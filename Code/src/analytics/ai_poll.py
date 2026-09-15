"""
ai_poll.py — ask the AI what a retail trader would ask it, every refresh.
=========================================================================

On every data refresh, ask a consumer AI model the questions a retail
trader would ask it ("what should I invest in", "best stocks after
memory") and record what it recommends, so the advice flowing into the
crowd can be triangulated over time.

Retail asks the same handful of consumer models the same handful of
questions — so the model's answers ARE a proxy for the advice flowing
into the crowd.  This module runs that poll at the end of every
update_data pass:

  * the PROMPT PANEL lives in config/ai_poll_prompts.csv (editable, one
    retail-style question per row).  It holds 12 prompts, one per
    family: the plain questions retail types, the scaffolds retail
    actually runs (hedge-fund-manager and Warren-Buffett personas from
    the most-starred AI-investing repos, the JSON-decision agent loop,
    the copy-paste screens), and balance prompts chosen so the panel is
    not read as purely a tech chase (gold, dividends, value screen and
    the boring-portfolio ask sit beside the single AI ask). A removed
    prompt_id's history stays in the store and its series simply ends;
    re-adding the same id later resumes it.
    The `family` column tags each one (plain,
    theme, persona, agent, screen, portfolio, momentum, risk, thesis) so
    the series can be read by TYPE of asker as well as in aggregate -
    whether the persona scaffolds recommend something different from the
    plain questions is itself a finding.
    ADDING is safe; REWORDING an existing prompt is not - it silently
    breaks that prompt_id's history. Add a new id instead;
  * each prompt is asked ONCE per run at temperature 0.8 — consumer
    products answer at a high temperature, so a single low-temperature
    reading would measure a machine retail never talks to; run-to-run
    noise is averaged out by the time series, which is the object we
    actually want;
  * the model answers naturally FIRST, then self-reports a structured
    extraction (tickers + direction + conviction, themes) — one call per
    prompt, ~12 calls/run, ~1 min at the gateway's pace;
  * rows land in Data/processed/ai_poll.parquet
        (run_date, prompt_id, kind ticker|theme, name, direction,
         conviction, rank, model, mock)
    and the full answer texts in Data/processed/ai_poll_answers.jsonl —
    both under Data/processed/, so nothing reaches a commit.

THE SERIES STARTS THE DAY YOU START POLLING.  There is no backfill —
nobody can ask 2024's model what it recommended — so the correlation
test against our flags is a FORWARD test, pre-registered in the
research record (Data/research_record/nb09_agentic_watch.json): it
activates itself once >=60 distinct poll days exist.

CLI:
    cd Code
    python -m src.analytics.ai_poll          # run the poll now
    python -m src.analytics.ai_poll --show   # print the latest run
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

import pandas as pd

from src import ai
from src.config import PROCESSED_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS_CSV = os.path.join(ROOT, "config", "ai_poll_prompts.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "ai_poll.parquet")
ANSWERS = os.path.join(PROCESSED_DIR, "ai_poll_answers.jsonl")
POLL_TEMPERATURE = 0.8      # consumer-product temperature, see docstring

_SYSTEM = (
    "Answer the user's question exactly as a helpful consumer AI "
    "assistant would answer a retail investor - your genuine, natural "
    "answer, including any caveats you would normally give. THEN, on a "
    "new line, write 'JSON:' followed by ONE json object summarising "
    "your own answer: {\"tickers\": [{\"symbol\": str, \"direction\": "
    "\"buy\"|\"sell\"|\"avoid\", \"conviction\": \"high\"|\"medium\"|"
    "\"low\"}], \"themes\": [str], \"summary\": \"<=25 words\"}. The "
    "JSON must list only names your answer actually recommended or "
    "warned about, in the order you mentioned them.")


def _prompts() -> list[dict]:
    with open(PROMPTS_CSV, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f)
                if (r.get("prompt") or "").strip()]


def _extract(answer: str) -> dict:
    """The self-reported JSON block at the tail of the natural answer."""
    if "JSON:" in answer:
        answer = answer.split("JSON:", 1)[1]
    try:
        return ai._parse_json(answer)             # noqa: SLF001
    except (ValueError, TypeError):
        return {}


def run(log=print) -> tuple[bool, str]:
    """Poll every prompt once; append this run to the stores."""
    if not ai.available():
        return False, f"LLM unavailable: {ai.explain_unavailable()}"
    prompts = _prompts()
    if not prompts:
        return False, "config/ai_poll_prompts.csv is empty"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log(f"AI POLL: asking {len(prompts)} retail prompts "
        f"(model {ai.active_model() or ai.MODEL}, T={POLL_TEMPERATURE})")
    rows, answers = [], []
    for _i, p in enumerate(prompts, 1):
        log(f"AI POLL: [{_i}/{len(prompts)}] asking: "
            f"\"{p['prompt'][:52]}\"")
        try:
            text = ai.chat(p["prompt"], system=_SYSTEM,
                           temperature=POLL_TEMPERATURE,
                           # 700 was sized for the gateway's model.
                           # The poll asks for a NATURAL answer plus a
                           # JSON summary, and a more verbose model
                           # spends the budget on the prose and is cut
                           # off before the JSON - which is exactly how
                           # this failed in production.
                           max_tokens=2500)
        except RuntimeError as e:
            log(f"AI POLL: stopped early - {e}")
            break
        parsed = _extract(str(text))
        log(f"AI POLL: [{_i}/{len(prompts)}] -> "
            f"{len(parsed.get('tickers', []) or [])} names, "
            f"{len(parsed.get('themes', []) or [])} themes")
        answers.append({"run_date": today, "prompt_id": p["prompt_id"],
                        "prompt": p["prompt"], "answer": str(text),
                        "model": ai.active_model() or ai.MODEL, "mock": ai.MOCK})
        for rank, t in enumerate(parsed.get("tickers", []) or [], 1):
            sym = str(t.get("symbol", "")).strip().upper()
            if not sym or len(sym) > 10:
                continue
            rows.append({"run_date": today, "prompt_id": p["prompt_id"],
                         "family": str(p.get("family") or ""),
                         "kind": "ticker", "name": sym,
                         "direction": str(t.get("direction", "buy")),
                         "conviction": str(t.get("conviction", "")),
                         "rank": rank, "model": ai.active_model() or ai.MODEL,
                         "mock": ai.MOCK})
        for rank, th in enumerate(parsed.get("themes", []) or [], 1):
            rows.append({"run_date": today, "prompt_id": p["prompt_id"],
                         "family": str(p.get("family") or ""),
                         "kind": "theme",
                         "name": str(th).strip().lower()[:40],
                         "direction": "buy", "conviction": "",
                         "rank": rank, "model": ai.active_model() or ai.MODEL,
                         "mock": ai.MOCK})
    if not rows:
        return False, "no structured recommendations parsed"
    new = pd.DataFrame(rows)
    new["run_date"] = pd.to_datetime(new["run_date"])
    if os.path.exists(OUT_PATH):
        old = pd.read_parquet(OUT_PATH)
        old["run_date"] = pd.to_datetime(old["run_date"])
        # one poll per day: a rerun the same day replaces that day
        old = old[old["run_date"] != pd.Timestamp(today)]
        new = pd.concat([old, new], ignore_index=True)
    tmp = OUT_PATH + ".tmp"
    new.to_parquet(tmp, index=False)         # atomic swap - never half-written
    os.replace(tmp, OUT_PATH)
    with open(ANSWERS, "a", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")
    n_tick = int((new["run_date"] == pd.Timestamp(today)).sum())
    log(f"AI POLL: {n_tick} recommendation rows recorded for {today} "
        f"-> {os.path.relpath(OUT_PATH, ROOT)}")
    return True, "ok"


def load_series() -> pd.DataFrame | None:
    if not os.path.exists(OUT_PATH):
        return None
    df = pd.read_parquet(OUT_PATH)
    df["run_date"] = pd.to_datetime(df["run_date"])
    return df


if __name__ == "__main__":
    import sys
    if "--show" in sys.argv:
        df = load_series()
        if df is None:
            print("no poll runs yet")
        else:
            last = df[df["run_date"] == df["run_date"].max()]
            print(last.groupby(["kind", "name", "direction"])
                  .size().sort_values(ascending=False).head(25))
    else:
        ok, msg = run()
        print(f"[{'OK' if ok else 'SKIP'}] {msg}")
        raise SystemExit(0 if ok else 1)
