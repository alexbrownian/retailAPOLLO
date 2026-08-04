"""
ai_pulse.py — the LLM writes the qualitative read of the market.
================================================================

The AI Pulse tab carried hand-written SAMPLE text since it shipped; this
module produces the real thing.  It runs at the end of every update_data
pass (and on demand from the dashboard), builds a compact EVIDENCE PACK
from the same aggregates every other tab reads, hands the freshest raw
posts to the firm's LLM through `src/ai.py` (the Apollo gateway), and
saves one JSON the dashboard renders verbatim:

    data/processed/ai_pulse.json
        as_of, generated_at, model, evidence (the numbers used),
        market_pulse, talk_of_the_town, mood_gauge{score,why},
        theme_briefs[], rally_watch[], catalyst_watch[], divergences[],
        agentic{digest, asks[], actions[], risk_note}

DESIGN RULES
  * NUMBERS COME FROM THE STORES, WORDS COME FROM THE MODEL.  The LLM
    never invents a statistic: every figure it may cite is handed to it
    in the evidence pack, and the pack itself is saved alongside the
    prose so any sentence can be audited against the inputs.
  * TWO CALLS PER RUN, not one per segment - the gateway round-trip is
    ~5s, so segments share a call: (1) the market read, (2) the agentic
    digest.  Budget-capped by AI_MAX_CALLS regardless.
  * PARAPHRASE, NEVER QUOTE.  Raw post text goes TO the model; only
    model-written summaries and paraphrases come back and are stored -
    no verbatim crowd text, no usernames, same text-free boundary as
    the committed aggregates.
  * DEGRADE, NEVER CRASH.  No VPN / no dimsum_lite / budget spent ->
    generate() returns (False, reason); update_data logs it and moves
    on; the dashboard keeps showing the last pulse (or the samples)
    with an honest banner.

CLI:
    python -m analytics.ai_pulse           # generate now
    python -m analytics.ai_pulse --dry     # print the evidence pack only
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone

import pandas as pd

from src import ai
from src.config import PROCESSED_DIR
from src.themes import THEME_ETFS, themes_in_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROCESSED_DIR, "ai_pulse.json")
RAW_DIR = os.path.join(ROOT, "data", "raw", "RedditComments")
POST_SAMPLE_N = 60          # newest posts handed to the model
POST_CLIP = 320             # chars per post - mood, not essays


def _read(name: str) -> pd.DataFrame | None:
    p = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _evidence() -> dict:
    """The numbers the model is allowed to use - nothing else."""
    ev: dict = {}
    tc = _read("daily_theme_counts.parquet")
    if tc is not None and len(tc):
        tc = tc[tc["theme"].isin(THEME_ETFS)]
        hi = tc["date"].max()
        w = tc[tc["date"] > hi - pd.Timedelta(days=7)]
        prev = tc[(tc["date"] <= hi - pd.Timedelta(days=7))
                  & (tc["date"] > hi - pd.Timedelta(days=35))]
        cur = w.groupby("theme")["mention_count"].sum()
        base = prev.groupby("theme")["mention_count"].sum() / 4.0
        share = (cur / cur.sum()).sort_values(ascending=False)
        ev["as_of"] = str(hi.date())
        ev["theme_mention_share_7d"] = {
            t: {"share": round(float(s), 4),
                "vs_4w_avg": (round(float(cur.get(t, 0)
                                          / base.get(t)), 2)
                              if base.get(t) else None)}
            for t, s in share.head(10).items()}
    cz = _read("daily_theme_conviction.parquet")
    if cz is not None and len(cz):
        last = cz[cz["date"] == cz["date"].max()]
        last = last[last["theme"].isin(THEME_ETFS)]
        top = last.nlargest(5, "conviction_z")
        bot = last.nsmallest(3, "conviction_z")
        ev["conviction_z"] = {
            r.theme: round(float(r.conviction_z), 2)
            for r in pd.concat([top, bot]).itertuples()}
    dk = _read("euphoria_desk.parquet")
    if dk is not None and len(dk):
        hi = dk["date"].max()
        recent = dk[dk["date"] > hi - pd.Timedelta(days=21)]
        ev["live_flags"] = (
            [{"name": r.name, "flag": "GET OUT",
              "date": str(r.date.date())}
             for r in recent[recent["get_out"].astype(bool)].itertuples()]
            + [{"name": r.name, "flag": "GET IN",
                "date": str(r.date.date())}
               for r in recent[recent["get_in"].astype(bool)]
               .itertuples()])
    eu = _read("euphoria_levels.parquet")
    if eu is not None and len(eu):
        last = eu[eu["date"] == eu["date"].max()]
        ev["euphoria_top5"] = {
            r.name: round(float(r.level))
            for r in last.nlargest(5, "level").itertuples()}
    ag = _read("daily_agentic_counts.parquet")
    if ag is not None and len(ag):
        hi = ag["date"].max()
        w = ag[ag["date"] > hi - pd.Timedelta(days=28)]
        tot = w[w["category"] == "_total_posts"]["mention_count"].sum()
        sig = w[w["category"] != "_total_posts"]
        ev["agentic_28d"] = {
            "per_10k_posts": (round(float(sig["mention_count"].sum())
                              / tot * 10000, 1) if tot else None),
            "by_category": {k: int(v) for k, v in
                            sig.groupby("category")["mention_count"]
                            .sum().items()},
            "top_themes": {k: int(v) for k, v in
                           sig[sig["theme"] != ""]
                           .groupby("theme")["mention_count"].sum()
                           .nlargest(5).items()},
        }
    return ev


def _fresh_posts(n: int = POST_SAMPLE_N) -> list[dict]:
    """The newest raw posts, theme-tagged, clipped - the model's ears."""
    import zstandard
    files = sorted((f for f in os.listdir(RAW_DIR)
                    if f.endswith(".jsonl.zst") and ".tmp" not in f),
                   key=lambda f: os.path.getmtime(
                       os.path.join(RAW_DIR, f)))
    out: list[dict] = []
    for fname in reversed(files):
        with open(os.path.join(RAW_DIR, fname), "rb") as fh:
            t = io.TextIOWrapper(
                zstandard.ZstdDecompressor().stream_reader(fh),
                encoding="utf-8", errors="replace")
            rows = []
            for line in t:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                body = str(d.get("body") or "")[:POST_CLIP]
                if len(body) < 25:
                    continue
                rows.append({"sub": d.get("subreddit", ""),
                             "themes": sorted(themes_in_text(body))[:3],
                             "text": body})
            # newest archive first; spread across themes, then fill
            themed = [r for r in rows if r["themes"]]
            plain = [r for r in rows if not r["themes"]]
            out += themed[-int(n * 0.8):] + plain[-int(n * 0.2):]
        if len(out) >= n:
            break
    return out[-n:]


_PULSE_SYSTEM = (
    "You write the daily qualitative read of retail-investor chatter for "
    "a professional trading desk. Rules, strictly: cite only numbers "
    "present in the EVIDENCE block; when you characterise the crowd's "
    "words, PARAPHRASE - never quote verbatim, never name users; be "
    "concrete and falsifiable, never vague; mark genuine uncertainty "
    "plainly; no investment advice language, this is a description of "
    "the crowd, not a recommendation. Answer ONLY with the requested "
    "JSON object, no prose around it.")


def _pulse_prompt(ev: dict, posts: list[dict]) -> str:
    seg = {
        "market_pulse": "one paragraph (<=120 words): the week's retail "
                        "read - what dominates, mood, rotation, what is "
                        "notably absent",
        "talk_of_the_town": "one paragraph (<=90 words): the specific "
                            "topics/threads the crowd keeps returning to",
        "mood_gauge": "object {score: 0-100 int (0 fear, 100 greed), "
                      "why: <=25 words}",
        "theme_briefs": "list of <=5 objects {theme, brief:<=40 words} "
                        "for the loudest themes - tone, framing, dissent",
        "rally_watch": "list of <=3 objects {target, verdict: one of "
                       "'clear rallying detected'|'early signs, watch'|"
                       "'no rallying detected', why:<=60 words, "
                       "example: a PARAPHRASE prefixed 'paraphrase - '} "
                       "- mobilising/recruiting language, coordinated "
                       "framing, evangelical tone",
        "catalyst_watch": "list of <=4 objects {event, themes[], "
                          "chatter:<=25 words} - events the crowd "
                          "positions for",
        "divergences": "list of <=3 objects {name, story:<=35 words} - "
                       "where the crowd's story disagrees with the "
                       "measured numbers in EVIDENCE",
    }
    return (f"EVIDENCE (the only numbers you may cite):\n"
            f"{json.dumps(ev, indent=1)}\n\n"
            f"FRESH POSTS (a sample of the newest raw crowd text, "
            f"theme-tagged):\n{json.dumps(posts, indent=0)}\n\n"
            f"Write the pulse. Return ONE JSON object with exactly "
            f"these keys:\n{json.dumps(seg, indent=1)}")


_AGENTIC_SYSTEM = _PULSE_SYSTEM


def _agentic_prompt(ev: dict, samples: list[dict]) -> str:
    return (
        "These are posts where retail traders describe USING AI to "
        "trade (asking models for picks, running AI agents/bots, "
        "building AI strategies, or mocking those who do). Counts for "
        f"context: {json.dumps(ev.get('agentic_28d', {}))}\n\n"
        f"POSTS:\n{json.dumps(samples, indent=0)}\n\n"
        "Return ONE JSON object: {digest: <=120 words - what retail is "
        "prompting AIs for and what the AIs appear to be telling/doing "
        "for them, as observed in these posts; asks: list of <=5 short "
        "strings - the typical prompts/questions; actions: list of <=5 "
        "short strings - what the AI reportedly said/did; risk_note: "
        "<=40 words - herding/correlation risk if many follow the same "
        "model outputs. PARAPHRASE only, no verbatim quotes, no "
        "usernames.}")


def generate(log=print) -> tuple[bool, str]:
    """Build the evidence, call the model, write ai_pulse.json.
    Returns (ok, message) - never raises for gateway problems."""
    log("AI PULSE: building evidence pack")
    ev = _evidence()
    if not ev:
        return False, "no aggregates on disk yet - run update_data first"
    if not ai.available():
        return False, f"LLM unavailable: {ai.explain_unavailable()}"
    posts = _fresh_posts()
    log(f"AI PULSE: {len(posts)} fresh posts, model {ai.MODEL}, "
        "generating (2 calls)")
    try:
        pulse = ai.chat(_pulse_prompt(ev, posts), system=_PULSE_SYSTEM,
                        want_json=True, max_tokens=2200)
        from src.agentic_watch import recent_samples
        agentic = ai.chat(
            _agentic_prompt(ev, recent_samples(per_cat=8)),
            system=_AGENTIC_SYSTEM, want_json=True, max_tokens=900)
    except (RuntimeError, ValueError) as e:
        return False, f"generation failed: {e}"
    doc = {
        "as_of": ev.get("as_of"),
        "generated_at": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M UTC"),
        "model": ai.MODEL,
        "mock": ai.MOCK,
        "evidence": ev,
    }
    doc.update(pulse if isinstance(pulse, dict) else {})
    doc["agentic"] = agentic if isinstance(agentic, dict) else {}
    json.dump(doc, open(OUT_PATH, "w", encoding="utf-8"), indent=1)
    log(f"AI PULSE: saved -> {os.path.relpath(OUT_PATH, ROOT)}")
    return True, "ok"


def load() -> dict | None:
    if not os.path.exists(OUT_PATH):
        return None
    try:
        return json.load(open(OUT_PATH, encoding="utf-8"))
    except ValueError:
        return None


if __name__ == "__main__":
    import sys
    if "--dry" in sys.argv:
        print(json.dumps(_evidence(), indent=1))
    else:
        ok, msg = generate()
        print(f"[{'OK' if ok else 'SKIP'}] {msg}")
        raise SystemExit(0 if ok else 1)
