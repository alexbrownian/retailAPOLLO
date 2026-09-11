"""
agentic_watch.py — how much of the crowd is trading WITH an AI?
===============================================================

Purpose: surface what retail traders are prompting AI with, and what
the AI is recommending or auto-trading for them — and whether that
chatter leads or lags the INCREASE / CUT EXPOSURE flags and the
boom/bust episodes (the research record, nb09_agentic_watch.json, runs
that test).

WHAT THIS MODULE DOES
  Scan the raw post archives for AI-TRADING LANGUAGE and build a daily,
  TEXT-FREE aggregate:

      Data/processed/daily_agentic_counts.parquet
          date, category, theme, mention_count
          (theme = "" for posts matching no theme keyword; a per-day
           "_total_posts" row per source carries the denominator)

  Four categories, defined in config/agentic_terms.csv (editable, same
  pattern as every other mapping in this project — one regex per row):

      ask_ai      asking an AI for picks/advice ("asked ChatGPT what to
                  buy", "Grok says NVDA to 200")
      auto_trade  an AI/bot EXECUTING trades ("my GPT agent bought",
                  "auto-trading with Claude", "bot sold my")
      build_bot   building AI trading systems ("coding a trading bot
                  with GPT", "backtesting my LLM strategy")
      skeptic     mocking/warning about AI trading (the contrarian side
                  of the same crowd)

  A small rolling sample of MATCHED post texts is kept LOCALLY in
  Data/reference/agentic_samples.jsonl (git-ignored, like every raw
  file) so the LLM digest and the research pass can quote-paraphrase; nothing
  with text ever reaches a committed store — the same boundary the rest
  of the pipeline enforces.

  Incremental: a ledger (Data/reference/agentic_scan_ledger.json) keyed
  on (path, size, mtime) skips already-scanned archives, so the
  update_data hook costs seconds after the first backfill.

CLI:
    cd Code
    python -m src.agentic_watch            # scan new archives, update store
    python -m src.agentic_watch --rebuild  # ignore the ledger, full rescan
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from src.config import PROCESSED_DIR, REFERENCE_DIR
from src.themes import themes_in_text
from src.config import DATA_DIR, PROJECT_DIR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIRS = {
    "reddit": os.path.join(DATA_DIR, "raw", "RedditComments"),
    "stocktwits": os.path.join(DATA_DIR, "raw", "StockTwits"),
    "reddit_live": os.path.join(DATA_DIR, "raw", "RedditLive"),
}
OUT_PATH = os.path.join(PROCESSED_DIR, "daily_agentic_counts.parquet")
LEDGER = os.path.join(REFERENCE_DIR, "agentic_scan_ledger.json")
SAMPLES = os.path.join(REFERENCE_DIR, "agentic_samples.jsonl")
TERMS_CSV = os.path.join(ROOT, "config", "agentic_terms.csv")
MAX_SAMPLES_PER_CAT_DAY = 3        # enough for a digest, tiny on disk
# the AI entities themselves - stripped before THEME attribution (below)
_ENTITY_RE = re.compile(
    r"chat\W?gpt|gpt-?\d\w*|\bgpt\b|claude|gemini|grok|deepseek|"
    r"copilot|\bai\b|a\.i\.|\bllm\b|artificial intelligence", re.I)


def load_patterns() -> dict[str, re.Pattern]:
    """category -> ONE compiled alternation. The CSV is the source of
    truth (desk-editable); patterns are combined per category so the hot
    loop runs a handful of C-level scans per post, not dozens."""
    per_cat: dict[str, list[str]] = defaultdict(list)
    with open(TERMS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat = (row.get("category") or "").strip()
            pat = (row.get("pattern") or "").strip()
            if not cat or not pat:
                continue
            re.compile(pat)                      # fail loudly on a typo
            per_cat[cat].append(f"(?:{pat})")
    if not per_cat:
        raise ValueError(f"{TERMS_CSV} has no usable rows")
    return {c: re.compile("|".join(ps), re.I | re.S)
            for c, ps in per_cat.items()}


def _iter_posts(path: str):
    """(date_str, text) for one raw archive, any of the three sources.
    Malformed lines are skipped, never fatal - archives contain them."""
    import zstandard
    base = os.path.basename(path)
    with open(path, "rb") as fh:
        stream = io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(fh),
            encoding="utf-8", errors="replace")
        for line in stream:
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            if "created_utc" in d:               # Reddit comment schema
                try:
                    day = datetime.fromtimestamp(
                        int(float(d["created_utc"])),
                        tz=timezone.utc).strftime("%Y-%m-%d")
                except (ValueError, TypeError, OSError):
                    continue
                text = " ".join(str(d.get(k) or "")
                                for k in ("title", "body"))
            elif "created_at" in d:              # StockTwits schema
                day = str(d.get("created_at", ""))[:10]
                text = str(d.get("body") or "")
            else:
                continue
            if len(day) == 10 and text.strip():
                yield day, text
    del base


def _archives() -> list[str]:
    out = []
    for d in RAW_DIRS.values():
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if f.endswith(".jsonl.zst")
                    and "_salvaged" not in f and ".tmp" not in f]
    return out


def _ledger_key(rel: str) -> str:
    """Ledger keys are project-relative, forward-slash paths under
    ``Data/`` (``Data/raw/RedditComments/x.jsonl.zst``). Keys written
    relative to the code folder or with a lower-case ``data/`` prefix
    are read as the same archive, so a moved working copy does not
    rescan everything it has already seen."""
    k = rel.replace("\\", "/")
    while k.startswith("../"):
        k = k[3:]
    if k.startswith("data/"):
        k = "Data/" + k[5:]
    return k


def scan(rebuild: bool = False, log=print) -> pd.DataFrame:
    """Scan new (or all) archives; merge into the daily store."""
    pats = load_patterns()
    ledger = {}
    if os.path.exists(LEDGER) and not rebuild:
        ledger = {_ledger_key(k): v for k, v in
                  json.load(open(LEDGER, encoding="utf-8")).items()}
    old = (pd.read_parquet(OUT_PATH)
           if os.path.exists(OUT_PATH) and not rebuild else None)

    counts: dict[tuple, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    samples: list[dict] = []
    seen_files, new_files = [], []
    for path in _archives():
        st = os.stat(path)
        key = _ledger_key(os.path.relpath(path, PROJECT_DIR))
        sig = [st.st_size, int(st.st_mtime)]
        if ledger.get(key) == sig:
            seen_files.append(key)
            continue
        new_files.append(key)
        n_posts = n_hits = 0
        per_cat_day_sampled: dict[tuple, int] = defaultdict(int)
        for day, text in _iter_posts(path):
            n_posts += 1
            totals[day] += 1
            low = text.lower()
            # cheap pre-filter: no AI entity word, no regex work at all
            if ("gpt" not in low and "chatgpt" not in low
                    and "claude" not in low and "gemini" not in low
                    and "grok" not in low and "copilot" not in low
                    and "deepseek" not in low and " ai " not in low
                    and "a.i." not in low and "llm" not in low
                    and not low.startswith("ai ") and "bot" not in low):
                continue
            for cat, pat in pats.items():
                if pat.search(text):
                    n_hits += 1
                    # theme attribution on the text WITH THE AI ENTITY
                    # WORDS REMOVED: "asked ChatGPT about uranium" must
                    # attribute to uranium_nuclear, not to the `ai` theme
                    # via its own "ChatGPT" keyword - otherwise every row
                    # in this store would read theme=ai and the record's
                    # lead/lag test would be measuring the keyword list.
                    found = themes_in_text(_ENTITY_RE.sub(" ", text))                         or {""}
                    for th in found:
                        counts[(day, cat, th)] += 1
                    k = (day, cat)
                    if per_cat_day_sampled[k] < MAX_SAMPLES_PER_CAT_DAY:
                        per_cat_day_sampled[k] += 1
                        samples.append({"date": day, "category": cat,
                                        "text": text[:600]})
        ledger[key] = sig
        log(f"  agentic scan: {key}  {n_posts:,} posts, {n_hits:,} hits")

    if not new_files and old is not None:
        log("  agentic scan: nothing new to scan")
        return old

    rows = [{"date": d, "category": c, "theme": t, "mention_count": n}
            for (d, c, t), n in counts.items()]
    rows += [{"date": d, "category": "_total_posts", "theme": "",
              "mention_count": n} for d, n in totals.items()]
    new = pd.DataFrame(rows)
    if old is not None and len(old):
        # a re-scanned day replaces its old rows (archives can overlap);
        # merged on the day level so overlapping files never double-count
        new_days = set(new["date"].unique())
        old = old[~old["date"].astype(str).isin(new_days)]
        new = pd.concat([old, new], ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])
    new = (new.groupby([new["date"], "category", "theme"], as_index=False)
           ["mention_count"].sum().sort_values(["date", "category"]))
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    new.to_parquet(OUT_PATH, index=False)
    json.dump(ledger, open(LEDGER, "w", encoding="utf-8"), indent=0)
    if samples:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        with open(SAMPLES, "a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    log(f"  agentic store: {len(new):,} rows -> "
        f"{os.path.relpath(OUT_PATH, ROOT)}")
    return new


def load_series() -> pd.DataFrame | None:
    """The daily store, or None before the first scan."""
    if not os.path.exists(OUT_PATH):
        return None
    df = pd.read_parquet(OUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def recent_samples(days: int = 7, per_cat: int = 12) -> list[dict]:
    """Latest locally-held matched texts, for the LLM digest ONLY -
    never rendered raw, never committed."""
    if not os.path.exists(SAMPLES):
        return []
    cutoff = (pd.Timestamp.now() - pd.Timedelta(days=days * 4))
    rows = []
    for line in open(SAMPLES, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    rows = [r for r in rows
            if pd.Timestamp(r.get("date", "1970-01-01")) >= cutoff]
    out, seen = [], defaultdict(int)
    for r in sorted(rows, key=lambda r: r["date"], reverse=True):
        if seen[r["category"]] < per_cat:
            seen[r["category"]] += 1
            out.append(r)
    return out


if __name__ == "__main__":
    import sys
    scan(rebuild="--rebuild" in sys.argv)
