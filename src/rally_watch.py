"""
rally_watch.py — is the crowd being MOBILISED, or just talking?
================================================================

The desk's question (2026-08-04, verbatim): "can we add a pump / rally
detector? e.g. if the tone of the posts are quite rallying or like lets
save this company or lets short squeeze! (e.g wendys or GME etc) we flag
that out."

WHY THIS IS A SEPARATE MEASUREMENT, NOT A PROMPT
------------------------------------------------
The AI Pulse already asked the model to spot "rallying", but it did so by
reading a 120-post sample and using its judgement.  That is the wrong
tool for the job twice over: the model saw a rounding error of the
corpus, and its verdict carried no number anyone could audit or backtest.
Mobilising language, unlike mood, is LEXICAL — people who are organising
a buy say a small, stable set of things ("diamond hands", "shorts are
trapped", "let's save this company", "everyone buy at 9:30").  That is
exactly what a regex bank over EVERY post does well and cheaply.

So the division of labour matches the rest of the project: this module
SCANS 100% OF THE POSTS and produces numbers; the LLM later reads the
posts those numbers point at and writes the words.  Off the VPN the
numbers still exist, so the dashboard's rally section keeps working with
no gateway at all.

WHAT IT PRODUCES
----------------
    data/processed/daily_rally_counts.parquet
        date, category, kind, name, mention_count
          kind "_all"     -> market-wide count for that category
                             (name "", plus a "_total_posts" denominator
                              row per day)
          kind "theme"    -> the same hit attributed to a theme
          kind "ticker"   -> ...and to any tickers named in the post
        plus category "_name_posts" rows: how many posts mentioned each
        theme/ticker AT ALL that day.

WHY THE DENOMINATOR IS COMPUTED HERE AND NOT BORROWED
-----------------------------------------------------
The obvious shortcut - divide these hits by daily_theme_counts /
daily_ticker_counts - is wrong, and quietly so.  Those stores are built
from posts.parquet (submissions: title + selftext); this scanner reads
the raw COMMENT archives.  Different populations, so the ratio is not a
share of anything: the first calibration run produced ticker "shares" of
250%, which is what a mismatched denominator looks like when it is polite
enough to be obvious.  Counting the denominator in the same pass over the
same posts with the same extractors costs about a minute on the full
archive and makes the ratio true by construction.

Seven categories, defined in config/rally_terms.csv (desk-editable, one
regex per row, same pattern as theme_keywords/agentic_terms):

    recruit           bringing others in ("get in before the institutions")
    squeeze           squeeze mechanics ("MOASS", "days to cover", "float
                      is locked")
    hold_the_line     coordinated refusal to sell ("diamond hands", "DRS")
    save_the_company  rescue framing ("let's save this company", buycotts,
                      "too iconic to fail") - the Wendy's case
    coordinate        explicit timing/headcount ("everyone buy at 9:30",
                      "we just need 50k people")
    moonshot          extreme-outcome claims ("10x", "generational wealth")
    pump_callout      THE CROWD'S OWN IMMUNE RESPONSE - "pump and dump",
                      "exit liquidity", "you're the bagholder".  Counted
                      but deliberately EXCLUDED from the rally score: it
                      is the contrarian side of the same conversation and
                      reading it as rallying would invert the signal.

THE SCORE (rally_frame)
-----------------------
Raw counts are useless on their own: GME says "diamond hands" every day
of its life, and a big theme collects more of everything simply by being
big.  So the measure is a SHARE against that name's own chatter —

    rally_share = mobilising hits / mentions of that name   (7d sums,
                  denominators taken from the SAME daily_ticker_counts /
                  daily_theme_counts stores every other tab reads)

— and a z-score of the daily hit series against the name's own trailing
normal, computed with `analytics.conviction.ewm_z`, i.e. THE SAME z
construction used everywhere else in this project (7d roll, EWM baseline,
strictly trailing).  No new statistics were invented for this module.

A name is called RALLYING when all three hold (see src/config.py for the
constants and their provenance): enough absolute evidence, an unusual
share of its own chatter, and an unusual level versus its own history.

BOUNDARIES
----------
  * DISPLAY AND RESEARCH ONLY.  This does not touch the frozen euphoria
    thresholds and does not create or suppress a GET IN / GET OUT flag.
    Its forward test is pre-registered in notebook 09; until that test
    passes, nothing here is allowed near the signal (handover rule 1).
  * TEXT-FREE.  Matched post text is kept LOCALLY in
    data/reference/rally_samples.jsonl (git-ignored) so the LLM can read
    what it is describing; only counts reach the committed stores, and
    only paraphrases reach the saved pulse.

CLI:
    python -m src.rally_watch             # scan new archives
    python -m src.rally_watch --rebuild   # ignore the ledger, full rescan
    python -m src.rally_watch --show      # print the current leaderboard
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

from src.config import (PROCESSED_DIR, REFERENCE_DIR, RALLY_MIN_HITS,
                        RALLY_MIN_SHARE, RALLY_MIN_Z, RALLY_WINDOW_D)
from src.themes import themes_in_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIRS = {
    "reddit": os.path.join(ROOT, "data", "raw", "RedditComments"),
    "stocktwits": os.path.join(ROOT, "data", "raw", "StockTwits"),
    "reddit_live": os.path.join(ROOT, "data", "raw", "RedditLive"),
}
OUT_PATH = os.path.join(PROCESSED_DIR, "daily_rally_counts.parquet")
LEDGER = os.path.join(REFERENCE_DIR, "rally_scan_ledger.json")
SAMPLES = os.path.join(REFERENCE_DIR, "rally_samples.jsonl")
TERMS_CSV = os.path.join(ROOT, "config", "rally_terms.csv")

# the contrarian category: measured, never scored as rallying (docstring)
COUNTER_CATEGORY = "pump_callout"
MAX_SAMPLES_PER_CAT_DAY = 6      # enough for the LLM to read, tiny on disk
SAMPLE_CLIP = 600


def load_patterns() -> dict[str, re.Pattern]:
    """category -> ONE compiled alternation, plus the "_any" union used as
    the hot-loop pre-filter.  The CSV is the source of truth; a bad regex
    raises here rather than silently matching nothing."""
    per_cat: dict[str, list[str]] = defaultdict(list)
    with open(TERMS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat = (row.get("category") or "").strip()
            pat = (row.get("pattern") or "").strip()
            if not cat or not pat:
                continue
            re.compile(pat)                       # fail loudly on a typo
            per_cat[cat].append(f"(?:{pat})")
    if not per_cat:
        raise ValueError(f"{TERMS_CSV} has no usable rows")
    out = {c: re.compile("|".join(ps), re.I | re.S)
           for c, ps in per_cat.items()}
    out["_any"] = re.compile(
        "|".join(p for ps in per_cat.values() for p in ps), re.I | re.S)
    return out


def _iter_posts(path: str):
    """(date, text, score) for one raw archive, any of the three sources.
    Malformed lines are skipped, never fatal - archives contain them."""
    import zstandard
    with open(path, "rb") as fh:
        stream = io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(fh),
            encoding="utf-8", errors="replace")
        for line in stream:
            try:
                d = json.loads(line)
            except (ValueError, TypeError):
                continue
            try:
                score = int(float(d.get("score") or 0))
            except (ValueError, TypeError):
                score = 0
            if "created_utc" in d:                # Reddit schema
                try:
                    day = datetime.fromtimestamp(
                        int(float(d["created_utc"])),
                        tz=timezone.utc).strftime("%Y-%m-%d")
                except (ValueError, TypeError, OSError):
                    continue
                text = " ".join(str(d.get(k) or "")
                                for k in ("title", "body"))
            elif "created_at" in d:               # StockTwits schema
                day = str(d.get("created_at", ""))[:10]
                text = str(d.get("body") or "")
            else:
                continue
            if len(day) == 10 and text.strip():
                yield day, text, score


def _archives() -> list[str]:
    out = []
    for d in RAW_DIRS.values():
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d))
                    if f.endswith(".jsonl.zst")
                    and "_salvaged" not in f and ".tmp" not in f]
    return out


def _universe() -> set[str]:
    try:
        from src.abstracted_data import load_universe
        return load_universe()
    except Exception:                             # noqa: BLE001
        return set()


def scan(rebuild: bool = False, log=print) -> pd.DataFrame:
    """Scan new (or all) archives; merge into the daily store."""
    pats = load_patterns()
    any_re = pats.pop("_any")
    ledger = {}
    if os.path.exists(LEDGER) and not rebuild:
        ledger = json.load(open(LEDGER, encoding="utf-8"))
    old = (pd.read_parquet(OUT_PATH)
           if os.path.exists(OUT_PATH) and not rebuild else None)

    counts: dict[tuple, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    samples: list[dict] = []
    new_files = []
    universe = _universe()
    from src.extract_tickers import extract_tickers_from_text

    for path in _archives():
        st = os.stat(path)
        key = os.path.relpath(path, ROOT).replace("\\", "/")
        sig = [st.st_size, int(st.st_mtime)]
        if ledger.get(key) == sig:
            continue
        new_files.append(key)
        n_posts = n_hits = 0
        per_cat_day: dict[tuple, int] = defaultdict(int)
        for day, text, score in _iter_posts(path):
            n_posts += 1
            totals[day] += 1
            # EVERY post is attributed, hit or not: these are the
            # denominators (see the docstring - they must come from this
            # same population, not from the submissions-based stores)
            themes = themes_in_text(text)
            tickers = (set(extract_tickers_from_text(
                text, universe, cashtags_only=False)) if universe else set())
            for th in themes:
                counts[(day, "_name_posts", "theme", th)] += 1
            for tk in tickers:
                counts[(day, "_name_posts", "ticker", tk)] += 1
            if not any_re.search(text):           # 1 scan for ~98% of posts
                continue
            hit_cats = [c for c, p in pats.items() if p.search(text)]
            if not hit_cats:
                continue
            n_hits += 1
            for cat in hit_cats:
                counts[(day, cat, "_all", "")] += 1
                for th in themes:
                    counts[(day, cat, "theme", th)] += 1
                for tk in tickers:
                    counts[(day, cat, "ticker", tk)] += 1
                k = (day, cat)
                if per_cat_day[k] < MAX_SAMPLES_PER_CAT_DAY:
                    per_cat_day[k] += 1
                    samples.append({
                        "date": day, "category": cat, "score": score,
                        "themes": sorted(themes)[:3],
                        "tickers": sorted(tickers)[:4],
                        "text": text[:SAMPLE_CLIP]})
        ledger[key] = sig
        log(f"  rally scan: {key}  {n_posts:,} posts, {n_hits:,} hits")

    if not new_files and old is not None:
        log("  rally scan: nothing new to scan")
        return old

    rows = [{"date": d, "category": c, "kind": k, "name": n,
             "mention_count": v} for (d, c, k, n), v in counts.items()]
    rows += [{"date": d, "category": "_total_posts", "kind": "_all",
              "name": "", "mention_count": n} for d, n in totals.items()]
    new = pd.DataFrame(rows, columns=["date", "category", "kind", "name",
                                      "mention_count"])
    if old is not None and len(old):
        # a re-scanned day replaces its old rows, so overlapping archives
        # can never double-count (same rule as the agentic store)
        new_days = set(new["date"].unique())
        old = old[~old["date"].astype(str).isin(new_days)]
        new = pd.concat([old, new], ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])
    new = (new.groupby([new["date"], "category", "kind", "name"],
                       as_index=False)["mention_count"].sum()
           .sort_values(["date", "category"]))
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    new.to_parquet(OUT_PATH, index=False)
    json.dump(ledger, open(LEDGER, "w", encoding="utf-8"), indent=0)
    if samples:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        with open(SAMPLES, "a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    log(f"  rally store: {len(new):,} rows -> "
        f"{os.path.relpath(OUT_PATH, ROOT)}")
    return new


def load_series() -> pd.DataFrame | None:
    """The daily store, or None before the first scan."""
    if not os.path.exists(OUT_PATH):
        return None
    df = pd.read_parquet(OUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def rally_frame(kind: str = "theme", window: int = RALLY_WINDOW_D,
                as_of: str | pd.Timestamp | None = None
                ) -> pd.DataFrame | None:
    """One row per name: how mobilised its chatter is right now.

    Columns: name, hits, mentions, share, z, rallying, top_category,
             by_category (dict), counter (pump_callout hits).
    `share` is the fraction of the name's chatter that is mobilising;
    `z` is the project-standard EWM trailing z of its daily hit series.

    `as_of` reads the frame as it stood on a past date - strictly
    trailing, nothing after it is touched - which is how notebook 09
    checks the detector against known episodes (does it light up on
    GME in 2021?) without any hindsight."""
    df = load_series()
    if df is None or not len(df):
        return None
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
        if not len(df):
            return None
    sig = df[(df["kind"] == kind)
             & (~df["category"].str.startswith("_"))]
    if not len(sig):
        return None
    hi = df["date"].max()
    lo = hi - pd.Timedelta(days=window)
    w = sig[sig["date"] > lo]
    if not len(w):
        return None
    den_rows = df[(df["kind"] == kind)
                  & (df["category"] == "_name_posts")]

    pos = w[w["category"] != COUNTER_CATEGORY]
    hits = pos.groupby("name")["mention_count"].sum()
    counter = (w[w["category"] == COUNTER_CATEGORY]
               .groupby("name")["mention_count"].sum())
    by_cat = (pos.groupby(["name", "category"])["mention_count"].sum()
              .unstack(fill_value=0))

    # z of the DAILY hit series, project-standard construction
    from analytics.conviction import ewm_z
    wide = (sig[sig["category"] != COUNTER_CATEGORY]
            .pivot_table(index="date", columns="name",
                         values="mention_count", aggfunc="sum")
            .asfreq("D").fillna(0.0))
    z_all = ewm_z(wide)
    z_last = (z_all.iloc[-1] if len(z_all) else pd.Series(dtype=float))

    mentions = (den_rows[den_rows["date"] > lo]
                .groupby("name")["mention_count"].sum()
                if len(den_rows) else pd.Series(dtype=float))

    out = []
    for name, h in hits.sort_values(ascending=False).items():
        m = float(mentions.get(name, 0.0))
        share = (float(h) / m) if m > 0 else None
        z = float(z_last.get(name)) if name in z_last.index else None
        if z is not None and pd.isna(z):
            z = None
        cats = ({c: int(v) for c, v in by_cat.loc[name].items() if v}
                if name in by_cat.index else {})
        out.append({
            "name": name, "hits": int(h), "mentions": int(m),
            "share": share, "z": z,
            "top_category": (max(cats, key=cats.get) if cats else None),
            "by_category": cats,
            "counter": int(counter.get(name, 0)),
            "rallying": bool(h >= RALLY_MIN_HITS
                             and share is not None
                             and share >= RALLY_MIN_SHARE
                             and z is not None and z >= RALLY_MIN_Z),
        })
    return pd.DataFrame(out)


def top_rallies(kind: str = "theme", n: int = 8,
                window: int = RALLY_WINDOW_D,
                as_of: str | pd.Timestamp | None = None) -> list[dict]:
    """The most-mobilised names, best first - the dashboard's and the
    evidence pack's view.  Sorted by share (how much of the name's own
    chatter is mobilising), not raw hits, so size does not win."""
    f = rally_frame(kind, window, as_of=as_of)
    if f is None or not len(f):
        return []
    f = f[f["hits"] >= max(3, RALLY_MIN_HITS // 3)]
    f = f.sort_values(["rallying", "share", "hits"],
                      ascending=[False, False, False])
    recs = f.head(n).to_dict("records")
    for r in recs:
        r["share"] = (round(r["share"], 4)
                      if r["share"] is not None else None)
        r["z"] = round(r["z"], 2) if r["z"] is not None else None
    return recs


def recent_samples(days: int = 14, per_cat: int = 8,
                   names: list[str] | None = None) -> list[dict]:
    """Matched post texts held locally, highest-engagement first - for the
    LLM to read.  Never rendered raw, never committed."""
    if not os.path.exists(SAMPLES):
        return []
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days * 3)
    rows = []
    for line in open(SAMPLES, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    rows = [r for r in rows
            if pd.Timestamp(r.get("date", "1970-01-01")) >= cutoff]
    if names:
        want = {str(n).lower() for n in names}
        keep = [r for r in rows
                if want & {str(x).lower() for x in
                           (r.get("themes") or []) + (r.get("tickers") or [])}]
        rows = keep or rows
    # most recent day first, then most-upvoted within it: the crowd's own
    # ranking of what mattered
    rows.sort(key=lambda r: (r.get("date", ""), int(r.get("score", 0))),
              reverse=True)
    out, seen = [], defaultdict(int)
    for r in rows:
        if seen[r["category"]] < per_cat:
            seen[r["category"]] += 1
            out.append(r)
    return out


if __name__ == "__main__":
    import sys
    if "--show" in sys.argv:
        for k in ("theme", "ticker"):
            print(f"\n=== most mobilised {k}s "
                  f"(last {RALLY_WINDOW_D}d) ===")
            for r in top_rallies(k, n=10):
                print(f"  {'RALLYING' if r['rallying'] else '        '} "
                      f"{r['name']:<22} hits={r['hits']:<5} "
                      f"share={r['share']!s:<8} z={r['z']!s:<6} "
                      f"{r['top_category']}")
    else:
        scan(rebuild="--rebuild" in sys.argv)
