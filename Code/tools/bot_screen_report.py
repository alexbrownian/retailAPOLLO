"""Run the bot screen over the raw post store and report what it excludes.

The screen runs inside every aggregate build (``ingestion/build_aggregates.py``
and ``ingestion/append_live_abstracted.py``) and writes a summary to
``Reports/bot_screen_last.json``. This tool runs the same screen
on demand so a threshold or weight change can be judged before it is
adopted, and prints examples of what is being excluded.

Usage::

    python Code/tools/bot_screen_report.py                  # trailing 90 days
    python Code/tools/bot_screen_report.py --start 2026-01-01 --examples 15
    python Code/tools/bot_screen_report.py --threshold 0.5  # try a stricter cut

Nothing is written. A copy without the raw store (``--mode aggregates``)
has no posts to screen; the tool then prints the last stored report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT)

from src.config import POSTS_PATH                               # noqa: E402
from ingestion.bot_screen import (apply_screen, format_report,  # noqa: E402
                                  screen_posts)
from src.config import REPORTS_DIR  # noqa: E402

# The examples are post text and author handles, which are not ASCII. A
# console encoding that cannot hold one of those characters would end
# the report on a print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load(start: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(POSTS_PATH)
    cols = [c for c in ("id", "date", "title", "selftext", "source",
                        "author", "subreddit") if c in pf.schema_arrow.names]
    parts = []
    for b in pf.iter_batches(columns=cols, batch_size=200_000):
        df = b.to_pandas()
        df = df[df["date"] >= start]
        if len(df):
            parts.append(df)
    return (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=cols))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default=None,
                    help="first date to screen (default: 90 days ago)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override bot_screen_threshold for this run")
    ap.add_argument("--examples", type=int, default=8,
                    help="how many excluded posts to print")
    args = ap.parse_args(argv)

    if not os.path.exists(POSTS_PATH):
        last = os.path.join(REPORTS_DIR, "bot_screen_last.json")
        print(f"no raw post store at {POSTS_PATH} (aggregates-only copy).")
        if os.path.exists(last):
            with open(last, encoding="utf-8") as f:
                rep = json.load(f)
            print("last stored report:", format_report(rep),
                  f"(checked {rep.get('checked_utc', '?')})")
        return 0

    start = args.start or (pd.Timestamp.today().normalize()
                           - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    posts = _load(start)
    print(f"{len(posts):,} posts since {start}")
    if posts.empty:
        return 0

    scored = screen_posts(posts)
    _, rep = apply_screen(posts, threshold=args.threshold, enabled=True)
    print(format_report(rep))

    thr = rep["threshold"]
    bad = scored[scored["bot_score"] >= thr]
    if len(bad):
        print("\nby forum:")
        if "subreddit" in bad:
            share = (bad["subreddit"].value_counts()
                     / posts["subreddit"].value_counts()).dropna()
            for sub, r in share.sort_values(ascending=False).head(10).items():
                print(f"  {sub:<24} {r:6.1%} of its posts excluded")
        print("\nreason mix (all flagged rows):")
        reasons = (scored.loc[scored["bot_score"] > 0, "bot_reasons"]
                   .str.split("|").explode().value_counts())
        for k, v in reasons.items():
            print(f"  {k:<16} {v:>8,}")
        print(f"\nexamples (score >= {thr:.2f}):")
        text = (bad["title"].fillna("").astype(str) + " "
                + bad.get("selftext", pd.Series("", index=bad.index))
                .fillna("").astype(str)).str.replace(r"\s+", " ", regex=True)
        sample = bad.assign(_t=text).sort_values("bot_score", ascending=False)
        sample = sample.drop_duplicates("_t").head(args.examples)
        for _, r in sample.iterrows():
            who = r.get("author", "") or "?"
            print(f"  [{r['bot_score']:.2f} {r['bot_reasons']}] "
                  f"u/{who} r/{r.get('subreddit', '?')}: {r['_t'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
