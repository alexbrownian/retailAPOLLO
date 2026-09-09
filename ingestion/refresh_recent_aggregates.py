"""Full-mode live fast path: recompute the recent tail of every aggregate.

Rebuilds the last N days of the five aggregate files straight from
``posts.parquet`` and splices that fresh tail onto the untouched history::

    python ingestion/refresh_recent_aggregates.py
    python ingestion/refresh_recent_aggregates.py --days 60
    python ingestion/refresh_recent_aggregates.py --dry-run

A full rebuild (``update_data.py --full``) recomputes every aggregate from
scratch, which is correct but slow: sentiment scoring alone can take 20-40
minutes. ``append_live_abstracted.py`` folds new posts in incrementally
with a seen-ids ledger, which is right for aggregates mode (no raw store),
but a ledger can drift out of sync with full rebuilds where a raw store
exists. This script has neither problem: ``posts.parquet`` is the single
source of truth, so recomputing its most recent days and replacing exactly
those days in the aggregates is always correct, however often it runs. It
uses the same aggregation code (``src.abstracted_data.aggregate_posts``) as
the full rebuild, so live numbers and full-rebuild numbers can never
disagree.

The rule: everything on or after the cutoff date is recomputed; everything
before it is left exactly as the last full rebuild wrote it.
"""

import argparse
import datetime
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:                     # posts contain emoji/links; avoid cp1252 crashes
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import abstracted_data                             # noqa: E402

POSTS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "posts.parquet")
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# 45 days of tail: comfortably more than the 7-day signal lookback and the
# 28-day z warm-up, still small enough to recompute in a minute or two.
DEFAULT_DAYS = 45


def load_recent_posts(cutoff):
    """Load the posts on or after ``cutoff`` with only the aggregator's columns.

    pyarrow pushes the date filter into the file scan, so this reads a tiny
    fraction of the store.

    Args:
        cutoff: ISO date string; rows with ``date >= cutoff`` are returned.

    Returns:
        DataFrame with columns ``id``, ``date``, ``title``, ``selftext``
        and ``source``.
    """
    table = pq.read_table(
        POSTS_PATH,
        columns=["id", "date", "title", "selftext", "source"],
        filters=[("date", ">=", cutoff)],
    )
    return table.to_pandas()


def splice(old, new_tail, cutoff, keys):
    """Replace the tail of an aggregate from ``cutoff`` on with a fresh one.

    Args:
        old: The aggregate as currently stored.
        new_tail: Freshly computed rows covering ``cutoff`` onwards.
        cutoff: ISO date string; ``old`` rows dated before it are kept
            unchanged, everything from it on comes from ``new_tail``.
        keys: Column names to sort the result by.

    Returns:
        The spliced aggregate, sorted by ``keys`` with a fresh index.
    """
    old = old.copy()
    old["date"] = pd.to_datetime(old["date"])
    head = old[old["date"] < pd.to_datetime(cutoff)]
    if len(new_tail):
        new_tail = new_tail.copy()
        new_tail["date"] = pd.to_datetime(new_tail["date"])
    out = pd.concat([head, new_tail], ignore_index=True)
    return out.sort_values(keys).reset_index(drop=True)


def main():
    """Recompute and splice the trailing window of every aggregate file.

    Returns:
        ``0`` on success or when there is nothing to do; ``1`` when
        ``posts.parquet`` is absent (this script only applies in full mode).
    """
    p = argparse.ArgumentParser(
        description="Rebuild the last N days of the aggregates from posts.parquet.")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS,
                   help=f"how many trailing days to recompute (default {DEFAULT_DAYS})")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would change; write nothing")
    args = p.parse_args()

    if not os.path.exists(POSTS_PATH):
        print("no posts.parquet - this script is for full mode only.")
        print("(aggregates mode uses ingestion/append_live_abstracted.py)")
        return 1

    cutoff = (datetime.date.today()
              - datetime.timedelta(days=args.days)).isoformat()
    print(f"recomputing all days >= {cutoff} from posts.parquet")

    posts = load_recent_posts(cutoff)
    print(f"loaded {len(posts):,} posts in the tail window")
    if posts.empty:
        print("no posts in the window - nothing to refresh.")
        return 0
    by_src = posts["source"].value_counts().to_dict()
    print(f"by source: {by_src} | dates {posts['date'].min()} -> {posts['date'].max()}")

    print("aggregating (tickers + themes + sentiment - same code as the notebooks)...")
    new_aggs = abstracted_data.aggregate_posts(posts)

    for name, (kind, keys) in abstracted_data.MERGE_RULES.items():
        path = os.path.join(PROCESSED, name)
        new_tail = new_aggs.get(name)
        if new_tail is None:
            continue
        if not os.path.exists(path):
            print(f"  (skip {name} - not in data/processed; run the full chain once first)")
            continue
        old = pd.read_parquet(path)
        merged = splice(old, new_tail, cutoff, keys)
        changed = len(merged) - len(old)
        if args.dry_run:
            print(f"  would write {name:<40} {len(old):,} -> {len(merged):,} rows "
                  f"({changed:+,})")
            continue
        abstracted_data._safe_write(merged, path)
        print(f"  spliced {name:<40} {len(old):,} -> {len(merged):,} rows ({changed:+,})")

    if args.dry_run:
        print("dry-run: nothing written.")
        return 0

    print("done. next: the analytics stage "
          "(update_data.py runs it: python -m analytics.run_analytics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
