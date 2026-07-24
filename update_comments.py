"""
update_comments.py
==================
The dedicated Reddit-comments + influence-board runner (desk decision,
2026-07-24: comments left the daily pipeline - they are the slow fetch -
and live here instead).

    python update_comments.py                      # incremental (watermark)
    python update_comments.py --lookback-days 30   # wider first window
    python update_comments.py --backfill 2026-01-01 2026-07-01
    python update_comments.py --estimate           # print the estimate, exit

WHAT ONE RUN DOES
    1. prints an upfront TIME ESTIMATE (from the per-subreddit watermarks
       - the honest driver of runtime is how much catching-up is owed);
    2. runs ingestion/fetch_reddit_comments.py (watermarked, resumable -
       Ctrl-C is always safe, the seen-file dedups on the next run);
    3. runs analytics.influence.update(): parse the new raw files, extend
       the committed text-free store, re-judge matured calls, rescore the
       board.

WHY THE ESTIMATE IS WHAT IT IS
    The Arctic Shift API is crawled politely: 100 comments/page, 1s
    pause, subreddits in sequence. Comments run 10-50x post volume, so
    a FIRST run (no watermark) over the default 3-day window costs
    roughly 10-25 minutes across the panel; a daily incremental run owes
    only the hours since the last watermark and typically lands in the
    1-4 minute range. Backfills are days-per-half-year territory for the
    busy subs - which is exactly why the desk scoped comments live-first.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

WM_FILE = os.path.join(ROOT, "data", "reference",
                       "reddit_comments_watermark.json")
SUBS_FILE = os.path.join(ROOT, "ingestion", "finance_subreddits.txt")


def _n_subs() -> int:
    try:
        with open(SUBS_FILE, encoding="utf-8") as f:
            return sum(1 for line in f
                       if line.strip() and not line.startswith("#"))
    except OSError:
        return 17


def estimate(lookback_days: int, backfill: list | None) -> str:
    """A human-honest runtime estimate. Drivers: number of subreddits,
    watermark age (how much catching-up is owed), and the API's polite
    pace (100/page, 1s/pause). Deliberately given as a RANGE - comment
    volume per sub varies 100x between a quiet Tuesday and a mania."""
    n = _n_subs()
    if backfill:
        return (f"BACKFILL {backfill[0]} -> {backfill[1]} across {n} "
                "subreddits: expect HOURS for busy subs (1s/page at 100/"
                "page; the desk's scope is the current year at most). "
                "Safe to Ctrl-C and resume any time.")
    marks = {}
    if os.path.exists(WM_FILE):
        try:
            marks = json.load(open(WM_FILE, encoding="utf-8"))
        except (ValueError, OSError):
            marks = {}
    if not marks:
        return (f"FIRST RUN (no watermark yet): {n} subreddits x "
                f"{lookback_days}d of comments at the API's polite pace "
                "- expect roughly 10-25 minutes. Every later run is "
                "incremental and typically takes 1-4 minutes.")
    oldest = min(int(v) for v in marks.values())
    owed_h = max(0.0, (time.time() - oldest) / 3600)
    if owed_h <= 30:
        return (f"incremental: watermarks are ~{owed_h:.0f}h old across "
                f"{n} subreddits - expect roughly 1-4 minutes.")
    return (f"incremental but stale: oldest watermark is ~{owed_h/24:.1f} "
            f"days old across {n} subreddits - expect roughly "
            f"{max(2, int(owed_h/24) * 2)}-{max(6, int(owed_h/24) * 6)} "
            "minutes (proportional to the catch-up owed).")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fetch Reddit comments + update the influence board "
                    "(the slow, optional half of ingestion)")
    p.add_argument("--lookback-days", type=int, default=3,
                   help="live window when no watermark exists (default 3)")
    p.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                   help="historical range YYYY-MM-DD YYYY-MM-DD")
    p.add_argument("--estimate", action="store_true",
                   help="print the time estimate and exit (no API calls)")
    p.add_argument("--skip-influence", action="store_true",
                   help="fetch only; do not update the influence store")
    args = p.parse_args()

    print("=" * 60)
    print("COMMENT PULL + INFLUENCE UPDATE")
    print(f"  estimate: {estimate(args.lookback_days, args.backfill)}")
    print("=" * 60)
    if args.estimate:
        return 0

    # ---- 1. fetch (child process, output streamed live) ----
    cmd = [sys.executable,
           os.path.join(ROOT, "ingestion", "fetch_reddit_comments.py"),
           "--lookback-days", str(args.lookback_days)]
    if args.backfill:
        cmd += ["--backfill", *args.backfill]
    t0 = time.time()
    code = subprocess.call(cmd, cwd=ROOT)
    print(f"fetch finished in {(time.time()-t0)/60:.1f} min "
          f"(exit {code})")
    if code != 0:
        print("fetch did not complete cleanly - the watermark only "
              "advances for completed subreddits, so simply re-run; "
              "nothing is duplicated.")

    # ---- 2. influence board update (ingest new raw -> extend store) ----
    if args.skip_influence:
        print("influence update skipped (--skip-influence)")
        return code
    print("--- influence board update ---")
    from analytics.influence import update as influence_update
    t1 = time.time()
    influence_update()
    print(f"influence update finished in {time.time()-t1:.0f}s")
    print("done. Board: python -m analytics.influence --top 20 | "
          "dashboard: Influence tracker tab")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
