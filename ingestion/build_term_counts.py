# build_term_counts.py
# ====================
# Build ABSTRACTED_DATA/daily_term_counts.parquet - the text-free daily term
# frequencies that let EMERGING-TERM detection run on the internal machine.
#
#     python ingestion/build_term_counts.py
#
# EXTERNAL machine (needs posts.parquet). Runs once over the last RETAIN_DAYS
# of history; after that, live folds keep the file current on any machine.
# Committing ABSTRACTED_DATA ships it to the HP like the other five.

import os
import sys

import pandas as pd
import pyarrow.parquet as pq

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.terms import (terms_in_text, TOTAL_MARKER, RETAIN_DAYS,       # noqa: E402
                       MIN_PER_DAY_WORD, MIN_PER_DAY_PAIR)
from src import abstracted_data                                        # noqa: E402

POSTS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "posts.parquet")
OUT_PATH = os.path.join(abstracted_data.ABSTRACTED_DIR, "daily_term_counts.parquet")


def main():
    if not os.path.exists(POSTS_PATH):
        print("no posts.parquet - run this on the external machine.")
        return 1

    pf = pq.ParquetFile(POSTS_PATH)

    # newest date first, so the retention floor is known before streaming
    newest = None
    for batch in pf.iter_batches(columns=["date"], batch_size=200_000):
        s = max(str(d)[:10] for d in batch.column("date").to_pylist())
        newest = s if newest is None or s > newest else newest
    floor = (pd.Timestamp(newest) - pd.Timedelta(days=RETAIN_DAYS)
             ).strftime("%Y-%m-%d")
    print(f"counting terms from {floor} -> {newest} "
          f"({pf.metadata.num_rows:,} posts total, older ones skipped)")

    # accumulate across ALL batches first, apply the per-day floors ONCE at
    # the end - a day whose posts straddle two batches must not lose terms
    counts: dict = {}
    day_totals: dict = {}
    seen = 0
    for batch in pf.iter_batches(columns=["date", "title", "selftext"],
                                 batch_size=100_000):
        dates = [str(d)[:10] for d in batch.column("date").to_pylist()]
        titles = batch.column("title").to_pylist()
        bodies = batch.column("selftext").to_pylist()
        for date, title, body in zip(dates, titles, bodies):
            if date < floor:
                continue
            day_totals[date] = day_totals.get(date, 0) + 1
            text = (title or "") + " " + (body or "")
            for term in terms_in_text(text):
                key = (date, term)
                counts[key] = counts.get(key, 0) + 1
        seen += len(dates)
        if seen % 1_000_000 < 100_000:
            print(f"  ... {seen:,} posts scanned", flush=True)

    rows = []
    for (date, term), n in counts.items():
        min_needed = MIN_PER_DAY_PAIR if " " in term else MIN_PER_DAY_WORD
        if n >= min_needed:
            rows.append((date, term, n))
    for date, n in day_totals.items():
        rows.append((date, TOTAL_MARKER, n))

    daily = (pd.DataFrame(rows, columns=["date", "term", "mention_count"])
             .sort_values(["date", "term"]).reset_index(drop=True))
    daily["date"] = pd.to_datetime(daily["date"])
    os.makedirs(abstracted_data.ABSTRACTED_DIR, exist_ok=True)
    abstracted_data._safe_write(daily, OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"wrote {len(daily):,} rows ({daily['date'].min().date()} -> "
          f"{daily['date'].max().date()}, {daily['term'].nunique():,} terms, "
          f"{size_mb:.1f} MB) -> {OUT_PATH}")
    if size_mb > 25:
        print("WARNING: file exceeds the 25 MB commit guard - raise "
              "MIN_PER_DAY_* in src/terms.py or lower RETAIN_DAYS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
