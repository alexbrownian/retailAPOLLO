"""Live-data smoke test: is fresh data flowing, and where does it stop?

Reports, layer by layer, whether fresh data is moving through the
pipeline. Read-only::

    python Code/ingestion/check_live_ingestion.py

It checks the four layers in flow order:

1. FETCH: are the raw live files growing? (StockTwits daily file,
   ``x_api_live.csv.zst``, Reddit live files)
2. DATASET: the newest post date in ``posts.parquet``, per source (full
   mode only; aggregates mode has no raw store and reports that as
   expected).
3. DERIVED: how fresh the derived analytics outputs are (counts,
   sentiment, signals).
4. RECORD: whether signal snapshots and a pipeline log are present.

Verdicts: ``[LIVE]`` means data within 2 days of today; ``[OK]`` an
expected state; ``[STALE]``/``[MISSING]`` means investigate (the message
says what to run). A layer that is stale while the one before it is
fresh points at the step between them.
"""

import datetime
import glob
import io
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
import zstandard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)        # run as a script from anywhere; src/ is under ROOT

from src.config import DATA_DIR, LOG_DIR  # noqa: E402

TODAY = datetime.date.today()


def verdict(day_str):
    """Return a freshness tag for an ISO date string (``[MISSING]`` if empty)."""
    if not day_str:
        return "[MISSING]"
    age = (TODAY - datetime.date.fromisoformat(str(day_str)[:10])).days
    return f"[LIVE +{age}d ago]" if age <= 2 else f"[STALE {age}d old]"


def section(title):
    """Print an underlined section heading."""
    print("\n" + title + "\n" + "-" * len(title))


# ---- 1. FETCH layer ----
section("1. FETCH - raw live files")
st_files = sorted(glob.glob(os.path.join(DATA_DIR, "raw", "StockTwits", "*.jsonl.zst")))
if st_files:
    latest = st_files[-1]
    blob = zstandard.ZstdDecompressor().decompress(open(latest, "rb").read())
    n_lines = blob.count(b"\n")
    day = os.path.basename(latest).replace("stocktwits_", "").replace(".jsonl.zst", "")
    print(f"StockTwits : {verdict(day)} latest file {os.path.basename(latest)} "
          f"({n_lines:,} raw messages)")
else:
    print("StockTwits : [MISSING] no raw files - run "
          "python Code/ingestion/fetch_all.py")

x_live = os.path.join(DATA_DIR, "raw", "X Data", "x_api_live.csv.zst")
if os.path.exists(x_live):
    blob = zstandard.ZstdDecompressor().decompress(open(x_live, "rb").read())
    df = pd.read_csv(io.BytesIO(blob))
    newest = str(df["created_at"].max())[:10] if len(df) else None
    print(f"X live     : {verdict(newest)} {len(df):,} tweets accumulated")
else:
    print("X live     : [NO DATA YET] no X fetch has run on this machine")

rl_files = sorted(glob.glob(os.path.join(DATA_DIR, "raw", "RedditLive", "*.jsonl.zst")))
if rl_files:
    latest = rl_files[-1]
    blob = zstandard.ZstdDecompressor().decompress(open(latest, "rb").read())
    day = os.path.basename(latest).replace("reddit_live_", "").replace(".jsonl.zst", "")
    print(f"Reddit live: {verdict(day)} latest file {os.path.basename(latest)} "
          f"({blob.count(chr(10).encode()):,} raw posts)")
else:
    print("Reddit live: [NO DATA YET] verify the key with "
          "'python Code/ingestion/test_fetchlayer.py', then run "
          "'python Code/ingestion/fetch_all.py' to start accumulating")

# ---- 2. DATASET layer ----
section("2. DATASET - posts.parquet, newest post per source")
posts = os.path.join(DATA_DIR, "processed", "posts.parquet")
if not os.path.exists(posts):
    print("[OK - aggregates mode] no posts.parquet here; live posts fold "
          "into Data/abstracted via ingestion/append_live_abstracted.py")
else:
    try:
        cols = pq.ParquetFile(posts).schema_arrow.names
        read_cols = ["date"] + (["source"] if "source" in cols else [])
        t = pq.read_table(posts, columns=read_cols).to_pandas()
        if "source" in t.columns:
            for src_name, grp in t.groupby("source"):
                print(f"{src_name:<11}: {verdict(grp['date'].max())} "
                      f"newest post {grp['date'].max()}")
        else:
            print(f"all        : {verdict(t['date'].max())} newest post {t['date'].max()}")
        print("(live Reddit, X and StockTwits raw all APPEND into posts.parquet via "
              "merge_live.py - run it, or update_data.py, to pull them in.)")
    except Exception as exc:
        print(f"[UNREADABLE] posts.parquet could not be opened ({exc}).")
        print("If a merge/swap is mid-flight or another program holds the file,")
        print("close it and re-run this check.")

# ---- 3. DERIVED layer ----
section("3. DERIVED - analytics outputs")
for label, fname, date_col in [
        ("counts        ", "daily_ticker_counts.parquet", "date"),
        ("tick sentiment", "daily_ticker_sentiment.parquet", "date"),
        ("theme sentimnt", "daily_theme_sentiment.parquet", "date"),
        ("trade signals ", "trade_signals.parquet", "signal_date")]:
    path = os.path.join(DATA_DIR, "processed", fname)
    if not os.path.exists(path):
        print(f"{label}: [MISSING] - run update_data.py")
        continue
    try:
        df = pd.read_parquet(path)
        newest = str(df[date_col].max())[:10] if len(df) else None
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%m-%d %H:%M")
        print(f"{label}: {verdict(newest)} newest row {newest} (file written {mtime})")
    except Exception as exc:
        print(f"{label}: [UNREADABLE] ({str(exc)[:60]}...) - file busy or mid-write")

# ---- 4. RECORD layer ----
section("4. RECORD - snapshots & logs")
snaps = sorted(glob.glob(os.path.join(DATA_DIR, "processed",
                                      "signal_snapshots", "*.parquet")))
print(f"snapshots  : {len(snaps)} files"
      + (f", latest {os.path.basename(snaps[-1])}" if snaps else
         " - none yet (update_data.py creates them)"))
logs = sorted(glob.glob(os.path.join(LOG_DIR, "run_*.log")))
if logs:
    tail = open(logs[-1], encoding="utf-8").read().strip().splitlines()[-1]
    print(f"last log   : {os.path.basename(logs[-1])} | last line: {tail}")
else:
    print("last log   : none - update_data.py has never run")

print("\nRule of thumb: layer N stale but layer N-1 fresh => the step between "
      "them did not run.")
