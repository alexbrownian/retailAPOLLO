# merge_live.py
# =============
# Append the RAW live files written by the ingestion/ fetchers into
# data/processed/posts.parquet, so live Reddit, X and StockTwits posts reach
# the signals - not just the raw folder. EXTERNAL machine only (the internal
# machine has no raw store; it uses ingestion/append_live_abstracted.py).
#
# Sources merged (each self-skips if its raw folder is empty):
#   data/raw/RedditLive/*.jsonl.zst    -> src.reddit_live_data
#   data/raw/StockTwits/*.jsonl.zst    -> src.stocktwits_data
#   data/raw/X Data/x_api_live.csv.zst -> src.x_data (normalise_x_api)
#
# THE DEDUP CONTRACT ("first seen wins"): a candidate post is dropped if its
# id is already in posts.parquet. The script is append-only and idempotent -
# it only ever adds posts it has never seen and never revises history.
#   (Distinct id prefixes keep the sources apart: Reddit base36, 'x_' tweets,
#    'st_' StockTwits - collisions are impossible across sources.)
#
# vs add_x_data.py: that script REBUILDS the X block from the frozen
# HuggingFace dumps (historical backfill). This one only appends fresh live
# posts. They do not conflict: merge_live never drops rows.
#
# HOW (memory-safe - the full store is never loaded at once):
#   1. normalise every raw live file -> candidate rows (standard 9 columns)
#   2. read ONLY the id column to find which candidates are new; if none,
#      exit immediately (no rewrite - the common no-op case costs seconds)
#   3. otherwise stream posts.parquet row group by row group into a new
#      file and append the new candidates as a final date-sorted block
#   4. verify row counts + schema, then swap the new file in atomically
#
# Run from anywhere:  python ingestion/merge_live.py
#   --dry-run   normalise + count what WOULD be added, write nothing
#   --posts / --raw-root / --out   override default paths

import argparse
import glob
import io
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Posts can contain emoji/links; the default Windows console codec (cp1252)
# would crash on them. Print UTF-8 and replace anything unprintable.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.clean_data import read_json_lines                 # noqa: E402
from src.reddit_live_data import normalise_reddit_live_records  # noqa: E402
from src.stocktwits_data import normalise_stocktwits       # noqa: E402
from src.x_data import normalise_x_api                     # noqa: E402

DEFAULT_POSTS = os.path.join(PROJECT_ROOT, "data", "processed", "posts.parquet")
DEFAULT_RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("date", pa.string()),
    ("author", pa.string()),
    ("score", pa.int64()),
    ("subreddit", pa.string()),
    ("title", pa.string()),
    ("selftext", pa.string()),
    ("num_comments", pa.int64()),
    ("source", pa.string()),
])
COLS = [f.name for f in SCHEMA]


# ---------------- collect candidate rows from the raw live files ----------
def collect_reddit_live(raw_root):
    files = sorted(glob.glob(os.path.join(raw_root, "RedditLive", "*.jsonl.zst")))
    records = []
    for path in files:
        records.extend(read_json_lines(path))
    if not records:
        return pd.DataFrame(columns=COLS)
    df = normalise_reddit_live_records(records)
    print(f"[reddit ] {len(df):,} posts from {len(files)} raw file(s)")
    return df


def collect_stocktwits(raw_root):
    files = sorted(glob.glob(os.path.join(raw_root, "StockTwits", "*.jsonl.zst")))
    messages = []
    for path in files:
        messages.extend(read_json_lines(path))
    if not messages:
        return pd.DataFrame(columns=COLS)
    df = normalise_stocktwits(messages)
    print(f"[stwits ] {len(df):,} messages from {len(files)} raw file(s)")
    return df


def collect_x_live(raw_root):
    path = os.path.join(raw_root, "X Data", "x_api_live.csv.zst")
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    blob = zstandard.ZstdDecompressor().decompress(open(path, "rb").read())
    raw = pd.read_csv(io.BytesIO(blob), dtype={"id": str})
    df = normalise_x_api(raw)
    print(f"[x live ] {len(df):,} tweets from x_api_live.csv.zst")
    return df


def collect_candidates(raw_root):
    parts = [collect_reddit_live(raw_root),
             collect_stocktwits(raw_root),
             collect_x_live(raw_root)]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset="id", keep="first")
    # Lock the dtypes so the arrow table matches SCHEMA exactly.
    for c in ("id", "date", "author", "subreddit", "title", "selftext", "source"):
        df[c] = df[c].fillna("").astype(str)
    for c in ("score", "num_comments"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df[COLS]


# ---------------- conform an existing row group to the output schema ------
def conform(t):
    if "source" not in t.schema.names:
        t = t.append_column("source", pa.array(["reddit"] * t.num_rows, pa.string()))
    return t.select(COLS).cast(SCHEMA)


# ---------------- a human-readable snapshot of what was pulled -------------
def print_snapshot(df):
    """Sanity view: posts per source, the reddit subreddit mix, and the
    newest few posts from each source - confirms a run really pulled
    live data (and what)."""
    print("\n" + "=" * 64)
    print("LIVE INGESTION SNAPSHOT  (what the fetchers pulled into raw)")
    print("=" * 64)
    print("posts by source:")
    for src, n in df["source"].value_counts().items():
        print(f"  {src:<12} {n:>7,}")
    print(f"  {'TOTAL':<12} {len(df):>7,}")

    red = df[df["source"] == "reddit"]
    if len(red):
        counts = red["subreddit"].value_counts()
        top8 = " | ".join(f"r/{s} {n}" for s, n in counts.head(8).items())
        print(f"\nreddit subreddits ({len(counts)}): {top8}")

    for src in ("reddit", "x", "stocktwits"):
        sub = df[df["source"] == src]
        if not len(sub):
            continue
        disp = sub.sort_values("date").tail(3)[["date", "title"]].copy()
        disp["title"] = (disp["title"].astype(str)
                         .str.replace(r"\s+", " ", regex=True).str.slice(0, 70))
        print(f"\n--- {src}: newest 3 of {len(sub):,} ---")
        print(disp.to_string(index=False, header=False))
    print("=" * 64 + "\n")


def main():
    p = argparse.ArgumentParser(description="Append live raw posts into posts.parquet")
    p.add_argument("--posts", default=DEFAULT_POSTS)
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--out", default=None, help="default: <posts dir>/posts_live_merged.parquet")
    p.add_argument("--dry-run", action="store_true",
                   help="report what WOULD be added, write nothing")
    args = p.parse_args()

    if not os.path.exists(args.posts):
        sys.exit(f"posts.parquet not found at {args.posts} - build the dataset first.")

    # ---- 1. normalise every raw live file
    cand = collect_candidates(args.raw_root)
    if cand.empty:
        print("no live raw posts found - nothing to merge.")
        return 0
    print(f"[all   ] {len(cand):,} candidate live posts "
          f"| dates {cand['date'].min()} -> {cand['date'].max()}")
    print_snapshot(cand)

    # ---- 2. FAST PRE-CHECK: read only the id column to find what is new.
    #         When everything was already merged (the common case for a
    #         re-run) this exits without touching the store at all.
    seen = set(pq.read_table(args.posts, columns=["id"]).column("id").to_pylist())
    fresh = cand[~cand["id"].isin(seen)].reset_index(drop=True)
    if fresh.empty:
        print(f"all {len(cand):,} candidate posts are already in posts.parquet "
              "- nothing new to append (store untouched).")
        return 0
    by_src = fresh.groupby("source").size().to_dict()
    print(f"[new   ] {len(fresh):,} of {len(cand):,} candidates are NEW: {by_src}")

    if args.dry_run:
        print("dry-run: nothing written.")
        return 0

    # ---- 3. stream the store through to a new file and append the fresh block
    pf = pq.ParquetFile(args.posts)
    out_path = args.out or os.path.join(os.path.dirname(args.posts),
                                        "posts_live_merged.parquet")
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
    kept = 0
    n_groups = pf.metadata.num_row_groups
    for i in range(n_groups):
        t = conform(pf.read_row_group(i))
        writer.write_table(t)
        kept += t.num_rows
        # one progress line every ~10 groups keeps the output readable
        if (i + 1) % 10 == 0 or i + 1 == n_groups:
            print(f"  copying store... {i + 1}/{n_groups} row groups "
                  f"({kept:,} rows)", flush=True)
    writer.write_table(pa.Table.from_pandas(fresh, schema=SCHEMA, preserve_index=False))
    writer.close()
    try:
        pf.close()
    except AttributeError:
        del pf

    # ---- 4. verify, then swap. Release the verify handle FIRST - Windows
    #         refuses to rename a file any process has open.
    new = pq.ParquetFile(out_path)
    total_rows, schema_names = new.metadata.num_rows, new.schema_arrow.names
    try:
        new.close()
    except AttributeError:
        del new
    assert schema_names == COLS, "schema mismatch"
    assert total_rows == kept + len(fresh), (
        f"row count mismatch: {total_rows} != {kept} + {len(fresh)}")
    print(f"verified: {kept:,} existing + {len(fresh):,} new = {total_rows:,} rows")

    try:
        # os.replace overwrites the target atomically (on Windows too) - no
        # separate delete, so there is never a moment with no posts file.
        os.replace(out_path, args.posts)
    except PermissionError:
        print()
        print("!" * 68)
        print("The merge SUCCEEDED but the final rename failed: another program")
        print("has posts.parquet open (usually a running Jupyter kernel).")
        print("Close all kernels / Excel / viewers, then re-run this script")
        print("(fast - it only appends), OR rename by hand:")
        print(f'  del "{args.posts}"')
        print(f'  ren "{out_path}" "{os.path.basename(args.posts)}"')
        print("!" * 68)
        return 1
    print("swapped in:", args.posts)
    print("next: re-run the notebook chain (update_data.py does this).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
