# append_live_abstracted.py
# =========================
# Fold NEW live posts into ABSTRACTED_DATA (the committable aggregates)
# WITHOUT keeping any raw text. This is the live-ingestion path for the
# INTERNAL machine, which does not hold posts.parquet.
#
#   python ingestion/append_live_abstracted.py             fold new posts in
#   python ingestion/append_live_abstracted.py --dry-run   show what WOULD fold in
#
# HOW IT DIFFERS FROM merge_live.py
#   merge_live.py appends raw live posts into posts.parquet (the EXTERNAL
#   machine's raw store). This script skips the raw store entirely: it
#   aggregates the new posts into daily counts + daily sentiment and merges
#   those rows into ABSTRACTED_DATA. Same fetchers, same normalisers, same
#   aggregation code - a different, text-free destination.
#
# THE FLOW
#   1. read live raw (RedditLive / StockTwits / X live) -> 9-column candidates
#      using the project normalisers (identical to merge_live.py)
#   2. keep only candidates that are
#        (a) dated >= LIVE_START - separates them from the committed HISTORICAL
#            block, so the two never overlap
#        (b) NOT already in the local seen-ids ledger ("first seen wins"
#            across re-runs, so running twice folds nothing the second time)
#   3. aggregate the survivors and merge the deltas into ABSTRACTED_DATA
#      (counts add; sentiment recombines weighted by n_posts)
#   4. record the new ids in the ledger, then hydrate ABSTRACTED_DATA ->
#      data/processed so the unchanged notebooks 08/09/10 see the update
#
# THE LEDGER + LIVE_START (data/reference/abstracted_live_meta.json)
#   Kept local and gitignored - post ids are mildly identifying, so they are
#   the one thing never committed. On a fresh machine the ledger starts empty
#   and LIVE_START freezes to (newest committed date) + 1 day, so live only
#   ever adds genuinely new days on top of the committed history.

import argparse
import glob
import io
import json
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Posts can contain emoji/links; keep the Windows console from crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import zstandard                                              # noqa: E402
from src import abstracted_data                               # noqa: E402
from src.clean_data import read_json_lines                    # noqa: E402
from src.reddit_live_data import normalise_reddit_live_records  # noqa: E402
from src.stocktwits_data import normalise_stocktwits          # noqa: E402
from src.x_data import normalise_x_api                        # noqa: E402

RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
META_PATH = os.path.join(PROJECT_ROOT, "data", "reference",
                         "abstracted_live_meta.json")
LEGACY_META = os.path.join(PROJECT_ROOT, "data", "reference",
                           "gic_live_meta.json")   # pre-rename ledger location
MAX_SEEN = 300_000            # ledger cap; newest ids kept (plenty for dedup)

# ---------------------------------------------------------------------
# THE PER-FILE SKIP LEDGER (2026-08-21).
#
# collect_reddit_live() globs EVERY raw file and reads all of them into
# one list, every run. That was fine when data/raw/RedditLive held a
# handful of daily pulls. After the 2026-08-21 backfill it holds 83
# files and ~600 MB compressed - roughly 1.9 M posts - and the ordinary
# `python update_data.py` sat at this step for many minutes, re-reading
# and re-normalising all of it to then DROP nearly all of it at the
# LIVE_START filter. And it gets worse with every backfill.
#
# The insight that makes skipping exact rather than approximate: a file
# whose NEWEST post is older than LIVE_START can never contribute a
# single row, no matter what the seen-id ledger contains, because the
# LIVE_START filter removes all of it by construction. So we record each
# file's max date once and skip it forever after - unless the file
# changes on disk (size or mtime), or LIVE_START itself moves.
#
# We deliberately do NOT skip on "we saw these ids before": seen_ids is
# capped at MAX_SEEN, so an old id can be evicted, and skipping on that
# basis could drop rows that should be folded.
FILE_LEDGER_KEY = "files_scanned"

# The columns aggregate_posts needs from a post.
NEEDED = ["id", "date", "title", "selftext", "source"]


# ---------------- collect candidate posts from the raw live files ----------
def _stat_key(path):
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


def _can_skip(path, ledger, live_start):
    """True when this file provably cannot contribute a single row."""
    rec = ledger.get(os.path.basename(path))
    if not rec or rec.get("stat") != _stat_key(path):
        return False                       # new or changed on disk
    if rec.get("live_start") != live_start:
        return False                       # the window moved; re-check it
    return rec.get("max_date", "9999") < live_start


def _note_file(path, ledger, live_start, max_date):
    ledger[os.path.basename(path)] = {"stat": _stat_key(path),
                                      "live_start": live_start,
                                      "max_date": max_date}


def collect_reddit_live(ledger=None, live_start=None):
    files = sorted(glob.glob(os.path.join(RAW_ROOT, "RedditLive", "*.jsonl.zst")))
    ledger = {} if ledger is None else ledger
    frames, skipped, read_n = [], 0, 0
    for path in files:
        if live_start and _can_skip(path, ledger, live_start):
            skipped += 1
            continue
        recs = read_json_lines(path)
        read_n += 1
        if not recs:
            if live_start:
                _note_file(path, ledger, live_start, "0000-00-00")
            continue
        d = normalise_reddit_live_records(recs)
        if len(d) and live_start:
            _note_file(path, ledger, live_start, str(d["date"].max())[:10])
        if len(d):
            frames.append(d)
        del recs
    if skipped:
        print(f"[reddit ] skipped {skipped} raw file(s) entirely - every post "
              f"in them predates LIVE_START, so they cannot contribute "
              f"(see FILE_LEDGER_KEY note)")
    if not frames:
        return pd.DataFrame(columns=NEEDED)
    df = pd.concat(frames, ignore_index=True)
    print(f"[reddit ] {len(df):,} posts from {read_n} raw file(s) read "
          f"of {len(files)} present")
    return df


def collect_stocktwits():
    files = sorted(glob.glob(os.path.join(RAW_ROOT, "StockTwits", "*.jsonl.zst")))
    messages = []
    for path in files:
        messages.extend(read_json_lines(path))
    if not messages:
        return pd.DataFrame(columns=NEEDED)
    df = normalise_stocktwits(messages)
    print(f"[stwits ] {len(df):,} messages from {len(files)} raw file(s)")
    return df


def collect_x_live():
    path = os.path.join(RAW_ROOT, "X Data", "x_api_live.csv.zst")
    if not os.path.exists(path):
        return pd.DataFrame(columns=NEEDED)
    blob = zstandard.ZstdDecompressor().decompress(open(path, "rb").read())
    raw = pd.read_csv(io.BytesIO(blob), dtype={"id": str})
    df = normalise_x_api(raw)
    print(f"[x live ] {len(df):,} tweets from x_api_live.csv.zst")
    return df


def collect_candidates(ledger=None, live_start=None):
    parts = [collect_reddit_live(ledger, live_start), collect_stocktwits(),
             collect_x_live()]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=NEEDED)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset="id", keep="first")
    # keep only what the aggregator needs; date as plain 'YYYY-MM-DD'
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    for c in ("id", "title", "selftext", "source"):
        df[c] = df[c].fillna("").astype(str)
    return df[NEEDED]


# ---------------- ledger + LIVE_START -------------------------------------
def load_meta():
    # migrate the pre-rename ledger transparently, so dedup history survives
    if not os.path.exists(META_PATH) and os.path.exists(LEGACY_META):
        try:
            os.replace(LEGACY_META, META_PATH)
            print("[setup ] migrated ledger gic_live_meta.json -> "
                  "abstracted_live_meta.json")
        except OSError:
            pass
    if os.path.exists(META_PATH):
        try:
            return json.load(open(META_PATH, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {}


def save_meta(meta):
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    # keep only the newest MAX_SEEN ids so the file cannot grow forever
    ids = meta.get("seen_ids", [])
    meta["seen_ids"] = ids[-MAX_SEEN:]
    tmp = META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp, META_PATH)


def newest_committed_date():
    """Newest date already in ABSTRACTED_DATA, or None if empty."""
    path = os.path.join(abstracted_data.ABSTRACTED_DIR, abstracted_data.TICKER_COUNTS)
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path, columns=["date"])
    if len(d) == 0:
        return None
    return pd.to_datetime(d["date"]).max()


def resolve_live_start(meta):
    """Frozen once: the first day live ingestion owns. Posts before it belong
    to the committed historical block and must never be re-folded here."""
    live_start = meta.get("live_start")
    if live_start:
        return live_start
    newest = newest_committed_date()
    if newest is None:
        live_start = "1970-01-01"        # empty store -> accept everything
    else:
        live_start = (newest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    meta["live_start"] = live_start      # freeze so it never moves again
    print(f"[setup ] LIVE_START frozen at {live_start} "
          f"(newest committed day: {None if newest is None else newest.date()})")
    return live_start


# ---------------- main -----------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Fold new live posts into ABSTRACTED_DATA (text-free aggregates).")
    p.add_argument("--dry-run", action="store_true",
                   help="report what WOULD be folded in; write nothing")
    p.add_argument("--no-hydrate", action="store_true",
                   help="update ABSTRACTED_DATA but skip the copy into data/processed")
    args = p.parse_args()

    # ---- 1. gather + 2. filter
    #
    # The ledger and LIVE_START are resolved BEFORE reading anything, so
    # collect_* can skip files that cannot possibly contribute. This used
    # to happen after the read, which is why the read was unconditional.
    meta = load_meta()
    live_start = resolve_live_start(meta)
    file_ledger = meta.get(FILE_LEDGER_KEY, {})

    cand = collect_candidates(file_ledger, live_start)
    meta[FILE_LEDGER_KEY] = file_ledger
    if cand.empty:
        print("no NEW live raw posts to consider - nothing to fold.")
        if not args.dry_run:
            save_meta(meta)        # keep the skip ledger even on a no-op
        return 0

    seen = set(meta.get("seen_ids", []))

    in_window = cand[cand["date"] >= live_start]
    fresh = in_window[~in_window["id"].isin(seen)].reset_index(drop=True)

    dropped_old = len(cand) - len(in_window)
    dropped_seen = len(in_window) - len(fresh)
    print(f"[filter] {len(cand):,} candidates | dropped {dropped_old:,} before "
          f"LIVE_START ({live_start}) | dropped {dropped_seen:,} already seen "
          f"| {len(fresh):,} NEW")
    if len(fresh):
        print(f"[filter] new by source: {fresh['source'].value_counts().to_dict()}"
              f" | dates {fresh['date'].min()} -> {fresh['date'].max()}")

    if fresh.empty:
        print("nothing new to fold - ABSTRACTED_DATA already up to date.")
        # Persist the skip ledger anyway. Without this the very common
        # "nothing new" run would learn nothing, and the next run would
        # re-read every file all over again - which is exactly the cost
        # this ledger exists to remove.
        if not args.dry_run:
            save_meta(meta)
        return 0

    if args.dry_run:
        print("\n--dry-run: nothing written. The above is what WOULD be folded in.")
        return 0

    # ---- 3. aggregate the new posts and merge the deltas
    print("\n--- aggregating new posts (tickers + sentiment) ---")
    new_aggs = abstracted_data.aggregate_posts(fresh)
    print("--- merging into ABSTRACTED_DATA ---")
    abstracted_data.merge_into_abstracted(new_aggs)

    # ---- 4. record the new ids, then hydrate for the local notebooks
    meta["seen_ids"] = meta.get("seen_ids", []) + fresh["id"].tolist()
    save_meta(meta)
    print(f"[ledger] +{len(fresh):,} ids "
          f"(ledger now holds {len(meta['seen_ids']):,})")

    if not args.no_hydrate:
        print("\n--- hydrate ABSTRACTED_DATA -> data/processed ---")
        abstracted_data.hydrate()

    print("\ndone. next: recompute the analytics "
          "(update_data.py does this: python -m analytics.run_analytics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
