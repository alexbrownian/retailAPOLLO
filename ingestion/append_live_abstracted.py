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

# The columns aggregate_posts needs from a post.
NEEDED = ["id", "date", "title", "selftext", "source"]


# ---------------- collect candidate posts from the raw live files ----------
def collect_reddit_live():
    files = sorted(glob.glob(os.path.join(RAW_ROOT, "RedditLive", "*.jsonl.zst")))
    records = []
    for path in files:
        records.extend(read_json_lines(path))
    if not records:
        return pd.DataFrame(columns=NEEDED)
    df = normalise_reddit_live_records(records)
    print(f"[reddit ] {len(df):,} posts from {len(files)} raw file(s)")
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


def collect_candidates():
    parts = [collect_reddit_live(), collect_stocktwits(), collect_x_live()]
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
    cand = collect_candidates()
    if cand.empty:
        print("no live raw posts found - nothing to fold.")
        return 0

    meta = load_meta()
    live_start = resolve_live_start(meta)
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

    print("\ndone. next: re-run notebooks 08/09/10 (update_data.py does this).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
