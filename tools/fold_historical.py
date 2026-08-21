# fold_historical.py
# ==================
# Fold HISTORICAL posts into ABSTRACTED_DATA on a machine that has no
# posts.parquet - the missing piece for the 2023-2025 coverage gap.
#
#     python tools/fold_historical.py --arctic            # files already on disk
#     python tools/fold_historical.py --dumps "data/raw/dumps/*_submissions.zst"
#     python tools/fold_historical.py --arctic --dry-run  # count, write nothing
#
# WHY THIS EXISTS (desk bug report 2026-08-18)
# --------------------------------------------
# `tools/backfill_reddit.py` pulled 904,026 posts across 16 monthly chunks
# (45.7 hours) and NOT ONE of them reached the aggregates. The reason is
# not a bug in the backfill: it is that this machine is the INTERNAL one
# (no posts.parquet), so the only fold-in path is
# `ingestion/append_live_abstracted.py` - and that script keeps only
# candidates dated >= LIVE_START, where LIVE_START is frozen at
# (newest committed day + 1). Every backfilled 2023-2025 post is older
# than LIVE_START, so it is dropped silently, by design. `update_data.py
# --full` cannot rescue it either: it rebuilds from posts.parquet, which
# only exists on the external machine, and its guard aborts rather than
# revert committed months.
#
# That filter is CORRECT for live ingestion - it is what stops the daily
# run re-folding the committed historical block and double-counting it.
# This script is the deliberate, separate door for the opposite job:
# adding history that was never there.
#
# WHAT IT DOES NOT CHANGE
#   * LIVE_START is read, never written. The live path keeps its guard.
#   * Text never lands on disk. Posts are streamed, aggregated in memory
#     and only the daily COUNT/SENTIMENT rows are written - the same
#     text-free boundary `append_live_abstracted.py` works under, and the
#     reason this is safe to run on the committable store.
#   * Counting rules are not reimplemented: aggregation goes through
#     `abstracted_data.aggregate_posts`, the same function the live fold
#     and the full rebuild use, so a folded month is indistinguishable
#     from a month that was there all along.
#
# IDEMPOTENCY - READ THIS BEFORE RE-RUNNING
# -----------------------------------------
# `merge_into_abstracted` ADDS counts. Folding the same posts twice would
# inflate them, and no downstream number would look obviously wrong. So
# every (file, month) block that is folded is recorded in
# data/reference/historical_fold_ledger.json, and a block already in the
# ledger is skipped. A crash mid-file therefore costs only the months
# that had not yet flushed, never a double count. `--force` overrides the
# ledger and is exactly as dangerous as it sounds.
#
# WHERE TO GET THE FILES
#   * ALREADY ON DISK: data/raw/RedditLive/*.jsonl.zst - what the backfill
#     runner pulled. Use --arctic. Nothing to download.
#   * TORRENT DUMPS: the per-subreddit monthly archives this project was
#     originally built from - `<subreddit>_submissions.zst`, the naming
#     documented in ingestion/finance_subreddits.txt. They are ordinary
#     NDJSON inside long-window zstd; src.clean_data.read_json_lines
#     already streams that format (max_window_size=2**31), so a dump can
#     be dropped straight into data/raw/dumps/ and pointed at with
#     --dumps. Submissions are what the historical block was built from;
#     comment archives are accepted with --include-comments but change
#     what a "post" means, so they are off by default.

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import abstracted_data                               # noqa: E402
from src.clean_data import read_json_lines                    # noqa: E402
from src import config                                        # noqa: E402

LEDGER_PATH = os.path.join(PROJECT_ROOT, "data", "reference",
                           "historical_fold_ledger.json")
LIVE_META = os.path.join(PROJECT_ROOT, "data", "reference",
                         "abstracted_live_meta.json")
ARCTIC_GLOB = os.path.join(PROJECT_ROOT, "data", "raw", "RedditLive",
                           "*.jsonl.zst")
# aggregate_posts needs exactly these; nothing else is carried, so no text
# can leak into a store by accident
NEEDED = ["id", "date", "title", "selftext", "source"]


# ---------------------------------------------------------------- ledger
def load_ledger():
    if os.path.exists(LEDGER_PATH):
        try:
            return json.load(open(LEDGER_PATH, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"blocks": {}}


def save_ledger(led):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(led, f, indent=1)
    os.replace(tmp, LEDGER_PATH)


def live_start():
    """The first day LIVE ingestion owns. Anything on or after it is the
    live path's business and must not be folded here - that is the double
    count this whole script exists to avoid."""
    if os.path.exists(LIVE_META):
        try:
            ls = json.load(open(LIVE_META, encoding="utf-8")).get("live_start")
            if ls:
                return ls
        except (ValueError, OSError):
            pass
    return None


# ------------------------------------------------------------ subreddits
def allowed_subreddits():
    """The 17 finance subreddits the committed history was built from -
    folding anything else in would change what the counts MEAN, not just
    how many there are."""
    try:
        with open(config.SUBREDDITS_FILE, encoding="utf-8") as f:
            return {ln.strip().lower() for ln in f
                    if ln.strip() and not ln.startswith("#")}
    except OSError:
        return set()


# -------------------------------------------------------------- records
def to_row(rec):
    """One raw record -> the 5 columns the aggregator needs, or None.

    Handles both shapes with the same code: a torrent SUBMISSION
    (title + selftext) and a COMMENT (body). Timestamps arrive as unix
    seconds in the dumps and as either seconds or ISO in the project's
    own pulls, so both are accepted."""
    if not isinstance(rec, dict):
        return None
    rid = rec.get("id") or rec.get("name") or ""
    if not rid:
        return None
    raw_ts = rec.get("created_utc", rec.get("created", ""))
    date = ""
    try:
        secs = float(raw_ts)
        if secs > 1_000_000_000:
            date = pd.Timestamp(secs, unit="s").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw_ts, errors="coerce", utc=True)
        if not pd.isna(parsed):
            date = parsed.strftime("%Y-%m-%d")
    if not date:
        return None
    title = str(rec.get("title") or "")
    body = str(rec.get("selftext") if rec.get("selftext") is not None
               else (rec.get("body") or ""))
    # deleted/removed bodies carry no signal and would dilute sentiment
    if body in ("[deleted]", "[removed]"):
        body = ""
    if not (title.strip() or body.strip()):
        return None
    return {"id": str(rid), "date": date, "title": title,
            "selftext": body, "source": "reddit",
            "_sub": str(rec.get("subreddit") or "").lower()}


# ----------------------------------------------------------------- fold
def flush(buf, label, dry, verbose=True):
    """Aggregate one (file, month) block and merge the deltas in."""
    df = pd.DataFrame(buf)[NEEDED]
    df = df.drop_duplicates(subset="id", keep="first")
    if dry:
        print(f"    [dry] {label}: {len(df):,} posts (nothing written)")
        return len(df)
    aggs = abstracted_data.aggregate_posts(df)
    abstracted_data.merge_into_abstracted(aggs, verbose=False)
    if verbose:
        print(f"    folded {label}: {len(df):,} posts")
    return len(df)


def fold_file(path, led, args, subs, lo, hi):
    """Stream one archive, month by month. Months already in the ledger
    are skipped without being parsed into memory twice."""
    base = os.path.basename(path)
    buf = defaultdict(list)
    kept = skipped_sub = skipped_win = 0
    seen_ids = set()
    total = 0
    for rec in read_json_lines(path):
        row = to_row(rec)
        if row is None:
            continue
        total += 1
        if subs and row["_sub"] and row["_sub"] not in subs:
            skipped_sub += 1
            continue
        if row["date"] < lo or row["date"] > hi:
            skipped_win += 1
            continue
        key = f"{base}:{row['date'][:7]}"
        if key in led["blocks"] and not args.force:
            continue
        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        buf[row["date"][:7]].append(row)
        kept += 1
        # flush a month once it is big enough to keep memory bounded; the
        # ledger entry is written at the same moment, so an interrupted
        # run never leaves folded-but-unrecorded posts behind
        if len(buf[row["date"][:7]]) >= args.chunk:
            m = row["date"][:7]
            n = flush(buf[m], f"{base}:{m}", args.dry_run)
            if not args.dry_run:
                led["blocks"][f"{base}:{m}"] = {"posts": n, "partial": True}
                save_ledger(led)
            buf[m] = []
    for m in sorted(buf):
        if not buf[m]:
            continue
        n = flush(buf[m], f"{base}:{m}", args.dry_run)
        if not args.dry_run:
            k = f"{base}:{m}"
            prev = led["blocks"].get(k, {}).get("posts", 0) if args.force else 0
            led["blocks"][k] = {"posts": n + prev,
                                "folded_utc": pd.Timestamp.utcnow()
                                .strftime("%Y-%m-%dT%H:%M:%S")}
            save_ledger(led)
    print(f"  {base}: read {total:,} | kept {kept:,} | "
          f"outside window {skipped_win:,} | other subreddit {skipped_sub:,}")
    return kept


def coverage_snapshot(lo, hi):
    """Mention rows per quarter in the target window - printed before and
    after so the fold's effect is a number, not a hope."""
    p = os.path.join(abstracted_data.ABSTRACTED_DIR,
                     abstracted_data.TICKER_COUNTS)
    if not os.path.exists(p):
        p = os.path.join(PROJECT_ROOT, "data", "processed",
                         "daily_ticker_counts.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= lo) & (d["date"] <= hi)]
    if not len(d):
        return pd.Series(dtype=int)
    return d.groupby(d["date"].dt.to_period("Q")).size()


def main():
    p = argparse.ArgumentParser(
        description="Fold historical posts into ABSTRACTED_DATA "
                    "(text-free; the door update_data.py does not have).")
    p.add_argument("--arctic", action="store_true",
                   help="fold data/raw/RedditLive/*.jsonl.zst (what the "
                        "backfill runner already pulled)")
    p.add_argument("--dumps", nargs="*", default=[],
                   help="torrent-style archives, e.g. "
                        "'data/raw/dumps/*_submissions.zst' (globs ok)")
    p.add_argument("--start", default="1970-01-01",
                   help="first day to fold (default: everything)")
    p.add_argument("--end", default=None,
                   help="last day to fold (default: the day before "
                        "LIVE_START, so the live path is never overlapped)")
    p.add_argument("--chunk", type=int, default=120_000,
                   help="posts per aggregation flush (memory ceiling)")
    p.add_argument("--all-subreddits", action="store_true",
                   help="do not restrict to ingestion/finance_subreddits.txt")
    p.add_argument("--include-comments", action="store_true",
                   help="accept comment archives too (changes what a post "
                        "means; submissions built the committed history)")
    p.add_argument("--dry-run", action="store_true",
                   help="count what WOULD fold in; write nothing")
    p.add_argument("--force", action="store_true",
                   help="ignore the ledger and re-fold (CAN DOUBLE COUNT)")
    p.add_argument("--no-hydrate", action="store_true",
                   help="skip the ABSTRACTED_DATA -> data/processed copy")
    args = p.parse_args()

    files = []
    if args.arctic:
        files += sorted(glob.glob(ARCTIC_GLOB))
    for pat in args.dumps:
        files += sorted(glob.glob(pat))
    # .tmp files are interrupted fetches - the runner renames only on
    # success, so their bytes are not a complete archive
    files = [f for f in files if not f.endswith(".tmp")]
    if not args.include_comments:
        files = [f for f in files if "_comments" not in os.path.basename(f)]
    if not files:
        print("no input archives found. Use --arctic and/or --dumps.")
        return 1

    ls = live_start()
    hi = args.end or (
        (pd.Timestamp(ls) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if ls else "2099-12-31")
    lo = args.start
    if ls and hi >= ls:
        print(f"REFUSED: --end {hi} reaches LIVE_START ({ls}). The live "
              "path already owns those days; folding them here would "
              "double count. Pick an earlier --end.")
        return 1

    subs = set() if args.all_subreddits else allowed_subreddits()
    print(f"--- historical fold: {len(files)} archive(s) ---")
    print(f"window   : {lo} -> {hi}"
          + (f"   (LIVE_START {ls}, untouched)" if ls else ""))
    print(f"subreddits: {'ALL' if not subs else f'{len(subs)} from finance_subreddits.txt'}")
    led = load_ledger()
    print(f"ledger   : {len(led['blocks'])} block(s) already folded")

    before = coverage_snapshot(lo, hi)
    if before is not None and len(before):
        print(f"before   : {int(before.sum()):,} mention rows in the window")
    elif before is not None:
        print("before   : 0 mention rows in the window")

    total = 0
    for f in files:
        total += fold_file(f, led, args, subs, lo, hi)

    print(f"\n{'[dry-run] would fold' if args.dry_run else 'folded'}: "
          f"{total:,} posts")
    if args.dry_run:
        print("nothing written.")
        return 0
    if total and not args.no_hydrate:
        print("\n--- hydrate ABSTRACTED_DATA -> data/processed ---")
        abstracted_data.hydrate()
        after = coverage_snapshot(lo, hi)
        if after is not None and len(after):
            print(f"\nafter    : {int(after.sum()):,} mention rows in the window")
            print(after.to_string())
    if total:
        print("\nNEXT: a fold rewrites the history the thresholds were "
              "chosen on, so re-open research once:\n"
              "  python -m analytics.run_analytics --what phases --research")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
