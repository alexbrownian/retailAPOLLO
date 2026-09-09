"""Fold historical posts into ``ABSTRACTED_DATA`` in aggregates mode.

A copy without ``posts.parquet`` has only one fold-in path,
``ingestion/append_live_abstracted.py``, and that script keeps only
candidates dated on or after ``LIVE_START`` (frozen at the newest
committed day + 1). Any backfilled post older than ``LIVE_START`` is
dropped silently, by design. ``update_data.py --full`` cannot fold it in
either: it rebuilds from ``posts.parquet``, which only exists in full
mode, and its guard aborts rather than revert committed months.

That filter is correct for live ingestion: it is what stops the daily run
re-folding the committed historical block and double-counting it. This
script is the deliberate, separate door for the opposite job, adding
history that was never there::

    python tools/fold_historical.py --arctic            # files already on disk
    python tools/fold_historical.py --dumps "data/raw/dumps/*_submissions.zst"
    python tools/fold_historical.py --arctic --dry-run  # count, write nothing

What it does not change:

* ``LIVE_START`` is read, never written. The live path keeps its guard.
* Text never lands on disk. Posts are streamed, aggregated in memory and
  only the daily count/sentiment rows are written; the same text-free
  boundary ``append_live_abstracted.py`` works under, and the reason
  this is safe to run on the committable store.
* Counting rules are not reimplemented: aggregation goes through
  ``abstracted_data.aggregate_posts``, the same function the live fold
  and the full rebuild use, so a folded month is indistinguishable from
  a month that was there all along.

Idempotency: ``merge_into_abstracted`` adds counts. Folding the same
posts twice would inflate them, and no downstream number would look
obviously wrong. So every ``(file, month)`` block that is folded is
recorded in ``data/reference/historical_fold_ledger.json``, and a
complete block already in the ledger is skipped. A crash mid-file
therefore costs only the months that had not yet flushed, never a double
count. ``--force`` overrides the ledger and can double count.

Input files:

* Already on disk: ``data/raw/RedditLive/*.jsonl.zst``, what the backfill
  runner pulled. Use ``--arctic``.
* Archive dumps: per-subreddit monthly archives named
  ``<subreddit>_submissions.zst`` (the naming documented in
  ``config/forums.csv``). They are ordinary NDJSON inside long-window
  zstd; ``src.clean_data.read_json_lines`` streams that format, so a dump
  can be dropped into ``data/raw/dumps/`` and pointed at with
  ``--dumps``. Submissions are what the historical block was built from;
  comment archives are accepted with ``--include-comments`` but change
  what a "post" means, so they are off by default.
"""

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
    """Return the fold ledger, or an empty one when absent or unreadable."""
    if os.path.exists(LEDGER_PATH):
        try:
            return json.load(open(LEDGER_PATH, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"blocks": {}}


def save_ledger(led):
    """Write the fold ledger atomically."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(led, f, indent=1)
    os.replace(tmp, LEDGER_PATH)


def live_start():
    """Return ``LIVE_START`` from the live ledger, or ``None`` if unset.

    Anything on or after it belongs to the live path and must not be
    folded here; that is the double count this script exists to avoid.
    """
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
    """Return the panel subreddits (lower-cased) from ``config/forums.csv``.

    The committed history was built from this panel; folding anything
    else in would change what the counts mean, not just how many there
    are. Returns an empty set when the settings module cannot be loaded.
    """
    try:
        from src.settings import load_forums
        return {s.lower() for s in load_forums()}
    except Exception:                                    # noqa: BLE001
        return set()


# -------------------------------------------------------------- records
def to_row(rec):
    """Reduce one raw record to the columns the aggregator needs.

    Handles both shapes with the same code: a submission (title plus
    selftext) and a comment (body). Timestamps arrive as unix seconds in
    the dumps and as either seconds or ISO in the project's own pulls, so
    both are accepted. Deleted/removed bodies are blanked because they
    carry no signal and would dilute sentiment.

    Args:
        rec: One decoded JSON record.

    Returns:
        Dict with the ``NEEDED`` columns plus ``_sub`` (lower-cased
        subreddit), or ``None`` when the record has no id, no parseable
        date or no text.
    """
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
    """Aggregate one ``(file, month)`` block and merge the deltas in.

    Args:
        buf: List of row dicts from ``to_row``.
        label: ``"<file>:<YYYY-MM>"`` for progress output.
        dry: Count only; write nothing.
        verbose: Print one line per block folded.

    Returns:
        Number of unique posts in the block.
    """
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
    """Stream one archive and fold it month by month.

    Complete months already in the ledger are skipped without being held
    in memory. Months are flushed once they reach ``args.chunk`` rows,
    with the ledger written at the same moment, so an interrupted run
    never leaves folded-but-unrecorded posts behind.

    Args:
        path: Archive to read.
        led: The fold ledger (mutated and saved as blocks complete).
        args: Parsed CLI arguments (``chunk``, ``dry_run``, ``force``).
        subs: Allowed subreddits; empty means all.
        lo: First day to fold (ISO).
        hi: Last day to fold (ISO, inclusive).

    Returns:
        Number of posts kept from this file.
    """
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
        # A mid-file chunk flush records the block with partial=True.
        # Skipping on presence alone would drop every remaining record
        # of that month in the same file after its first flush, while
        # the ledger asserted the block was done. Only a COMPLETE block
        # skips.
        _blk = led["blocks"].get(key)
        if _blk is not None and not _blk.get("partial") and not args.force:
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
            # Carry forward whatever the chunk flushes already counted
            # for this block, and drop the partial marker: the file is
            # now fully read, so the block is complete.
            _p = led["blocks"].get(k, {})
            prev = _p.get("posts", 0) if (args.force or _p.get("partial")) else 0
            led["blocks"][k] = {"posts": n + prev,
                                "folded_utc": pd.Timestamp.utcnow()
                                .strftime("%Y-%m-%dT%H:%M:%S")}
            save_ledger(led)
    print(f"  {base}: read {total:,} | kept {kept:,} | "
          f"outside window {skipped_win:,} | other subreddit {skipped_sub:,}")
    return kept


def coverage_snapshot(lo, hi):
    """Count ticker-mention rows per quarter in the target window.

    Printed before and after the fold so its effect is a number.

    Args:
        lo: Window start (ISO).
        hi: Window end (ISO, inclusive).

    Returns:
        Series indexed by quarter period, empty when the window has no
        rows, or ``None`` when no counts file exists yet.
    """
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
    """Fold the selected archives below ``LIVE_START`` and hydrate.

    Returns:
        ``0`` on success; ``1`` when no input archives were found or the
        requested window reaches ``LIVE_START``.
    """
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
                   help="do not restrict to the panel in config/forums.csv")
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
    print(f"subreddits: {'ALL' if not subs else f'{len(subs)} from config/forums.csv'}")
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
