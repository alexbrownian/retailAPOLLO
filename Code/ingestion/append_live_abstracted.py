"""Fold new live posts into ``Data/abstracted`` without keeping raw text.

This is the live-ingestion path for aggregates mode, where the copy holds
no ``posts.parquet``::

    python Code/ingestion/append_live_abstracted.py             fold new posts in
    python Code/ingestion/append_live_abstracted.py --dry-run   show what would fold in

``merge_live.py`` appends raw live posts into ``posts.parquet`` (the
full-mode raw store). This script skips the raw store entirely: it
aggregates the new posts into daily counts and daily sentiment and merges
those rows into ``Data/abstracted``. Same fetchers, same normalisers, same
aggregation code; a different, text-free destination.

The flow:

1. read the live raw files (RedditLive / StockTwits / X live) into
   candidate rows using the project normalisers, exactly as
   ``merge_live.py`` does;
2. keep only candidates that are (a) dated on or after ``LIVE_START``,
   which separates them from the committed historical block so the two
   never overlap, and (b) not already in the local seen-ids ledger
   ("first seen wins" across re-runs, so running twice folds nothing the
   second time);
3. aggregate the survivors with ``src.abstracted_data.aggregate_posts``
   and merge the deltas into ``Data/abstracted`` (counts add; sentiment
   recombines weighted by ``n_posts``);
4. record the new ids in the ledger, then hydrate ``Data/abstracted`` into
   ``Data/processed`` so the analytics stage sees the update.

The ledger and ``LIVE_START`` live in
``Data/reference/abstracted_live_meta.json`` (plus the seen-id parquet
beside it). They are kept local and gitignored: post ids are mildly
identifying, so they are the one thing never committed. On a fresh copy
the ledger starts empty and ``LIVE_START`` freezes to (newest committed
date) + 1 day, so live ingestion only ever adds genuinely new days on top
of the committed history.
"""

import argparse
import datetime
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
from src.config import REFERENCE_DIR                          # noqa: E402
from src import abstracted_data                               # noqa: E402
from src.clean_data import read_json_lines                    # noqa: E402
from src.reddit_live_data import normalise_reddit_live_records  # noqa: E402
from src.stocktwits_data import normalise_stocktwits          # noqa: E402
from src.x_data import normalise_x_api                        # noqa: E402
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402

RAW_ROOT = os.path.join(DATA_DIR, "raw")
META_PATH = os.path.join(DATA_DIR, "reference",
                         "abstracted_live_meta.json")
LEGACY_META = os.path.join(DATA_DIR, "reference",
                           "gic_live_meta.json")   # older ledger file name,
                                                   # read if present
MAX_SEEN = 300_000            # legacy cap on the JSON seen_ids array; the
                              # parquet set (SEEN_PATH) is uncapped. Kept
                              # only so an old ledger can still be read.

# ---------------------------------------------------------------------
# Dedup set. src/abstracted_data.merge_counts is additive: the same
# (date, ticker) row has its mention_count summed into the store, and
# the aggregates carry no post ids, so a post counted twice can never
# afterwards be detected or undone. The seen-id set is therefore the
# sole guard against permanent double counting and is kept durable and
# uncapped in its own parquet (raw files accumulate indefinitely by
# design, so a lost or truncated id set would re-fold all of them). The
# JSON ledger retains live_start and the file-scan ledger; any legacy
# seen_ids array it holds is migrated on first read and kept in place as
# a fallback.
SEEN_PATH = os.path.join(DATA_DIR, "reference",
                         "abstracted_seen_ids.parquet")
# Rolling copies of the whole reference dir, taken only after a run that
# actually changed something. Cheap insurance against a lost or
# truncated id set.
BACKUP_DIR = os.path.join(DATA_DIR, "reference", "_backups")
BACKUP_KEEP = 7

# ---------------------------------------------------------------------
# Per-file skip ledger. The raw folder accumulates files indefinitely
# (backfills included), and reading every file on every run scales with
# history rather than with new data. The skip is exact, not heuristic: a
# file whose newest post predates LIVE_START cannot contribute a single
# row, because the LIVE_START filter removes all of it by construction.
# Each file's max date is recorded once and the file is skipped until it
# changes on disk (size or mtime) or LIVE_START itself moves. The skip
# is keyed on dates, never on previously-seen ids: a file that predates
# the window is provably empty of new rows, whereas a partially-seen
# file may still hold rows that should be folded.
FILE_LEDGER_KEY = "files_scanned"

# The columns aggregate_posts needs from a post.
NEEDED = ["id", "date", "title", "selftext", "source"]


# ---------------- collect candidate posts from the raw live files ----------
def _stat_key(path):
    """Return ``"<size>:<mtime>"`` for a file, the ledger's change key."""
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


def _can_skip(path, ledger, live_start):
    """Return True when this file provably cannot contribute a single row.

    Args:
        path: Raw file path.
        ledger: The per-file scan ledger (mutated by ``_note_file``).
        live_start: The frozen ``LIVE_START`` date string.
    """
    rec = ledger.get(os.path.basename(path))
    if not rec or rec.get("stat") != _stat_key(path):
        return False                       # new or changed on disk
    if rec.get("live_start") != live_start:
        return False                       # the window moved; re-check it
    return rec.get("max_date", "9999") < live_start


def _note_file(path, ledger, live_start, max_date):
    """Record a file's change key, window and newest post date in the ledger."""
    ledger[os.path.basename(path)] = {"stat": _stat_key(path),
                                      "live_start": live_start,
                                      "max_date": max_date}


def collect_reddit_live(ledger=None, live_start=None):
    """Normalise the ``RedditLive/*.jsonl.zst`` files that may hold new rows.

    Args:
        ledger: The per-file scan ledger; updated in place with each file
            read. ``None`` disables the ledger.
        live_start: The frozen ``LIVE_START`` date string. When given,
            files whose newest post predates it are skipped without being
            read.

    Returns:
        DataFrame of normalised posts; empty when nothing was read.
    """
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
    """Normalise every ``StockTwits/*.jsonl.zst`` raw file.

    Returns:
        DataFrame of normalised messages; empty when there are none.
    """
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
    """Normalise ``X Data/x_api_live.csv.zst``.

    Returns:
        DataFrame of normalised tweets; empty when the file is absent.
    """
    path = os.path.join(RAW_ROOT, "X Data", "x_api_live.csv.zst")
    if not os.path.exists(path):
        return pd.DataFrame(columns=NEEDED)
    blob = zstandard.ZstdDecompressor().decompress(open(path, "rb").read())
    raw = pd.read_csv(io.BytesIO(blob), dtype={"id": str})
    df = normalise_x_api(raw)
    print(f"[x live ] {len(df):,} tweets from x_api_live.csv.zst")
    return df


def collect_candidates(ledger=None, live_start=None):
    """Gather candidate posts from all three live sources.

    Duplicate ids are dropped (first kept) and the frame is reduced to the
    ``NEEDED`` columns with ``date`` as a plain ``YYYY-MM-DD`` string.

    Args:
        ledger: Per-file scan ledger passed to ``collect_reddit_live``.
        live_start: ``LIVE_START`` passed to ``collect_reddit_live``.

    Returns:
        DataFrame with exactly the ``NEEDED`` columns; empty when no
        source produced rows.
    """
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
    """Load the JSON ledger, migrating the pre-rename file if present.

    Returns:
        The ledger dict, or an empty dict when none exists or it is
        unreadable.
    """
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
    """Persist ``live_start`` and the file-scan ledger atomically.

    The seen-id set is not written here; it lives in ``SEEN_PATH``,
    uncapped. Any legacy ``seen_ids`` array already in the file is left
    exactly as it is: harmless, and a fallback if the parquet is ever
    lost.

    Args:
        meta: The ledger dict to write.
    """
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    tmp = META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp, META_PATH)


def load_seen(meta):
    """Load the durable dedup set.

    Reads the parquet at ``SEEN_PATH``; falls back to (and migrates from)
    the legacy JSON array in ``meta`` the first time. Ids present only in
    the legacy array are always added, so the set never shrinks.

    Args:
        meta: The JSON ledger dict (may hold a legacy ``seen_ids`` array).

    Returns:
        Set of post id strings.
    """
    if os.path.exists(SEEN_PATH):
        try:
            got = set(pd.read_parquet(SEEN_PATH)["id"].astype(str))
            legacy = {str(i) for i in meta.get("seen_ids", [])}
            missing = legacy - got
            if missing:                    # belt and braces: never shrink
                got |= missing
                print(f"[ledger] recovered {len(missing):,} id(s) present "
                      f"only in the legacy JSON ledger")
            return got
        except Exception as exc:           # noqa: BLE001
            print(f"[ledger] WARNING: {os.path.basename(SEEN_PATH)} "
                  f"unreadable ({exc}); falling back to the JSON ledger")
    legacy = {str(i) for i in meta.get("seen_ids", [])}
    if legacy:
        print(f"[ledger] migrating {len(legacy):,} id(s) from the legacy "
              f"JSON ledger into {os.path.basename(SEEN_PATH)} (uncapped)")
    return legacy


def save_seen(ids):
    """Write the dedup set atomically.

    Writes beside the target and then replaces it, so a crash mid-write
    leaves the previous set intact rather than a truncated one.

    Args:
        ids: Iterable of post id strings.
    """
    os.makedirs(os.path.dirname(SEEN_PATH), exist_ok=True)
    tmp = SEEN_PATH + ".tmp"
    pd.DataFrame({"id": sorted(ids)}).to_parquet(tmp, index=False)
    os.replace(tmp, SEEN_PATH)


def backup_reference():
    """Snapshot ``Data/reference`` after a run that changed something.

    The ledgers are small (a few MB) and losing one is unrecoverable, so a
    handful of dated copies is the cheapest possible insurance. Keeps
    ``BACKUP_KEEP`` snapshots and deletes the rest; never recurses into
    itself. A backup failure is reported but never stops a good run.
    """
    import shutil
    try:
        src_dir = os.path.dirname(META_PATH)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dest = os.path.join(BACKUP_DIR, stamp)
        # two runs inside the same second would otherwise overwrite each
        # other and make the rotation count wrong
        _n = 1
        while os.path.exists(dest):
            dest = os.path.join(BACKUP_DIR, f"{stamp}_{_n}")
            _n += 1
        os.makedirs(dest, exist_ok=True)
        for fn in os.listdir(src_dir):
            fp = os.path.join(src_dir, fn)
            if os.path.isfile(fp) and not fn.endswith(".tmp"):
                shutil.copy2(fp, os.path.join(dest, fn))
        kept = sorted(d for d in os.listdir(BACKUP_DIR)
                      if os.path.isdir(os.path.join(BACKUP_DIR, d)))
        for old_dir in kept[:-BACKUP_KEEP]:
            shutil.rmtree(os.path.join(BACKUP_DIR, old_dir),
                          ignore_errors=True)
        print(f"[backup] Data/reference -> _backups/{stamp} "
              f"(keeping {min(len(kept), BACKUP_KEEP)})")
    except Exception as exc:               # noqa: BLE001
        # a backup failure must never stop a good run
        print(f"[backup] skipped ({exc})")


def newest_committed_date():
    """Return the newest date already in ``Data/abstracted``, or ``None``."""
    path = os.path.join(abstracted_data.ABSTRACTED_DIR, abstracted_data.TICKER_COUNTS)
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path, columns=["date"])
    if len(d) == 0:
        return None
    return pd.to_datetime(d["date"]).max()


def resolve_live_start(meta):
    """Return ``LIVE_START``, freezing it into ``meta`` on first use.

    ``LIVE_START`` is the first day live ingestion owns. Posts before it
    belong to the committed historical block and must never be re-folded
    here. On first use it is set to the newest committed day plus one
    (or ``1970-01-01`` for an empty store) and never moves again.

    Args:
        meta: The JSON ledger dict; ``meta["live_start"]`` is set when
            absent.

    Returns:
        The ``YYYY-MM-DD`` date string.
    """
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
    """Gather, filter, aggregate and merge the new live posts.

    Returns:
        ``0`` on success or when there is nothing to fold.
    """
    p = argparse.ArgumentParser(
        description="Fold new live posts into Data/abstracted (text-free aggregates).")
    p.add_argument("--dry-run", action="store_true",
                   help="report what WOULD be folded in; write nothing")
    p.add_argument("--no-hydrate", action="store_true",
                   help="update Data/abstracted but skip the copy into Data/processed")
    args = p.parse_args()

    # ---- 1. gather + 2. filter
    #
    # The ledger and LIVE_START are resolved BEFORE reading anything, so
    # collect_* can skip files that cannot possibly contribute.
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

    seen = load_seen(meta)

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
        print("nothing new to fold - Data/abstracted already up to date.")
        # Persist the skip ledger anyway. Without this the very common
        # "nothing new" run would learn nothing, and the next run would
        # re-read every file all over again - which is exactly the cost
        # this ledger exists to remove.
        if not args.dry_run:
            save_meta(meta)
        return 0

    # ---- bot screen: automated / duplicated posts never reach an
    #      aggregate. The raw files are untouched; the ids are still
    #      recorded as seen so the same rows are not re-judged next run.
    from ingestion.bot_screen import apply_screen, format_report, write_report
    _screened, _rep = apply_screen(fresh)
    print("[filter] " + format_report(_rep))
    write_report(_rep, os.path.join(REPORTS_DIR, "bot_screen_last.json"))
    _excluded_ids = (set(fresh["id"].astype(str)) - set(_screened["id"].astype(str))
                     if _rep["rows_excluded"] else set())
    fresh = _screened
    if fresh.empty:
        print("every new post was excluded by the bot screen - nothing to fold.")
        if not args.dry_run:
            seen |= _excluded_ids
            save_seen(seen)
            save_meta(meta)
        return 0

    if args.dry_run:
        print("\n--dry-run: nothing written. The above is what WOULD be folded in.")
        return 0

    # ---- 3. aggregate the new posts and merge the deltas
    print("\n--- aggregating new posts (tickers + sentiment) ---")
    new_aggs = abstracted_data.aggregate_posts(fresh)
    print("--- merging into Data/abstracted ---")
    abstracted_data.merge_into_abstracted(new_aggs)

    # ---- 4. record the new ids, then hydrate Data/processed
    #
    # ORDER MATTERS. The aggregates were merged above; if the process
    # dies before the ids are written, the next run would fold those
    # posts AGAIN. So the dedup set is saved immediately after the
    # merge, before anything slower (hydrate) can fail.
    seen |= set(fresh["id"].astype(str)) | _excluded_ids
    save_seen(seen)
    save_meta(meta)
    print(f"[ledger] +{len(fresh):,} ids (dedup set now holds "
          f"{len(seen):,}, uncapped)")
    backup_reference()

    if not args.no_hydrate:
        print("\n--- hydrate Data/abstracted -> Data/processed ---")
        abstracted_data.hydrate()

    print("\ndone. next: recompute the analytics "
          "(update_data.py does this: cd Code && python -m src.analytics.run_analytics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
