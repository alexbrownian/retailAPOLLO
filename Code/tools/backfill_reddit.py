"""Resumable Reddit history backfill in chunks.

Fills a multi-month gap in the raw Reddit store without needing an
uninterrupted multi-hour connection::

    python Code/tools/backfill_reddit.py                      # full gap, monthly chunks
    python Code/tools/backfill_reddit.py --start 2024-01-01 --end 2024-07-01
    python Code/tools/backfill_reddit.py --chunk-days 14      # smaller bites
    python Code/tools/backfill_reddit.py --status             # what is left to do
    python Code/tools/backfill_reddit.py --redo 2024-03-01    # re-run one chunk

``ingestion/fetch_reddit_arctic.py --backfill START END`` does the actual
pulling, but it is one long process that writes to a ``.tmp`` file and
only renames it to the real ``.jsonl.zst`` at the very end. Kill it, or
lose the network for more than a few minutes, and the whole run is
discarded; a backfill deliberately does not move the watermark, so there
is nothing to resume from.

This wrapper cuts the window into chunks and runs the fetcher once per
chunk as its own process. Each chunk that finishes writes its own raw
file and is recorded in the ledger, so an interrupted run picks up at the
first unfinished chunk. Dedup is by post id all the way down the
pipeline, so re-running a chunk is always safe.

Ledger: ``Data/reference/reddit_backfill_progress.json``::

    {"chunks": {"2024-03-01_2024-04-01": {"posts": 41230, "done_utc": ...}}}

After it finishes, the right follow-up depends on the mode of this copy.

Aggregates mode (no ``Data/processed/posts.parquet``)::

    python Code/tools/fold_historical.py --arctic
    cd Code && python -m src.analytics.run_analytics --what phases --research

Full mode (``posts.parquet`` present)::

    python Code/update_data.py --skip-fetch
    cd Code && python -m src.analytics.run_analytics --what phases --research

Do not run ``update_data.py --skip-fetch`` in aggregates mode after a
backfill: ``ingestion/append_live_abstracted.py`` keeps only posts dated
on or after ``LIVE_START`` by design, so every backfilled post is read
and then dropped. ``tools/fold_historical.py`` is the correct path for
historical posts in that mode. ``print_next_steps`` picks the right pair
automatically, so the message printed at the end of a run is always the
one for this copy.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from src.config import DATA_DIR  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
FETCHER = os.path.join(PROJECT_ROOT, "ingestion", "fetch_reddit_arctic.py")
LEDGER = os.path.join(DATA_DIR, "reference",
                      "reddit_backfill_progress.json")

# The default window covers the span where ticker-mention rows collapse
# ~90% and only recover at its end.
DEFAULT_START = "2023-04-01"
DEFAULT_END = "2026-01-01"


def load_ledger() -> dict:
    """Return the progress ledger, or an empty one when absent or unreadable."""
    if os.path.exists(LEDGER):
        try:
            return json.load(open(LEDGER, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"chunks": {}}


def save_ledger(led: dict) -> None:
    """Write the progress ledger atomically."""
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER + ".tmp", "w", encoding="utf-8") as f:
        json.dump(led, f, indent=1)
    os.replace(LEDGER + ".tmp", LEDGER)


def make_chunks(start: str, end: str, chunk_days: int):
    """Split ``[start, end)`` into fetch windows.

    Args:
        start: ISO start date.
        end: ISO end date (exclusive).
        chunk_days: Window width in days; ``<= 0`` means calendar months.

    Returns:
        List of ``(start, end)`` ISO date pairs.

    Raises:
        SystemExit: When ``end`` is not after ``start``.
    """
    d0 = datetime.date.fromisoformat(start)
    d1 = datetime.date.fromisoformat(end)
    if d1 <= d0:
        raise SystemExit("end must be after start")
    out = []
    cur = d0
    while cur < d1:
        if chunk_days > 0:
            nxt = cur + datetime.timedelta(days=chunk_days)
        else:                                   # first of the next month
            nxt = (cur.replace(day=28)
                   + datetime.timedelta(days=4)).replace(day=1)
        nxt = min(nxt, d1)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return out


# Which subreddits this run covers. Part of the ledger key - see below.
SUBS_TAG = "all"


def key_of(a: str, b: str) -> str:
    """Build the ledger key from the window plus the subreddit set.

    A chunk is only complete for the subreddits it actually fetched, so
    the set belongs in the key: with a date-only key, re-running a
    finished window with additional subreddits would be skipped as
    already done and silently fetch nothing. Re-running the same set
    still skips; dedup is by post id, so any overlap is harmless.

    Args:
        a: Window start (ISO date).
        b: Window end (ISO date).

    Returns:
        ``"<a>_<b>"`` for the full panel, else ``"<a>_<b>#<subs tag>"``.
    """
    return f"{a}_{b}" if SUBS_TAG == "all" else f"{a}_{b}#{SUBS_TAG}"


def subs_tag(spec: str) -> str:
    """Return a stable tag for a comma-separated subreddit set.

    The names are lower-cased and sorted, so the order typed on the
    command line cannot create a second, spurious ledger entry. An empty
    spec yields ``"all"``.
    """
    if not spec.strip():
        return "all"
    subs = sorted({x.strip().lower() for x in spec.split(",") if x.strip()})
    return "+".join(subs) if subs else "all"


# Extra flags handed to every fetcher subprocess (set in main()).
FETCH_OPTS: list = []


def run_chunk(a: str, b: str) -> tuple[bool, int]:
    """Run the fetcher for one window as a subprocess.

    The child is started with ``-u`` and that flag is load-bearing: the
    fetcher's progress lines use a bare ``print()``, and Python
    block-buffers stdout (about 8 KB) whenever it is a pipe rather than
    a terminal. One chunk emits only a few hundred bytes, so without
    ``-u`` nothing would appear until the child exits many minutes later,
    which looks exactly like a hang. ``bufsize=1`` on the ``Popen`` is
    line buffering on the parent's read side and says nothing about how
    the child writes.

    Args:
        a: Window start (ISO date).
        b: Window end (ISO date).

    Returns:
        Tuple ``(ok, posts_written)``. ``ok`` is False when the child
        failed or any subreddit gave up mid-window, in which case the
        chunk must be re-run.
    """
    cmd = [sys.executable, "-u", FETCHER, "--backfill", a, b]
    if FETCH_OPTS:
        cmd += FETCH_OPTS
    posts = 0
    gave_up = 0
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    # HEARTBEAT. Even unbuffered, a single busy subreddit can page for
    # many minutes without printing anything, and silence on a 13-hour
    # job is indistinguishable from a hang. A daemon thread says how
    # long it has been quiet so the operator never has to guess.
    _last = [time.time()]
    _stop = threading.Event()

    def _beat():
        while not _stop.wait(120):
            _quiet = time.time() - _last[0]
            if _quiet >= 115:
                print(f"      ... still working - {_quiet / 60:.0f} min "
                      f"since the last line (a busy subreddit pages for "
                      f"a while; Ctrl-C is safe, finished chunks are "
                      f"kept)", flush=True)

    threading.Thread(target=_beat, daemon=True).start()
    for line in proc.stdout:
        _last[0] = time.time()
        line = line.rstrip()
        if "giving up this run" in line:
            gave_up += 1
        if "arctic reddit:" in line:
            try:
                posts = int(line.split(":")[1].split()[0].replace(",", ""))
            except (IndexError, ValueError):
                pass
        if line.strip():
            print("    " + line, flush=True)
    proc.wait()
    _stop.set()
    if proc.returncode != 0:
        return False, posts
    if gave_up:
        # Partial coverage: some subreddit windows were never fetched.
        # Do NOT mark done - the chunk must be re-run.
        print(f"    !! {gave_up} subreddit(s) gave up - chunk NOT marked "
              f"done, re-run to complete it", flush=True)
        return False, posts
    return True, posts


def print_next_steps(header: str) -> None:
    """Print the correct follow-up commands for this copy's mode.

    Decided by the same test ``update_data.py`` uses: whether
    ``posts.parquet`` exists.

    Args:
        header: Line printed before the commands.
    """
    posts = os.path.join(DATA_DIR, "processed", "posts.parquet")
    internal = not os.path.exists(posts)
    print(header)
    if internal:
        print("  python Code/tools/fold_historical.py --arctic")
        print("  cd Code && python -m src.analytics.run_analytics --what phases --research")
        print("\n  (aggregates mode - no posts.parquet. fold_historical is")
        print("   the door for historical posts. `update_data.py --skip-fetch`")
        print("   would drop every one of them: append_live_abstracted keeps")
        print("   only dates >= LIVE_START. See research.ipynb.)")
    else:
        print("  python Code/update_data.py --skip-fetch")
        print("  cd Code && python -m src.analytics.run_analytics --what phases --research")
        print("\n  (full mode - posts.parquet present, so the ordinary")
        print("   rebuild path sees the backfilled posts.)")


def main() -> int:
    """Run the pending chunks, or one of the status/redo/estimate modes.

    Returns:
        ``0`` on success; ``1`` when a chunk or the probe failed.
    """
    p = argparse.ArgumentParser(description="Resumable Reddit backfill.")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--chunk-days", type=int, default=0,
                   help="0 = one calendar month per chunk (default)")
    p.add_argument("--status", action="store_true",
                   help="print what is done / left and exit")
    p.add_argument("--redo", metavar="START",
                   help="forget the chunk beginning on this date so the "
                        "next run fetches it again")
    p.add_argument("--max-chunks", type=int, default=0,
                   help="stop after N chunks this run (0 = all)")
    p.add_argument("--workers", type=int, default=4,
                   help="subreddits fetched concurrently INSIDE each chunk "
                        "(default 4). Subreddits are independent; pages "
                        "within one stay sequential because each page's "
                        "cursor comes from the previous page.")
    p.add_argument("--subreddits", default="",
                   help="comma-separated subset instead of all 17. Coverage "
                        "measured on the healthy 2026 window (share of "
                        "covered name-days retained): wallstreetbets 24%%, "
                        "+valueinvesting+stocks 59%%, +dividends+bogleheads "
                        "76%%, +personalfinance+pennystocks+daytrading 90%%. "
                        "Fewer subreddits is the single biggest lever if "
                        "you want SOME history rather than all of it.")
    p.add_argument("--no-fast", action="store_true",
                   help="disable limit=auto + minimal fields (fast mode). "
                        "Only for comparing against the conservative "
                        "path - it is strictly slower and returns "
                        "identical posts.")
    p.add_argument("--estimate", action="store_true",
                   help="pull ONE probe day, measure it, and print a "
                        "runtime for the whole window from THIS machine's "
                        "throughput (no typed-in numbers)")
    args = p.parse_args()

    led = load_ledger()
    global FETCH_OPTS, SUBS_TAG
    SUBS_TAG = subs_tag(args.subreddits)
    FETCH_OPTS = []
    if args.no_fast:
        FETCH_OPTS.append("--no-fast")
    if args.workers and args.workers != 1:
        FETCH_OPTS += ["--workers", str(args.workers)]
    if args.subreddits:
        FETCH_OPTS += ["--subreddits", args.subreddits]

    chunks = make_chunks(args.start, args.end, args.chunk_days)

    if args.estimate:
        # A probe day in the middle of the window: representative volume,
        # real network, real pacing. The probe's posts land in the raw
        # dir like any other pull (dedup makes that harmless).
        d0 = datetime.date.fromisoformat(args.start)
        d1 = datetime.date.fromisoformat(args.end)
        mid = d0 + (d1 - d0) / 2
        a, b = mid.isoformat(), (mid + datetime.timedelta(days=1)).isoformat()
        print(f"probing one day ({a}) to measure this machine ...")
        t0 = time.time()
        ok, posts = run_chunk(a, b)
        secs = time.time() - t0
        days = (d1 - d0).days
        if not ok or secs <= 0:
            print("probe failed - check the connection and retry")
            return 1
        eta_h = secs * days / 3600.0
        print(f"\n  probe: {posts:,} posts in {secs:.0f}s "
              f"({posts / secs:.0f} posts/s)")
        print(f"  window: {days} days ({len(chunks)} chunks)")
        print(f"  ESTIMATE: ~{posts * days:,.0f} posts, "
              f"~{eta_h:.1f} hours of fetching")
        print(f"  (fixed per-chunk overhead adds roughly "
              f"{len(chunks) * 17 / 60:.0f} min across {len(chunks)} chunks)")
        return 0

    if args.redo:
        hits = [k for k in led["chunks"] if k.startswith(args.redo + "_")]
        # --redo forgets the window for EVERY subreddit set, which is
        # what "run this window again" should mean.
        for k in hits:
            led["chunks"].pop(k)
        save_ledger(led)
        print(f"forgot {len(hits)} chunk(s) starting {args.redo}")
        return 0

    # Legacy entries (written before the key carried the subreddit set)
    # are ambiguous: we cannot tell which subreddits they covered. Say
    # so rather than guessing either way.
    _legacy = [k for k in led["chunks"] if "#" not in k]
    if _legacy and SUBS_TAG != "all":
        print(f"  NOTE: {len(_legacy)} chunk(s) in the ledger predate "
              f"per-subreddit tracking.\n        They are treated as "
              f"NOT done for the set '{SUBS_TAG}', which may re-fetch "
              f"some\n        posts. That is harmless - dedup is by post "
              f"id - but it costs time.\n        `--status` lists them.")

    todo = [c for c in chunks if key_of(*c) not in led["chunks"]]
    done = len(chunks) - len(todo)
    got = sum(v.get("posts", 0) for v in led["chunks"].values())

    print(f"  subreddits: {SUBS_TAG}")
    print(f"BACKFILL {args.start} -> {args.end}: {len(chunks)} chunks, "
          f"{done} done ({got:,} posts so far), {len(todo)} to go")
    if args.status:
        for a, b in chunks:
            rec = led["chunks"].get(key_of(a, b))
            mark = f"done  {rec['posts']:>7,} posts" if rec else "TO DO"
            print(f"  {a} -> {b}   {mark}")
        return 0
    if not todo:
        print_next_steps("nothing left to fetch. Now run:")
        return 0

    t_run = time.time()
    for i, (a, b) in enumerate(todo, 1):
        if args.max_chunks and i > args.max_chunks:
            print(f"--max-chunks {args.max_chunks} reached - stopping "
                  f"early; re-run to continue")
            break
        print(f"\n[{i}/{len(todo)}] {a} -> {b} ...", flush=True)
        t0 = time.time()
        ok, posts = run_chunk(a, b)
        secs = time.time() - t0
        if ok:
            led["chunks"][key_of(a, b)] = {
                "posts": posts, "secs": round(secs, 1),
                "done_utc": datetime.datetime.utcnow().isoformat(
                    timespec="seconds")}
            save_ledger(led)          # durable after EVERY chunk
            rate = posts / secs if secs else 0
            remaining = len(todo) - i
            eta = remaining * secs
            print(f"  chunk done: {posts:,} posts in {secs/60:.1f} min "
                  f"({rate:.0f}/s) | ~{eta/3600:.1f} h left at this rate",
                  flush=True)
        else:
            print(f"  chunk FAILED after {secs/60:.1f} min - it stays on "
                  f"the to-do list. Fix the connection and re-run this "
                  f"same command; finished chunks are skipped.", flush=True)
            return 1

    left = [c for c in chunks if key_of(*c) not in led["chunks"]]
    total = sum(v.get("posts", 0) for v in led["chunks"].values())
    print(f"\n{len(chunks) - len(left)}/{len(chunks)} chunks complete, "
          f"{total:,} posts pulled, {(time.time() - t_run)/60:.1f} min "
          f"this run")
    if not left:
        print_next_steps("ALL CHUNKS DONE. Now fold and re-score:")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
