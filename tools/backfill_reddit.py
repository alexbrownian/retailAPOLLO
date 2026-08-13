"""
backfill_reddit.py — RESUMABLE Reddit history backfill
======================================================
Desk request 2026-08-11. Fills the 2023-04 -> 2026-01 data drought
without needing an uninterrupted 8-hour connection.

    python tools/backfill_reddit.py                      # full gap, monthly chunks
    python tools/backfill_reddit.py --start 2024-01-01 --end 2024-07-01
    python tools/backfill_reddit.py --chunk-days 14      # smaller bites
    python tools/backfill_reddit.py --status             # what is left to do
    python tools/backfill_reddit.py --redo 2024-03-01    # re-run one chunk

WHY THIS EXISTS
    ingestion/fetch_reddit_arctic.py --backfill START END does the actual
    pulling, but it is ONE long process that writes to a .tmp file and
    only renames it to the real .jsonl.zst at the very end. Kill it,
    close the laptop, or lose the network for more than ~3 minutes and
    the whole run is discarded - a backfill deliberately does not move
    the watermark, so there is nothing to resume from.

    This wrapper cuts the window into chunks and runs the fetcher once
    per chunk as its own process. Each chunk that finishes writes its
    own raw file and is recorded in the ledger below, so an interrupted
    run picks up at the first unfinished chunk. Dedup is by post id all
    the way down the pipeline, so re-running a chunk is always safe.

LEDGER: data/reference/reddit_backfill_progress.json
    {"chunks": {"2024-03-01_2024-04-01": {"posts": 41230, "done_utc": ...}}}

AFTER IT FINISHES, fold and re-score as usual:
    python update_data.py --skip-fetch
    python -m analytics.run_analytics --what phases --research
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
FETCHER = os.path.join(PROJECT_ROOT, "ingestion", "fetch_reddit_arctic.py")
LEDGER = os.path.join(PROJECT_ROOT, "data", "reference",
                      "reddit_backfill_progress.json")

# The drought: ticker-mention rows collapse ~90% from 2023Q2 and only
# recover in 2026Q1 (see docs/research/coverage_gap.md).
DEFAULT_START = "2023-04-01"
DEFAULT_END = "2026-01-01"


def load_ledger() -> dict:
    if os.path.exists(LEDGER):
        try:
            return json.load(open(LEDGER, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"chunks": {}}


def save_ledger(led: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER + ".tmp", "w", encoding="utf-8") as f:
        json.dump(led, f, indent=1)
    os.replace(LEDGER + ".tmp", LEDGER)


def make_chunks(start: str, end: str, chunk_days: int):
    """Calendar months when chunk_days<=0, else fixed-width windows."""
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


def key_of(a: str, b: str) -> str:
    return f"{a}_{b}"


def run_chunk(a: str, b: str) -> tuple[bool, int]:
    """Run the fetcher for one window. Returns (ok, posts_written).

    `-u` IS LOAD-BEARING, not a style choice. The fetcher's progress
    lines use a bare print(), and Python BLOCK-buffers stdout whenever
    it is a pipe rather than a terminal - about 8 KB. One chunk emits
    roughly 700 bytes (seventeen short subreddit lines), so the buffer
    never fills and NOTHING appeared until the child exited ~26 minutes
    later. Run straight from a terminal the same fetcher prints live,
    which is why this only showed up under the wrapper and looked
    exactly like a hang at "[1/30]".

    `bufsize=1` below does NOT fix it: that is line buffering on the
    PARENT's read side and says nothing about how the child writes.
    """
    cmd = [sys.executable, "-u", FETCHER, "--backfill", a, b]
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


def main() -> int:
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
    p.add_argument("--estimate", action="store_true",
                   help="pull ONE probe day, measure it, and print a "
                        "runtime for the whole window from THIS machine's "
                        "throughput (no typed-in numbers)")
    args = p.parse_args()

    led = load_ledger()
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
        for k in hits:
            led["chunks"].pop(k)
        save_ledger(led)
        print(f"forgot {len(hits)} chunk(s) starting {args.redo}")
        return 0

    todo = [c for c in chunks if key_of(*c) not in led["chunks"]]
    done = len(chunks) - len(todo)
    got = sum(v.get("posts", 0) for v in led["chunks"].values())

    print(f"BACKFILL {args.start} -> {args.end}: {len(chunks)} chunks, "
          f"{done} done ({got:,} posts so far), {len(todo)} to go")
    if args.status:
        for a, b in chunks:
            rec = led["chunks"].get(key_of(a, b))
            mark = f"done  {rec['posts']:>7,} posts" if rec else "TO DO"
            print(f"  {a} -> {b}   {mark}")
        return 0
    if not todo:
        print("nothing left to fetch. Now run:\n"
              "  python update_data.py --skip-fetch\n"
              "  python -m analytics.run_analytics --what phases --research")
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
        print("ALL CHUNKS DONE. Now fold and re-score:\n"
              "  python update_data.py --skip-fetch\n"
              "  python -m analytics.run_analytics --what phases --research")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
