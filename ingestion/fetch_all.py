# fetch_all.py
# ============
# The single entry point for all live API calls - now with the three source
# fetchers running IN PARALLEL (they are independent network jobs, so there
# is no reason to wait for Reddit before starting X and StockTwits; a full
# fetch that used to take fetchers' summed time now takes the slowest one's).
#
#   TESTING MODE   python ingestion/fetch_all.py --test
#       Makes exactly ONE FetchLayer call, prints what came back, writes
#       nothing. Verifies the key works. Pick the endpoint with --source:
#           --test                 (Reddit: r/wallstreetbets newest 5)
#           --test --source x      (X: newest tweets for a few cashtags)
#
#   NORMAL MODE    python ingestion/fetch_all.py
#       1. Checks .env - a source with its key filled is CALLED; a source
#          with an empty key is SKIPPED (no request sent).
#       2. Runs every enabled fetcher CONCURRENTLY (each writes raw files;
#          they touch disjoint folders, so parallelism is safe).
#       3. Appends the new posts - destination picked automatically:
#            * external machine (posts.parquet exists) -> merge_live.py
#              appends the raw posts into posts.parquet (first seen wins)
#            * internal machine (no posts.parquet, or --abstracted) ->
#              append_live_abstracted.py folds them into ABSTRACTED_DATA
#              as text-free aggregates
#          Skip the append entirely with --no-merge (update_data.py does -
#          it owns the append step itself).
#
#   Other flags:
#       --check       print the .env key check only, call nothing
#       --no-merge    NORMAL mode but stop after writing raw (no append)
#       --abstracted  force the ABSTRACTED_DATA append (internal-machine path)
#       --serial      run the fetchers one after another (debugging - the
#                     interleaved parallel output can be hard to read)
#
# Sources and their keys (.env at the project root):
#   StockTwits : no key needed          - always called
#   Reddit     : Arctic Shift public API - no key needed (FetchLayer is the
#                manual fallback in fetch_reddit_live.py)
#   X          : FETCHLAYER_KEY         (or official X_BEARER_TOKEN)
#
# WHY SUBPROCESSES (not threads calling functions directly): each fetcher is
# also a standalone script with its own CLI, retries and rate-limit logic.
# Running them as subprocesses keeps that independence (one crashing can
# never take the others down), gives each a hard timeout, and lets the
# pipeline show each fetcher's exit status separately.

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import FETCH_TIMEOUT_S  # noqa: E402


def read_env():
    """Read .env directly (no dependency). Values are never printed."""
    keys = {}
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys


def fetch_plan():
    """THE CHECK. Returns [(source, will_call, reason, script_args)].

    Only sources whose credentials exist are called - a fetcher is never
    started just to fail on a missing key."""
    keys = read_env()

    def have(*names):                       # is ANY of these keys filled?
        return any(keys.get(n, "").strip() for n in names)

    fetchlayer = have("FETCHLAYER_KEY", "FETCHLAYER_API_KEY")
    x_ok = fetchlayer or have("X_BEARER_TOKEN")
    x_why = ("FetchLayer key found" if fetchlayer
             else "official X_BEARER_TOKEN found" if x_ok
             else "no FETCHLAYER_KEY / X_BEARER_TOKEN")

    return [
        ("StockTwits", True, "public API - no key needed", ["fetch_stocktwits.py"]),
        # Reddit comes from Arctic Shift: complete per-subreddit coverage,
        # near-real-time, free (no key). FetchLayer stays for X;
        # fetch_reddit_live.py remains available as a manual fallback.
        ("Reddit", True, "Arctic Shift public API - no key needed",
         ["fetch_reddit_arctic.py"]),
        ("X", x_ok, x_why, ["fetch_x_live.py"]),
    ]


def run_script(script_args, extra=None, timeout=None):
    """Run one fetcher to completion, capturing its output so parallel
    fetchers don't interleave lines. Returns (exit_code, captured_output)."""
    cmd = [sys.executable, os.path.join(THIS_DIR, script_args[0]),
           *script_args[1:], *(extra or [])]
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return 124, partial + f"\n[timeout] killed after {timeout}s"


def run_test(source):
    """TESTING MODE: one FetchLayer call, prints output, writes nothing."""
    print("=" * 60)
    print(f"TESTING MODE - one FetchLayer call ({source}), nothing is written")
    print("=" * 60)
    script = ["fetch_x_live.py", "--test"] if source == "x" else ["test_fetchlayer.py"]
    code, out = run_script(script, timeout=120)
    print(out)
    return code


def main():
    p = argparse.ArgumentParser(
        description="Live API calls: --test (1 call) or normal (parallel fetch + append)")
    p.add_argument("--test", action="store_true",
                   help="TESTING MODE: one FetchLayer call, print it, write nothing")
    p.add_argument("--source", choices=["reddit", "x"], default="reddit",
                   help="which endpoint --test hits (default: reddit)")
    p.add_argument("--check", action="store_true",
                   help="print the .env key check only - make no API calls")
    p.add_argument("--no-merge", action="store_true",
                   help="NORMAL mode but skip the parquet append (raw files only)")
    p.add_argument("--abstracted", action="store_true",
                   help="fold new posts into ABSTRACTED_DATA (text-free "
                        "aggregates) instead of posts.parquet - the "
                        "internal-machine path")
    p.add_argument("--serial", action="store_true",
                   help="run fetchers one at a time (easier-to-read output)")
    p.add_argument("--lookback-days", type=int, default=7,
                   help="how far back the fetch reaches (top posts of the "
                        "last N days); overlap never duplicates")
    p.add_argument("--max-credits", type=int, default=90,
                   help="FetchLayer credit cap per source per run")
    args = p.parse_args()

    # ---- TESTING MODE ---------------------------------------------------
    if args.test:
        return run_test(args.source)

    # ---- NORMAL MODE ----------------------------------------------------
    plan = fetch_plan()
    print("NORMAL MODE - API key check (.env at project root):")
    for name, will_call, why, _ in plan:
        print(f"  {'CALL' if will_call else 'SKIP'}  {name:<10} ({why})")
    if args.check:
        print("\n--check: no calls made. Run without --check to fetch.")
        return 0

    # the two knobs travel to every fetcher that understands them
    knobs = {"fetch_reddit_arctic.py": ["--lookback-days", str(args.lookback_days)],
             "fetch_reddit_live.py": ["--lookback-days", str(args.lookback_days),
                                      "--max-credits", str(args.max_credits)],
             "fetch_x_live.py": ["--lookback-days", str(args.lookback_days),
                                 "--max-credits", str(args.max_credits)]}

    to_run = [(name, script_args) for name, will_call, _, script_args in plan
              if will_call]
    failed = []
    t0 = time.time()
    print(f"\nfetching {len(to_run)} sources "
          f"{'SERIALLY' if args.serial else 'IN PARALLEL'} "
          f"(per-fetcher timeout {FETCH_TIMEOUT_S}s)...")

    if args.serial:
        for name, script_args in to_run:
            print(f"\n--- {name} ---")
            code, out = run_script(script_args, extra=knobs.get(script_args[0]),
                                   timeout=FETCH_TIMEOUT_S)
            print(out)
            if code != 0:
                failed.append(name)
    else:
        # One worker per source: the fetchers are network-bound and write to
        # disjoint folders (RedditLive/, StockTwits/, X Data/), so running
        # them simultaneously is safe and cuts wall-clock time to the
        # slowest fetcher instead of the sum of all three.
        #
        # Each fetcher's output prints as ONE block when it completes (so
        # parallel logs never interleave). The cost of that choice is a
        # silent terminal while the slow fetchers grind - hence the
        # HEARTBEAT below: every 30s, one line naming what is still
        # running and for how long, so a long fetch never looks frozen.
        # (The slow parts are deliberate: every fetcher sleeps between
        # requests to respect API rate limits - StockTwits ~1.5s/symbol,
        # X 5s/request, Arctic Shift 1s/page - so a full pull is minutes
        # of polite pacing, not seconds of work.)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_run)) as pool:
            futures = {pool.submit(run_script, script_args,
                                   knobs.get(script_args[0]),
                                   FETCH_TIMEOUT_S): name
                       for name, script_args in to_run}
            pending = set(futures)
            last_beat = time.time()
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    name = futures[fut]
                    code, out = fut.result()
                    print(f"\n--- {name} "
                          f"({'ok' if code == 0 else f'FAILED rc={code}'}, "
                          f"{time.time() - t0:.0f}s elapsed) ---")
                    print(out.rstrip(), flush=True)
                    if code != 0:
                        failed.append(name)
                # heartbeat: nothing finished in the last ~30s but work
                # continues - say so, with elapsed time per fetcher
                if pending and time.time() - last_beat >= 30:
                    still = ", ".join(sorted(futures[f] for f in pending))
                    print(f"[{time.time() - t0:.0f}s] still fetching: {still} "
                          "(rate-limit pacing - this is normal)", flush=True)
                    last_beat = time.time()

    print(f"\nfetch done in {time.time() - t0:.0f}s."
          + (f" FAILED: {', '.join(failed)}" if failed else " all sources ok."))

    # ---- APPEND the fresh raw ------------------------------------------
    # Two possible destinations, picked automatically:
    #   * posts.parquet   (external machine - the raw store exists)
    #       -> ingestion/merge_live.py
    #   * ABSTRACTED_DATA (internal machine - no raw store allowed;
    #       --abstracted forces this, and it is also chosen automatically
    #       when posts.parquet is absent)
    #       -> ingestion/append_live_abstracted.py
    posts_path = os.path.join(PROJECT_ROOT, "data", "processed", "posts.parquet")
    use_abstracted = args.abstracted or not os.path.exists(posts_path)

    if args.no_merge:
        target = "ABSTRACTED_DATA" if use_abstracted else "posts.parquet"
        later = ("ingestion/append_live_abstracted.py" if use_abstracted
                 else "ingestion/merge_live.py")
        print(f"--no-merge: raw written, {target} NOT touched. "
              f"To append later:  python {later}")
    elif use_abstracted:
        why = "--abstracted" if args.abstracted else "no posts.parquet found (internal machine)"
        print(f"\n--- APPEND: folding new posts into ABSTRACTED_DATA ({why}) ---")
        rc = subprocess.run(
            [sys.executable, os.path.join(THIS_DIR, "append_live_abstracted.py")],
            cwd=PROJECT_ROOT).returncode
        if rc != 0:
            failed.append("abstracted-append")
    else:
        print("\n--- MERGE: appending new posts into posts.parquet ---")
        merge_rc = subprocess.run(
            [sys.executable, os.path.join(THIS_DIR, "merge_live.py")],
            cwd=PROJECT_ROOT).returncode
        if merge_rc != 0:
            failed.append("merge")

    print("\nsee what landed:  python check_live_ingestion.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
