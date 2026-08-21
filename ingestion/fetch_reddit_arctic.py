# fetch_reddit_arctic.py
# ======================
# LIVE Reddit ingestion via the Arctic Shift public API - the DEFAULT
# Reddit source (FetchLayer stays for X; fetch_reddit_live.py remains as a
# fallback). Why Arctic:
#   * COMPLETE coverage: every post in every tracked subreddit, not a
#     top-engagement sample
#   * near-real-time: posts are archived within minutes of being written
#   * free - no key, no credits (be polite: paced requests)
#   * records are official/Pushshift shape, so the existing normaliser
#     (src/reddit_live_data.py, "_backend": "official") handles them as-is
#
#   python ingestion/fetch_reddit_arctic.py                  # fetch_all calls this
#   python ingestion/fetch_reddit_arctic.py --lookback-days 14
#   python ingestion/fetch_reddit_arctic.py --test           # one page, writes nothing
#   python ingestion/fetch_reddit_arctic.py --backfill 2023-04-01 2023-07-01
#                                                          # fill a HISTORICAL gap
#
# OUTPUT: data/raw/RedditLive/reddit_live_arctic_<timestamp>.jsonl.zst
#   one line per post, raw JSON + "_backend": "official". The same
#   merge/fold machinery consumes it (merge_live.py / append_live_abstracted
#   glob RedditLive/*.jsonl.zst) - dedup by id as always, so overlap with
#   FetchLayer pulls or previous runs is harmless.
# PERMANENCE: raw files accumulate forever (nothing is ever re-pulled) and
#   the fold ledgers guarantee each post enters the pipeline exactly once.
# SPEED - THE WATERMARK: Arctic Shift archives by CREATION TIME with
#   complete coverage, so once a subreddit has been fetched through time T,
#   posts created before T can never appear later - re-fetching them is
#   pure waste. A per-subreddit watermark (newest created_utc seen, kept in
#   data/reference/reddit_arctic_watermark.json) lets every run after the
#   first fetch only what is NEW (minus a 1-day safety overlap for posts
#   that reach the archive late). A watermark only advances when the sub's
#   pagination COMPLETED - a run that gave up mid-sub re-covers the window
#   next time. First run / --lookback-days farther back than the watermark:
#   behaves exactly as before. Result: a daily run's Reddit pass drops from
#   many minutes (full week, every sub, every page) to ~1 minute.

import argparse
import datetime
import json
import os
import sys
import threading
import time

import requests
import zstandard

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "RedditLive")
SEEN_FILE = os.path.join(PROJECT_ROOT, "data", "reference",
                         "reddit_arctic_seen.json")
SUBS_FILE = os.path.join(PROJECT_ROOT, "ingestion",
                         "finance_subreddits.txt")
WATERMARK_FILE = os.path.join(PROJECT_ROOT, "data", "reference",
                              "reddit_arctic_watermark.json")
OVERLAP_S = 86400          # 1-day overlap behind the watermark (late arrivals)
API = "https://arctic-shift.photon-reddit.com/api/posts/search"
PAGE = 100
PAUSE_S = 1.0
MAX_SEEN = 50_000     # rolling window of recently-written ids

# ---------------------------------------------------------------------
# BACKFILL SPEED (2026-08-21). The 2024-08 chunk took 712 minutes and
# still lost three subreddits. Four things were costing that time, none
# of them the network's fault:
#
#   1. limit=100. Arctic Shift accepts limit="auto", which returns
#      between 100 and 1000 rows depending on server capacity - up to
#      TEN TIMES fewer round trips for the same posts.
#   2. Every field. Arctic returns the full Reddit object (preview,
#      media_metadata, all_awardings, gildings, ...). This project reads
#      exactly EIGHT of those fields - see src/reddit_live_data.py, which
#      normalises through src.clean_data.normalise. The rest is
#      downloaded, decompressed, parsed and thrown away. `fields=` makes
#      the request return only what is kept, which is lossless here.
#   3. A flat 1-second sleep after every page, whether or not the server
#      wanted one. Arctic publishes X-RateLimit-Remaining; pacing off
#      that sleeps when the server is actually near its limit and not
#      otherwise.
#   4. 4xx treated as a network hiccup. A 422 is the server saying the
#      REQUEST is malformed - retrying it four times with 20/40/60/80s
#      backoff burns 200 seconds to be told the same thing again, and
#      then counts as "gave up", which is what left chunks unfinished.
#
# The live daily path is unchanged: these apply to --backfill, or to any
# run that passes --fast explicitly.
# ---------------------------------------------------------------------

# The only fields anything downstream reads. Keep in step with
# src/reddit_live_data.py::OUTPUT_COLUMNS and src.clean_data.normalise.
KEEP_FIELDS = ("id,created_utc,author,score,subreddit,title,selftext,"
               "num_comments")

RATELIMIT_FLOOR = 5       # start waiting when this few requests remain
FAST_PAUSE_S = 0.0        # pacing comes from the rate-limit header instead

# 4xx handling, split by how confident we can be about the cause.
#
# HONEST NOTE (2026-08-21): the desk reported "422 https stuff" during the
# backfill. I could NOT reproduce a 422 from this file's own pagination -
# I built a stand-in server that answers an inverted range with 422 and
# drove the original loop at it with dense boundaries and duplicate
# timestamps, and the loop terminated cleanly every time. So the 422 is
# coming from the server for a reason we have not identified, and the
# code must NOT pretend to know which.
#
# Therefore: 400/404 are unambiguous - the URL or a parameter is wrong,
# and retrying is pointless. 403/422 get a SHORT retry (a server under
# load may answer 422 transiently) and the response BODY is printed, so
# the next run tells us what Arctic actually objects to instead of us
# guessing again. Either way the old behaviour - four retries over 200
# seconds, then marking the whole subreddit "gave up" - is gone.
HARD_4XX = {400, 404}         # the request is malformed; stop
SOFT_4XX = {403, 422}         # might be transient; a couple of quick tries
SOFT_4XX_TRIES = 3
SOFT_4XX_BACKOFF = (2, 5, 10)


def read_subreddits():
    subs = []
    with open(SUBS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                subs.append(line)
    return subs


def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            return list(json.load(open(SEEN_FILE, encoding="utf-8")))
        except Exception:
            return []
    return []


def save_seen(seen_list):
    os.makedirs(os.path.dirname(SEEN_FILE), exist_ok=True)
    with open(SEEN_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(seen_list[-MAX_SEEN:], f)
    os.replace(SEEN_FILE + ".tmp", SEEN_FILE)


def load_watermarks():
    if os.path.exists(WATERMARK_FILE):
        try:
            return json.load(open(WATERMARK_FILE, encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def save_watermarks(marks):
    os.makedirs(os.path.dirname(WATERMARK_FILE), exist_ok=True)
    with open(WATERMARK_FILE + ".tmp", "w", encoding="utf-8") as f:
        json.dump(marks, f)
    os.replace(WATERMARK_FILE + ".tmp", WATERMARK_FILE)


class Stop(Exception):
    """The server refused the REQUEST. Retrying cannot help."""


def fetch_page(sub, after, before, retries=4, session=None, fast=False):
    """One page. Returns rows, or None when the window was not covered.

    Raises Stop on a 4xx that means the request itself is wrong - the
    caller ends this subreddit cleanly instead of spending 200 seconds
    re-asking a question the server has already rejected.
    """
    get = (session or requests).get
    params = {"subreddit": sub, "after": after, "before": before,
              "limit": "auto" if fast else PAGE}
    if fast:
        params["fields"] = KEEP_FIELDS
    soft = 0
    for attempt in range(retries):
        try:
            r = get(API, params=params, timeout=(10, 120))
            if r.status_code == 200:
                _pace(r, fast)
                return r.json().get("data", [])
            if r.status_code == 429:
                # the one 4xx that IS about the moment
                wait = float(r.headers.get("Retry-After") or 30)
                print(f"    rate limited - waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in HARD_4XX:
                raise Stop(f"HTTP {r.status_code}: {r.text[:300]}")
            if r.status_code in SOFT_4XX:
                soft += 1
                body = r.text[:300].replace("\n", " ")
                print(f"    HTTP {r.status_code} from r/{sub} "
                      f"(try {soft}/{SOFT_4XX_TRIES}) - server said: "
                      f"{body}", flush=True)
                if soft >= SOFT_4XX_TRIES:
                    raise Stop(f"HTTP {r.status_code} {soft}x: {body}")
                time.sleep(SOFT_4XX_BACKOFF[min(soft - 1,
                                                len(SOFT_4XX_BACKOFF) - 1)])
                continue
            print(f"    HTTP {r.status_code} - backing off "
                  f"{20 * (attempt + 1)}s...", flush=True)
        except requests.RequestException as e:
            print(f"    network hiccup ({e}) - retrying...", flush=True)
        time.sleep(20 * (attempt + 1))
    print(f"    r/{sub}: giving up this run (next run re-covers the window)")
    return None                    # None = FAILED (vs [] = genuinely empty)


def _pace(resp, fast):
    """Sleep only when the SERVER says to, not on a fixed timer."""
    if not fast:
        return
    try:
        remaining = int(resp.headers.get("X-RateLimit-Remaining", "999"))
    except ValueError:
        remaining = 999
    if remaining <= RATELIMIT_FLOOR:
        try:
            reset = float(resp.headers.get("X-RateLimit-Reset", "5"))
        except ValueError:
            reset = 5.0
        time.sleep(max(0.5, min(reset, 60.0)))
    elif FAST_PAUSE_S:
        time.sleep(FAST_PAUSE_S)


def main():
    p = argparse.ArgumentParser(description="Live Reddit via Arctic Shift.")
    p.add_argument("--lookback-days", type=int, default=7,
                   help="fetch posts from the last N days (overlap dedups)")
    p.add_argument("--max-credits", type=int, default=0,
                   help="ignored - Arctic Shift is free (accepted so the "
                        "shared fetch knobs don't error)")
    p.add_argument("--test", action="store_true",
                   help="one page from one subreddit, print, write nothing")
    p.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                   help="fetch an explicit PAST window (YYYY-MM-DD "
                        "YYYY-MM-DD) and IGNORE the watermark - the only "
                        "way to fill a historical gap, because the "
                        "incremental window is max(lookback, watermark) "
                        "and therefore cannot walk backwards. The "
                        "watermark is left untouched by a backfill, so "
                        "the next ordinary run still resumes from the "
                        "present. Dedup is by post id, so overlapping "
                        "an already-fetched span is harmless.")
    p.add_argument("--fast", dest="fast", action="store_true", default=None,
                   help="limit=auto + minimal fields + rate-limit pacing. "
                        "ON by default for --backfill; the daily live run "
                        "keeps the old conservative behaviour unless you "
                        "ask for it here.")
    p.add_argument("--no-fast", dest="fast", action="store_false",
                   help="force the old one-page-at-a-time behaviour")
    p.add_argument("--subreddits", default="",
                   help="comma-separated subset to fetch instead of all of "
                        "finance_subreddits.txt. Coverage measured on the "
                        "healthy 2026 window: wallstreetbets alone keeps "
                        "24%% of covered name-days, +valueinvesting+stocks "
                        "59%%, +dividends+bogleheads 76%%.")
    p.add_argument("--workers", type=int, default=1,
                   help="fetch this many subreddits concurrently. "
                        "Subreddits are independent, so this is the one "
                        "safe axis to parallelise; pages within a "
                        "subreddit stay strictly sequential because each "
                        "page's cursor comes from the one before it.")
    args = p.parse_args()

    subs = read_subreddits()
    if args.subreddits:
        want = {x.strip().lower() for x in args.subreddits.split(",")
                if x.strip()}
        missing = want - {s.lower() for s in subs}
        subs = [s for s in subs if s.lower() in want]
        if missing:
            print(f"NOTE: not in finance_subreddits.txt, fetching anyway: "
                  f"{sorted(missing)}")
            subs += sorted(missing)
        if not subs:
            p.error("--subreddits matched nothing")

    # fast is the default for a backfill and only for a backfill
    fast = args.fast if args.fast is not None else bool(args.backfill)
    today = datetime.date.today()
    if args.backfill:
        after, before = args.backfill
        try:
            datetime.date.fromisoformat(after)
            datetime.date.fromisoformat(before)
        except ValueError:
            p.error("--backfill dates must be YYYY-MM-DD")
        print(f"BACKFILL {after} -> {before} across {len(subs)} "
              "subreddits (watermark ignored and left unchanged)")
    else:
        after = (today
                 - datetime.timedelta(days=args.lookback_days)).isoformat()
        before = (today + datetime.timedelta(days=1)).isoformat()

    if args.test:
        rows = fetch_page(subs[0], after, before, fast=fast) or []
        print(f"TEST: r/{subs[0]} returned {len(rows)} posts "
              f"({after} -> {before}); first titles:")
        for rec in rows[:3]:
            print("  -", str(rec.get("title", ""))[:70])
        return 0

    seen_list = load_seen()
    seen = set(seen_list)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"reddit_live_arctic_{stamp}.jsonl.zst")

    marks = load_watermarks()
    lookback_epoch = int(time.time()) - args.lookback_days * 86400
    total = 0
    writer = zstandard.ZstdCompressor().stream_writer(
        open(out_path + ".tmp", "wb"))
    wlock = threading.Lock()          # writer + seen are shared across workers

    def _epoch(v):
        """Accept 'YYYY-MM-DD' or an epoch string; return int seconds."""
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return int(datetime.datetime.fromisoformat(
                str(v)).replace(tzinfo=datetime.timezone.utc).timestamp())

    before_epoch = _epoch(before)

    def do_sub(sub, session):
        # INCREMENTAL WINDOW: never before the requested lookback, but if a
        # watermark exists, start just behind it - everything older was
        # already fetched (Arctic archives by creation time, complete).
        sub_after = after
        wm = marks.get(sub)
        if wm and not args.backfill:
            sub_after = str(max(lookback_epoch, int(wm) - OVERLAP_S))
        after_epoch = _epoch(sub_after)
        got = 0
        pages = 0
        newest_seen = int(wm) if wm else 0
        completed = True                      # pagination reached the end?
        cursor = before_epoch
        while True:
            # THE 422. The cursor walks BACKWARDS (each page's oldest post
            # becomes the next page's `before`). Once it reaches the start
            # of the window, `before` <= `after` - an empty, invalid range,
            # which Arctic answers with 422. The old loop then treated that
            # as a network fault and retried it four times over 200 seconds
            # before declaring the subreddit "gave up", which is why chunks
            # finished partial. It is not an error at all: it is the end of
            # the window, and the right response is to stop.
            if cursor <= after_epoch:
                break
            try:
                rows = fetch_page(sub, sub_after, str(cursor),
                                  session=session, fast=fast)
            except Stop as e:
                print(f"    r/{sub}: server refused the request ({e}) - "
                      f"ending this subreddit", flush=True)
                completed = False
                break
            if rows is None:               # gave up after retries: the
                completed = False          # window was NOT fully covered,
                break                      # so the watermark must not move
            if not rows:
                break                      # clean end: no more posts
            pages += 1
            batch = []
            oldest = cursor
            for rec in rows:
                pid = str(rec.get("id", ""))
                created = int(rec.get("created_utc", 0) or 0)
                if created:
                    oldest = min(oldest, created)
                newest_seen = max(newest_seen, created)
                if not pid:
                    continue
                rec["_backend"] = "official"      # Pushshift/official shape
                batch.append((pid, rec))
            with wlock:
                for pid, rec in batch:
                    if pid in seen:
                        continue
                    seen.add(pid)
                    seen_list.append(pid)
                    writer.write((json.dumps(rec) + "\n").encode("utf-8"))
                    got += 1
            # NO PROGRESS GUARD. If every row on a page shares the oldest
            # timestamp, `oldest` never moves and the old loop would ask
            # for the same page forever. Step one second past it.
            nxt = oldest if oldest < cursor else cursor - 1
            if nxt >= cursor:
                break
            cursor = nxt
            if not fast:
                time.sleep(PAUSE_S)
        wm_note = " (incremental)" if wm else ""
        print(f"  r/{sub:<24} {got:>5} new posts  [{pages} page(s)]{wm_note}",
              flush=True)
        # advance the watermark only on a clean finish with data seen -
        # and NEVER on a backfill: the watermark is "how far forward we
        # have come", and a historical window would drag it backwards
        if completed and newest_seen and not args.backfill:
            marks[sub] = newest_seen
        return got

    workers = max(1, int(args.workers))
    if workers == 1:
        with requests.Session() as sess:
            for sub in subs:
                total += do_sub(sub, sess)
                if not fast:
                    time.sleep(PAUSE_S)
    else:
        # One Session PER WORKER: a Session is not documented as thread
        # safe, and sharing one is the classic source of "connection pool
        # is full" warnings and cross-talk between requests.
        from concurrent.futures import ThreadPoolExecutor
        print(f"  fetching {len(subs)} subreddits with {workers} workers",
              flush=True)
        local = threading.local()

        def _run(sub):
            if not hasattr(local, "sess"):
                local.sess = requests.Session()
            return do_sub(sub, local.sess)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for n in ex.map(_run, subs):
                total += n
    writer.close()
    save_watermarks(marks)

    if total == 0:
        os.remove(out_path + ".tmp")
        print("no new posts this run (all already seen) - nothing written")
        return 0
    os.replace(out_path + ".tmp", out_path)
    save_seen(seen_list)
    print(f"arctic reddit: {total:,} new posts -> {os.path.basename(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
