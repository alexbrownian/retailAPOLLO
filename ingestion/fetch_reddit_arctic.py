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


def fetch_page(sub, after, before, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(API, params={"subreddit": sub, "after": after,
                                          "before": before, "limit": PAGE},
                             timeout=(10, 60))
            if r.status_code == 200:
                return r.json().get("data", [])
            print(f"    HTTP {r.status_code} - backing off "
                  f"{20 * (attempt + 1)}s...")
        except requests.RequestException as e:
            print(f"    network hiccup ({e}) - retrying...")
        time.sleep(20 * (attempt + 1))
    print(f"    r/{sub}: giving up this run (next run re-covers the window)")
    return None                    # None = FAILED (vs [] = genuinely empty)


def main():
    p = argparse.ArgumentParser(description="Live Reddit via Arctic Shift.")
    p.add_argument("--lookback-days", type=int, default=7,
                   help="fetch posts from the last N days (overlap dedups)")
    p.add_argument("--max-credits", type=int, default=0,
                   help="ignored - Arctic Shift is free (accepted so the "
                        "shared fetch knobs don't error)")
    p.add_argument("--test", action="store_true",
                   help="one page from one subreddit, print, write nothing")
    args = p.parse_args()

    subs = read_subreddits()
    today = datetime.date.today()
    after = (today - datetime.timedelta(days=args.lookback_days)).isoformat()
    before = (today + datetime.timedelta(days=1)).isoformat()

    if args.test:
        rows = fetch_page(subs[0], after, before) or []
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
    for sub in subs:
        # INCREMENTAL WINDOW: never before the requested lookback, but if a
        # watermark exists, start just behind it - everything older was
        # already fetched (Arctic archives by creation time, complete).
        sub_after = after
        wm = marks.get(sub)
        if wm:
            sub_after = str(max(lookback_epoch, int(wm) - OVERLAP_S))
        got = 0
        newest_seen = int(wm) if wm else 0
        completed = True                      # pagination reached the end?
        cursor = before
        while True:
            rows = fetch_page(sub, sub_after, cursor)
            if rows is None:               # gave up after retries: the
                completed = False          # window was NOT fully covered,
                break                      # so the watermark must not move
            if not rows:
                break                      # clean end: no more posts
            for rec in rows:
                pid = str(rec.get("id", ""))
                created = int(rec.get("created_utc", 0) or 0)
                newest_seen = max(newest_seen, created)
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                seen_list.append(pid)
                rec["_backend"] = "official"      # Pushshift/official shape
                writer.write((json.dumps(rec) + "\n").encode("utf-8"))
                got += 1
            oldest = min(int(r["created_utc"]) for r in rows)
            if len(rows) < PAGE:
                break
            cursor = str(oldest)
            time.sleep(PAUSE_S)
        wm_note = " (incremental)" if wm else ""
        print(f"  r/{sub:<24} {got:>5} new posts{wm_note}")
        # advance the watermark only on a clean finish with data seen
        if completed and newest_seen:
            marks[sub] = newest_seen
        total += got
        time.sleep(PAUSE_S)
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
