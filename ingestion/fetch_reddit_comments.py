# fetch_reddit_comments.py
# ========================
# COMMENT ingestion from the Arctic Shift public API - the data source for
# the INFLUENCE TRACKER (per-author calls + the reply graph that maps who
# responds to whom). Comments carry: author, body, created_utc, link_id
# (the post they belong to) and parent_id (what they reply to) - the last
# two are the edges of the social interaction graph.
#
#   python ingestion/fetch_reddit_comments.py                       # live: last 7d
#   python ingestion/fetch_reddit_comments.py --lookback-days 30
#   python ingestion/fetch_reddit_comments.py --backfill 2021-01-01 2021-06-30
#   python ingestion/fetch_reddit_comments.py --test                # one page
#
# DATA BOUNDARY (desk decision, July 2026 - REVISED): the raw comment
#   files stay LOCAL (gitignored, like all raw text), but the influence
#   STORE derived from them (calls/scores/edges - text-free, pseudonymous)
#   is committed and shared. Raw text never crosses git; metadata does.
#
# SCOPE (desk decision, July 2026): comments run LIVE-FIRST - fetch_all
#   calls this script on every live pass (watermarked, incremental), and
#   the recommended one-off backfill is the CURRENT YEAR only. Deep
#   multi-year comment history was descoped: at the API's polite rate
#   (1s/page, 100/page) busy subreddits cost hours per half-year, and the
#   influence tracker's value is who is right NOW.
#
# TIME PARAMETERS: after/before are normalised to EPOCH SECONDS before
#   the first request and stay epoch for every page. (The first version
#   sent page 1 with ISO dates and then paginated with an epoch cursor -
#   mixed formats in one request drew HTTP 422s from the API. One format,
#   chosen once, everywhere.)
#
# OUTPUT: data/raw/RedditComments/comments_<range>.jsonl.zst
#   one raw JSON comment per line. Dedup by comment id via a rolling
#   seen-file; a watermark per subreddit makes repeat live runs
#   incremental exactly like the post fetcher.

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

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "RedditComments")
SEEN_FILE = os.path.join(PROJECT_ROOT, "data", "reference",
                         "reddit_comments_seen.json")
WM_FILE = os.path.join(PROJECT_ROOT, "data", "reference",
                       "reddit_comments_watermark.json")
SUBS_FILE = os.path.join(PROJECT_ROOT, "ingestion", "finance_subreddits.txt")
API = "https://arctic-shift.photon-reddit.com/api/comments/search"
PAGE = 100
PAUSE_S = 1.0
MAX_SEEN = 200_000
# the fields the influence tracker needs - dropping the rest keeps the raw
# files a fraction of full-comment size
KEEP = ("id", "author", "body", "created_utc", "subreddit",
        "link_id", "parent_id", "score")


def read_subreddits():
    subs = []
    with open(SUBS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                subs.append(line)
    return subs


def _load(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {}


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def to_epoch(v) -> int:
    """One time format everywhere: YYYY-MM-DD (or any ISO date) -> epoch
    seconds at UTC midnight; a value that is already all digits passes
    through as-is. Called ONCE on the CLI bounds, so every request the
    crawl makes - first page and every cursor page after it - uses the
    same format. (Mixing 'after=2021-01-01' with an epoch 'before' cursor
    is what earned the HTTP 422s in the first version.)"""
    s = str(v)
    if s.isdigit():
        return int(s)
    d = datetime.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    return int(d.timestamp())


def fetch_page(sub, after, before, retries=4):
    """One API page. Retry policy by FAILURE TYPE:
    - 429 / 5xx / network drop: the server or the connection is having a
      moment - waiting and retrying is the right move (20s, 40s, 60s...).
    - other 4xx: OUR request is malformed - retrying the identical bad
      request four times cannot help, so print the server's explanation
      and stop immediately."""
    for attempt in range(retries):
        try:
            r = requests.get(API, params={"subreddit": sub,
                                          "after": int(after),
                                          "before": int(before),
                                          "limit": PAGE},
                             timeout=(10, 60))
            if r.status_code == 200:
                return r.json().get("data", [])
            if r.status_code == 429 or r.status_code >= 500:
                print(f"    HTTP {r.status_code} - backing off "
                      f"{20*(attempt+1)}s")
            else:
                print(f"    HTTP {r.status_code} (client error - not "
                      f"retrying): {r.text[:200]}")
                return None
        except requests.RequestException as e:
            print(f"    network problem ({type(e).__name__}) - retrying in "
                  f"{20*(attempt+1)}s. (wifi/VPN drop? safe to Ctrl-C and "
                  "re-run later: the seen-file dedups everything already "
                  "saved)")
        time.sleep(20 * (attempt + 1))
    print(f"    r/{sub}: giving up this run")
    return None


def main():
    p = argparse.ArgumentParser(description="Arctic Shift comment ingestion "
                                            "(influence tracker source)")
    p.add_argument("--lookback-days", type=int, default=3,
                   help="live window (default 3d - the per-subreddit "
                        "watermark makes every later run incremental, so "
                        "only the FIRST live run pays for the window)")
    p.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                   help="historical range YYYY-MM-DD YYYY-MM-DD (end excl); "
                        "desk scope is the current year only, e.g. "
                        "2026-01-01 <today>")
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    subs = read_subreddits()
    if args.backfill:
        label = f"{args.backfill[0]}_{args.backfill[1]}"
        # ONE format for the whole crawl - see to_epoch's docstring
        after = to_epoch(args.backfill[0])
        before = to_epoch(args.backfill[1])
        incremental = False
    else:
        today = datetime.date.today()
        after = to_epoch((today - datetime.timedelta(
            days=args.lookback_days)).isoformat())
        before = to_epoch((today + datetime.timedelta(days=1)).isoformat())
        label = today.isoformat()
        incremental = True

    if args.test:
        rows = fetch_page(subs[0], after, before) or []
        print(f"TEST: r/{subs[0]} returned {len(rows)} comments; sample:")
        for rec in rows[:3]:
            print(f"  u/{rec.get('author')}: {str(rec.get('body',''))[:60]}")
        return 0

    seen_obj = _load(SEEN_FILE)
    seen_list = seen_obj.get("ids", [])
    seen = set(seen_list)
    marks = _load(WM_FILE)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"comments_{label}.jsonl.zst")
    writer = zstandard.ZstdCompressor().stream_writer(
        open(out_path + ".tmp", "wb"))
    total = 0
    for sub in subs:
        sub_after = after                     # already epoch (see to_epoch)
        wm = marks.get(sub)
        if incremental and wm:
            # start from the watermark (minus a 1-day overlap for late
            # arrivals) if that is LATER than the lookback window start
            sub_after = max(after, int(wm) - 86400)
        got, newest, completed = 0, int(wm) if wm else 0, True
        cursor = before                       # epoch, stays epoch
        while True:
            rows = fetch_page(sub, sub_after, cursor)
            if rows is None:
                completed = False
                break
            if not rows:
                break
            for rec in rows:
                cid = str(rec.get("id", ""))
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                seen_list.append(cid)
                slim = {k: rec.get(k) for k in KEEP}
                newest = max(newest, int(rec.get("created_utc", 0) or 0))
                writer.write((json.dumps(slim) + "\n").encode("utf-8"))
                got += 1
            oldest = min(int(r["created_utc"]) for r in rows)
            if len(rows) < PAGE:
                break
            cursor = oldest                   # epoch int, same as page 1
            time.sleep(PAUSE_S)
        print(f"  r/{sub:<24} {got:>6} new comments")
        if incremental and completed and newest:
            marks[sub] = newest
        total += got
        time.sleep(PAUSE_S)
    writer.close()

    if total == 0:
        os.remove(out_path + ".tmp")
        print("no new comments this run")
        return 0
    os.replace(out_path + ".tmp", out_path)
    seen_obj["ids"] = seen_list[-MAX_SEEN:]
    _save(SEEN_FILE, seen_obj)
    if incremental:
        _save(WM_FILE, marks)
    print(f"comments: {total:,} new -> {os.path.basename(out_path)}")
    print("next:  python -m analytics.influence --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
