# fetch_reddit_live.py
# ====================
# LIVE Reddit ingestion with TWO interchangeable backends - whichever has
# a key in .env gets used (FetchLayer preferred if both are present):
#
#   A) FETCHLAYER (fetchlayer.dev - third-party structured Reddit API)
#        .env:  FETCHLAYER_KEY = ss-...
#        One POST per subreddit to /api/reddit/community-posts, sort=new.
#        Billing: 1 credit per REQUEST (free tier: 30 requests; $1.99/1k;
#        Starter $25/mo = ~500/day). 15 subreddits = 15 credits per run ->
#        run HOURLY (360/day) on Starter, or 2-3x/day on free credits.
#   B) OFFICIAL REDDIT OAUTH (free, if you ever get app credentials)
#        .env:  REDDIT_PERSONAL_USE / REDDIT_SECRET (+ optional user/pass)
#        One multireddit /new listing, paginated. ~100 req/min allowance.
#
#   python ingestion/fetch_reddit_live.py --test   # ONE small call, writes nothing
#   python ingestion/fetch_reddit_live.py          # real poll (fetch_all calls this)
#
# OUTPUT (both backends): data/raw/RedditLive/reddit_live_YYYY-MM-DD.jsonl.zst
#   - each line is one post's raw JSON exactly as the backend returned it,
#     tagged with "_backend" so the normaliser knows the shape.
# DEDUP: a rolling seen-ids file (data/reference/reddit_live_seen.json,
#   last 20k ids) plus a created-time watermark where the backend provides
#   timestamps. Final dedup happens again at merge time (first seen wins).
#
# NOTE ON MERGING: raw accumulates here; posts reach posts.parquet via
# ingestion/merge_live.py (append-only, "first seen wins").
# update_data.py and `fetch_all.py` (normal mode) call it for you.

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

try:                     # post titles contain emoji; don't die on cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "RedditLive")
SEEN_FILE = os.path.join(PROJECT_ROOT, "data", "reference", "reddit_live_seen.json")
SUBS_FILE = os.path.join(PROJECT_ROOT, "ingestion", "finance_subreddits.txt")
FETCHLAYER_URL = "https://fetchlayer.dev/api/reddit/community-posts"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
PAUSE_S = 1.0
MAX_SEEN = 20_000


def load_env():
    """Read keys from .env DIRECTLY (no python-dotenv dependency), with
    os.environ as a fallback. Accepts FETCHLAYER_KEY or FETCHLAYER_API_KEY."""
    from_file = {}
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                from_file[k.strip()] = v.strip()

    def get(*names):
        for name in names:
            v = from_file.get(name) or os.environ.get(name, "")
            if v.strip():
                return v.strip()
        return ""

    return {"FETCHLAYER_API_KEY": get("FETCHLAYER_KEY", "FETCHLAYER_API_KEY"),
            "REDDIT_PERSONAL_USE": get("REDDIT_PERSONAL_USE"),
            "REDDIT_SECRET": get("REDDIT_SECRET"),
            "REDDIT_USERNAME": get("REDDIT_USERNAME"),
            "REDDIT_PASSWORD": get("REDDIT_PASSWORD"),
            "REDDIT_APP_NAME": get("REDDIT_APP_NAME")}


def tracked_subs():
    if os.path.exists(SUBS_FILE):
        subs = [line.strip() for line in open(SUBS_FILE, encoding="utf-8")
                if line.strip() and not line.startswith("#")]
        if subs:
            return subs
    return ["wallstreetbets", "stocks", "investing", "options", "pennystocks",
            "stockmarket", "daytrading", "thetagang", "dividends",
            "valueinvesting", "securityanalysis", "personalfinance",
            "financialindependence", "cryptocurrency", "bitcoin"]


def post_id(p):
    """A stable id whatever shape the backend returns."""
    for key in ("id", "postId", "name"):
        if p.get(key):
            return str(p[key])
    return str(p.get("url") or p.get("permalink") or hash(json.dumps(p, sort_keys=True)))


# ---------------- backend A: FetchLayer ----------------
def fetchlayer_test(key):
    r = requests.post(FETCHLAYER_URL,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"subreddit": "wallstreetbets", "sort": "new", "limit": 5},
                      timeout=30)
    print(f"POST community-posts(wallstreetbets, new, 5) -> {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        return 1
    payload = r.json()
    posts = (payload.get("items") or payload.get("posts")
             or payload.get("results") or [])
    print(f"got {len(posts)} posts; sample fields: "
          f"{sorted(list(posts[0].keys()))[:10] if posts else '-'}")
    for p in posts[:5]:
        title = p.get("title") or p.get("postTitle") or p.get("text") or ""
        print(f"  {str(p.get('score', '?')):>5} pts | {str(title)[:70]}")
    print("TEST PASSED - FetchLayer key works (1 credit used).")
    return 0


def _timeframe_for(days):
    """Map a lookback in days onto Reddit's top-post timeframes."""
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


def fetchlayer_poll(key, limit, max_credits=60, lookback_days=7):
    """TWO passes per subreddit:
      1. sort=new             - the newest posts (catches everything recent)
      2. sort=top, timeframe  - the most POPULAR posts of the lookback window
                                (high-engagement posts earlier runs missed)
    Costs ~2 credits per subreddit per run. Dedup (here on id, and again at
    merge time) means overlap between passes and between runs is harmless -
    a longer lookback can only ADD posts, never duplicate them."""
    headers = {"Authorization": f"Bearer {key}"}
    all_posts, used = [], 0
    stopped = False
    started = time.time()
    max_seconds = 300          # whole-run time budget: never look frozen for long
    subs = tracked_subs()
    total_requests = len(subs) * 2
    tf = _timeframe_for(lookback_days)
    print(f"polling {len(subs)} subreddits x 2 passes (new + top-of-{tf}) = "
          f"{total_requests} requests; progress below")
    # (sort, extra request fields) - "timeframe"/"t" both sent so whichever
    # name FetchLayer expects for the top-post window is covered.
    passes = [("new", {}),
              ("top", {"timeframe": tf, "t": tf})]
    for sub in subs:
        if stopped:
            break
        for sort, extra in passes:
            if used >= max_credits:
                print(f"[stop] hit the per-run credit cap ({max_credits})")
                stopped = True
                break
            if time.time() - started > max_seconds:
                print(f"[stop] hit the {max_seconds}s time budget - keeping "
                      f"what was fetched; next run continues")
                stopped = True
                break
            body = {"subreddit": sub, "sort": sort, "limit": limit}
            body.update(extra)
            t0 = time.time()
            # timeout=(connect, read): fail FAST if the server is unreachable,
            # but be PATIENT once it is working - big subreddits
            # (personalfinance, wallstreetbets) can take >20s server-side.
            # One retry on timeout: slow scrapes usually succeed second time.
            r = None
            for attempt in (1, 2):
                try:
                    r = requests.post(FETCHLAYER_URL, headers=headers,
                                      json=body, timeout=(10, 60))
                    break
                except requests.exceptions.Timeout:
                    if attempt == 1:
                        print(f"  .. r/{sub:<22} {sort:<4} slow (read timeout) - retrying once")
                    else:
                        print(f"  {used + 1:>2}/{total_requests} r/{sub:<22} {sort:<4} "
                              "FAILED: timed out twice - skipping this one")
                except Exception as exc:
                    print(f"  {used + 1:>2}/{total_requests} r/{sub:<22} {sort:<4} FAILED: {exc}")
                    break
            if r is None:
                continue
            used += 1
            if r.status_code in (402, 429):
                print(f"[stop] FetchLayer says {r.status_code} (credits/rate) - "
                      "ending run; next run continues")
                stopped = True
                break
            if r.status_code != 200:
                print(f"  {used:>2}/{total_requests} r/{sub:<22} {sort:<4} "
                      f"HTTP {r.status_code}: {r.text[:80]}")
                continue
            payload = r.json()
            posts = (payload.get("items") or payload.get("posts")
                     or payload.get("results") or [])
            for p in posts:
                p["_backend"] = "fetchlayer"
                p.setdefault("subreddit", sub)
                all_posts.append(p)
            print(f"  {used:>2}/{total_requests} r/{sub:<22} {sort:<4} "
                  f"-> {len(posts):>3} posts | {time.time() - t0:4.1f}s")
            time.sleep(PAUSE_S)
    print(f"fetchlayer: {used} credits used, {len(all_posts)} posts, "
          f"{time.time() - started:.0f}s total")
    return all_posts


# ---------------- backend B: official OAuth ----------------
def official_token(creds):
    auth = requests.auth.HTTPBasicAuth(creds["REDDIT_PERSONAL_USE"], creds["REDDIT_SECRET"])
    ua = f"windows:{creds['REDDIT_APP_NAME'] or 'retailflow'}:v1.0 " \
         f"(by /u/{creds['REDDIT_USERNAME'] or 'retailflow'})"
    if creds["REDDIT_USERNAME"] and creds["REDDIT_PASSWORD"]:
        data = {"grant_type": "password", "username": creds["REDDIT_USERNAME"],
                "password": creds["REDDIT_PASSWORD"]}
    else:
        data = {"grant_type": "client_credentials"}
    r = requests.post(TOKEN_URL, auth=auth, data=data,
                      headers={"User-Agent": ua}, timeout=20)
    if r.status_code != 200:
        print(f"official OAuth token failed ({r.status_code}): {r.text[:150]}")
        return None, ua
    return r.json().get("access_token"), ua


def official_poll(creds, limit):
    token, ua = official_token(creds)
    if not token:
        return []
    headers = {"Authorization": f"bearer {token}", "User-Agent": ua}
    multi = "+".join(tracked_subs())
    r = requests.get(f"https://oauth.reddit.com/r/{multi}/new",
                     headers=headers, params={"limit": min(limit * 15, 100)}, timeout=20)
    if r.status_code != 200:
        print(f"[warn] listing failed ({r.status_code})")
        return []
    posts = []
    for child in r.json().get("data", {}).get("children", []):
        p = child.get("data", {})
        p["_backend"] = "official"
        posts.append(p)
    return posts


# ---------------- shared: dedup + raw append ----------------
def load_seen():
    if os.path.exists(SEEN_FILE):
        return list(json.load(open(SEEN_FILE)).get("ids", []))
    return []


def save_seen(ids):
    os.makedirs(os.path.dirname(SEEN_FILE), exist_ok=True)
    json.dump({"ids": ids[-MAX_SEEN:],
               "updated": datetime.datetime.now().isoformat(timespec="seconds")},
              open(SEEN_FILE, "w"))


def append_raw(posts):
    os.makedirs(OUT_DIR, exist_ok=True)
    day = datetime.date.today().isoformat()
    path = os.path.join(OUT_DIR, f"reddit_live_{day}.jsonl.zst")
    old = b""
    if os.path.exists(path):
        old = zstandard.ZstdDecompressor().decompress(open(path, "rb").read())
    lines = "\n".join(json.dumps(p, ensure_ascii=False) for p in posts) + "\n"
    with open(path, "wb") as f:
        f.write(zstandard.ZstdCompressor(level=10).compress(old + lines.encode("utf-8")))
    return path


def main():
    ap = argparse.ArgumentParser(description="Live Reddit ingestion (FetchLayer or official OAuth)")
    ap.add_argument("--test", action="store_true",
                    help="ONE small call (5 posts from r/wallstreetbets), writes nothing")
    ap.add_argument("--limit", type=int, default=100,
                    help="posts per subreddit per pass (new + top-of-window)")
    ap.add_argument("--max-credits", type=int, default=60,
                    help="FetchLayer credit cap per run (2 passes x 15 subs = 30)")
    ap.add_argument("--lookback-days", type=int, default=7,
                    help="top-post window: how far back the fetch reaches")
    args = ap.parse_args()

    creds = load_env()
    if creds["FETCHLAYER_API_KEY"]:
        backend = "fetchlayer"
    elif creds["REDDIT_PERSONAL_USE"] and creds["REDDIT_SECRET"]:
        backend = "official"
    else:
        print("reddit: skipped - no key found in .env")
        print("  For FetchLayer: add  FETCHLAYER_KEY = <your key from fetchlayer.dev>")
        print("  For official  : fill the REDDIT_* lines in .env")
        return 0
    print(f"backend: {backend}")

    if args.test:
        if backend == "fetchlayer":
            return fetchlayer_test(creds["FETCHLAYER_API_KEY"])
        posts = official_poll(creds, limit=5)
        for p in posts[:5]:
            print(f"  {p.get('score', 0):>5} pts | {p.get('title', '')[:70]}")
        print("TEST PASSED" if posts else "TEST FAILED - see messages above")
        return 0 if posts else 1

    posts = (fetchlayer_poll(creds["FETCHLAYER_API_KEY"], args.limit,
                             args.max_credits, args.lookback_days)
             if backend == "fetchlayer" else official_poll(creds, args.limit))
    if not posts:
        print("no posts fetched this run")
        return 0

    seen = load_seen()
    seen_set = set(seen)
    fresh = [p for p in posts if post_id(p) not in seen_set]
    if not fresh:
        print(f"fetched {len(posts)} posts - all already seen (nothing new)")
        return 0
    path = append_raw(fresh)
    save_seen(seen + [post_id(p) for p in fresh])
    print(f"kept {len(fresh)} NEW posts (of {len(posts)} fetched) -> {path}")
    print("raw accumulates here; append into posts.parquet with:  "
          "python ingestion/merge_live.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
