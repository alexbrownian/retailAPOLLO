"""Manual fallback for live Reddit ingestion with two interchangeable backends.

``fetch_reddit_arctic.py`` is the default Reddit source; this script
remains available when a keyed backend is preferred. Whichever backend has
a key in ``.env`` is used (FetchLayer preferred if both are present):

A. FetchLayer (fetchlayer.dev, a third-party structured Reddit API).
   ``.env``: ``FETCHLAYER_KEY = ss-...``. One POST per subreddit and pass
   to ``/api/reddit/community-posts``. Billing is one credit per request,
   so a 15-subreddit panel costs about 30 credits per run.
B. Official Reddit OAuth (free, given app credentials).
   ``.env``: ``REDDIT_PERSONAL_USE`` / ``REDDIT_SECRET`` (plus optional
   username/password). One multireddit ``/new`` listing.

Usage::

    python ingestion/fetch_reddit_live.py --test   # one small call, writes nothing
    python ingestion/fetch_reddit_live.py          # real poll

Output (both backends): ``data/raw/RedditLive/reddit_live_YYYY-MM-DD.jsonl.zst``.
Each line is one post's raw JSON exactly as the backend returned it,
tagged with ``"_backend"`` so the normaliser knows the shape.

Dedup: a rolling seen-ids file (``data/reference/reddit_live_seen.json``,
last ``MAX_SEEN`` ids). Final dedup happens again at merge time (first
seen wins). Raw accumulates here; posts reach ``posts.parquet`` via
``ingestion/merge_live.py`` (append-only), which ``update_data.py`` and
``fetch_all.py`` run for you.
"""

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
# Forum panel: config/forums.csv via src.settings.load_forums().
FETCHLAYER_URL = "https://fetchlayer.dev/api/reddit/community-posts"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
PAUSE_S = 1.0
MAX_SEEN = 20_000


def load_env():
    """Read credentials from ``.env`` directly, with ``os.environ`` as fallback.

    Accepts ``FETCHLAYER_KEY`` or ``FETCHLAYER_API_KEY`` for FetchLayer.

    Returns:
        Dict with the ``FETCHLAYER_API_KEY`` and ``REDDIT_*`` values;
        missing keys are empty strings.
    """
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
    """Return the enabled forums from ``config/forums.csv``.

    Falls back to a built-in panel when the settings module cannot be
    loaded, so the script still runs standalone.
    """
    try:
        from src.settings import load_forums
        return list(load_forums())
    except Exception:                                    # noqa: BLE001
        pass
    return ["wallstreetbets", "stocks", "investing", "options", "pennystocks",
            "stockmarket", "daytrading", "thetagang", "dividends",
            "valueinvesting", "securityanalysis", "personalfinance",
            "financialindependence", "cryptocurrency", "bitcoin"]


def post_id(p):
    """Return a stable id for a post whatever shape the backend returns."""
    for key in ("id", "postId", "name"):
        if p.get(key):
            return str(p[key])
    return str(p.get("url") or p.get("permalink") or hash(json.dumps(p, sort_keys=True)))


# ---------------- backend A: FetchLayer ----------------
def fetchlayer_test(key):
    """Make one five-post FetchLayer call and print the result.

    Args:
        key: FetchLayer API key.

    Returns:
        ``0`` when the call succeeded, ``1`` otherwise.
    """
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
    """Poll every tracked subreddit through FetchLayer.

    Two passes per subreddit: ``sort=new`` for the newest posts, and
    ``sort=top`` over the lookback timeframe for the most popular posts of
    the window (high-engagement posts earlier runs missed). Costs about
    two credits per subreddit per run. Dedup (here on id, and again at
    merge time) makes overlap between passes and between runs harmless, so
    a longer lookback can only add posts, never duplicate them. The run
    stops early at the credit cap, at a fixed time budget, or when
    FetchLayer answers 402/429.

    Args:
        key: FetchLayer API key.
        limit: Posts requested per subreddit per pass.
        max_credits: Requests allowed this run.
        lookback_days: Window for the top-post pass.

    Returns:
        List of raw post dicts tagged with ``"_backend": "fetchlayer"``.
    """
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
    """Obtain an OAuth token from Reddit.

    Args:
        creds: The dict returned by ``load_env``.

    Returns:
        Tuple ``(access_token, user_agent)``; the token is ``None`` when
        the request failed.
    """
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
    """Fetch one multireddit ``/new`` listing through the official API.

    Args:
        creds: The dict returned by ``load_env``.
        limit: Posts per subreddit; the listing asks for ``min(limit * 15,
            100)`` posts in total.

    Returns:
        List of raw post dicts tagged with ``"_backend": "official"``;
        empty on any failure.
    """
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
    """Return the rolling list of previously written post ids."""
    if os.path.exists(SEEN_FILE):
        return list(json.load(open(SEEN_FILE)).get("ids", []))
    return []


def save_seen(ids):
    """Write the newest ``MAX_SEEN`` ids of ``ids`` with a timestamp."""
    os.makedirs(os.path.dirname(SEEN_FILE), exist_ok=True)
    json.dump({"ids": ids[-MAX_SEEN:],
               "updated": datetime.datetime.now().isoformat(timespec="seconds")},
              open(SEEN_FILE, "w"))


def append_raw(posts):
    """Append posts to today's raw file, rewriting the zstd blob.

    Args:
        posts: Raw post dicts to write, one JSON line each.

    Returns:
        Path of the file written.
    """
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
    """Pick a backend from the credentials, poll it and append new posts.

    Returns:
        ``0`` on success or when nothing new was fetched; ``1`` when the
        official-backend test found no posts.
    """
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
