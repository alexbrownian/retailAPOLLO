# fetch_x_live.py
# ===============
# LIVE X (Twitter) ingestion with TWO interchangeable backends - whichever
# has a key in .env gets used (FetchLayer preferred if both are present):
#
#   A) FETCHLAYER (fetchlayer.dev - third-party structured social API)
#        .env:  FETCHLAYER_KEY = ss-...           (the SAME key as Reddit)
#        One POST per cashtag-chunk to /api/twitter/search, product=Latest.
#        Billing: 1 credit per REQUEST (same pool as the Reddit fetcher).
#        NO paid X developer account needed - this is the path that works
#        today when you only have a FetchLayer key.
#   B) OFFICIAL X v2 API (needs a PAID developer account)
#        .env:  X_BEARER_TOKEN = AAAA....         (from developer.x.com)
#        v2 recent-search (last 7 days). Armed but off until you pay.
#
#   python ingestion/fetch_x_live.py --test   # ONE small call, writes nothing
#   python ingestion/fetch_x_live.py          # real poll (fetch_all calls this)
#   python ingestion/fetch_x_live.py --max-tweets 150
#
# OUTPUT (both backends): data/raw/X Data/x_api_live.csv.zst
#   a flat csv (id, created_at, text, author, likes) - a REGISTERED dataset
#   (x_api_live in src/x_data.py, normalise_x_api). Real tweet ids share the
#   'x_' prefix with the historical dumps, so overlaps dedupe automatically
#   (first seen wins). The merge into posts.parquet is done by
#   ingestion/merge_live.py (update_data.py calls it).
#
# QUOTA NOTES:
#   FetchLayer - 1 credit per chunk-request; the default symbol list is a
#     few chunks per run, so a run costs a handful of credits.
#   Official   - reads capped per MONTH and ~60 req/15min; --max-tweets is
#     your seat belt. Both backends stop instantly on HTTP 429/402.

import argparse
import datetime
import io
import os
import re
import sys
import time

import pandas as pd
import requests
import zstandard

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:                     # tweets contain emoji/links; don't die on cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.themes import THEME_ETFS  # noqa: E402

OUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "X Data", "x_api_live.csv.zst")
FETCHLAYER_URL = "https://fetchlayer.dev/api/twitter/search"
SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
PAGE_SIZE = 100          # max_results per request (10-100)
PAUSE_S = 5.0            # polite gap between requests (X scraping is the
                         # expensive endpoint - going faster earns 429s)

# DISCOVERY queries - broad finance chatter, NO fixed tickers. The point:
# our extractor pulls every valid ticker out of post TEXT, so scraping the
# week's top finance posts catches names NOBODY put on a list yet (the next
# GME). min_faves keeps it to posts with real engagement. The targeted
# cashtag queries below still guarantee the theme anchors are covered.
DISCOVERY_QUERIES = [
    '(stocks OR "stock market" OR investing) min_faves:50 lang:en -is:retweet',
    '("short squeeze" OR "to the moon" OR tendies OR YOLO) min_faves:20 lang:en -is:retweet',
    '(calls OR puts) (stock OR earnings OR market) min_faves:20 lang:en -is:retweet',
    '(earnings OR guidance) (beat OR miss OR surged OR crashed) min_faves:20 lang:en -is:retweet',
    '(bitcoin OR ethereum OR crypto) (buy OR rally OR crash) min_faves:50 lang:en -is:retweet',
    '("bought shares" OR "loading up" OR "all in on") min_faves:10 lang:en -is:retweet',
    # sector channels - niche trades live in niche vocabulary, and the broad
    # queries above rarely surface them (lower min_faves: smaller communities)
    '(robotics OR humanoid OR bearings OR actuators) (stock OR stocks OR supplier OR trade) min_faves:10 lang:en -is:retweet',
    '("data center" OR datacenter OR "AI capex" OR GPUs) (stock OR stocks OR power OR demand) min_faves:20 lang:en -is:retweet',
    '(defense OR defence OR rearmament OR NATO) (stocks OR contractor OR budget) min_faves:10 lang:en -is:retweet',
    '(uranium OR copper OR "rare earth" OR lithium) (stocks OR miners OR price) min_faves:10 lang:en -is:retweet',
    '(Rheinmetall OR Nikkei OR "European stocks" OR Softbank OR TSMC) min_faves:10 lang:en -is:retweet',
    '(Fed OR inflation OR "rate cut" OR recession) (stocks OR market) min_faves:50 lang:en -is:retweet',
]
STATUS_ID = re.compile(r"/status/(\d+)")


def load_env():
    """Read keys from .env DIRECTLY (no python-dotenv), os.environ fallback.
    Accepts FETCHLAYER_KEY or FETCHLAYER_API_KEY (same key the Reddit
    fetcher uses)."""
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
            "X_BEARER_TOKEN": get("X_BEARER_TOKEN")}


def build_queries(chunk_size=6, lookback_days=None):
    """Cashtag queries chunked to stay well under any query-length limit.
    Smaller chunks = more queries (more credits) but far better coverage per
    symbol, because each request returns up to PAGE_SIZE tweets for its whole
    chunk. lookback_days adds since:<N days ago> so the Top product ranks the
    most popular posts of exactly that window."""
    core = {"GME", "AMC", "NVDA", "TSLA", "AAPL", "PLTR", "COIN", "MSTR", "SMCI"}
    symbols = sorted(set(THEME_ETFS.values()) | core)
    suffix = " lang:en -is:retweet"
    if lookback_days:
        since = (datetime.date.today()
                 - datetime.timedelta(days=lookback_days)).isoformat()
        suffix += f" since:{since}"
    queries, chunk = [], []
    for sym in symbols:
        chunk.append(f"${sym}")
        if len(chunk) == chunk_size:
            queries.append(f"({' OR '.join(chunk)}){suffix}")
            chunk = []
    if chunk:
        queries.append(f"({' OR '.join(chunk)}){suffix}")
    return queries


# ---------------- backend A: FetchLayer ----------------
def _fl_row(t):
    """Map ONE FetchLayer tweet object to our flat raw schema. FetchLayer's
    exact field names can shift, so every field is looked up defensively.
    Returns None if the tweet has no usable numeric status id."""
    url = t.get("url") or t.get("tweetUrl") or ""
    tweet_id = ""
    m = STATUS_ID.search(url) if isinstance(url, str) else None
    if m:
        tweet_id = m.group(1)
    else:
        for key in ("id", "tweetId", "id_str", "restId"):
            if str(t.get(key, "")).isdigit():
                tweet_id = str(t[key])
                break
    if not tweet_id:
        return None
    author = t.get("author") or {}
    handle = (author.get("handle") or author.get("username") or author.get("screenName")
              if isinstance(author, dict) else author) or ""
    created = (t.get("createdAt") or t.get("created_at") or t.get("date")
               or t.get("time") or "")
    if not created:                              # "Latest" => essentially now
        created = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Engagement lives either at the top level or inside a "counts" object.
    counts = t.get("counts") if isinstance(t.get("counts"), dict) else {}
    likes = (t.get("likeCount") or t.get("likes") or t.get("favoriteCount")
             or counts.get("likes") or counts.get("favorites")
             or counts.get("likeCount") or 0)
    return {
        "id": tweet_id,
        "created_at": created,
        "text": t.get("text") or t.get("fullText") or t.get("content") or "",
        "author": str(handle),
        "likes": likes,
    }


def fetchlayer_search(key, query, count):
    r = requests.post(FETCHLAYER_URL,
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"},
                      json={"query": query, "product": "Latest", "count": count},
                      timeout=30)
    return r


def fetchlayer_test(key):
    """ONE call, prints what came back, writes nothing (a few credits)."""
    query = "($NVDA OR $TSLA OR $GME) lang:en -is:retweet"
    print(f"POST twitter/search(product=Latest, count=5)\n  query: {query}")
    r = fetchlayer_search(key, query, 5)
    print(f"-> HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        return 1
    payload = r.json()
    results = (payload.get("results") or payload.get("tweets")
               or payload.get("data") or [])
    print(f"got {len(results)} tweets; sample fields: "
          f"{sorted(results[0].keys())[:12] if results else '-'}")
    for t in results[:5]:
        row = _fl_row(t)
        if row:
            print(f"  {str(row['likes']):>5} likes | @{row['author']:<15} | "
                  f"{row['text'][:60]}")
    print("TEST PASSED - FetchLayer X search works.")
    return 0


def fetchlayer_poll(key, max_tweets, max_credits=60, lookback_days=7):
    """TWO passes over the cashtag chunks (this is where the volume comes
    from - the signals need a WEEK of X chatter, not a trickle):
      1. product=Top     of the last 7 days - the most POPULAR tweets, the
                         main input for the 1-week trading lookback
      2. product=Latest  - the newest tweets, so nothing recent is missed
    Dedup on tweet id (here and at merge time) makes overlap harmless."""
    rows, used = [], 0
    per_chunk = min(PAGE_SIZE, max(10, max_tweets))
    since = (datetime.date.today()
             - datetime.timedelta(days=lookback_days)).isoformat()
    discovery = [q + f" since:{since}" for q in DISCOVERY_QUERIES]
    # DISCOVERY first (catch unknown names), then the targeted cashtag chunks.
    passes = [("Top", discovery + build_queries(lookback_days=lookback_days)),
              ("Latest", build_queries())]
    stopped = False
    for product, queries in passes:
        if stopped:
            break
        for query in queries:
            if len(rows) >= max_tweets:
                stopped = True
                break
            if used >= max_credits:
                print(f"[stop] hit the per-run credit cap ({max_credits})")
                stopped = True
                break
            # 429 = "too fast", NOT "out of credits" - back off and retry
            # instead of abandoning the whole run like before.
            r = None
            for wait in (0, 30, 90):
                if wait:
                    print(f"[rate] 429 - waiting {wait}s, then retrying")
                    time.sleep(wait)
                try:
                    r = requests.post(FETCHLAYER_URL,
                                      headers={"Authorization": f"Bearer {key}",
                                               "Content-Type": "application/json"},
                                      json={"query": query, "product": product,
                                            "count": per_chunk},
                                      timeout=(10, 60))
                except Exception as exc:
                    print(f"[warn] query failed: {exc}")
                    r = None
                    break
                if r.status_code != 429:
                    break
            if r is None:
                continue
            used += 1
            if r.status_code == 429:
                print("[stop] still rate-limited after two backoffs - ending "
                      "this run; next run continues where this left off")
                stopped = True
                break
            if r.status_code == 402:
                print("[stop] FetchLayer says 402 - out of credits; top up or "
                      "wait for the plan to reset")
                stopped = True
                break
            if r.status_code != 200:
                print(f"[warn] query ({product}) {r.status_code}: {r.text[:120]}")
                continue
            payload = r.json()
            results = (payload.get("results") or payload.get("tweets")
                       or payload.get("data") or [])
            got = 0
            for t in results:
                row = _fl_row(t)
                if row:
                    rows.append(row)
                    got += 1
            print(f"  {product:<6} | {got:>3} tweets | {query[:60]}")
            time.sleep(PAUSE_S)
    print(f"fetchlayer: {used} credits used this run")
    return rows[:max_tweets]


# ---------------- backend B: official v2 API ----------------
def official_poll(token, max_tweets):
    headers = {"Authorization": f"Bearer {token}"}
    rows = []
    for query in build_queries():
        if len(rows) >= max_tweets:
            break
        params = {
            "query": query,
            "max_results": min(PAGE_SIZE, max(10, max_tweets - len(rows))),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username",
        }
        r = requests.get(SEARCH_URL, headers=headers, params=params, timeout=20)
        if r.status_code == 429:
            print("[stop] rate limited (429) - ending this run; next run catches up.")
            break
        if r.status_code != 200:
            print(f"[warn] query failed ({r.status_code}): {r.text[:120]}")
            continue
        payload = r.json()
        users = {u["id"]: u.get("username", "")
                 for u in payload.get("includes", {}).get("users", [])}
        for t in payload.get("data", []):
            rows.append({
                "id": t["id"],
                "created_at": t.get("created_at", ""),
                "text": t.get("text", ""),
                "author": users.get(t.get("author_id"), ""),
                "likes": (t.get("public_metrics") or {}).get("like_count", 0),
            })
        time.sleep(PAUSE_S)
    return rows


# ---------------- shared: raw append ----------------
def append_to_raw(rows):
    """Append new tweets to the registered raw file, deduping on id."""
    new = pd.DataFrame(rows)
    if os.path.exists(OUT_FILE):
        old_bytes = zstandard.ZstdDecompressor().decompress(open(OUT_FILE, "rb").read())
        old = pd.read_csv(io.BytesIO(old_bytes), dtype={"id": str})
        new = pd.concat([old, new.astype({"id": str})], ignore_index=True)
    new = new.drop_duplicates(subset="id", keep="first")
    buf = new.to_csv(index=False).encode("utf-8")
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "wb") as f:
        f.write(zstandard.ZstdCompressor(level=10).compress(buf))
    print(f"raw file now holds {len(new):,} unique tweets -> {OUT_FILE}")


def main():
    p = argparse.ArgumentParser(description="Live X ingestion (FetchLayer or official v2 API)")
    p.add_argument("--test", action="store_true",
                   help="ONE small call, prints the tweets, writes nothing")
    p.add_argument("--max-tweets", type=int, default=3000,
                   help="budget cap per run (Top-of-week + Latest passes)")
    p.add_argument("--max-credits", type=int, default=60,
                   help="FetchLayer credit cap per run (1 credit per request)")
    p.add_argument("--lookback-days", type=int, default=7,
                   help="Top-product window: how far back the fetch reaches")
    args = p.parse_args()

    env = load_env()
    if env["FETCHLAYER_API_KEY"]:
        backend = "fetchlayer"
    elif env["X_BEARER_TOKEN"]:
        backend = "official"
    else:
        print("X live ingestion is OFF: no FETCHLAYER_KEY or X_BEARER_TOKEN in .env.")
        print("  Easiest: add  FETCHLAYER_KEY = <your key from fetchlayer.dev>")
        print("           (the same key the Reddit fetcher uses - no paid X account needed)")
        print("  Or pay for X API access and set  X_BEARER_TOKEN = <bearer token>")
        return 0
    print(f"backend: {backend}")

    if args.test:
        if backend == "fetchlayer":
            return fetchlayer_test(env["FETCHLAYER_API_KEY"])
        rows = official_poll(env["X_BEARER_TOKEN"], max_tweets=5)
        for row in rows[:5]:
            print(f"  {str(row['likes']):>5} likes | @{row['author']} | {row['text'][:60]}")
        print("TEST PASSED" if rows else "TEST FAILED - see messages above")
        return 0 if rows else 1

    rows = (fetchlayer_poll(env["FETCHLAYER_API_KEY"], args.max_tweets,
                            args.max_credits, args.lookback_days)
            if backend == "fetchlayer"
            else official_poll(env["X_BEARER_TOKEN"], args.max_tweets))
    if not rows:
        print("no tweets fetched this run")
        return 0
    append_to_raw(rows)
    print("raw accumulates here; merge into posts.parquet is the next pipeline "
          "step:  python ingestion/merge_live.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
