"""Smallest possible check that a FetchLayer key pulls Reddit posts.

Has no project dependencies (only ``requests``), reads ``.env`` by hand,
makes exactly one call (one credit) and writes nothing::

    python Code/ingestion/test_fetchlayer.py

Every failure mode (missing key, rejected key, no credits, unreachable
host, changed endpoint) prints a plain-English diagnosis and returns a
non-zero exit code.
"""

import json
import os
import sys

import requests

try:                     # post titles contain emoji; don't die on cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)          # .env lives at the project root
URL = "https://fetchlayer.dev/api/reddit/community-posts"


def read_key():
    """Return the FetchLayer key from ``.env``, or ``None`` with a diagnosis."""
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        print("FAIL: no .env file at", env_path)
        return None
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in ("FETCHLAYER_KEY", "FETCHLAYER_API_KEY") and v.strip():
            return v.strip()
    print("FAIL: .env exists but has no filled FETCHLAYER_KEY line")
    return None


def main():
    """Make the one test call and print what came back.

    Returns:
        ``0`` on success, ``1`` on any failure.
    """
    key = read_key()
    if not key:
        return 1
    print(f"key found in .env (starts '{key[:5]}...', {len(key)} chars)")
    print("calling FetchLayer: 5 newest posts from r/wallstreetbets (1 credit)...")

    try:
        r = requests.post(URL,
                          headers={"Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"},
                          json={"subreddit": "wallstreetbets", "sort": "new", "limit": 5},
                          timeout=30)
    except requests.exceptions.ConnectionError as e:
        print("FAIL: could not reach fetchlayer.dev at all - network/proxy/"
              "firewall issue on this machine:", str(e)[:150])
        return 1
    except requests.exceptions.Timeout:
        print("FAIL: fetchlayer.dev timed out - try again / check network")
        return 1

    print("HTTP status:", r.status_code)
    if r.status_code == 401 or r.status_code == 403:
        print("FAIL: key REJECTED - copy it again from your fetchlayer.dev "
              "dashboard (no quotes, no spaces) into .env")
        print("server said:", r.text[:200])
        return 1
    if r.status_code == 402:
        print("FAIL: out of credits - top up / subscribe at fetchlayer.dev")
        return 1
    if r.status_code == 404:
        print("FAIL: endpoint not found - FetchLayer may have changed paths;")
        print("check https://fetchlayer.dev/reddit-scraper#endpoints")
        print("server said:", r.text[:200])
        return 1
    if r.status_code != 200:
        print("FAIL: unexpected response:", r.text[:300])
        return 1

    payload = r.json()
    # FetchLayer returns the posts under "items" (seen in a real response);
    # keep the other names as fallbacks in case the API changes.
    posts = (payload.get("items") or payload.get("posts")
             or payload.get("results") or payload.get("data") or [])
    if not posts and isinstance(payload, list):
        posts = payload
    print(f"\nSUCCESS - {len(posts)} posts returned. Fields in a post:")
    if posts:
        print(" ", sorted(posts[0].keys()))
        print("\nnewest r/wallstreetbets posts:")
        for p in posts[:5]:
            title = p.get("title") or p.get("postTitle") or p.get("text") or ""
            score = p.get("score", p.get("upvotes", "?"))
            print(f"  {str(score):>6} pts | {str(title)[:75]}")
    else:
        print("  (200 OK but no posts array recognised - raw response below)")
        print(json.dumps(payload, indent=1)[:600])
    print("\nYour pipeline can pull from FetchLayer. Next:")
    print("  python Code/ingestion/fetch_all.py        (check keys + fetch everything)")
    print("  python Code/ingestion/check_live_ingestion.py       (see it registered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
