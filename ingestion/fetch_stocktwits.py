"""Append the latest StockTwits messages for every tracked symbol to raw.

Pulls the ~30 newest messages per symbol from the public symbol streams
and appends them raw to
``data/raw/StockTwits/stocktwits_YYYY-MM-DD.jsonl.zst`` (one JSON message
per line, zstd-compressed; the same immutable-raw approach as the Reddit
dumps). Run it on a schedule; duplicates across runs are fine, because the
normaliser dedupes on message id ("first seen wins").

No API key is needed for these read-only streams, but the script stays
polite: the unauthenticated limit is about 200 requests per hour per IP,
so with the default symbol list (~40) an hourly run uses ~40 requests.
Do not run it more than ~4x per hour with a big list. A 429 response ends
the run early; the next scheduled run picks up the rest.

The author's own Bullish/Bearish label is preserved in the raw lines; it
is the ground truth for calibrating the rule-based sentiment score.

Usage::

    python ingestion/fetch_stocktwits.py
    python ingestion/fetch_stocktwits.py --symbols GME,NVDA,GLD   # override the list
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

from src.themes import THEME_ETFS  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "StockTwits")
URL = "https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
PAUSE_S = 1.5   # polite gap between requests


def default_symbols():
    """Return the theme anchor ETFs plus a core set of retail-heavy names."""
    core = {"GME", "AMC", "NVDA", "TSLA", "AAPL", "PLTR", "COIN", "MSTR", "SMCI"}
    return sorted(set(THEME_ETFS.values()) | core)


def main():
    """Fetch every symbol's stream and append the messages to today's file."""
    p = argparse.ArgumentParser(description="Append raw StockTwits messages")
    p.add_argument("--symbols", default=None,
                   help="comma-separated override, e.g. GME,NVDA,GLD")
    args = p.parse_args()
    symbols = (args.symbols.split(",") if args.symbols else default_symbols())

    os.makedirs(OUT_DIR, exist_ok=True)
    today = datetime.date.today().isoformat()
    out_path = os.path.join(OUT_DIR, f"stocktwits_{today}.jsonl.zst")

    lines, fetched, skipped = [], 0, 0
    for sym in symbols:
        try:
            r = requests.get(URL.format(sym=sym), headers=HEADERS, timeout=15)
        except Exception as exc:
            print(f"[warn] {sym}: {exc}")
            continue
        if r.status_code == 429:
            print("[stop] rate limited (429) - ending this run early; "
                  "the next scheduled run picks up the rest.")
            break
        if r.status_code != 200:
            skipped += 1
            continue
        msgs = (r.json() or {}).get("messages", [])
        for m in msgs:
            lines.append(json.dumps(m, ensure_ascii=False))
        fetched += len(msgs)
        time.sleep(PAUSE_S)

    if not lines:
        print("nothing fetched"); return

    # Append-compress: read existing day file (if any), add the new lines.
    old = b""
    if os.path.exists(out_path):
        old = zstandard.ZstdDecompressor().decompress(open(out_path, "rb").read())
    blob = old + ("\n".join(lines) + "\n").encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(zstandard.ZstdCompressor(level=10).compress(blob))

    print(f"fetched {fetched} messages from {len(symbols) - skipped} symbols "
          f"-> {out_path}")
    print("dedup happens at merge time (normalise_stocktwits, first seen wins).")


if __name__ == "__main__":
    main()
