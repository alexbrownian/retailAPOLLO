"""
screen_tickers.py
=================
Decide which tickers are "word-tickers" (EDGE, LOAN, RENT, ...) that should
only be counted when written as a cashtag ($EDGE), using two signals:

  Signal 1 - CASE RATIO (measured on OUR OWN corpus):
      English words appear mostly lowercase in real posts ("the edge of"),
      real tickers appear mostly in ALL CAPS ("bought NVDA calls").
          caps_share = caps_count / (caps_count + lower_count)
      Low caps_share -> it's a word. Measured on this dataset:
      EDGE ~ 0.02, LOAN ~ 0.002, NVDA ~ 0.93, TSLA ~ 0.93.

  Signal 2 - WORDFREQ (general-English frequency, fallback only):
      zipf_frequency("edge", "en") ~ 4.7  -> common English word
      zipf_frequency("nvda", "en") ~ 2.1  -> not a word
      Used only when the corpus doesn't contain the token often enough
      for the case ratio to be trustworthy (< MIN_SIGHTINGS).

  Corpus evidence WINS over wordfreq when both exist, because it reflects
  how Reddit actually writes. Example: AMD scores 3.5 in wordfreq (looks
  like a word) but the corpus shows it caps-heavy like a ticker; SNAP
  scores 4.2 but measures caps_share ~ 0.56 -> kept as a ticker.

Output: data/reference/ticker_classification.csv
  (ticker, caps_count, lower_count, caps_share, zipf, decided_by,
   classification). extract_tickers.py loads it at import and demotes
  every 'cashtag_only' ticker automatically: bare-caps mentions stop
  counting, $CASHTAG mentions still count ("demote, don't delete").

Run standalone (samples the corpus itself, ~1 min):
    python -m src.screen_tickers
or from a notebook with posts already loaded: see notebook 01, section
"Screen word-tickers".

Known limitations (see README "Screening word-tickers"):
  - Caps-typed jargon (HODL) passes the case test; the manual
    BARE_PROSE_STOP list in extract_tickers.py still covers those.
  - Brand-name tickers people type lowercase (SOFI, HOOD, COIN) get
    demoted -> lower recall, higher precision. Their cashtags still count.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from wordfreq import zipf_frequency

logger = logging.getLogger(__name__)

# ---- thresholds (tune here; see README "Screening word-tickers") ---------
MIN_SIGHTINGS = 30          # need >= this many sightings to trust the ratio
CAPS_SHARE_THRESHOLD = 0.5  # caps_share below this = mostly a word
ZIPF_WORD_THRESHOLD = 3.5   # zipf >= this = common English word (fallback)

# Only 4-5 letter words can collide with the bare-caps ticker regex in
# extract_tickers.py (WORD_BARE), so that's all we need to screen.
WORD_RE = re.compile(r"\b[A-Za-z]{4,5}\b")

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "reference" / "ticker_classification.csv"
DEFAULT_POSTS = Path(__file__).resolve().parent.parent / "data" / "processed" / "posts.parquet"


def count_caps_vs_lower(texts, candidates):
    """One pass over all texts. For each candidate ticker count how often it
    appears as strict ALL CAPS ("EDGE") vs any other casing ("edge", "Edge").
    "Edge" at a sentence start counts as word evidence - that's deliberate."""
    caps = {t: 0 for t in candidates}
    lower = {t: 0 for t in candidates}
    for text in texts:
        if not isinstance(text, str):
            continue
        for word in WORD_RE.findall(text):
            if word.isupper():
                if word in caps:
                    caps[word] += 1
            else:
                key = word.upper()
                if key in caps:
                    lower[key] += 1
    return caps, lower


def classify_ticker(caps_count: int, lower_count: int, zipf: float):
    """Classify ONE ticker. Returns (classification, decided_by).

    classification: 'normal'      -> count bare caps AND cashtags
                    'cashtag_only' -> count only $CASHTAG mentions
    decided_by:     'case_ratio' (corpus evidence) or 'wordfreq' (fallback)
    """
    total = caps_count + lower_count
    if total >= MIN_SIGHTINGS:
        caps_share = caps_count / total
        if caps_share < CAPS_SHARE_THRESHOLD:
            return "cashtag_only", "case_ratio"
        return "normal", "case_ratio"
    # Too rare in our corpus to judge -> fall back to general English.
    if zipf >= ZIPF_WORD_THRESHOLD:
        return "cashtag_only", "wordfreq"
    return "normal", "wordfreq"


def screen_tickers(texts, candidates) -> pd.DataFrame:
    """Build the classification table for every candidate ticker.

    texts      : iterable of post texts (title + selftext)
    candidates : set of ticker symbols to screen (4-5 letters, uppercase)
    """
    caps, lower = count_caps_vs_lower(texts, candidates)
    rows = []
    for t in sorted(candidates):
        z = zipf_frequency(t.lower(), "en")
        cls, why = classify_ticker(caps[t], lower[t], z)
        total = caps[t] + lower[t]
        rows.append({
            "ticker": t,
            "caps_count": caps[t],
            "lower_count": lower[t],
            "caps_share": round(caps[t] / total, 3) if total else None,
            "zipf": z,
            "decided_by": why,
            "classification": cls,
        })
    return pd.DataFrame(rows)


def sample_texts_from_parquet(posts_path, sample_size: int = 300_000, seed: int = 0):
    """Stream title+selftext in batches and keep a random ~sample_size of
    them. Streaming means we never hold all 7.9M posts in memory, and the
    sample covers every subreddit block of the file."""
    pf = pq.ParquetFile(posts_path)
    frac = min(1.0, sample_size / pf.metadata.num_rows)
    rng = random.Random(seed)
    texts = []
    for batch in pf.iter_batches(columns=["title", "selftext"], batch_size=65536):
        for t, b in zip(batch.column("title").to_pylist(),
                        batch.column("selftext").to_pylist()):
            if rng.random() < frac:
                texts.append((t or "") + " " + (b or ""))
    return texts


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Classify word-tickers (case ratio + wordfreq)")
    p.add_argument("--posts", type=Path, default=DEFAULT_POSTS, help="posts.parquet path")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output CSV path")
    p.add_argument("--sample-size", type=int, default=300_000,
                   help="how many posts to sample for the case-ratio count")
    p.add_argument("--universe-cache", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data" / "reference",
                   help="dir with cached nasdaqlisted.txt / otherlisted.txt")
    args = p.parse_args(argv)

    # Import here so the module works without ticker_universe's deps.
    from .ticker_universe import load_us_ticker_universe

    universe = load_us_ticker_universe(args.universe_cache, max_cache_age_days=365)
    candidates = {t for t in universe if 4 <= len(t) <= 5}
    logger.info("screening %s candidate tickers (4-5 letters)", len(candidates))

    texts = sample_texts_from_parquet(args.posts, sample_size=args.sample_size)
    logger.info("sampled %s posts", len(texts))

    df = screen_tickers(texts, candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    demoted = (df["classification"] == "cashtag_only").sum()
    logger.info("wrote %s -> %s tickers demoted to cashtag-only, %s kept normal",
                args.out, demoted, len(df) - demoted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
