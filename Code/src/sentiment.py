"""Post-level sentiment scoring, rolled up per ticker and per theme per day.

Scoring is lexicon-based (VADER, optionally layered with FinVADER's
financial dictionaries, plus a hand-set WSB/finance slang lexicon):
cheap, offline and explainable.

How it works:

1. VADER scores each post's text with a 'compound' score in [-1, +1].
   Every word carries a valence, negation flips it ("not good" < 0),
   intensifiers scale it ("very good" > "good").
2. Plain VADER misreads finance: "calls"/"puts" are neutral to it,
   "short" reads as generically negative, "moon"/"tendies" mean nothing.
   WSB_LEXICON injects finance/WSB slang with hand-set valences (VADER's
   scale runs about -4..+4; ~2 is clearly positive).
3. Per day and entity (ticker or theme) the post scores aggregate into::

       n_posts        - how many scored posts mentioned it
       avg_sentiment  - mean compound score, [-1, +1]
       net_bullish    - (share of bullish posts) - (share of bearish posts),
                        also [-1, +1]. A post is bullish if compound > +0.05,
                        bearish if < -0.05 (VADER's conventional cutoffs).

   net_bullish is the headline metric: it is robust to one extreme post
   dragging the mean and reads naturally ("+0.3 = 30 points more bulls
   than bears"). avg_sentiment is kept for comparison.

Noise: a ticker with 5 posts/day swings wildly, so readers mask thin days
and use rolling means. Sarcasm, loss-porn irony and "puts printing"
defeat any lexicon; treat levels as noisy and read changes against a
name's own baseline. Theme-level lines aggregate hundreds of posts per
day and are meaningfully more stable.

Key functions: ``score_text()`` scores one string; ``add_sentiment()``
is the single-core reference; ``add_sentiment_fast()`` scores in
parallel across CPU cores (VADER is embarrassingly parallel) and, with
the 300-char truncation, reaches several thousand posts per second per
core; ``add_sentiment_cached()`` adds the permanent per-engine score
store so every post is scored exactly once. Long selftexts are truncated
(TRUNCATE_CHARS) because sentiment saturates within the first few hundred
characters; the tail adds cost, not signal.
``build_daily_ticker_sentiment()`` and ``build_daily_theme_sentiment()``
produce the daily aggregate frames.
"""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.config import DATA_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# WSB / finance slang. VADER valence scale is roughly -4 .. +4. Single
# tokens only (VADER's lexicon is token-based; phrases do not match).
# Changing this changes the engine's output; the score store is keyed by
# engine name, not lexicon contents, so rescore after editing.
# ---------------------------------------------------------------------------
WSB_LEXICON: dict[str, float] = {
    # bullish
    "moon": 2.5, "mooning": 2.5, "rocket": 1.5, "tendies": 2.0,
    "calls": 1.5, "call": 0.8, "long": 1.0, "bull": 2.0, "bullish": 2.5,
    "buy": 1.0, "btfd": 2.0, "hodl": 1.5, "hold": 0.5, "stonks": 1.5,
    "lambo": 2.0, "printing": 1.5, "breakout": 1.5, "undervalued": 1.5,
    "squeeze": 1.0, "rip": 1.0, "ripping": 2.0, "pump": 0.5,
    # bearish
    "puts": -1.5, "put": -0.8, "short": -1.0, "bear": -2.0, "bearish": -2.5,
    "sell": -1.0, "dump": -2.0, "dumping": -2.5, "crash": -2.5, "tank": -2.0,
    "tanking": -2.5, "drill": -2.0, "drilling": -2.0, "bagholder": -2.5,
    "bagholders": -2.5, "bags": -1.5, "rug": -2.5, "rugged": -3.0,
    "rekt": -2.5, "overvalued": -1.5, "bubble": -1.5, "scam": -3.0,
    "worthless": -3.0, "delisted": -2.5, "bankrupt": -3.0, "bankruptcy": -3.0,
}

TRUNCATE_CHARS = 300    # Text beyond this adds cost, not signal.
BULL_CUTOFF = 0.05      # VADER convention: compound > +0.05 = positive post.
BEAR_CUTOFF = -0.05     #                   compound < -0.05 = negative post.

_ANALYZER: SentimentIntensityAnalyzer | None = None
SENTIMENT_ENGINE = "vader+wsb"      # Reported by get_analyzer(); becomes
                                    # "finvader+wsb" when finvader is installed.


def _finvader_lexicons() -> dict:
    """Returns FinVADER's two financial dictionaries, or {} if unavailable.

    SentiBignomics (~7.3k terms) and Henry (~190 terms) fix plain VADER's
    blind spots on finance language ('impairment', 'covenant breach',
    'guidance raised'...). An empty result degrades the engine gracefully
    to classic VADER."""
    try:
        # Importing finvader triggers nltk.download("vader_lexicon"), which
        # contacts the NLTK server to compare versions every time: one
        # network round-trip per worker process, the main startup cost of
        # a parallel scoring run. The lexicon file is already on disk, so
        # the downloader is temporarily no-op'd while the package
        # initialises.
        import nltk
        real_download = nltk.download
        nltk.download = lambda *args, **kwargs: True
        try:
            from finvader.SentiBignomics import lexicon1
            from finvader.Henry import lexicon2
            return {**lexicon1(), **lexicon2()}
        finally:
            nltk.download = real_download
    except Exception:
        return {}


def get_analyzer() -> SentimentIntensityAnalyzer:
    """Returns the process-wide analyzer, built on first use.

    Lexicon layering (later wins): VADER base -> FinVADER financial
    dictionaries -> WSB slang. The slang goes last so meme vocabulary
    keeps its hand-set valences. Sets SENTIMENT_ENGINE to the layering
    actually achieved.
    """
    global _ANALYZER, SENTIMENT_ENGINE
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
        fin = _finvader_lexicons()
        if fin:
            _ANALYZER.lexicon.update(fin)
            SENTIMENT_ENGINE = "finvader+wsb"
        _ANALYZER.lexicon.update(WSB_LEXICON)
        # Only the main process announces the engine; the parallel workers
        # each build their own analyzer too and would otherwise print this
        # line once per core.
        import multiprocessing
        if multiprocessing.parent_process() is None:
            print(f"sentiment engine: {SENTIMENT_ENGINE} "
                  f"({len(_ANALYZER.lexicon):,} lexicon terms)")
    return _ANALYZER


def score_text(text: str) -> float:
    """Returns the compound sentiment of one text, in [-1, +1]; 0.0 for
    empty or non-string input."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return get_analyzer().polarity_scores(text[:TRUNCATE_CHARS])["compound"]


def add_sentiment(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of posts_df with a 'sentiment' column scored from
    title + selftext on a single core. The reference implementation;
    add_sentiment_fast() gives the same output in parallel."""
    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str).str.slice(0, TRUNCATE_CHARS)
    out = posts_df.copy()
    out["sentiment"] = [score_text(t + " " + b) for t, b in zip(titles, bodies)]
    return out


def _score_batch(texts: list) -> list:
    """Scores one chunk of texts. Module-level so joblib workers can import
    it (each worker builds its own analyzer on first call)."""
    analyzer = get_analyzer()
    out = []
    for t in texts:
        if isinstance(t, str) and t.strip():
            out.append(analyzer.polarity_scores(t[:TRUNCATE_CHARS])["compound"])
        else:
            out.append(0.0)
    return out


def add_sentiment_fast(posts_df: pd.DataFrame, n_jobs: int = -1,
                       chunk_size: int = 20_000) -> pd.DataFrame:
    """Parallel version of add_sentiment().

    Splits the posts into chunks and scores them across CPU cores via
    joblib. Same output as add_sentiment(), roughly n_cores times faster.

    Args:
        posts_df: Posts frame with title and selftext columns.
        n_jobs: joblib worker count; -1 uses every core.
        chunk_size: Posts per worker task.

    Returns:
        A copy of posts_df with a 'sentiment' column.
    """
    from joblib import Parallel, delayed

    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str).str.slice(0, TRUNCATE_CHARS)
    texts = (titles + " " + bodies).tolist()
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(_score_batch)(c) for c in chunks)

    out = posts_df.copy()
    out["sentiment"] = [s for part in results for s in part]
    return out


# ---------------------------------------------------------------------------
# Permanent score store: every post is scored exactly once per engine.
# The store (id, sentiment) appends forever; any rebuild or live fold
# looks scores up by post id and only scores the ids it has never seen. A
# post's score never changes for a given engine, so the lookup is always
# correct, and because the store is keyed by the engine name in its
# filename, switching engines rescores automatically.
# ---------------------------------------------------------------------------
def get_engine_name() -> str:
    """Returns the active engine name without building the analyzer.

    find_spec only checks whether the finvader package exists on disk; it
    does not run finvader's __init__ (which would trigger an nltk network
    check)."""
    import importlib.util
    if importlib.util.find_spec("finvader") is not None:
        return "finvader+wsb"
    return "vader+wsb"


def _store_path() -> str:
    """Returns the score store path for the active engine."""
    import os
    safe = get_engine_name().replace("+", "_")
    return os.path.join(DATA_DIR, "processed", f"sentiment_scores_{safe}.parquet")


def add_sentiment_cached(posts_df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """add_sentiment_fast() backed by the permanent score store.

    Known ids are looked up; unknown ids are scored in parallel and
    appended to the store atomically. After the first full build,
    historical rebuilds cost a lookup, not a rescore. A frame without an
    'id' column is scored directly and nothing is stored.

    Args:
        posts_df: Posts frame with title and selftext columns, and
            normally an id column.
        n_jobs: joblib worker count; -1 uses every core.

    Returns:
        A copy of posts_df with a 'sentiment' column.
    """
    import os

    if "id" not in posts_df.columns:
        # No ids: nothing to key the store on, so score directly.
        return add_sentiment_fast(posts_df, n_jobs=n_jobs)

    path = _store_path()
    known = pd.DataFrame(columns=["id", "sentiment"])
    if os.path.exists(path):
        known = pd.read_parquet(path)

    ids = posts_df["id"].astype(str)
    known_map = known.set_index("id")["sentiment"] if len(known) else pd.Series(dtype=float)
    hit = ids.isin(known_map.index)
    n_new = int((~hit).sum())
    print(f"sentiment store [{get_engine_name()}]: {int(hit.sum()):,} cached, "
          f"{n_new:,} new to score")

    out = posts_df.copy()
    out["sentiment"] = ids.map(known_map).values
    if n_new:
        fresh = add_sentiment_fast(posts_df.loc[~hit.values], n_jobs=n_jobs)
        out.loc[~hit.values, "sentiment"] = fresh["sentiment"].values
        addition = pd.DataFrame({"id": ids[~hit.values].values,
                                 "sentiment": fresh["sentiment"].values})
        combined = (pd.concat([known, addition], ignore_index=True)
                    .drop_duplicates(subset="id", keep="first"))
        tmp = path + ".tmp"
        combined.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        print(f"sentiment store now holds {len(combined):,} scored posts")
    return out


def _aggregate(rows: list[tuple]) -> pd.DataFrame:
    """Turns rows of (date, entity, sentiment) into daily per-entity
    aggregates (n_posts, avg_sentiment, net_bullish)."""
    if not rows:
        return pd.DataFrame(columns=["date", "entity", "n_posts",
                                     "avg_sentiment", "net_bullish"])
    df = pd.DataFrame(rows, columns=["date", "entity", "sentiment"])
    df["is_bull"] = (df["sentiment"] > BULL_CUTOFF).astype(int)
    df["is_bear"] = (df["sentiment"] < BEAR_CUTOFF).astype(int)
    daily = (
        df.groupby(["date", "entity"])
        .agg(n_posts=("sentiment", "size"),
             avg_sentiment=("sentiment", "mean"),
             bull=("is_bull", "sum"),
             bear=("is_bear", "sum"))
        .reset_index()
    )
    daily["net_bullish"] = (daily["bull"] - daily["bear"]) / daily["n_posts"]
    return daily.drop(columns=["bull", "bear"])


def build_daily_ticker_sentiment(posts_with_sent: pd.DataFrame,
                                 universe: set,
                                 cashtags_only: bool = False) -> pd.DataFrame:
    """Builds daily sentiment per ticker.

    Reuses the same extractor as the mention counts, so screening and
    stop lists apply identically. Each post casts one vote per ticker it
    mentions.

    Args:
        posts_with_sent: Posts frame with a 'sentiment' column.
        universe: Valid ticker symbols.
        cashtags_only: Count only $-prefixed mentions.

    Returns:
        DataFrame(date, ticker, n_posts, avg_sentiment, net_bullish).
    """
    from .extract_tickers import extract_tickers_from_text

    rows = []
    titles = posts_with_sent["title"].fillna("").astype(str)
    bodies = posts_with_sent["selftext"].fillna("").astype(str)
    for date, title, body, sent in zip(posts_with_sent["date"].astype(str),
                                       titles, bodies,
                                       posts_with_sent["sentiment"]):
        tickers = set(extract_tickers_from_text(title + " " + body, universe,
                                                cashtags_only=cashtags_only))
        for ticker in tickers:          # One vote per post per ticker.
            rows.append((date, ticker, sent))
    return _aggregate(rows).rename(columns={"entity": "ticker"})


def build_daily_theme_sentiment(posts_with_sent: pd.DataFrame) -> pd.DataFrame:
    """Builds daily sentiment per theme (keyword matching from src/themes.py).

    Args:
        posts_with_sent: Posts frame with a 'sentiment' column.

    Returns:
        DataFrame(date, theme, n_posts, avg_sentiment, net_bullish).
    """
    from .themes import themes_in_text

    rows = []
    titles = posts_with_sent["title"].fillna("").astype(str)
    bodies = posts_with_sent["selftext"].fillna("").astype(str)
    for date, title, body, sent in zip(posts_with_sent["date"].astype(str),
                                       titles, bodies,
                                       posts_with_sent["sentiment"]):
        for theme in themes_in_text(title + " " + body):
            rows.append((date, theme, sent))
    return _aggregate(rows).rename(columns={"entity": "theme"})
