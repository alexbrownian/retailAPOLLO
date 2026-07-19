"""
sentiment.py
============
Post-level sentiment scoring (VADER + a WSB/finance lexicon), rolled up
per ticker and per theme per day. This is "Stage 2 lite": cheap, offline,
explainable - built to test whether direction-aware signals add anything
before investing in a finance-tuned transformer (FinBERT/FinTwitBERT).

HOW IT WORKS
  1. VADER scores each post's text with a 'compound' score in [-1, +1].
     VADER is a lexicon: every word carries a valence, negation flips it
     ("not good" < 0), intensifiers scale it ("very good" > "good").
  2. Plain VADER misspeaks finance: "calls"/"puts" are neutral to it,
     "short" reads as generically negative, "moon"/"tendies" mean nothing.
     WSB_LEXICON below injects finance/WSB slang with hand-set valences
     (VADER's scale runs about -4..+4; ~2 = clearly positive).
  3. Per day and entity (ticker or theme) we aggregate post scores into:
        n_posts        - how many scored posts mentioned it
        avg_sentiment  - mean compound score, [-1, +1]
        net_bullish    - (share of bullish posts) - (share of bearish posts),
                         also [-1, +1]. A post is bullish if compound > +0.05,
                         bearish if < -0.05 (VADER's conventional cutoffs).
     net_bullish is the headline metric: it is robust to one extreme post
     dragging the mean, and reads naturally ("+0.3 = 30 points more bulls
     than bears"). avg_sentiment is kept for comparison.

NOISE WARNING (read before trusting a chart)
  - A ticker with 5 posts/day swings wildly: mask days below MIN_POSTS
    and use 7-day rolling means in the notebooks.
  - Sarcasm, loss-porn irony and "puts printing" defeat any lexicon.
    Treat levels as noisy; CHANGES vs a ticker's own baseline are the
    more trustworthy read. Theme-level lines aggregate hundreds of posts
    per day and are meaningfully more stable (that is why JPM's published
    chart is sector-level, not single-stock).

Speed: use add_sentiment_fast() - it scores in parallel across CPU cores
(VADER is embarrassingly parallel) and, with the 300-char truncation,
reaches roughly 4-5k posts/sec PER CORE. On a typical 8-thread machine a
1-year slice (~2.8M posts) lands around the 2-minute mark; the notebooks
additionally cap the number of scored posts (seeded sample) to guarantee
it. Long selftexts are truncated (TRUNCATE_CHARS=300) - sentiment
saturates within the first few hundred characters; the tail adds cost,
not signal. add_sentiment() is the simple single-core reference.
"""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# WSB / finance slang. VADER valence scale is roughly -4 .. +4.
# Single TOKENS only (VADER's lexicon is token-based; phrases won't match).
# Extend freely - then re-run the scoring notebook.
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

TRUNCATE_CHARS = 300    # text beyond this adds cost, not signal (~4.5x faster than 1000)
BULL_CUTOFF = 0.05      # VADER convention: compound > +0.05 = positive post
BEAR_CUTOFF = -0.05     #                   compound < -0.05 = negative post

_ANALYZER: SentimentIntensityAnalyzer | None = None
SENTIMENT_ENGINE = "vader+wsb"      # reported by get_analyzer(); becomes
                                    # "finvader+wsb" when finvader is installed


def _finvader_lexicons() -> dict:
    """FinVADER's two financial dictionaries (SentiBignomics ~7.3k terms,
    Henry ~190 terms), if the finvader package is installed. They fix plain
    VADER's blind spots on finance language ('impairment', 'covenant
    breach', 'guidance raised'...). Returns {} when unavailable, so the
    engine degrades gracefully to classic VADER."""
    try:
        # importing finvader triggers nltk.download("vader_lexicon"), which
        # phones the NLTK server to compare versions EVERY time - one network
        # round-trip per worker process, the main startup cost of a parallel
        # scoring run. The lexicon file is already on disk, so temporarily
        # no-op the downloader while the package initialises.
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
    """One analyzer, built on first use. Lexicon layering (later wins):
    VADER base -> FinVADER financial dictionaries -> WSB slang. The slang
    goes last so meme vocabulary keeps its hand-set valences."""
    global _ANALYZER, SENTIMENT_ENGINE
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
        fin = _finvader_lexicons()
        if fin:
            _ANALYZER.lexicon.update(fin)
            SENTIMENT_ENGINE = "finvader+wsb"
        _ANALYZER.lexicon.update(WSB_LEXICON)
        # only the MAIN process announces the engine - the parallel workers
        # each build their own analyzer too, which used to print this line
        # once per core and flood the log
        import multiprocessing
        if multiprocessing.parent_process() is None:
            print(f"sentiment engine: {SENTIMENT_ENGINE} "
                  f"({len(_ANALYZER.lexicon):,} lexicon terms)")
    return _ANALYZER


def score_text(text: str) -> float:
    """Compound sentiment of one text, in [-1, +1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return get_analyzer().polarity_scores(text[:TRUNCATE_CHARS])["compound"]


def add_sentiment(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of posts_df with a 'sentiment' column scored from
    title + selftext. THE slow step - run once, save, iterate on the file."""
    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str).str.slice(0, TRUNCATE_CHARS)
    out = posts_df.copy()
    out["sentiment"] = [score_text(t + " " + b) for t, b in zip(titles, bodies)]
    return out


def _score_batch(texts: list) -> list:
    """Score one chunk of texts. Module-level so joblib workers can import
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
    """Parallel version of add_sentiment(): splits the posts into chunks and
    scores them on every CPU core via joblib (n_jobs=-1 = all cores).
    Same output, ~n_cores times faster."""
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
# PERMANENT SCORE STORE - every post is scored exactly ONCE per engine.
# The store (id, sentiment) appends forever; any rebuild or live fold looks
# scores up by post id and only scores the ids it has never seen. A post's
# score never changes for a given engine, so this is always correct - and it
# removes the stale-cache class of bugs entirely (the store is keyed by the
# ENGINE NAME in its filename, so switching engines rescoring automatically).
# ---------------------------------------------------------------------------
def get_engine_name() -> str:
    """The active engine, decided WITHOUT building the analyzer.
    find_spec only checks whether the package EXISTS on disk - it does not
    run finvader's __init__ (which would trigger an nltk network check)."""
    import importlib.util
    if importlib.util.find_spec("finvader") is not None:
        return "finvader+wsb"
    return "vader+wsb"


def _store_path() -> str:
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safe = get_engine_name().replace("+", "_")
    return os.path.join(root, "data", "processed", f"sentiment_scores_{safe}.parquet")


def add_sentiment_cached(posts_df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """add_sentiment_fast + the permanent store: known ids are looked up,
    unknown ids are scored in parallel and APPENDED to the store. After the
    first full build, historical rebuilds cost a lookup, not a rescore."""
    import os

    if "id" not in posts_df.columns:
        # no ids -> nothing to key the store on; just score directly
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
    """rows of (date, entity, sentiment) -> daily per-entity aggregates."""
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
    """Daily sentiment per TICKER. Reuses the exact same extractor as the
    mention counts, so screening/stop lists apply identically.
    Returns DataFrame(date, ticker, n_posts, avg_sentiment, net_bullish)."""
    from .extract_tickers import extract_tickers_from_text

    rows = []
    titles = posts_with_sent["title"].fillna("").astype(str)
    bodies = posts_with_sent["selftext"].fillna("").astype(str)
    for date, title, body, sent in zip(posts_with_sent["date"].astype(str),
                                       titles, bodies,
                                       posts_with_sent["sentiment"]):
        tickers = set(extract_tickers_from_text(title + " " + body, universe,
                                                cashtags_only=cashtags_only))
        for ticker in tickers:          # one vote per post per ticker
            rows.append((date, ticker, sent))
    return _aggregate(rows).rename(columns={"entity": "ticker"})


def build_daily_theme_sentiment(posts_with_sent: pd.DataFrame) -> pd.DataFrame:
    """Daily sentiment per THEME (keyword matching from src/themes.py).
    Returns DataFrame(date, theme, n_posts, avg_sentiment, net_bullish)."""
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
