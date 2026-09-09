"""Normalisation of X (Twitter) dumps into the standard posts schema.

Turns raw X data into the project's standard posts shape, so tweets and
Reddit posts live in one table. Three HuggingFace datasets and the live
v2 API feed are supported; each has its own small normaliser, but they
all funnel into the same 9 columns::

    id           <- real tweet/status id where the dataset has one
                    (prefixed 'x_'), else a dataset-scoped row id
                    ('x_smt_<n>'). Prefixes guarantee no collision with
                    Reddit base36 ids or with each other.
    date         <- tweet timestamp, cut to YYYY-MM-DD
    author       <- where available, else ''
    score        <- like count where available, else 0
    subreddit    <- 'x_twitter' (so subreddit filters/blocks still work)
    title        <- the tweet text (tweets have no title/body split)
    selftext     <- ''
    num_comments <- reply/comment count where available, else 0
    source       <- 'x'  (Reddit rows carry source='reddit')

The DATASETS registry (bottom of this file) maps a short key to the
HuggingFace repo and the right normaliser; raw files live at
data/raw/X Data/<key>.csv.zst. The live feed is the one consumed by the
pipeline: ingestion/fetch_x_live.py writes x_api_live.csv.zst and
ingestion/merge_live.py / append_live_abstracted.py normalise it with
``normalise_x_api()``. To add another dataset: write one normaliser, add
one registry line.

Datasets and their quirks:
  financial_tweets (StephanAkkerman/financial-tweets)
      ~315k rows, Nov 2023 onwards. Real status ids in 'url'; tweet text in
      'description'; author parsed from 'embed_title'. NO like counts.
  stock_market_tweets_data (StephanAkkerman/stock-market-tweets-data)
      ~924k rows, Apr 9 - Jul 16 2020 (S&P 500 tags). Columns are just
      id / created_at / text - the id is a ROW NUMBER, not a tweet id, so
      it gets the 'x_smt_' prefix. No author, NO like counts. Many rows are
      retweets ("RT @..."), which still carry the cashtags being echoed.
  stock_market_tweets (mjw/stock_market_tweets)
      Millions of rows, 2015-2020 (top S&P companies). Real 'tweet_id';
      text in 'body'; 'writer' is the author; HAS engagement: like_num ->
      score, comment_num -> num_comments. The file repeats a tweet once
      per 'ticker_symbol' it mentions; the id dedup collapses those back
      to one row (the extractor re-finds all tickers from the text).

Score caveat: only the mjw dataset carries likes; the score column is
kept in the schema for spam filtering only. All counting uses raw mention
counts (one post = 1); score-based weighting is not used anywhere in the
project because archived scores leak future information.
"""

from __future__ import annotations

import re

import pandas as pd

# The 8 classic columns + the 'source' column.
OUTPUT_COLUMNS = ["id", "date", "author", "score", "subreddit",
                  "title", "selftext", "num_comments", "source"]

STATUS_ID = re.compile(r"/status/(\d+)")


def _dates_from(series) -> pd.Series:
    """Converts timestamps to 'YYYY-MM-DD' strings.

    Handles ISO strings and unix seconds or milliseconds (dumps vary).
    Mixed formats are expected, so pandas' per-element-parse warning is
    suppressed.

    The numeric path is range-guarded: a raw feed row can carry a huge
    numeric in created_at (a tweet/status id is ~2e18), and feeding that
    to to_datetime(unit='s') multiplies toward nanoseconds and overflows,
    which raises FloatingPointError where numpy is set to raise. Only
    values inside a sane band are treated as timestamps: seconds
    ~1973-2128, and the matching millisecond band; anything else (ids,
    garbage) becomes NaT and the row is dropped downstream. A snowflake
    id is deliberately not decoded as a nanosecond stamp; it would
    produce a plausible-looking wrong date.
    """
    import warnings
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    # The numeric fallback is best-effort on top of the string parse and
    # must never kill the fold. The range bands and errstate do not stop
    # a FloatingPointError on every numpy/pandas pairing, so the whole
    # fallback sits behind a try: a row the fallback cannot date is
    # dropped downstream, which is the correct cost.
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            numeric = pd.to_numeric(series, errors="coerce")
            _sec = numeric.where((numeric >= 1e8) & (numeric < 5e9))
            _ms = numeric.where((numeric >= 1e11) & (numeric < 5e12))
            unix = pd.to_datetime(_sec, unit="s", utc=True,
                                  errors="coerce")
            unix = unix.fillna(pd.to_datetime(_ms, unit="ms",
                                              utc=True,
                                              errors="coerce"))
        parsed = parsed.fillna(unix)
    except Exception:                                    # noqa: BLE001
        pass
    return parsed.dt.strftime("%Y-%m-%d")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Shared final step for every normaliser: drops unusable rows,
    dedups on id (first seen wins, the same rule as the Reddit pipeline)
    and sorts by date so the rows form date-ordered blocks in the parquet."""
    df = df[(df["date"].notna()) & (df["id"] != "") & (df["title"].str.strip() != "")]
    df = df.drop_duplicates(subset="id", keep="first")
    return df.sort_values("date").reset_index(drop=True)[OUTPUT_COLUMNS]


def author_from_embed_title(embed_title) -> str:
    """Extracts the author from a financial-tweets embed title, for
    example 'Crypto Mikey tweeted about PRIME, AXS' -> 'Crypto Mikey'.
    Returns '' when no known marker is present."""
    if not isinstance(embed_title, str):
        return ""
    for marker in (" tweeted about ", " retweeted ", " quoted "):
        if marker in embed_title:
            return embed_title.split(marker)[0].strip()
    return ""


def normalise_tweets(raw: pd.DataFrame, keep_tweet_types=None) -> pd.DataFrame:
    """Normalises StephanAkkerman/financial-tweets (Nov 2023+, no likes).

    Args:
        raw: The raw dataset frame.
        keep_tweet_types: Values of tweet_type to keep, for example
            ['tweet'] to drop retweets and quotes; None keeps all.

    Returns:
        DataFrame with OUTPUT_COLUMNS.
    """
    df = raw.copy()
    if keep_tweet_types and "tweet_type" in df.columns:
        df = df[df["tweet_type"].isin(keep_tweet_types)]

    def make_id(url):
        if isinstance(url, str):
            m = STATUS_ID.search(url)
            if m:
                return "x_" + m.group(1)
        return ""

    out = pd.DataFrame({
        "id": df.get("url", pd.Series(dtype=str)).map(make_id),
        "date": _dates_from(df.get("timestamp")),
        "author": df.get("embed_title", pd.Series(dtype=str)).map(author_from_embed_title),
        "score": 0,
        "subreddit": "x_twitter",
        "title": df.get("description", pd.Series(dtype=str)).fillna("").astype(str),
        "selftext": "",
        "num_comments": 0,
        "source": "x",
    })
    return _finish(out)


def normalise_smt(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalises StephanAkkerman/stock-market-tweets-data (Apr-Jul 2020).

    Columns are id (a row number, not a tweet id), created_at and text;
    no author, no likes.
    """
    df = raw.copy()
    ids = pd.to_numeric(df.get("id"), errors="coerce")
    out = pd.DataFrame({
        "id": ids.map(lambda v: f"x_smt_{int(v)}" if pd.notna(v) else ""),
        "date": _dates_from(df.get("created_at")),
        "author": "",
        "score": 0,
        "subreddit": "x_twitter",
        "title": df.get("text", pd.Series(dtype=str)).fillna("").astype(str),
        "selftext": "",
        "num_comments": 0,
        "source": "x",
    })
    return _finish(out)


def normalise_x_api(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalises live X data from the official v2 API.

    fetch_x_live.py stores a flat csv: id, created_at, text, author,
    likes. Real tweet ids get the same 'x_' prefix as the historical
    dumps, so a tweet present in both is never double-counted (first
    seen wins).
    """
    df = raw.copy()
    ids = pd.to_numeric(df.get("id"), errors="coerce")
    out = pd.DataFrame({
        "id": ids.map(lambda v: f"x_{int(v)}" if pd.notna(v) else ""),
        "date": _dates_from(df.get("created_at")),
        "author": df.get("author", pd.Series(dtype=str)).fillna("").astype(str),
        "score": pd.to_numeric(df.get("likes"), errors="coerce").fillna(0).astype(int),
        "subreddit": "x_twitter",
        "title": df.get("text", pd.Series(dtype=str)).fillna("").astype(str),
        "selftext": "",
        "num_comments": 0,
        "source": "x",
    })
    return _finish(out)


def normalise_mjw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalises mjw/stock_market_tweets (2015-2020, top S&P companies).

    Columns: tweet_id, writer, post_date, body, comment_num, retweet_num,
    like_num, ticker_symbol. The same tweet_id repeats once per
    ticker_symbol; _finish()'s id dedup collapses that.
    """
    df = raw.copy()
    ids = pd.to_numeric(df.get("tweet_id"), errors="coerce")
    out = pd.DataFrame({
        "id": ids.map(lambda v: f"x_{int(v)}" if pd.notna(v) else ""),
        "date": _dates_from(df.get("post_date")),
        "author": df.get("writer", pd.Series(dtype=str)).fillna("").astype(str),
        "score": pd.to_numeric(df.get("like_num"), errors="coerce").fillna(0).astype(int),
        "subreddit": "x_twitter",
        "title": df.get("body", pd.Series(dtype=str)).fillna("").astype(str),
        "selftext": "",
        "num_comments": pd.to_numeric(df.get("comment_num"), errors="coerce").fillna(0).astype(int),
        "source": "x",
    })
    return _finish(out)


# ---------------------------------------------------------------------
# The registry: one entry per dataset. Raw file = data/raw/X Data/<key>.csv.zst
# ---------------------------------------------------------------------
DATASETS = {
    "financial_tweets": {
        "repo": "StephanAkkerman/financial-tweets",
        "normaliser": normalise_tweets,
    },
    "stock_market_tweets_data": {
        "repo": "StephanAkkerman/stock-market-tweets-data",
        "normaliser": normalise_smt,
    },
    "stock_market_tweets": {
        "repo": "mjw/stock_market_tweets",
        "normaliser": normalise_mjw,
    },
    # Live X via the official v2 API. fetch_x_live.py appends to this
    # file whenever X_BEARER_TOKEN is set in .env; without the token the
    # feed is inactive. No HF repo.
    "x_api_live": {
        "repo": None,
        "normaliser": normalise_x_api,
    },
}
