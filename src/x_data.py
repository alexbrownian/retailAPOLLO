"""
x_data.py
=========
Turn RAW X (Twitter) dumps into the project's standard posts shape, so
tweets and Reddit posts live in ONE table. THREE HuggingFace datasets are
supported; each has its own tiny normaliser, but they all funnel into the
SAME 9 columns - that is the whole trick for "same format, no new files":

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

THE REGISTRY (bottom of this file) maps a short key to the HuggingFace repo
and the right normaliser. fetch_x_data.py downloads each registry entry to
data/raw/X Data/<key>.csv.zst; add_x_data.py normalises every raw file it
finds there and rebuilds the x block of posts.parquet in one go.
To add ANOTHER dataset later: write one normaliser, add one registry line.

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
      score, comment_num -> num_comments. NOTE: the file repeats a tweet
      once per 'ticker_symbol' it mentions - the id dedup below collapses
      those back to one row (our extractor re-finds all tickers from the
      text anyway).

SCORE CAVEAT: only the mjw dataset carries likes; the score column is kept
in the schema for spam filtering only. ALL counting uses raw mention counts
(one post = 1) - score-based weighting was removed project-wide (see
design_decisions.xlsx #30).
"""

from __future__ import annotations

import re

import pandas as pd

# The 8 classic columns + the 'source' column.
OUTPUT_COLUMNS = ["id", "date", "author", "score", "subreddit",
                  "title", "selftext", "num_comments", "source"]

STATUS_ID = re.compile(r"/status/(\d+)")


def _dates_from(series) -> pd.Series:
    """Timestamps -> 'YYYY-MM-DD'. Handles ISO strings AND unix seconds
    (some dumps store post_date as seconds-since-1970). Mixed formats are
    expected here, so pandas' per-element-parse warning is suppressed."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        numeric = pd.to_numeric(series, errors="coerce")
        unix = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    parsed = parsed.fillna(unix)
    return parsed.dt.strftime("%Y-%m-%d")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Shared final step for every normaliser: drop unusable rows,
    dedup on id (first seen wins - same rule as the Reddit pipeline),
    sort by date so the rows form date-ordered blocks in the parquet."""
    df = df[(df["date"].notna()) & (df["id"] != "") & (df["title"].str.strip() != "")]
    df = df.drop_duplicates(subset="id", keep="first")
    return df.sort_values("date").reset_index(drop=True)[OUTPUT_COLUMNS]


def author_from_embed_title(embed_title) -> str:
    """'Crypto Mikey tweeted about PRIME, AXS' -> 'Crypto Mikey'."""
    if not isinstance(embed_title, str):
        return ""
    for marker in (" tweeted about ", " retweeted ", " quoted "):
        if marker in embed_title:
            return embed_title.split(marker)[0].strip()
    return ""


def normalise_tweets(raw: pd.DataFrame, keep_tweet_types=None) -> pd.DataFrame:
    """StephanAkkerman/financial-tweets (Nov 2023+, no like counts).
    keep_tweet_types: e.g. ['tweet'] to drop retweets/quotes; None = all."""
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
    """StephanAkkerman/stock-market-tweets-data (Apr-Jul 2020).
    Columns: id (row number!), created_at, text. No author, no likes."""
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
    """LIVE X data from the official v2 API (fetch_x_live.py writes it).
    The fetcher stores a flat csv: id, created_at, text, author, likes.
    Real tweet ids get the same 'x_' prefix as the historical dumps, so a
    tweet present in both can never be double-counted (first seen wins)."""
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
    """mjw/stock_market_tweets (2015-2020, top S&P companies).
    Columns: tweet_id, writer, post_date, body, comment_num, retweet_num,
    like_num, ticker_symbol. The same tweet_id repeats once per
    ticker_symbol - _finish()'s id dedup collapses that."""
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
# THE REGISTRY - one line per dataset. Raw file = data/raw/X Data/<key>.csv.zst
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
    # LIVE X via the official v2 API - fetch_x_live.py appends to this file
    # whenever X_BEARER_TOKEN is set in .env (pipeline armed, off until paid).
    # No HF repo: fetch_x_data.py skips entries without one.
    "x_api_live": {
        "repo": None,
        "normaliser": normalise_x_api,
    },
}
