"""
build_mentions.py
=================
Bridge between the cleaned posts and the analysis notebooks.

It takes the tidy posts table (from clean_data.py), runs the ticker extractor
on each post's title + selftext, and returns a daily count table:

    date, ticker, mention_count

That table is exactly what the notebooks (mentions-over-time and
first-derivative) expect. We keep the heavy ticker logic in extract_tickers.py
and the valid-symbol list in ticker_universe.py - this file just wires them
together so a notebook can do it in one line.

WHY THERE IS NO UPVOTE-WEIGHTED COUNT ANY MORE (removed 2026-07-06):
The old weighted_count summed score**2 per post. But archived Reddit dumps
carry each post's FINAL score - the upvotes it collected over days or weeks
AFTER posting. A backtest that weights day-t mentions by final scores is
therefore reading the future ("this post will go viral") - a look-ahead
leak that silently inflates any result built on it. Raw mention counts are
immune: one post = 1 the moment it exists. If weighting ever returns, it
must use scores AS OF the mention day (live re-poll pipeline) - see
design_decisions.xlsx #30 and the README live-data checklist.
"""

import pandas as pd

from .extract_tickers import extract_tickers_from_text


def build_daily_counts(posts_df, universe, cashtags_only=False):
    """
    posts_df : DataFrame with columns date, title, selftext
    universe : set of valid ticker symbols (from load_us_ticker_universe)
    cashtags_only : True = only count $TICKER (cleaner, fewer false hits)

    Returns a DataFrame with columns:
        date, ticker, mention_count

    mention_count = number of distinct posts that mention the ticker that day
    (each post counted once regardless of how many times the ticker appears
    in it - breadth of attention, not verbosity).
    """
    rows = []
    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str)
    dates = posts_df["date"].astype(str)

    for date, title, body in zip(dates, titles, bodies):
        text = title + " " + body
        # set() deduplicates so each ticker contributes at most once per post.
        for ticker in set(extract_tickers_from_text(text, universe, cashtags_only=cashtags_only)):
            rows.append({"date": date, "ticker": ticker})

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "mention_count"])

    long_df = pd.DataFrame(rows)
    daily = (
        long_df.groupby(["date", "ticker"], as_index=False)
        .size()
        .rename(columns={"size": "mention_count"})
    )
    return daily
