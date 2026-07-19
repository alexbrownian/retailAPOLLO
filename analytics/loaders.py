"""
loaders.py
==========
Shared data access for the analytics package and the dashboard: read the
aggregate parquet files, clip them to a view window, and turn the long
"one row per (date, entity)" tables into the WIDE "dates x entities"
matrices every downstream computation wants.

WHY WIDE MATRICES
-----------------
The aggregates are stored long (date, ticker, mention_count) because that
is compact and merge-friendly. But every analytic - rolling sums, z-scores,
crossings - is a column-wise time-series operation. Pivoting once into a
DataFrame whose index is EVERY calendar day and whose columns are entities
means:

  * missing days become explicit zeros (a quiet day is data, not a gap -
    skipping it would corrupt every rolling window and derivative);
  * one vectorised pandas call processes every entity at once (this is the
    core reason the .py analytics finish in seconds where the notebook
    chain took minutes: no per-entity Python loops, no chart rendering).

CACHING
-------
`load(name)` memoises on (path, file-mtime): repeated calls within one
process are free, and the cache invalidates itself the moment the pipeline
rewrites a file. The dashboard adds its own st.cache_data layer on top with
the same key - the two never disagree.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

from src.config import PROCESSED_DIR, PRICES_PATH

# Canonical filenames (identical to RetailFlow1, so the two projects'
# outputs remain directly comparable file-for-file).
TICKER_COUNTS = "daily_ticker_counts.parquet"
TICKER_COUNTS_BY_SOURCE = "daily_ticker_counts_by_source.parquet"
TICKER_SENT = "daily_ticker_sentiment.parquet"
THEME_COUNTS = "daily_theme_counts.parquet"
THEME_SENT = "daily_theme_sentiment.parquet"
TICKER_CONVICTION = "daily_ticker_conviction.parquet"
THEME_CONVICTION = "daily_theme_conviction.parquet"
THEME_SIGNALS = "trade_signals.parquet"
TICKER_SIGNALS = "trade_signals_tickers.parquet"


def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@lru_cache(maxsize=64)
def _read_cached(path: str, mtime: float) -> pd.DataFrame:
    """The real read. mtime is part of the cache key ONLY - a rewritten
    file (new mtime) misses the cache and is re-read automatically."""
    df = pd.read_parquet(path)
    for col in ("date", "action_date", "signal_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load(name: str, folder: str = PROCESSED_DIR) -> pd.DataFrame | None:
    """Load one aggregate by filename; None if it does not exist yet.
    Returns a COPY so callers can mutate freely without corrupting the
    cache (the frames are small; the copy is cheap)."""
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    return _read_cached(path, _mtime(path)).copy()


def load_prices() -> pd.DataFrame | None:
    """The Bloomberg price store (date, symbol, px_last), or None."""
    if not os.path.exists(PRICES_PATH):
        return None
    return _read_cached(PRICES_PATH, _mtime(PRICES_PATH)).copy()


def clip_window(df: pd.DataFrame, col: str, lo, hi) -> pd.DataFrame:
    """Rows with lo <= df[col] <= hi. hi=None means 'to the newest'."""
    out = df[df[col] >= lo]
    return out if hi is None else out[out[col] <= hi]


def price_series(prices: pd.DataFrame, symbol: str, lo, hi) -> pd.Series:
    """One symbol's daily close as a CONTINUOUS daily series.

    Prices only exist on trading days; forward-filling weekends/holidays
    gives a gap-free line that aligns date-for-date with the (calendar-
    daily) mention series, so overlays and as-of lookups never miss."""
    one = prices[prices["symbol"] == symbol].sort_values("date")
    s = one.set_index("date")["px_last"]
    if not s.empty:
        s = s.asfreq("D").ffill()
    s = s[s.index >= lo]
    return s if hi is None else s[s.index <= hi]


def to_wide(long_df: pd.DataFrame, entity_col: str, value_col: str,
            all_days: pd.DatetimeIndex | None = None,
            fill: float | None = 0.0) -> pd.DataFrame:
    """Long (date, entity, value) -> wide (all calendar days x entities).

    all_days : the full daily index to align onto; None = span of the data.
    fill     : what a missing (day, entity) cell becomes. 0.0 is right for
               COUNTS ("no posts" = zero mentions). None keeps NaN, which
               is right for RATIOS like net_bullish ("no posts" = no
               opinion, NOT a neutral one - averaging in fake zeros would
               drag every quiet name toward neutrality)."""
    if all_days is None:
        all_days = pd.date_range(long_df["date"].min(),
                                 long_df["date"].max(), freq="D")
    wide = (long_df.pivot_table(index="date", columns=entity_col,
                                values=value_col)
            .reindex(all_days))
    if fill is not None:
        wide = wide.fillna(fill)
    return wide


def day_span(*frames: pd.DataFrame) -> pd.DatetimeIndex:
    """One calendar-daily index covering every date in every given frame,
    so all the wide matrices in a computation share one aligned index."""
    lo = min(f["date"].min() for f in frames if f is not None and len(f))
    hi = max(f["date"].max() for f in frames if f is not None and len(f))
    return pd.date_range(lo, hi, freq="D")
