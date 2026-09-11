"""
robust_share.py
===============
Coverage-robust SHARE-OF-CHATTER estimation - the estimator behind every
attention series (charts AND signal features).

WHY A PLAIN DAILY RATIO DOES NOT WORK
-------------------------------------
The naive estimator computes "share of the forum" per DAY as mentions /
that day's total, then 7d-averages it. Two facts about the live archive
break that estimator:

1. THE PULL CADENCE IS UNEVEN. `update_data` runs ~2x/week, and a catch-up
   pull lands several days of posts at once, so daily totals can swing
   by more than 10x between adjacent days. On a thin day a real name
   reads a fake zero while genuinely being discussed, and on a fat day
   every share is diluted at once. Averaging the daily RATIOS weights a
   10-post day exactly as much as a 4,000-post day.

2. THE SOURCE MIX IS A REGIME, NOT A CONSTANT. Reddit runs back to 2017,
   StockTwits only ramps from Feb-2026, X only exists from Jul-2026. A
   denominator that pools all sources makes a Reddit-heavy name's share
   collapse the day a big StockTwits pull lands - a composition
   artifact, not a crowd movement.

THE ESTIMATOR (three standard techniques, one line each)
--------------------------------------------------------
* RATIO-OF-SUMS (a.k.a. the ratio estimator): share over the trailing
  7d window = (sum of the name's mentions) / (sum of ALL mentions),
  not the mean of daily ratios. Thin days then contribute in proportion
  to the evidence they actually carry. (Cochran, Sampling Techniques -
  the same reason CTRs are computed clicks/impressions, not averaged
  daily rates.)
* PER-SOURCE STRATIFICATION (tickers, where a by-source aggregate
  exists): the share is computed WITHIN each source, then combined with
  weights equal to each source's own trailing-90d volume share,
  renormalised over the sources actually present in the window. A
  missing StockTwits pull then just means "use the sources we do have"
  instead of "halve everyone's share".
* EMPIRICAL-BAYES SHRINKAGE toward the name's own trailing-120d share:
  posterior = (k + tau * p0) / (N + tau), with prior strength
  tau = MIN_TOTAL posts (the project's existing too-thin-to-trust floor;
  no new constant). With hundreds of posts the data speaks for itself
  (tau is negligible); with a handful, the estimate leans on the name's
  own recent baseline instead of printing 0% or 100%.

All windows are TRAILING (day t uses only data <= t) - nothing here can
leak the future into a backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MIN_TOTAL

# The house windows, reused - no new constants:
SHARE_WINDOW_D = 7        # ROLL, the one-week attention window (A1, E1)
PRIOR_WINDOW_D = 120      # the A1 hype gate's own baseline window
SOURCE_MIX_WINDOW_D = 90  # ~one quarter of trailing volume sets the mix
PRIOR_STRENGTH = MIN_TOTAL  # posts of evidence at which data outweighs prior


# ---------------------------------------------------------------------------
# cached day-total tables (the same frames are passed once per instrument -
# ~100x per pipeline run - so the groupby is paid once, not 100 times)
# ---------------------------------------------------------------------------
_TOTALS_CACHE: dict = {}


def _day_totals(counts_long: pd.DataFrame) -> pd.Series:
    """Total mentions per day, cached on the identity of the frame."""
    key = (id(counts_long), len(counts_long))
    hit = _TOTALS_CACHE.get(key)
    if hit is not None:
        return hit
    tot = counts_long.groupby("date")["mention_count"].sum()
    if len(_TOTALS_CACHE) > 16:      # a run touches ~4 tables; stay tiny
        _TOTALS_CACHE.clear()
    _TOTALS_CACHE[key] = tot
    return tot


def _source_totals(by_source: pd.DataFrame) -> pd.DataFrame:
    """date x source total-mention matrix, cached like _day_totals."""
    key = (id(by_source), len(by_source), "src")
    hit = _TOTALS_CACHE.get(key)
    if hit is not None:
        return hit
    tot = (by_source.groupby(["date", "source"])["mention_count"].sum()
           .unstack(fill_value=0.0))
    _TOTALS_CACHE[key] = tot
    return tot


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------
def _shrunk_ratio(k: pd.Series, n: pd.Series,
                  window: int, prior_days: int,
                  prior_strength: float) -> pd.Series:
    """Ratio-of-sums over `window` days with empirical-Bayes shrinkage
    toward the trailing `prior_days` ratio. k = the name's daily
    mentions, n = the day totals (same index). Returns a 0-1 share."""
    k7 = k.rolling(window, min_periods=1).sum()
    n7 = n.rolling(window, min_periods=1).sum()
    k120 = k.rolling(prior_days, min_periods=window).sum()
    n120 = n.rolling(prior_days, min_periods=window).sum()
    p0 = (k120 / n120.where(n120 > 0))
    # no baseline yet (young series): fall back to the raw window ratio
    raw = k7 / n7.where(n7 > 0)
    p0 = p0.fillna(raw)
    return (k7 + prior_strength * p0) / (n7 + prior_strength)


def robust_share(counts_long: pd.DataFrame, entity_col: str, name: str,
                 all_days: pd.DatetimeIndex,
                 by_source: pd.DataFrame | None = None,
                 window: int = SHARE_WINDOW_D,
                 prior_days: int = PRIOR_WINDOW_D,
                 prior_strength: float = PRIOR_STRENGTH) -> pd.Series:
    """The coverage-robust share (in %, like the series it replaces).

    by_source: the per-source long frame (tickers only - themes have no
    by-source aggregate and use the unstratified estimator, which still
    gets the ratio-of-sums + shrinkage fixes)."""
    if by_source is not None and "source" in by_source.columns:
        one = by_source[by_source[entity_col] == name]
        src_tot = _source_totals(by_source).reindex(all_days).fillna(0.0)
        if len(one):
            k_by = (one.groupby(["date", "source"])["mention_count"].sum()
                    .unstack(fill_value=0.0)
                    .reindex(index=all_days, columns=src_tot.columns)
                    .fillna(0.0))
        else:
            k_by = pd.DataFrame(0.0, index=all_days,
                                columns=src_tot.columns)
        shares, weights = [], []
        n7_by = src_tot.rolling(window, min_periods=1).sum()
        w90 = src_tot.rolling(SOURCE_MIX_WINDOW_D, min_periods=1).sum()
        for s in src_tot.columns:
            sh = _shrunk_ratio(k_by[s], src_tot[s], window, prior_days,
                               prior_strength)
            # a source with nothing in the window has no estimate to give
            shares.append(sh.where(n7_by[s] > 0))
            weights.append(w90[s])
        sh = pd.concat(shares, axis=1)
        w = pd.concat(weights, axis=1)
        w = w.where(sh.notna())              # only sources present today
        wsum = w.sum(axis=1)
        combined = (sh * w).sum(axis=1) / wsum.where(wsum > 0)
        return combined * 100
    m = (counts_long[counts_long[entity_col] == name]
         .groupby("date")["mention_count"].sum()
         .reindex(all_days).fillna(0.0))
    tot = _day_totals(counts_long).reindex(all_days).fillna(0.0)
    return _shrunk_ratio(m, tot, window, prior_days, prior_strength) * 100


def window_totals(counts_long: pd.DataFrame, all_days: pd.DatetimeIndex,
                  window: int = SHARE_WINDOW_D) -> pd.Series:
    """Total posts inside the trailing `window` days - the coverage mass
    behind each share value. The display layer masks stretches where even
    the whole window holds under MIN_TOTAL posts (there is genuinely
    nothing to estimate from)."""
    tot = _day_totals(counts_long).reindex(all_days).fillna(0.0)
    return tot.rolling(window, min_periods=1).sum()
