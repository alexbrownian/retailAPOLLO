"""
signals.py
==========
The BUY/SELL decision engine - the direct replacement for RetailFlow1
notebook 10, with identical rules, thresholds and output schema. Pure
computation: a full rerun over nine years of aggregates takes ~2 seconds.

THE PHILOSOPHY: FEWER TRADES, MORE CONVICTION
---------------------------------------------
A signal only fires when MOMENTUM and SENTIMENT agree - neither alone is
enough. Everything is measured against each name's OWN trailing 84-day
baseline (never whole-window statistics), so a signal on day t uses only
information available on day t: what you see in a backtest is exactly what
the live run would have produced.

WHAT MAKES A BUY - all three must hold on the same day:
  1. A momentum trigger CROSSES up: attention z or conviction z crosses
     above +K (crossing means yesterday <= K, today > K - so one surge
     produces exactly ONE trade, not a signal every day the surge lasts).
     K = 2.5 means only the top ~1% most abnormal days even qualify.
  2. Sentiment agrees: the 5-day change of the net-bullish share is
     POSITIVE - the mood is improving vs its own recent past, not just loud.
  3. Score >= 4 of 5 (the checklist below).

WHAT MAKES A SELL - the mirror image, with a deliberately harder bar
(retail skews bullish, so bearish evidence must be stronger): a bearish
trigger (conviction z crosses below -K, or the CROWDED-TOP divergence
newly activates - attention above the surge line while the mood
deteriorates, the classic distribution pattern), plus a NEGATIVE 5-day
sentiment change, plus a sell score >= 4 of 5.

THE SCORE - one point per independent check:
  | # | BUY check                        | SELL check                      |
  |---|----------------------------------|---------------------------------|
  | 1 | attention z > K (crowd unusually large)                | same      |
  | 2 | 5d sentiment change > 0 (improving) | 5d change < 0 (deteriorating)|
  | 3 | conviction z > K (large AND bullish) | conviction z < -K           |
  | 4 | crowded-top flag NOT active      | crowded-top flag ACTIVE         |
  | 5 | Reddit AND X mentions both rising (where X has coverage) | same    |

Every accepted trade's `reason` column spells out exactly which checks
fired with their actual numbers - no signal is a black box.

BOOKKEEPING RULES
  * signal_date vs action_date: the signal is computed on day t from data
    through day t; the order is stamped for the NEXT day (no look-ahead).
  * cooldown: once a name signals, the SAME side is suppressed for 21
    days - one episode, one trade.

TWO LEVELS, ONE ENGINE
  * THEMES -> anchor ETFs (the primary strategy; a theme aggregates
    hundreds of posts/day, so its lines are stable).
  * individual TICKERS (for backtest purposes; noisier by construction,
    hence the higher volume floor).

OUTPUT FILES (identical schema to notebook 10)
  trade_signals.parquet          themes:  signal_date, action_date, action,
                                 theme, etf, score, att_z, conv_z,
                                 sent_5d_chg, reason
  trade_signals_tickers.parquet  tickers: same, with ticker/trade columns
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import (ROLL, PROCESSED_DIR, SIG_K, SIG_MIN_SCORE,
                        SIG_MIN_SCORE_SELL, SIG_COOLDOWN_DAYS,
                        MIN_DAILY_MENTIONS, MIN_TICKER_MENTIONS,
                        SENT_CHANGE_HORIZON)
from src.themes import THEME_ETFS, build_ticker_to_themes
from analytics.loaders import (load, to_wide, TICKER_COUNTS,
                               TICKER_COUNTS_BY_SOURCE, TICKER_SENT,
                               THEME_SENT, THEME_SIGNALS, TICKER_SIGNALS)
from analytics.conviction import trailing_z


# ---------------------------------------------------------------------------
# crossing helpers - the "one surge = one trade" primitive
# ---------------------------------------------------------------------------
def crosses_above(z: pd.Series, k: float) -> pd.Series:
    """True on the day z moves from <= k to > k. NaN-safe: a warm-up day
    (NaN) can never register as a crossing."""
    return (z > k) & (z.shift(1) <= k)


def crosses_below(z: pd.Series, k: float) -> pd.Series:
    return (z < k) & (z.shift(1) >= k)


def _val(frame: pd.DataFrame | None, day, name):
    """One cell, or None when the frame/column/value is missing. The
    decision loop treats None as 'this evidence is unavailable' - a check
    that cannot be evaluated simply scores 0, it never crashes the run."""
    if frame is None or name not in frame.columns:
        return None
    v = frame.at[day, name]
    return None if pd.isna(v) else float(v)


# ---------------------------------------------------------------------------
# ingredient builders
# ---------------------------------------------------------------------------
def theme_attention_from_ticker_counts(counts: pd.DataFrame,
                                       all_days: pd.DatetimeIndex):
    """Theme-level ATTENTION built from ticker mentions rolled up through
    the ticker->theme map (NVDA counts toward semiconductors, ai AND
    ai_megacap). This inherits all the extractor's precision (stop lists,
    cashtag screening) rather than re-matching keywords.

    Returns the wide daily mention matrix per theme, volume-floored:
    themes averaging under MIN_DAILY_MENTIONS/day are dropped entirely -
    a z-score over a handful of posts is noise dressed as signal."""
    lookup = build_ticker_to_themes()
    c = counts.copy()
    c["themes"] = c["ticker"].map(lambda t: lookup.get(t, []))
    theme_rows = c.explode("themes").dropna(subset=["themes"])
    theme_daily = (theme_rows.groupby(["date", "themes"])["mention_count"].sum()
                   .rename("mentions").reset_index()
                   .rename(columns={"themes": "theme"}))
    wide = to_wide(theme_daily, "theme", "mentions", all_days, fill=0.0)
    return wide.loc[:, wide.mean() >= MIN_DAILY_MENTIONS]


def cross_source_rising(by_source: pd.DataFrame | None, entity_col: str,
                        all_days: pd.DatetimeIndex,
                        to_theme: bool = False) -> pd.DataFrame | None:
    """Check #5: are Reddit AND X mentions BOTH rising (5d change of the
    7d rolling sum positive on each)? Two independent crowds agreeing is
    stronger evidence than one platform's echo chamber.

    X coverage has gaps (2021-2023 in the archive), so the check is gated
    on 'X active': X must have shown ANY mention in the trailing 30 days,
    otherwise the check is False (not spuriously true or false from a
    coverage hole). Returns None when a source is missing entirely -
    the caller then skips the check for every name."""
    if by_source is None:
        return None
    bs = by_source.copy()
    bs["date"] = pd.to_datetime(bs["date"])
    if to_theme:
        lookup = build_ticker_to_themes()
        bs["themes"] = bs["ticker"].map(lambda t: lookup.get(t, []))
        bs = (bs.explode("themes").dropna(subset=["themes"])
              .groupby(["date", "source", "themes"])["mention_count"].sum()
              .reset_index().rename(columns={"themes": entity_col}))
    if not set(bs["source"].unique()) >= {"reddit", "x"}:
        return None

    def src_roll(source_name):
        w = to_wide(bs[bs["source"] == source_name], entity_col,
                    "mention_count", all_days, fill=0.0)
        return w.rolling(ROLL, min_periods=1).sum()

    r7, x7 = src_roll("reddit"), src_roll("x")
    x_active = x7.rolling(30, min_periods=1).max() > 0
    # align the two matrices (a name may exist on one platform only)
    cols = r7.columns.union(x7.columns)
    r7 = r7.reindex(columns=cols, fill_value=0.0)
    x7 = x7.reindex(columns=cols, fill_value=0.0)
    x_active = x_active.reindex(columns=cols, fill_value=False)
    return (r7.diff(5) > 0) & (x7.diff(5) > 0) & x_active


# ---------------------------------------------------------------------------
# THE DECISION ENGINE - shared by themes and tickers
# ---------------------------------------------------------------------------
def make_decisions(names, all_days, az_f, cz_f, dv_f, crowd_f, xrise_f,
                   instrument_of, k=SIG_K, min_score=SIG_MIN_SCORE,
                   min_score_sell=SIG_MIN_SCORE_SELL,
                   cooldown_days=SIG_COOLDOWN_DAYS) -> pd.DataFrame:
    """Apply the momentum + sentiment + score rules to every name.

    Parameters (all wide daily matrices sharing all_days as index):
      az_f    attention z          cz_f    conviction z
      dv_f    5d sentiment change  crowd_f crowded-top flags (bool)
      xrise_f cross-source both-rising (bool) or None
      instrument_of  name -> tradeable instrument (theme -> ETF, or
                     ticker -> itself)

    Returns the trade log, one row per accepted decision, with the
    `reason` string reconstructing the whole scorecard in words.
    """
    out = []
    last_signal = {}   # (name, side) -> last accepted day, for the cooldown
    for name in names:
        if name not in cz_f.columns:
            continue
        # momentum triggers (vectorised per name, then iterated only on
        # the handful of trigger days - this loop is tiny)
        buy_t = crosses_above(az_f[name], k) | crosses_above(cz_f[name], k)
        newly_crowded = (crowd_f[name] & ~crowd_f[name].shift(1).fillna(False)
                         if name in crowd_f.columns
                         else pd.Series(False, index=all_days))
        sell_t = crosses_below(cz_f[name], -k) | newly_crowded

        for day in all_days[(buy_t | sell_t).fillna(False)]:
            az = _val(az_f, day, name)
            cz = _val(cz_f, day, name)
            dv = _val(dv_f, day, name)
            is_crowded = (bool(crowd_f.at[day, name])
                          if name in crowd_f.columns else False)
            xr = (bool(xrise_f.at[day, name])
                  if xrise_f is not None and name in xrise_f.columns else None)

            # GATE: momentum fired, but sentiment must AGREE or no trade.
            if bool(buy_t.get(day, False)) and dv is not None and dv > 0:
                side, min_needed = "BUY", min_score
                checks = [
                    (az is not None and az > k,
                     f"attention surged (z {az:+.2f} > {k})" if az is not None else ""),
                    (True, f"mood improving (5d sentiment {dv:+.3f} > 0)"),
                    (cz is not None and cz > k,
                     f"crowd leaning bullish (conviction z {cz:+.2f} > {k})" if cz is not None else ""),
                    (not is_crowded, "no crowded-top warning"),
                    (xr is True, "Reddit AND X both rising"),
                ]
            elif bool(sell_t.get(day, False)) and dv is not None and dv < 0:
                side, min_needed = "SELL", min_score_sell
                checks = [
                    (az is not None and az > k,
                     f"attention surged (z {az:+.2f} > {k})" if az is not None else ""),
                    (True, f"mood deteriorating (5d sentiment {dv:+.3f} < 0)"),
                    (cz is not None and cz < -k,
                     f"crowd leaning bearish (conviction z {cz:+.2f} < -{k})" if cz is not None else ""),
                    (is_crowded, "crowded-top divergence ACTIVE (crowd up, mood down)"),
                    (xr is True, "Reddit AND X both rising"),
                ]
            else:
                continue   # momentum fired but sentiment disagreed -> NO trade

            score = sum(1 for ok, _ in checks if ok)
            if score < min_needed:
                continue   # fewer trades, more conviction

            # COOLDOWN: the same call on the same name within the window is
            # the same move still going - one signal is enough to act on.
            prev = last_signal.get((name, side))
            if cooldown_days and prev is not None and (day - prev).days < cooldown_days:
                continue
            last_signal[(name, side)] = day

            reason = " + ".join(txt for ok, txt in checks if ok and txt)
            instr = instrument_of(name)
            out.append({"signal_date": day.date(),
                        "action_date": (day + pd.Timedelta(days=1)).date(),
                        "action": side, "name": name, "instrument": instr,
                        "score": score, "att_z": az, "conv_z": cz,
                        "sent_5d_chg": dv,
                        "reason": f"{side} {instr}: {reason} -> score {score}/5"})
    if not out:
        return pd.DataFrame()
    return (pd.DataFrame(out).sort_values("signal_date")
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# the two runs: themes and tickers
# ---------------------------------------------------------------------------
def compute_theme_signals(start=None, end=None) -> pd.DataFrame:
    """The primary strategy: theme-level decisions traded via anchor ETFs.

    start/end clip the INPUT window ('YYYY-MM-DD', end exclusive) - normally
    left None so signals exist over the full aggregate history and any view
    window just filters them."""
    counts = load(TICKER_COUNTS)
    theme_sent = load(THEME_SENT)
    if counts is None or theme_sent is None:
        raise FileNotFoundError(
            "need daily_ticker_counts + daily_theme_sentiment - run the "
            "pipeline (update_data.py) so the aggregates exist first")
    counts["date"] = pd.to_datetime(counts["date"])
    if start:
        counts = counts[counts["date"] >= start]
    if end:
        counts = counts[counts["date"] < end]

    all_days = pd.date_range(counts["date"].min(), counts["date"].max(), freq="D")

    # ingredient 1: attention z per theme (from ticker counts, volume-floored)
    wide = theme_attention_from_ticker_counts(counts, all_days)
    att_z = trailing_z(wide)

    # ingredients 2-4: conviction z, sentiment change, crowded flags -
    # all from theme sentiment. NOTE: net_bullish pivots with NaN (not 0)
    # for missing days - "no posts" is "no opinion", and the (n * b)
    # product then correctly contributes zero pressure via fillna(0).
    ts = theme_sent.copy()
    ts["date"] = pd.to_datetime(ts["date"])
    wn = to_wide(ts, "theme", "n_posts", all_days, fill=0.0)
    wb = to_wide(ts, "theme", "net_bullish", all_days, fill=None)
    pressure = (wn * wb).fillna(0)
    conv_z = trailing_z(pressure)
    share = (pressure.rolling(ROLL, min_periods=1).sum()
             / wn.rolling(ROLL, min_periods=1).sum().replace(0, np.nan))
    sent_change = share.diff(SENT_CHANGE_HORIZON)
    crowded = (att_z.reindex(columns=share.columns) > 1) & (sent_change < -0.10)

    # ingredient 5: cross-source confirmation
    both_rising = cross_source_rising(load(TICKER_COUNTS_BY_SOURCE), "theme",
                                      all_days, to_theme=True)

    trades = make_decisions(wide.columns, all_days, att_z, conv_z,
                            sent_change, crowded, both_rising,
                            lambda th: THEME_ETFS.get(th, "?"))
    if len(trades):
        trades = trades.rename(columns={"name": "theme", "instrument": "etf"})
    return trades


def compute_ticker_signals(start=None, end=None) -> pd.DataFrame:
    """The SAME engine per single ticker: attention from its own mention
    counts, sentiment from the per-ticker file, identical gates and
    scoring. The instrument bought/sold is the ticker itself. Noisier than
    themes by construction - hence the higher volume floor."""
    counts = load(TICKER_COUNTS)
    ticker_sent = load(TICKER_SENT)
    if counts is None:
        raise FileNotFoundError("daily_ticker_counts.parquet missing")
    if ticker_sent is None:
        return pd.DataFrame()   # sentiment is REQUIRED, same as for themes
    counts["date"] = pd.to_datetime(counts["date"])
    if start:
        counts = counts[counts["date"] >= start]
    if end:
        counts = counts[counts["date"] < end]

    all_days = pd.date_range(counts["date"].min(), counts["date"].max(), freq="D")

    t_wide = to_wide(counts, "ticker", "mention_count", all_days, fill=0.0)
    t_wide = t_wide.loc[:, t_wide.mean() >= MIN_TICKER_MENTIONS]
    t_att_z = trailing_z(t_wide)

    tks = ticker_sent.copy()
    tks["date"] = pd.to_datetime(tks["date"])
    t_wn = to_wide(tks, "ticker", "n_posts", all_days, fill=0.0)
    t_wb = to_wide(tks, "ticker", "net_bullish", all_days, fill=None)
    t_pressure = (t_wn * t_wb).fillna(0)
    t_conv_z = trailing_z(t_pressure)
    t_share = (t_pressure.rolling(ROLL, min_periods=1).sum()
               / t_wn.rolling(ROLL, min_periods=1).sum().replace(0, np.nan))
    t_sent_change = t_share.diff(SENT_CHANGE_HORIZON)
    t_crowded = ((t_att_z.reindex(columns=t_share.columns) > 1)
                 & (t_sent_change < -0.10))

    t_both_rising = cross_source_rising(load(TICKER_COUNTS_BY_SOURCE),
                                        "ticker", all_days, to_theme=False)

    trades = make_decisions(t_wide.columns, all_days, t_att_z, t_conv_z,
                            t_sent_change, t_crowded, t_both_rising,
                            lambda t: t)   # instrument = the ticker itself
    if len(trades):
        trades = trades.rename(columns={"name": "ticker",
                                        "instrument": "trade"})
    return trades


def rebuild_signal_files(start=None, end=None, verbose: bool = True) -> dict:
    """The pipeline entry point: recompute both signal files on disk
    (atomic writes). Returns {filename: n_signals}.

    start/end ('YYYY-MM-DD', end exclusive) clip the input window, exactly
    like notebook 10's START_DATE/END_DATE params did. Left None (the
    default, and what live runs use) the engine sees the whole aggregate
    history. NOTE the volume floors are MEANS over the input window, so a
    ticker that is loud recently but quiet for years passes a windowed run
    yet fails a full-history one - windowed runs are how the old ticker
    backtests were produced."""
    from src.abstracted_data import _safe_write

    written = {}
    theme_trades = compute_theme_signals(start, end)
    if len(theme_trades):
        _safe_write(theme_trades, os.path.join(PROCESSED_DIR, THEME_SIGNALS))
    written[THEME_SIGNALS] = len(theme_trades)
    if verbose:
        n_buy = int((theme_trades["action"] == "BUY").sum()) if len(theme_trades) else 0
        n_sell = int((theme_trades["action"] == "SELL").sum()) if len(theme_trades) else 0
        print(f"  {len(theme_trades)} THEME decisions (BUY {n_buy} / SELL {n_sell}) "
              f"-> {THEME_SIGNALS}")

    ticker_trades = compute_ticker_signals(start, end)
    if len(ticker_trades):
        _safe_write(ticker_trades, os.path.join(PROCESSED_DIR, TICKER_SIGNALS))
    written[TICKER_SIGNALS] = len(ticker_trades)
    if verbose:
        n_buy = int((ticker_trades["action"] == "BUY").sum()) if len(ticker_trades) else 0
        n_sell = int((ticker_trades["action"] == "SELL").sum()) if len(ticker_trades) else 0
        print(f"  {len(ticker_trades)} TICKER decisions (BUY {n_buy} / SELL {n_sell}) "
              f"-> {TICKER_SIGNALS}")
    return written
