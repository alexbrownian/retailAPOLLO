"""
overlays.py
===========
Price-overlay analytics - the computations behind RetailFlow1's overlay
notebooks 11-16, reduced to pure functions that return DATA (series and
frames). The dashboard renders them as interactive Plotly; nothing here
draws anything, which is exactly why it is fast enough to recompute live
on every dashboard interaction.

WHAT EACH FUNCTION REPLACES
  mention_share_series()      nb 11 - mention share vs price
  chatter_change_series()     nb 12/13 - attention first derivative
  gradient_analysis()         nb 12 - chatter change -> forward price move
                              (scatter/deciles) + lead/lag correlation scan
  direction_flips()           nb 12 evidence view - chatter turning points
  conviction_crossings()      nb 14 - conviction z crossings of +/-1.5
  signal_scorecard()          nb 15/16 - the report card: hold every
                              signal HOLD_DAYS and tally the P&L

THE NORMALISATION RULE (applies to every mention series here)
-------------------------------------------------------------
Raw counts jump at the archive->live boundary: the archive held millions
of posts/day, a live fetch collects hundreds. Dividing by the day's TOTAL
mentions gives SHARE-OF-CHATTER (%), which is comparable across eras -
"NVDA was 8% of everything retail said today" means the same thing in
2021 and now. Days with fewer than MIN_TOTAL total mentions are masked
(NaN): a 1-post day would otherwise read as a fake 100% spike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ROLL, DERIV_SMOOTH, MIN_TOTAL, HOLD_DAYS, CROSS_AT, MIN_GAP
from analytics.loaders import price_series, clip_window


# ---------------------------------------------------------------------------
# mention share + first derivative
# ---------------------------------------------------------------------------
def mention_share_series(counts: pd.DataFrame, entity_col: str, name: str,
                         lo, hi, normalise: bool = True,
                         roll: int = ROLL) -> pd.Series:
    """One entity's daily mention line over the window.

    normalise=True  -> share of that day's total chatter (%), 7d-smoothed
                       (the era-safe default, see module docstring)
    normalise=False -> raw 7d rolling mean of mention counts
    """
    c = clip_window(counts, "date", lo, hi)
    day_totals = c.groupby("date")["mention_count"].sum()
    m = (c[c[entity_col] == name].sort_values("date")
         .set_index("date")["mention_count"].asfreq("D").fillna(0))
    if m.empty:
        return m
    if normalise:
        # keep everything float64: where() puts NaN where totals is zero,
        # so no pd.NA ever enters the series (rolling needs plain floats)
        totals = day_totals.reindex(m.index).fillna(0).astype("float64")
        m = (m.astype("float64") / totals.where(totals > 0)) * 100
        m[totals < MIN_TOTAL] = float("nan")
    return m.rolling(roll, min_periods=1).mean()


def chatter_change_series(counts: pd.DataFrame, entity_col: str, name: str,
                          lo, hi, smooth: int = DERIV_SMOOTH) -> pd.Series:
    """The FIRST DERIVATIVE of attention: day-to-day change of the smoothed
    mention share, then a short moving average over it.

    Think of the share as distance and this as speed: when it jumps from
    ~0 to clearly positive, attention is ACCELERATING - the crowd is
    arriving - which historically happens before the loudest headline days.
    The smoothing keeps it reactive within a week while killing the
    day-to-day sawtooth."""
    base = mention_share_series(counts, entity_col, name, lo, hi)
    return base.diff().rolling(smooth, min_periods=1).mean()


# ---------------------------------------------------------------------------
# does chatter change PREDICT price? (nb 12's three-panel analysis)
# ---------------------------------------------------------------------------
def gradient_analysis(counts: pd.DataFrame, entity_col: str, names: list,
                      prices: pd.DataFrame, resolve, lo, hi,
                      fwd_days: int = 5, max_lag: int = 15):
    """For a set of names, relate today's chatter CHANGE to the price move
    over the NEXT `fwd_days` days. Forward, not same-day - a relationship
    here means chatter PREDICTS, not just coincides.

    resolve : name -> priced symbol (identity for tickers, anchor-ETF
              resolution for themes).

    Returns (pairs, curves, lags):
      pairs  - long frame, one row per name-day: chatter_chg, fwd_ret, name
               (the dashboard scatters it and buckets it into deciles -
               a red->green staircase across deciles = real, monotonic edge)
      curves - {name: [corr at each lag]} where lag k>0 correlates today's
               chatter change with the price move k days LATER. A peak
               RIGHT of zero = chatter leads price (the tradeable case);
               a peak LEFT of zero = price moves first, chatter chases.
      lags   - the lag axis, range(-max_lag, +max_lag+1)
    """
    lags = list(range(-max_lag, max_lag + 1))
    pairs, curves = [], {}
    for name in names:
        symbol = resolve(name)
        if symbol is None:
            continue
        chg = chatter_change_series(counts, entity_col, name, lo, hi)
        px = price_series(prices, symbol, lo, hi)
        if px.empty or chg.dropna().empty:
            continue
        fwd = (px.shift(-fwd_days) / px - 1) * 100
        both = pd.DataFrame({"chatter_chg": chg, "fwd_ret": fwd}).dropna()
        both["name"] = name
        pairs.append(both)
        dr = px.pct_change() * 100
        curves[name] = [chg.corr(dr.shift(-k)) for k in lags]
    pairs = pd.concat(pairs) if pairs else pd.DataFrame()
    return pairs, curves, lags


def direction_flips(chg: pd.Series, px: pd.Series, lead_days: int,
                    min_gap: int = 7, eps_std: float = 0.25):
    """Chatter DIRECTION FLIPS marked on the price line (nb 12 evidence
    view). A state machine with hysteresis: +1 once the smoothed change
    clears +eps, -1 once it clears -eps, where eps = eps_std * std(change)
    - micro-wiggles around zero are ignored, and a flip only counts when
    entering the OPPOSITE state (plus a min_gap between counted flips).

    Returns (up_days, down_days, up_moves, down_moves, baseline):
      up_days/down_days   the flip dates (crowd re-engaging / losing interest)
      up_moves/down_moves the % price move over the next lead_days after each
      baseline            the unconditional lead_days drift, for comparison -
                          flips only mean something if they beat this."""
    eps = float(chg.std() * eps_std) if chg.notna().any() else 0.0
    up, down, state = [], [], 0
    for d, v in chg.dropna().items():
        if state <= 0 and v > eps:
            if not up or (d - up[-1]).days >= min_gap:
                up.append(d)
            state = 1
        elif state >= 0 and v < -eps:
            if not down or (d - down[-1]).days >= min_gap:
                down.append(d)
            state = -1

    def fwd_moves(events):
        out = []
        for d in events:
            p0 = px.asof(d)
            p1 = px.asof(d + pd.Timedelta(days=lead_days))
            if pd.notna(p0) and pd.notna(p1) and p0 != 0:
                out.append((p1 / p0 - 1) * 100)
        return out

    baseline = float(((px.shift(-lead_days) / px - 1) * 100).mean())
    return up, down, fwd_moves(up), fwd_moves(down), baseline


# ---------------------------------------------------------------------------
# conviction crossings (nb 14) - marked on the price line
# ---------------------------------------------------------------------------
def spaced(idx, gap: int) -> list:
    """First-in-a-burst wins: drop dates closer than `gap` days to the
    previously kept one, so a week-long surge marks once, not daily."""
    out = []
    for d in idx:
        if not out or (d - out[-1]).days >= gap:
            out.append(d)
    return out


def conviction_crossings(cz: pd.Series, cross_at: float = CROSS_AT,
                         min_gap: int = MIN_GAP):
    """The days conviction z CROSSES the +/-cross_at lines (crossing =
    yesterday inside, today outside - one surge, one marker), burst-spaced.
    Returns (bullish_days, bearish_days)."""
    up = spaced(cz[(cz > cross_at) & (cz.shift(1) <= cross_at)].index, min_gap)
    dn = spaced(cz[(cz < -cross_at) & (cz.shift(1) >= -cross_at)].index, min_gap)
    return up, dn


# ---------------------------------------------------------------------------
# the report card (nb 15/16) - did the signals make money?
# ---------------------------------------------------------------------------
def crossing_exits(cz: pd.Series, ups: list, dns: list,
                   exit_level: float, max_days: int = 60):
    """The EXIT companion to conviction_crossings: after each entry
    crossing, the first day z reverts inside +/-exit_level = the signal
    has expired ("back to neutral") - the validated early-exit point.

    Evidence (July-2026 study, real prices): exiting +2.5-crossing longs
    on reversion returned +0.83%/trade in ~10 days held vs +1.36% in 20
    days for the fixed hold - LESS per trade but MORE per day of capital
    (0.080 vs 0.065 %/day), i.e. the same money can work ~2x as often.

    Returns (long_exits, short_exits): one exit date per entry (capped at
    max_days after entry so an exit always exists)."""
    long_exits, short_exits = [], []
    for entries, out, cond in [
            (ups, long_exits, lambda w: w < exit_level),
            (dns, short_exits, lambda w: w > -exit_level)]:
        for d in entries:
            path = cz[d:d + pd.Timedelta(days=max_days)]
            hit = path[cond(path)]
            out.append(hit.index[0] if len(hit)
                       else d + pd.Timedelta(days=max_days))
    return long_exits, short_exits


def signal_scorecard(sig: pd.DataFrame, prices: pd.DataFrame, priced: set,
                     lo, hold_days: int = HOLD_DAYS,
                     instrument_col: str = "etf") -> pd.DataFrame:
    """Mechanically hold every signal for `hold_days` and tally the result.

    Entry = the instrument's price as of action_date; exit = as of
    action_date + hold_days; SELLs are counted short (sign flipped), so
    every number reads as 'money made by following the signal'.
    Rows: ALL / BUY only / SELL only, with trade count, avg per trade,
    hit rate, total P&L, per-trade Sharpe and an annualised Sharpe
    (x sqrt(252/hold_days) - the standard scaling)."""
    trades = []
    for _, row in sig.iterrows():
        instr = row.get(instrument_col)
        if instr not in priced:
            continue
        px = price_series(prices, instr, lo, None)
        if px.empty:
            continue
        p0 = px.asof(row["action_date"])
        p1 = px.asof(row["action_date"] + pd.Timedelta(days=hold_days))
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            continue
        sign = 1 if row["action"] == "BUY" else -1
        trades.append({"side": row["action"],
                       "ret": sign * (p1 / p0 - 1) * 100})

    out = []
    groups = {"ALL (buy+sell)": trades,
              "BUY only": [t for t in trades if t["side"] == "BUY"],
              "SELL only": [t for t in trades if t["side"] == "SELL"]}
    for name, g in groups.items():
        r = pd.Series([t["ret"] for t in g])
        if len(r) < 2:
            out.append({"strategy": name, "trades": len(r)})
            continue
        sharpe = r.mean() / r.std() if r.std() > 0 else float("nan")
        out.append({"strategy": name, "trades": len(r),
                    "avg/trade %": round(r.mean(), 2),
                    "hit rate %": round((r > 0).mean() * 100),
                    "total P&L %": round(r.sum(), 1),
                    "sharpe/trade": round(sharpe, 2),
                    "annualised": round(sharpe * (252 / hold_days) ** 0.5, 2)})
    return pd.DataFrame(out)


def trade_desk(sig: pd.DataFrame, prices: pd.DataFrame | None, priced: set,
               today, hold_days: int = HOLD_DAYS,
               instrument_col: str = "etf") -> pd.DataFrame:
    """The live ledger: one row per signal, MOST RECENT FIRST - entry
    price/date, the dated hold_days exit, OPEN/closed status, days left,
    and signed P&L so far (marked at min(exit, today))."""
    rows = []
    for _, r in sig.sort_values("action_date", ascending=False).iterrows():
        instr = r.get(instrument_col)
        entry_d = r["action_date"]
        exit_d = entry_d + pd.Timedelta(days=hold_days)
        row = {"signal date": entry_d.date(), "action": r["action"],
               "theme": r.get("theme", r.get("ticker", "")),
               "instrument": instr,
               "exit by": exit_d.date(),
               "status": "OPEN" if exit_d > today else "closed",
               "days left": max((exit_d - today).days, 0),
               "score": f"{r.get('score', '?')}/5",
               "conv z": round(float(r.get("conv_z", float("nan"))), 2)}
        if prices is not None and instr in priced:
            px = price_series(prices, instr,
                              entry_d - pd.Timedelta(days=5), None)
            p0 = px.asof(entry_d) if not px.empty else float("nan")
            mark_d = min(exit_d, today)
            p1 = px.asof(mark_d) if not px.empty else float("nan")
            if pd.notna(p0) and pd.notna(p1) and p0:
                sign = 1 if r["action"] == "BUY" else -1
                row["entry px"] = round(float(p0), 2)
                row["P&L so far %"] = round(sign * (p1 / p0 - 1) * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def certainty_table(sig: pd.DataFrame) -> pd.DataFrame:
    """The desk's ranking metric, exactly as the old dashboard defined it:
    certainty = score (breadth of evidence)
              + |conviction z| capped at 3 (strength)
              + a recency bonus fading linearly over 90 days
                (a live edge beats an old one)."""
    cert = sig.copy()
    cert["strength"] = cert["conv_z"].abs().clip(upper=3)
    age = (cert["action_date"].max() - cert["action_date"]).dt.days
    cert["recency"] = (1 - age / 90).clip(lower=0)
    cert["certainty"] = cert["score"] + cert["strength"] + cert["recency"]
    return cert.sort_values("certainty", ascending=False)
