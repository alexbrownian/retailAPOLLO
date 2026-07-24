"""
euphoria.py
===========
THE EUPHORIA DETECTOR - retailAPOLLO's core signal since the July-2026
re-aim: detect retail EUPHORIA and use it to call PRICE TOPS.

THE AIM (what "success" means, exactly)
---------------------------------------
For every instrument in the universe (theme anchor ETFs + hot single
names + retail commodities), raise a EUPHORIA ALERT that lands inside
[peak - 30 days, peak + 1 day] of a genuine price top - the closer to the
peak the better - while keeping false alarms rare. A "genuine top" is
price-defined (below), so the test is objective and walk-forward.

WHAT IS EUPHORIA? (the definition, in words)
--------------------------------------------
The state where the crowd has stopped analysing and started celebrating:
everyone is watching one name (attention at extremes), everyone has been
bullish for weeks (sustained one-way sentiment), new people keep arriving
(crowd influx), the price itself has been rising FASTER THAN EXPONENTIAL
(the bubble signature), and - at the very end - the mood starts to roll
over while the crowd is still at maximum size (the crowded-top fade).
Every one of those clauses is a measurable rule below; the composite
0-100 EUPHORIA LEVEL is their average, and an ALERT is a hard-threshold
crossing with prerequisites - nothing arbitrary, nothing discretionary.

RESEARCH GROUNDING (why these rules and not others)
---------------------------------------------------
* Attention extremes predict REVERSAL, not continuation: retail attention
  herding is followed by negative abnormal returns (Barber & Odean's
  attention-induced buying; WSB/GameStop discussion-board studies;
  "Dumb money: social network attention herding" 2025). -> rules E1/E3.
* One-sided sentiment at extremes is contrarian: aggregated bullishness
  peaks coincide with local price tops in Reddit/Twitter studies.
  -> rule E2 (persistence, not level alone: a single loud day is hype,
  weeks of one-way lean is euphoria).
* The PRICE signature of a bubble is super-exponential (faster-than-
  exponential) growth - the core of Sornette's LPPLS bubble/crash
  framework, applied to meme stocks in arXiv:2110.06190. We use an
  "LPPLS-lite" convexity measure: the quadratic coefficient of a rolling
  regression of log-price on time. Positive convexity = the log price is
  CURVING UPWARD = growth is accelerating beyond any steady exponential
  - unsustainable by construction. -> rule E5.
* The end-phase divergence (crowd maximal, mood fading) is the classic
  distribution pattern - it was already this project's "crowded top"
  flag, validated directionally in the July-2026 conviction study
  (post-peak sentiment fade preceded drawdowns). -> rule E4.

REDDIT-ONLY PREDICTION (desk rule, July 2026)
---------------------------------------------
Every PREDICTIVE input below is built from the Reddit-derived aggregates
(mention counts + scored sentiment). PRICE IS NEVER AN INPUT to the
euphoria level or the alert - price appears ONLY in the ground-truth
peak definition and the scoring, i.e. only to TEST the detector. This
keeps the claim clean: "the crowd alone called the top", not "the crowd
plus the chart called the top". (An earlier revision used a price-
convexity feature and a price-boom gate; both were removed under this
rule and replaced with their attention-space analogues below - the
ablation table quantifies what that costs/buys.)

THE RULES (all trailing - day t uses only data <= t; all thresholds are
percentile ranks against the SAME instrument's own trailing 365 days, so
"extreme" always means "extreme for this name", never absolute counts
that coverage shifts could fake)
------------------------------------------------------------------------
  E1  ATTENTION EXTREMITY   pct-rank of the 7d mention share
  E2  SUSTAINED BULLISHNESS pct-rank of the 28d mean net-bullish share,
                            gated by persistence: >=75% of the last 28
                            days net-bullish (rule: "super bullish AND
                            has been for a long time")
  E3  CROWD INFLUX          pct-rank of the 28d CHANGE in mention share
                            (the crowd is still arriving)
  E5  SUPER-EXPONENTIAL     pct-rank of positive log-ATTENTION convexity
      ATTENTION GROWTH      (rolling 60d quadratic fit of log mentions).
                            Sornette's super-exponential bubble signature
                            applied to the crowd instead of the price:
                            attention spreading is an epidemic process,
                            and when its growth rate is itself growing,
                            the contagion phase is terminal - it must
                            saturate (everyone who can arrive has), and
                            saturation of attention is where tops form.
  ----------------------------------------------------------------------
  EUPHORIA LEVEL = 100 * mean(E1, E2, E3, E5)          [0..100]
  ----------------------------------------------------------------------
  E4  FADE FLAG (the trigger sharpener, not part of the level): attention
      still >= its 90th percentile while the 14d change of the bullish
      share is NEGATIVE - the crowd is maximal but the mood is rolling
      over. An alert fires at a LOWER level when the fade is active,
      because the fade is the latest (and historically last) stage.

THE ALERT (the red line on the dashboard)
-----------------------------------------
On day t, EUPHORIA is DECLARED for an instrument when ALL hold:
  A0  coverage gate: >= EUPHORIA_MIN_COVERAGE scored posts in the last
      28 days. Percentile "extremes" computed on a handful of posts are
      noise wearing a costume - the thin-coverage era (2023-2025, before
      the backfill) generates floods of fake extremes without this gate.
      Same philosophy as MIN_TOTAL on the mention charts.
  A1  hype prerequisite: the 7d mention share >= HYPE_MULT x its own
      trailing 120d median ("something has to go euphoric first" -
      measured in the crowd, not the chart: the audience must have
      genuinely swollen, not just wobbled at a normal size)
  A2  attention gate: E1 >= 0.90 (you cannot be euphoric quietly)
  A2b sustained-bullishness gate: E2 > 0, i.e. at least 75% of the last
      28 days were net-bullish (the desk's rule 1 - "super bullish AND
      has been for a long time" - as a hard prerequisite, not just a
      score component)
  A3  level trigger:  EUPHORIA LEVEL >= threshold        (walk-forward)
      OR level >= threshold - FADE_DISCOUNT and E4 active
  A4  cooldown: >= 21 days since this instrument's last alert
The ONLY fitted quantity is the threshold in A3, and it is chosen
WALK-FORWARD: for each test year, the threshold is picked purely on the
years BEFORE it (maximising hits minus FA_PENALTY * false alarms), then
applied unchanged. Every other number is fixed a priori and documented
in src/config.py.

GROUND TRUTH (what counts as a top - price-only, so the test is honest)
-----------------------------------------------------------------------
A day P is a PEAK for an instrument when:
  G1  local maximum: close(P) is the highest close in [P-21d, P+21d]
  G2  it followed a boom: close(P) >= (1+BOOM_MIN) * min close over the
      preceding 120d   (BOOM_MIN: 25% ETFs/themes, 50% single names)
  G3  it was followed by a bust: drawdown from close(P) reaches at least
      CRASH_MIN within the next 90d (CRASH_MIN: 15% ETFs, 30% singles -
      single names are structurally more volatile, per the desk's call)
Peaks closer than 30d apart collapse to the higher one.

SCORING (the report card the walk-forward prints)
-------------------------------------------------
  peak capture   % of ground-truth peaks with >=1 alert in
                 [peak - 30d, peak + 1d]   <- the stated aim
  median lead    days from the capturing alert to the peak (positive =
                 early; the aim says within a month, closer the better)
  false alarms   alerts with NO qualifying peak within [alert, alert+45d]
                 (rate reported per instrument-year)
All reported per year AND per regime so one era cannot carry the signal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from src.config import (PROCESSED_DIR, EUPHORIA_MIN_COVERAGE,
                        EUPHORIA_ATT_GATE, EUPHORIA_BOOM_MIN_ETF,
                        EUPHORIA_BOOM_MIN_SINGLE, EUPHORIA_CRASH_MIN_ETF,
                        EUPHORIA_CRASH_MIN_SINGLE, EUPHORIA_COOLDOWN_DAYS,
                        EUPHORIA_FADE_DISCOUNT, EUPHORIA_FA_PENALTY,
                        EUPHORIA_PCT_WINDOW, EUPHORIA_MIN_HISTORY,
                        EUPHORIA_EXCLUDED_THEMES, EUPHORIA_SINGLE_TOP_N,
                        EUPHORIA_MIN_NAME_POSTS, EUPHORIA_HYPE_MULT)
from src.themes import THEME_ETFS, THEME_ETF_FALLBACKS
from analytics.loaders import (load, to_wide, THEME_COUNTS, THEME_SENT,
                               TICKER_COUNTS, TICKER_SENT)


# ---------------------------------------------------------------------------
# small building blocks
# ---------------------------------------------------------------------------
def trailing_pct_rank(s: pd.Series, window: int = EUPHORIA_PCT_WINDOW,
                      min_periods: int = EUPHORIA_MIN_HISTORY) -> pd.Series:
    """Where does today sit inside THIS name's own trailing `window` days?
    0 = the lowest seen lately, 1 = the highest. Percentiles (not z's)
    because euphoria features are heavily skewed - a z on a fat-tailed
    series over/under-reacts, a rank never does. Strictly trailing."""
    return s.rolling(window, min_periods=min_periods).rank(pct=True)


def log_convexity(s: pd.Series, window: int = 60) -> pd.Series:
    """LPPLS-lite: the quadratic coefficient of log(series) ~ a+b*t+c*t^2
    over a rolling window. c > 0 means the LOG of the series curves
    upward - growth is accelerating beyond any constant exponential rate,
    the mathematical signature of an unsustainable (self-reinforcing)
    process (Sornette). Under the Reddit-only rule this is applied to
    ATTENTION (log(1 + mentions)), not price: contagion whose growth
    rate is itself growing must saturate, and attention saturation is
    where tops form. Computed with a closed-form polyfit on a fixed
    design matrix (the window is constant, so the pseudo-inverse is
    built once)."""
    t = np.arange(window, dtype=float)
    t = (t - t.mean()) / window          # centred, scaled -> stable fit
    X = np.column_stack([np.ones(window), t, t * t])
    pinv = np.linalg.pinv(X)             # (3, window), built once
    logp = np.log(s.where(s > 0))

    def _c(win):
        if np.isnan(win).any():
            return np.nan
        return float(pinv[2] @ win)      # the quadratic coefficient

    return logp.rolling(window).apply(_c, raw=True)


# kept as an alias so older imports/tests keep working - same maths
log_price_convexity = log_convexity


# ---------------------------------------------------------------------------
# the instrument universe (equities + retail commodities ONLY)
# ---------------------------------------------------------------------------
def euphoria_themes() -> dict:
    """theme -> anchor symbol, excluding the non-equity/commodity themes
    the desk removed (rates_bonds, real_estate, ...). Commodities stay
    via their retail ETFs (gold GLD, silver in fallbacks, oil XLE/USO,
    uranium URA)."""
    return {t: sym for t, sym in THEME_ETFS.items()
            if t not in EUPHORIA_EXCLUDED_THEMES}


def single_name_universe(prices: pd.DataFrame,
                         top_n: int = EUPHORIA_SINGLE_TOP_N) -> list:
    """The hottest single names: top-N most-mentioned tickers that are
    priced and carry enough scored posts for sentiment to mean anything.
    Chosen from the data, not a hand list - today's NVDA is tomorrow's
    something else, and the whole point is catching the next one."""
    counts = load(TICKER_COUNTS)
    sent = load(TICKER_SENT)
    if counts is None or sent is None:
        return []
    priced = set(prices["symbol"].unique())
    posts_per = sent.groupby("ticker")["n_posts"].sum()
    ranked = (counts.groupby("ticker")["mention_count"].sum()
              .sort_values(ascending=False))
    out = []
    for tick in ranked.index:
        if tick in priced and posts_per.get(tick, 0) >= EUPHORIA_MIN_NAME_POSTS:
            out.append(tick)
        if len(out) >= top_n:
            break
    return out


def resolve_anchor(theme: str, priced: set):
    for sym in ([THEME_ETFS.get(theme)] if THEME_ETFS.get(theme) else []) \
            + THEME_ETF_FALLBACKS.get(theme, []):
        if sym in priced:
            return sym
    return None


# ---------------------------------------------------------------------------
# features -> euphoria level, per instrument
# ---------------------------------------------------------------------------
@dataclass
class EuphoriaSeries:
    """Everything the detector produces for ONE instrument."""
    name: str                  # theme name or ticker
    symbol: str                # the priced symbol behind it
    kind: str                  # "theme" | "single"
    level: pd.Series           # 0-100 euphoria level, daily
    e1: pd.Series = None       # attention extremity   (0-1)
    e2: pd.Series = None       # sustained bullishness (0-1)
    e3: pd.Series = None       # crowd influx          (0-1)
    e5: pd.Series = None       # super-exponential attention (0-1)
    fade: pd.Series = None     # E4 flag (bool)
    boom_ok: pd.Series = None  # A1 hype prerequisite (bool, Reddit-only)
    coverage_ok: pd.Series = None  # A0 gate (bool): enough scored posts
    alerts: list = field(default_factory=list)   # filled by detect_alerts


def _mention_share(counts_long, entity_col, name, all_days):
    """(7d-smoothed share of the day's total mentions in %,
        7d-smoothed raw mention count). The share powers E1/E3/A1 (it is
    coverage-shift-proof); the raw count powers E5's contagion fit."""
    day_tot = counts_long.groupby("date")["mention_count"].sum()
    m = (counts_long[counts_long[entity_col] == name]
         .groupby("date")["mention_count"].sum()
         .reindex(all_days).fillna(0.0))
    tot = day_tot.reindex(all_days).fillna(0.0)
    share = (m / tot.where(tot > 0)) * 100
    return (share.rolling(7, min_periods=1).mean(),
            m.rolling(7, min_periods=1).mean())


def _bullish_series(sent_long, entity_col, name, all_days):
    """(28d net-bullish share, persistence, 14d change, 28d post count)."""
    one = sent_long[sent_long[entity_col] == name]
    n = one.groupby("date")["n_posts"].sum().reindex(all_days).fillna(0.0)
    nb = one.groupby("date").apply(
        lambda g: (g["n_posts"] * g["net_bullish"]).sum(),
        include_groups=False).reindex(all_days).fillna(0.0)
    roll_n = n.rolling(28, min_periods=7).sum()
    share28 = nb.rolling(28, min_periods=7).sum() / roll_n.replace(0, np.nan)
    daily_share = nb / n.replace(0, np.nan)
    # persistence over days WITH posts: a zero-post day is "no evidence",
    # not "not bullish" - counting it against persistence would mute the
    # detector exactly when coverage thins
    has_posts = (n > 0).rolling(28, min_periods=7).sum()
    bull_days = (daily_share > 0).rolling(28, min_periods=7).sum()
    persist = bull_days / has_posts.replace(0, np.nan)
    share14 = (nb.rolling(14, min_periods=7).sum()
               / n.rolling(14, min_periods=7).sum().replace(0, np.nan))
    fade_chg = share14.diff(14)
    return share28, persist, fade_chg, roll_n


def compute_euphoria(name, symbol, kind, counts_long, sent_long,
                     entity_col) -> EuphoriaSeries | None:
    """Build the full euphoria series for one instrument - from the
    Reddit aggregates ONLY (no price input; price is for testing).
    Returns None when there is not enough data to say anything honest."""
    all_days = pd.date_range(counts_long["date"].min(),
                             counts_long["date"].max(), freq="D")

    share, m7 = _mention_share(counts_long, entity_col, name, all_days)
    share28, persist, fade_chg, posts28 = _bullish_series(
        sent_long, entity_col, name, all_days)

    e1 = trailing_pct_rank(share)
    # E2: rank of the 28d bullish level, GATED by persistence - weeks of
    # one-way lean, not one loud day (persistence gate multiplies to 0
    # when under 75% of recent days were net-bullish)
    e2 = trailing_pct_rank(share28) * (persist >= 0.75).astype(float)
    e3 = trailing_pct_rank(share.diff(28))
    # E5: super-exponential ATTENTION growth (LPPLS-lite on log mentions)
    conv = log_convexity(1.0 + m7)
    e5 = trailing_pct_rank(conv.clip(lower=0))     # only UPWARD curvature
    fade = (e1 >= EUPHORIA_ATT_GATE) & (fade_chg < 0)

    # A1 hype prerequisite - the crowd must have genuinely swollen: the
    # 7d share is at least HYPE_MULT x its own trailing 120d median
    # (median, not min: a min of ~zero would make the gate vacuous)
    base_share = share.rolling(120, min_periods=60).median()
    boom_ok = share >= EUPHORIA_HYPE_MULT * base_share.where(base_share > 0)

    coverage_ok = posts28 >= EUPHORIA_MIN_COVERAGE
    level = 100 * pd.concat([e1, e2, e3, e5], axis=1).mean(axis=1)
    return EuphoriaSeries(name=name, symbol=symbol, kind=kind, level=level,
                          e1=e1, e2=e2, e3=e3, e5=e5, fade=fade,
                          boom_ok=boom_ok.fillna(False),
                          coverage_ok=coverage_ok)


# ---------------------------------------------------------------------------
# alerts + ground truth + scoring
# ---------------------------------------------------------------------------
def detect_alerts(es: EuphoriaSeries, threshold: float,
                  cooldown: int = EUPHORIA_COOLDOWN_DAYS,
                  att_gate: bool = True, e2_gate: bool = True,
                  fade_on: bool = True, hype_gate: bool = True) -> list:
    """Apply the alert rule A1-A4 for one instrument at one threshold.
    Returns the alert dates. The keyword switches exist ONLY so the
    ablation study (below) can knock out one rule at a time and measure
    what it was contributing - live runs always use the defaults."""
    lvl, fade, boom, e1 = es.level, es.fade, es.boom_ok, es.e1
    gate = es.coverage_ok.copy()
    if hype_gate:
        gate = gate & boom
    if att_gate:
        gate = gate & (e1 >= EUPHORIA_ATT_GATE)
    if e2_gate:
        gate = gate & (es.e2 > 0)
    trigger = lvl >= threshold
    if fade_on:
        trigger = trigger | ((lvl >= threshold - EUPHORIA_FADE_DISCOUNT)
                             & fade)
    fire = gate & trigger
    alerts, last = [], None
    for d in lvl.index[fire.fillna(False)]:
        if last is None or (d - last).days >= cooldown:
            alerts.append(d)
            last = d
    return alerts


def ground_truth_peaks(px: pd.Series, kind: str) -> list:
    """The price-only definition of a top (G1-G3 in the module docstring).
    Returns the peak dates."""
    boom_min = (EUPHORIA_BOOM_MIN_SINGLE if kind == "single"
                else EUPHORIA_BOOM_MIN_ETF)
    crash_min = (EUPHORIA_CRASH_MIN_SINGLE if kind == "single"
                 else EUPHORIA_CRASH_MIN_ETF)
    px = px.dropna()
    if len(px) < 240:
        return []
    # G1 local max over +/-21d
    is_max = px == px.rolling(43, center=True, min_periods=22).max()
    cands = px.index[is_max.fillna(False)]
    peaks = []
    for d in cands:
        p = px.loc[d]
        prior = px.loc[d - pd.Timedelta(days=120):d]
        if len(prior) < 60 or p < (1 + boom_min) * prior.min():   # G2 boom
            continue
        after = px.loc[d:d + pd.Timedelta(days=90)]
        if len(after) < 5 or after.min() > (1 - crash_min) * p:   # G3 bust
            continue
        peaks.append(d)
    # collapse peaks closer than 30d (keep the higher close)
    out = []
    for d in peaks:
        if out and (d - out[-1]).days < 30:
            if px.loc[d] > px.loc[out[-1]]:
                out[-1] = d
            continue
        out.append(d)
    return out


def judgeable_window(px: pd.Series):
    """The date range where an alert can be honestly judged: price data
    must exist AT the alert (else no peak could even be defined there)
    and for 45 days AFTER it (else 'no peak followed' is not knowable
    yet - those alerts are PENDING, not false). Without this clip, every
    pre-price-history alert would be scored a false alarm by default,
    which is not evidence - it is missing data."""
    px = px.dropna()
    if px.empty:
        return None, None
    return px.index.min(), px.index.max() - pd.Timedelta(days=45)


def score_alerts(alerts: list, peaks: list):
    """The report card for one instrument: which peaks were captured (an
    alert inside [peak-30d, peak+1d]), with what lead, and which alerts
    were false (no qualifying peak within [alert, alert+45d])."""
    captured, leads = [], []
    for p in peaks:
        window = [a for a in alerts
                  if p - pd.Timedelta(days=30) <= a <= p + pd.Timedelta(days=1)]
        if window:
            captured.append(p)
            leads.append((p - max(window)).days)   # nearest alert's lead
    false = [a for a in alerts
             if not any(a <= p <= a + pd.Timedelta(days=45) for p in peaks)]
    return captured, leads, false


# ---------------------------------------------------------------------------
# the walk-forward
# ---------------------------------------------------------------------------
def build_all_series(prices: pd.DataFrame) -> list:
    """Every instrument's euphoria series: themes (via anchors) + the
    data-chosen single names."""
    theme_counts = load(THEME_COUNTS)
    theme_sent = load(THEME_SENT)
    tick_counts = load(TICKER_COUNTS)
    tick_sent = load(TICKER_SENT)
    for df in (theme_counts, theme_sent, tick_counts, tick_sent):
        if df is None:
            raise FileNotFoundError("aggregates missing - run the pipeline")
        df["date"] = pd.to_datetime(df["date"])
    pxmap = {s: g.sort_values("date").set_index("date")["px_last"]
             .asfreq("D").ffill() for s, g in prices.groupby("symbol")}
    priced = set(pxmap)

    out = []
    for theme in euphoria_themes():
        sym = resolve_anchor(theme, priced)
        if sym is None:
            continue
        es = compute_euphoria(theme, sym, "theme", theme_counts,
                              theme_sent, "theme")
        if es is not None:
            out.append(es)
    for tick in single_name_universe(prices):
        es = compute_euphoria(tick, tick, "single", tick_counts,
                              tick_sent, "ticker")
        if es is not None:
            out.append(es)
    return out, pxmap


def peak_maps(series: list, pxmap: dict):
    """(peaks_by, detectable_by) for a set of instruments.
    DETECTABLE peaks: those where the coverage gate was satisfied at any
    point in the alert window [peak-30d, peak]. A top that happened while
    there were a handful of posts/day is real for the DENOMINATOR of an
    "all peaks" rate, but no sentiment detector can honestly be scored on
    it - both rates are reported, neither is hidden."""
    peaks_by = {es.name: ground_truth_peaks(pxmap[es.symbol], es.kind)
                for es in series}
    detectable_by = {}
    for es in series:
        det = []
        for p in peaks_by[es.name]:
            win = es.coverage_ok.loc[p - pd.Timedelta(days=30):p]
            if len(win) and win.any():
                det.append(p)
        detectable_by[es.name] = det
    return peaks_by, detectable_by


def walk_forward(series: list, pxmap: dict,
                 thresholds=(50, 55, 60, 65, 70, 75, 80, 85),
                 fa_penalty: float = EUPHORIA_FA_PENALTY,
                 detect_kwargs: dict | None = None) -> dict:
    """The honest evaluation. For each test year Y: pick the threshold
    that maximises (hits - fa_penalty * false alarms) on the years BEFORE
    Y only, apply it to Y unchanged, tally. The rules never change; only
    the trigger level is learned, and never from the future.
    detect_kwargs is for the ablation study only (rule knock-outs)."""
    dk = detect_kwargs or {}
    peaks_by, detectable_by = peak_maps(series, pxmap)
    years = sorted({d.year for es in series
                    for d in es.level.dropna().index})
    years = [y for y in years if y >= min(years) + 1]   # need 1 train year

    def tally(year_set, thr):
        hits = fas = npeaks = ndet = 0
        leads = []
        for es in series:
            j0, j1 = judgeable_window(pxmap[es.symbol])
            alerts = [a for a in detect_alerts(es, thr, **dk)
                      if a.year in year_set
                      and j0 is not None and j0 <= a <= j1]
            peaks = [p for p in peaks_by[es.name] if p.year in year_set]
            det = [p for p in detectable_by[es.name] if p.year in year_set]
            cap, ld, fa = score_alerts(alerts, det)
            hits += len(cap); npeaks += len(peaks); ndet += len(det)
            fas += len(fa); leads += ld
        return hits, npeaks, fas, leads, ndet

    report = {"per_year": {}, "thresholds": {}}
    tot_hits = tot_peaks = tot_fas = 0
    all_leads = []
    for y in years:
        train = {yy for yy in years if yy < y} | {min(years) - 1}
        # do-no-harm default: with NO training evidence (young data, or
        # price history not yet covering the early years) every threshold
        # ties at utility 0 - the tie must go to the MOST conservative
        # trigger, not the loosest. Strict '>' below means a threshold
        # only loosens when past evidence actually paid for it.
        best_thr = thresholds[-1]
        best_u = -1e9
        for thr in reversed(thresholds):     # conservative first
            h, npk, fa, _, _ = tally(train, thr)
            u = h - fa_penalty * fa
            if u > best_u:
                best_u, best_thr = u, thr
        h, npk, fa, leads, ndet = tally({y}, best_thr)
        report["per_year"][y] = {"threshold": best_thr, "peaks": npk,
                                 "detectable": ndet,
                                 "captured": h, "false_alarms": fa,
                                 "leads": leads}
        report["thresholds"][y] = best_thr
        tot_hits += h; tot_peaks += npk; tot_fas += fa; all_leads += leads
        tot_det = report.setdefault("_det", 0) + ndet
        report["_det"] = tot_det
    n_iy = max(len(series) * len(years), 1)
    tot_det = report.pop("_det", 0)
    report["overall"] = {
        "peaks": tot_peaks, "detectable_peaks": tot_det,
        "captured": tot_hits,
        "capture_rate_all": round(tot_hits / tot_peaks, 3) if tot_peaks else None,
        "capture_rate_detectable":
            round(tot_hits / tot_det, 3) if tot_det else None,
        "false_alarms": tot_fas,
        "fa_per_instrument_year": round(tot_fas / n_iy, 2),
        "median_lead_days": float(np.median(all_leads)) if all_leads else None,
        "instruments": len(series), "years": years,
    }
    return report


# ---------------------------------------------------------------------------
# the ablation study (design ported from Chan's thesis, section 7.2.1:
# remove one ingredient at a time, re-run the FULL evaluation, and report
# how the headline metrics move - the honest way to show which rules earn
# their place and which are passengers)
# ---------------------------------------------------------------------------
def _level_without(es: EuphoriaSeries, drop: str) -> EuphoriaSeries:
    """A copy of one instrument's series with one feature removed from the
    LEVEL (the gates are ablated separately, via detect_alerts switches)."""
    feats = {"e1": es.e1, "e2": es.e2, "e3": es.e3, "e5": es.e5}
    kept = [v for k, v in feats.items() if k != drop]
    return replace(es, level=100 * pd.concat(kept, axis=1).mean(axis=1),
                   alerts=[])


def ablation(series: list, pxmap: dict) -> list:
    """Knock out each rule/feature, re-run the whole walk-forward, tabulate.
    Two caveats the thesis itself flags, which apply here too: (1) features
    are correlated, so single-feature drops UNDERSTATE the value of
    overlapping ingredients; (2) with few peaks, small deltas are noise -
    read the big movements, not the decimals."""
    variants = [
        ("FULL (all rules)", series, {}),
        ("- E1 attention (level)", [_level_without(es, "e1") for es in series], {}),
        ("- E2 sustained bull (level)", [_level_without(es, "e2") for es in series], {}),
        ("- E3 crowd influx (level)", [_level_without(es, "e3") for es in series], {}),
        ("- E5 super-exp attention (level)", [_level_without(es, "e5") for es in series], {}),
        ("- hype gate A1", series, {"hype_gate": False}),
        ("- attention gate A2", series, {"att_gate": False}),
        ("- sustained-bull gate A2b", series, {"e2_gate": False}),
        ("- fade trigger E4", series, {"fade_on": False}),
    ]
    rows = []
    for label, ser, dk in variants:
        o = walk_forward(ser, pxmap, detect_kwargs=dk)["overall"]
        rows.append({"variant": label,
                     "capture_detectable": o["capture_rate_detectable"],
                     "captured": o["captured"],
                     "false_alarms": o["false_alarms"],
                     "fa_per_iy": o["fa_per_instrument_year"],
                     "median_lead": o["median_lead_days"]})
    base = rows[0]
    for r in rows:
        cap0 = base["capture_detectable"] or 0
        cap = r["capture_detectable"] or 0
        r["d_capture"] = round(cap - cap0, 3)
        r["d_fa"] = r["false_alarms"] - base["false_alarms"]
    return rows


# ---------------------------------------------------------------------------
# the ML challenger (thesis chapter 6, scaled to our data): can a learned
# model beat the hand-written alert rule on the SAME features and the SAME
# walk-forward discipline? The thesis compared feature-only baselines vs
# graph models; here the comparison is hand-rules vs a logistic regression
# - deliberately the simplest possible learner, because with ~70 positive
# events, anything bigger memorises the past instead of learning from it
# (the thesis's own GNNs only reached ~12% precision on 133 positives -
# a warning against model appetite exceeding label supply).
# ---------------------------------------------------------------------------
ML_FEATURES = ["e1", "e2", "e3", "e5", "fade"]


def _ml_frame(series: list, detectable_by: dict,
              pxmap: dict) -> pd.DataFrame:
    """One row per (instrument, candidate day). Candidate days are the
    days that pass the NON-fitted prerequisites (hype + coverage) - the ML
    replaces only the fitted part (the trigger), so the comparison with
    the rules is apples-to-apples. Rows outside the judgeable price
    window are dropped (their labels would be missing data, not truth).
    Label y=1 when an alert on this day would capture a detectable peak
    (peak within [day-1, day+30])."""
    frames = []
    for es in series:
        ok = (es.boom_ok & es.coverage_ok).fillna(False)
        df = pd.DataFrame({"e1": es.e1, "e2": es.e2, "e3": es.e3,
                           "e5": es.e5,
                           "fade": es.fade.astype(float)})[ok]
        df = df.dropna()
        j0, j1 = judgeable_window(pxmap[es.symbol])
        if j0 is None:
            continue
        df = df[(df.index >= j0) & (df.index <= j1)]
        if df.empty:
            continue
        peaks = detectable_by[es.name]
        y = np.zeros(len(df))
        for i, d in enumerate(df.index):
            if any(d - pd.Timedelta(days=1) <= p <= d + pd.Timedelta(days=30)
                   for p in peaks):
                y[i] = 1
        df["y"] = y
        df["name"] = es.name
        df["year"] = df.index.year
        frames.append(df.reset_index(names="date"))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=ML_FEATURES + ["y", "name", "year",
                                                     "date"]))


def _prob_alerts(dates, probs, thr, cooldown=EUPHORIA_COOLDOWN_DAYS):
    """Turn a probability series into alert dates: fire on prob >= thr,
    then apply the same cooldown the rules use."""
    alerts, last = [], None
    for d, p in zip(dates, probs):
        if p >= thr and (last is None or (d - last).days >= cooldown):
            alerts.append(d)
            last = d
    return alerts


def ml_walk_forward(series: list, pxmap: dict,
                    fa_penalty: float = EUPHORIA_FA_PENALTY) -> dict:
    """Same discipline as walk_forward, but the trigger is a logistic
    regression fitted on PAST years only (features -> P(peak within 30d)),
    with its probability cut-off chosen on the SAME past years by the SAME
    utility (hits - fa_penalty * false alarms). Nothing sees the future."""
    from sklearn.linear_model import LogisticRegression

    peaks_by, detectable_by = peak_maps(series, pxmap)
    frame = _ml_frame(series, detectable_by, pxmap)
    if frame.empty:
        return {"error": "no candidate days"}
    years = sorted(int(y) for y in frame["year"].unique())
    years = [y for y in years if y > min(years)]

    def tally(sub, thr, year_set):
        hits = fas = ndet = 0
        leads = []
        for es in series:
            g = sub[sub["name"] == es.name].sort_values("date")
            alerts = _prob_alerts(g["date"].tolist(), g["prob"].tolist(), thr)
            det = [p for p in detectable_by[es.name] if p.year in year_set]
            cap, ld, fa = score_alerts(alerts, det)
            hits += len(cap); ndet += len(det); fas += len(fa); leads += ld
        return hits, fas, leads, ndet

    report = {"per_year": {}}
    tot_hits = tot_fas = tot_det = 0
    all_leads = []
    coefs = None
    for y in years:
        train = frame[frame["year"] < y]
        test = frame[frame["year"] == y]
        if train["y"].sum() < 3 or test.empty:
            report["per_year"][y] = {"skipped": "under 3 training peaks"}
            continue
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(train[ML_FEATURES], train["y"])
        train = train.assign(prob=model.predict_proba(
            train[ML_FEATURES])[:, 1])
        test = test.assign(prob=model.predict_proba(test[ML_FEATURES])[:, 1])
        # cut-off from the TRAIN years only, same utility as the rules
        grid = np.unique(np.percentile(train["prob"], range(50, 100, 5)))
        best_thr, best_u = grid[-1], -1e9    # conservative default, as above
        train_years = set(train["year"].unique())
        for thr in grid[::-1]:               # conservative first
            h, fa, _, _ = tally(train, thr, train_years)
            u = h - fa_penalty * fa
            if u > best_u:
                best_u, best_thr = u, thr
        h, fa, leads, ndet = tally(test, best_thr, {y})
        report["per_year"][y] = {"prob_threshold": round(float(best_thr), 3),
                                 "detectable": ndet, "captured": h,
                                 "false_alarms": fa}
        tot_hits += h; tot_fas += fa; tot_det += ndet; all_leads += leads
        coefs = {f: float(round(c, 3))
                 for f, c in zip(ML_FEATURES, model.coef_[0])}
    n_iy = max(len(series) * len(years), 1)
    report["overall"] = {
        "detectable_peaks": tot_det, "captured": tot_hits,
        "capture_rate_detectable":
            round(tot_hits / tot_det, 3) if tot_det else None,
        "false_alarms": tot_fas,
        "fa_per_instrument_year": round(tot_fas / n_iy, 2),
        "median_lead_days": float(np.median(all_leads)) if all_leads else None,
        "last_model_coefficients": coefs,
    }
    return report


def _stored_report() -> dict | None:
    import json
    path = os.path.join(PROCESSED_DIR, "euphoria_report.json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except (ValueError, OSError):
        return None


def needs_research(stored: dict | None, data_max_year: int) -> bool:
    """When is a full research pass (walk-forward + ablation + ML)
    actually REQUIRED, rather than a frozen threshold being enough?

    The walk-forward trains each year's threshold on STRICTLY EARLIER
    years, so within a calendar year the daily recompute is provably a
    no-op - today's data is not in any threshold's training set. The
    threshold can only legitimately change when (a) no report exists yet,
    or (b) the data has rolled into a year the stored thresholds do not
    cover. Everything else (backfills, rule changes) is an explicit
    --research run - a research decision, not a side effect of a pull.
    (Desk decision, 2026-07-24: research runs once, deliberately; live
    runs score.)"""
    if not stored or not stored.get("thresholds"):
        return True
    return data_max_year > max(int(y) for y in stored["thresholds"])


def main(research: bool | None = None):
    """CLI: build the euphoria series and save daily levels + alerts.

    research=None (the pipeline default): auto - run the FULL validation
      (walk-forward + ablation + ML challenger) only when needs_research
      says a frozen threshold cannot be trusted; otherwise score at the
      stored threshold in seconds.
    research=True (run_analytics --research / the notebooks): always run
      the full validation and refresh euphoria_report.json.
    """
    from src.config import PRICES_PATH
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    series, pxmap = build_all_series(prices)
    print(f"euphoria universe: {len(series)} instruments "
          f"({sum(1 for s in series if s.kind == 'theme')} themes, "
          f"{sum(1 for s in series if s.kind == 'single')} single names)")

    stored = _stored_report()
    data_max_year = int(max(es.level.dropna().index.max()
                            for es in series).year)
    if research is None:
        research = needs_research(stored, data_max_year)
        if research and stored:
            print("  research pass auto-triggered: data covers "
                  f"{data_max_year}, stored thresholds stop at "
                  f"{max(stored['thresholds'])}")

    if not research:
        # LIVE FAST PATH: the frozen threshold, today's data, seconds.
        # The report json (walk-forward record, ablation, ML verdict) is
        # deliberately untouched - those are research artifacts refreshed
        # by research runs, and the dashboard states their as-of years.
        thr_now = stored["thresholds"][max(stored["thresholds"])]
        rows = []
        for es in series:
            alerts = set(detect_alerts(es, thr_now))
            df = pd.DataFrame({"date": es.level.index, "name": es.name,
                               "symbol": es.symbol, "kind": es.kind,
                               "level": es.level.values,
                               # components + gate states, so the
                               # dashboard can EXPLAIN every alert in
                               # plain English (all text-free floats)
                               "e1": es.e1.values, "e2": es.e2.values,
                               "e3": es.e3.values, "e5": es.e5.values,
                               "fade": es.fade.reindex(
                                   es.level.index).fillna(False).values,
                               "hype_ok": es.boom_ok.reindex(
                                   es.level.index).fillna(False).values})
            df["alert"] = df["date"].isin(alerts)
            rows.append(df.dropna(subset=["level"]))
        out = pd.concat(rows, ignore_index=True)
        from src.abstracted_data import _safe_write
        _safe_write(out, os.path.join(PROCESSED_DIR,
                                      "euphoria_levels.parquet"))
        print(f"  LIVE mode: scored at frozen threshold {thr_now} "
              f"(walk-forward convention: intra-year recompute is a "
              f"no-op) -> euphoria_levels.parquet ({len(out):,} rows). "
              "Full validation: run_analytics --what euphoria --research")
        return 0

    report = walk_forward(series, pxmap)
    o = report["overall"]
    print(f"\nWALK-FORWARD ({o['years'][0]}-{o['years'][-1]}):")
    print(f"  peaks defined     : {o['peaks']} "
          f"(detectable with coverage: {o['detectable_peaks']})")
    print(f"  captured in window: {o['captured']}  "
          f"(rate: {o['capture_rate_detectable']} of detectable, "
          f"{o['capture_rate_all']} of all)")
    print(f"  median lead       : {o['median_lead_days']} days before the peak")
    print(f"  false alarms      : {o['false_alarms']} "
          f"({o['fa_per_instrument_year']}/instrument-year)")
    for y, r in report["per_year"].items():
        print(f"  {y}: thr {r['threshold']} | peaks {r['peaks']} "
              f"(det {r['detectable']}) | captured {r['captured']} "
              f"| FAs {r['false_alarms']}")

    # --- ablation study (thesis 7.2.1 style) ---
    print("\nABLATION (drop one rule, re-run the whole walk-forward):")
    abl = ablation(series, pxmap)
    for r in abl:
        print(f"  {r['variant']:<30} capture {r['capture_detectable']}"
              f" ({r['d_capture']:+.3f}) | FAs {r['false_alarms']}"
              f" ({r['d_fa']:+d}) | lead {r['median_lead']}")

    # --- the ML challenger, same features + same discipline ---
    print("\nML CHALLENGER (logistic regression, walk-forward):")
    ml = ml_walk_forward(series, pxmap)
    mo = ml.get("overall", {})
    if mo:
        print(f"  capture {mo['capture_rate_detectable']} of detectable | "
              f"FAs {mo['false_alarms']} "
              f"({mo['fa_per_instrument_year']}/instr-yr) | "
              f"lead {mo['median_lead_days']}")
        print(f"  coefficients: {mo['last_model_coefficients']}")
    # THE ADOPTION RULE (stated before looking at the numbers, so the
    # choice is a criterion, not a preference): the ML replaces the rules
    # only if, on the SAME test years (the rules get one extra early test
    # year the ML cannot be scored on - excluded for fairness), it wins on
    # (a) total utility (hits - penalty*FAs), AND (b) captures at least
    # as many peaks in the MOST RECENT year, AND (c) captures at least as
    # many peaks in total. (b) exists because the newest regime is the one
    # the desk actually trades; (c) exists because a near-silent model can
    # "win" on utility purely by never firing - and a top detector that
    # never fires is not a better top detector.
    ml_years = [y for y, r in ml.get("per_year", {}).items()
                if "skipped" not in r]
    ru = mu = 0.0
    rules_recent = ml_recent = rules_cap = ml_cap = 0
    if ml_years:
        recent = max(ml_years)
        for y in ml_years:
            rr = report["per_year"].get(y, {})
            mr = ml["per_year"][y]
            ru += (rr.get("captured", 0)
                   - EUPHORIA_FA_PENALTY * rr.get("false_alarms", 0))
            mu += (mr["captured"] - EUPHORIA_FA_PENALTY * mr["false_alarms"])
            rules_cap += rr.get("captured", 0)
            ml_cap += mr["captured"]
            if y == recent:
                rules_recent = rr.get("captured", 0)
                ml_recent = mr["captured"]
    adopted = ("ml" if ml_years and mu > ru and ml_recent >= rules_recent
               and ml_cap >= rules_cap
               else "rules")
    matched = {"years": ml_years, "rules_utility": ru, "ml_utility": mu,
               "rules_captured": rules_cap, "ml_captured": ml_cap,
               "rules_recent_captured": rules_recent,
               "ml_recent_captured": ml_recent}
    print(f"  matched years {ml_years}: utility rules {ru:.1f} vs ml {mu:.1f}"
          f" | capture rules {rules_cap} vs ml {ml_cap}"
          f" | recent-year rules {rules_recent} vs ml {ml_recent}")
    print(f"  verdict: {'ML ADOPTED' if adopted == 'ml' else 'rules kept'}")

    # persist for the dashboard: daily levels + the final-year threshold
    rows = []
    thr_now = report["thresholds"][max(report["thresholds"])]
    for es in series:
        alerts = set(detect_alerts(es, thr_now))
        df = pd.DataFrame({"date": es.level.index, "name": es.name,
                           "symbol": es.symbol, "kind": es.kind,
                           "level": es.level.values,
                           "e1": es.e1.values, "e2": es.e2.values,
                           "e3": es.e3.values, "e5": es.e5.values,
                           "fade": es.fade.reindex(
                               es.level.index).fillna(False).values,
                           "hype_ok": es.boom_ok.reindex(
                               es.level.index).fillna(False).values})
        df["alert"] = df["date"].isin(alerts)
        rows.append(df.dropna(subset=["level"]))
    out = pd.concat(rows, ignore_index=True)
    from src.abstracted_data import _safe_write
    _safe_write(out, os.path.join(PROCESSED_DIR, "euphoria_levels.parquet"))
    import json
    with open(os.path.join(PROCESSED_DIR, "euphoria_report.json"), "w") as f:
        json.dump({k: v for k, v in report.items() if k != "per_year"}
                  | {"per_year": {str(y): {kk: vv for kk, vv in r.items()
                                           if kk != "leads"}
                                  for y, r in report["per_year"].items()},
                     "ablation": abl,
                     "ml_test": {"overall": mo,
                                 "per_year": {str(y): r for y, r in
                                              ml.get("per_year", {}).items()},
                                 "matched_comparison": matched},
                     "adopted": adopted},
                  f, indent=1)
    print(f"\nsaved euphoria_levels.parquet ({len(out):,} rows) "
          f"+ euphoria_report.json (live threshold {thr_now})")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
