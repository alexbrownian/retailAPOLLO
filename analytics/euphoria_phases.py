"""
euphoria_phases.py
==================
Ground truth + feature bank for the euphoria PHASES study (July 2026):
detect not just the END of retail euphoria (the existing top detector in
analytics/euphoria.py) but also its START - the onset of a GME-scale
rally - from the crowd alone.

WHAT THIS MODULE OWNS
---------------------
1. EPISODES - the price-defined ground truth. An episode is the full
   boom-bust arc trough -> peak -> bust, built from the SAME peak
   definition the top detector is scored on (G1-G3 in euphoria.py), so
   the two detectors share one notion of "a genuine euphoria event".
2. The ONSET FEATURE BANK - six trailing, crowd-only candidate features
   aimed at the LEFT side of an episode (the crowd arriving), evaluated
   in notebook 02 with the thesis-style importance battery.
3. The LABELLED DAY FRAME - one row per (instrument, candidate day) with
   features + onset/top labels, shared by notebooks 02 and 03 and by the
   production detector, so research and dashboard can never drift apart.

DESK DECISIONS (July 2026, recorded before any result was computed)
-------------------------------------------------------------------
* ONSET HIT WINDOW: an onset alert is a HIT when it lands inside
  [trough, min(trough + 45d, peak)]. The trough is the 120d low the boom
  is measured from (G2), so "the start" is anchored to the same low the
  peak definition already uses - no new fitted quantity. The window is
  CAPPED AT THE PEAK: for fast rallies an uncapped +45d would let an
  alert fired AFTER the top count as "caught the start".
* LATE is not FALSE: an onset alert inside (window end, peak] fired
  during the rally but after its start. It is reported as LATE -
  separately from hits AND from false alarms - because calling it a hit
  inflates the onset claim and calling it false punishes an alert that
  was inside a genuine episode. Only alerts outside the whole episode
  count as false alarms.
* CROWD-ONLY PREDICTION (unchanged hard rule): price never enters any
  feature or alert below - price appears ONLY here, in the ground-truth
  episode definition and the scoring.

All feature rules are trailing (day t uses only data <= t) and are
percentile ranks against the SAME instrument's own trailing history, for
the same reasons as E1-E5 (fat tails -> ranks, coverage shifts -> shares).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (EUPHORIA_CRASH_MIN_ETF, EUPHORIA_CRASH_MIN_SINGLE,
                        EUPHORIA_PCT_WINDOW, EUPHORIA_MIN_HISTORY)
from analytics.euphoria import (EuphoriaSeries, ground_truth_peaks,
                                judgeable_window, trailing_pct_rank,
                                log_convexity, _mention_share,
                                _bullish_series)
from analytics.loaders import load, TICKER_COUNTS_BY_SOURCE

# Desk decision (July 2026): the onset hit window length. Anchored to the
# trough that G2 already defines, capped at the peak (see module docstring).
ONSET_WINDOW_DAYS = 45

# The window in which a peak "answers" a TOP alert - identical to the
# stated aim in euphoria.py ([peak-30d, peak+1d]), repeated here so the
# labelled frame can be built without re-deriving it.
TOP_LEAD_DAYS = 30


# ---------------------------------------------------------------------------
# 1. EPISODES - the price-defined ground truth (trough -> peak -> bust)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    """One complete boom-bust arc for one instrument. Frozen: ground
    truth is evidence, never something downstream code may edit."""
    name: str                      # theme name or ticker
    symbol: str                    # priced symbol behind it
    kind: str                      # "theme" | "single"
    trough: pd.Timestamp           # t_start: the 120d low the boom grew from
    peak: pd.Timestamp             # the confirmed top (G1-G3)
    bust_date: pd.Timestamp | None  # first close down >= crash_min from peak
    boom_pct: float                # trough -> peak gain (e.g. 0.5 = +50%)
    bust_pct: float                # peak -> 90d-min drawdown (negative)
    run_days: int                  # trough -> peak, calendar days
    onset_lo: pd.Timestamp         # = trough
    onset_hi: pd.Timestamp         # = min(trough + ONSET_WINDOW_DAYS, peak)


def find_episodes(px: pd.Series, name: str, symbol: str,
                  kind: str) -> list[Episode]:
    """Extend each confirmed peak (the existing G1-G3 definition) into a
    full episode by walking BACKWARD to its trough and FORWARD to its
    bust. Nothing here is new ground truth - the trough is exactly the
    'min close over the preceding 120d' that G2 already measures the
    boom against, and the bust date is the first day G3's drawdown
    condition is met."""
    crash_min = (EUPHORIA_CRASH_MIN_SINGLE if kind == "single"
                 else EUPHORIA_CRASH_MIN_ETF)
    px = px.dropna()
    episodes = []
    for peak in ground_truth_peaks(px, kind):
        prior = px.loc[peak - pd.Timedelta(days=120):peak]
        trough = prior.idxmin()
        peak_close = px.loc[peak]
        after = px.loc[peak:peak + pd.Timedelta(days=90)]
        drawdown = after / peak_close - 1.0
        busted = drawdown[drawdown <= -crash_min]
        episodes.append(Episode(
            name=name, symbol=symbol, kind=kind,
            trough=trough, peak=peak,
            bust_date=busted.index[0] if len(busted) else None,
            boom_pct=float(peak_close / prior.min() - 1.0),
            bust_pct=float(drawdown.min()),
            run_days=int((peak - trough).days),
            onset_lo=trough,
            onset_hi=min(trough + pd.Timedelta(days=ONSET_WINDOW_DAYS), peak),
        ))
    return episodes


def episode_catalog(series: list, pxmap: dict) -> pd.DataFrame:
    """Every episode for every instrument, as one flat table, with the
    two detectability flags the report always shows side by side:

    onset_detectable : the coverage gate (A0) held on >= 1 day of the
                       onset window - a start no crowd was measurable
                       for stays in the "all" denominator but cannot
                       honestly score a crowd detector;
    top_detectable   : same flag over [peak-30d, peak] (the existing
                       definition in euphoria.peak_maps)."""
    rows = []
    for es in series:
        for ep in find_episodes(pxmap[es.symbol], es.name, es.symbol, es.kind):
            onset_cov = es.coverage_ok.loc[ep.onset_lo:ep.onset_hi]
            top_cov = es.coverage_ok.loc[ep.peak - pd.Timedelta(days=30):ep.peak]
            rows.append({
                "name": ep.name, "symbol": ep.symbol, "kind": ep.kind,
                "trough": ep.trough, "peak": ep.peak,
                "bust_date": ep.bust_date,
                "boom_pct": ep.boom_pct, "bust_pct": ep.bust_pct,
                "run_days": ep.run_days,
                "onset_lo": ep.onset_lo, "onset_hi": ep.onset_hi,
                "year": int(ep.peak.year),
                "onset_detectable": bool(len(onset_cov) and onset_cov.any()),
                "top_detectable": bool(len(top_cov) and top_cov.any()),
            })
    cols = ["name", "symbol", "kind", "trough", "peak", "bust_date",
            "boom_pct", "bust_pct", "run_days", "onset_lo", "onset_hi",
            "year", "onset_detectable", "top_detectable"]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(["peak", "name"]).reset_index(drop=True))


# ---------------------------------------------------------------------------
# 2. THE ONSET FEATURE BANK - six crowd-only candidates for "the start"
# ---------------------------------------------------------------------------
# Names + one-line definitions (all trailing pct-ranks vs own history):
#   O1 attention_accel   7d mention share minus 28d share - the crowd is
#                        arriving faster than its own monthly norm
#   O2 hype_ratio        7d share / own trailing 120d median share - the
#                        continuous version of the A1 hype gate
#   O3 bull_inflection   14d change of the 14d net-bullish share - the
#                        mood is TURNING up (vs E2's "has been up")
#   O4 influx_speed      14d change of the mention share - E3's crowd
#                        influx measured at twice the speed
#   O5 attention_convexity  E5's super-exponential growth signature -
#                        inherently an EARLY-phase feature (contagion)
#   O6 source_breadth    how many of the 4 sources mentioned the name in
#                        the last 7d - a real crowd spreads across
#                        platforms, a single loud thread does not
ONSET_FEATURES = ["attention_accel", "hype_ratio", "bull_inflection",
                  "influx_speed", "attention_convexity", "source_breadth"]

# The LOCKED production bank (notebook 02's verdict): source_breadth is
# EXCLUDED - its apparent skill was a coverage-regime artifact (X and
# StockTwits exist in the archive only from 2026, so "breadth" mostly
# encoded "which year is it"). Notebooks 03/04 and the live detector all
# import THIS list - research and production cannot drift.
ONSET_BANK = ["attention_accel", "hype_ratio", "bull_inflection",
              "influx_speed", "attention_convexity"]

# The incumbent bank, for side-by-side evaluation in the notebooks.
TOP_FEATURES = ["e1", "e2", "e3", "e5", "fade"]


def _source_breadth(name: str, all_days: pd.DatetimeIndex) -> pd.Series | None:
    """O6: count of distinct sources mentioning the name in the trailing
    7d. Only tickers have a by-source aggregate; for themes this returns
    None and the feature is simply absent (the frame builder drops it)."""
    by_src = load(TICKER_COUNTS_BY_SOURCE)
    if by_src is None or "source" not in by_src.columns:
        return None
    one = by_src[by_src["ticker"] == name]
    if one.empty:
        return None
    one = one.assign(date=pd.to_datetime(one["date"]))
    daily = (one.groupby(["date", "source"])["mention_count"].sum()
             .unstack(fill_value=0).reindex(all_days).fillna(0.0))
    return (daily.rolling(7, min_periods=1).sum() > 0).sum(axis=1).astype(float)


def compute_onset_features(name: str, counts_long: pd.DataFrame,
                           sent_long: pd.DataFrame, entity_col: str,
                           with_breadth: bool = True) -> pd.DataFrame:
    """The onset feature bank for one instrument - crowd aggregates ONLY
    (no price argument exists, by design; enforced by a unit test).
    Returns a daily DataFrame with one column per available feature."""
    all_days = pd.date_range(counts_long["date"].min(),
                             counts_long["date"].max(), freq="D")
    share, m7 = _mention_share(counts_long, entity_col, name, all_days)

    # O1: is this week's crowd bigger than this month's? (both in share
    # space, so a platform-wide busy day cancels out)
    share28 = share.rolling(28, min_periods=7).mean()
    o1 = trailing_pct_rank(share - share28)

    # O2: the hype gate's ratio as a continuous feature - how many times
    # its own 120d median the current 7d share is
    base = share.rolling(120, min_periods=60).median()
    o2 = trailing_pct_rank(share / base.where(base > 0))

    # O3: the mood turning up - 14d change of the 14d net-bullish share
    one = sent_long[sent_long[entity_col] == name]
    n = one.groupby("date")["n_posts"].sum().reindex(all_days).fillna(0.0)
    nb = one.groupby("date").apply(
        lambda g: (g["n_posts"] * g["net_bullish"]).sum(),
        include_groups=False).reindex(all_days).fillna(0.0)
    share14 = (nb.rolling(14, min_periods=7).sum()
               / n.rolling(14, min_periods=7).sum().replace(0, np.nan))
    o3 = trailing_pct_rank(share14.diff(14))

    # O4: E3 at double speed - the crowd influx over 14d, not 28d
    o4 = trailing_pct_rank(share.diff(14))

    # O5: the super-exponential attention signature (shared with E5 -
    # contagion accelerating is an EARLY-phase phenomenon)
    o5 = trailing_pct_rank(log_convexity(1.0 + m7).clip(lower=0))

    out = pd.DataFrame({"attention_accel": o1, "hype_ratio": o2,
                        "bull_inflection": o3, "influx_speed": o4,
                        "attention_convexity": o5,
                        # NOT a bank feature - the RAW hype ratio, kept so
                        # the onset prerequisite gate (share above its own
                        # 120d median, multiplier 1 = parameter-free) can
                        # be applied identically to every model
                        "hype_raw": share / base.where(base > 0)})
    if with_breadth and entity_col == "ticker":
        breadth = _source_breadth(name, all_days)
        if breadth is not None:
            out["source_breadth"] = trailing_pct_rank(breadth)
    return out


# ---------------------------------------------------------------------------
# 3. THE LABELLED DAY FRAME - the one table notebooks 02/03 both stand on
# ---------------------------------------------------------------------------
def label_days(index: pd.DatetimeIndex, episodes: pd.DataFrame,
               name: str) -> pd.DataFrame:
    """Three {0,1} labels for every day of one instrument's series:

    y_onset : day inside an episode's onset window [trough, capped end]
    y_late  : day inside (onset end, peak] - in the rally, past its start
    y_top   : day inside [peak - TOP_LEAD_DAYS, peak + 1d] - the existing
              aim window of the top detector
    """
    eps = episodes[episodes["name"] == name]
    y_onset = pd.Series(0, index=index)
    y_late = pd.Series(0, index=index)
    y_top = pd.Series(0, index=index)
    for ep in eps.itertuples():
        y_onset.loc[ep.onset_lo:ep.onset_hi] = 1
        if ep.onset_hi < ep.peak:
            y_late.loc[ep.onset_hi + pd.Timedelta(days=1):ep.peak] = 1
        y_top.loc[ep.peak - pd.Timedelta(days=TOP_LEAD_DAYS):
                  ep.peak + pd.Timedelta(days=1)] = 1
    return pd.DataFrame({"y_onset": y_onset, "y_late": y_late,
                         "y_top": y_top})


def build_day_frame(series: list, pxmap: dict,
                    episodes: pd.DataFrame,
                    counts: dict, sents: dict,
                    clip_judgeable: bool = True) -> pd.DataFrame:
    """One row per (instrument, candidate day): every onset feature, every
    incumbent feature, and the three labels. Candidate days are the days
    the detector is even allowed to speak on - coverage gate satisfied
    AND inside the judgeable price window (an unjudgeable label is
    missing data, not truth). counts/sents map entity_col -> long frame,
    e.g. {"theme": theme_counts, "ticker": tick_counts}.

    clip_judgeable=False keeps the most recent ~45 days (whose labels
    cannot be judged yet - alerts there are PENDING, not false): research
    must clip them, but the LIVE dashboard must score them."""
    frames = []
    for es in series:
        entity_col = "theme" if es.kind == "theme" else "ticker"
        onset = compute_onset_features(es.name, counts[entity_col],
                                       sents[entity_col], entity_col)
        top = pd.DataFrame({"e1": es.e1, "e2": es.e2, "e3": es.e3,
                            "e5": es.e5, "fade": es.fade.astype(float)})
        df = onset.join(top)
        labels = label_days(df.index, episodes, es.name)
        df = df.join(labels)
        # candidate-day mask: measurable (A0) and judgeable (price truth)
        j0, j1 = judgeable_window(pxmap[es.symbol])
        if j0 is None:
            continue
        ok = es.coverage_ok.reindex(df.index).fillna(False)
        mask = ok & (df.index >= j0)
        if clip_judgeable:
            mask = mask & (df.index <= j1)
        df = df[mask]
        # features must exist (percentiles need EUPHORIA_MIN_HISTORY days)
        df = df.dropna(subset=[c for c in ONSET_FEATURES + TOP_FEATURES
                               if c in df.columns
                               and c not in ("source_breadth", "hype_raw")])
        if df.empty:
            continue
        df = df.assign(name=es.name, kind=es.kind, year=df.index.year,
                       hype_ok=es.boom_ok.reindex(df.index).fillna(False))
        frames.append(df.reset_index(names="date"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. TOURNAMENT MACHINERY - walk-forward model comparison (notebook 03),
#    written here so the production detector runs the WINNING code path,
#    not a re-implementation of it.
# ---------------------------------------------------------------------------
from src.config import EUPHORIA_COOLDOWN_DAYS  # noqa: E402  (single source)


def alerts_from_scores(dates: list, scores: list, threshold: float,
                       cooldown: int = EUPHORIA_COOLDOWN_DAYS) -> list:
    """Scores -> sparse alert dates: fire on score >= threshold, then
    apply the standard cooldown (same rule as the top detector's A4).
    Only the threshold crossings are scanned (the cooldown pass is
    inherently sequential, but crossings are rare by construction)."""
    s = np.asarray(scores, dtype=float)
    alerts, last = [], None
    for i in np.flatnonzero(s >= threshold):
        d = dates[i]
        if last is None or (d - last).days >= cooldown:
            alerts.append(d)
            last = d
    return alerts


def _day_ints(x) -> np.ndarray:
    """Timestamps -> integer day numbers. Every judging comparison below
    runs in int64 day-space: the cooldown/tally sweep is called thousands
    of times per tournament, and Timestamp arithmetic in a Python loop is
    ~50x slower than integer arithmetic."""
    return (pd.DatetimeIndex(x).values.astype("datetime64[D]")
            .astype(np.int64))


def _eps_arrays(eps: pd.DataFrame) -> dict:
    """One instrument's episodes as parallel int-day arrays."""
    return {"trough": _day_ints(eps["trough"]) if len(eps) else np.array([], np.int64),
            "lo": _day_ints(eps["onset_lo"]) if len(eps) else np.array([], np.int64),
            "hi": _day_ints(eps["onset_hi"]) if len(eps) else np.array([], np.int64),
            "peak": _day_ints(eps["peak"]) if len(eps) else np.array([], np.int64)}


def classify_onset_alerts(alert_days: np.ndarray, ea: dict) -> dict:
    """Judge one instrument's onset alerts (int days) against its
    episodes (int-day arrays from _eps_arrays).

    HIT  : alert inside [onset_lo, onset_hi] (each episode captured once);
    LATE : alert inside (onset_hi, peak] - in the rally, past its start;
    FA   : everything else.
    Leads are recorded both ways: days after the trough, days before the
    peak."""
    captured, late, fa, leads = set(), [], [], []
    for a in alert_days:
        hit = np.flatnonzero((ea["lo"] <= a) & (a <= ea["hi"]))
        if hit.size:
            for i in hit:
                p = int(ea["peak"][i])
                if p not in captured:
                    captured.add(p)
                    leads.append({"after_trough": int(a - ea["trough"][i]),
                                  "before_peak": int(p - a)})
            continue
        if np.any((ea["hi"] < a) & (a <= ea["peak"])):
            late.append(int(a))
        else:
            fa.append(int(a))
    return {"captured": captured, "late": late, "fa": fa, "leads": leads}


def classify_top_alerts(alert_days: np.ndarray, ea: dict) -> dict:
    """Judge one instrument's top alerts - the existing aim, unchanged:
    HIT = alert inside [peak-30d, peak+1d]; FA = no episode peak within
    [alert, alert+45d]. (Structurally parallel to the onset judge.)"""
    captured, fa, leads = set(), [], []
    for a in alert_days:
        hit = np.flatnonzero((ea["peak"] - TOP_LEAD_DAYS <= a)
                             & (a <= ea["peak"] + 1))
        if hit.size:
            for i in hit:
                p = int(ea["peak"][i])
                if p not in captured:
                    captured.add(p)
                    leads.append({"before_peak": int(p - a)})
            continue
        if not np.any((ea["peak"] >= a) & (ea["peak"] <= a + 45)):
            fa.append(int(a))
    return {"captured": captured, "late": [], "fa": fa, "leads": leads}


def walk_forward_scores(frame: pd.DataFrame, feats: list, label: str,
                        fit_score) -> pd.DataFrame:
    """The walk-forward spine shared by every model in the tournament.

    For each test year y (needing >= 3 positive days in the years before,
    the existing ml_walk_forward convention): fit on years < y, score both
    the train years (threshold selection evidence) and year y (the test).
    fit_score(train_df, apply_df, feats) -> score array on apply_df; a
    rule-based model simply ignores train_df. Returns the frame plus
    'score' and 'test_year' columns, test rows only, all years stacked."""
    out = []
    years = sorted(frame.year.unique())
    for y in years:
        train = frame[frame.year < y]
        test = frame[frame.year == y]
        if train[label].sum() < 3 or test.empty:
            continue
        out.append(test.assign(score=fit_score(train, test, feats),
                               train_score=np.nan, test_year=y))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=list(frame.columns) + ["score",
                                                             "test_year"]))


def _alerts_int(days: np.ndarray, scores: np.ndarray, threshold: float,
                cooldown: int = EUPHORIA_COOLDOWN_DAYS) -> np.ndarray:
    """alerts_from_scores in int-day space (the tournament hot path)."""
    alerts, last = [], None
    for i in np.flatnonzero(scores >= threshold):
        if last is None or days[i] - last >= cooldown:
            alerts.append(days[i])
            last = days[i]
    return np.asarray(alerts, dtype=np.int64)


def _pregroup(scored: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """Group once, tally many times: {name: (day_ints, scores,
    eps_int_arrays)}. Threshold sweeps re-scan these arrays instead of
    re-grouping frames."""
    eps_by = dict(tuple(episodes.groupby("name")))
    empty = episodes.iloc[0:0]
    return {name: (_day_ints(g["date"]),
                   g["score"].to_numpy(dtype=float),
                   _eps_arrays(eps_by.get(name, empty)))
            for name, g in scored.sort_values("date").groupby("name")}


def _tally(groups: dict, episodes: pd.DataFrame, threshold: float,
           mode: str, denominator_mask) -> dict:
    """Alerts at one threshold -> operational scorecard, pooled over all
    the scored rows. mode: 'onset' | 'top'."""
    judge = classify_onset_alerts if mode == "onset" else classify_top_alerts
    det_col = "onset_detectable" if mode == "onset" else "top_detectable"
    captured, late, fa, leads = set(), [], [], []
    for name, (days, scores, ea) in groups.items():
        alerts = _alerts_int(days, scores, threshold)
        if not alerts.size:
            continue
        res = judge(alerts, ea)
        captured |= {(name, p) for p in res["captured"]}
        late += res["late"]; fa += res["fa"]; leads += res["leads"]
    det = episodes[denominator_mask(episodes) & episodes[det_col]]
    det_keys = {(r.name, int(p)) for r, p in
                zip(det.itertuples(), _day_ints(det["peak"]))}
    hits = len(captured & det_keys)
    return {"captured": hits, "detectable": len(det_keys),
            "late": len(late), "false_alarms": len(fa), "leads": leads}


def choose_threshold(train_scored: pd.DataFrame, episodes: pd.DataFrame,
                     mode: str, fa_budget_per_iy: float,
                     n_instruments: int) -> float:
    """The PRE-STATED criterion, applied on TRAINING years only:
    among percentile thresholds of the train scores, keep those whose
    train FA rate is within the budget; of those, maximise captured;
    tie -> the more conservative (higher) threshold. If nothing fits the
    budget, the most conservative threshold wins (do-no-harm default -
    the same convention as euphoria.walk_forward)."""
    years = sorted(train_scored.year.unique())
    n_iy = max(n_instruments * len(years), 1)
    in_years = lambda eps: eps.year.isin(years)  # noqa: E731
    grid = np.unique(np.percentile(train_scored["score"].dropna(),
                                   np.arange(50, 100, 2.5)))
    groups = _pregroup(train_scored, episodes)
    best_thr, best_key = grid[-1], (-1, -np.inf)
    for thr in grid[::-1]:                     # conservative first
        r = _tally(groups, episodes, thr, mode, in_years)
        if r["false_alarms"] / n_iy > fa_budget_per_iy:
            continue
        key = (r["captured"], thr)             # capture, then conservatism
        if key > best_key:
            best_key, best_thr = key, thr
    return float(best_thr)


def run_tournament_entry(frame: pd.DataFrame, episodes: pd.DataFrame,
                         feats: list, label: str, mode: str,
                         fit_score, fa_budget_per_iy: float) -> dict:
    """One model through the whole discipline: walk-forward scores, a
    threshold chosen per test year on its train years only, alerts,
    pooled scorecard + threshold-independent AP/AUROC on the stacked
    test scores (the thesis's §6.2.3 separation of score quality from
    operating point)."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    n_instruments = frame["name"].nunique()
    scored = walk_forward_scores(frame, feats, label, fit_score)
    if scored.empty:
        return {"error": "no scoreable years"}
    test_years = sorted(scored.test_year.unique())

    # thresholds are chosen per test year, on that year's TRAIN years -
    # scored again by the same fitted model (rule models are identical)
    parts = []
    thresholds = {}
    for y in test_years:
        train = frame[frame.year < y]
        train_scored = train.assign(
            score=fit_score(train, train, feats))
        thr = choose_threshold(train_scored, episodes, mode,
                               fa_budget_per_iy, n_instruments)
        thresholds[int(y)] = thr
        parts.append(scored[scored.test_year == y].assign(threshold=thr))
    test = pd.concat(parts, ignore_index=True)

    # operational scorecard: each year's alerts at its own threshold
    eps_by = dict(tuple(episodes.groupby("name")))
    empty = episodes.iloc[0:0]
    per_name_alerts: dict[str, list] = {}
    captured, late, fa, leads = set(), [], [], []
    judge = classify_onset_alerts if mode == "onset" else classify_top_alerts
    for (name, y), g in test.groupby(["name", "test_year"]):
        g = g.sort_values("date")
        alerts = _alerts_int(_day_ints(g["date"]),
                             g["score"].to_numpy(dtype=float),
                             g["threshold"].iloc[0])
        per_name_alerts.setdefault(name, []).extend(alerts.tolist())
    for name, alerts in per_name_alerts.items():
        ea = _eps_arrays(eps_by.get(name, empty))
        res = judge(np.asarray(sorted(alerts), dtype=np.int64), ea)
        captured |= {(name, p) for p in res["captured"]}
        late += res["late"]; fa += res["fa"]; leads += res["leads"]

    det_col = "onset_detectable" if mode == "onset" else "top_detectable"
    det = episodes[episodes.year.isin(test_years) & episodes[det_col]]
    det_keys = {(r.name, int(p)) for r, p in
                zip(det.itertuples(), _day_ints(det["peak"]))}
    n_iy = max(n_instruments * len(test_years), 1)
    y_true, y_score = test[label].values, test["score"].values
    hits = len(captured & det_keys)

    def _ts(day: int) -> pd.Timestamp:
        return pd.Timestamp(np.datetime64(int(day), "D"))

    return {
        "test_years": [int(y) for y in test_years],
        "thresholds": thresholds,
        "captured": hits, "detectable": len(det_keys),
        "capture_rate": round(hits / len(det_keys), 3) if det_keys else None,
        "late": len(late), "false_alarms": len(fa),
        "fa_per_iy": round(len(fa) / n_iy, 3),
        "leads": leads,
        "auroc": round(float(roc_auc_score(y_true, y_score)), 3)
        if len(set(y_true)) > 1 else None,
        "ap": round(float(average_precision_score(y_true, y_score)), 3)
        if len(set(y_true)) > 1 else None,
        "ap_baseline": round(float(np.mean(y_true)), 3),
        "alerts_by_name": {k: [_ts(a) for a in v]
                           for k, v in per_name_alerts.items() if v},
    }


# ---------------------------------------------------------------------------
# 5. PRODUCTION - the live onset detector (the NB03 tournament winner:
#    the RULES model - un-weighted mean of ONSET_BANK - which beat every
#    learner under the pre-stated criterion). Called from run_analytics on
#    every rebuild, exactly like the top detector.
# ---------------------------------------------------------------------------
def onset_score(df: pd.DataFrame) -> pd.Series:
    """The winning onset score: the un-weighted mean of the locked bank
    (the euphoria-LEVEL construction applied to the onset features).
    Takes NO price argument, by design - enforced by a unit test."""
    return df[ONSET_BANK].mean(axis=1)


def _stored_onset_report() -> dict | None:
    import json as _json
    import os as _os
    from src.config import PROCESSED_DIR
    path = _os.path.join(PROCESSED_DIR, "euphoria_onset_report.json")
    if not _os.path.exists(path):
        return None
    try:
        return _json.load(open(path))
    except (ValueError, OSError):
        return None


def onset_needs_research(stored: dict | None, data_max_year: int) -> bool:
    """Mirror of euphoria.needs_research: the walk-forward trains on
    strictly earlier years, so a frozen threshold is exactly right until
    (a) no report exists yet, or (b) the data rolls into a year the
    stored record does not cover. Backfills / rule changes are explicit
    --research runs."""
    if not stored or "live_threshold" not in stored:
        return True
    years = stored.get("walk_forward", {}).get("test_years") or []
    return not years or data_max_year > max(int(y) for y in years)


def rebuild_phase_files(verbose: bool = True,
                        research: bool | None = None) -> dict:
    """Rebuild what the dashboard's start/end panes read.

    LIVE mode (the pipeline default when a valid report exists and the
    year has not rolled over): build today's features, score at the
    FROZEN live threshold, refresh episodes.parquet +
    euphoria_onset.parquet. Seconds beyond the unavoidable feature
    build; the stored walk-forward scorecard is left untouched (it is a
    research artifact with an as-of range, not a daily statistic).

    RESEARCH mode (research=True, or auto-triggered by
    onset_needs_research): additionally re-runs the winner's full
    walk-forward scorecard and re-selects the live threshold - on FULL
    years strictly before the current data year (the incumbent's
    convention, so the threshold is stable within a year by
    construction) - and rewrites euphoria_onset_report.json."""
    import json as _json
    import os as _os

    from src.config import PRICES_PATH, PROCESSED_DIR
    from analytics.euphoria import build_all_series
    from analytics.loaders import load as _load, THEME_COUNTS, THEME_SENT, \
        TICKER_COUNTS, TICKER_SENT

    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    series, pxmap = build_all_series(prices)
    episodes = episode_catalog(series, pxmap)

    counts = {"theme": _load(THEME_COUNTS), "ticker": _load(TICKER_COUNTS)}
    sents = {"theme": _load(THEME_SENT), "ticker": _load(TICKER_SENT)}
    for d in list(counts.values()) + list(sents.values()):
        d["date"] = pd.to_datetime(d["date"])

    # the LIVE frame (through today - recent alerts are PENDING, not
    # false) is needed in both modes; it is the unavoidable cost
    frame_live = build_day_frame(series, pxmap, episodes, counts, sents,
                                 clip_judgeable=False)
    onset_live = frame_live[frame_live.hype_raw >= 1].copy()
    data_max_year = int(pd.to_datetime(frame_live["date"]).max().year)

    stored = _stored_onset_report()
    if research is None:
        research = onset_needs_research(stored, data_max_year)
        if research and stored and verbose:
            print("  onset research pass auto-triggered (report missing "
                  "the current year)")

    if research:
        # the JUDGED frame (labels need 45d of future price) powers the
        # honest scorecard + threshold selection
        frame = build_day_frame(series, pxmap, episodes, counts, sents)
        onset_frame = frame[frame.hype_raw >= 1].copy()

        # the derived FA budget: the noise level the desk already
        # accepted from the validated top detector
        rep_path = _os.path.join(PROCESSED_DIR, "euphoria_report.json")
        fa_budget = 0.23
        if _os.path.exists(rep_path):
            fa_budget = _json.load(open(rep_path))["overall"][
                "fa_per_instrument_year"]

        def _rules(train, apply, feats):
            return onset_score(apply).values

        wf = run_tournament_entry(onset_frame, episodes, ONSET_BANK,
                                  "y_onset", "onset", _rules, fa_budget)

        # live threshold from FULL years strictly before the current
        # data year - stable within a year by construction
        train = onset_frame[onset_frame["year"] < data_max_year]
        if train.empty:                      # very young data: use all
            train = onset_frame
        train_scored = train.assign(score=onset_score(train))
        thr_live = choose_threshold(train_scored, episodes, "onset",
                                    fa_budget, train["name"].nunique())
        stored = {
            "live_threshold": thr_live,
            "fa_budget_per_iy": fa_budget,
            "walk_forward": {k: v for k, v in wf.items()
                             if k not in ("leads", "alerts_by_name")},
            "onset_window_days": ONSET_WINDOW_DAYS,
            "bank": ONSET_BANK,
        }
        with open(_os.path.join(PROCESSED_DIR,
                                "euphoria_onset_report.json"), "w") as f:
            _json.dump(stored, f, indent=1, default=str)
        if verbose:
            print(f"  onset RESEARCH pass: {wf['captured']}/"
                  f"{wf['detectable']} detectable onsets captured, "
                  f"{wf['late']} late, {wf['false_alarms']} FA "
                  f"({wf['fa_per_iy']}/instr-yr vs budget {fa_budget}) | "
                  f"live threshold {thr_live:.3f}")
    else:
        thr_live = float(stored["live_threshold"])
        if verbose:
            print(f"  onset LIVE mode: frozen threshold {thr_live:.3f} "
                  "(intra-year recompute is a no-op by the walk-forward "
                  "convention). Full validation: run_analytics --what "
                  "phases --research")

    # daily score + alert table for the dashboard (both modes)
    live_scored = onset_live.assign(score=onset_score(onset_live))
    rows = []
    for name, g in live_scored.sort_values("date").groupby("name"):
        alerts = set(alerts_from_scores(g["date"].tolist(),
                                        g["score"].tolist(), thr_live))
        cols = {
            "date": g["date"].values, "name": name,
            "kind": g["kind"].values,
            # "onset_score", not "score": the FORBIDDEN-columns contract
            # reserves "score" for the raw Reddit post field it bans
            "onset_score": g["score"].values,
            # raw hype ratio (7d share / own 120d median) so the
            # dashboard can hold single names to the FULL A1 bar (2x)
            # for display - "really euphoric", not marginal flicker
            "hype_raw": g["hype_raw"].values,
            "alert": g["date"].isin(alerts).values}
        # the bank components, so the dashboard can EXPLAIN each onset
        for c in ONSET_BANK:
            cols[c] = g[c].values
        rows.append(pd.DataFrame(cols))
    out = pd.concat(rows, ignore_index=True)
    sym_by = {es.name: es.symbol for es in series}
    out["symbol"] = out["name"].map(sym_by)

    from src.abstracted_data import _safe_write
    _safe_write(episodes, _os.path.join(PROCESSED_DIR, "episodes.parquet"))
    _safe_write(out, _os.path.join(PROCESSED_DIR, "euphoria_onset.parquet"))
    if verbose:
        print(f"  saved episodes.parquet ({len(episodes)}), "
              f"euphoria_onset.parquet ({len(out):,})")
    return stored


# ---------------------------------------------------------------------------
# 6. EPISODE COHERENCE - the desk-facing state machine (2026-07-24)
# ---------------------------------------------------------------------------
def episode_coherent_alerts(onset_dates, top_dates,
                            cooldown: int = EUPHORIA_COOLDOWN_DAYS):
    """The desk-facing state machine, ASYMMETRIC by evidence
    (2026-07-24): a new START within `cooldown` days AFTER an END is a
    contradictory flip and is SUPPRESSED (you cannot start euphoria the
    desk was just told is ending); a fast START -> END is a REAL,
    violent mania and the ENDING (risk) signal is NEVER suppressed.

    The asymmetry was measured, not assumed: the symmetric rule cost the
    top detector half its walk-forward captures (17 -> 9) for only 8
    fewer false alarms - fast episodes legitimately run start-to-end
    inside one cooldown - while the onset direction cost 2 captures and
    removed 7 FAs. The cooldown (21d) is the project's existing
    one-episode timescale; no new constant. Same-day tie: the END wins
    and the same-day START is suppressed.

    Returns (kept_onset_dates, kept_top_dates), chronological."""
    tops = sorted(pd.Timestamp(d) for d in top_dates)
    kept_onset = []
    for d in sorted(pd.Timestamp(x) for x in onset_dates):
        blocked = any(0 <= (d - t).days < cooldown for t in tops)
        if not blocked:
            kept_onset.append(d)
    return kept_onset, tops
