"""Euphoria phases: episode ground truth, onset features, tournament, and the GET IN / GET OUT pair.

This module extends the top detector in analytics/euphoria.py (which calls
the END of retail euphoria) with the machinery to also call its START, and
with the deployed two-sided signal the dashboard shows. It owns:

1. EPISODES - the price-defined ground truth. An episode is one full
   boom-bust arc (trough -> peak -> bust) built from the same peak
   definition the top detector is scored on (G1-G3 in euphoria.py), so
   both detectors share one notion of "a genuine euphoria event".
   `find_episodes` / `episode_catalog`.
2. The ONSET FEATURE BANK - trailing, crowd-only features aimed at the
   left side of an episode (the crowd arriving). `compute_onset_features`,
   `ONSET_FEATURES` (candidates), `ONSET_BANK` (the locked production
   list).
3. The LABELLED DAY FRAME - one row per (instrument, candidate day) with
   every feature and the onset/late/top labels, shared by the research
   tooling and the production detector so the two cannot drift.
   `label_days`, `build_day_frame`.
4. TOURNAMENT MACHINERY - the walk-forward model comparison every
   detector in this project is judged by: `walk_forward_scores`,
   `choose_threshold`, `run_tournament_entry`, the alert judges
   (`classify_onset_alerts`, `classify_top_alerts`) and the alert
   triggers (`alerts_from_scores`, `alerts_from_scores_shaped`).
5. PRODUCTION - `rebuild_phase_files`, the pipeline stage that scores
   today's data and writes every phase file the dashboard reads.
6. The GET IN / GET OUT pair - the deployed signal family (rule-based
   fallback in `desk_end_fit` / `desk_onset_fit` / `desk_candidacy`; the
   learned families live in analytics/ml_detector.py).
7. EPISODE COHERENCE - `episode_coherent_alerts`, the display-facing rule
   that keeps a START from landing on top of an END.

Inputs (all under PROCESSED_DIR unless stated):
    prices.parquet (PRICES_PATH)             daily closes per symbol
    theme/ticker counts and sentiment stores  via analytics.loaders
    euphoria_onset_report.json               frozen onset threshold
    euphoria_desk_report.json                frozen GET IN / GET OUT
                                             cuts, re-arm levels, model
                                             family, tournament table

Outputs written by `rebuild_phase_files`:
    episodes.parquet                  the episode catalog
    euphoria_onset.parquet            crowd-only onset score + alerts
    euphoria_desk.parquet             GET IN / GET OUT scores, alerts,
                                      inflection marker, price-blind and
                                      retail-flow columns
    euphoria_desk_components.parquet  per-day feature readings for hover
    desk_model_insight.json           logit weights + GBM permutation
                                      importance of the live fit
    readiness_alerts.json             names within 90% of a strict cut
    euphoria_onset_report.json / euphoria_desk_report.json
                                      rewritten on research passes only

Scoring conventions (fixed before any result was computed):

* ONSET HIT WINDOW: an onset alert is a HIT when it lands inside
  [trough, min(trough + ONSET_WINDOW_DAYS, peak)]. The trough is the 120d
  low the boom is measured from (G2), so "the start" is anchored to the
  same low the peak definition already uses - no new fitted quantity. The
  window is capped at the peak because, for fast rallies, an uncapped
  +45d would let an alert fired AFTER the top count as "caught the start".
* LATE is not FALSE: an onset alert inside (window end, peak] fired
  during the rally but after its start. It is reported as LATE,
  separately from hits and from false alarms: calling it a hit inflates
  the onset claim, calling it false punishes an alert inside a genuine
  episode. Only alerts outside the whole episode are false alarms.
* CROWD-ONLY PREDICTION for the onset detector: price never enters any
  onset feature or alert - it appears only in the episode definition and
  the scoring. The GET IN / GET OUT pair is a separately labelled family
  that is allowed two price features (see section 6 and ml_detector.py).
* WALK-FORWARD BY YEAR: every threshold and every learned model is chosen
  on full years strictly before the year it is scored on. Thresholds are
  therefore stable within a year by construction, and a live run scoring
  at a frozen threshold is the out-of-sample use the evaluation licenses.

All feature rules are trailing (day t uses only data <= t) and are
percentile ranks against the same instrument's own trailing history, for
the same reasons as E1-E5 in euphoria.py: the underlying series are
fat-tailed (ranks do not over-react where z-scores would) and coverage
shifts over time (shares and own-history ranks are immune, raw counts are
not).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (EUPHORIA_CRASH_MIN_ETF, EUPHORIA_CRASH_MIN_SINGLE,
                        EUPHORIA_BOOM_LOOKBACK_D, EUPHORIA_CRASH_WINDOW_D,
                        EUPHORIA_MIN_HISTORY,
                        EUPHORIA_FA_BUDGET_PER_IY,
                        EUPHORIA_INFLECTION_ENABLED, EUPHORIA_INFLECTION_CUT_Q,
                        EUPHORIA_INFLECTION_REARM_Q, EUPHORIA_INFLECTION_SPACING_D,
                        EUPHORIA_XP_ENABLED)
from analytics.euphoria import (EuphoriaSeries, ground_truth_peaks,
                                judgeable_window, trailing_pct_rank,
                                log_convexity, _mention_share,
                                _bullish_series)
from analytics.loaders import load, TICKER_COUNTS_BY_SOURCE

# The onset hit window length. Anchored to the
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
    """One complete boom-bust arc for one instrument.

    Frozen: ground truth is evidence, never something downstream code may
    edit.
    """
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
                  kind: str, boom_min: float | None = None,
                  crash_min: float | None = None) -> list[Episode]:
    """Extend each confirmed peak (G1-G3) into a full episode.

    Walks BACKWARD from the peak to its trough and FORWARD to its bust.
    Nothing here is new ground truth: the trough is exactly the "min
    close over the preceding EUPHORIA_BOOM_LOOKBACK_D days" that G2
    already measures the boom against, and the bust date is the first
    day G3's drawdown condition is met.

    Args:
        px: Daily close series indexed by date (NaNs are dropped).
        name: Theme name or ticker.
        symbol: The priced symbol behind the instrument.
        kind: "theme" or "single".
        boom_min: Sweep override for the G2 boom bar; the config default
            for `kind` when None (see ground_truth_peaks).
        crash_min: Sweep override for the G3 crash bar; the config
            default for `kind` when None.

    Returns:
        A list of Episode records, one per confirmed peak, in date order.
    """
    if crash_min is None:
        crash_min = (EUPHORIA_CRASH_MIN_SINGLE if kind == "single"
                     else EUPHORIA_CRASH_MIN_ETF)
    px = px.dropna()
    episodes = []
    for peak in ground_truth_peaks(px, kind, boom_min=boom_min,
                                   crash_min=crash_min):
        prior = px.loc[peak - pd.Timedelta(days=EUPHORIA_BOOM_LOOKBACK_D):peak]
        trough = prior.idxmin()
        peak_close = px.loc[peak]
        after = px.loc[peak:peak + pd.Timedelta(days=EUPHORIA_CRASH_WINDOW_D)]
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


def episode_catalog(series: list, pxmap: dict,
                    gt_override: dict | None = None) -> pd.DataFrame:
    """Every episode for every instrument, as one flat table.

    Two detectability flags are always reported side by side:

    onset_detectable : the coverage gate (A0) held on >= 1 day of the
                       onset window. A start no crowd was measurable for
                       stays in the "all" denominator but cannot honestly
                       score a crowd detector.
    top_detectable   : the same flag over [peak-30d, peak] (the
                       definition in euphoria.peak_maps).

    Args:
        series: EuphoriaSeries objects, one per instrument.
        pxmap: symbol -> daily close series.
        gt_override: Optional ground-truth overrides keyed
            "boom_<kind>" / "crash_<kind>" (used by the ground-truth
            sweep); config defaults when absent.

    Returns:
        A DataFrame with one row per episode (columns name, symbol, kind,
        trough, peak, bust_date, boom_pct, bust_pct, run_days, onset_lo,
        onset_hi, year, onset_detectable, top_detectable), sorted by
        peak then name.
    """
    gt = gt_override or {}
    rows = []
    for es in series:
        for ep in find_episodes(
                pxmap[es.symbol], es.name, es.symbol, es.kind,
                boom_min=gt.get(f"boom_{es.kind}"),
                crash_min=gt.get(f"crash_{es.kind}")):
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

# The LOCKED production bank (feature battery:
# reference/research_record/nb02_feature_stats.json): source_breadth is
# EXCLUDED because its apparent skill was a coverage-regime artifact - X
# and StockTwits exist in the archive only from 2026, so "breadth" mostly
# encoded "which year is it". The research tooling and the live detector
# all import THIS list, so research and production cannot drift.
ONSET_BANK = ["attention_accel", "hype_ratio", "bull_inflection",
              "influx_speed", "attention_convexity"]

# The top detector's bank (the rule-based baseline), for side-by-side
# evaluation.
TOP_FEATURES = ["e1", "e2", "e3", "e5", "fade"]


def _source_breadth(name: str, all_days: pd.DatetimeIndex) -> pd.Series | None:
    """O6: count of distinct sources mentioning the name in the trailing 7d.

    Only tickers have a by-source aggregate; for themes this returns None
    and the feature is simply absent (the frame builder drops it).
    """
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
                           with_breadth: bool = True,
                           by_source: pd.DataFrame | None = None
                           ) -> pd.DataFrame:
    """The onset feature bank for one instrument, from crowd aggregates only.

    No price argument exists, by design; a unit test enforces this.

    Args:
        name: Theme name or ticker.
        counts_long: Long-form mention counts (date, entity, mention_count).
        sent_long: Long-form scored sentiment (date, entity, n_posts,
            net_bullish).
        entity_col: "theme" or "ticker" - the entity column in both frames.
        with_breadth: Also compute source_breadth (tickers only).
        by_source: Optional per-source ticker counts for the stratified
            mention share.

    Returns:
        A daily DataFrame with one column per available feature, plus
        `hype_raw` (the un-ranked hype ratio used by the onset
        prerequisite gate; not a bank feature).
    """
    all_days = pd.date_range(counts_long["date"].min(),
                             counts_long["date"].max(), freq="D")
    share, m7 = _mention_share(counts_long, entity_col, name, all_days,
                               by_source=by_source)

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
    # Vectorised weighted sum: a `groupby("date").apply(lambda ...)` does
    # the same arithmetic once per group in Python (~500 ms per instrument
    # on a 300k-row sentiment store) instead of once in C (~2 ms).
    nb = ((one["n_posts"] * one["net_bullish"]).groupby(one["date"]).sum()
          .reindex(all_days).fillna(0.0))
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
# 3. THE LABELLED DAY FRAME - the one table research and production share
# ---------------------------------------------------------------------------
def label_days(index: pd.DatetimeIndex, episodes: pd.DataFrame,
               name: str) -> pd.DataFrame:
    """Three {0,1} labels for every day of one instrument's series.

    y_onset : day inside an episode's onset window [trough, capped end]
    y_late  : day inside (onset end, peak] - in the rally, past its start
    y_top   : day inside [peak - TOP_LEAD_DAYS, peak + 1d] - the aim
              window of the top detector

    Args:
        index: The instrument's daily date index.
        episodes: The episode catalog (episode_catalog output).
        name: The instrument to label.

    Returns:
        A DataFrame indexed like `index` with columns y_onset, y_late,
        y_top.
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
    """One row per (instrument, candidate day) with features and labels.

    Each row carries every onset feature, every top-detector feature, the
    un-gated sentiment ingredients, and the three labels. Candidate days
    are the days the detector is allowed to speak on: coverage gate
    satisfied AND inside the judgeable price window (an unjudgeable label
    is missing data, not truth).

    Args:
        series: EuphoriaSeries objects, one per instrument.
        pxmap: symbol -> daily close series.
        episodes: The episode catalog.
        counts: entity_col -> long-form counts frame, e.g.
            {"theme": theme_counts, "ticker": tick_counts}.
        sents: entity_col -> long-form sentiment frame, same keys.
        clip_judgeable: When False, keep the most recent ~45 days, whose
            labels cannot be judged yet. Alerts there are PENDING, not
            false: research must clip them, the live dashboard must
            score them.

    Returns:
        A DataFrame with columns date, name, kind, year, hype_ok, every
        feature, and the labels; empty if no instrument yields rows.
    """
    by_src = load(TICKER_COUNTS_BY_SOURCE)
    if by_src is not None:
        by_src = by_src.assign(date=pd.to_datetime(by_src["date"]))
    frames = []
    for es in series:
        entity_col = "theme" if es.kind == "theme" else "ticker"
        onset = compute_onset_features(
            es.name, counts[entity_col], sents[entity_col], entity_col,
            by_source=by_src if entity_col == "ticker" else None)
        top = pd.DataFrame({"e1": es.e1, "e2": es.e2, "e3": es.e3,
                            "e5": es.e5, "fade": es.fade.astype(float),
                            # un-gated sentiment ingredients (ML bank)
                            "bull_level": es.bull_level,
                            "bull_persist": es.bull_persist})
        df = onset.join(top)
        labels = label_days(df.index, episodes, es.name)
        df = df.join(labels)
        # candidate-day mask: measurable (A0) and judgeable (price truth)
        j0, j1 = judgeable_window(pxmap[es.symbol])
        if j0 is None:
            continue
        ok = es.coverage_ok.reindex(df.index).fillna(False).astype(bool)
        mask = ok & (df.index >= j0)
        if clip_judgeable:
            mask = mask & (df.index <= j1)
        df = df[mask]
        # features must exist (percentiles need EUPHORIA_MIN_HISTORY days)
        df = df.dropna(subset=[c for c in (ONSET_FEATURES + TOP_FEATURES
                                           + ["bull_level", "bull_persist"])
                               if c in df.columns
                               and c not in ("source_breadth", "hype_raw")])
        if df.empty:
            continue
        df = df.assign(name=es.name, kind=es.kind, year=df.index.year,
                       hype_ok=es.boom_ok.reindex(df.index)
                       .fillna(False).astype(bool))
        frames.append(df.reset_index(names="date"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. TOURNAMENT MACHINERY - the walk-forward model comparison, written
#    here so the production detector runs the WINNING code path, not a
#    re-implementation of it.
# ---------------------------------------------------------------------------
from src.config import EUPHORIA_COOLDOWN_DAYS  # noqa: E402  (single source)


def alerts_from_scores(dates: list, scores: list, threshold: float,
                       cooldown: int = EUPHORIA_COOLDOWN_DAYS) -> list:
    """Scores -> sparse alert dates.

    Fires on score >= threshold, then applies the standard cooldown (the
    same rule as the top detector's A4). Only the threshold crossings are
    scanned: the cooldown pass is inherently sequential, but crossings
    are rare by construction.

    Args:
        dates: Dates aligned with `scores`, in chronological order.
        scores: Score per date.
        threshold: Fire when score >= threshold.
        cooldown: Minimum days between two alerts for one name.

    Returns:
        The alert dates, in order.
    """
    s = np.asarray(scores, dtype=float)
    alerts, last = [], None
    for i in np.flatnonzero(s >= threshold):
        d = dates[i]
        if last is None or (d - last).days >= cooldown:
            alerts.append(d)
            last = d
    return alerts


def alerts_from_scores_shaped(dates: list, scores: list, gate,
                              threshold: float, rearm: float,
                              spacing: int) -> list:
    """The SHAPED trigger: one clear call per boom.

    Evidence: reference/research_record/alert_shape_sweep.json.

    * Fire only on an UPWARD CROSSING of the cut, and only on days the
      PHASE GATE allows (GET OUT: the name has already boomed by the
      ground truth's own 120d bar; GET IN: it has not).
    * After a fire, the score must drop below the RE-ARM level before it
      may fire again (GET OUT re-arms at the cut; GET IN re-arms at the
      train-median score - the crowd must fully cool).
    * One call per name per `spacing` days (63 - one per quarter).

    Walk-forward, this took GET OUT from 305 calls / 1.6 per boom to 130
    calls / 1.2 per boom at precision 0.36 -> 0.53, and made IN-vs-OUT
    adjacency structurally impossible outside the bar-crossing moment
    (the display layer then drops the few INs that remain within 21d of
    an OUT).

    Args:
        dates: Dates aligned with `scores`, in chronological order.
        scores: Score per date.
        gate: Boolean per date; a fire is allowed only where True.
        threshold: The cut.
        rearm: The score must fall below this before the next fire.
        spacing: Minimum days between two fires for one name.

    Returns:
        The alert dates, in order.
    """
    s = np.asarray(scores, dtype=float)
    g = np.asarray(gate, dtype=bool)
    alerts, last, armed = [], None, True
    for d, v, ok in zip(dates, s, g):
        if v >= threshold:
            if ok and armed and (last is None
                                 or (d - last).days >= spacing):
                alerts.append(d)
                last = d
            armed = False
        elif v < rearm:
            armed = True
    return alerts


def boomed120_frame(series, pxmap) -> pd.DataFrame:
    """The 120d boom phase gate, raw and display-stabilised.

    Returns name/date/boomed120 (+ boomed120_stable): run-up vs the
    trailing 120d low >= the G2 boom bar (20% themes / 40% singles) - the
    EPISODE DEFINITION's own boom test, reused as the alert phase gate
    (no new constant). A top may only be called after a boom as the
    ground truth defines booms; a start only before the boom has
    completed.

    boomed120 is the MODEL's gate and never changes: every frozen
    threshold was calibrated against it.

    boomed120_stable is a DISPLAY-ONLY twin. The raw gate is an
    instantaneous test against a hard bar, so a name parked near it
    flips sides on ordinary noise: measured over 12 months of the live
    store, 96% of side flips happened within 5pp of the bar and one name
    (URA) flipped 25 times with a median run of 3 days. Two standard
    cures, applied in order:

      * HYSTERESIS (Schmitt trigger): enter the run-up state at the bar,
        leave it only once the run-up decays to bar - 5pp. A name must
        genuinely give back part of the move to be treated as
        no-longer-run-up, instead of jittering on a rounding error.
      * DEBOUNCE, ASYMMETRIC: leaving the run-up state is accepted only
        after 3 consecutive days, which removes the residue. ENTERING is
        immediate and deliberately un-debounced: a real breakout must
        register at once, and delaying it put 14 of the store's 124 GET
        OUT calls on a day the display still showed as pre-boom (a red
        marker on a teal band). The asymmetry costs nothing: 89 flips a
        year against 88 for the symmetric version.

    Measured on the live store: flips 201 -> 89 a year (-56%), median
    time on a side 8 -> ~48 days, and runs shorter than 3 days go from
    19% of all runs to zero. It is SIGNAL-NEUTRAL: replayed against every
    GET OUT call in the store, all 124 survive on a displayed-boomed day
    and no new day becomes gate-eligible (fires happen deep inside a
    run-up, never at the knife edge), so nothing about the model's calls
    changes - only which side of the bar is displayed.

    Args:
        series: EuphoriaSeries objects, one per instrument.
        pxmap: symbol -> daily close series.

    Returns:
        A DataFrame with columns name, date, boomed120, boomed120_stable.
    """
    from src.config import (EUPHORIA_BOOM_MIN_ETF,
                            EUPHORIA_BOOM_MIN_SINGLE)
    _EXIT_GIVEBACK = 0.05      # leave the run-up state at bar - 5pp
    _DEBOUNCE_DAYS = 3

    def _stable(runup, bar):
        enter, leave = bar, bar - _EXIT_GIVEBACK
        on = False
        held = []                                   # hysteresis pass
        for v in runup:
            if not np.isnan(v):
                if not on and v >= enter:
                    on = True
                elif on and v < leave:
                    on = False
            held.append(on)
        out, cur, run = [], (held[0] if held else False), 0
        for x in held:                              # debounce pass
            if x == cur:
                run = 0
            else:
                run += 1
                # ON is immediate; only leaving the state waits
                if x or run >= _DEBOUNCE_DAYS:
                    cur, run = x, 0
            out.append(cur)
        return out

    rows = []
    for es in series:
        px = pxmap[es.symbol].dropna().asfreq("D").ffill()
        low120 = px.rolling(120, min_periods=60).min()
        bar = (EUPHORIA_BOOM_MIN_SINGLE if es.kind == "single"
               else EUPHORIA_BOOM_MIN_ETF)
        runup = (px / low120 - 1)
        rows.append(pd.DataFrame({
            "name": es.name, "date": px.index,
            "boomed120": (runup >= bar).values,
            "boomed120_stable": _stable(runup.values, bar)}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# INFLECTION - the reversal context marker
# ---------------------------------------------------------------------------
# Trigger swept in reference/research_record/inflection_trigger_sweep.json;
# the constants, and the reason it is a CONTEXT MARKER rather than a call,
# are in src/config.py.
#
# The four features below were the only price-free additions that
# improved the inflection head (reference/research_record/nb08_inflection.json).
# They are used by the INFLECTION head ONLY. GET IN and GET OUT keep the
# shipped bank untouched: changing their inputs is a separate adoption
# that needs its own research re-freeze, and bundling it here would make
# the inflection head impossible to evaluate against the record it joins.
INFLECTION_EXTRA_FEATURES = ["att_vol_21", "bull_dispersion", "att_x_mood",
                       "breadth_chg"]


def inflection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four INFLECTION-only crowd features. Price-free by design.

    att_vol_21      how UNSTABLE attention has been, not how high
    bull_dispersion spread of daily mood - disagreement, not direction
    att_x_mood      the interaction; loud-and-bullish differs from
                    loud-and-bearish and two additive terms cannot say so
    breadth_chg     change in how many sources carry the name

    Args:
        df: A day frame with name, date, hype_raw, bull_level, e1 and
            optionally source_breadth.

    Returns:
        A copy of `df` (original row order) with the four columns added,
        NaNs filled with 0.
    """
    d = df.sort_values(["name", "date"]).copy()
    g = d.groupby("name", sort=False)
    d["att_vol_21"] = g["hype_raw"].transform(
        lambda s: s.rolling(21, min_periods=8).std())
    d["bull_dispersion"] = g["bull_level"].transform(
        lambda s: s.rolling(21, min_periods=8).std())
    d["att_x_mood"] = d["e1"] * d["bull_level"]
    if "source_breadth" in d:
        d["breadth_chg"] = g["source_breadth"].transform(
            lambda s: s - s.rolling(21, min_periods=8).mean())
    else:
        d["breadth_chg"] = 0.0
    for c in INFLECTION_EXTRA_FEATURES:
        d[c] = d[c].fillna(0.0)
    return d.loc[df.index]


def inflection_label_frame(series: list, pxmap: dict) -> pd.DataFrame:
    """name/date/y_inflection - is a real reversal about to land?

    A day is an EXTREMUM when its close is the max (or min) of the
    +/- EUPHORIA_INFLECTION_WIN_D window AND the excess move away from it
    over the next EUPHORIA_INFLECTION_HORIZON_D days is at least
    EUPHORIA_INFLECTION_MIN_MOVE. The second test is what separates a
    reversal from a flat drift that happens to contain a local maximum -
    without it the label is noise.

    Excess, not raw: a raw forward move is contaminated by whatever the
    whole market did that month. The excess is the name's forward return
    minus the cross-sectional median of the tracked universe on the same
    day - the same date-matched control used by the max-performance
    evaluation (reference/research_record/max_performance.json).

    y_inflection(t) = 1 when such a day falls in
    (t, t + EUPHORIA_INFLECTION_LOOKAHEAD_D], so a signal on day t is
    allowed to be early rather than exact.

    Args:
        series: EuphoriaSeries objects, one per instrument.
        pxmap: symbol -> daily close series.

    Returns:
        A DataFrame with columns name, date, y_inflection (int 0/1).
    """
    from src.config import (EUPHORIA_INFLECTION_WIN_D, EUPHORIA_INFLECTION_MIN_MOVE,
                            EUPHORIA_INFLECTION_LOOKAHEAD_D,
                            EUPHORIA_INFLECTION_HORIZON_D)
    wide = pd.DataFrame({sym: px for sym, px in pxmap.items()})
    fwd = wide.shift(-EUPHORIA_INFLECTION_HORIZON_D) / wide - 1.0
    excess = fwd.sub(fwd.median(axis=1), axis=0)
    win = 2 * EUPHORIA_INFLECTION_WIN_D + 1
    out = []
    for es in series:
        px = pxmap.get(es.symbol)
        if px is None:
            continue
        px = px.dropna()
        if len(px) < 260 or es.symbol not in excess.columns:
            continue
        hi = px.rolling(win, center=True).max()
        lo = px.rolling(win, center=True).min()
        ex = excess[es.symbol].reindex(px.index)
        is_peak = (px >= hi) & (ex <= -EUPHORIA_INFLECTION_MIN_MOVE)
        is_trough = (px <= lo) & (ex >= EUPHORIA_INFLECTION_MIN_MOVE)
        ext = (is_peak | is_trough).fillna(False).astype(bool)
        fut = (ext.iloc[::-1]
               .rolling(EUPHORIA_INFLECTION_LOOKAHEAD_D, min_periods=1)
               .max().iloc[::-1].shift(-1).fillna(0))
        out.append(pd.DataFrame({"name": es.name, "date": px.index,
                                 "y_inflection": fut.astype(int).values}))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["name", "date", "y_inflection"]))


def inflection_alerts(dates, scores, threshold, rearm, spacing):
    """Inflection alert dates: shaped trigger with no phase gate.

    Upward crossings of `threshold`, re-armed below `rearm`, at most one
    per `spacing` days. NO PHASE GATE, deliberately: a reversal is
    exactly as interesting at the bottom of a bust as at the top of a
    boom, so the gates that gave GET IN and GET OUT their shape would
    throw away half of what this head exists to see.
    """
    gate = [True] * len(dates)
    return alerts_from_scores_shaped(list(dates), list(scores), gate,
                                     threshold, rearm, spacing)


def _day_ints(x) -> np.ndarray:
    """Timestamps -> integer day numbers.

    Every judging comparison below runs in int64 day-space: the
    cooldown/tally sweep is called thousands of times per tournament, and
    Timestamp arithmetic in a Python loop is ~50x slower than integer
    arithmetic.
    """
    return (pd.DatetimeIndex(x).values.astype("datetime64[D]")
            .astype(np.int64))


def _eps_arrays(eps: pd.DataFrame) -> dict:
    """One instrument's episodes as parallel int-day arrays."""
    return {"trough": _day_ints(eps["trough"]) if len(eps) else np.array([], np.int64),
            "lo": _day_ints(eps["onset_lo"]) if len(eps) else np.array([], np.int64),
            "hi": _day_ints(eps["onset_hi"]) if len(eps) else np.array([], np.int64),
            "peak": _day_ints(eps["peak"]) if len(eps) else np.array([], np.int64)}


def classify_onset_alerts(alert_days: np.ndarray, ea: dict) -> dict:
    """Judge one instrument's onset alerts against its episodes.

    HIT  : alert inside [onset_lo, onset_hi] (each episode captured once);
    LATE : alert inside (onset_hi, peak] - in the rally, past its start;
    FA   : everything else.
    Leads are recorded both ways: days after the trough, days before the
    peak.

    Args:
        alert_days: Alert dates as int day numbers (see _day_ints).
        ea: The instrument's episodes as int-day arrays (_eps_arrays).

    Returns:
        A dict with keys captured (set of peak day ints), late (list),
        fa (list) and leads (list of dicts with peak, after_trough,
        before_peak).
    """
    captured, late, fa, leads = set(), [], [], []
    for a in alert_days:
        hit = np.flatnonzero((ea["lo"] <= a) & (a <= ea["hi"]))
        if hit.size:
            for i in hit:
                p = int(ea["peak"][i])
                if p not in captured:
                    captured.add(p)
                    leads.append({"peak": p,
                                  "after_trough": int(a - ea["trough"][i]),
                                  "before_peak": int(p - a)})
            continue
        if np.any((ea["hi"] < a) & (a <= ea["peak"])):
            late.append(int(a))
        else:
            fa.append(int(a))
    return {"captured": captured, "late": late, "fa": fa, "leads": leads}


def classify_top_alerts(alert_days: np.ndarray, ea: dict) -> dict:
    """Judge one instrument's top alerts against its episodes.

    HIT = alert inside [peak-30d, peak+1d]; FA = no episode peak within
    [alert, alert+45d]. Structurally parallel to classify_onset_alerts
    (the `late` list is always empty for tops).

    Args:
        alert_days: Alert dates as int day numbers.
        ea: The instrument's episodes as int-day arrays.

    Returns:
        A dict with keys captured, late, fa, leads.
    """
    captured, fa, leads = set(), [], []
    for a in alert_days:
        hit = np.flatnonzero((ea["peak"] - TOP_LEAD_DAYS <= a)
                             & (a <= ea["peak"] + 1))
        if hit.size:
            for i in hit:
                p = int(ea["peak"][i])
                if p not in captured:
                    captured.add(p)
                    leads.append({"peak": p, "before_peak": int(p - a)})
            continue
        if not np.any((ea["peak"] >= a) & (ea["peak"] <= a + 45)):
            fa.append(int(a))
    return {"captured": captured, "late": [], "fa": fa, "leads": leads}


def walk_forward_scores(frame: pd.DataFrame, feats: list, label: str,
                        fit_score) -> pd.DataFrame:
    """The walk-forward spine shared by every model in the tournament.

    For each test year y with >= 3 positive days in the years before it
    (the euphoria.ml_walk_forward convention): fit on years < y and score
    year y. Fitting by calendar year, rather than by a rolling window,
    keeps every threshold stable within a year and makes "scored
    out-of-sample" mean the same thing for every model.

    Args:
        frame: The labelled day frame (needs `year` and `label`).
        feats: Feature columns handed to `fit_score`.
        label: The label column.
        fit_score: Callable (train_df, apply_df, feats) -> score array
            aligned with apply_df. A rule-based model ignores train_df.

    Returns:
        The test rows of every scoreable year stacked, with `score`,
        `train_score` (NaN) and `test_year` columns added; an empty frame
        with those columns when no year is scoreable.
    """
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
    """The FA-budget threshold rule, applied on TRAINING years only.

    Among percentile thresholds of the train scores, keep those whose
    train false-alarm rate is within the budget; of those, maximise
    captured episodes; tie -> the more conservative (higher) threshold.
    If nothing fits the budget, the most conservative threshold wins
    (the do-no-harm default, the same convention as
    euphoria.walk_forward).

    Args:
        train_scored: Train rows with `score`, `year`, `name`, `date`.
        episodes: The episode catalog.
        mode: "onset" or "top" - which judge to apply.
        fa_budget_per_iy: Allowed false alarms per instrument-year.
        n_instruments: Instruments in the frame (for the per-iy rate).

    Returns:
        The chosen threshold as a float.
    """
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
                         fit_score, fa_budget_per_iy: float,
                         chooser=None) -> dict:
    """Run one model through the whole walk-forward discipline.

    Walk-forward scores, a threshold chosen per test year on its train
    years only, alerts, a pooled operational scorecard, and
    threshold-independent AP/AUROC on the stacked test scores. Reporting
    both keeps score QUALITY (AP/AUROC, threshold-free) separate from the
    OPERATING POINT chosen on it.

    Args:
        frame: The labelled candidate-day frame.
        episodes: The episode catalog.
        feats: Feature columns handed to `fit_score`.
        label: "y_onset" or "y_top".
        mode: "onset" or "top" - which judge to apply.
        fit_score: Callable (train_df, apply_df, feats) -> score array.
        fa_budget_per_iy: Passed through to the chooser.
        chooser: The threshold-selection rule, called as
            chooser(train_scored, episodes, mode, fa_budget_per_iy,
            n_instruments). None selects the FA-budget rule
            (choose_threshold); the ML tournament passes its budget-free
            F1 chooser (analytics.ml_detector.choose_threshold_f1).

    Returns:
        A dict with test_years, thresholds (per test year), captured,
        detectable, capture_rate, late, false_alarms, fa_per_iy, leads,
        auroc, ap, ap_baseline (the positive rate of the test rows) and
        alerts_by_name; or {"error": ...} when no year is scoreable.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if chooser is None:
        chooser = choose_threshold
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
        thr = chooser(train_scored, episodes, mode,
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
# 5. PRODUCTION - the live crowd-only onset detector. The tournament winner
#    (reference/research_record/nb03_tournament.json) is the RULES model -
#    the un-weighted mean of ONSET_BANK - which beat every learner under
#    the pre-stated criterion. Called from run_analytics on every rebuild,
#    exactly like the top detector.
# ---------------------------------------------------------------------------
def onset_score(df: pd.DataFrame) -> pd.Series:
    """The onset score: the un-weighted mean of the locked bank.

    This is the euphoria-LEVEL construction applied to the onset
    features. Takes NO price argument, by design; a unit test enforces
    this.
    """
    return df[ONSET_BANK].mean(axis=1)


def _stored_onset_report() -> dict | None:
    """The frozen onset record (euphoria_onset_report.json), or None."""
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
    """Must a data pull derive the onset threshold for itself?

    Mirror of euphoria.needs_research: a data pull derives a threshold in
    EXACTLY ONE case - there is no usable frozen record to read. A record
    that exists but stops at an earlier year is not a reason to re-fit
    inside a refresh job; it is a reason to print a notice (see
    `onset_record_lags_data`) and leave the refit to an explicit
    `--research` run. The full argument is in `needs_research` in
    analytics/euphoria.py.

    Args:
        stored: The parsed euphoria_onset_report.json, or None.
        data_max_year: Newest year in the data (accepted for signature
            symmetry with `onset_record_lags_data`; unused).

    Returns:
        True when no usable frozen record exists.
    """
    if not stored or "live_threshold" not in stored:
        return True
    return not (stored.get("walk_forward", {}).get("test_years") or [])


def onset_record_lags_data(stored: dict | None, data_max_year: int):
    """Newest walk-forward test year when it lags the data, else None."""
    if not stored or "live_threshold" not in stored:
        return None
    years = stored.get("walk_forward", {}).get("test_years") or []
    if not years:
        return None
    newest = max(int(y) for y in years)
    return newest if data_max_year > newest else None


def rebuild_phase_files(verbose: bool = True,
                        research: bool | None = None) -> dict:
    """Rebuild every phase file the dashboard's start/end panes read.

    LIVE mode (the pipeline default whenever a frozen record exists at
    all): build today's features, score at the FROZEN thresholds, and
    refresh episodes.parquet, euphoria_onset.parquet, euphoria_desk.parquet
    and the sidecars listed in the module docstring. Seconds beyond the
    unavoidable feature build; the stored walk-forward scorecards are
    left untouched (they are research artifacts with an as-of range, not
    daily statistics). A record that lags the data prints one notice and
    is still used - see `onset_needs_research` for why that is the
    methodologically correct default, not a shortcut.

    RESEARCH mode (research=True, or the one-off bootstrap when no record
    exists): additionally re-runs the full walk-forward scorecard for the
    crowd-only onset detector and the GET IN / GET OUT model tournament,
    re-selects every live threshold on FULL years strictly before the
    current data year (the same convention as the top detector, so a
    threshold is stable within a year by construction), and rewrites
    euphoria_onset_report.json and euphoria_desk_report.json.

    Args:
        verbose: Print progress and the notices described above.
        research: True forces a research pass, False forces live mode,
            None (default) chooses live mode unless no frozen record
            exists.

    Returns:
        The onset record dict (the content of euphoria_onset_report.json).

    Raises:
        RuntimeError: When DESK_MODEL_FAMILY is pinned and that family
            errored during its fit, so it did not survive selection.
    """
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
        if research and verbose:
            print("  onset: no frozen record here - deriving the "
                  "threshold once (bootstrap)")
        elif verbose:
            _lag = onset_record_lags_data(stored, data_max_year)
            if _lag is not None:
                print(f"  NOTE: onset data reaches {data_max_year}; the "
                      f"frozen threshold was confirmed through {_lag}. "
                      "Scoring at it out-of-sample, as designed. Refresh "
                      "the record deliberately with "
                      "analytics.run_analytics --what phases --research")

    if research:
        # the JUDGED frame (labels need 45d of future price) powers the
        # honest scorecard + threshold selection
        frame = build_day_frame(series, pxmap, episodes, counts, sents)
        onset_frame = frame[frame.hype_raw >= 1].copy()

        # THE FA BUDGET IS A CONSTANT, not a reading off the last run:
        # loading it from euphoria_report.json here would race with the
        # euphoria stage rewriting that file in parallel and make the
        # adoption bar depend on stage finishing order. See the block
        # beside EUPHORIA_FA_BUDGET_PER_IY in src/config.py.
        fa_budget = EUPHORIA_FA_BUDGET_PER_IY

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

    # ---- THE GET IN / GET OUT PAIR (section 6) --------------------------
    # The dashboard's headline signals. Research passes re-run the full
    # walk-forward for both heads and refreeze their thresholds; live
    # passes score today's data at the frozen thresholds in seconds.
    boom = boom_state_frame(series, pxmap)
    fpx_live = frame_live.merge(boom, on=["name", "date"], how="left")
    # astype(bool) after the fill: the merge leaves an object column
    # holding True/False/NaN, and pandas 2.x deprecated silently
    # downcasting that back to bool on fillna. Saying the dtype out loud
    # keeps the behaviour identical and drops the FutureWarning.
    fpx_live["boom_state"] = fpx_live["boom_state"].eq(True)

    desk_path = _os.path.join(PROCESSED_DIR, "euphoria_desk_report.json")
    desk_stored = None
    if _os.path.exists(desk_path):
        try:
            desk_stored = _json.load(open(desk_path))
        except (ValueError, OSError):
            desk_stored = None
    desk_research = research or desk_needs_research(desk_stored,
                                                    data_max_year)
    _desk_lag = (None if desk_research
                 else desk_record_lags_data(desk_stored, data_max_year))

    fa_budget = EUPHORIA_FA_BUDGET_PER_IY      # frozen - see src/config.py

    # The deployed model family is selected, not assumed. A research pass
    # runs the full model tournament (analytics/ml_detector.py: the
    # rule-based baseline + logistic + monotone GBM + MLP + ensemble, all
    # walk-forward) and freezes the winner under the pre-stated criterion
    # (one family for both heads, combined AP lift). Live passes re-fit
    # the frozen family deterministically on full years strictly before
    # the data year (same convention as every frozen threshold here) and
    # score today at the frozen probability cut.
    from analytics import ml_detector as mld    # local: avoids cycle

    frame_j = None

    def _judged_frame():
        nonlocal frame_j
        if frame_j is None:
            frame_j = (frame if research
                       else build_day_frame(series, pxmap, episodes,
                                            counts, sents))
        return frame_j

    def _frozen_ml(cand_j, maker, label, mode, chooser=None):
        train = cand_j[cand_j["year"] < data_max_year]
        if train.empty:                        # very young data: use all
            train = cand_j
        fit = maker(label)
        train_scored = train.assign(score=fit(train, train,
                                              mld.DESK_ML_BANK))
        chooser = chooser or mld.choose_threshold_f1
        return chooser(train_scored, episodes, mode, fa_budget,
                       train["name"].nunique())

    def _frozen_ml_pair(cand_j, maker, label, mode):
        """(standard F1 cut, strict F0.5 cut, train-median re-arm level)
        from ONE train scoring. The re-arm level is the GET IN trigger's
        'the crowd must fully cool before another start call' floor
        (shaped-trigger convention)."""
        train = cand_j[cand_j["year"] < data_max_year]
        if train.empty:
            train = cand_j
        fit = maker(label)
        train_scored = train.assign(score=fit(train, train,
                                              mld.DESK_ML_BANK))
        n = train["name"].nunique()
        return (mld.choose_threshold_f1(train_scored, episodes, mode,
                                        fa_budget, n),
                mld.choose_threshold_strict(train_scored, episodes, mode,
                                            fa_budget, n),
                float(np.percentile(train_scored["score"].dropna(), 50)))

    if desk_research:
        from src.config import DESK_MODEL_FAMILY
        fj = _judged_frame()
        _fams = [DESK_MODEL_FAMILY] if DESK_MODEL_FAMILY else None
        tournament = mld.run_ml_tournament(fj, episodes, pxmap, sym_by,
                                           series=series, boom=boom,
                                           families=_fams)
        # With a family pinned there is exactly one candidate, so
        # pick_winner is a formality - but it still runs, so the pinned
        # path and the tournament path adopt through the SAME code and
        # cannot drift apart.
        model_name = mld.pick_winner(tournament)
        if DESK_MODEL_FAMILY and model_name != DESK_MODEL_FAMILY:
            raise RuntimeError(
                "pinned family %r did not survive selection (got %r) - "
                "this means the pinned family errored during its fit; "
                "check the tournament output above."
                % (DESK_MODEL_FAMILY, model_name))
        wf_out = tournament["get_out"][model_name]
        wf_in = tournament["get_in"][model_name]

        if model_name == "rules":
            fpx_j = fj.merge(boom, on=["name", "date"], how="left")
            fpx_j["boom_state"] = fpx_j["boom_state"].eq(True)
            end_j, onset_j = desk_candidacy(fpx_j)

            def _frozen(cand, fit, feats, mode):
                train = cand[cand["year"] < data_max_year]
                if train.empty:                # very young data: use all
                    train = cand
                train_scored = train.assign(score=fit(train, train, feats))
                return choose_threshold(train_scored, episodes, mode,
                                        fa_budget,
                                        train["name"].nunique())

            thr_out = _frozen(end_j, desk_end_fit, TOP_FEATURES, "top")
            thr_in = _frozen(onset_j, desk_onset_fit, ONSET_BANK, "onset")
            thr_out_strict = thr_in_strict = None   # strict = ML-only
            rearm_in = None
        else:
            cand_j = mld.attach_price_features(
                mld.candidate_frame(fj), series, pxmap)
            maker = mld.ML_FITS[model_name]
            thr_out, thr_out_strict, _ = _frozen_ml_pair(
                cand_j, maker, "y_top", "top")
            thr_in, thr_in_strict, rearm_in = _frozen_ml_pair(
                cand_j, maker, "y_onset", "onset")

        # a compact copy of the whole tournament rides in the record so
        # the dashboard can show the comparison table without re-running
        _tbl_keys = ("auroc", "ap", "ap_baseline", "captured",
                     "detectable", "capture_rate", "late",
                     "false_alarms", "fa_per_iy", "precision",
                     "median_lead_days", "n_alerts", "forward_returns",
                     "test_years")
        _tbl = {h: {m: {k: r.get(k) for k in _tbl_keys}
                    for m, r in tournament[h].items() if "error" not in r}
                for h in ("get_out", "get_in")}
        desk_stored = {
            "model": model_name,
            "get_out": {"live_threshold": thr_out,
                        "strict_threshold": thr_out_strict,
                        "walk_forward": wf_out},
            "get_in": {"live_threshold": thr_in,
                       "strict_threshold": thr_in_strict,
                       "rearm_level": rearm_in,
                       "walk_forward": wf_in},
            "conditioning": {
                "version": 2,
                "trigger": "shaped crossing (phase gates + re-arm + "
                           "63d spacing)",
                "get_out": "only after the 120d G2 boom bar; re-arms "
                           "at the cut",
                "get_in": "only before the boom completes; re-arms at "
                          "the train-median score",
                "separation": "no GET IN within 21d of a GET OUT, "
                              "either direction",
                "standard": "F1 cut",
                "strict": "F0.5 cut (precision weighted 2x)",
                "evidence": "reference/research_record/alert_shape_sweep.json",
                "decided": "2026-08-09",
            },
            "tournament": _tbl,
            "selection_rule": ("one family for both heads; combined "
                               "test AP lift; ties AUROC then fewer "
                               "false alarms"),
            "operating_point_rule": ("probability cut maximising "
                                     "episode-level F1 on train years"
                                     if model_name != "rules" else
                                     "baseline FA-budget rule"),
            "fa_budget_per_iy": fa_budget,
            "smooth_days": ROLL,
            "bank_get_out": (mld.DESK_ML_BANK if model_name != "rules"
                             else TOP_FEATURES),
            "bank_get_in": (mld.DESK_ML_BANK if model_name != "rules"
                            else ONSET_BANK),
        }
        with open(desk_path, "w") as f:
            _json.dump(desk_stored, f, indent=1, default=str)
        if verbose:
            print(f"  DESK research pass (winner: {model_name}): GET OUT "
                  f"cap {wf_out['captured']}/{wf_out['detectable']} FA "
                  f"{wf_out['false_alarms']} AP {wf_out['ap']} | GET IN "
                  f"cap {wf_in['captured']}/{wf_in['detectable']} late "
                  f"{wf_in['late']} FA {wf_in['false_alarms']} | frozen "
                  f"thr in {thr_in:.3f} / out {thr_out:.3f}")
    else:
        model_name = (desk_stored or {}).get("model", "rules")
        thr_out = float(desk_stored["get_out"]["live_threshold"])
        thr_in = float(desk_stored["get_in"]["live_threshold"])
        thr_out_strict = desk_stored["get_out"].get("strict_threshold")
        thr_in_strict = desk_stored["get_in"].get("strict_threshold")
        rearm_in = desk_stored["get_in"].get("rearm_level")
        _cond_v = (desk_stored.get("conditioning") or {}).get("version")
        if model_name != "rules" and (_cond_v != 2
                                      or thr_out_strict is None
                                      or thr_in_strict is None
                                      or rearm_in is None):
            # upgrade an older record in place: derive the strict cuts
            # and the GET IN re-arm level once, persist them
            cand_up = mld.attach_price_features(
                mld.candidate_frame(_judged_frame()), series, pxmap)
            maker_up = mld.ML_FITS[model_name]
            thr_out2, thr_out_strict, _ = _frozen_ml_pair(
                cand_up, maker_up, "y_top", "top")
            thr_in2, thr_in_strict, rearm_in = _frozen_ml_pair(
                cand_up, maker_up, "y_onset", "onset")
            desk_stored["get_out"]["strict_threshold"] = thr_out_strict
            desk_stored["get_in"]["strict_threshold"] = thr_in_strict
            desk_stored["get_in"]["rearm_level"] = rearm_in
            desk_stored["conditioning"] = {
                "version": 2,
                "trigger": "shaped crossing (phase gates + re-arm + "
                           "63d spacing)",
                "get_out": "only after the 120d G2 boom bar; re-arms "
                           "at the cut",
                "get_in": "only before the boom completes; re-arms at "
                          "the train-median score",
                "separation": "no GET IN within 21d of a GET OUT, "
                              "either direction",
                "standard": "F1 cut",
                "strict": "F0.5 cut (precision weighted 2x)",
                "evidence": "reference/research_record/alert_shape_sweep.json",
                "decided": "2026-08-09",
            }
            with open(desk_path, "w") as f:
                _json.dump(desk_stored, f, indent=1, default=str)
            if verbose:
                print(f"  desk record upgraded to conditioning v2 "
                      f"(strict in {thr_in_strict:.3f} / out "
                      f"{thr_out_strict:.3f}; GET IN re-arm "
                      f"{rearm_in:.3f})")
        if verbose:
            print(f"  DESK live mode ({model_name}): frozen thresholds "
                  f"in {thr_in:.3f} / out {thr_out:.3f}")
            if _desk_lag is not None:
                print(f"  NOTE: desk data reaches {data_max_year}; these "
                      f"thresholds were confirmed through {_desk_lag}. "
                      "Refresh the record deliberately with "
                      "analytics.run_analytics --what phases --research")

    if model_name == "rules":
        end_live, onset_live_f = desk_candidacy(fpx_live)
        end_scored = end_live.assign(
            dscore=desk_end_fit(end_live, end_live, TOP_FEATURES))
        onset_scored = onset_live_f.assign(
            dscore=desk_onset_fit(onset_live_f, onset_live_f, ONSET_BANK))
        # the inflection and price-blind heads are ML heads; the rules
        # fallback has neither
        inflection_scored, _ttrain_sc = None, None
        xp_in_scored = xp_out_scored = xp_train = None
    else:
        # ML path: candidacy is the coverage gate only - the hype, boom
        # and end-stage doors are FEATURES now, not gates
        cand_j = mld.attach_price_features(
            mld.candidate_frame(_judged_frame()), series, pxmap)
        train = cand_j[cand_j["year"] < data_max_year]
        if train.empty:
            train = cand_j
        live_cand = mld.attach_price_features(
            mld.candidate_frame(fpx_live), series, pxmap)
        maker = mld.ML_FITS[model_name]
        _fit_top, _fit_onset = maker("y_top"), maker("y_onset")
        end_scored = live_cand.assign(
            dscore=_fit_top(train, live_cand, mld.DESK_ML_BANK))
        onset_scored = live_cand.assign(
            dscore=_fit_onset(train, live_cand, mld.DESK_ML_BANK))
        # Persist THIS fit and the population it scored, so the
        # dashboard can score a name outside the universe (the ETF
        # lookup) with the same model at the same frozen cut. Read-only
        # from here on; a research pass rewrites it. Never fatal.
        try:
            _bundle_path = _os.path.join(PROCESSED_DIR, mld.DESK_BUNDLE)
            mld.save_desk_bundle(
                _bundle_path, {"y_top": _fit_top, "y_onset": _fit_onset},
                mld.DESK_ML_BANK,
                {"model": model_name, "data_max_year": int(data_max_year),
                 "thr_in": float(thr_in), "thr_out": float(thr_out),
                 "thr_in_strict": thr_in_strict,
                 "thr_out_strict": thr_out_strict, "rearm_in": rearm_in,
                 "live_rows": int(len(live_cand)),
                 "train_rows": int(len(train))})
            if verbose:
                print(f"  saved {mld.DESK_BUNDLE} (fitted {model_name} + "
                      f"live population, for the ETF lookup)")
        except Exception as _bd_e:                       # noqa: BLE001
            if verbose:
                print(f"  (model bundle skipped: {_bd_e})")
        # The inflection head - a third, independent score on
        # the same candidate frame, price-free bank, fitted the same way
        # and on the same train years. It reads nothing the other two
        # write and neither of them reads it back: if this head is ever
        # withdrawn, GET IN and GET OUT are bit-identical without it.
        if EUPHORIA_INFLECTION_ENABLED:
            _tl = inflection_label_frame(series, pxmap)
            _tbank = [c for c in mld.DESK_ML_BANK
                      if c not in mld.PRICE_FEATURES] + INFLECTION_EXTRA_FEATURES
            _ttrain = inflection_features(train).merge(_tl, on=["name", "date"],
                                                 how="left")
            _ttrain = _ttrain[_ttrain["y_inflection"].notna()].copy()
            _ttrain["y_inflection"] = _ttrain["y_inflection"].astype(int)
            _tlive = inflection_features(live_cand)
            if len(_ttrain) > 400 and _ttrain["y_inflection"].sum() >= 12:
                inflection_scored = _tlive.assign(
                    tscore=mld.make_ens_fit("y_inflection")(_ttrain, _tlive,
                                                      _tbank))
                _ttrain_sc = _ttrain.assign(
                    tscore=mld.make_ens_fit("y_inflection")(_ttrain, _ttrain,
                                                      _tbank))
            else:
                inflection_scored, _ttrain_sc = None, None
                if verbose:
                    print("  inflection head skipped: too few labelled train "
                          "days yet")
        else:
            inflection_scored, _ttrain_sc = None, None
        # The experimental price-blind pair: crowd-only by design. A
        # SECOND GET IN / GET OUT scoring with price removed from BOTH
        # places it enters the primary pair: the two price features are
        # dropped from the bank, and the 120d boom phase gate is dropped
        # from the trigger (see the alert block below). The bank is the
        # nine crowd features plus the four price-free inflection extras
        # - that addition measured +0.02 AP on GET IN with nothing given
        # up. LOGIT, not the ensemble the primary pair uses: the same
        # selection rule (combined AP lift, ties -> AUROC) picks logit on
        # the price-blind bank (frozen in
        # reference/research_record/nb08_price_blind.json). Everything
        # else - train years, F1/F0.5 cuts, re-arm, spacing, coherence
        # sweep - is the primary pair's machinery unchanged, so the two
        # modes differ ONLY in what the model is allowed to see.
        if EUPHORIA_XP_ENABLED:
            _xp_bank = list(mld.ML_BANK) + INFLECTION_EXTRA_FEATURES
            xp_cand_j = inflection_features(
                mld.candidate_frame(_judged_frame()))
            xp_train = xp_cand_j[xp_cand_j["year"] < data_max_year]
            if xp_train.empty:
                xp_train = xp_cand_j
            xp_live = inflection_features(mld.candidate_frame(fpx_live))
            xp_in_scored = xp_live.assign(
                dscore=mld.make_logit_fit("y_onset")(xp_train, xp_live,
                                                     _xp_bank))
            xp_out_scored = xp_live.assign(
                dscore=mld.make_logit_fit("y_top")(xp_train, xp_live,
                                                   _xp_bank))
        else:
            xp_in_scored = xp_out_scored = xp_train = None
        # the explainability sidecar the dashboard's "what drives the
        # calls" expander reads: logit weights + GBM permutation
        # importance for the live fit (one computation, every surface)
        try:
            insight = mld.model_insight(cand_j)
            insight["model"] = model_name
            with open(_os.path.join(PROCESSED_DIR,
                                    "desk_model_insight.json"), "w") as f:
                _json.dump(insight, f, indent=1)
            if verbose:
                print("  saved desk_model_insight.json (feature weights "
                      "for the dashboard)")
        except Exception as _e:                          # noqa: BLE001
            if verbose:
                print(f"  (model insight skipped: {_e})")

    # The shaped trigger (evidence
    # reference/research_record/alert_shape_sweep.json): phase gates from the
    # ground truth's own 120d boom bar, re-arm levels, 63d spacing.
    from src.config import EUPHORIA_ALERT_SPACING_D
    _b120 = boomed120_frame(series, pxmap)

    def _alert_dates(scored, thr, rearm, gate_boomed):
        """gate_boomed True -> fire only after the boom (GET OUT);
        False -> only before it (GET IN). rearm None -> re-arm at the
        cut."""
        by = {}
        sc = scored.merge(_b120, on=["name", "date"], how="left")
        sc["boomed120"] = sc["boomed120"].eq(True)
        for name, g in sc.sort_values("date").groupby("name"):
            gate = (g["boomed120"] if gate_boomed
                    else ~g["boomed120"]).tolist()
            by[name] = set(alerts_from_scores_shaped(
                g["date"].tolist(), g["dscore"].tolist(), gate, thr,
                rearm if rearm is not None else thr,
                EUPHORIA_ALERT_SPACING_D))
        return by

    in_alerts = _alert_dates(onset_scored, thr_in, rearm_in, False)
    out_alerts = _alert_dates(end_scored, thr_out, None, True)
    in_alerts_s = (_alert_dates(onset_scored, thr_in_strict, rearm_in,
                                False)
                   if thr_in_strict is not None else {})
    out_alerts_s = (_alert_dates(end_scored, thr_out_strict, None, True)
                    if thr_out_strict is not None else {})

    # Ungated GET IN (production; evidence
    # reference/research_record/nb08_single_dial.json). The 120d phase
    # gate was measured throwing away roughly three quarters of the GET
    # IN calls the model earns (captured episodes 28 -> 116 ungated, at
    # under double the false alarms): episodes chain, so a re-onset
    # routinely arrives while the name is still past the boom bar and
    # the IN side is dark. These columns are the SAME scores at the SAME
    # frozen cuts with the SAME shaped trigger - only the phase gate is
    # dropped. The dashboard shows them by DEFAULT; its "stricter
    # threshold (price gate)" checkbox switches back to the gated columns
    # above. GET OUT keeps its gate everywhere: removing it doubles false
    # alarms for a handful of captures, so no ungated OUT variant exists
    # on purpose.
    def _alert_dates_nogate(scored, thr, rearm):
        by = {}
        for name, g in scored.sort_values("date").groupby("name"):
            by[name] = set(alerts_from_scores_shaped(
                g["date"].tolist(), g["dscore"].tolist(),
                [True] * len(g), thr,
                rearm if rearm is not None else thr,
                EUPHORIA_ALERT_SPACING_D))
        return by

    in_alerts_ng = _alert_dates_nogate(onset_scored, thr_in, rearm_in)
    in_alerts_ng_s = (_alert_dates_nogate(onset_scored, thr_in_strict,
                                          rearm_in)
                      if thr_in_strict is not None else {})

    ds = fpx_live[["date", "name", "kind", "hype_raw",
                   "boom_state"]].copy()
    ds["end_stage"] = end_stage_mask(fpx_live).values
    ds = ds.merge(onset_scored[["name", "date", "dscore"]]
                  .rename(columns={"dscore": "in_score"}),
                  on=["name", "date"], how="left")
    ds = ds.merge(end_scored[["name", "date", "dscore"]]
                  .rename(columns={"dscore": "out_score"}),
                  on=["name", "date"], how="left")
    for _suffix, _ia, _oa in (("", in_alerts, out_alerts),
                              ("_strict", in_alerts_s, out_alerts_s)):
        ds[f"get_in{_suffix}"] = [d in _ia.get(n, ())
                                  for n, d in zip(ds["name"], ds["date"])]
        ds[f"get_out{_suffix}"] = [d in _oa.get(n, ())
                                   for n, d in zip(ds["name"],
                                                   ds["date"])]
    for _suffix, _ia in (("_nogate", in_alerts_ng),
                         ("_nogate_strict", in_alerts_ng_s)):
        ds[f"get_in{_suffix}"] = [d in _ia.get(n, ())
                                  for n, d in zip(ds["name"], ds["date"])]
    # Display coherence (invariant: a name never shows a GET IN and a GET
    # OUT close together): a START is never shown on an end-stage day,
    # and never within one cooldown of an END call IN EITHER DIRECTION.
    # GET OUT is never suppressed - it is the risk signal. The phase
    # gates make same-day contradiction structurally impossible; this
    # rule sweeps the boundary cases. Applied to every GET IN variant,
    # including the ungated pair - coherence is a display promise, not a
    # gate, so dropping the gate does not lift it (the ungated IN is
    # judged against the GATED OUT, the only OUT that exists).
    for _suffix, _go_suffix in (("", ""), ("_strict", "_strict"),
                                ("_nogate", ""),
                                ("_nogate_strict", "_strict")):
        _gi, _go = f"get_in{_suffix}", f"get_out{_go_suffix}"
        ds.loc[ds["end_stage"].astype(bool), _gi] = False
        for _n, _g in ds.groupby("name"):
            outs = _g.loc[_g[_go], "date"]
            if not len(outs):
                continue
            ins = _g.loc[_g[_gi], "date"]
            for d in ins:
                if any(abs((d - t).days) <= EUPHORIA_COOLDOWN_DAYS
                       for t in outs):
                    ds.loc[(ds["name"] == _n) & (ds["date"] == d),
                           _gi] = False
    # INFLECTION COLUMNS. Threshold and re-arm are score values frozen from
    # the TRAIN years' own distribution, exactly like the GET IN / GET OUT
    # cuts - a percentile taken on the live scores would move every run
    # and the marker could not be compared week to week.
    ds["inflection_score"] = np.nan
    ds["inflection"] = False
    if inflection_scored is not None and _ttrain_sc is not None:
        # SAME CONTRACT AS THE GET IN / GET OUT CUTS: the threshold is
        # FROZEN on disk and only re-derived on a research pass.
        # Recomputing it every live run would leave nothing on disk
        # describing how today's marker differs from yesterday's, and a
        # threshold nobody can reconstruct cannot be defended - the
        # reasoning is written out beside `needs_research`. The
        # bootstrap (no record yet) is the one exception, exactly as for
        # GET IN and GET OUT.
        _frozen_turn = (desk_stored or {}).get("inflection") or {}
        if research or "threshold" not in _frozen_turn:
            _thr_t = float(_ttrain_sc["tscore"].quantile(
                EUPHORIA_INFLECTION_CUT_Q))
            _rearm_t = float(_ttrain_sc["tscore"].quantile(
                EUPHORIA_INFLECTION_REARM_Q))
            _infl_src = "research" if research else "bootstrap"
        else:
            _thr_t = float(_frozen_turn["threshold"])
            _rearm_t = float(_frozen_turn["rearm"])
            _infl_src = "frozen"
        if verbose:
            print(f"  inflection head ({_infl_src}): threshold {_thr_t:.3f} / "
                  f"re-arm {_rearm_t:.3f}")
        _tmap = {}
        for _n, _g in inflection_scored.sort_values("date").groupby("name"):
            _tmap[_n] = set(inflection_alerts(_g["date"].tolist(),
                                        _g["tscore"].tolist(), _thr_t,
                                        _rearm_t,
                                        EUPHORIA_INFLECTION_SPACING_D))
        ds = ds.merge(inflection_scored[["name", "date", "tscore"]]
                      .rename(columns={"tscore": "inflection_score_new"}),
                      on=["name", "date"], how="left")
        ds["inflection_score"] = ds.pop("inflection_score_new")
        ds["inflection"] = [d in _tmap.get(n, ())
                      for n, d in zip(ds["name"], ds["date"])]
        if isinstance(desk_stored, dict) and _infl_src != "frozen":
            # A record written under this head's earlier name carries a
            # "turn" block; drop it rather than leave two thresholds in
            # one file, where the next reader has to guess which is live.
            desk_stored.pop("turn", None)
            desk_stored["inflection"] = {
                "threshold": _thr_t, "rearm": _rearm_t,
                "cut_q": EUPHORIA_INFLECTION_CUT_Q,
                "rearm_q": EUPHORIA_INFLECTION_REARM_Q,
                "spacing_d": EUPHORIA_INFLECTION_SPACING_D,
                "bank": _tbank,
                "evidence": "reference/research_record/inflection_trigger_sweep.json",
                "role": ("CONTEXT MARKER - never a call, never in the "
                         "watchlist, never gates GET IN or GET OUT"),
                "derived": _infl_src}
            with open(desk_path, "w") as _f:
                _json.dump(desk_stored, _f, indent=1, default=str)
    # EXPERIMENTAL PRICE-BLIND COLUMNS. Same frozen-threshold contract as
    # the GET IN / GET OUT cuts and the inflection head: the cuts live in
    # euphoria_desk_report.json, are re-derived only on research (or
    # once, on bootstrap), and every live run scores at them. The
    # trigger is the shaped trigger WITHOUT the phase gate - no price
    # anywhere between a post and a call. The end-stage suppression and
    # the 21d IN/OUT separation are kept: both are crowd-only reads
    # (end_stage_mask is e1/e2/hype - no price term).
    ds["in_score_xp"] = np.nan
    ds["out_score_xp"] = np.nan
    for _c in ("get_in_xp", "get_out_xp",
               "get_in_xp_strict", "get_out_xp_strict"):
        ds[_c] = False
    if xp_in_scored is not None and xp_train is not None:
        _frozen_xp = (desk_stored or {}).get("experimental_price_blind") \
            or {}
        if research or "get_in" not in _frozen_xp:
            _xp_rec = {"model": "logit", "bank": _xp_bank,
                       "derived": "research" if research else "bootstrap",
                       "trigger": "shaped crossing WITHOUT phase gates "
                                  "(re-arm + 63d spacing only)",
                       "evidence": "reference/research_record/nb08_price_blind.json",
                       "role": ("EXPERIMENTAL dashboard mode - crowd "
                                "features only, no price features, no "
                                "price gate. Never the default.")}
            for _hd, _lab, _md in (("get_in", "y_onset", "onset"),
                                   ("get_out", "y_top", "top")):
                _fit_xp = mld.make_logit_fit(_lab)
                _tr_sc = xp_train.assign(
                    score=_fit_xp(xp_train, xp_train, _xp_bank))
                _n_xp = xp_train["name"].nunique()
                _xp_rec[_hd] = {
                    "live_threshold": mld.choose_threshold_f1(
                        _tr_sc, episodes, _md, fa_budget, _n_xp),
                    "strict_threshold": mld.choose_threshold_strict(
                        _tr_sc, episodes, _md, fa_budget, _n_xp),
                    "rearm_level": float(
                        _tr_sc["score"].dropna().median())}
            if isinstance(desk_stored, dict):
                desk_stored["experimental_price_blind"] = _xp_rec
                with open(desk_path, "w") as _f:
                    _json.dump(desk_stored, _f, indent=1, default=str)
            _frozen_xp = _xp_rec
        if verbose:
            print(f"  price-blind pair ({_frozen_xp.get('derived')}): "
                  f"in {_frozen_xp['get_in']['live_threshold']:.3f}/"
                  f"{_frozen_xp['get_in']['strict_threshold']:.3f} out "
                  f"{_frozen_xp['get_out']['live_threshold']:.3f}/"
                  f"{_frozen_xp['get_out']['strict_threshold']:.3f}")

        def _xp_alerts(scored, thr, rearm):
            """No phase gate, deliberately - price routes nothing."""
            by = {}
            for _n, _g in scored.sort_values("date").groupby("name"):
                by[_n] = set(alerts_from_scores_shaped(
                    _g["date"].tolist(), _g["dscore"].tolist(),
                    [True] * len(_g), thr,
                    rearm if rearm is not None else thr,
                    EUPHORIA_ALERT_SPACING_D))
            return by

        _rin = _frozen_xp["get_in"].get("rearm_level")
        for _sfx, _thr_key in (("", "live_threshold"),
                               ("_strict", "strict_threshold")):
            _ia = _xp_alerts(xp_in_scored,
                             float(_frozen_xp["get_in"][_thr_key]), _rin)
            _oa = _xp_alerts(xp_out_scored,
                             float(_frozen_xp["get_out"][_thr_key]), None)
            ds[f"get_in_xp{_sfx}"] = [d in _ia.get(n, ())
                                      for n, d in zip(ds["name"],
                                                      ds["date"])]
            ds[f"get_out_xp{_sfx}"] = [d in _oa.get(n, ())
                                       for n, d in zip(ds["name"],
                                                       ds["date"])]
        ds = ds.merge(xp_in_scored[["name", "date", "dscore"]]
                      .rename(columns={"dscore": "in_score_xp_new"}),
                      on=["name", "date"], how="left")
        ds["in_score_xp"] = ds.pop("in_score_xp_new")
        ds = ds.merge(xp_out_scored[["name", "date", "dscore"]]
                      .rename(columns={"dscore": "out_score_xp_new"}),
                      on=["name", "date"], how="left")
        ds["out_score_xp"] = ds.pop("out_score_xp_new")
        # the same coherence sweep as the primary pair (both crowd-only
        # rules, so nothing here re-admits price)
        for _sfx in ("_xp", "_xp_strict"):
            _gi, _go = f"get_in{_sfx}", f"get_out{_sfx}"
            ds.loc[ds["end_stage"].astype(bool), _gi] = False
            for _n, _g in ds.groupby("name"):
                outs = _g.loc[_g[_go], "date"]
                if not len(outs):
                    continue
                for d in _g.loc[_g[_gi], "date"]:
                    if any(abs((d - t).days) <= EUPHORIA_COOLDOWN_DAYS
                           for t in outs):
                        ds.loc[(ds["name"] == _n) & (ds["date"] == d),
                               _gi] = False
        if verbose:
            print(f"  price-blind pair: {int(ds['get_in_xp'].sum())} GET "
                  f"IN / {int(ds['get_out_xp'].sum())} GET OUT all-time "
                  f"({int(ds['get_in_xp_strict'].sum())}/"
                  f"{int(ds['get_out_xp_strict'].sum())} strict)")
    ds = ds.merge(_b120, on=["name", "date"], how="left")
    ds["boomed120"] = ds["boomed120"].eq(True)
    # the display twin travels with it; absent on older bundles, where
    # the dashboard falls back to the raw gate
    if "boomed120_stable" in ds.columns:
        ds["boomed120_stable"] = ds["boomed120_stable"].eq(True)
    ds["symbol"] = ds["name"].map(sym_by)
    # Retail-flow dial (production; record
    # reference/research_record/nb08_retail_flow.json). Failure-isolated:
    # euphoria_desk.parquet must never be lost to a dial bug, so a dial
    # error degrades to missing columns and one printed line, never a
    # crash. Adds roughly 2-4 minutes to a phases run - it refits the
    # dial's walk-forward (deterministic, random_state=0) rather than
    # caching a model, the same recompute-from-stores policy every other
    # derived column follows.
    try:
        from analytics.retail_flow import attach_retail_flow
        if verbose:
            print("  retail-flow dial (notebook 08 §9, ~2-4 min):",
                  flush=True)
        ds = attach_retail_flow(ds, fpx_live, prices, sym_by,
                                verbose=verbose)
    except Exception as _rf_e:                            # noqa: BLE001
        if verbose:
            print(f"  (retail-flow dial skipped: {_rf_e})")
    # PER-DAY MODEL COMPONENTS: the DESK_ML_BANK readings per scored day,
    # so the dashboard hover can draw each day's reading-x-weight stack
    # (factor + factor + factor vs the threshold). Raw feature values
    # only - the frozen weights live in desk_model_insight.json; no
    # text, no ids, so the file is publishable. Failure-isolated like
    # the dial: a components bug must never cost euphoria_desk.parquet.
    try:
        _lc_cmp = live_cand
        _cmp_cols = [c for c in mld.DESK_ML_BANK
                     if c in _lc_cmp.columns]
        if len(_cmp_cols) >= 8:
            _cmp = (_lc_cmp[["date", "name"] + _cmp_cols]
                    .dropna(subset=_cmp_cols, how="all").copy())
            for _cc in _cmp_cols:
                _cmp[_cc] = _cmp[_cc].astype("float32")
            _safe_write(_cmp, _os.path.join(
                PROCESSED_DIR, "euphoria_desk_components.parquet"))
            if verbose:
                print(f"  saved euphoria_desk_components.parquet "
                      f"({len(_cmp):,} rows, {len(_cmp_cols)} "
                      "readings)")
    except NameError:
        pass               # rules fallback: no ML bank to publish
    except Exception as _cmp_e:                           # noqa: BLE001
        if verbose:
            print(f"  (components store skipped: {_cmp_e})")
    _safe_write(ds, _os.path.join(PROCESSED_DIR, "euphoria_desk.parquet"))
    if verbose:
        print(f"  saved euphoria_desk.parquet ({len(ds):,} rows, "
              f"{int(ds['get_in'].sum())} GET IN / "
              f"{int(ds['get_out'].sum())} GET OUT alerts all-time, "
              f"{int(ds['inflection'].sum())} inflection markers, "
              f"model {model_name})")
    # Readiness alerts: every name whose signed readiness crosses +/-90%.
    # Computed here so the alert file always matches the store it was
    # cut from; the dashboard banners it and the pipeline run prints it.
    # Signed readiness is the dashboard's display convention: the
    # phase-routed side's score over its frozen strict cut, negative for
    # GET OUT.
    try:
        _al_rows = []
        if thr_in_strict and thr_out_strict:
            # Themes only. The bands and the radar still show every
            # name; the ALERT - the thing that interrupts - is reserved
            # for the tradeable theme ETFs, not single-name tickers.
            _dsr = ds[ds["kind"] == "theme"].dropna(
                subset=["in_score", "out_score"])
            _fresh_bar = ds["date"].max() - pd.Timedelta(days=60)
            for _n, _g in _dsr.groupby("name"):
                _c = _g.sort_values("date").iloc[-1]
                # STALE ROWS DO NOT ALERT: a name whose last scored day
                # is months old (it left the scored universe) would
                # otherwise ping forever on a frozen reading - the same
                # 60-day rule the ETF radar sorts by.
                if pd.Timestamp(_c["date"]) < _fresh_bar:
                    continue
                _sr = (-float(_c["out_score"]) / float(thr_out_strict)
                       if bool(_c.get("boomed120", False))
                       else float(_c["in_score"]) / float(thr_in_strict))
                if abs(_sr) >= 0.90:
                    _al_rows.append({
                        "name": _n, "symbol": _c.get("symbol"),
                        "signed_readiness": round(float(_sr), 3),
                        "side": ("GET OUT" if _sr < 0 else "GET IN"),
                        "as_of": str(pd.Timestamp(_c["date"]).date())})
        _al_rows.sort(key=lambda r: r["signed_readiness"])
        with open(_os.path.join(PROCESSED_DIR,
                                "readiness_alerts.json"), "w",
                  encoding="utf-8") as _fh:
            _json.dump({"threshold": 0.90, "built":
                        pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M"),
                        "alerts": _al_rows}, _fh, indent=1)
        if verbose and _al_rows:
            print("  READINESS ALERTS (|signed readiness| >= 90% of the "
                  "cut):")
            for _r in _al_rows:
                print(f"    {_r['name']:<20} {_r['side']:<8} "
                      f"{100 * _r['signed_readiness']:+.0f}%  "
                      f"(as of {_r['as_of']})")
        elif verbose:
            print("  readiness alerts: none at the ±90% line")
    except Exception as _ra_e:                            # noqa: BLE001
        if verbose:
            print(f"  (readiness alerts skipped: {_ra_e})")
    return stored


# ---------------------------------------------------------------------------
# 6. THE GET IN / GET OUT PAIR - the signal family the dashboard shows.
#    This is the rule-based configuration of the pair; it is the
#    tournament baseline and the fallback when no learned family is
#    adopted (model "rules" in euphoria_desk_report.json).
#
#    This family is allowed price. It is a SECOND, clearly labelled signal
#    family: the crowd-only detectors above are unchanged - their claim
#    ("the crowd alone called it") is different, not worse - and they
#    remain the research baseline.
#
#    GET OUT (euphoria ending) = the top-detector rules score with two
#    measured upgrades (reference/research_record/nb06_desk_config.json):
#      * candidacy requires an ACTUAL PRICE BOOM - G2's own size
#        thresholds above the trailing EUPHORIA_BOOM_WINDOW_D low, past
#        prices only; no new constant. Walk-forward: capture 16 -> 26 of
#        122, AP 0.286 -> 0.435, capture-gain CI [+3.5pp, +13pp].
#      * the trigger runs on the ROLL-day (7d) SMOOTHED score: AP 0.435
#        -> 0.449, FA 41 -> 39, at a recorded cost of 2 captures (26 ->
#        24). One loud afternoon can no longer fire a one-day episode.
#
#    GET IN (euphoria starting) = the tournament-winning onset rules with
#    PHASE-AWARE candidacy + the same 7d smoothing: a day that already
#    satisfies every END gate (A1 crowd swollen AND A2 attention >= its
#    90th pct AND A3's persistence e2 > 0 - the detector's own existing
#    gates, no new constant) is END-STAGE, and declaring a START there is
#    definitionally incoherent. Measured (same record):
#    START-within-21d-of-END adjacency 20 -> 2, LATE starts 21 -> 10, FA
#    169 -> 124, at a recorded cost of captures 29 -> 20 of 125. A START
#    landing on top of an END is the error that destroys user trust in
#    the signal, so the adjacency priority overrules the raw-capture
#    utility rule; the cost is recorded, not hidden.
# ---------------------------------------------------------------------------
from src.config import (ROLL, EUPHORIA_ATT_GATE,  # noqa: E402
                        EUPHORIA_BOOM_MIN_ETF, EUPHORIA_BOOM_MIN_SINGLE,
                        EUPHORIA_BOOM_WINDOW_D, EUPHORIA_BOOM_WINDOW_MIN_D,
                        EUPHORIA_ONSET_HYPE_MIN)


def boom_state_frame(series: list, pxmap: dict) -> pd.DataFrame:
    """The prediction-time boom gate for GET OUT candidacy.

    Returns name/date/boom_state: is the price >= its G2 boom threshold
    above its own trailing EUPHORIA_BOOM_WINDOW_D low? Trailing only (day
    t uses closes <= t); the SIZE thresholds are the ground-truth
    constants, reused, so the gate introduces no new size number.

    THE WINDOW IS NOT THE GROUND TRUTH'S WINDOW. `find_episodes` walks
    back EUPHORIA_BOOM_LOOKBACK_D (120) days because that is the
    yardstick an episode is DEFINED by; this gate walks back
    EUPHORIA_BOOM_WINDOW_D (54) because that is a prediction-time choice,
    and 120 let crash-rebounds through: a name 20-25% below its own
    prior peak, bouncing off a bear-market bottom, could still clear a
    120d bar and fire GET OUT. The window was swept; the table, the
    selection rule and the recorded cost (the walk-forward loses its
    earliest test year) are in src/config.py beside the constant.
    Deliberately two windows, deliberately not shared.

    Args:
        series: EuphoriaSeries objects, one per instrument.
        pxmap: symbol -> daily close series.

    Returns:
        A DataFrame with columns name, date, boom_state (bool).
    """
    rows = []
    for es in series:
        px = pxmap[es.symbol].dropna().asfreq("D").ffill()
        low_w = px.rolling(EUPHORIA_BOOM_WINDOW_D,
                           min_periods=EUPHORIA_BOOM_WINDOW_MIN_D).min()
        bm = (EUPHORIA_BOOM_MIN_SINGLE if es.kind == "single"
              else EUPHORIA_BOOM_MIN_ETF)
        rows.append(pd.DataFrame({"name": es.name, "date": px.index,
                                  "boom_state": ((px / low_w - 1) >= bm)
                                  .values}))
    return pd.concat(rows, ignore_index=True)


def end_stage_mask(df: pd.DataFrame) -> pd.Series:
    """END-STAGE mask: the day already satisfies every END gate.

    A1 (hype_ok) + A2 (e1 >= EUPHORIA_ATT_GATE) + A3 persistence
    (e2 > 0). Built ONLY from the detector's own frozen gate constants -
    no new threshold enters the system.
    """
    return ((df["e1"] >= EUPHORIA_ATT_GATE) & (df["e2"] > 0)
            & df["hype_ok"].astype(bool))


def _smooth_by_name(scores: pd.Series, names: pd.Series,
                    dates: pd.Series | None = None) -> pd.Series:
    """Trigger smoothing for the GET IN / GET OUT pair.

    A trailing ROLL-day (7d - the house one-week window, the same as A1's
    mention-share window) mean over each instrument's own candidate
    days. Trailing => no look-ahead.

    CALENDAR-AWARE when `dates` is given, and the distinction matters.
    Rolling over each name's candidate-ROW sequence (`rolling(ROLL)` =
    the last 7 rows) silently bridges a multi-year candidacy gap: a
    7-row window can contain six candidate days from an episode years
    earlier plus one gate-zeroed day from today, so a call fires on
    evidence from a mania long past. With `dates`, the window is the
    last ROLL CALENDAR days, closed on the right, so evidence older than
    one week can never reach a trigger. On a dense daily candidate run
    the two windows contain identical rows, so ordinary in-episode
    behaviour is unchanged; only gap-bridging is removed. Thresholds
    were re-frozen through the standard walk-forward on this basis (see
    euphoria_desk_report.json).

    Args:
        scores: The per-row scores.
        names: Instrument name per row (aligned with `scores`).
        dates: Date per row. When None, the row-based window is used
            (the unit contract for callers without dates).

    Returns:
        The smoothed scores, indexed like `scores`.
    """
    if dates is None:
        return scores.groupby(names.values).transform(
            lambda g: g.rolling(ROLL, min_periods=1).mean())
    frame = pd.DataFrame({"s": scores.values,
                          "d": pd.to_datetime(pd.Series(dates).values),
                          "n": pd.Series(names).values},
                         index=scores.index)
    out = pd.Series(index=scores.index, dtype="float64")
    for _, g in frame.groupby("n", sort=False):
        g2 = g.sort_values("d")
        s = pd.Series(g2["s"].values, index=pd.DatetimeIndex(g2["d"]))
        sm = s.rolling(f"{ROLL}D", min_periods=1).mean()
        out.loc[g2.index] = sm.values
    return out


def desk_end_fit(train, apply, feats):
    """The rule-based GET OUT score (fitting is a no-op).

    Mean of the top-detector bank (`feats`, normally TOP_FEATURES),
    zeroed where the A2/A3 gates fail (gates applied in score space, so
    one threshold governs), then 7-calendar-day smoothed. Matches the
    fit_score(train, apply, feats) protocol; `train` is ignored.
    """
    sc = apply[feats].mean(axis=1)
    sc = sc.where((apply["e1"] >= EUPHORIA_ATT_GATE) & (apply["e2"] > 0),
                  0.0)
    return _smooth_by_name(sc, apply["name"], apply["date"]).values


def desk_onset_fit(train, apply, feats):
    """The rule-based GET IN score (fitting is a no-op).

    The tournament-winning onset rules (mean of the locked bank),
    7-calendar-day smoothed. Phase-awareness lives in CANDIDACY (the
    frame passed in), not in the score. `train` is ignored.
    """
    return _smooth_by_name(apply[feats].mean(axis=1),
                           apply["name"], apply["date"]).values


def desk_candidacy(frame_px: pd.DataFrame) -> tuple:
    """The two rule-based candidate frames for the GET IN / GET OUT pair.

    THE ONSET FLOOR IS EUPHORIA_ONSET_HYPE_MIN (1.10), NOT 1.0: at 1.0
    this rule breached its own false-alarm budget (0.255 vs 0.23). The
    sweep, the cost (one capture) and what it buys (budget compliance,
    late starts 6 -> 2) are in src/config.py beside the constant. The
    CROWD-ONLY onset store still uses 1.0 on purpose: different detector,
    different record, not swept here.

    Args:
        frame_px: A day frame already merged with boom_state.

    Returns:
        (end_frame, onset_frame): GET OUT candidates are hype_ok AND
        boom_state days; GET IN candidates are days with hype_raw >=
        EUPHORIA_ONSET_HYPE_MIN that are not end-stage.
    """
    end_f = frame_px[frame_px["hype_ok"].astype(bool)
                     & frame_px["boom_state"].astype(bool)].copy()
    onset_f = frame_px[(frame_px["hype_raw"] >= EUPHORIA_ONSET_HYPE_MIN)
                       & ~end_stage_mask(frame_px)].copy()
    return end_f, onset_f


def _desk_test_years(stored: dict | None) -> list:
    """Every walk-forward test year the stored GET IN / GET OUT record covers.

    Shared by the bootstrap test and the staleness notice so the two can
    never disagree about what the record contains.
    """
    if not stored or "get_in" not in stored or "get_out" not in stored:
        return []
    years = []
    for k in ("get_in", "get_out"):
        years += stored[k].get("walk_forward", {}).get("test_years") or []
    return years


def desk_needs_research(stored: dict | None, data_max_year: int) -> bool:
    """Must a data pull derive the GET IN / GET OUT thresholds itself?

    Same convention as onset_needs_research: research only to BOOTSTRAP
    a copy with no usable frozen record. A record that lags the data is
    a notice, not a refit - see `desk_record_lags_data`.
    """
    return not _desk_test_years(stored)


def desk_record_lags_data(stored: dict | None, data_max_year: int):
    """Newest GET IN / GET OUT walk-forward test year when it lags the
    data, else None."""
    years = _desk_test_years(stored)
    if not years:
        return None
    newest = max(int(y) for y in years)
    return newest if data_max_year > newest else None


# ---------------------------------------------------------------------------
# 7. EPISODE COHERENCE - the display-facing state machine
# ---------------------------------------------------------------------------
def episode_coherent_alerts(onset_dates, top_dates,
                            cooldown: int = EUPHORIA_COOLDOWN_DAYS):
    """Suppress START calls that contradict a recent END call.

    The rule is ASYMMETRIC by evidence: a new START within `cooldown`
    days AFTER an END is a contradictory flip and is SUPPRESSED (a
    euphoria that was just declared ending cannot be starting); a fast
    START -> END is a REAL, violent mania and the ENDING (risk) signal
    is NEVER suppressed.

    The asymmetry was measured, not assumed: the symmetric rule cost the
    top detector half its walk-forward captures (17 -> 9) for only 8
    fewer false alarms - fast episodes legitimately run start-to-end
    inside one cooldown - while the onset direction cost 2 captures and
    removed 7 FAs. The cooldown (21d) is the project's existing
    one-episode timescale; no new constant. Same-day tie: the END wins
    and the same-day START is suppressed.

    Args:
        onset_dates: Candidate START dates.
        top_dates: END dates (never filtered).
        cooldown: Days after an END during which a START is suppressed.

    Returns:
        (kept_onset_dates, kept_top_dates), both chronological.
    """
    tops = sorted(pd.Timestamp(d) for d in top_dates)
    kept_onset = []
    for d in sorted(pd.Timestamp(x) for x in onset_dates):
        blocked = any(0 <= (d - t).days < cooldown for t in tops)
        if not blocked:
            kept_onset.append(d)
    return kept_onset, tops
