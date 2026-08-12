"""
ml_detector.py
==============
The LEARNED euphoria detectors (August 2026) - one shared crowd-only
feature bank, two heads (GET IN / GET OUT), three model families, judged
by the SAME walk-forward discipline as everything else in this project.

WHY THIS EXISTS (desk brief, 2026-08-07)
----------------------------------------
The incumbent desk signals are hand rules stacked on gates, and every
gate carries a constant that has to be defended one by one in a
presentation (2.0x hype, 1.10x onset floor, 0.90 attention percentile,
75% persistence, a 0.23 FA budget, a 0.5 FA penalty...). The brief:
replace the rule/threshold stack with a model that (a) is at least as
accurate, (b) needs only a handful of explainable numbers, and (c) is
technically presentable ("not just rules-based"). No prior FA budget is
imposed - the operating point is chosen for accuracy alone.

WHAT SURVIVES OF THE OLD SYSTEM (three numbers, each one sentence)
------------------------------------------------------------------
1. COVERAGE FLOOR - a name must carry >= EUPHORIA_MIN_COVERAGE scored
   posts in the last 28d before anything can fire ("we do not diagnose a
   crowd we cannot see").
2. COOLDOWN - one alert per name per 21 days ("one call per episode").
3. THE TRIGGER - a probability cut chosen walk-forward on PAST years by
   maximising F1 (the standard balance of precision and recall - no
   hand-set budget, no penalty constant).
Everything else - the hype gates, the boom gates, the persistence gate,
the end-stage exclusion - is handed to the model as FEATURES, so the
data decides how much each matters instead of a hard-coded constant.

THE FEATURE BANK (9 features, all crowd-only, all trailing, all
percentile-ranked against the SAME name's own history - no embedded
gate constants)
----------------------------------------------------------------
  attention_level      how loud is the name vs its own last year (E1)
  attention_change     is the crowd bigger than a month ago (E3)
  influx_speed         is the crowd bigger than a fortnight ago (O4)
  attention_accel      is this week busier than this month (O1)
  hype_ratio           this week vs the name's own 120d norm (O2)
  attention_convexity  is attention growth itself accelerating -
                       Sornette's super-exponential signature (E5/O5)
  bull_level           how bullish is the mood vs its own year
  bull_persist         what fraction of the last month leaned bullish
  bull_inflection      is the mood turning (O3)

THE MODELS (the tournament; criterion pre-stated below)
-------------------------------------------------------
  rules     the incumbent desk configuration, unchanged (baseline)
  logit     L2 logistic regression - one weight per feature, readable
            as a formula
  gbm       gradient-boosted trees with MONOTONE constraints: every
            feature is constrained so that MORE crowd-heat can only
            RAISE the score. That single design choice is what makes a
            300-tree ensemble presentable - the model physically cannot
            learn "high attention is sometimes safe", so its behaviour
            is globally directional like a rule, while the tree
            structure learns the interactions the rules hard-coded.
            (Standard technique: monotonic gradient boosting, e.g.
            XGBoost/LightGBM monotone_constraints; here sklearn's
            HistGradientBoostingClassifier.)
  mlp       a small neural network (16-8 hidden units) - the ceiling
            check: if it beats gbm materially, structure is being
            missed; if not, gbm is capturing what is learnable.

ADOPTION CRITERION (stated before the numbers were computed): the
walk-forward TEST-YEAR Average Precision (threshold-free score quality,
the thesis convention) decides the winner per head; ties break by
AUROC, then by fewer false alarms at the chosen operating point. The
winner replaces the incumbent desk fit only if it beats the incumbent's
AP on the same test years.

Price NEVER enters any feature (crowd-only rule unchanged) - price
appears only in the ground truth and the scoring, exactly as before.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.config import (PROCESSED_DIR, PRICES_PATH,
                        EUPHORIA_COOLDOWN_DAYS)
from analytics.euphoria import build_all_series, judgeable_window
from analytics.euphoria_phases import (episode_catalog, build_day_frame,
                                       run_tournament_entry, _pregroup,
                                       _tally, desk_end_fit,
                                       desk_onset_fit, desk_candidacy,
                                       boom_state_frame)
from analytics.loaders import (load, THEME_COUNTS, THEME_SENT,
                               TICKER_COUNTS, TICKER_SENT)

# ---------------------------------------------------------------------------
# the bank (one list, two heads - names as documented above)
# ---------------------------------------------------------------------------
ML_BANK = ["attention_accel", "hype_ratio", "bull_inflection",
           "influx_speed", "attention_convexity", "e1", "e3",
           "bull_level", "bull_persist"]

# THE PRICE PAIR (desk heads only). The desk configuration lifted the
# crowd-only restriction in July 2026 ("we should be using both price
# and the social media"); the incumbent spent that licence on a HARD
# boom gate (>=25%/50% above the 54d low - two more constants). The ML
# bank spends it on two CONTINUOUS features and lets the model find the
# cut: measured 2026-08-07, they move GET OUT AUROC 0.63 -> 0.74 and
# GET IN 0.58 -> 0.77 walk-forward. The crowd-only variant is still run
# and reported alongside (the "crowd alone" claim keeps its own record).
PRICE_FEATURES = ["price_runup", "price_ret21"]
DESK_ML_BANK = ML_BANK + PRICE_FEATURES

# plain-English labels for charts/tables (kept beside the bank so the
# dashboard and the deck can never drift from the actual column names)
ML_BANK_LABELS = {
    "e1": "attention level (vs own year)",
    "e3": "attention change (1 month)",
    "influx_speed": "attention change (2 weeks)",
    "attention_accel": "week vs month",
    "hype_ratio": "week vs own normal",
    "attention_convexity": "attention accelerating",
    "bull_level": "bullishness level",
    "bull_persist": "bullishness persistence",
    "bull_inflection": "mood turning",
    "price_runup": "price run-up off its 54d low",
    "price_ret21": "price return, 1 month",
}

RANDOM_STATE = 0     # determinism: same data -> same fitted model


def price_feature_frame(series: list, pxmap: dict) -> pd.DataFrame:
    """name/date/price_runup/price_ret21 - trailing only (day t uses
    closes <= t). The run-up window is EUPHORIA_BOOM_WINDOW_D (54), the
    same one the incumbent's boom gate already uses - no new constant."""
    from src.config import (EUPHORIA_BOOM_WINDOW_D,
                            EUPHORIA_BOOM_WINDOW_MIN_D)
    rows = []
    for es in series:
        px = pxmap[es.symbol].dropna().asfreq("D").ffill()
        low = px.rolling(EUPHORIA_BOOM_WINDOW_D,
                         min_periods=EUPHORIA_BOOM_WINDOW_MIN_D).min()
        rows.append(pd.DataFrame({
            "name": es.name, "date": px.index,
            "price_runup": (px / low - 1).values,
            "price_ret21": px.pct_change(21).values}))
    return pd.concat(rows, ignore_index=True)


def attach_price_features(frame: pd.DataFrame, series: list,
                          pxmap: dict) -> pd.DataFrame:
    """Merge the price pair onto a day frame (rows without a judgeable
    price feature drop - they could never have been scored live)."""
    pf = price_feature_frame(series, pxmap)
    return (frame.merge(pf, on=["name", "date"], how="left")
            .dropna(subset=PRICE_FEATURES))


# ---------------------------------------------------------------------------
# model fits (the fit_score(train, apply, feats) protocol the tournament
# machinery already speaks)
# ---------------------------------------------------------------------------
def _balanced_weights(y: np.ndarray) -> np.ndarray:
    """class_weight='balanced' as explicit sample weights (HistGBM takes
    weights, not a class_weight argument)."""
    pos = max(y.sum(), 1)
    neg = max(len(y) - y.sum(), 1)
    w = np.where(y > 0, len(y) / (2 * pos), len(y) / (2 * neg))
    return w


def _label_col(train: pd.DataFrame) -> str:
    """The label this head is being fitted on (exactly one is attached
    by the caller via .attrs)."""
    return train.attrs.get("label", "y_top")


def make_logit_fit(label: str):
    from sklearn.linear_model import LogisticRegression

    def fit(train, apply, feats):
        y = train[label].values
        if len(np.unique(y)) < 2:
            return np.zeros(len(apply))
        m = LogisticRegression(class_weight="balanced", max_iter=2000)
        m.fit(train[feats], y)
        return m.predict_proba(apply[feats])[:, 1]
    fit.__name__ = f"logit_{label}"
    return fit


def make_gbm_fit(label: str):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit(train, apply, feats):
        y = train[label].values
        if len(np.unique(y)) < 2:
            return np.zeros(len(apply))
        m = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=200,
            monotonic_cst=[1] * len(feats),   # more heat -> more risk, only
            random_state=RANDOM_STATE)
        m.fit(train[feats], y, sample_weight=_balanced_weights(y))
        return m.predict_proba(apply[feats])[:, 1]
    fit.__name__ = f"gbm_{label}"
    return fit


def make_ens_fit(label: str):
    """logit + gbm rank ensemble: each model scores, the two probability
    RANKINGS are averaged (rank space, so neither model's calibration
    dominates). The classic two-heads-are-better check."""
    lg, gb = make_logit_fit(label), make_gbm_fit(label)

    def fit(train, apply, feats):
        a = pd.Series(lg(train, apply, feats)).rank(pct=True)
        b = pd.Series(gb(train, apply, feats)).rank(pct=True)
        return ((a + b) / 2).values
    fit.__name__ = f"ens_{label}"
    return fit


def make_mlp_fit(label: str):
    from sklearn.neural_network import MLPClassifier

    def fit(train, apply, feats):
        y = train[label].values
        if len(np.unique(y)) < 2:
            return np.zeros(len(apply))
        m = MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-3,
                          max_iter=200, early_stopping=True,
                          random_state=RANDOM_STATE)
        # MLPClassifier has no class weighting: rebalance by resampling
        # the positives (with replacement, capped) toward the negatives
        pos = train[y > 0]
        neg = train[y == 0]
        if len(pos) and len(neg) > len(pos):
            k = min(len(neg) // max(len(pos), 1), 5)
            boost = pd.concat([pos] * k, ignore_index=False)
            fit_df = pd.concat([neg, pos, boost])
        else:
            fit_df = train
        m.fit(fit_df[feats], fit_df[label].values)
        return m.predict_proba(apply[feats])[:, 1]
    fit.__name__ = f"mlp_{label}"
    return fit


# the registry the production path (euphoria_phases.rebuild_phase_files)
# dispatches on - the stored desk report names one of these
ML_FITS = {"logit": make_logit_fit, "gbm": make_gbm_fit,
           "mlp": make_mlp_fit, "ens": make_ens_fit}


# ---------------------------------------------------------------------------
# the budget-free operating point: maximise F-beta on the TRAIN years
# ---------------------------------------------------------------------------
def _choose_threshold_fbeta(train_scored: pd.DataFrame,
                            episodes: pd.DataFrame, mode: str,
                            beta: float) -> float:
    """Among percentile thresholds of the train scores, pick the one
    maximising episode-level F-beta on the train years:
      precision = captured / (captured + false alarms)
      recall    = captured / detectable episodes
      F-beta    = (1+b^2) * P * R / (b^2 * P + R)
    beta=1 balances the two (the STANDARD operating point); beta=0.5
    weights precision twice as heavily (the STRICT operating point -
    fewer false alarms, fewer captures). Tie -> the more conservative
    (higher) threshold."""
    years = sorted(train_scored.year.unique())
    in_years = lambda eps: eps.year.isin(years)          # noqa: E731
    grid = np.unique(np.percentile(train_scored["score"].dropna(),
                                   np.arange(50, 100, 2.5)))
    groups = _pregroup(train_scored, episodes)
    b2 = beta * beta
    best_thr, best_f = float(grid[-1]), -1.0
    for thr in grid[::-1]:                                # conservative first
        r = _tally(groups, episodes, thr, mode, in_years)
        pd_, rd = r["captured"] + r["false_alarms"], r["detectable"]
        if not pd_ or not rd:
            continue
        p, rec = r["captured"] / pd_, r["captured"] / rd
        denom = b2 * p + rec
        f = (1 + b2) * p * rec / denom if denom else 0.0
        if f > best_f:
            best_f, best_thr = f, float(thr)
    return best_thr


def choose_threshold_f1(train_scored: pd.DataFrame, episodes: pd.DataFrame,
                        mode: str, fa_budget_per_iy: float,
                        n_instruments: int) -> float:
    """Chooser for run_tournament_entry (same signature as the incumbent
    choose_threshold; fa_budget_per_iy is accepted and IGNORED - that is
    the point): episode-level F1 on the train years."""
    return _choose_threshold_fbeta(train_scored, episodes, mode, beta=1.0)


def choose_threshold_strict(train_scored: pd.DataFrame,
                            episodes: pd.DataFrame, mode: str,
                            fa_budget_per_iy: float,
                            n_instruments: int) -> float:
    """The STRICT operating point (desk request 2026-08-09: 'a stricter
    setting with fewer false alarms'): identical machinery, F0.5 - the
    standard precision-weighted F-measure, no new constant beyond the
    textbook beta=0.5. Same walk-forward convention, same grid."""
    return _choose_threshold_fbeta(train_scored, episodes, mode, beta=0.5)


# ---------------------------------------------------------------------------
# WITHDRAWN 2026-08-10 - the "Max Performance" operating point.
# The desk asked for a cut chosen purely for outcome (most negative
# forward move after a GET OUT). It was built and measured walk-forward
# against the DATE-MATCHED EXCESS move - the name's forward return minus
# what the whole tracked universe did that day, because the universe
# itself drifts ~+1% per 21d and a RAW median can never go negative in
# a bull run. Verdict: the edge it found on the train years did not
# survive out of sample (GET OUT +0.1%, GET IN 0.0% excess), while the
# plain precision-weighted STRICT cut delivered -1.4% / +0.7%. Removed
# at the desk's instruction; the measurement stays in
# docs/research/max_performance.json as the record of a tested null.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# forward-return battery: what did price do after the alerts?
# (evaluation only - the finance-audience numbers; never a feature)
# ---------------------------------------------------------------------------
FWD_HORIZONS = (5, 21, 84)


def forward_returns(alerts_by_name: dict, sym_by: dict, pxmap: dict,
                    horizons=FWD_HORIZONS) -> dict:
    """Median % price move after each alert, per horizon (calendar days),
    with the count of judgeable alerts. GET OUT alerts done well should
    show flat-to-negative medians; GET IN alerts positive ones."""
    moves = {h: [] for h in horizons}
    for name, alerts in alerts_by_name.items():
        sym = sym_by.get(name)
        px = pxmap.get(sym)
        if px is None or px.dropna().empty:
            continue
        px = px.dropna()
        for a in alerts:
            a = pd.Timestamp(a)
            p0 = px.asof(a)
            if pd.isna(p0) or p0 == 0:
                continue
            for h in horizons:
                target = a + pd.Timedelta(days=h)
                if target > px.index.max():
                    continue                       # pending, not evidence
                p1 = px.asof(target)
                if pd.notna(p1):
                    moves[h].append((p1 / p0 - 1) * 100)
    out = {}
    for h in horizons:
        r = pd.Series(moves[h])
        out[f"fwd_{h}d"] = {
            "n": int(len(r)),
            "median_pct": round(float(r.median()), 2) if len(r) else None,
            "mean_pct": round(float(r.mean()), 2) if len(r) else None,
            "pct_positive": round(float((r > 0).mean() * 100), 1)
            if len(r) else None,
        }
    return out


# ---------------------------------------------------------------------------
# the tournament (research entry point)
# ---------------------------------------------------------------------------
def _load_all():
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    series, pxmap = build_all_series(prices)
    counts = {"theme": load(THEME_COUNTS), "ticker": load(TICKER_COUNTS)}
    sents = {"theme": load(THEME_SENT), "ticker": load(TICKER_SENT)}
    for d in list(counts.values()) + list(sents.values()):
        d["date"] = pd.to_datetime(d["date"])
    return prices, series, pxmap, counts, sents


def candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """ML candidacy = the coverage gate only (already applied inside
    build_day_frame) - NO hype gate, NO boom gate, NO end-stage
    exclusion. Those become features, not doors."""
    return frame.dropna(subset=ML_BANK)


def run_ml_tournament(frame: pd.DataFrame, episodes: pd.DataFrame,
                      pxmap: dict, sym_by: dict,
                      series=None, boom=None) -> dict:
    """Every model through both heads; returns the full comparison table
    (the deck's Table 1). The incumbent rules run on their OWN candidacy
    (gates and budget intact - that IS the incumbent); the learners run
    on the coverage-only frame with the F1 chooser. Each learner is run
    twice: crowd-only (the clean-claim record) and crowd+price (the
    desk configuration the winner is drawn from)."""
    from src.config import EUPHORIA_FA_BUDGET_PER_IY

    cand = candidate_frame(frame)
    cand_px = (attach_price_features(cand, series, pxmap)
               if series is not None else None)
    results = {"get_out": {}, "get_in": {}}

    entries = [("logit", make_logit_fit), ("gbm", make_gbm_fit),
               ("mlp", make_mlp_fit), ("ens", make_ens_fit)]
    import time as _time
    _t_all = _time.time()
    _done = 0
    _total = len(entries) * 2 * (2 if cand_px is not None else 1)
    print(f"  model tournament: {_total} fits to run "
          f"(4 families x 2 heads x crowd-only/crowd+price). This is "
          f"the slow stage - typically 10-20 min on a laptop; every "
          f"fit prints as it starts.", flush=True)
    for head, label, mode in (("get_out", "y_top", "top"),
                              ("get_in", "y_onset", "onset")):
        for mname, maker in entries:
            variants = [(mname + "_crowd", cand, ML_BANK)]
            if cand_px is not None:
                variants.append((mname, cand_px, DESK_ML_BANK))
            for vname, vframe, vbank in variants:
                import time as _time
                _t0 = _time.time()
                # ANNOUNCE BEFORE, not only after (desk 2026-08-11: "it
                # stopped at [get_in] stuff" - it had not stopped, the
                # next model was fitting in silence for minutes). The
                # entry line prints immediately, the timing line
                # completes it, so a long fit looks like work rather
                # than a hang.
                _done += 1
                print(f"    [{head}] {vname}: fitting "
                      f"({_done}/{_total}, {len(vframe):,} days) ...",
                      end="", flush=True)
                wf = run_tournament_entry(vframe, episodes, vbank, label,
                                          mode, maker(label),
                                          EUPHORIA_FA_BUDGET_PER_IY,
                                          chooser=choose_threshold_f1)
                _summarise_entry(wf, sym_by, pxmap, mode)
                results[head][vname] = wf
                print(f" done in {_time.time() - _t0:.0f}s "
                      f"(elapsed {_time.time() - _t_all:.0f}s)",
                      flush=True)

    # the incumbents, unchanged, for the same table
    if series is not None and boom is not None:
        fpx = frame.merge(boom, on=["name", "date"], how="left")
        fpx["boom_state"] = fpx["boom_state"].eq(True)
        end_f, onset_f = desk_candidacy(fpx)
        for head, cand_f, fit, feats, label, mode in (
                ("get_out", end_f, desk_end_fit,
                 ["e1", "e2", "e3", "e5", "fade"], "y_top", "top"),
                ("get_in", onset_f, desk_onset_fit,
                 ["attention_accel", "hype_ratio", "bull_inflection",
                  "influx_speed", "attention_convexity"],
                 "y_onset", "onset")):
            wf = run_tournament_entry(cand_f, episodes, feats, label,
                                      mode, fit,
                                      EUPHORIA_FA_BUDGET_PER_IY)
            _summarise_entry(wf, sym_by, pxmap, mode)
            results[head]["rules"] = wf
    return results


def _summarise_entry(wf: dict, sym_by: dict, pxmap: dict,
                     mode: str) -> None:
    """Fold the bulky per-alert fields into the summary numbers the
    table needs: forward returns, precision, median lead."""
    if "alerts_by_name" in wf:
        wf["forward_returns"] = forward_returns(
            wf["alerts_by_name"], sym_by, pxmap)
        wf["n_alerts"] = int(sum(len(v) for v in
                                 wf["alerts_by_name"].values()))
        wf.pop("alerts_by_name")
    leads = wf.pop("leads", None)
    if leads is not None:
        key = "before_peak" if mode == "top" else "after_trough"
        vals = [ld[key] for ld in leads if key in ld]
        wf["median_lead_days"] = (int(np.median(vals)) if vals else None)
    cap, fa = wf.get("captured"), wf.get("false_alarms")
    if cap is not None and fa is not None and (cap + fa):
        wf["precision"] = round(cap / (cap + fa), 3)


def pick_winner(results: dict) -> str:
    """The pre-stated criterion (stated 2026-08-07 BEFORE the numbers
    were computed):

    * ONE model family serves both heads - two different learners for
      GET IN and GET OUT would double the explanation burden for a
      marginal gain, and the point of this exercise is fewer moving
      parts, not more.
    * The family with the highest COMBINED AP LIFT (test-year AP divided
      by its own frame's base rate, summed over the two heads) wins.
      LIFT, not raw AP: the incumbent's candidacy gates give it a frame
      where 40-60%% of candidate days are already labelled positive, so
      its raw AP is inflated by construction - "how many times better
      than guessing on your own frame" is the number that compares.
    * Ties by combined AUROC, then by fewer total false alarms.

    Only deployable entries compete: the incumbent and the crowd+price
    learners (the *_crowd variants are the clean-claim record, kept in
    the table but not adoptable - the desk configuration is allowed
    price and should use it)."""
    heads = ("get_out", "get_in")
    names = set()
    for h in heads:
        names |= {k for k, v in results.get(h, {}).items()
                  if "error" not in v and not k.endswith("_crowd")}

    def lift(r):
        ap, base = r.get("ap"), r.get("ap_baseline")
        return (ap / base) if ap and base else 0.0

    def key(name):
        rs = [results[h].get(name, {}) for h in heads]
        if any("error" in r or not r for r in rs):
            return (-1, 0, 0)
        return (sum(lift(r) for r in rs),
                sum(r.get("auroc") or 0 for r in rs),
                -sum(r.get("false_alarms") or 10**9 for r in rs))
    if not names:
        return "rules"
    return max(names, key=key)


# ---------------------------------------------------------------------------
# ground-truth sweep (task: "consider lowering the 25%/50% boom bars")
# ---------------------------------------------------------------------------
# explicit values, never {}: the config defaults MOVED on 2026-08-07 (to
# the 20/40, 12/25 row this sweep selected), and a grid entry that means
# "whatever the defaults are today" would silently re-label itself
GT_GRID = {
    "GT-old (25/50, 15/30)": {"boom_theme": 0.25, "boom_single": 0.50,
                              "crash_theme": 0.15, "crash_single": 0.30},
    "GT-ADOPTED (20/40, 12/25)": {"boom_theme": 0.20, "boom_single": 0.40,
                                  "crash_theme": 0.12,
                                  "crash_single": 0.25},
    "GT-loosest (15/30, 10/20)": {"boom_theme": 0.15, "boom_single": 0.30,
                                  "crash_theme": 0.10,
                                  "crash_single": 0.20},
}


def ground_truth_sweep(series, pxmap, counts, sents, sym_by,
                       model_maker=None) -> dict:
    """Re-run episode extraction AND the best learner under each ground
    truth in GT_GRID. Reported per variant: episode count, which themes
    ever have an episode (the desk asks for gold / meme / semis by
    name), and the learner's AP/AUROC/capture - so 'did loosening hurt'
    is answered by the same instrument that will be presented."""
    from src.config import EUPHORIA_FA_BUDGET_PER_IY
    if model_maker is None:
        model_maker = make_logit_fit
    out = {}
    for gt_name, gt in GT_GRID.items():
        eps = episode_catalog(series, pxmap, gt_override=gt or None)
        frame = build_day_frame(series, pxmap, eps, counts, sents)
        cand = attach_price_features(candidate_frame(frame), series, pxmap)
        row = {"episodes": int(len(eps)),
               "themes_with_episode": sorted(
                   eps[eps.kind == "theme"]["name"].unique().tolist()),
               "singles_with_episode": int(
                   eps[eps.kind == "single"]["name"].nunique())}
        for head, label, mode in (("get_out", "y_top", "top"),
                                  ("get_in", "y_onset", "onset")):
            wf = run_tournament_entry(cand, eps, DESK_ML_BANK, label,
                                      mode, model_maker(label),
                                      EUPHORIA_FA_BUDGET_PER_IY,
                                      chooser=choose_threshold_f1)
            wf.pop("leads", None)
            wf.pop("alerts_by_name", None)
            row[head] = {k: wf.get(k) for k in
                         ("auroc", "ap", "ap_baseline", "capture_rate",
                          "captured", "detectable", "false_alarms",
                          "fa_per_iy", "late")}
        out[gt_name] = row
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(sweep: bool = False) -> int:
    prices, series, pxmap, counts, sents = _load_all()
    sym_by = {es.name: es.symbol for es in series}
    episodes = episode_catalog(series, pxmap)
    frame = build_day_frame(series, pxmap, episodes, counts, sents)
    boom = boom_state_frame(series, pxmap)

    print(f"universe: {len(series)} instruments | episodes: "
          f"{len(episodes)} | candidate days: {len(frame):,}")

    results = run_ml_tournament(frame, episodes, pxmap, sym_by,
                                series=series, boom=boom)
    for head in ("get_out", "get_in"):
        print(f"\n{head.upper()} tournament:")
        for m, r in results[head].items():
            if "error" in r:
                print(f"  {m:<12} {r['error']}")
                continue
            lift = (round(r["ap"] / r["ap_baseline"], 2)
                    if r.get("ap") and r.get("ap_baseline") else None)
            print(f"  {m:<12} AP {r['ap']} (base {r['ap_baseline']}, "
                  f"lift {lift}x) | AUROC {r['auroc']} | capture "
                  f"{r['captured']}/{r['detectable']} | FA "
                  f"{r['false_alarms']} ({r['fa_per_iy']}/iy)")
    winner = pick_winner(results)
    results["winner"] = winner
    print(f"\nwinner (one family, both heads, combined AP lift): {winner}")

    payload = {"bank": ML_BANK, "desk_bank": DESK_ML_BANK,
               "results": results}
    if sweep:
        print("\nGROUND-TRUTH SWEEP (winning learner under each "
              "episode definition):")
        sw = ground_truth_sweep(series, pxmap, counts, sents, sym_by)
        for k, v in sw.items():
            print(f"  {k}: {v['episodes']} episodes | GET OUT AP "
                  f"{v['get_out']['ap']} cap {v['get_out']['captured']}/"
                  f"{v['get_out']['detectable']} | GET IN AP "
                  f"{v['get_in']['ap']}")
        payload["ground_truth_sweep"] = sw

    out_path = os.path.join("docs", "research", "ml_tournament.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0,
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true",
                   help="also run the ground-truth sweep")
    a = p.parse_args()
    raise SystemExit(main(sweep=a.sweep))


# ---------------------------------------------------------------------------
# explainability sidecar: which measurements the live model leans on.
# Written by the pipeline (euphoria_phases.rebuild_phase_files) whenever a
# learned model ships, read by the dashboard's "what drives the calls"
# expander and by notebook 03 §SS2 - one computation, every surface.
# ---------------------------------------------------------------------------
def model_insight(cand: pd.DataFrame, eval_year: int | None = None,
                  max_eval_rows: int = 4000) -> dict:
    """Two independent reads on what the fitted ensemble members weigh,
    per head:

    * the LOGIT WEIGHTS - one signed coefficient per measurement,
      readable as a formula (positive = more of this, more risk);
    * the GBM PERMUTATION IMPORTANCE - shuffle one measurement on
      held-out days and record how much Average Precision drops (Breiman
      2001 / the standard sklearn procedure): a model-agnostic "how much
      does the model actually USE this" that trees cannot fake.

    Fit on years < eval_year (default: the newest year in the frame),
    importance evaluated on eval_year - so the read matches the live
    walk-forward convention. Deterministic (random_state=0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    if eval_year is None:
        eval_year = int(cand["year"].max())
    tr = cand[cand["year"] < eval_year]
    ev = cand[cand["year"] == eval_year]
    if tr.empty or ev.empty or tr["y_top"].nunique() < 2:
        tr = ev = cand
    if len(ev) > max_eval_rows:
        ev = ev.sample(max_eval_rows, random_state=RANDOM_STATE)

    out = {"bank": DESK_ML_BANK,
           "plain_labels": {f: ML_BANK_LABELS.get(f, f)
                            for f in DESK_ML_BANK},
           "fitted_on_years_before": eval_year,
           "importance_evaluated_on": eval_year}
    for label, head in (("y_top", "get_out"), ("y_onset", "get_in")):
        lg = LogisticRegression(class_weight="balanced", max_iter=2000)
        lg.fit(tr[DESK_ML_BANK], tr[label])
        gb = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=200,
            monotonic_cst=[1] * len(DESK_ML_BANK),
            random_state=RANDOM_STATE)
        gb.fit(tr[DESK_ML_BANK], tr[label],
               sample_weight=_balanced_weights(tr[label].values))
        if ev[label].nunique() > 1:
            pi = permutation_importance(
                gb, ev[DESK_ML_BANK], ev[label], n_repeats=3,
                random_state=RANDOM_STATE, scoring="average_precision")
            imp = {f: round(float(v), 4)
                   for f, v in zip(DESK_ML_BANK, pi.importances_mean)}
        else:
            imp = {f: None for f in DESK_ML_BANK}
        out[head] = {
            "logit_weights": {f: round(float(c), 3) for f, c in
                              zip(DESK_ML_BANK, lg.coef_[0])},
            "gbm_permutation_importance": imp,
        }
    return out
