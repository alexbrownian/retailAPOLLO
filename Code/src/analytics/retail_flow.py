"""
retail_flow.py
==============
The continuous retail-flow dial (research record:
Data/research_record/nb08_retail_flow.json). One smooth number per name
per day in [-1, +1]:

    retail_flow      the walk-forward gauge itself (state-space filtered)
    retail_flow_disp the display scaling: trailing 365d per-name rank
                     mapped to [-1, +1] - "bullish for THIS name" means
                     bullish against its own recent range

Construction, exactly as the record froze it:

  * ANCHORED HALF - posts-only P(onset) and P(top) (logit + monotone
    GBM, each calibrated through its TRAIN-year score ECDF),
    subtracted. The anchors are the production get_in / get_out
    definitions made continuous; nothing about them changes here.
  * SLOW HALF - ridge + GBM regression of 63-day forward EXCESS return
    on the slowest crowd measures (attention age/saturation, breadth,
    cross-name rotation), tanh-squashed. Price appears in the TARGET at
    training time only; every feature is price-free. This half is where
    the measured 1-3 week lead over price lives.
  * FILTER - steady-state Kalman (a local-level state-space model),
    gain chosen per fold from {0.05, 0.1, 0.2} by train-year IC against
    42-day forward excess. No lookahead, no lockout rule: smoothness is
    model structure, not a bolt-on average.

Walk-forward discipline throughout: every year is scored by models
fitted strictly on earlier years, including the current partial year.
The whole computation is deterministic (random_state=0), so two runs on
the same stores produce identical columns.

WHAT THIS IS NOT. Not a trigger: nothing fires from these columns, and
the fence tests hold the line. The dial is the trend/context layer -
IC ~+0.01-0.02 pooled (a tide, not a ticket) - while get_in / get_out
remain the calls. And it deliberately does NOT hard-flip inside boom
regimes: the record's sweep shows the flip helps tops only by hurting
starts more, with the winner unstable across eras.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

# SILENCE ONE KNOWN-COSMETIC WARNING, narrowly. Some sklearn/joblib
# version pairs emit "sklearn.utils.parallel.delayed should be used
# with sklearn.utils.parallel.Parallel" from INSIDE sklearn's own fit
# paths - hundreds of times per walk-forward, drowning the progress
# lines under a mismatched joblib pair. It is sklearn's internal usage, not
# ours, and harmless; upgrading joblib+scikit-learn together clears it
# at the source. The filter is message-targeted so every other warning
# still surfaces.
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message=r".*sklearn\.utils\.parallel\.delayed.*")

RANDOM_STATE = 0
KALMAN_GAINS = (0.05, 0.1, 0.2)
SLOW_FEATURES = ["e1", "bull_level", "bull_persist", "att_age", "att_sat",
                 "att_fade60", "bull_age", "xname_rank", "breadth_level",
                 "e2", "att_vol_21"]
# the slow-half additions on top of the 13-feature price-blind bank
EXTRA_FEATURES = ["att_age", "att_sat", "att_fade60", "mood_slope10",
                  "bull_age", "xname_rank", "xname_rank_chg21",
                  "breadth_level"]


def _run_age(s: pd.Series) -> pd.Series:
    out, run = np.zeros(len(s)), 0.0
    for i, v in enumerate(s.to_numpy()):
        run = run + 1 if v > 0 else 0.0
        out[i] = run
    return pd.Series(out, index=s.index)


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """The slow-half price-free additions: HOW LONG the crowd has been hot
    (age, saturation), whether it is off its own peak, and where the
    name sits across the whole universe (rotation). Every threshold is
    a trailing per-name quantile shifted one day - day t is graded only
    against days strictly before it."""
    d = df.sort_values(["name", "date"]).reset_index(drop=True).copy()
    g = d.groupby("name", sort=False)
    q60h = g["hype_raw"].transform(
        lambda s: s.rolling(730, min_periods=120).quantile(0.60).shift(1))
    d["att_age"] = np.log1p((d["hype_raw"] > q60h).astype(float)
                            .groupby(d["name"], sort=False)
                            .transform(_run_age))
    d["att_sat"] = np.log1p((d["hype_raw"] - q60h).clip(lower=0)
                            .groupby(d["name"], sort=False)
                            .transform(lambda s: s.rolling(
                                90, min_periods=10).sum()))
    mx60 = g["hype_raw"].transform(
        lambda s: s.rolling(60, min_periods=10).max())
    d["att_fade60"] = (d["hype_raw"] / mx60.replace(0, np.nan)).fillna(1.0)
    d["mood_slope10"] = g["bull_level"].transform(
        lambda s: s.diff(10) / 10.0)
    q60b = g["bull_level"].transform(
        lambda s: s.rolling(730, min_periods=120).quantile(0.60).shift(1))
    d["bull_age"] = np.log1p((d["bull_level"] > q60b).astype(float)
                             .groupby(d["name"], sort=False)
                             .transform(_run_age))
    d["xname_rank"] = d.groupby("date")["e1"].rank(pct=True)
    d["xname_rank_chg21"] = g["xname_rank"].transform(lambda s: s.diff(21))
    d["breadth_level"] = (d["source_breadth"].fillna(0.0)
                          if "source_breadth" in d else 0.0)
    if "fade" in d:
        d["fade"] = d["fade"].astype(float)
    for c in EXTRA_FEATURES:
        d[c] = d[c].fillna(0.0)
    return d


def _forward_excess(df: pd.DataFrame, prices: pd.DataFrame,
                    sym_by: dict, horizons=(42, 63)) -> pd.DataFrame:
    """Excess-of-market forward returns on each symbol's OWN trading
    calendar (never a union index), calendar days mapped to the
    next trading day. Used as TRAIN targets only."""
    fw = {}
    for sym, gp in prices.groupby("symbol"):
        px = gp.set_index("date")["px_last"].sort_index().dropna()
        fw[sym] = {h: (px.shift(-h) / px - 1.0) for h in horizons}
    d = df.assign(symbol=df["name"].map(sym_by))
    for h in horizons:
        F = pd.DataFrame({s: v[h] for s, v in fw.items()})
        EX = F.sub(F.median(axis=1), axis=0)
        outs = []
        for sym, gp in d.sort_values("date").groupby("symbol"):
            if sym not in EX.columns:
                gp = gp.copy()
                gp[f"ex{h}"] = np.nan
                outs.append(gp)
                continue
            e = EX[sym].dropna().rename(f"ex{h}").reset_index()
            e.columns = ["date", f"ex{h}"]
            outs.append(pd.merge_asof(gp, e.sort_values("date"), on="date",
                                      direction="forward",
                                      tolerance=pd.Timedelta(days=5)))
        d = (pd.concat(outs).sort_values(["name", "date"])
             .reset_index(drop=True))
    return d.drop(columns=["symbol"])


def _bal_w(y):
    pos, neg = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    return np.where(y > 0, len(y) / (2 * pos), len(y) / (2 * neg))


def _ecdf(train_scores, x):
    s = np.sort(train_scores)
    return np.searchsorted(s, x, side="right") / max(len(s), 1)


def _heads_net(tr, te, bank):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    net_te, net_tr = np.zeros(len(te)), np.zeros(len(tr))
    for sign, lb in ((1, "y_onset"), (-1, "y_top")):
        y = tr[lb].values
        lg = LogisticRegression(class_weight="balanced",
                                max_iter=2000).fit(tr[bank], y)
        gb = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.1, max_iter=200,
            random_state=RANDOM_STATE).fit(tr[bank], y,
                                           sample_weight=_bal_w(y))
        for m in (lg, gb):
            ptr = m.predict_proba(tr[bank])[:, 1]
            net_te += sign * _ecdf(ptr, m.predict_proba(te[bank])[:, 1]) / 2
            net_tr += sign * _ecdf(ptr, ptr) / 2
    return net_te, net_tr


def _slow_tilt(tr, te, feats):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr63 = tr.dropna(subset=["ex63"])
    z = ((tr63["ex63"] - tr63["ex63"].mean())
         / max(tr63["ex63"].std(), 1e-9)).clip(-3, 3)
    mu, sd = tr63[feats].mean(), tr63[feats].std().replace(0, 1)
    rd = Ridge(alpha=30.0).fit((tr63[feats] - mu) / sd, z)
    gb = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.06, max_iter=250,
        random_state=RANDOM_STATE).fit(tr63[feats], z)
    p_te = np.tanh((rd.predict((te[feats] - mu) / sd)
                    + gb.predict(te[feats])) / 2)
    p_tr = np.tanh((rd.predict((tr63[feats] - mu) / sd)
                    + gb.predict(tr63[feats])) / 2)
    return p_te, p_tr, tr63.index


def _kalman(names, vals, gain):
    out, prev, last = np.empty(len(vals)), 0.0, None
    for i, (nm, o) in enumerate(zip(names, vals)):
        if nm != last:
            prev, last = 0.0, nm
        prev = prev + gain * (o - prev)
        out[i] = prev
    return out


def _ic(df, col, target):
    from scipy.stats import spearmanr
    d = df.dropna(subset=[col, target])
    return float(spearmanr(d[col], d[target])[0]) if len(d) > 100 else np.nan


def build_retail_flow(frame: pd.DataFrame, prices: pd.DataFrame,
                      sym_by: dict, verbose: bool = True) -> pd.DataFrame:
    """name/date/retail_flow/retail_flow_disp for every scoreable
    name-day. `frame` is the coverage-gated day frame with the
    13-feature price-blind bank already attached (the same object the
    experimental trigger scores from)."""
    import src.analytics.ml_detector as mld
    from src.analytics.euphoria_phases import (inflection_features,
                                           INFLECTION_EXTRA_FEATURES)
    t0 = time.time()
    RF = add_flow_features(inflection_features(mld.candidate_frame(frame)))
    bank = (list(mld.ML_BANK) + list(INFLECTION_EXTRA_FEATURES)
            + EXTRA_FEATURES)
    RF = _forward_excess(RF, prices, sym_by)
    RF["year"] = pd.to_datetime(RF["date"]).dt.year
    oos = []
    for Y in sorted(RF["year"].unique()):
        tr = RF[RF["year"] < Y].sort_values(["name", "date"]) \
            .reset_index(drop=True)
        te = RF[RF["year"] == Y].sort_values(["name", "date"]) \
            .reset_index(drop=True)
        if (len(tr) < 3000 or not len(te) or tr["y_onset"].sum() < 12
                or tr["y_top"].sum() < 12):
            continue
        nh_te, nh_tr = _heads_net(tr, te, bank)
        st_te, st_tr, st_ix = _slow_tilt(tr, te, SLOW_FEATURES)
        te["_raw"] = 0.5 * nh_te + 0.5 * st_te
        tr["_raw"] = 0.5 * nh_tr
        tr.loc[st_ix, "_raw"] += 0.5 * st_tr
        gain, best = KALMAN_GAINS[0], -9.0
        for K in KALMAN_GAINS:
            tr["_s"] = _kalman(tr["name"].to_numpy(),
                               tr["_raw"].to_numpy(), K)
            v = _ic(tr, "_s", "ex42")
            if v == v and v > best:
                best, gain = v, K
        te["retail_flow"] = _kalman(te["name"].to_numpy(),
                                    te["_raw"].to_numpy(), gain)
        oos.append(te[["name", "date", "retail_flow"]])
        if verbose:
            print(f"    retail flow {Y}: gain {gain} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not oos:
        return pd.DataFrame(columns=["name", "date", "retail_flow",
                                     "retail_flow_disp"])
    out = (pd.concat(oos).sort_values(["name", "date"])
           .reset_index(drop=True))
    out["retail_flow_disp"] = out.groupby("name", sort=False)[
        "retail_flow"].transform(
        lambda s: 2 * s.rolling(365, min_periods=60).rank(pct=True) - 1)
    if verbose:
        print(f"    retail flow: {len(out):,} name-days in "
              f"{time.time() - t0:.0f}s", flush=True)
    return out


def attach_retail_flow(ds: pd.DataFrame, frame: pd.DataFrame,
                       prices: pd.DataFrame, sym_by: dict,
                       verbose: bool = True) -> pd.DataFrame:
    """Merge the dial columns onto the signal store. Failure-isolated by
    the caller: the signal store must never be lost to a dial bug."""
    flow = build_retail_flow(frame, prices, sym_by, verbose=verbose)
    ds = ds.drop(columns=[c for c in ("retail_flow", "retail_flow_disp")
                          if c in ds.columns])
    return ds.merge(flow, on=["name", "date"], how="left")
