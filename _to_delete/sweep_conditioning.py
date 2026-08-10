"""
sweep_conditioning.py — the operating-point conditioning study
(desk request 2026-08-09: "flipping between GET IN and GET OUT too
quickly ... GET OUT signals all the way up a price graph ... should be
slightly stricter").

Three candidate conditionings of the FROZEN winner's score, evaluated
walk-forward under the identical discipline as the tournament — the
model is NOT retrained, only how its daily score becomes an alert:

  smoothing  raw score  vs  trailing 7d mean per name (reuses the ROLL
             constant already in the system; trailing-only, no lookahead)
  cut        F1 (balanced, the shipped standard)  vs  F0.5 (precision
             weighted twice — the textbook strict operating point)
  trigger    level (score >= cut fires, 21d cooldown — shipped)  vs
             crossing (fires only on an UPWARD crossing of the cut;
             the score must first drop below the cut to re-arm - kills
             the repeated fire all the way up a rally)

Pre-stated adoption rule (written before the numbers): adopt, for the
STANDARD display mode, the conditioning with the fewest IN/OUT flips
per instrument-year among those that (a) cut false alarms vs the
shipped config and (b) keep capture within 5 points of it, per head;
one conditioning family for both heads (fewest total flips). The F0.5
cut of the SAME conditioning becomes the STRICT display mode.

Writes docs/research/conditioning_sweep.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import ml_detector as mld                     # noqa: E402
from analytics.euphoria import build_all_series              # noqa: E402
from analytics.euphoria_phases import (                      # noqa: E402
    build_day_frame, episode_catalog, classify_top_alerts,
    classify_onset_alerts, _eps_arrays, _day_ints, _alerts_int)
from analytics.loaders import (load, THEME_COUNTS, THEME_SENT,  # noqa: E402
                               TICKER_COUNTS, TICKER_SENT)
from src.config import EUPHORIA_COOLDOWN_DAYS                # noqa: E402

SMOOTH_D = 7          # the existing ROLL constant, reused - not new


def smooth_scores(df: pd.DataFrame, col: str = "score") -> pd.Series:
    """Trailing 7-calendar-day mean per name (no lookahead)."""
    out = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("name"):
        s = pd.Series(g[col].values,
                      index=pd.DatetimeIndex(g["date"]))
        out.loc[g.index] = s.rolling(f"{SMOOTH_D}D",
                                     min_periods=1).mean().values
    return out


def alerts_crossing(days: np.ndarray, scores: np.ndarray, thr: float,
                    cooldown: int = EUPHORIA_COOLDOWN_DAYS) -> np.ndarray:
    """Fire on UPWARD crossings only: after a fire, the score must fall
    below the cut before it can fire again (plus the usual cooldown)."""
    alerts, last, armed = [], None, True
    for d, s in zip(days, scores):
        if s >= thr:
            if armed and (last is None or d - last >= cooldown):
                alerts.append(d)
                last = d
            armed = False
        else:
            armed = True
    return np.asarray(alerts, dtype=np.int64)


def main() -> int:
    t0 = time.time()
    prices = pd.read_parquet("data/prices/prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    series, pxmap = build_all_series(prices)
    episodes = episode_catalog(series, pxmap)
    counts = {"theme": load(THEME_COUNTS), "ticker": load(TICKER_COUNTS)}
    sents = {"theme": load(THEME_SENT), "ticker": load(TICKER_SENT)}
    for d in list(counts.values()) + list(sents.values()):
        d["date"] = pd.to_datetime(d["date"])
    frame = build_day_frame(series, pxmap, episodes, counts, sents)
    cand = mld.attach_price_features(mld.candidate_frame(frame), series,
                                     pxmap)
    print(f"[{time.time()-t0:.0f}s] frame ready: {len(cand):,} days")

    # walk-forward TEST scores (notebook 07's cache - re-run notebook
    # 07 first if this is missing or stale)
    cache = "docs/research/nb07_wf_scores.parquet"
    wf = pd.read_parquet(cache)
    fp = json.loads(wf["fingerprint"].iloc[0])
    assert fp["rows"] == len(cand), "stale WF cache - re-run notebook 07"
    wf = wf.drop(columns=["fingerprint"])

    eps_by = dict(tuple(episodes.groupby("name")))
    empty = episodes.iloc[0:0]
    n_inst = cand["name"].nunique()

    results = {}
    for head, label, mode in (("GET OUT", "y_top", "top"),
                              ("GET IN", "y_onset", "onset")):
        sc = wf[wf["head"] == head].copy().sort_values(["name", "date"])
        sc["score_smooth"] = smooth_scores(sc)
        test_years = sorted(sc["test_year"].unique())

        # per-year cuts for each (smoothing, beta): chosen on the TRAIN
        # years scored by the same fitted family (production convention)
        cuts = {}
        for y in test_years:
            tr = cand[cand["year"] < y]
            if tr.empty or tr[label].nunique() < 2:
                continue
            fit = mld.make_ens_fit(label)
            trs = tr.assign(score=fit(tr, tr, mld.DESK_ML_BANK))
            trs = trs.sort_values(["name", "date"]).reset_index(drop=True)
            trs["score_smooth"] = smooth_scores(trs)
            for sm in (False, True):
                col = "score_smooth" if sm else "score"
                s2 = trs.assign(score=trs[col])
                for beta in (1.0, 0.5):
                    cuts[(y, sm, beta)] = mld._choose_threshold_fbeta(
                        s2, episodes, mode, beta)
            print(f"[{time.time()-t0:.0f}s] {head} cuts for {y} done",
                  flush=True)

        judge = (classify_top_alerts if mode == "top"
                 else classify_onset_alerts)
        det_col = ("top_detectable" if mode == "top"
                   else "onset_detectable")
        det = episodes[episodes.year.isin(test_years)
                       & episodes[det_col]]

        for sm in (False, True):
            col = "score_smooth" if sm else "score"
            for beta in (1.0, 0.5):
                for trig in ("level", "crossing"):
                    per_name = {}
                    for (n, y), g in sc.groupby(["name", "test_year"]):
                        thr = cuts.get((y, sm, beta))
                        if thr is None:
                            continue
                        g = g.sort_values("date")
                        fn = (alerts_crossing if trig == "crossing"
                              else _alerts_int)
                        al = fn(_day_ints(g["date"]),
                                g[col].to_numpy(float), thr)
                        per_name.setdefault(n, []).extend(al.tolist())
                    captured, fa = set(), 0
                    for n, al in per_name.items():
                        res = judge(np.asarray(sorted(al), dtype=np.int64),
                                    _eps_arrays(eps_by.get(n, empty)))
                        captured |= {(n, p) for p in res["captured"]}
                        fa += len(res["fa"])
                    cap_rate = len(captured) / max(len(det), 1)
                    fa_iy = fa / max(n_inst * len(test_years), 1)
                    npos = len(captured) + fa
                    prec = len(captured) / npos if npos else 0.0
                    f1 = (2 * prec * cap_rate / (prec + cap_rate)
                          if prec + cap_rate else 0.0)
                    key = f"{'smooth' if sm else 'raw'}|F{beta}|{trig}"
                    results.setdefault(head, {})[key] = {
                        "captured": len(captured),
                        "detectable": len(det),
                        "capture_rate": round(cap_rate, 3),
                        "false_alarms": fa,
                        "fa_per_iy": round(fa_iy, 3),
                        "precision": round(prec, 3),
                        "episode_f1": round(f1, 3),
                        "n_alerts": int(sum(len(v)
                                            for v in per_name.values())),
                        "alerts_by_name": {n: [str(np.datetime64(int(d), "D"))
                                               for d in sorted(al)]
                                           for n, al in per_name.items()},
                    }
                    r = results[head][key]
                    print(f"[{time.time()-t0:.0f}s] {head:8s} {key:22s} "
                          f"cap {r['captured']}/{r['detectable']} "
                          f"FA {r['false_alarms']} ({r['fa_per_iy']}/iy) "
                          f"prec {r['precision']} F1 {r['episode_f1']}",
                          flush=True)

    # flips: a GET IN and a GET OUT on the same name within 30d of each
    # other (either order) - the exact complaint being fixed
    flips = {}
    for key in next(iter(results.values())):
        a_in = results.get("GET IN", {}).get(key, {}).get(
            "alerts_by_name", {})
        a_out = results.get("GET OUT", {}).get(key, {}).get(
            "alerts_by_name", {})
        n_flip = 0
        for n in set(a_in) | set(a_out):
            di = [np.datetime64(d) for d in a_in.get(n, [])]
            do = [np.datetime64(d) for d in a_out.get(n, [])]
            for x in di:
                if any(abs((x - y).astype(int)) <= 30 for y in do):
                    n_flip += 1
        flips[key] = n_flip
    for key, nf in flips.items():
        for head in results:
            results[head][key]["flips_within_30d"] = nf

    # strip the bulky alert lists from the persisted record (keep counts)
    slim = {h: {k: {kk: vv for kk, vv in r.items()
                    if kk != "alerts_by_name"}
                for k, r in results[h].items()} for h in results}
    out = {"as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
           "smooth_days": SMOOTH_D,
           "adoption_rule": ("fewest flips among configs cutting FAs "
                             "with capture within 5pts of shipped, "
                             "per head; one family both heads"),
           "results": slim}
    with open("docs/research/conditioning_sweep.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{time.time()-t0:.0f}s] saved "
          "docs/research/conditioning_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
