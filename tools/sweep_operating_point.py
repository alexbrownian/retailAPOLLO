"""
sweep_operating_point.py - choose the "Standard" cut on evidence
================================================================
Desk request 2026-08-23: the shipped setting declines gold's 2026 top by
about a hundredth of probability, and the question is whether the
operating point is too tight IN GENERAL - not whether gold specifically
can be made to fire.

    python tools/sweep_operating_point.py
    python tools/sweep_operating_point.py --betas 0.5,0.75,1.0,1.25
    python tools/sweep_operating_point.py --watch gold_metals,uranium_nuclear

WHAT IT DOES
    Re-runs the REAL walk-forward judge (analytics.ml_detector) over a
    grid of beta for the strict/"Standard" operating point, and reports
    capture, precision, false alarms and FA-per-instrument-year for each
    - both heads. It writes nothing: no thresholds move, no store is
    touched. It is a measurement, not a change.

HOW TO USE THE OUTPUT HONESTLY
    Write down the criterion BEFORE reading the table. The project's
    pre-stated one is the false-alarm budget in src/config.py
    (EUPHORIA_FA_BUDGET_PER_IY): the loosest beta whose FA/instrument-yr
    still respects it. If you instead pick the beta that happens to make
    a name you already looked at fire, you have fitted a parameter to a
    known outcome and the walk-forward claim is void - which is the one
    thing this file exists to prevent.

    --watch prints, for information only, whether the named instruments
    would fire at each beta. It is there so you can SEE the consequence
    of a choice, not so you can choose by it.

THEN
    set EUPHORIA_STRICT_BETA in src/config.py, re-run
        python -m analytics.run_analytics --what phases --research
    and record the before/after in docs/PARAMETER_REGISTER.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep the Standard operating point.")
    p.add_argument("--betas", default="0.5,0.625,0.75,0.875,1.0",
                   help="comma-separated beta grid (default 0.5..1.0)")
    p.add_argument("--watch", default="gold_metals",
                   help="comma-separated instruments to report fire/no-fire "
                        "for, FOR INFORMATION - never as the choice criterion")
    args = p.parse_args()
    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    watch = [w.strip() for w in args.watch.split(",") if w.strip()]

    from src.config import EUPHORIA_FA_BUDGET_PER_IY, EUPHORIA_STRICT_BETA
    from analytics import euphoria_phases as ep
    from analytics import ml_detector as mld

    print("loading the judged frame (this is the slow part) ...", flush=True)
    prices, series, pxmap, counts, sents = mld._load_all()
    episodes = ep.episode_catalog(series, pxmap)
    frame = ep.build_day_frame(series, pxmap, episodes, counts, sents)
    cand = mld.attach_price_features(mld.candidate_frame(frame), series, pxmap)
    years = sorted(cand["year"].unique())
    print(f"  {len(cand):,} candidate days | {cand['name'].nunique()} "
          f"instruments | {len(episodes):,} episodes | "
          f"years {years[0]}-{years[-1]}")
    print(f"  current EUPHORIA_STRICT_BETA = {EUPHORIA_STRICT_BETA}")
    print(f"  pre-stated FA budget         = {EUPHORIA_FA_BUDGET_PER_IY}"
          f" per instrument-year\n")

    # Reuse the tournament's own walk-forward entry point rather than
    # re-implementing the fold loop: same folds, same judge, same alert
    # shaping as the shipped numbers. The ONLY thing that varies here is
    # the chooser, which is the whole point of the sweep.
    rows = []
    for head, label, mode in (("get_out", "y_top", "top"),
                              ("get_in", "y_onset", "onset")):
        fit = mld.make_ens_fit(label)
        for beta in betas:
            def chooser(ts, eps, m, fa, n, _b=beta):
                return mld._choose_threshold_fbeta(ts, eps, m, beta=_b)
            wf = ep.run_tournament_entry(cand, episodes, mld.DESK_ML_BANK,
                                         label, mode, fit,
                                         EUPHORIA_FA_BUDGET_PER_IY,
                                         chooser=chooser)
            if "error" in wf:
                print(f"  {head} beta {beta}: {wf['error']}")
                continue
            alerts = wf["captured"] + wf["false_alarms"]
            prec = wf["captured"] / alerts if alerts else 0.0
            fired = {w: (wf.get("alerts_by_name", {}).get(w) or ["-"])[0]
                     for w in watch}
            rows.append(dict(head=head, beta=beta,
                             captured=wf["captured"],
                             detectable=wf["detectable"],
                             capture_rate=wf["capture_rate"],
                             precision=round(prec, 3),
                             false_alarms=wf["false_alarms"],
                             fa_per_iy=wf["fa_per_iy"],
                             auroc=wf.get("auroc"), ap=wf.get("ap"),
                             **{f"first_{w}": fired[w] for w in watch}))
            over = (" <- OVER the pre-stated budget"
                    if wf["fa_per_iy"] > EUPHORIA_FA_BUDGET_PER_IY else "")
            print(f"  {head:<8} beta {beta:<6} capture "
                  f"{wf['capture_rate']:>6}  precision {prec:6.1%}  "
                  f"FA {wf['false_alarms']:>4}  FA/instr-yr "
                  f"{wf['fa_per_iy']:<6}{over}", flush=True)

    out = pd.DataFrame(rows)
    dest = os.path.join(PROJECT_ROOT, "docs", "research",
                        "operating_point_sweep.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_json(dest, orient="records", indent=1)
    print(f"\nwrote {os.path.relpath(dest, PROJECT_ROOT)}")
    print("\nNOTHING WAS CHANGED. To adopt a beta: set EUPHORIA_STRICT_BETA "
          "in src/config.py,\nre-run `--what phases --research`, and record "
          "the before/after in PARAMETER_REGISTER.md.")
    if watch and len(out):
        cols = ["head", "beta"] + [f"first_{w}" for w in watch]
        print(f"\nFOR INFORMATION ONLY - first alert per watched name.")
        print("Do NOT choose beta from this column; choose it from the "
              "FA budget above.")
        print(out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
