"""
sweep_boom_min.py — how much does the BOOM_MIN choice actually matter?
======================================================================

THE DESK'S QUESTION (2026-08-04): "can we try and do a more scientific
method for the boom_min? e.g. we can try and vary the % to see if we can
improve the model performance."

THE METHODOLOGICAL TRAP, STATED FIRST
-------------------------------------
`EUPHORIA_BOOM_MIN_ETF` (0.25) and `EUPHORIA_BOOM_MIN_SINGLE` (0.50) are
used in TWO places that look alike and are not:

  1. GROUND TRUTH (`analytics/euphoria.py::ground_truth_peaks`, G2, 120d
     window): how big a run-up must precede a price high before that
     high COUNTS as a top. This is the exam.
  2. THE LIVE GATE (`analytics/euphoria_phases.py::boom_state_frame`,
     54d window): whether a name is allowed to be a GET OUT candidate
     today. This is the answer.

Tuning (1) to "improve model performance" is circular: you would be
choosing the definition of a top so that the detector looks good at
finding tops, and the improvement would be an artefact by construction.
Tuning (2) is ordinary, legitimate model selection.

The two WINDOWS were already separated for this exact reason (54 vs 120,
desk decision 2026-07-29). The two MAGNITUDES were not — they still
share a constant — which is why this study runs them as two different
kinds of experiment:

  STUDY A - THE GATE (selection).   Ground truth held FIXED at the frozen
      0.25/0.50; only the gate multiplier moves. Any improvement here is
      real, and is judged by the project's existing pre-stated rule:
      captures - EUPHORIA_FA_PENALTY x false alarms, out of sample.
  STUDY B - THE EXAM (sensitivity, NOT selection).  Gate held FIXED;
      the ground-truth multiplier moves. This CANNOT be used to pick a
      value. It answers a different and more important question: is the
      detector's scorecard an artefact of one arbitrary 25%/50% choice,
      or does the conclusion survive a wide range of definitions? A
      result that only exists at one definition of "a top" is not a
      result.

Both studies move the two magnitudes together through a single
MULTIPLIER, so the ETF/single ratio the desk set (0.25 vs 0.50, singles
boom harder) is preserved and the sweep stays one-dimensional and
readable.

USAGE
    python tools/sweep_boom_min.py                 # both studies
    python tools/sweep_boom_min.py --study gate    # just A
    python tools/sweep_boom_min.py --mults 0.6,1.0,1.4

Each point re-runs the full walk-forward research pass (~70s), so a
default run is about 12 minutes. THE STORES ARE RESTORED AFTERWARDS -
the sweep is a measurement, not a re-fit, and it must never leave the
frozen record pointing at one of its own experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:                 # runnable as `python tools/...`
    sys.path.insert(0, ROOT)

# the frozen values this study perturbs (src/config.py)
BASE_ETF, BASE_SINGLE = 0.25, 0.50
DEFAULT_MULTS = [0.6, 0.8, 1.0, 1.2, 1.4]
# files a research pass rewrites - saved and put back
TOUCHED = ["euphoria_desk_report.json", "euphoria_desk.parquet",
           "euphoria_onset_report.json", "euphoria_onset.parquet",
           "episodes.parquet", "phase_day_frame.parquet"]


def _set(gate_mult: float, truth_mult: float) -> None:
    """Point the two call sites at different magnitudes. This is the
    separation the constants do not yet have; the study exists to show
    whether it is worth making permanent."""
    import analytics.euphoria as E
    import analytics.euphoria_phases as P
    E.EUPHORIA_BOOM_MIN_ETF = BASE_ETF * truth_mult
    E.EUPHORIA_BOOM_MIN_SINGLE = BASE_SINGLE * truth_mult
    P.EUPHORIA_BOOM_MIN_ETF = BASE_ETF * gate_mult
    P.EUPHORIA_BOOM_MIN_SINGLE = BASE_SINGLE * gate_mult


def _one(gate_mult: float, truth_mult: float) -> dict:
    from analytics.euphoria_phases import rebuild_phase_files
    from src.config import PROCESSED_DIR
    _set(gate_mult, truth_mult)
    t0 = time.time()
    rebuild_phase_files(verbose=False, research=True)
    rep = json.load(open(os.path.join(
        PROCESSED_DIR, "euphoria_desk_report.json"), encoding="utf-8"))
    out = {"gate_mult": gate_mult, "truth_mult": truth_mult,
           "secs": round(time.time() - t0, 1)}
    for side in ("get_out", "get_in"):
        w = rep[side]["walk_forward"]
        out[side] = {
            "captured": w["captured"], "detectable": w["detectable"],
            "capture_rate": w["capture_rate"],
            "false_alarms": w["false_alarms"],
            "fa_per_iy": w["fa_per_iy"],
            "median_lead_days": w.get("median_lead_days"),
            "threshold": rep[side]["live_threshold"],
            # the project's pre-stated adoption rule, both directions
            "utility": w["captured"] - w["false_alarms"],
        }
    return out


def _table(rows: list, moving: str) -> str:
    lab = "gate x" if moving == "gate" else "truth x"
    key = "gate_mult" if moving == "gate" else "truth_mult"
    head = (f"{lab:>8}  {'GET OUT: cap/detect':>21} {'rate':>6} {'FA':>4} "
            f"{'lead':>5} {'util':>5}   {'GET IN: cap/detect':>20} "
            f"{'rate':>6} {'FA':>4} {'util':>6}")
    lines = [head, "-" * len(head)]
    for r in rows:
        o, i = r["get_out"], r["get_in"]
        lines.append(
            f"{r[key]:>8.2f}  {o['captured']:>9}/{o['detectable']:<11} "
            f"{o['capture_rate']:>6.3f} {o['false_alarms']:>4} "
            f"{str(o['median_lead_days']):>5} {o['utility']:>5}   "
            f"{i['captured']:>8}/{i['detectable']:<11} "
            f"{i['capture_rate']:>6.3f} {i['false_alarms']:>4} "
            f"{i['utility']:>6}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["gate", "truth", "both"],
                    default="both")
    ap.add_argument("--mults", default=None,
                    help="comma-separated multipliers of the frozen "
                         "0.25/0.50 (default 0.6,0.8,1.0,1.2,1.4)")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "docs", "research", "boom_min_sweep.json"))
    args = ap.parse_args()
    mults = ([float(x) for x in args.mults.split(",")]
             if args.mults else DEFAULT_MULTS)

    from src.config import PROCESSED_DIR
    stash = tempfile.mkdtemp(prefix="boomsweep_")
    for f in TOUCHED:
        p = os.path.join(PROCESSED_DIR, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(stash, f))
    print(f"stored files stashed in {stash} - they WILL be restored\n")

    result = {"base": {"etf": BASE_ETF, "single": BASE_SINGLE},
              "mults": mults, "gate": [], "truth": []}
    try:
        if args.study in ("gate", "both"):
            print("STUDY A - THE GATE (selection; ground truth held "
                  "fixed at the frozen 0.25/0.50)")
            for m in mults:
                r = _one(gate_mult=m, truth_mult=1.0)
                result["gate"].append(r)
                print(f"  gate x{m:.2f} done in {r['secs']}s")
            print("\n" + _table(result["gate"], "gate") + "\n")
        if args.study in ("truth", "both"):
            print("STUDY B - THE EXAM (SENSITIVITY ONLY - these numbers "
                  "may not be used to choose a value)")
            for m in mults:
                r = _one(gate_mult=1.0, truth_mult=m)
                result["truth"].append(r)
                print(f"  truth x{m:.2f} done in {r['secs']}s")
            print("\n" + _table(result["truth"], "truth") + "\n")
    finally:
        for f in TOUCHED:
            src = os.path.join(stash, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(PROCESSED_DIR, f))
        print(f"frozen stores restored from {stash}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w", encoding="utf-8"),
              indent=1, default=str)
    print(f"saved -> {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
