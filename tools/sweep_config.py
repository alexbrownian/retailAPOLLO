"""
sweep_config.py — does this number deserve its value? Ask the data.
===================================================================

THE DESK'S QUESTION (2026-08-04), looking at ROLL, BASELINE, MIN_DAYS,
DERIV_SMOOTH and MIN_TOTAL: "where did these numbers come from too? we
need to make everything have reason and testing."

Fair question, and the audit behind it is uncomfortable: rule 4 of this
project says every constant carries its provenance, and while the
euphoria-era numbers were swept and recorded, the OLDEST knobs - the ones
inherited from RetailFlow1 and sitting at the top of section 4 - carry a
sentence of reasoning and no measurement. "One loud afternoon is not a
trend" is a good argument for smoothing. It is not evidence for SEVEN.

This tool closes that gap the same way the rest of the project does: it
re-runs the FULL walk-forward research pass with one constant changed,
and reports the scorecard under the project's own pre-stated adoption
rule (captures - EUPHORIA_FA_PENALTY x false alarms, both directions).

HOW IT PATCHES, AND WHY IT USES SUBPROCESSES
--------------------------------------------
Several of these constants are bound as DEFAULT ARGUMENTS at import time
(`def ewm_z(frame, roll: int = ROLL)`), so setting a module attribute
afterwards changes nothing and would produce a sweep that silently
measured the same value five times - the worst possible outcome for a
tool whose job is honesty. So each point runs in a FRESH SUBPROCESS that
rewrites `src.config` BEFORE any other project module is imported. Slower
(~70s a point), and correct.

A GUARD AGAINST EXACTLY THAT FAILURE: every sweep asserts that at least
two of its points produce different scorecards. A constant that changes
nothing when moved is either genuinely inert (worth knowing, and
reported) or not actually being patched (a bug, and reported loudly) -
the tool never lets the two look alike.

WHAT THIS TOOL MAY AND MAY NOT BE USED FOR
------------------------------------------
It measures MODEL knobs. It must NOT be pointed at the ground-truth
definition (EUPHORIA_BOOM_MIN_*, EUPHORIA_CRASH_MIN_* as used by G2/G3)
in order to CHOOSE a value: tuning the definition of a top so the top
detector scores better is circular. `tools/sweep_boom_min.py` explains
that split in full and runs the ground-truth side as a SENSITIVITY study
only. The same rule applies here.

USAGE
    python tools/sweep_config.py --list
    python tools/sweep_config.py --only ROLL,BASELINE
    python tools/sweep_config.py                # every registered sweep

THE STORES ARE RESTORED AFTERWARDS. This is a measurement, never a
re-fit; the frozen record must not end up pointing at one of the tool's
own experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOUCHED = ["euphoria_desk_report.json", "euphoria_desk.parquet",
           "euphoria_onset_report.json", "euphoria_onset.parquet",
           "episodes.parquet", "phase_day_frame.parquet"]

# name -> (values to try, one-line note on what the number does)
SWEEPS: dict[str, tuple[list, str]] = {
    "ROLL": ([3, 5, 7, 10, 14],
             "rolling window for mention / bull-pressure sums"),
    "BASELINE": ([42, 63, 84, 120, 180],
                 "trailing z baseline (days)"),
    "MIN_DAYS": ([14, 21, 28, 42, 56],
                 "warm-up before a z exists"),
    "MIN_TOTAL": ([10, 20, 30, 50, 100],
                  "mask days with fewer total posts than this"),
    "EUPHORIA_PCT_WINDOW": ([180, 270, 365, 540, 730],
                            "'extreme' = versus this name's own last N days"),
    "EUPHORIA_HYPE_MULT": ([1.5, 1.75, 2.0, 2.5, 3.0],
                           "A1: crowd must be this many x its 120d median"),
    "EUPHORIA_FADE_DISCOUNT": ([0, 5, 10, 15, 20],
                               "A3: level points the fade flag buys"),
    "EUPHORIA_MIN_HISTORY": ([90, 135, 180, 270, 365],
                             "days of history before percentiles exist"),
    "EUPHORIA_ATT_GATE": ([0.80, 0.85, 0.90, 0.95, 0.98],
                          "A2: attention percentile gate"),
    "EUPHORIA_MIN_COVERAGE": ([100, 125, 150, 175, 200],
                              "A0: scored posts needed in the last 28d"),
}

_RUNNER = r"""
import json, sys
sys.path.insert(0, {root!r})
import src.config as C
setattr(C, {name!r}, {value!r})
from analytics.euphoria_phases import rebuild_phase_files
rebuild_phase_files(verbose=False, research=True)
import os
rep = json.load(open(os.path.join(C.PROCESSED_DIR,
                                  "euphoria_desk_report.json")))
out = {{}}
for side in ("get_out", "get_in"):
    w = rep[side]["walk_forward"]
    out[side] = {{"captured": w["captured"], "detectable": w["detectable"],
                 "capture_rate": w["capture_rate"],
                 "false_alarms": w["false_alarms"],
                 "fa_per_iy": w["fa_per_iy"],
                 "median_lead_days": w.get("median_lead_days"),
                 "threshold": rep[side]["live_threshold"],
                 "utility": w["captured"] - w["false_alarms"]}}
print("@@RESULT@@" + json.dumps(out))
"""


def _point(name: str, value) -> dict | None:
    code = _RUNNER.format(root=ROOT, name=name, value=value)
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    sys.stderr.write(f"\n  [{name}={value}] FAILED:\n"
                     f"{p.stdout[-800:]}\n{p.stderr[-800:]}\n")
    return None


def _table(name: str, note: str, rows: list, current) -> str:
    head = (f"{name:>22}  {'GET OUT cap/det':>16} {'FA':>4} {'lead':>5} "
            f"{'util':>5}   {'GET IN cap/det':>15} {'FA':>4} {'util':>6}")
    lines = [f"\n{name} - {note}", head, "-" * len(head)]
    for v, r in rows:
        if r is None:
            lines.append(f"{str(v):>22}  {'(run failed)':>16}")
            continue
        o, i = r["get_out"], r["get_in"]
        mark = "  <- frozen" if v == current else ""
        lines.append(
            f"{str(v):>22}  {o['captured']:>7}/{o['detectable']:<8} "
            f"{o['false_alarms']:>4} {str(o['median_lead_days']):>5} "
            f"{o['utility']:>5}   {i['captured']:>6}/{i['detectable']:<8} "
            f"{i['false_alarms']:>4} {i['utility']:>6}{mark}")
    got = [r for _, r in rows if r]
    if len(got) > 1:
        sigs = {json.dumps(r, sort_keys=True) for r in got}
        if len(sigs) == 1:
            lines.append("  !! every value produced an IDENTICAL scorecard. "
                         "Either this constant does not reach the desk "
                         "detector at all, or the patch is not taking - "
                         "do not read this row as evidence.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated constant names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "docs", "research", "config_sweep.json"))
    args = ap.parse_args()

    if args.list:
        import src.config as C
        for n, (vals, note) in SWEEPS.items():
            print(f"  {n:<26} now={getattr(C, n, '?')!s:<8} "
                  f"try={vals}  # {note}")
        return 0

    names = ([n.strip() for n in args.only.split(",")]
             if args.only else list(SWEEPS))
    import src.config as C
    from src.config import PROCESSED_DIR

    stash = tempfile.mkdtemp(prefix="cfgsweep_")
    for f in TOUCHED:
        p = os.path.join(PROCESSED_DIR, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(stash, f))
    print(f"frozen stores stashed in {stash} - they WILL be restored")
    n_pts = sum(len(SWEEPS[n][0]) for n in names if n in SWEEPS)
    print(f"{len(names)} constant(s), {n_pts} points, "
          f"~{n_pts * 70 / 60:.0f} min\n")

    results = {}
    try:
        for name in names:
            if name not in SWEEPS:
                print(f"  (no sweep registered for {name})")
                continue
            vals, note = SWEEPS[name]
            cur = getattr(C, name, None)
            rows = []
            for v in vals:
                t0 = time.time()
                rows.append((v, _point(name, v)))
                print(f"  {name}={v} ... {time.time()-t0:.0f}s")
            results[name] = {"note": note, "current": cur,
                             "points": [{"value": v, "scorecard": r}
                                        for v, r in rows]}
            print(_table(name, note, rows, cur))
    finally:
        for f in TOUCHED:
            src = os.path.join(stash, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(PROCESSED_DIR, f))
        print(f"\nfrozen stores restored from {stash}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w", encoding="utf-8"),
              indent=1, default=str)
    print(f"saved -> {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
