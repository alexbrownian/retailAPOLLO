"""Sweep the BUBBLE/BUST ground truth and report what actually changes.

Desk request 2026-08-05: *"can you test for me if we change the criteria
(the ground truths) for bubbles and bust if we can improve the hit rate
that way? ... but keep good results like memory / gold / game stop etc.
right now its like 25 for etf 50 for individual"*

THE TRAP THIS TOOL IS BUILT AROUND
----------------------------------
Hit rate is `captured / detectable`, and BOTH move when the ground truth
moves. Loosen the bars and `detectable` grows faster than `captured`, so
the rate falls. Tighten them and `detectable` collapses, so the rate
rises — while the detector has done nothing differently and is simply
being graded on fewer, larger events.

So "improve the hit rate" is a request that a naive sweep answers by
recommending the TIGHTEST bars on the grid, every time, for a reason that
has nothing to do with the detector being better.

AND IT IS WORSE THAN THAT - the correction that this tool's first draft
got wrong and the sweep itself exposed. False alarms are not
definition-independent either. An alert is a false alarm when no
gradeable peak follows it, so LOOSENING the bars converts existing false
alarms into hits without the detector doing anything: measured on this
store, FA/instrument-year falls 0.210 -> 0.140 as the bars go to 0.5x.
Threshold selection then moves too, because it optimises
(hits - penalty x false alarms) against whatever ground truth it is
handed.

The conclusion is uncomfortable and worth stating plainly: **no column in
a walk-forward is comparable across two different ground truths.**
Changing the bars changes the exam, not the student. Every row below is
internally valid and cross-row comparisons of `rate`, `captured` and
`fa_per_iy` are all, to some degree, artefacts.

What the sweep CAN honestly answer is two things:

  * `anchors_kept` - do the episodes the desk NAMED still exist? The
                     memory top, the gold run, the GameStop squeeze. A
                     setting that improves a ratio by deleting GameStop
                     from the ground truth has improved nothing, and this
                     column is what catches it.
  * `instruments_with_any_episode` - how much of the universe has a
                     gradeable record at all. This is a real, decidable
                     consequence: at 0.5x it goes 43 -> 58 names.

And one question the sweep cannot answer at all, which is the one that
should decide it: at 0.5x an ETF "bubble" is a 12.5% run-up and a 7.5%
fall. That is an ordinary quarter. Whether the word should still be
"bubble" is a judgement about meaning, not a number.

Read the output that way: `rate` is context, the other three decide.

Usage
-----
    python tools/sweep_ground_truth.py                # the default grid
    python tools/sweep_ground_truth.py --quick        # 3 settings, fast
    python tools/sweep_ground_truth.py --json out.json

Nothing here writes to config or to any store. It prints, and optionally
saves a JSON record. Adopting a row is a separate, deliberate edit to
`src/config.py` followed by a full `--what euphoria --research` pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analytics.euphoria as E                                # noqa: E402
from src.config import (EUPHORIA_BOOM_MIN_ETF,                # noqa: E402
                        EUPHORIA_BOOM_MIN_SINGLE,
                        EUPHORIA_CRASH_MIN_ETF,
                        EUPHORIA_CRASH_MIN_SINGLE,
                        PRICES_PATH)

# The episodes the desk named as "good results" that must survive any
# change. Each is (instrument, the window it should fall in, what it was).
# A setting that loses one of these is rejected however good its ratios
# look - that is the whole point of naming them in advance rather than
# reading the winner's episode list afterwards.
ANCHORS = [
    ("memory", "2026-01-01", "2026-12-31", "the 2026 memory/HBM top"),
    ("gold_metals", "2025-09-01", "2026-12-31", "the gold run"),
    ("GME", "2021-01-01", "2021-12-31", "the GameStop squeeze"),
    ("semiconductors", "2024-01-01", "2026-12-31", "the semis cycle"),
]

# (label, boom_etf, crash_etf, boom_single, crash_single)
# The incumbent sits in the middle so the grid reads outward from it.
GRID = [
    ("tighter  (1.4x)", 0.35, 0.21, 0.70, 0.42),
    ("tighter  (1.2x)", 0.30, 0.18, 0.60, 0.36),
    ("INCUMBENT", EUPHORIA_BOOM_MIN_ETF, EUPHORIA_CRASH_MIN_ETF,
     EUPHORIA_BOOM_MIN_SINGLE, EUPHORIA_CRASH_MIN_SINGLE),
    ("looser   (0.8x)", 0.20, 0.12, 0.40, 0.24),
    ("looser   (0.6x)", 0.15, 0.09, 0.30, 0.18),
    ("looser   (0.5x)", 0.125, 0.075, 0.25, 0.15),
    ("crash-only looser", 0.25, 0.10, 0.50, 0.20),
    ("boom-only looser", 0.15, 0.15, 0.30, 0.30),
]
QUICK = {"INCUMBENT", "looser   (0.8x)", "looser   (0.6x)"}


def _set_bars(boom_etf, crash_etf, boom_single, crash_single):
    """Patch the module-level constants the ground truth reads.

    `ground_truth_peaks` resolves them at call time out of
    `analytics.euphoria`, so patching there is enough and the built
    series do not need rebuilding - the bars gate GRADING, not scoring.
    """
    E.EUPHORIA_BOOM_MIN_ETF = boom_etf
    E.EUPHORIA_CRASH_MIN_ETF = crash_etf
    E.EUPHORIA_BOOM_MIN_SINGLE = boom_single
    E.EUPHORIA_CRASH_MIN_SINGLE = crash_single


def _anchor_check(series, pxmap):
    """Which named episodes still exist as ground truth, and how many
    instruments have any at all."""
    peaks_by, _ = E.peak_maps(series, pxmap)
    kept = []
    for name, lo, hi, _what in ANCHORS:
        got = peaks_by.get(name) or []
        lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
        kept.append(any(lo_t <= pd.Timestamp(d) <= hi_t for d in got))
    measurable = sum(1 for v in peaks_by.values() if len(v))
    return kept, measurable


def _lookback_sweep(series, pxmap, json_path=None):
    """How far back should the run-up be measured?

    Desk question 2026-08-05: "the comparison to trailing low (how many
    days now and why)?" 120 days, and this is the why. The number was
    typed inside `ground_truth_peaks` and never surfaced until it was
    asked about; it is now EUPHORIA_BOOM_LOOKBACK_D in src/config.py.

    Note it is NOT the 54d live boom gate. Short window = the detector
    may only fire while a boom is CURRENT. Long window = a top that took
    four months to build is still graded as a top."""
    base = E.EUPHORIA_BOOM_LOOKBACK_D
    rows = []
    print(f"{'lookback':>9s} {'peaks':>6s} {'det':>5s} {'cap':>4s} "
          f"{'rate':>6s} {'FA/iy':>7s} {'anchors':>8s}")
    try:
        for d in (54, 90, 120, 180, 250, 365):
            E.EUPHORIA_BOOM_LOOKBACK_D = d
            kept, _meas = _anchor_check(series, pxmap)
            o = E.walk_forward(series, pxmap)["overall"]
            rows.append({"lookback_d": d, "peaks": o["peaks"],
                         "detectable": o["detectable_peaks"],
                         "captured": o["captured"],
                         "rate": o["capture_rate_detectable"],
                         "fa_per_iy": o["fa_per_instrument_year"],
                         "anchors_kept": sum(kept)})
            _r = o["capture_rate_detectable"]
            _f = o["fa_per_instrument_year"]
            print(f"{d:>7d}d {o['peaks']:>6d} {o['detectable_peaks']:>5d} "
                  f"{o['captured']:>4d} "
                  f"{(f'{_r:.3f}' if _r is not None else '   -'):>6s} "
                  f"{(f'{_f:.3f}' if _f is not None else '   -'):>7s} "
                  f"{sum(kept)}/{len(ANCHORS):>6}", flush=True)
    finally:
        E.EUPHORIA_BOOM_LOOKBACK_D = base
    print(f"\nincumbent = {base}d. A LONGER lookback admits slow-built "
          "tops; a shorter one")
    print("grades them as noise because the run-up began outside the "
          "window.")
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({"lookback_sweep": rows}, fh, indent=2)
        print(f"wrote {json_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="three settings instead of the full grid")
    ap.add_argument("--lookback", action="store_true",
                    help="sweep the trailing-low GRADING window instead "
                         "of the boom/crash sizes")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the record as JSON")
    args = ap.parse_args()

    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    series, pxmap = E.build_all_series(prices)
    print(f"universe: {len(series)} instruments "
          f"({sum(1 for s in series if s.kind == 'theme')} themes, "
          f"{sum(1 for s in series if s.kind == 'single')} singles)")
    print("grading bars are swept; the DETECTOR is untouched throughout\n")

    if args.lookback:
        return _lookback_sweep(series, pxmap, args.json)
    grid = [g for g in GRID if not args.quick or g[0] in QUICK]
    rows = []
    base = (EUPHORIA_BOOM_MIN_ETF, EUPHORIA_CRASH_MIN_ETF,
            EUPHORIA_BOOM_MIN_SINGLE, EUPHORIA_CRASH_MIN_SINGLE)
    try:
        for label, be, ce, bs, cs in grid:
            _set_bars(be, ce, bs, cs)
            kept, measurable = _anchor_check(series, pxmap)
            rep = E.walk_forward(series, pxmap)
            o = rep["overall"]
            rows.append({
                "setting": label,
                "boom_etf": be, "crash_etf": ce,
                "boom_single": bs, "crash_single": cs,
                "peaks": o["peaks"],
                "detectable": o["detectable_peaks"],
                "captured": o["captured"],
                "rate_of_detectable": o["capture_rate_detectable"],
                "false_alarms": o["false_alarms"],
                "fa_per_iy": o["fa_per_instrument_year"],
                "median_lead_d": o.get("median_lead_days"),
                "instruments_with_any_episode": measurable,
                "anchors_kept": sum(kept),
                "anchors": {ANCHORS[i][0]: kept[i]
                            for i in range(len(ANCHORS))},
            })
            print(f"  {label:<20s} captured {o['captured']:>3d}  "
                  f"det {o['detectable_peaks']:>4d}  "
                  f"rate {o['capture_rate_detectable']:.3f}  "
                  f"FA/iy {o['fa_per_instrument_year']:.3f}  "
                  f"anchors {sum(kept)}/{len(ANCHORS)}", flush=True)
    finally:
        _set_bars(*base)      # never leave the module patched

    print("\n" + "=" * 100)
    print("RESULTS - ranked by CAPTURED (absolute tops caught), not by rate")
    print("=" * 100)
    hdr = (f"{'setting':<20s} {'boom/crash ETF':>15s} {'single':>13s} "
           f"{'cap':>4s} {'det':>5s} {'rate':>6s} {'FA/iy':>7s} "
           f"{'lead':>5s} {'names':>6s} {'anchors':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["captured"]):
        print(f"{r['setting']:<20s} "
              f"{r['boom_etf']:>7.1%}/{r['crash_etf']:<7.1%} "
              f"{r['boom_single']:>5.0%}/{r['crash_single']:<6.0%} "
              f"{r['captured']:>4d} {r['detectable']:>5d} "
              f"{r['rate_of_detectable']:>6.3f} {r['fa_per_iy']:>7.3f} "
              f"{str(r['median_lead_d']):>5s} "
              f"{r['instruments_with_any_episode']:>6d} "
              f"{r['anchors_kept']}/{len(ANCHORS):<6d}")
    print()
    print("READ THIS BEFORE THE TABLE. No column here is comparable")
    print("across rows: changing the ground truth changes the EXAM, not")
    print("the detector. `det` moves by construction; `rate` therefore")
    print("moves; and `FA/iy` moves too, because loosening the bars turns")
    print("alerts that had no peak near them into hits. Threshold")
    print("selection then shifts as well, since it optimises")
    print("(hits - penalty x FA) against whatever truth it is given.")
    print()
    print("What IS decidable: do the named episodes survive (anchors),")
    print("and how much of the universe becomes gradeable (names).")
    print("names = instruments with at least one gradeable episode.")
    print(f"anchors = of {len(ANCHORS)} desk-named episodes still present: "
          + ", ".join(a[0] for a in ANCHORS))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"universe": len(series), "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
