"""
run_analytics.py
================
Recompute every derived output in one command - the .py replacement for
executing notebooks 08, 09 and 10 through nbconvert.

    python -m analytics.run_analytics            # conviction + signals
    python -m analytics.run_analytics --what conviction
    python -m analytics.run_analytics --what signals

WHY THIS IS FAST WHERE THE NOTEBOOKS WERE SLOW
----------------------------------------------
The notebook chain paid three separate taxes on every run:
  1. process startup x N notebooks (a fresh Jupyter kernel each, ~5-10s a
     time before any work starts);
  2. matplotlib rendering of every chart into the .ipynb file (hundreds of
     static images nobody looks at until they open the notebook - and the
     reason the .ipynb files grew to 1-4 MB each);
  3. JSON-serialising the whole notebook back to disk, with the atomic
     tmp-file dance to survive interruption.
Here the same mathematics runs once, in one process, with zero rendering -
the dashboard draws charts interactively from the saved parquet outputs
instead. Full recompute over nine years of aggregates: a few seconds.

WHAT GETS WRITTEN (identical filenames/schemas to the notebooks)
  daily_ticker_conviction.parquet    (was notebook 08's output)
  daily_theme_conviction.parquet     (was notebook 09's output)
  trade_signals.parquet              (was notebook 10's theme output)
  trade_signals_tickers.parquet      (was notebook 10's ticker output)

The two stages are independent (conviction files are a dashboard input,
the signal engine builds its own ingredients from the raw aggregates), so
they run in PARALLEL - each in its own worker - and the whole step costs
only the slower of the two.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time

# allow "python analytics/run_analytics.py" as well as "-m analytics.run_analytics"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_conviction():
    from analytics.conviction import rebuild_conviction_files
    return rebuild_conviction_files(verbose=True)


def run_signals(start=None, end=None):
    from analytics.signals import rebuild_signal_files
    return rebuild_signal_files(start=start, end=end, verbose=True)


def run_euphoria(research=None):
    """The top-detector. LIVE mode (default): score today's data at the
    FROZEN walk-forward threshold - seconds; the validation record
    (walk-forward + ablation + ML challenger) is a research artifact,
    refreshed only by --research runs or auto-triggered when the data
    rolls into an uncovered year (intra-year recompute is a no-op by the
    walk-forward convention - thresholds train on strictly earlier
    years). Needs prices; skips gracefully when absent."""
    import os
    from src.config import PRICES_PATH
    if not os.path.exists(PRICES_PATH):
        print("  (euphoria skipped - no prices.parquet; run "
              "pull_bloomberg_prices.py first)")
        return {}
    from analytics.euphoria import main as euphoria_main
    return euphoria_main(research=research)


def run_influence():
    """The influence tracker's LIVE hook: parse any raw files that landed
    since the last run, extend the committed store, rescore the board.
    Silent no-op on a machine with no raw data and no store yet."""
    from analytics.influence import update as influence_update
    return influence_update()


def run_phases(research=None):
    """The euphoria ONSET detector (the July-2026 phases study winner).
    LIVE mode (default): today's scores/alerts at the frozen threshold;
    RESEARCH (--research or auto on year rollover): re-run the winner's
    walk-forward scorecard + threshold selection. Needs prices; skips
    gracefully when they are absent, like euphoria."""
    import os
    from src.config import PRICES_PATH
    if not os.path.exists(PRICES_PATH):
        print("  (phases skipped - no prices.parquet; run "
              "pull_bloomberg_prices.py first)")
        return {}
    from analytics.euphoria_phases import rebuild_phase_files
    return rebuild_phase_files(research=research)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Recompute conviction + trading signals from the aggregates.")
    p.add_argument("--what",
                   choices=["all", "conviction", "signals", "euphoria",
                            "influence", "phases"],
                   default="all")
    p.add_argument("--start", default=None,
                   help="signal engine window start (YYYY-MM-DD; default: "
                        "whole aggregate history - the live behaviour)")
    p.add_argument("--end", default=None,
                   help="signal engine window end, exclusive (windowed runs "
                        "are how the old ticker backtests were produced)")
    p.add_argument("--serial", action="store_true",
                   help="run the stages one after another (clearer output "
                        "when debugging)")
    p.add_argument("--research", action="store_true",
                   help="force the FULL validation pass for euphoria and "
                        "phases (walk-forward + ablation + ML challenger, "
                        "threshold re-selection). Without it, live runs "
                        "score at the frozen thresholds in seconds; a "
                        "research pass also auto-triggers when the data "
                        "rolls into a year the stored record does not "
                        "cover, or when a report is missing. Run after "
                        "backfills or rule changes (desk decision "
                        "2026-07-24: research decides once, live scores)")
    args = p.parse_args(argv)

    t0 = time.time()
    jobs = []
    if args.what in ("all", "conviction"):
        jobs.append(("conviction (was nb 08+09)", run_conviction))
    if args.what in ("all", "signals"):
        # functools.partial (not a lambda): partials of module-level
        # functions can be pickled into the process-pool workers below.
        import functools
        jobs.append(("signals (was nb 10)",
                     functools.partial(run_signals, args.start, args.end)))
    import functools as _ft
    research = True if args.research else None      # None = auto-decide
    if args.what in ("all", "euphoria"):
        jobs.append(("euphoria (the top detector)",
                     _ft.partial(run_euphoria, research)))
    if args.what in ("all", "phases"):
        jobs.append(("phases (the onset detector)",
                     _ft.partial(run_phases, research)))
    if args.what in ("all", "influence"):
        jobs.append(("influence (live board update)", run_influence))

    print(f"analytics: {len(jobs)} stage(s), "
          f"{'serial' if args.serial or len(jobs) == 1 else 'parallel'}")
    failed = []
    if args.serial or len(jobs) == 1:
        for label, fn in jobs:
            print(f"--- {label} ---")
            try:
                fn()
            except Exception as exc:
                print(f"  FAILED: {exc}")
                failed.append(label)
    else:
        # ProcessPool (not threads): both stages are pandas-heavy and hold
        # the GIL, so separate processes are what actually overlaps them.
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {pool.submit(fn): label for label, fn in jobs}
            for fut in concurrent.futures.as_completed(futures):
                label = futures[fut]
                try:
                    fut.result()
                    print(f"--- {label}: done ---")
                except Exception as exc:
                    print(f"--- {label}: FAILED: {exc} ---")
                    failed.append(label)

    print(f"analytics finished in {time.time() - t0:.1f}s"
          + (f" | FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
