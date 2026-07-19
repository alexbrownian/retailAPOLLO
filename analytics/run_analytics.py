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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Recompute conviction + trading signals from the aggregates.")
    p.add_argument("--what", choices=["all", "conviction", "signals"],
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
