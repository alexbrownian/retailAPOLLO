"""
run_analytics.py
================
Recompute every derived output in one command.

    cd Code
    python -m src.analytics.run_analytics            # conviction + signals
    python -m src.analytics.run_analytics --what conviction
    python -m src.analytics.run_analytics --what signals

(``python Code/src/analytics/run_analytics.py`` from the project root
works as well.)

The mathematics runs once, in one process, with no chart rendering; the
dashboard draws its charts interactively from the saved parquet outputs.
A full recompute over nine years of aggregates takes a few seconds.

WHAT GETS WRITTEN
  daily_theme_conviction.parquet     per-theme conviction (ticker
                                     conviction is computed live by the
                                     dashboard; no file)
  trade_signals.parquet              theme signals
  trade_signals_tickers.parquet      ticker signals

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

# allow "python src/analytics/run_analytics.py" as well as "-m src.analytics.run_analytics"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_conviction():
    from src.analytics.conviction import rebuild_conviction_files
    return rebuild_conviction_files(verbose=True)


def run_signals(start=None, end=None):
    from src.analytics.signals import rebuild_signal_files
    return rebuild_signal_files(start=start, end=end, verbose=True)


def run_euphoria(research=None):
    """The top-detector. LIVE mode (default): score today's data at the
    FROZEN walk-forward threshold - seconds; the validation record
    (walk-forward + ablation + ML challenger) is a research artifact,
    refreshed only by --research runs or when no usable record exists
    yet (the bootstrap case). A record that lags the data is reported,
    never refitted as a side effect. Needs prices; skips gracefully
    when absent."""
    import os
    from src.config import PRICES_PATH
    if not os.path.exists(PRICES_PATH):
        print("  (euphoria skipped - no prices.parquet; run "
              "Code/ingestion/pull_prices.py first)")
        return {}
    from src.analytics.euphoria import main as euphoria_main
    return euphoria_main(research=research)


def run_influence():
    """The influence tracker's LIVE hook: parse any raw files that landed
    since the last run, extend the committed store, rescore the board.
    Silent no-op on a machine with no raw data and no store yet."""
    from src.analytics.influence import update as influence_update
    return influence_update()


def run_phases(research=None):
    """The euphoria ONSET detector (the phases-study winner).
    LIVE mode (default): today's scores/alerts at the frozen threshold;
    RESEARCH (--research, or automatically when no record exists yet):
    re-run the winner's walk-forward scorecard + threshold selection.
    Needs prices; skips gracefully when they are absent, like euphoria."""
    import os
    from src.config import PRICES_PATH
    if not os.path.exists(PRICES_PATH):
        print("  (phases skipped - no prices.parquet; run "
              "Code/ingestion/pull_prices.py first)")
        return {}
    from src.analytics.euphoria_phases import rebuild_phase_files
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
                        "are how ticker backtests are produced)")
    p.add_argument("--serial", action="store_true",
                   help="run the stages one after another (clearer output "
                        "when debugging)")
    p.add_argument("--research", action="store_true",
                   help="force the FULL validation pass for euphoria and "
                        "phases (walk-forward + ablation + ML challenger, "
                        "threshold re-selection). Without it, live runs "
                        "score at the frozen thresholds in seconds; a "
                        "research pass also runs automatically when a "
                        "report is missing (the bootstrap case). A record "
                        "that lags the data is reported, not refitted. "
                        "Run after backfills or rule changes (research "
                        "decides once, live scores)")
    args = p.parse_args(argv)

    t0 = time.time()
    jobs = []
    if args.what in ("all", "conviction"):
        jobs.append(("conviction (ticker + theme)", run_conviction))
    if args.what in ("all", "signals"):
        # functools.partial (not a lambda): partials of module-level
        # functions can be pickled into the process-pool workers below.
        import functools
        jobs.append(("signals (theme + ticker)",
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
