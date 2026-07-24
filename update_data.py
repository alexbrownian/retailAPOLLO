#!/usr/bin/env python
"""
update_data.py - the single pipeline entry point. Set the window, run it.

    python update_data.py

TWO MODES, ONE KNOB (the window in src/config.py, overridable per run):

  LIVE MODE      END_DATE = ""   (the day-to-day default)
      Fast path. Fetch the lookback window of the most popular posts from
      every live source (all three fetchers IN PARALLEL), fold them into
      the stores, recompute conviction + signals (pure .py analytics - a
      few seconds, no notebooks anywhere), pull prices. The dashboard then
      renders everything interactively.

  BACKTEST MODE  END_DATE = "YYYY-MM-DD"
      View a past regime - instant. The aggregates are WINDOW-INDEPENDENT
      (built once over BUILD_START_DATE -> today via --full), so a
      backtest fetches nothing and rebuilds nothing unless the derived
      outputs are stale. The window drives the price pull and what the
      dashboard shows.

    Examples:
        --start 2021-01-01 --end 2021-11-01   -> view Jan-Oct 2021
        --start 2021-01-01                    -> LIVE, 2021 -> today

RUNS ON BOTH MACHINES (auto-detected, exactly like RetailFlow1)
    * EXTERNAL machine (data/processed/posts.parquet present)
        Holds the raw post store. Live mode merges new posts into the
        store and splices the aggregate tail; --full rebuilds every
        aggregate over the whole build range from raw text.
    * INTERNAL machine (no posts.parquet)
        Holds only ABSTRACTED_DATA - text-free daily aggregates. Live
        posts fold straight into the aggregates; raw text never lands.
    Either way the run ends by verifying ABSTRACTED_DATA carries no text.

WHAT REPLACED THE NOTEBOOKS
    old: nbconvert-executes 08/09/10 (+ overlays 11-16), minutes + JSON
         re-serialisation + truncation risk
    new: analytics/run_analytics.py - the identical mathematics as importable
         functions, parallelised, seconds. The overlay charts are computed
         on demand by the dashboard from the same saved outputs.

EVERY RUN ALSO PRINTS
    * a DATA COVERAGE table (posts per month, per source) - gaps at a glance
    * a WINDOW CHECK - whether the chosen view window actually has data,
      per source, so an empty chart is never a mystery
    * a RUN SUMMARY - the key facts in one glance
    * the SAFETY CHECK verdict - PASS before committing ABSTRACTED_DATA

FLAGS
    --full           rebuild the aggregates over BUILD_START_DATE -> today
                     (external machine; run once initially and after
                     theme/schema changes)
    --fetch          force API fetching in backtest mode
    --skip-fetch     recompute only, no API calls
    --skip-prices    skip the Bloomberg pull
    --start / --end  override the window for this run only
    --external / --internal   force a machine mode instead of auto-detecting
    --dry-run        print the plan, run nothing
"""

import argparse
import datetime
import os
import subprocess
import sys

try:                     # posts contain emoji/links; avoid cp1252 crashes
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import config                                        # noqa: E402
from src.config import (LOG_DIR, SNAPSHOT_DIR, PROCESSED_DIR,  # noqa: E402
                        ABSTRACTED_DIR, PRICES_PATH, POSTS_PATH,
                        BUILD_START_DATE, FETCH_LOOKBACK_DAYS,
                        FETCH_MAX_CREDITS, FORBIDDEN_COLS, MAX_ABSTRACTED_MB)

SIGNAL_FILES = ["trade_signals.parquet", "trade_signals_tickers.parquet"]


def log(msg, fh=None):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def run(cmd, fh, dry, show=False):
    """Run a child command. show=True streams its output live to the
    terminal (long steps show progress); otherwise output is captured
    quietly. Returns the exit code (0 on --dry-run)."""
    log("RUN  " + " ".join(cmd), fh)
    if dry:
        return 0
    if show:
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            log("FAIL (see the output above)", fh)
        return r.returncode
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        log("FAIL " + (r.stderr or r.stdout)[-800:], fh)
    return r.returncode


# ---------------------------------------------------------------------------
# DATA COVERAGE - what the store holds, month by month
# ---------------------------------------------------------------------------
def _compact(n):
    """1234 -> '1.2k', 2500000 -> '2.5M' - keeps the coverage table narrow."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(int(n))


def print_data_coverage(fh, internal):
    """A year x month table of data held, per source, so coverage gaps are
    visible at a glance after every run. The external machine counts POSTS
    from posts.parquet; the internal machine has no raw store, so it counts
    MENTIONS from the aggregates (same table shape, same gaps)."""
    import pandas as pd

    if not internal:
        import pyarrow.parquet as pq
        if not os.path.exists(POSTS_PATH):
            return
        df = pq.read_table(POSTS_PATH, columns=["date", "source"]).to_pandas()
        value_label = "posts"
    else:
        path = os.path.join(PROCESSED_DIR, "daily_ticker_counts_by_source.parquet")
        if not os.path.exists(path):
            return
        df = pd.read_parquet(path)
        value_label = "ticker mentions"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for source in sorted(df["source"].unique()):
        one = df[df["source"] == source]
        if not internal:
            counts = one.groupby(["year", "month"]).size()
        else:
            counts = one.groupby(["year", "month"])["mention_count"].sum()
        log(f"--- DATA COVERAGE: {value_label} per month | source = {source} ---", fh)
        log("  year " + "".join(f"{m:>7}" for m in months), fh)
        for year in sorted(one["year"].unique()):
            cells = []
            for mo in range(1, 13):
                n = counts.get((year, mo), 0)
                cells.append(f"{_compact(n):>7}" if n else f"{'.':>7}")
            log(f"  {year} " + "".join(cells), fh)
    log("('.' = NO data that month - a gap, not a quiet month)", fh)


def check_window_coverage(fh, start, end):
    """Flag up-front whether the chosen VIEW window actually has data, per
    source, so an empty chart is never a mystery. Returns True if at least
    one source covers part of the window."""
    import pandas as pd

    path = os.path.join(PROCESSED_DIR, "daily_ticker_counts_by_source.parquet")
    if not os.path.exists(path):
        log("window check: no aggregates yet - hydrate (internal) or run "
            "'python update_data.py --full' (external) to build them", fh)
        return False

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    agg_lo, agg_hi = df["date"].min(), df["date"].max()
    lo = pd.to_datetime(start)
    hi = pd.to_datetime(end) if end else agg_hi

    log(f"--- WINDOW CHECK: view {lo.date()} -> {hi.date()} | "
        f"aggregates span {agg_lo.date()} -> {agg_hi.date()} ---", fh)
    win = df[(df["date"] >= lo) & (df["date"] <= hi)]
    any_data = False
    for source in sorted(df["source"].unique()):
        one = win[win["source"] == source]
        if len(one):
            any_data = True
            days = one["date"].nunique()
            span_days = max((hi - lo).days, 1)
            pct = min(100, round(100 * days / span_days))
            log(f"  {source:<10} [OK] {one['mention_count'].sum():,} mentions "
                f"over {days} days (~{pct}% of the window)", fh)
        else:
            full = df[df["source"] == source]
            log(f"  {source:<10} [NO DATA in window] this source spans "
                f"{full['date'].min().date()} -> {full['date'].max().date()}", fh)
    if not any_data:
        log("  >>> NO source has data in this window. If the raw store covers "
            "it, run 'python update_data.py --full' on the external machine; "
            "otherwise widen the window.", fh)
    return any_data


# ---------------------------------------------------------------------------
# SAFETY CHECK - the committed aggregates must stay text-free
# ---------------------------------------------------------------------------
def verify_abstracted(fh):
    """Confirm ABSTRACTED_DATA is present, small, and carries no columns
    that could reveal raw posts. Reads only each file's schema footer, so
    the check is instant. Returns True if safe to commit."""
    import pyarrow.parquet as pq
    from src import abstracted_data

    log("safety check: ABSTRACTED_DATA is text-free?", fh)
    all_present = all_safe = True
    for name in abstracted_data.FILES:
        path = os.path.join(ABSTRACTED_DIR, name)
        if not os.path.exists(path):
            log(f"  MISSING  {name}", fh)
            all_present = False
            all_safe = False
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        cols = list(pq.ParquetFile(path).schema_arrow.names)
        bad = [c for c in cols if c.lower() in FORBIDDEN_COLS]
        flag = ""
        if bad:
            flag = f"  <-- LEAK {bad}"
            all_safe = False
        if size_mb > MAX_ABSTRACTED_MB:
            flag += f"  <-- TOO BIG ({size_mb:.1f} MB)"
            all_safe = False
        log(f"  {name:<40} {size_mb:5.2f} MB | {cols}{flag}", fh)

    ok = all_present and all_safe
    log(f"safety check: {'PASS - safe to commit' if ok else 'FAIL - do NOT commit'}", fh)
    return ok


def main():
    p = argparse.ArgumentParser(
        description="Refresh all data for the configured window (src/config.py).")
    p.add_argument("--start", default=config.START_DATE,
                   help="override START_DATE this run")
    p.add_argument("--end", default=config.END_DATE,
                   help="override END_DATE this run ('' = live)")
    p.add_argument("--full", action="store_true",
                   help="rebuild the aggregates over BUILD_START_DATE -> today")
    p.add_argument("--fetch", action="store_true",
                   help="force API fetching in backtest mode")
    p.add_argument("--skip-fetch", action="store_true",
                   help="recompute only - no API calls")
    p.add_argument("--skip-prices", action="store_true",
                   help="skip the Bloomberg price pull")
    p.add_argument("--external", action="store_true", help="force external-machine mode")
    p.add_argument("--internal", action="store_true", help="force internal-machine mode")
    p.add_argument("--with-comments", action="store_true",
                   help="ALSO fetch Reddit comments in this run. Desk "
                        "decision 2026-07-24: comments are OFF in the "
                        "daily pipeline (they are the slow fetch - "
                        "10-50x post volume at the API's polite 1s/page); "
                        "the dedicated runner is update_comments.py, "
                        "which also updates the influence board")
    p.add_argument("--skip-panel-review", action="store_true",
                   help="skip the monthly dynamic-panel review (subreddit "
                        "discovery; it is watermarked and only actually "
                        "runs when >= PANEL_REVIEW_DAYS have passed)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = p.parse_args()
    dry = args.dry_run
    py = sys.executable

    # ---- machine mode: external (has raw store) vs internal ----
    if args.external:
        internal = False
    elif args.internal:
        internal = True
    else:
        internal = not os.path.exists(POSTS_PATH)

    # ---- live vs backtest, fast vs full ----
    live = (args.end == "")
    # backtest covers the past, which does not change - no fetching by default
    do_fetch = (live and not args.skip_fetch) or (args.fetch and not args.skip_fetch)
    full_chain = args.full and not internal

    config.ensure_dirs()
    today = datetime.date.today().isoformat()
    fh = open(os.path.join(LOG_DIR, f"run_{today}.log"), "a", encoding="utf-8")

    end_label = args.end if args.end else "LIVE (newest)"
    log("=" * 60, fh)
    log("UPDATE DATA (retailAPOLLO - notebook-free pipeline)", fh)
    log(f"  window : {args.start} -> {end_label}", fh)
    log(f"  machine: {'INTERNAL (abstracted data only)' if internal else 'EXTERNAL (raw store)'} "
        f"(posts.parquet {'present' if os.path.exists(POSTS_PATH) else 'absent'})", fh)
    if full_chain:
        path_label = f"FULL rebuild over {BUILD_START_DATE} -> today"
    elif live:
        path_label = "LIVE fast (incremental fold + analytics)"
    else:
        path_label = "BACKTEST view (analytics only if stale)"
    log(f"  path   : {path_label}", fh)
    log(f"  fetch  : {'yes' if do_fetch else 'no (backtest or --skip-fetch)'}", fh)
    log("=" * 60, fh)

    # ---- 0. ENVIRONMENT PRE-FLIGHT: every package the run needs must live
    # in THIS interpreter (multiple installed Pythons is the classic cause
    # of intermittent failures). Checking everything up front gives one
    # clear message with one fix command. ----
    if not dry:
        needed = ["pandas", "pyarrow", "zstandard", "requests",
                  "vaderSentiment", "joblib",        # sentiment scoring
                  "wordfreq",                        # word-ticker screening
                  "plotly", "streamlit"]             # the dashboard
        missing = []
        for name in needed:
            try:
                __import__(name)
            except ImportError:
                missing.append(name)
        if missing:
            log(f"ABORT: this python ({py}) is missing: {', '.join(missing)}", fh)
            log(f"fix:  {py} -m pip install {' '.join(missing)} --user", fh)
            log("(or:  pip install -r requirements.txt --user  with the same python)", fh)
            return 1

    # The VIEW window travels to pull_bloomberg_prices.py (and any other
    # child process) through these env vars - one window, every script.
    os.environ["PIPELINE_START_DATE"] = args.start
    os.environ["PIPELINE_END_DATE"] = args.end

    # ---- 1. FETCH (raw only; the append steps below own the stores).
    #         The three fetchers run in PARALLEL inside fetch_all.py. ----
    if do_fetch:
        fetch_cmd = [py, "ingestion/fetch_all.py", "--no-merge",
                     "--lookback-days", str(FETCH_LOOKBACK_DAYS),
                     "--max-credits", str(FETCH_MAX_CREDITS)]
        if not args.with_comments:
            # comments are the slow species - the daily pipeline skips
            # them (desk decision 2026-07-24); update_comments.py is the
            # dedicated comments + influence runner
            fetch_cmd.append("--skip-comments")
        run(fetch_cmd, fh, dry, show=True)
    else:
        log("fetch skipped", fh)

    # ---- 1b. MONTHLY PANEL REVIEW (dynamic subreddit list). Watermarked
    # inside the script: costs one file-stat when not due, so it rides
    # every live run and actually fires ~monthly. --if-due exits quietly.
    if do_fetch and not args.skip_panel_review:
        run([py, "ingestion/discover_subreddits.py", "--if-due"],
            fh, dry, show=True)

    # ---- 2. APPEND into the right store (idempotent either way) ----
    if internal:
        log("folding live raw -> ABSTRACTED_DATA + hydrate", fh)
        run([py, "ingestion/append_live_abstracted.py"], fh, dry, show=True)
    else:
        if do_fetch or full_chain:
            log("merging live raw -> posts.parquet (close viewers first)", fh)
            run([py, "ingestion/merge_live.py"], fh, dry, show=True)
        else:
            # the merge streams the ENTIRE master (minutes) - pointless in a
            # backtest where nothing was fetched, so skip it
            log("backtest, nothing fetched - live merge skipped", fh)
        if live and not full_chain:
            # LIVE FAST PATH: recompute the last ~45 days of the aggregates
            # straight from posts.parquet and splice them onto the untouched
            # history. Same aggregation code as the full rebuild, minutes
            # not hours, always in sync with the raw store.
            #
            # Guard: if the aggregates end long ago, the tail splice would
            # leave a hole in the middle - a full rebuild is required first.
            if not dry:
                import pandas as pd
                agg_path = os.path.join(PROCESSED_DIR, "daily_ticker_counts.parquet")
                if os.path.exists(agg_path):
                    newest = pd.to_datetime(
                        pd.read_parquet(agg_path, columns=["date"])["date"]).max()
                    age = (pd.Timestamp.today() - newest).days
                    if age > 90:
                        log(f"ABORT: the aggregates end {newest.date()} ({age} days "
                            "ago). Run 'python update_data.py --full' once to "
                            "restore full history before live fast runs.", fh)
                        return 1
            log("live fast path: refreshing the aggregate tail from posts.parquet", fh)
            code = run([py, "ingestion/refresh_recent_aggregates.py"],
                       fh, dry, show=True)
            if code != 0:
                log("ABORT: aggregate tail refresh failed", fh)
                return 1

    # The internal machine mirrors the latest ABSTRACTED_DATA into
    # data/processed (covers a fresh git pull as well as a local append) so
    # the analytics never read stale aggregates.
    if internal and not dry:
        from src import abstracted_data
        abstracted_data.hydrate(verbose=False)
        log("hydrated ABSTRACTED_DATA -> data/processed", fh)

    # ---- 2b. DATA COVERAGE + WINDOW CHECK ----
    if not dry:
        print_data_coverage(fh, internal)
        check_window_coverage(fh, args.start, args.end)

    # ---- 3. COMPUTE - the notebook-free analytics.
    # live -> always recompute (new data just folded in); --full -> rebuild
    # the aggregates from raw text first, then recompute; backtest ->
    # recompute only when the aggregates are NEWER than the derived outputs
    # (they are derived locally and do not travel through git, so a fresh
    # pull needs one recompute on this machine).
    def signals_stale():
        agg = os.path.join(PROCESSED_DIR, "daily_ticker_counts.parquet")
        if not os.path.exists(agg):
            return False
        agg_mtime = os.path.getmtime(agg)
        derived = ["daily_ticker_conviction.parquet", "daily_theme_conviction.parquet",
                   "trade_signals.parquet"]
        for f in derived:
            p_ = os.path.join(PROCESSED_DIR, f)
            if not os.path.exists(p_) or os.path.getmtime(p_) < agg_mtime:
                return True
        return False

    if full_chain:
        # GUARD (same as RetailFlow1): months can be folded into
        # ABSTRACTED_DATA on the OTHER machine. A --full here rebuilds from
        # THIS machine's posts.parquet - if the committed aggregates run
        # ahead of the local master, the rebuild would silently REVERT
        # those months. Abort and explain instead.
        agg_p = os.path.join(ABSTRACTED_DIR, "daily_theme_counts.parquet")
        if (os.path.exists(POSTS_PATH) and os.path.exists(agg_p)
                and not os.environ.get("FORCE_FULL")):
            import pyarrow.parquet as _pq
            import pandas as _pd
            _pf = _pq.ParquetFile(POSTS_PATH)
            posts_max = None
            for _b in _pf.iter_batches(columns=["date"], batch_size=500_000):
                _mx = max(_b.column("date").to_pylist())
                posts_max = _mx if posts_max is None or _mx > posts_max else posts_max
            agg_max = _pd.to_datetime(
                _pd.read_parquet(agg_p, columns=["date"])["date"]).max()
            if agg_max - _pd.Timestamp(str(posts_max)[:10]) > _pd.Timedelta(days=14):
                log("ABORT: the committed aggregates reach "
                    f"{agg_max.date()} but this machine's posts.parquet only "
                    f"reaches {str(posts_max)[:10]}.", fh)
                log("A --full rebuild here would REVERT months folded on the "
                    "other machine. Fold/fetch the missing months into this "
                    "machine's master first, then rerun --full.", fh)
                log("Override (data loss!): set FORCE_FULL=1 in the environment.", fh)
                return 1
        log(f"full chain: building aggregates over {BUILD_START_DATE} -> today", fh)
        code = run([py, "ingestion/build_aggregates.py",
                    "--start", BUILD_START_DATE], fh, dry, show=True)
        if code != 0:
            log("ABORT: aggregate build failed", fh)
            return 1
        # rolling term frequencies (emerging-term detection); live folds
        # keep it current between fulls
        log("building rolling term counts (emerging-term detection)", fh)
        run([py, "ingestion/build_term_counts.py"], fh, dry, show=True)
        compute = True
    elif live:
        compute = True
        log("live: recomputing conviction + signals off the aggregates", fh)
    elif signals_stale():
        compute = True
        log("backtest: the aggregates are NEWER than the derived outputs "
            "(fresh pull?) - recomputing conviction + signals once", fh)
    else:
        compute = False
        log(f"backtest: aggregates + signals are up to date - the dashboard "
            f"renders {args.start} -> {end_label} directly", fh)

    if compute:
        analytics_cmd = [py, "-m", "analytics.run_analytics"]
        if args.full:
            # a full rebuild rewrites history - the frozen thresholds and
            # validation records must be re-derived (research decides
            # once, but a backfill IS a new research question)
            analytics_cmd.append("--research")
        code = run(analytics_cmd, fh, dry, show=True)
        if code != 0:
            log("ABORT: analytics failed - later steps skipped", fh)
            return 1

    # ---- 4. SNAPSHOT the signals (never revised) ----
    import shutil
    for fname in SIGNAL_FILES:
        src_path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(src_path):
            dest = os.path.join(SNAPSHOT_DIR, f"{today}_{fname}")
            if not dry and not os.path.exists(dest):
                shutil.copy2(src_path, dest)
            log(f"snapshot -> {dest}", fh)

    # ---- 4b. PRICES: pull Bloomberg closes for the window. Non-fatal:
    #          without a Terminal/blpapi the pull is skipped and the
    #          dashboard's price panels show their 'no prices' hint. ----
    if not dry and not args.skip_prices:
        log("pulling Bloomberg prices (Terminal must be open)", fh)
        run([py, "pull_bloomberg_prices.py"], fh, dry, show=True)
        if not os.path.exists(PRICES_PATH):
            log("no data/prices/prices.parquet - price overlays will be "
                "empty. Open the Bloomberg Terminal (and pip install "
                "blpapi), then re-run or use the dashboard button.", fh)

    # ---- 5. PUBLISH aggregates to ABSTRACTED_DATA (external machine, in
    #         live or --full runs; a backtest changes nothing to publish) ----
    if not internal and (live or full_chain) and not dry:
        from src import abstracted_data
        log("publishing aggregates -> ABSTRACTED_DATA", fh)
        abstracted_data.export(verbose=False)
    elif not internal and not live:
        log("backtest view: nothing rebuilt, nothing published", fh)

    # ---- 6. SAFETY CHECK the committed data ----
    safe = True
    if not dry:
        safe = verify_abstracted(fh)

    # ---- 7. RUN SUMMARY: the key facts in one glance ----
    if not dry:
        import pandas as pd
        log("", fh)
        log("=" * 60, fh)
        log("RUN SUMMARY", fh)
        log(f"  window        : {args.start} -> {end_label}", fh)
        log(f"  machine/path  : {'INTERNAL' if internal else 'EXTERNAL'} | {path_label}", fh)
        if not internal and os.path.exists(POSTS_PATH):
            import pyarrow.parquet as pq
            t = pq.read_table(POSTS_PATH, columns=["date", "source"]).to_pandas()
            log(f"  post store    : {len(t):,} posts total", fh)
            for src_name, grp in t.groupby("source"):
                log(f"    {src_name:<10} newest {grp['date'].max()}", fh)
        agg = os.path.join(PROCESSED_DIR, "daily_ticker_counts.parquet")
        if os.path.exists(agg):
            dates = pd.to_datetime(pd.read_parquet(agg, columns=["date"])["date"])
            log(f"  aggregates    : {dates.min().date()} -> {dates.max().date()}", fh)
        for fname, label in [("trade_signals.parquet", "theme signals"),
                             ("trade_signals_tickers.parquet", "ticker signals")]:
            path_ = os.path.join(PROCESSED_DIR, fname)
            if os.path.exists(path_):
                s = pd.read_parquet(path_)
                log(f"  {label:<13} : {len(s)} on file", fh)
        log(f"  prices        : {'present' if os.path.exists(PRICES_PATH) else 'MISSING (run pull_bloomberg_prices.py with the Terminal open)'}", fh)
        log(f"  safety check  : {'PASS' if safe else 'FAIL - do NOT commit ABSTRACTED_DATA'}", fh)
        log(f"  dashboard     : python -m streamlit run dashboard.py", fh)
        log("=" * 60, fh)
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
