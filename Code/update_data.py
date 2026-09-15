#!/usr/bin/env python
"""Pipeline entry point: refresh data, rescore, publish.

    python Code/update_data.py               # live refresh (the daily job)
    python Code/update_data.py --daily        # live refresh, prices from the theme anchors only
    python Code/update_data.py --prices-only  # the price step and nothing else
    python Code/update_data.py --full         # rebuild every aggregate from raw text
    python Code/update_data.py --ai           # refresh only the AI-generated panels

Modes
-----
The script detects which of two modes applies from what is on disk, and
``--mode`` overrides the detection:

``full``
    ``Data/processed/posts.parquet`` is present. This copy holds the raw
    post store: a live run merges newly fetched posts into it and splices
    the tail of every aggregate; ``--full`` rebuilds every aggregate over
    the whole build range from raw text.

``aggregates``
    No raw store. This copy holds only ``Data/abstracted`` (text-free
    daily aggregates). Newly fetched posts fold straight into the
    aggregates and raw text is never written to disk.

Either way the run ends by verifying that ``Data/abstracted`` carries no
text (the text-free boundary), and by publishing the dashboard bundle.

Windows
-------
``END_DATE`` in ``src/config.py`` (overridable per run) selects the view:

* ``END_DATE = ""`` is the live default: fetch the lookback window from
  every source in parallel, fold it in, rescore, pull prices (Bloomberg
  or Tiingo, per ``--provider`` / ``price_provider``).
* ``END_DATE = "YYYY-MM-DD"`` views a past regime. Aggregates are
  window-independent, so nothing is fetched or rebuilt unless a derived
  output is stale; the window drives the price pull and the dashboard.

What a run never does
---------------------
It never chooses a model, re-selects a threshold, or re-runs the
walk-forward tournament. It refreshes data and scores it with the frozen
model every time, so a run is one predictable job. The only exception is
a copy with no frozen record at all, which derives one once before it can
score (the bootstrap). To re-open the research question deliberately::

    cd Code && python -m src.analytics.run_analytics --what phases --research
    python Code/update_data.py --full     # a backfill rewrites the history
                                          # the thresholds were chosen on

If the frozen record stops at an earlier year than the data, the run
prints one notice and keeps scoring with it; that is out-of-sample use,
which the walk-forward validation licenses.

Every run prints
----------------
* a data-coverage table (posts per month, per source);
* a window check (whether the selected view has data, per source);
* a run summary;
* the text-free safety verdict.

Flags
-----
``--full``          rebuild the aggregates from raw text (``full`` mode)
``--ai``            refresh only the AI pulse and poll
``--daily``         price the theme anchor ETFs only; every other stage
                    runs as it normally does
``--prices-only``   run the price step and nothing else
``--fetch``         force API fetching in a backtest window
``--skip-fetch``    recompute only, no API calls
``--provider``      price source for this run: auto | bloomberg | tiingo
``--skip-prices``   skip the price pull
``--start/--end``   override the window for this run
``--mode``          ``full`` or ``aggregates`` instead of auto-detecting
``--dry-run``       print the plan, run nothing

Unattended operation
--------------------
A scheduled run has nobody watching it, so it judges itself on four
conditions: a stage or the run itself exited non-zero; the publish did
not reach the hosted app; a paid dependency ran out of credit or had its
key refused; or the run finished clean without the newest data day
advancing. The ALERT block at the end of the run names every condition
that fired, or says that none did.

Three of the four leave data that looks healthy from outside the
machine, so the verdict also travels: the publish step writes
``run_status.json`` into the dashboard bundle beside the publish
manifest, and ``tools/check_published_freshness.py`` reads it from a
checkout. ``tools/setup_schedule.py`` creates the scheduled task.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import traceback

try:                     # posts contain emoji/links; avoid cp1252 crashes
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import config                                        # noqa: E402
from src import pipeline_budget                               # noqa: E402
from src.config import (LOG_DIR, SNAPSHOT_DIR, PROCESSED_DIR,  # noqa: E402
                        ABSTRACTED_DIR, PRICES_PATH, POSTS_PATH, DATA_DIR,
                        REPORTS_DIR,
                        BUILD_START_DATE, FETCH_LOOKBACK_DAYS,
                        FETCH_MAX_CREDITS, FORBIDDEN_COLS, MAX_ABSTRACTED_MB,
                        PIPELINE_BUDGET_S)

SIGNAL_FILES = ["trade_signals.parquet", "trade_signals_tickers.parquet"]

# What one run leaves behind for the alert check at the end of it. The
# run log is the record a person reads, and these three are the same
# record in a shape the check can ask questions of: every line the run
# printed, every child command that exited non-zero, and the handful of
# verdicts the RUN SUMMARY prints. They are module-level because the
# alert is raised OUTSIDE main() - a stage that aborts returns from
# main() at the point of failure, and a wrapper around it sees every one
# of those exits through one return value.
RUN_LOG_LINES = []
STAGE_FAILURES = []
RUN_FACTS = {}


def _read_panel_subs():
    """The enabled forum panel from config/forums.csv, in file order.

    The same list the fetchers crawl, so cadence advice is computed over
    the panel that will actually be fetched.
    """
    from src import settings
    return list(settings.load_forums())


def log(msg, fh=None):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    RUN_LOG_LINES.append(line)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def _stream(cmd, fh):
    """Run ``cmd``, echoing each line to the terminal AND into the run log.

    A long stage has to show progress while it works, and an unattended
    run has to leave behind the reason it went the way it did. Those two
    need the same lines in two places: the terminal for the person
    sitting there, the log for the alert that is raised hours later and
    for whoever reads it afterwards. A provider's quota refusal is
    printed by the child, not by this script, so a log holding only this
    script's own lines cannot show why a run came up short.

    Returns the child's exit code.
    """
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        print(line, flush=True)
        RUN_LOG_LINES.append(line)
        if fh:
            fh.write(line + "\n")
            fh.flush()
    proc.stdout.close()
    return proc.wait()


def run(cmd, fh, dry, show=False, stage=None):
    """Run a child command. show=True streams its output live to the
    terminal and into the run log (long steps show progress); otherwise
    output is captured quietly. Returns the exit code (0 on --dry-run).

    stage=<name> also TIMES the step and folds the wall clock into
    Data/reference/pipeline_stage_times.json. That ledger is what makes the
    comment fetch budget honest: the pages the crawl may spend are the
    ceiling MINUS what THIS machine actually spends on everything else,
    rather than minus a number someone typed. Only successful runs are
    recorded - a stage that crashed after two seconds is not evidence that
    it takes two seconds."""
    log("RUN  " + " ".join(cmd), fh)
    if dry:
        return 0
    t0 = time.time()
    if show:
        rc = _stream(cmd, fh)
        if rc != 0:
            log(f"FAIL exit {rc} (the stage's own output is above)", fh)
    else:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        rc = r.returncode
        if rc != 0:
            log("FAIL " + (r.stderr or r.stdout)[-800:], fh)
    if rc != 0:
        # The per-stage exit codes an unattended run is judged on. A
        # stage that fails without stopping the run - the price pull is
        # the standing case - leaves no other trace once the summary has
        # scrolled past.
        STAGE_FAILURES.append((" ".join(cmd), rc))
    if stage and rc == 0:
        try:
            pipeline_budget.record_stage(stage, time.time() - t0)
        except OSError:
            pass          # a cost ledger is an optimisation, never a blocker
    return rc


# ---------------------------------------------------------------------------
# UNATTENDED OPERATION - the four conditions a run reports
# ---------------------------------------------------------------------------
# A scheduled run is read by nobody unless it asks to be. These are the
# states in which it asks, and nothing else is: a run that fetched,
# scored and published reports nothing.
#
# The judgement is made from what the run already records - the per-stage
# exit codes `run` collects, the publish string `_git_autopush` returns,
# and the lines the run printed - rather than from a second set of
# checks that could disagree with the summary a person reads.

# Where the previous run's newest data day is remembered. Under Reports/,
# which is machine-local and git-ignored: it describes THIS machine's
# last run, and travelling through git would make one machine's run
# history look like another's.
RUN_STATE_PATH = os.path.join(REPORTS_DIR, "run_state.json")

# The publish outcomes that are not a problem. `_git_autopush` returns a
# sentence; these two prefixes are the only ones that mean the hosted app
# has this run's data. "pushed and verified" is the commit cross-checked
# against the publish manifest, and "nothing new" is the honest case
# where the data did not change and there was nothing to push. Every
# other return - a refusal, a failed add, a commit that carries the
# previous run's bundle, a push that failed - means the hosted app is
# serving something older than this run.
PUBLISH_OK_PREFIXES = ("pushed and verified", "nothing new - data unchanged")

# How many days the newest data day may sit behind today before a clean
# run counts as stale. Two absorbs a weekend: the crowd posts every day,
# but a Saturday run after a quiet Friday night is not a fault. It also
# has to clear the case of two runs on one day, which necessarily read
# the same newest day as each other.
DATA_STALE_DAYS = 2

# Text that means a paid dependency is OUT, as opposed to unhappy. Each
# marker is matched case-insensitively against the run log.
#
# The distinction that matters here is exhaustion versus a transient
# error, because the two need different answers: exhaustion needs the
# owner to top up, raise a plan or issue a new key, and a transient
# error needs nothing at all - the next run recovers. So the markers
# below are the ones a service prints only when it has stopped serving
# this account:
#
# * Tiingo answers HTTP 429 when the key's request quota for the window
#   is spent; src/prices.py turns that one status into the first
#   sentence below, and an ordinary failed request into something else.
#   HTTP 401/403 is the same key refused outright.
# * FetchLayer bills one credit per request and answers HTTP 402 when
#   the balance is gone. Its HTTP 429 is pacing, which the fetchers back
#   off and retry, so 429 is deliberately NOT here - nor is "[stop] hit
#   the per-run credit cap", which is this pipeline's own budget doing
#   its job.
# * The AI layer has two ceilings. AI_MAX_CALLS is this process's own
#   cap, which stops a runaway loop before a provider does, and the
#   provider's own refusals name a quota, a credit balance or a key.
#   "LLM unavailable" is NOT here: a machine with no AI key configured
#   is a supported state, not an exhausted one, and alerting on it would
#   make every run of such a copy raise a condition.
EXHAUSTION_MARKERS = (
    ("request quota reached for this key",
     "Tiingo: the request quota for this key is spent (HTTP 429)"),
    ("tiingo: key rejected",
     "Tiingo: the key was refused (HTTP 401/403)"),
    ("fetchlayer says 402",
     "FetchLayer: out of credits (HTTP 402)"),
    ("ai_max_calls budget",
     "AI layer: this run's own call budget is spent (AI_MAX_CALLS)"),
    ("insufficient_quota",
     "AI layer: the provider reports no quota left on the key"),
    ("credit balance is too low",
     "AI layer: the provider reports no credit left on the account"),
    ("authenticationerror",
     "AI layer: the provider refused the key"),
    ("authentication_error",
     "AI layer: the provider refused the key"),
    ("invalid_api_key",
     "AI layer: the provider refused the key"),
    ("permissiondeniederror",
     "AI layer: the provider refused the key"),
)


def newest_data_day():
    """The newest day in the aggregates, as ``YYYY-MM-DD``, or ``""``.

    The same file and the same date the RUN SUMMARY prints as
    ``aggregates``, so the staleness verdict and the summary can never
    describe different data.
    """
    path = os.path.join(PROCESSED_DIR, "daily_ticker_counts.parquet")
    if not os.path.exists(path):
        return ""
    try:
        import pandas as pd
        dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
        return "" if dates.empty else str(dates.max().date())
    except Exception:                                        # noqa: BLE001
        return ""            # an unreadable store is the fold's problem


def read_run_state():
    """The previous run's newest data day and finish time, or ``{}``.

    An absent or unreadable state file means no previous run to compare
    against, which is silence rather than an alert: a first run on a
    machine has nothing to have advanced from.
    """
    try:
        with open(RUN_STATE_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def write_run_state(state):
    """Replace the run state with ``state``.

    Staged beside the target and swapped in, like every other write in
    this project: a run killed mid-write would otherwise leave a
    truncated file under the real name, and the next run would read it
    as "no previous run" and go quiet on a stale store.
    """
    tmp_path = RUN_STATE_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, RUN_STATE_PATH)
    except OSError:
        pass          # the state aids the next run; it never blocks this one


def _lines_matching(needles, limit=6):
    """Run-log lines containing any of ``needles`` (lower-cased), newest last."""
    hits = [ln for ln in RUN_LOG_LINES
            if any(n in ln.lower() for n in needles)]
    return hits[-limit:]


def alert_conditions(facts, prev_state):
    """Every condition this run raises.

    Args:
        facts: What the run recorded about itself (``RUN_FACTS``): the
            exit code, the publish string, the newest data day, the log
            path. A ``facts`` with no ``publish`` key leaves condition 2
            unjudged, which is how the publish step calls this - the
            commit and push happen after it.
        prev_state: The previous run's state, from :func:`read_run_state`.

    Returns:
        List of ``(tag, headline, detail_lines)``. ``tag`` is the short
        form the ALERT block and ``run_status.json`` carry; an empty list
        means the run is clean.
    """
    out = []

    # ---- 1. a stage errored, or the run itself did ----
    rc = facts.get("exit_code", 0)
    detail = [f"{cmd} -> exit {code}" for cmd, code in STAGE_FAILURES]
    aborts = [ln for ln in RUN_LOG_LINES if "ABORT:" in ln][-4:]
    if facts.get("traceback"):
        detail.append("the run raised an exception; the traceback is below")
    if rc != 0 or STAGE_FAILURES or facts.get("traceback"):
        head = (f"the run exited {rc}" if rc != 0
                else f"the run exited 0 but {len(STAGE_FAILURES)} stage(s) "
                     "failed")
        out.append(("run failed", head, detail + aborts))

    # ---- 2. the publish did not reach the hosted app ----
    # Only judged when the run got as far as attempting one: a
    # --prices-only or --dry-run never publishes, and a run that aborted
    # before the publish is already reported by condition 1.
    publish = facts.get("publish")
    if publish is not None and not publish.startswith(PUBLISH_OK_PREFIXES):
        out.append(("publish did not land",
                    "the refreshed data did not reach the hosted app",
                    [publish,
                     "the hosted dashboard redeploys from the repository, "
                     "so it is still serving the previous run's numbers"]))

    # ---- 3. a paid dependency is out of credit, or its key is refused ----
    seen = []
    for needle, meaning in EXHAUSTION_MARKERS:
        hits = _lines_matching((needle,), limit=2)
        if hits and meaning not in [m for m, _ in seen]:
            seen.append((meaning, hits))
    if seen:
        detail = []
        for meaning, hits in seen:
            detail.append(meaning)
            detail += [f"    {h}" for h in hits]
        out.append(("credit or key exhausted",
                    "a paid dependency stopped serving this account",
                    detail))

    # ---- 4. the run was clean and the data has fallen behind ----
    # The case that has no other symptom: every stage returns 0, the
    # summary reads normally, the bundle publishes, and the dashboard
    # serves a day that keeps getting older.
    #
    # The measure is the newest data day against TODAY, not against what
    # the previous run saw. Two runs on one day legitimately read the
    # same newest day, and so does a run made before that day's posts
    # land; neither is a fault, and a rule that fires on them would put
    # a false alarm in front of the owner often enough to be ignored.
    # DATA_STALE_DAYS absorbs a weekend, where the crowd is quiet but
    # nothing is broken.
    today_day = facts.get("newest_data_day") or ""
    if rc == 0 and today_day:
        try:
            behind = (datetime.date.today()
                      - datetime.date.fromisoformat(today_day)).days
        except ValueError:
            behind = None
        if behind is not None and behind > DATA_STALE_DAYS:
            since = prev_state.get("newest_data_day_since") or "an earlier run"
            out.append(("data stale",
                        f"the run finished clean and the newest data day is "
                        f"{today_day}, {behind} day(s) behind today",
                        [f"it has been {today_day} since {since}",
                         f"the previous run finished "
                         f"{prev_state.get('finished') or 'at an unrecorded time'}",
                         "nothing new is reaching the aggregates: check the "
                         "fetch stage above, and whether the sources "
                         "answered"]))
    return out


def run_status_record(exit_code):
    """What the published bundle carries about the run that built it.

    The run log stays on the machine; the bundle travels. Three of the
    four conditions leave data that looks perfectly healthy from a
    checkout - a stage that failed without stopping the run, a paid
    dependency that stopped serving, a store whose newest day did not
    move - so a watcher that measures only the published data day cannot
    see them. This record is how it does.

    Condition 2 is deliberately out of scope here. This is built by the
    publish step, which runs before the commit and the push, so whether
    the bundle reached the remote is not knowable yet; that question is
    answered from outside instead, by whether ``data_through`` advances
    between one day's bundle and the next.

    Args:
        exit_code: The code ``main`` is going to return.

    Returns:
        A JSON-ready dict: when the run finished, its exit code, the
        newest day in the aggregates, and one entry per condition.
    """
    facts = {"exit_code": exit_code, "newest_data_day": newest_data_day()}
    fired = alert_conditions(facts, read_run_state())
    return {"finished": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
            "exit_code": exit_code,
            "data_through": facts["newest_data_day"],
            "conditions": [{"tag": tag, "reason": headline}
                           for tag, headline, _detail in fired]}


def raise_run_alert(facts, prev_state, fh=None):
    """Judge the finished run and print the ALERT block.

    Never raises and never changes the run's exit code: the caller has
    already finished the work this block describes. The block is the
    last thing the run prints, and it names every condition that fired
    together with the log lines behind it, so a person reading the log
    top to bottom ends on the verdict.

    Returns:
        One line describing the outcome, already logged.
    """
    try:
        conditions = alert_conditions(facts, prev_state)
    except Exception as exc:                                 # noqa: BLE001
        # The whole point of this function is that a run which has
        # already done its work cannot be undone by the thing that
        # reports on it, so the report's own faults end here too.
        log(f"ALERT: the alert check itself failed - {type(exc).__name__}: "
            f"{exc} (the run's own result is unchanged)", fh)
        return "alert check failed"

    log("", fh)
    log("=" * 60, fh)
    log("ALERT", fh)
    log(f"  exit code     : {facts.get('exit_code', 0)}", fh)
    log(f"  newest data   : {facts.get('newest_data_day') or 'unknown'}", fh)
    log(f"  log file      : {facts.get('log_path') or 'not opened'}", fh)
    log(f"  publish       : {facts.get('publish') or 'not attempted'}", fh)
    if not conditions:
        log("  conditions    : none fired - the run is clean", fh)
        log("=" * 60, fh)
        return "nothing fired"
    log(f"  conditions    : {len(conditions)} fired", fh)
    for n, (tag, headline, detail) in enumerate(conditions, start=1):
        log(f"  {n}. {tag} - {headline}", fh)
        for line in detail:
            log(f"       {line}", fh)
    if facts.get("traceback"):
        log("  traceback", fh)
        for line in facts["traceback"].rstrip().splitlines():
            log(f"       {line}", fh)
    log("=" * 60, fh)
    return ", ".join(tag for tag, _h, _d in conditions)


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


def print_data_coverage(fh, aggregates_only):
    """A year x month table of data held, per source, so coverage gaps are
    visible at a glance after every run. In full mode this counts posts
    from posts.parquet; in aggregates mode there is no raw store, so it counts
    MENTIONS from the aggregates (same table shape, same gaps)."""
    import pandas as pd

    if not aggregates_only:
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
        if not aggregates_only:
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
        log("window check: no aggregates yet - hydrate Data/abstracted, or run "
            "'python Code/update_data.py --full' on a copy with the raw store", fh)
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
            "it, run 'python Code/update_data.py --full' on the copy with the raw store; "
            "otherwise widen the window.", fh)
    return any_data


# ---------------------------------------------------------------------------
# SAFETY CHECK - the committed aggregates must stay text-free
# ---------------------------------------------------------------------------
def verify_abstracted(fh):
    """Confirm Data/abstracted is present, non-empty, small, and carries no
    columns that could reveal raw posts. Reads only each file's footer, so
    the check is instant. Returns True if safe to commit."""
    import pyarrow.parquet as pq
    from src import abstracted_data

    log("safety check: Data/abstracted is text-free?", fh)
    all_present = all_safe = True
    for name in abstracted_data.FILES:
        path = os.path.join(ABSTRACTED_DIR, name)
        if not os.path.exists(path):
            log(f"  MISSING  {name}", fh)
            all_present = False
            all_safe = False
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        pf = pq.ParquetFile(path)
        cols = list(pf.schema_arrow.names)
        rows = pf.metadata.num_rows
        bad = [c for c in cols if c.lower() in FORBIDDEN_COLS]
        flag = ""
        if bad:
            flag = f"  <-- LEAK {bad}"
            all_safe = False
        if size_mb > MAX_ABSTRACTED_MB:
            flag += f"  <-- TOO BIG ({size_mb:.1f} MB)"
            all_safe = False
        # The row count comes from the same footer, so it costs nothing and
        # it catches the failure the column check cannot see: an empty file
        # carries no forbidden column either, and committing one replaces
        # the history the hosted dashboard draws with nothing.
        if rows == 0:
            flag += "  <-- EMPTY (0 rows)"
            all_safe = False
        log(f"  {name:<40} {size_mb:5.2f} MB | {rows:>9,} rows | {cols}{flag}", fh)

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
    p.add_argument("--provider", choices=("auto", "bloomberg", "tiingo"),
                   default=None,
                   help="price source for this run: auto (Bloomberg if "
                        "reachable, else Tiingo), bloomberg (falls back "
                        "to Tiingo on failure), or tiingo. Default: "
                        "price_provider in config/settings.csv")
    p.add_argument("--skip-prices", action="store_true",
                   help="skip the Bloomberg price pull")
    p.add_argument("--daily", action="store_true",
                   help="LIGHT PRICE STEP: pull closes for the theme "
                        "anchor ETFs only (about 27 symbols) instead of "
                        "the whole priced universe, extending each "
                        "stored series with this run's provider. It "
                        "changes the price step and nothing else, so it "
                        "combines with any other flag")
    p.add_argument("--prices-only", action="store_true",
                   help="PRICE STEP ONLY: pull closes and stop. Fetches "
                        "nothing, folds nothing, recomputes nothing and "
                        "publishes nothing - for exercising the price "
                        "step on its own. Combine with --daily for the "
                        "light universe and with --provider to pin the "
                        "source")
    p.add_argument("--mode", choices=("auto", "full", "aggregates"),
                   default="auto",
                   help="full = raw post store present; aggregates = "
                        "text-free aggregates only (default: detect)")
    p.add_argument("--skip-comments", action="store_true",
                   help="do NOT fetch Reddit comments this run (the "
                        "influence board then rescores the same data it "
                        "already had). Comments are ON by default in live "
                        "runs, because the influence board is only current "
                        "if the comments behind it are. They are the slow "
                        "species, so "
                        "their cost is BUDGETED against "
                        f"PIPELINE_BUDGET_S={PIPELINE_BUDGET_S}s rather "
                        "than switched off - see src/pipeline_budget.py")
    p.add_argument("--with-comments", action="store_true",
                   help=argparse.SUPPRESS)
    #   ^ accepted and ignored: comments ride every live run by default.
    #     Kept so the command lines printed in the RUNBOOK, the research
    #     report and any existing scheduled task keep working instead of
    #     dying on an unrecognised argument.
    p.add_argument("--ai", action="store_true",
                   help="AI LAYER ONLY: agentic scan, poll, pulse and "
                        "the keyword audit, then republish the "
                        "dashboard bundle. Fetches nothing, folds "
                        "nothing, recomputes nothing and pulls no "
                        "prices - for regenerating the AI panels "
                        "against data already on disk")
    p.add_argument("--skip-publish", action="store_true",
                   help="do not stage the hosted dashboard's display "
                        "bundle (Data/dashboard/) at the end of the run")
    p.add_argument("--skip-panel-review", action="store_true",
                   help="skip the monthly dynamic-panel review (subreddit "
                        "discovery; it is watermarked and only actually "
                        "runs when >= PANEL_REVIEW_DAYS have passed)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = p.parse_args()
    dry = args.dry_run
    py = sys.executable

    RUN_LOG_LINES.clear()
    STAGE_FAILURES.clear()
    RUN_FACTS.clear()
    RUN_FACTS["dry"] = dry

    # ---- mode: full (raw post store present) vs aggregates-only ----
    if args.mode == "full":
        aggregates_only = False
    elif args.mode == "aggregates":
        aggregates_only = True
    else:
        aggregates_only = not os.path.exists(POSTS_PATH)

    # ---- live vs backtest, fast vs full ----
    live = (args.end == "")
    # backtest covers the past, which does not change - no fetching by default
    do_fetch = (live and not args.skip_fetch) or (args.fetch and not args.skip_fetch)

    # AI-ONLY MODE. The AI layer reads the stores rather than building
    # them, so it is the one stage that is meaningful on its own: a
    # pulse that failed, or a prompt that changed, can be regenerated
    # without spending an API budget on a fetch or half an hour on a
    # recompute. Everything that WRITES a store is switched off here
    # rather than guarded stage by stage, so a stage added later
    # cannot quietly start running in this mode.
    ai_only = args.ai
    if ai_only:
        do_fetch = False
        args.skip_prices = True
        args.skip_panel_review = True

    # PRICES-ONLY MODE. The price pull is the long pole of a run and the
    # one stage whose cost is set by a vendor rather than by this
    # machine, so it needs to be runnable on its own - to time it, to
    # test a provider, to catch a series up after a failure. Everything
    # that WRITES a store is switched off here rather than guarded stage
    # by stage, so a stage added later cannot quietly start running in
    # this mode.
    prices_only = args.prices_only
    if prices_only:
        do_fetch = False
        args.skip_panel_review = True
        args.skip_publish = True
        args.full = False

    # Both single-stage modes leave every store but their own alone, so
    # the guards below ask this one question instead of naming each mode.
    single_stage = ai_only or prices_only
    full_chain = args.full and not aggregates_only
    price_scope = ("theme anchors only (--daily)" if args.daily
                   else "full priced universe")

    config.ensure_dirs()
    today = datetime.date.today().isoformat()
    fh = open(os.path.join(LOG_DIR, f"run_{today}.log"), "a", encoding="utf-8")
    RUN_FACTS["log_path"] = os.path.join(LOG_DIR, f"run_{today}.log")

    end_label = args.end if args.end else "LIVE (newest)"
    log("=" * 60, fh)
    log("UPDATE DATA (retailAPOLLO)", fh)
    log(f"  window : {args.start} -> {end_label}", fh)
    log(f"  mode: {'aggregates (text-free aggregates only)' if aggregates_only else 'full (raw post store)'} "
        f"(posts.parquet {'present' if os.path.exists(POSTS_PATH) else 'absent'})", fh)
    if full_chain:
        path_label = f"FULL rebuild over {BUILD_START_DATE} -> today"
    elif prices_only:
        path_label = "PRICES ONLY (no fetch, fold, recompute or publish)"
    elif live:
        path_label = ("AI LAYER ONLY (no fetch, fold, recompute or prices)"
                      if ai_only
                      else "LIVE fast (incremental fold + analytics)")
    else:
        path_label = "BACKTEST view (analytics only if stale)"
    log(f"  path   : {path_label}", fh)
    log(f"  fetch  : {'yes' if do_fetch else 'no (backtest or --skip-fetch)'}", fh)
    log(f"  prices : {'skipped' if args.skip_prices else price_scope}", fh)
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

    # The VIEW window travels to pull_prices.py (and any other
    # child process) through these env vars - one window, every script.
    os.environ["PIPELINE_START_DATE"] = args.start
    os.environ["PIPELINE_END_DATE"] = args.end

    # ---- 1. FETCH (raw only; the append steps below own the stores).
    #         The three fetchers run in PARALLEL inside fetch_all.py. ----
    if do_fetch:
        fetch_cmd = [py, "ingestion/fetch_all.py", "--no-merge",
                     "--lookback-days", str(FETCH_LOOKBACK_DAYS),
                     "--max-credits", str(FETCH_MAX_CREDITS)]
        if args.skip_comments:
            fetch_cmd.append("--skip-comments")
        else:
            # Comments ride every live run (see research.ipynb) so the
            # influence board is rescored on data that is actually new. They
            # are the slow species, so the crawl gets a PAGE ALLOWANCE: the
            # runtime ceiling minus what this machine measurably
            # spends on everything else, converted to pages at the API's
            # contracted request rate. The comment fetch runs in PARALLEL
            # with the post fetchers inside fetch_all.py, so those pages are
            # spent alongside the other sources, not after them.
            allowance = pipeline_budget.allowance_pages(
                skip_prices=args.skip_prices)
            fetch_cmd += ["--comment-pages", str(allowance)]
            other_s, measured = pipeline_budget.non_fetch_seconds()
            log(f"comment budget: {allowance} pages "
                f"({pipeline_budget.fmt_minutes(allowance / (config.COMMENT_RATE_PER_S or 1))}) "
                f"= ceiling {pipeline_budget.fmt_minutes(PIPELINE_BUDGET_S)} "
                f"- other stages {pipeline_budget.fmt_minutes(other_s)} "
                f"({'measured on this machine' if measured else 'bootstrap prior'})",
                fh)
        run(fetch_cmd, fh, dry, show=True, stage="fetch")
    else:
        log("fetch skipped", fh)

    # ---- 1b. MONTHLY PANEL REVIEW (dynamic subreddit list). Watermarked
    # inside the script: costs one file-stat when not due, so it rides
    # every live run and actually fires ~monthly. --if-due exits quietly.
    if do_fetch and not args.skip_panel_review:
        run([py, "ingestion/discover_subreddits.py", "--if-due"],
            fh, dry, show=True)

    # ---- 2. APPEND into the right store (idempotent either way) ----
    if ai_only:
        log("AI-only run: no fetch, no fold, no recompute, "
            "no prices - regenerating the AI layer against the "
            "stores already on disk", fh)
    elif prices_only:
        log("prices-only run: no fetch, no fold, no recompute, "
            "no publish - the price step against the stores already "
            "on disk", fh)
    elif aggregates_only:
        log("folding live raw -> Data/abstracted + hydrate", fh)
        # The fold's exit code must be checked: ignoring it would turn
        # a crashed fold into a green run (analytics recomputed on
        # unchanged aggregates, the bundle published, "safety check:
        # PASS", exit 0). Worse, a fold that dies after writing some of
        # the six aggregate files but before recording the seen-ids will
        # DOUBLE COUNT on the next run - so a silent failure here is
        # the one that corrupts the store. Stop the run on it: every
        # later step reads the aggregates this step was writing, and
        # the hydrate, the publish and the git push would carry a
        # half-written store out to the hosted dashboard.
        fold_rc = run([py, "ingestion/append_live_abstracted.py"], fh,
                      dry, show=True, stage="fold")
        if fold_rc:
            log("ABORT: FOLD FAILED - the aggregates may be partially "
                "written, so later steps are skipped and nothing is "
                "published or pushed. Do NOT re-run until the log above "
                "is read: a partial fold that did not record its "
                "seen-ids will double count on the next run.", fh)
            return 1
    else:
        if do_fetch or full_chain:
            log("merging live raw -> posts.parquet (close viewers first)", fh)
            fold_rc = run([py, "ingestion/merge_live.py"], fh, dry,
                          show=True, stage="fold")
            if fold_rc:
                log("ABORT: MERGE FAILED - posts.parquet may be "
                    "incomplete, so later steps are skipped and nothing "
                    "is published or pushed; read the log above before "
                    "re-running.", fh)
                return 1
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
                            "ago). Run 'python Code/update_data.py --full' once to "
                            "restore full history before live fast runs.", fh)
                        return 1
            log("live fast path: refreshing the aggregate tail from posts.parquet", fh)
            code = run([py, "ingestion/refresh_recent_aggregates.py"],
                       fh, dry, show=True, stage="fold")
            if code != 0:
                log("ABORT: aggregate tail refresh failed", fh)
                return 1

    # An aggregates-only copy mirrors the latest Data/abstracted into
    # Data/processed (covers a fresh git pull as well as a local append) so
    # the analytics never read stale aggregates.
    # A prices-only run leaves Data/processed as it found it: the price
    # universe reads it, and a day-old copy of it asks for the same
    # symbols.
    if aggregates_only and not dry and not prices_only:
        from src import abstracted_data
        _t = time.time()
        abstracted_data.hydrate(verbose=False)
        pipeline_budget.record_stage("hydrate", time.time() - _t)
        log("hydrated Data/abstracted -> Data/processed", fh)

    # ---- 2b. DATA COVERAGE + WINDOW CHECK ----
    if not dry and not prices_only:
        _t = time.time()
        print_data_coverage(fh, aggregates_only)
        check_window_coverage(fh, args.start, args.end)
        pipeline_budget.record_stage("coverage", time.time() - _t)

    # ---- 2c. PRICES: pull daily closes for the window through the
    #          configured provider (Bloomberg or Tiingo; see
    #          src/prices.py). Before the analytics, because the scores
    #          need the closes: the boom state and the two price features
    #          are read from the store, and a name is scored only up to
    #          its newest close. Non-fatal: on failure the run continues
    #          on the prices already on disk and the summary says so. ----
    prices_rc = None
    if not args.skip_prices:
        from src import settings as _settings
        _prov = args.provider or _settings.get("price_provider")
        price_cmd = [py, "ingestion/pull_prices.py", "--provider", _prov]
        if args.daily:
            price_cmd.append("--daily")
        log(f"pulling prices (provider: {_prov}, {price_scope})", fh)
        prices_rc = run(price_cmd, fh, dry, show=True, stage="prices")
        if not dry and prices_rc != 0:
            # NOT fatal, but it must not pass silently either: the rest
            # of the run is valid on the prices already on disk, and the
            # RUN SUMMARY says how stale they now are.
            log("PRICE PULL FAILED - continuing on the prices already "
                "on disk. With provider=bloomberg this usually means no "
                "Terminal is logged in AND the Tiingo fallback could "
                "not reach api.tiingo.com; with provider=tiingo it is a network "
                "or symbol-mapping problem (see the pull log above). "
                "Re-run with `--provider tiingo`, or "
                "`--skip-prices` to stop trying. Everything else in "
                "this run is unaffected.", fh)
        if not dry and not os.path.exists(PRICES_PATH):
            log("no Data/prices/prices.parquet - price overlays will be "
                "empty. Open the Bloomberg Terminal (and pip install "
                "blpapi), then re-run or use the dashboard button.", fh)

    # ---- 3. COMPUTE - the analytics.
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
        # GUARD: months can be folded into
        # Data/abstracted on the OTHER machine. A --full here rebuilds from
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
        if not single_stage:
            log("live: recomputing conviction + signals off the "
                "aggregates", fh)
    elif signals_stale():
        compute = True
        log("backtest: the aggregates are NEWER than the derived outputs "
            "(fresh pull?) - recomputing conviction + signals once", fh)
    else:
        compute = False
        log(f"backtest: aggregates + signals are up to date - the dashboard "
            f"renders {args.start} -> {end_label} directly", fh)

    if compute and not single_stage:
        analytics_cmd = [py, "-m", "src.analytics.run_analytics"]
        if args.full:
            # a full rebuild rewrites history - the frozen thresholds and
            # validation records must be re-derived (research decides
            # once, but a backfill IS a new research question)
            analytics_cmd.append("--research")
        code = run(analytics_cmd, fh, dry, show=True,
                   stage="analytics" if not args.full else None)
        #      ^ a --full run also re-derives thresholds (--research), which
        #        is a different and far larger job than a live recompute;
        #        recording it as "analytics" would poison the ledger with a
        #        cost the live path never pays.
        if code != 0:
            log("ABORT: analytics failed - later steps skipped", fh)
            return 1

    # ---- 4. SNAPSHOT the signals (never revised) ----
    import shutil
    for fname in ([] if single_stage else SIGNAL_FILES):
        src_path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(src_path):
            dest = os.path.join(SNAPSHOT_DIR, f"{today}_{fname}")
            if not dry and not os.path.exists(dest):
                # Copy beside the target and swap it in. A snapshot is
                # never revised, so the "already there" test above would
                # keep a half-copied parquet from an interrupted run for
                # good; only a whole file is ever allowed to appear under
                # the final name.
                shutil.copy2(src_path, dest + ".tmp")
                os.replace(dest + ".tmp", dest)
            log(f"snapshot -> {dest}", fh)

    # ---- 5. PUBLISH aggregates to Data/abstracted (full mode, in
    #         live or --full runs; a backtest changes nothing to publish) ----
    if not aggregates_only and (live or full_chain) and not dry and not single_stage:
        from src import abstracted_data
        log("publishing aggregates -> Data/abstracted", fh)
        abstracted_data.export(verbose=False)
    elif not aggregates_only and not live:
        log("backtest view: nothing rebuilt, nothing published", fh)

    # ---- 5b. AI LAYER - every update_data run also refreshes the AI
    #          pulse.  Two parts, both non-fatal by design:
    #          * the AGENTIC SCAN - pure python over any new raw
    #            archives (incremental via its ledger; seconds when
    #            nothing is new), no gateway needed;
    #          * the AI PULSE - the LLM's qualitative read, via the
    #            AI gateway (src/ai.py). Without a provider it skips with
    #            the reason logged and the dashboard keeps the last
    #            pulse; the pipeline NEVER fails on the AI stage.
    ai_poll_msg = ai_pulse_msg = "not run"
    if not dry and not prices_only:
        # CALL-BUDGET PRE-FLIGHT. The AI stages are non-fatal, so a
        # budget too small for the configured work fails on the LAST
        # call of the pulse - after every expensive call has been spent
        # and with nothing written. That happened for real: the pulse's
        # batch count is ceil(themes / THEMES_PER_CALL), so halving that
        # constant doubled the calls and silently pushed a run past a
        # 40-call cap. Check the arithmetic BEFORE spending anything.
        try:
            from src import ai as _ai_mod
            from src.analytics import ai_pulse as _ap_mod
            _n_themes = len(getattr(_ap_mod, "THEME_ETFS", {}) or {}) or 33
            _batches = max(1, -(-_n_themes // _ap_mod.THEMES_PER_CALL))
            _need = 30 + 1 + _batches + 1 + 1 + 1     # poll+pulse+audit
            if _ai_mod.MAX_CALLS < _need * 1.5:
                log(f"  NOTE: AI_MAX_CALLS={_ai_mod.MAX_CALLS} and this "
                    f"configuration needs about {_need} "
                    f"({_batches} theme batches). Raise it in .env to at "
                    f"least {int(_need * 1.5)} so one retry cannot "
                    "exhaust the budget on the pulse's last call.", fh)
        except Exception:                                # noqa: BLE001
            pass                                          # advisory only

        try:
            from src.agentic_watch import scan as _agentic_scan
            _agentic_scan(log=lambda m: log(m, fh))
        except Exception as e:                           # noqa: BLE001
            log(f"  agentic scan skipped: {type(e).__name__}: {e}", fh)
        try:
            from src.analytics.ai_poll import run as _run_poll
            _ok, _msg = _run_poll(log=lambda m: log(m, fh))
            ai_poll_msg = "ok" if _ok else f"FAILED - {_msg}"
            if not _ok:
                log(f"AI POLL: skipped - {_msg}", fh)
        except Exception as e:                           # noqa: BLE001
            ai_poll_msg = f"FAILED - {type(e).__name__}: {e}"
            log(f"AI POLL: skipped - {type(e).__name__}: {e}", fh)
        try:
            from src.analytics.ai_pulse import generate as _gen_pulse
            _ok, _msg = _gen_pulse(log=lambda m: log(m, fh))
            ai_pulse_msg = "ok" if _ok else f"FAILED - {_msg}"
            if not _ok:
                log(f"AI PULSE: skipped - {_msg}", fh)
        except Exception as e:                           # noqa: BLE001
            ai_pulse_msg = f"FAILED - {type(e).__name__}: {e}"
            log(f"AI PULSE: skipped - {type(e).__name__}: {e}", fh)
        # the keyword-map auditor, WEEKLY: if the newest suggestions
        # file is older than 7 days and the gateway is up, a fresh audit
        # is written for review. NEVER auto-applied
        # - apply is a human step (tools/ai_keyword_audit.py --apply).
        try:
            import glob as _glob
            _sugg = sorted(_glob.glob(os.path.join(
                DATA_DIR, "reference", "keyword_suggestions",
                "keyword_suggestions_*.csv")))
            _age_ok = True
            if _sugg:
                _age_ok = (time.time() - os.path.getmtime(_sugg[-1])
                           > 7 * 86400)
            if _age_ok:
                from src import ai as _ai
                if _ai.available() and not _ai.MOCK:
                    from tools.ai_keyword_audit import audit as _audit
                    _p = _audit()
                    log(f"KEYWORD AUDIT: suggestions -> {_p} "
                        "(review + approve, then --apply)", fh)
        except SystemExit:
            pass
        except Exception as e:                           # noqa: BLE001
            log(f"KEYWORD AUDIT: skipped - {type(e).__name__}: {e}", fh)

    # ---- 6. SAFETY CHECK the committed data ----
    safe = True
    if not dry:
        safe = verify_abstracted(fh)

    # ---- 6b. PUBLISH the hosted dashboard's display bundle ----
    # Deliberately AFTER the safety check and gated on it: the bundle is
    # committed and served publicly, so it must never be staged from a
    # tree that just failed the text-free rule. Non-fatal like the AI
    # stage - a bundle problem must not fail a data refresh that already
    # succeeded. tools/publish_dashboard.py runs its own per-file guard
    # on top of this one.
    #
    # The run status rides along with the bundle. It is judged here, at
    # the last moment before the bundle is written, so that everything
    # the run has logged is already in evidence; `safe` is the only
    # thing left that decides the exit code, which is why the code can
    # be stated this early. Building it is wrapped: a verdict that
    # cannot be reached costs the status file and nothing else.
    if not dry and not args.skip_publish:
        if not safe:
            log("dashboard bundle NOT published - the text-free check "
                "failed; resolve that first", fh)
        else:
            try:
                _status = run_status_record(0 if safe else 1)
            except Exception as e:             # noqa: BLE001
                _status = None
                log(f"run status skipped: {type(e).__name__}: {e}", fh)
            try:
                from tools.publish_dashboard import publish as _publish
                _publish(log=lambda m: log(m, fh), per_file=False,
                         run_status=_status)
            except SystemExit as e:            # the guard refused a file
                log(f"dashboard bundle NOT published - {e}", fh)
            except Exception as e:             # noqa: BLE001
                log(f"dashboard bundle skipped: {type(e).__name__}: {e}", fh)

    # ---- 6c. GIT AUTO-PUBLISH: commit + push the refreshed data ----
    # A hosted dashboard redeploys from the repository, so a refresh
    # that stops short of a push never reaches it. DATA PATHS ONLY
    # (Data/dashboard + Data/abstracted): code edits are never swept
    # into an auto-commit. Refuses to act when other files are already
    # staged - that is a commit in progress. Never fatal; the run
    # summary reports what happened.
    def _git_autopush():
        import json as _json
        import subprocess as _sp
        _env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        # Only the paths that exist: `git add` rejects the whole pathspec
        # when one entry matches nothing, so a copy that has never staged
        # a bundle would otherwise fail to publish its aggregates either.
        _dirs = [(n, os.path.join(DATA_DIR, n))
                 for n in ("dashboard", "abstracted")
                 if os.path.exists(os.path.join(DATA_DIR, n))]
        if not _dirs:
            return "skipped - neither Data/dashboard nor Data/abstracted exists"

        def _git(*a, timeout=120):
            # git reports paths as UTF-8; decoding them with the console's
            # locale encoding instead turns a non-ASCII filename in the
            # status output into a decode error on a cp1252 machine, which
            # would abort the commit step over a name it only had to read.
            return _sp.run(["git", *a], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=_env)
        try:
            r = _git("rev-parse", "--is-inside-work-tree")
        except FileNotFoundError:
            return "skipped - git is not installed on this machine"
        if r.returncode != 0:
            return "skipped - not a git repo"
        # Two states where a commit would touch more than the data paths,
        # so the auto-publish keeps its hands off both. On a detached HEAD
        # the commit lands on no branch and the next checkout loses it.
        # Mid-merge (or rebase, cherry-pick, revert) `git commit`
        # CONCLUDES that operation and sweeps in every path it has
        # already staged - the owner's work, committed under this
        # script's message.
        if _git("symbolic-ref", "--quiet", "HEAD").returncode != 0:
            return ("skipped - HEAD is detached; check out a branch and "
                    "commit the data yourself")
        r = _git("rev-parse", "--absolute-git-dir")
        _gitdir = r.stdout.strip() if r.returncode == 0 else ""
        _busy = [n for n in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD",
                             "rebase-merge", "rebase-apply")
                 if _gitdir and os.path.exists(os.path.join(_gitdir, n))]
        if _busy:
            return (f"skipped - {_busy[0]} is present, so a merge or "
                    "rebase is in progress; finish it and re-run")
        r = _git("diff", "--cached", "--name-only")
        if r.returncode == 0 and r.stdout.strip():
            return ("skipped - you already have files staged; finish "
                    "that commit (or unstage) and re-run")

        # THE SPELLING THE INDEX USES. A checkout taken on a
        # case-insensitive filesystem can hold the data folder as
        # "data/..." while the same folder is reachable on disk as
        # "Data/...". git matches a pathspec against index entries
        # case-sensitively, so a pathspec built from the on-disk spelling
        # names no tracked file: `git add` succeeds, stages none of the
        # bundle, and the publish is silent. The spelling is read out of
        # git rather than taken from disk for that reason.
        r = _git("rev-parse", "--show-toplevel")
        _top = r.stdout.strip()
        if r.returncode != 0 or not _top:
            return "skipped - the repository root could not be read"
        # `--full-name -- :/` because this runs from Code/ and `git
        # ls-files` otherwise answers for the current directory only,
        # which would report the data folder as untracked.
        r = _git("ls-files", "-z", "--full-name", "--", ":/")
        _tracked = [n for n in r.stdout.split("\0") if n] \
            if r.returncode == 0 else []

        def _rel(path):
            """`path` as git spells paths: relative to the repository root."""
            return os.path.relpath(path, _top).replace(os.sep, "/")

        def _indexed(rel):
            """How the index spells directory `rel`.

            None when the index holds nothing under it - a fresh clone, a
            folder never committed - where the on-disk spelling is the
            right one and the files are new.
            """
            want = rel.lower() + "/"
            for name in _tracked:
                if name.lower().startswith(want):
                    return name[:len(rel)]
            return None

        _spec, _differs = {}, []
        for _name, _path in _dirs:
            _on_disk = _rel(_path)
            _in_index = _indexed(_on_disk)
            _spec[_name] = _in_index or _on_disk
            if _in_index and _in_index != _on_disk:
                _differs.append(f"{_on_disk} is indexed as {_in_index}")
        # A repository state the owner has to know about: every git
        # command they type against the on-disk spelling misses the same
        # files this one would have missed.
        _note = ("  [" + "; ".join(_differs) + "]") if _differs else ""

        def _say(msg):
            """The run summary prints one line, so the spelling disagreement
            rides along with the outcome instead of living somewhere else."""
            return msg + _note

        # What this run has to publish, measured with NO pathspec. A
        # pathspec is the thing under suspicion here, so the answer has to
        # come from somewhere a pathspec cannot filter.
        r = _git("status", "--porcelain", "-z", "--untracked-files=all")
        if r.returncode != 0:
            return _say("status failed - "
                        f"{(r.stderr or r.stdout).strip()[:160]}")
        _roots = tuple(_rel(p).lower() + "/" for _n, p in _dirs)
        _pending = {e[3:] for e in r.stdout.split("\0")
                    if len(e) > 3 and e[3:].lower().startswith(_roots)}
        if not _pending:
            return _say("nothing new - data unchanged since the last commit")

        # Anchored at the repository root with `:/`, because git resolves a
        # pathspec against the working directory and this runs from Code/.
        r = _git("add", "--", *(":/" + s for s in _spec.values()))
        if r.returncode != 0:
            return _say(f"add failed - {(r.stderr or r.stdout).strip()[:160]}")
        r = _git("diff", "--cached", "--name-only", "-z")
        _staged = {n.lower() for n in r.stdout.split("\0") if n}
        # A NON-EMPTY index is not evidence of a publish. What has to hold
        # is that everything the data paths changed went in - the manifest
        # and the aggregates, not whichever files the add happened to
        # catch - because a commit carrying a subset still pushes, still
        # redeploys, and still serves the previous run's numbers.
        _missed = sorted(n for n in _pending if n.lower() not in _staged)
        if not _staged:
            return _say(
                f"NOT published - {len(_pending)} changed file(s) under the "
                "data paths and the index took none of them; the likeliest "
                "cause is the pathspec and the index disagreeing on how the "
                "data folder is spelled. The bundle is uncommitted and "
                "nothing is pushed")
        if _missed:
            return _say(
                f"NOT published - {len(_missed)} of {len(_pending)} changed "
                f"file(s) under the data paths stayed out of the index, "
                f"starting with {_missed[0]}; the likeliest cause is the "
                "pathspec and the index disagreeing on how the data folder "
                "is spelled. The bundle is uncommitted and nothing is pushed")
        _cm = time.strftime("data refresh %Y-%m-%d %H:%M (auto-publish)")
        r = _git("commit", "-m", _cm)
        if r.returncode != 0:
            return _say(f"commit failed - "
                        f"{(r.stderr or r.stdout).strip()[:160]}")

        # WHAT WENT IN AGAINST WHAT WAS JUST PUBLISHED. The publish
        # manifest names the newest date in the aggregates the bundle
        # carries, so reading it back out of the commit is the one check
        # that separates a commit carrying THIS run's bundle from one
        # carrying the last run's. From the outside the two are
        # indistinguishable: both push, both redeploy, both look current.
        def _through(text):
            """The `data_through` date a manifest carries, or None."""
            try:
                return _json.loads(text).get("data_through")
            except Exception:                            # noqa: BLE001
                return None

        _published = None
        if "dashboard" in _spec:
            _mf = os.path.join(DATA_DIR, "dashboard", "publish_manifest.json")
            if os.path.exists(_mf):
                try:
                    with open(_mf, encoding="utf-8") as _f:
                        _published = _through(_f.read())
                except OSError:
                    _published = None
        if _published is not None:
            r = _git("show",
                     f"HEAD:{_spec['dashboard']}/publish_manifest.json")
            _committed = _through(r.stdout) if r.returncode == 0 else None
            if _committed != _published:
                return _say(
                    "NOT published - the commit carries data through "
                    f"{_committed or 'no manifest'} while this run published "
                    f"data through {_published}; the commit is local and "
                    "nothing is pushed")

        r = _git("push", timeout=300)
        if r.returncode != 0:
            return _say("committed locally but push FAILED - "
                        f"{(r.stderr or r.stdout).strip()[:160]} "
                        "- run `git push` yourself; the commit is made")
        if _published is None:
            return _say("pushed - the commit carries the staged data paths; "
                        "this copy has no publish manifest to cross-check it "
                        "against; the hosted dashboard redeploys in ~1-2 min")
        return _say(f"pushed and verified - the commit carries this run's "
                    f"bundle, data through {_published}; the hosted "
                    "dashboard redeploys in ~1-2 min")

    git_msg = "not run"
    if not dry and not prices_only:
        if not safe:
            git_msg = "skipped - the safety check failed"
        else:
            try:
                git_msg = _git_autopush()
            except Exception as e:                       # noqa: BLE001
                git_msg = f"FAILED - {type(e).__name__}: {e}"
        log(f"GIT PUBLISH: {git_msg}", fh)
        # The publish string is the contract the alert check reads: it is
        # recorded only when a publish was actually attempted, so a mode
        # that never publishes is not reported as one that failed to.
        RUN_FACTS["publish"] = git_msg

    # ---- 7. RUN SUMMARY: the key facts in one glance ----
    if not dry:
        import pandas as pd
        log("", fh)
        log("=" * 60, fh)
        log("RUN SUMMARY", fh)
        log(f"  window        : {args.start} -> {end_label}", fh)
        log(f"  mode/path     : {'aggregates' if aggregates_only else 'full'} | {path_label}", fh)
        if not aggregates_only and os.path.exists(POSTS_PATH):
            import pyarrow.parquet as pq
            t = pq.read_table(POSTS_PATH, columns=["date", "source"]).to_pandas()
            log(f"  post store    : {len(t):,} posts total", fh)
            for src_name, grp in t.groupby("source"):
                log(f"    {src_name:<10} newest {grp['date'].max()}", fh)
        agg = os.path.join(PROCESSED_DIR, "daily_ticker_counts.parquet")
        if os.path.exists(agg):
            dates = pd.to_datetime(pd.read_parquet(agg, columns=["date"])["date"])
            log(f"  aggregates    : {dates.min().date()} -> {dates.max().date()}", fh)
            RUN_FACTS["newest_data_day"] = str(dates.max().date())
        for fname, label in [("trade_signals.parquet", "theme signals"),
                             ("trade_signals_tickers.parquet", "ticker signals")]:
            path_ = os.path.join(PROCESSED_DIR, fname)
            if os.path.exists(path_):
                s = pd.read_parquet(path_)
                log(f"  {label:<13} : {len(s)} on file", fh)
        # prices: say how FRESH, not merely whether the file exists -
        # a stale store after a failed pull is the case that matters
        if os.path.exists(PRICES_PATH):
            try:
                import pandas as pd          # local: keeps startup fast
                _pxd = pd.read_parquet(PRICES_PATH, columns=["date"])
                _newest = pd.to_datetime(_pxd["date"]).max()
                _lag = (pd.Timestamp.now().normalize()
                        - _newest.normalize()).days
                _fresh = (f"newest close {_newest:%Y-%m-%d} "
                          f"({_lag}d old)")
            except Exception:                            # noqa: BLE001
                _fresh = "present (could not read the newest date)"
            if prices_rc not in (None, 0):
                _msg = (f"STALE - PULL FAILED THIS RUN ({price_scope}) "
                        f"| {_fresh}")
            elif prices_rc == 0:
                _msg = f"updated, {price_scope} | {_fresh}"
            else:
                _msg = f"not pulled this run | {_fresh}"
        else:
            _msg = "MISSING (run ingestion/pull_prices.py)"
        log(f"  prices        : {_msg}", fh)
        # ---- INFLUENCE BOARD + the cadence this run's own timings imply ----
        infl = os.path.join(config.REFERENCE_DIR, "influence", "author_scores.parquet")
        if os.path.exists(infl):
            a = pd.read_parquet(infl, columns=["author"])
            age_h = (time.time() - os.path.getmtime(infl)) / 3600.0
            log(f"  influence     : {len(a):,} authors scored, rebuilt "
                f"{age_h:.1f}h ago"
                f"{' (comments SKIPPED this run - same data rescored)' if args.skip_comments else ''}",
                fh)
        if do_fetch and not args.skip_comments and not dry:
            # This is the number the signal acts on, and it is entirely
            # measured: the fetch budget this machine has left, divided by
            # the panel's observed comment volume. Run at least this often
            # and no comments are ever deferred.
            try:
                subs = _read_panel_subs()
                cadence = pipeline_budget.derived_cadence_days(
                    subs, skip_prices=args.skip_prices)
                log(f"  run cadence   : every {cadence:.1f} days or less keeps "
                    f"the influence board fully current "
                    f"(~{7 / max(cadence, 0.1):.1f}x/week; measured, not "
                    "assumed - src/pipeline_budget.py)", fh)
            except OSError:
                pass
        log(f"  safety check  : {'PASS' if safe else 'FAIL - do NOT commit Data/abstracted'}", fh)
        # The AI stages are non-fatal, which is right - a dead gateway
        # must not stop a data refresh - so a failed pulse would leave no
        # trace while the dashboard quietly served a months-old file.
        # State the verdict where the run is read.
        log(f"  AI poll       : {ai_poll_msg[:78]}", fh)
        log(f"  AI pulse      : {ai_pulse_msg[:78]}", fh)
        log(f"  git publish   : {git_msg[:78]}", fh)
        log("  dashboard     : python -m streamlit run Code/dashboard.py", fh)
        log("=" * 60, fh)
    # A failed text-free check is a failed run, so a scheduler or wrapper
    # never sees a store that must not be committed as success. The fold
    # and the analytics abort earlier, with their own nonzero exit.
    return 0 if safe else 1


def run_with_alerts():
    """Run the pipeline, then raise the one alert an unattended run needs.

    A wrapper rather than a step inside :func:`main`, because a stage
    that aborts the run returns from ``main`` at the point of failure -
    the fold, the merge, the aggregate build and the analytics each do -
    and a wrapper sees every one of those exits through one return
    value. It returns exactly what ``main`` returned: nothing the ALERT
    block does may change the run's verdict.

    A ``--dry-run`` is excluded. It refreshes nothing, so it has no
    result to judge, and it must stay the one command that changes
    nothing at all.
    """
    argv = sys.argv[1:]
    quiet = "--dry-run" in argv
    prev_state = {} if quiet else read_run_state()
    started = datetime.datetime.now()
    try:
        rc = main()
    except Exception:                                        # noqa: BLE001
        # An unhandled exception is the failure most worth reporting, so
        # it is caught and described rather than allowed to end the
        # process before the ALERT block runs. The exit code is the 1 an
        # uncaught traceback would have produced, and the traceback
        # itself goes to the console and the run log.
        rc = 1
        # Through the redactor before it is printed or logged. A
        # traceback carries the arguments of every frame on the way
        # down, and this one is written to a file.
        from src import prices as _prices
        RUN_FACTS["traceback"] = _prices.redact(traceback.format_exc())
        log(f"RUN FAILED: {RUN_FACTS['traceback'].rstrip()}")
    if quiet:
        return rc

    finished = datetime.datetime.now()
    RUN_FACTS["exit_code"] = rc
    RUN_FACTS["finished"] = f"{finished:%Y-%m-%d %H:%M}"
    RUN_FACTS.setdefault("newest_data_day", newest_data_day())
    # The verdict belongs in the run log next to the run it judges, so
    # that reading the log answers "what did this run raise" without a
    # second file. A log that cannot be opened - a full disk, a folder
    # gone - costs the log lines and nothing else.
    today = datetime.date.today().isoformat()
    try:
        with open(os.path.join(LOG_DIR, f"run_{today}.log"), "a",
                  encoding="utf-8") as fh:
            raise_run_alert(RUN_FACTS, prev_state, fh)
    except Exception as exc:                                 # noqa: BLE001
        # Deliberately every exception, and deliberately one line: the
        # run is finished and its verdict is already decided, so the
        # worst an unreachable log or an unexpected fault in the alert
        # path may cost is this message.
        print(f"ALERT: not raised - {type(exc).__name__}: {exc}", flush=True)

    # The state the NEXT run compares against. `newest_data_day_since`
    # keeps the finish time of the run that first saw this day, so a
    # store that has been stuck for a week says a week rather than
    # repeating "since the last run" every day.
    day = RUN_FACTS.get("newest_data_day") or ""
    since = (prev_state.get("newest_data_day_since")
             if day and day == (prev_state.get("newest_data_day") or "")
             else RUN_FACTS["finished"])
    write_run_state({"newest_data_day": day,
                     "newest_data_day_since": since or RUN_FACTS["finished"],
                     "finished": RUN_FACTS["finished"],
                     "exit_code": rc,
                     "started": f"{started:%Y-%m-%d %H:%M}"})
    return rc          # whatever main() decided, unchanged by any of this


if __name__ == "__main__":
    raise SystemExit(run_with_alerts())
