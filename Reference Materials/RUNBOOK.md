# Runbook

Operating instructions for retailAPOLLO. All commands run from the
project root (the folder that holds `Code/`, `Data/` and this file's
folder); the few `python -m src...` commands run from inside `Code/`. Routine operation is one command; the rest of this page
covers setup, publishing, maintenance and recovery. For how the method
works and why each parameter has its value, see `research.ipynb`.

Layout:

| Folder | Contents |
|---|---|
| `Code/` | Everything runnable: `dashboard.py`, `update_data.py`, `src/` (shared code and the model in `src/analytics/`), `ingestion/` (fetchers, bot screen, fold, price pull), `tools/` (operations), `tests/`, `config/` (everything a user edits), `.env`. |
| `Data/` | `abstracted/` (committed text-free aggregates), `dashboard/` (the published display bundle), `research_record/` (the frozen research evidence), and the git-ignored runtime stores `raw/`, `processed/`, `prices/`, `reference/`. |
| `Reports/` | One log per run in `logs/`, plus the bot-screen summary and the forum-panel review. Machine-local, git-ignored. |
| `Reference Materials/` | This runbook; `research.ipynb`, the write-up of how the method works; and `ice_source_review.ipynb`, the completed assessment of a vendor Reddit source. `archive/` (git-ignored) holds long-form notes and older research notebooks. |
| `Presentations/` | Decks and demo material. Nothing runnable reads from it. |

The repository root also carries `README.md`, `requirements.txt` and
`.streamlit/` (the dashboard theme; Streamlit reads it from the folder
the server is started in, which is why the dashboard is started from
the root).

## 1. Setup

```bash
pip install -r requirements.txt
cp Code/example.env Code/.env                     # optional keys; comments in example.env
python Code/tools/preflight.py
```

`Code/tools/preflight.py` runs eight checks and prints one line per
finding: every config CSV validates (the full rule set in
`Code/tools/validate_config.py`); the four core aggregate stores are
present; every tradeable theme has a priced line somewhere in its anchor
chain; no theme is quietly drawn on a fallback instead of its named
anchor; every mapped ticker still appears in the listed symbol directory;
the frozen detector constants match the stored research record; no
committed aggregate carries post text or an author; and no file cited in
the code is missing (`Code/tools/verify_deps.py`). It writes nothing, so
it is safe to run at any time. Its one network call refreshes the cached
symbol directory; a machine that cannot reach the listing host warns and
reads the cached copy. A clean preflight is the definition of "this copy
is ready to run".

Python 3.11 or 3.12 is supported. `blpapi` (Bloomberg) installs from
Bloomberg's package index and is optional: without it, prices come from
Tiingo, which needs `TIINGO_API_KEY` in `.env` (a free key from tiingo.com).

## 2. Routine operation

| Task | Command |
|---|---|
| Refresh everything | `python Code/update_data.py` |
| Refresh with a light price step | `python Code/update_data.py --daily` (section 2.4) |
| Run the price step on its own | `python Code/update_data.py --prices-only` |
| Open the dashboard | `python -m streamlit run dashboard.py` (the root file runs `Code/dashboard.py`) |
| Recompute without any network call | `python Code/update_data.py --skip-fetch --skip-prices` |
| Analytics only (seconds) | `python Code/src/analytics/run_analytics.py` |
| Read-only health report | `python Code/tools/data_health.py` |
| Full soundness check | `python Code/tools/preflight.py` |
| Dangling reference / dependency check | `python Code/tools/verify_deps.py` |
| Bot-screen report on the raw store | `python Code/tools/bot_screen_report.py` |
| Refresh prices only | `python Code/ingestion/pull_prices.py` (`--dry-run` to preview) |
| Diagnose the price step | `python Code/ingestion/pull_prices.py --check` (section 2.5) |
| What live data landed | `python Code/ingestion/check_live_ingestion.py` |
| Test suite | `python -m pytest Code/tests -q` (`pip install -r Code/tools/requirements-dev.txt` once) |
| Create the daily scheduled task | `python Code/tools/setup_schedule.py` (section 9.1; `--register` creates it) |
| Check the published bundle from a checkout | `python Code/tools/check_published_freshness.py` (section 9.2) |

One refresh does, in order: fetch (all sources in parallel), bot screen,
fold into the aggregates, coverage check, price pull, analytics at the
frozen thresholds (a name is scored up to its newest close, so the
scores run one trading day behind the posts), AI layer (if configured),
text-free safety check, publish bundle, summary. `--dry-run` prints the plan without
running anything.

Run cadence: twice a week is the design point. The comment fetch is
budgeted so that at that cadence nothing is deferred; after a longer gap
the oldest pages are deferred to the next run (deferral, never loss —
a capped forum keeps its watermark).

### 2.1 Run modes

`update_data.py --mode` selects how posts are stored:

| Mode | When | What happens |
|---|---|---|
| `full` | `Data/processed/posts.parquet` exists | New posts are appended to the raw store and every aggregate can be rebuilt from text. |
| `aggregates` | No raw store | New posts are screened, aggregated and folded into `Data/abstracted/` as text-free deltas. The raw files stay in `Data/raw/`. |
| `auto` (default) | — | Detects which of the above applies. |

Both modes produce the same aggregate tables and the same dashboard. A
fresh clone is an aggregates-mode copy.

### 2.2 Price provider

`Code/config/settings.csv → price_provider` sets the default; `--provider`
overrides it for one run, on `update_data.py` and on
`ingestion/pull_prices.py`.

| Value | Behaviour |
|---|---|
| `auto` (default) | Bloomberg if a Terminal answers, otherwise Tiingo. |
| `bloomberg` | Bloomberg; falls back to Tiingo if the Terminal fails mid-run unless `--no-fallback` is given. |
| `tiingo` | Tiingo only: an authenticated daily-price API, `TIINGO_API_KEY` in `.env` (free key at tiingo.com), one request per symbol. Split-adjusted closes, same convention as `PX_LAST`. |

A Terminal that fails repeatedly is dropped for the rest of a run:
`bloomberg_max_failures` in `Code/config/settings.csv` (default 2) is how
many failed attempts are tolerated first. Each attempt can block for
about two minutes while `//blp/refdata` times out, so the pull stops
paying that wait once the Terminal has proved unreachable; the next run
tries it again from scratch.

Each symbol is stored from one source at a time (`source` column in
`prices.parquet`). Switching provider re-pulls a symbol's full window
rather than splicing two sources; `--force` re-pulls everything.
`--daily` is the one mode that suspends that rule (section 2.4).
Non-US lines need a `tiingo` symbol in
`Code/config/approved_instruments.csv` or they are skipped with a message.

### 2.3 Bot screen

Automated and duplicated posts are excluded from the aggregates before
anything is counted (`Code/ingestion/bot_screen.py`). The screen runs inside
every refresh and prints one line (`bot screen: excluded N of M posts`);
the same summary is stored in `Reports/bot_screen_last.json`.
Thresholds live in `Code/config/settings.csv` (`bot_screen_*`). To see what a
setting change would exclude before adopting it:

    python Code/tools/bot_screen_report.py --threshold 0.5 --examples 15

### 2.4 The daily run without a Terminal

A full pull asks the provider for every priced symbol — 267 of them, and
Tiingo answers one symbol per request. `--daily` prices the theme anchor
ETFs alone, about 27 symbols, which is the smallest set that keeps every
theme's line on the dashboard current:

    python Code/update_data.py --daily --provider tiingo

It changes the price step and nothing else: the fetch, the fold, the
rescore, the AI layer and the publish all run as they do in any live
refresh, and it combines with every other flag. A symbol already current
costs no request at all, so a run the day after a run is close to free.

Two consequences worth knowing:

- The anchors are the only lines that stay current. The single names,
  the fallback anchors and the approved-instrument benchmarks catch up
  on the next full pull, so run one (`python Code/update_data.py`) when
  a Terminal is available or when a wider set is needed.
- `--daily` extends a stored series with whichever provider answers
  instead of re-pulling it, so a series can hold Bloomberg history and
  Tiingo closes after it. Two vendors' closes differ after a corporate
  action, so that seam is reported rather than hidden: the pull prints
  how many series carry two vendors, the `source` column records which
  rows came from where, and `python Code/tools/data_health.py` names
  the series on its `mixed-source series` line. To put a symbol back on
  one vendor, re-pull it with a normal (non-`--daily`) run, which
  replaces its whole window.

To exercise only the price step — it is the long pole of a run, and the
one stage whose cost is set by the vendor:

    python Code/update_data.py --prices-only            # full universe
    python Code/update_data.py --prices-only --daily    # the anchors

That fetches nothing, folds nothing, recomputes nothing and publishes
nothing.

### 2.5 Price-step diagnostic

`--check` connects far enough to measure and writes nothing:

    python Code/ingestion/pull_prices.py --check
    python Code/ingestion/pull_prices.py --check --daily --provider tiingo

It prints each provider in the chain with the exact reason an unusable
one is unusable; the store's rows, symbols, span, how stale the newest
close is and how many series carry two vendors; what a daily run and a
full run would each cost today (spans and symbol-requests); and one
timed single-symbol round trip against the live provider, so seconds ×
symbol-requests gives the cost of a whole pull on this machine. With no
provider usable it says so and skips the timing. `--daily` makes the
daily plan the headline. Exit 0 when it could report.

## 3. Dashboard

The dashboard has no mode selectors: the production configuration is
fixed in code and read from the frozen record. Branding and the
pipeline buttons come from `Code/config/settings.csv`.

**An edit should appear on its own.** The file watcher is on
(`.streamlit/config.toml` at the repository root), and it is the thing
that clears Streamlit's compiled-script cache: with it off, a running
server keeps executing the build it started with, so every local edit
needs a restart and every push to the hosted copy needs its Reboot
button. It is fenced with `folderWatchBlacklist` rather than disabled, so
the parquet stores the pipeline rewrites in place are not watched and a
refresh cannot trigger a mid-read reload. Save a file under `Code/` and
the page reloads by itself; push, and the hosted app follows within a
minute or two.

If an edit still does not appear, a second `streamlit run` has probably
taken the next port (8501 → 8502) while the pinned tab keeps serving the
first process. Stop them all and start one:

    Get-NetTCPConnection -LocalPort 8501,8502,8503 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
    python -m streamlit run Code/dashboard.py

The sidebar shows the build time and the serving port. A copy that turns
the watcher off keeps the stale-build banner, which says so in the
sidebar and on the page.

**Sidebar pipeline controls.** `show_pipeline_controls` in
`Code/config/settings.csv` is `false` in the committed file so a hosted copy
never offers a button that spends API credit. On a machine that runs the
pipeline, `Code/tools/publish_dashboard.py` writes the git-ignored
`Code/config/settings.local.csv` with the setting on; `RETAILAPOLLO_CONTROLS=1`
in the environment does the same for one process.

**Freshness line.** The masthead shows how many business days old the
data is and, on a hosted copy, whether the deployed bundle is behind the
one last published (`publish_manifest.json`). A viewer sees one plain
sentence; the machine with the pipeline controls sees the diagnosis and
the command. The remedy for either is a refresh and a push (section 4).

**ETF lookup (beta).** The landing page has a search box for any ETF,
configured or not. It resolves the text (configured themes, the local
`Code/config/etf_catalogue.csv`, then the Terminal's security lookup when one
answers), reads the ETF's top holdings (from the Terminal, or typed by
hand), derives a word list
from the names, assembles attention and sentiment from the aggregates,
pulls the price (store first, then the price provider, cached under
`Data/prices/lookups/`), and scores the result two ways: the rule-based
euphoria level with the frozen gauge bands, and the production
INCREASE / CUT EXPOSURE model applied read-only at its frozen cuts. The
model comes from `Data/processed/euphoria_desk_model.joblib`, written by
every analytics pass (the fitted ensemble plus the population it
scored, so a new name is ranked on the same scale); nothing is fitted or
re-thresholded for a lookup, and no capture rate is claimed for it.
Self-test: `cd Code && python -m src.lookup india` (add `--no-terminal` to skip the
Terminal). Heading text: `Code/config/settings.csv → lookup_section_title`.

## 4. Hosted dashboard

A hosted dashboard (for example Streamlit Community Cloud) is a **view**:
it never fetches, never recomputes and never reaches a Terminal. It
renders the bundle in `Data/dashboard/`.

`python Code/update_data.py` stages the bundle as its last step, so the
routine refresh needs no second command, only a commit:

    python Code/update_data.py
    git add Data/abstracted Data/dashboard
    git commit -m "publish dashboard"
    git push

| Task | Command |
|---|---|
| Stage the bundle on its own | `python Code/tools/publish_dashboard.py` (`--dry-run` to preview) |
| Refresh without touching the bundle | `python Code/update_data.py --skip-publish` |
| Include the author-level influence board | `python Code/tools/publish_dashboard.py --with-influence` |

The push is the deployment. Deployment settings: main file
`dashboard.py` (the root entry point, which runs `Code/dashboard.py`),
branch `main`, Python 3.12. No secrets are needed on the
host: nothing there reads a credential.

Two safeguards are automatic. `publish_dashboard.py` refuses to stage any
frame carrying an identity column (post text, author, id, forum), the
same rule that protects `Data/abstracted/`; and the bundle step runs
only after the text-free check has passed. On first load the host copies
the bundle into `Data/processed/` — a copy, never a recompute.

## 5. AI layer (optional)

Two providers behind one interface (`Code/src/ai.py`); `AI_PROVIDER` in `.env`
selects.

| Value | Behaviour |
|---|---|
| `auto` (default) | The `apollo` gateway first, the Anthropic API if it does not resolve. |
| `apollo` | An OpenAI-compatible gateway reached through the `dimsum_lite` client. Requires that package and its network route. |
| `anthropic` | The Anthropic API; needs only `ANTHROPIC_API_KEY`. |

With neither configured, `available()` is False and every AI consumer
degrades to a banner instead of failing the run. Model ids: `AI_MODEL`
(gateway deployment name, default `gpt-4o`) and `ANTHROPIC_MODEL`
(default `claude-sonnet-5`). `AI_MAX_CALLS` caps calls per process.

    cd Code && python -m src.ai --selftest   # prints the resolved provider and model
    python Code/update_data.py --ai          # regenerate only the AI panels

## 6. Research and re-validation

| Task | Command |
|---|---|
| Full re-validation (walk-forward + threshold re-freeze) | `python Code/src/analytics/run_analytics.py --what phases --research` |
| Rebuild aggregates from the raw store, then re-validate | `python Code/update_data.py --full` (full mode only) |
| Operating-point sweep (writes nothing) | `python "Reference Materials/archive/research/tools/sweep_operating_point.py"` (optional folder) |
| Rebuild `research.ipynb` from the current data | `python Code/tools/build_research_notebook.py` |

Research re-opens in exactly three ways: `--research`, `--full`, or
automatically when a frozen record is missing (a fresh copy). A record
that lags the data is reported in the run log, never refitted as a side
effect of a refresh. **Any backfill or
historical fold must be followed by `--what phases --research`**: a fold
rewrites the history the thresholds were fitted on, and scoring new
history against thresholds fitted on the old one is a silent lookahead.

The model family is pinned (`DESK_MODEL_FAMILY = "ens"` in
`Code/src/config.py`): a research pass fits and judges that family only.
Set it to `None` to re-open the full tournament, which is required if
the feature bank, the labels or the universe change.

## 7. Historical data

| Task | Command |
|---|---|
| Backfill a Reddit gap (resumable, chunked) | `python Code/tools/backfill_reddit.py --start YYYY-MM-DD --end YYYY-MM-DD` (`--status` to inspect, `--estimate` to size) |
| Fold pulled history into the aggregates | `python Code/tools/fold_historical.py --arctic` (`--dry-run` first) |
| Fold dump archives | `python Code/tools/fold_historical.py --dumps FILE...` |

In aggregates mode, `Code/tools/fold_historical.py` is the only path by which
historical posts reach the aggregates: the live fold keeps only posts
dated on or after `LIVE_START` by design. The fold ledger is
per-(file, month), so re-running cannot double count; the backfill ledger
is keyed by window and forum set, so adding forums later correctly
re-opens a finished window. After any fold, run the research pass
(section 6).

## 8. Ingestion internals worth knowing

- **Dedup** is an uncapped, atomically written parquet
  (`Data/reference/abstracted_seen_ids.parquet`). Aggregate merges are
  additive with no post ids, so this set is the only guard against
  permanent double counting. `Data/reference/` is snapshotted into
  `_backups/` (last 7) after every successful fold. Posts excluded by
  the bot screen are recorded as seen so they are not re-judged.
- **Watermarks** advance only on a clean pagination finish; an
  interrupted forum re-covers its window on the next run.
- **Forum panel**: a monthly review may add at most one forum
  (`exploration` tier) to `Code/config/forums.csv`; every addition is logged
  in `Data/reference/subreddit_panel.json` and the review text is
  written to `Reports/panel_review_latest.md`.
- **Fetch fast mode** (backfills): `limit=auto`, minimal fields,
  header-based pacing. The live fetch uses the conservative path.

## 9. Scheduling and unattended operation

A run that nobody watches needs two things beyond a command in a
scheduler: it has to happen even on a day the machine was switched off
at the scheduled minute, and it has to say so when it goes wrong.
`Code/tools/setup_schedule.py` covers the first. A GitHub Actions
workflow covers the second, from outside the machine and without a
single credential.

### 9.1 The daily task

`Code/tools/setup_schedule.py` builds the task definition and registers
it. Printing is the default, so the definition can be read before it
exists anywhere:

```powershell
python Code\tools\setup_schedule.py                   # print, change nothing
python Code\tools\setup_schedule.py --register        # create the task
```

Run it with the interpreter the task should use — it puts that
interpreter in the definition. Defaults: the name `retailAPOLLO daily
refresh`, 17:30 local, `Code\update_data.py --daily`, the project folder
as the working directory. `--time`, `--task-name`, `--args`, `--python`
and `--time-limit` change any of those.

Why 17:30 and why `--daily`: the newest US close lands at 04:00 local in
UTC+8, so a late-afternoon run has more than thirteen hours of margin on
it, and `--daily` prices the theme anchors rather than the full universe
(section 2.4), which is the price step a machine without a Terminal can
finish quickly every day.

The definition is registered as XML rather than built out of `schtasks`
switches because the switches cannot express the one setting that makes
the schedule unattended:

| Setting | Value | Why |
|---|---|---|
| `StartWhenAvailable` | true | A start missed while the machine was off runs as soon as it is back. Without it, a day switched off is a day skipped. |
| `WakeToRun` | false | `--wake` sets it true. Off by default: waking depends on firmware and the power plan, so a task that relies on it can look scheduled and never fire, and `StartWhenAvailable` already recovers the missed day. |
| `MultipleInstancesPolicy` | IgnoreNew | A run that overruns is never joined by the next day's. |
| `DisallowStartIfOnBatteries` | false | A laptop on battery still refreshes. |
| `ExecutionTimeLimit` | PT4H | A run takes minutes; one still going after four hours is stuck, and a stuck run would block the next day's start. |
| `LogonType` | InteractiveToken | The task runs as the logged-on account and stores no password. |

Afterwards:

```powershell
schtasks /Query  /TN "retailAPOLLO daily refresh" /V /FO LIST   # inspect
schtasks /Run    /TN "retailAPOLLO daily refresh"               # run it now
schtasks /Change /TN "retailAPOLLO daily refresh" /DISABLE      # pause
schtasks /Change /TN "retailAPOLLO daily refresh" /ENABLE       # resume
schtasks /Delete /TN "retailAPOLLO daily refresh" /F            # remove
```

`--register` refuses to touch a task of the same name that already
exists; `--force` replaces it, and `--task-name` registers a second one
alongside. On a machine that is not Windows there is no `schtasks` to
call: the script prints the command and the definition, says that
nothing was registered, and exits.

### 9.2 The freshness watchdog

The pipeline runs on one machine and reports on itself, which leaves two
failures it cannot see by construction: a run that never started (the
machine was off, the task did not fire, the interpreter died on import),
and a publish that finished locally and never reached the remote. Both
look identical from the repository — the committed bundle stops
advancing — so the watching is done from there instead.

`.github/workflows/publish-freshness.yml` is the whole of it. It checks
the repository out on a GitHub runner and runs

```bash
python Code/tools/check_published_freshness.py --max-age-days 3
```

against the checkout. The script reads two small files that the pipeline
commits inside the dashboard bundle: `publish_manifest.json`, which
carries the newest data day the bundle holds, and `run_status.json`,
which carries the verdict the run reached on itself. A job that fails
notifies the repository owner through GitHub's own notification
settings, which is why the watchdog needs no credential of any kind and
this repository holds no secret.

**When it runs.** 13:00 UTC every day, a few hours after the machine's
own 17:30 local slot, so a run that started on time has finished and
pushed before the job looks. `workflow_dispatch` is also on: **Actions →
published bundle freshness → Run workflow** runs it by hand, on any
branch, which is how to check a change to the script or to see the
current verdict without waiting for tomorrow.

**What each result means.** The script's exit code is the job's result,
and every run writes its output to the job summary page whether it
passed or failed:

| Exit | Result | Meaning |
|---|---|---|
| 0 | `CURRENT` | The published data day is inside the allowance and the run that published it raised nothing. |
| 1 | `STALE` | The newest published data day is further behind today than `MAX_AGE_DAYS`. Either the daily run is not firing, or it is running and its publish is not reaching the remote. |
| 1 | `UNREADABLE` | A manifest or a run status in the bundle does not parse, which means the publish that wrote it did not complete. |
| 2 | `NO BUNDLE` | Neither spelling of the bundle folder holds a manifest, so the hosted dashboard has no data to serve. |
| 3 | `RUN RAISED` | The published data is current, and the run that published it raised at least one condition (section 9.3). The publishing works; the fault is inside the run. |

**Tuning the allowance.** `MAX_AGE_DAYS` is an `env:` value on the check
step in the workflow file. Three days absorbs a weekend on a machine
that is switched off; tighten it to 2 once daily runs are reliable, or
widen it while travelling. Editing the file and pushing is the whole
change — there is nothing to register and nothing to restart.

**Reading `run_status.json`.** It sits beside `publish_manifest.json` in
the committed bundle under `Data/dashboard/`, and it is written by the
publish step of every real refresh:

```json
{
  "conditions": [
    {
      "reason": "a paid dependency stopped serving this account",
      "tag": "credit or key exhausted"
    }
  ],
  "data_through": "2026-09-14",
  "exit_code": 0,
  "finished": "2026-09-14 17:41"
}
```

`finished` and `exit_code` describe the run, `data_through` is the
newest day in the aggregates it scored, and `conditions` is one entry
per condition raised, empty on a clean run. It exists because a healthy
bundle is not a healthy run: a run can lose a stage, or have a paid key
refused, and still publish a bundle whose newest day is today, and from
a checkout those are invisible. The full detail — the log lines behind
each condition — stays in the ALERT block in `Reports/logs/` on the
machine.

One thing is deliberately **not** in it: whether the push landed. The
publish step runs before the commit and the push, so that answer does
not exist yet when the file is written. The watchdog measures publishing
the other way, by whether `data_through` advances from one day's bundle
to the next.

An older bundle carries no run status at all, and the script treats a
missing one as silence rather than as a failure.

**GitHub disables a scheduled workflow after 60 days of no activity on
the repository.** This one is pushed to every day by the pipeline
itself, so the clock is reset daily in normal operation — but a copy
parked for two months comes back with its schedule switched off. The
Actions tab says so and offers a button to enable it again.

### 9.3 What each condition means

Every run ends with an ALERT block naming every condition it raised, or
one line saying none did. The tags below are what the block prints and
what `run_status.json` carries out to the watchdog.

| Tag | What happened | What to do |
|---|---|---|
| `run failed` | The run exited non-zero, or a stage did while the run carried on (the price pull is the usual case). The block names each command and its exit code. | Read the named stage in the run log. |
| `publish did not land` | The refreshed data did not reach the hosted app. Everything except a verified push and the honest "nothing new — data unchanged" counts: a refusal, a failed `git add`, a commit carrying the previous run's bundle, a push that failed. The block carries the sentence the publish step returned. This one is the ALERT block alone: it is decided after the bundle is written, so it never reaches `run_status.json`. | Section 4. The dashboard is serving the previous run's numbers until this is resolved. |
| `credit or key exhausted` | A paid dependency stopped serving this account: Tiingo's request quota (HTTP 429) or a refused key (401/403), FetchLayer out of credits (HTTP 402), or the AI provider reporting no quota or credit left. Pacing (HTTP 429 from FetchLayer or StockTwits) and this pipeline's own per-run credit cap are **not** this — they are normal and stay quiet. | Top up, raise the plan, or issue a new key. The run itself is valid on whatever it did get. |
| `data stale` | The run finished clean and the newest day in the aggregates is the same as the previous run's. Nothing new reached the store even though every stage reported success. | Check the fetch stage in the log and whether the sources answered. |

The comparison for `data stale` is kept in `Reports/run_state.json` —
the previous run's newest data day, when that day first appeared, and
when the run finished. It is machine-local and git-ignored like the rest
of `Reports/`. Deleting it costs one run of staleness detection: the
next run has nothing to compare against and stays quiet.

Two runs on the same day over unchanged data will raise `data stale`,
which is the detector being right rather than noisy — the day genuinely
did not advance.

### 9.4 How to pause it

- Pause the daily run and keep the watchdog:
  `schtasks /Change /TN "retailAPOLLO daily refresh" /DISABLE`. Expect
  the watchdog to start failing once the published data day falls past
  `MAX_AGE_DAYS`, which is the watchdog doing its job.
- Pause the watchdog and keep the daily run: **Actions → published
  bundle freshness → ⋯ → Disable workflow**. Re-enable it from the same
  menu.
- Stop both: disable the workflow and delete the task
  (`schtasks /Delete /TN "retailAPOLLO daily refresh" /F`).

Nothing else changes in any of these cases. Running
`python Code/update_data.py` by hand prints the same ALERT block and
writes the same `run_status.json` as a scheduled run.

## 10. Recovery

| Symptom | Action |
|---|---|
| Aggregates look doubled for a month | `python Code/tools/data_health.py` (double-count scan), then check `Data/reference/historical_fold_ledger.json` for two entries covering that month. |
| Ledger lost or corrupted | Restore the newest snapshot from `Data/reference/_backups/`. |
| Dashboard shows a stale build | Restart the server (section 3). |
| Hosted dashboard says the bundle is behind | Run a refresh, commit `Data/abstracted Data/dashboard`, push. |
| A backfill finished but coverage did not move | The fold is a separate step: `python Code/tools/fold_historical.py --arctic`, then the research pass. |
| Price pull fails on Bloomberg | It falls back to Tiingo automatically; `--provider tiingo` pins it. `python Code/ingestion/pull_prices.py --check` reports which providers answer and what a pull would cost; `cd Code && python -m src.prices --provider tiingo` tests Tiingo on its own. Check `data_health.py` for a corporate-action warning if a series jumps. |
| The price step is too slow to sit through | `python Code/update_data.py --daily` (anchors only, section 2.4), or `--prices-only` to run just that step. |
| Bot screen excludes too much or too little | `python Code/tools/bot_screen_report.py --examples 20`, adjust `bot_screen_*` in `Code/config/settings.csv`, re-run. |
| A config edit is rejected | `python Code/tools/validate_config.py` prints the exact row and rule. |
| The watchdog never reports anything | Run it by hand from the Actions tab (section 9.2); a job that passes prints `CURRENT` on its summary page. A repository with no activity for 60 days has its schedule disabled by GitHub. |
| The watchdog reports `data stale` every day | The fetch stage is returning nothing new. Read the fetch output in `Reports/logs/`, then `python Code/ingestion/check_live_ingestion.py`. |
| The task exists but never fires | `schtasks /Query /TN "retailAPOLLO daily refresh" /V /FO LIST` shows the last result and the next run time. A machine asleep at the scheduled minute runs the task when it wakes, not while it sleeps, unless the task was registered with `--wake` (section 9.1). |
