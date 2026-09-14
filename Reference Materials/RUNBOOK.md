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

`Code/tools/preflight.py` checks the interpreter, the packages, every config
CSV (`Code/tools/validate_config.py`), the data stores and the frozen record,
and prints one line per finding. A clean preflight is the definition of
"this copy is ready to run".

Python 3.11 or 3.12 is supported. `blpapi` (Bloomberg) installs from
Bloomberg's package index and is optional: without it, prices come from
Tiingo, which needs `TIINGO_API_KEY` in `.env` (a free key from tiingo.com).

## 2. Routine operation

| Task | Command |
|---|---|
| Refresh everything | `python Code/update_data.py` |
| Open the dashboard | `python -m streamlit run dashboard.py` (the root file runs `Code/dashboard.py`) |
| Recompute without any network call | `python Code/update_data.py --skip-fetch --skip-prices` |
| Analytics only (seconds) | `python Code/src/analytics/run_analytics.py` |
| Read-only health report | `python Code/tools/data_health.py` |
| Full soundness check | `python Code/tools/preflight.py` |
| Dangling reference / dependency check | `python Code/tools/verify_deps.py` |
| Bot-screen report on the raw store | `python Code/tools/bot_screen_report.py` |
| Refresh prices only | `python Code/ingestion/pull_prices.py` (`--dry-run` to preview) |
| What live data landed | `python Code/ingestion/check_live_ingestion.py` |
| Test suite | `python -m pytest Code/tests -q` (`pip install -r Code/tools/requirements-dev.txt` once) |

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

## 9. Scheduling

Windows Task Scheduler, twice a week:

```powershell
schtasks /Create /SC WEEKLY /D MON,THU /ST 06:30 /TN "retailAPOLLO refresh" ^
  /TR "C:\Python312\python.exe <path-to-project>\Code\update_data.py"
```

cron equivalent: `30 6 * * 1,4 cd <project> && python Code/update_data.py >> Reports/logs/cron.log 2>&1`.

## 10. Recovery

| Symptom | Action |
|---|---|
| Aggregates look doubled for a month | `python Code/tools/data_health.py` (double-count scan), then check `Data/reference/historical_fold_ledger.json` for two entries covering that month. |
| Ledger lost or corrupted | Restore the newest snapshot from `Data/reference/_backups/`. |
| Dashboard shows a stale build | Restart the server (section 3). |
| Hosted dashboard says the bundle is behind | Run a refresh, commit `Data/abstracted Data/dashboard`, push. |
| A backfill finished but coverage did not move | The fold is a separate step: `python Code/tools/fold_historical.py --arctic`, then the research pass. |
| Price pull fails on Bloomberg | It falls back to Tiingo automatically; `--provider tiingo` pins it. `cd Code && python -m src.prices --provider tiingo` tests Tiingo on its own. Check `data_health.py` for a corporate-action warning if a series jumps. |
| Bot screen excludes too much or too little | `python Code/tools/bot_screen_report.py --examples 20`, adjust `bot_screen_*` in `Code/config/settings.csv`, re-run. |
| A config edit is rejected | `python Code/tools/validate_config.py` prints the exact row and rule. |
