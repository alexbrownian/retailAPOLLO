# Runbook

Operating instructions for retailAPOLLO. All commands run from the
project root. Routine operation is one command; the rest of this page
covers setup, publishing, maintenance and recovery. For *why* a
parameter has its value, see `reference/KEY_PARAMETERS.md`; for how the
pieces fit together, `docs/ARCHITECTURE.md`.

## 1. Setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp example.env .env                               # optional keys; see docs/LIVE_INGESTION.md
python tools/preflight.py
```

`tools/preflight.py` checks the interpreter, the packages, every config
CSV (`tools/validate_config.py`), the data stores and the frozen record,
and prints one line per finding. A clean preflight is the definition of
"this copy is ready to run".

Python 3.11 or 3.12 is supported. `blpapi` (Bloomberg) installs from
Bloomberg's package index and is optional: without it, prices come from
Yahoo Finance.

## 2. Routine operation

| Task | Command |
|---|---|
| Refresh everything | `python update_data.py` |
| Open the dashboard | `python -m streamlit run dashboard.py` |
| Recompute without any network call | `python update_data.py --skip-fetch --skip-prices` |
| Analytics only (seconds) | `python -m analytics.run_analytics` |
| Read-only health report | `python tools/data_health.py` |
| Full soundness check | `python tools/preflight.py` |
| Dangling reference / dependency check | `python tools/verify_deps.py` |
| Bot-screen report on the raw store | `python tools/bot_screen_report.py` |
| Refresh prices only | `python pull_prices.py` (`--dry-run` to preview) |
| What live data landed | `python check_live_ingestion.py` |
| Test suite | `python -m pytest tests/ -q` |

One refresh does, in order: fetch (all sources in parallel), bot screen,
fold into the aggregates, coverage check, analytics at the frozen
thresholds, price pull, AI layer (if configured), text-free safety
check, publish bundle, summary. `--dry-run` prints the plan without
running anything.

Run cadence: twice a week is the design point. The comment fetch is
budgeted so that at that cadence nothing is deferred; after a longer gap
the oldest pages are deferred to the next run (deferral, never loss —
a capped forum keeps its watermark).

### 2.1 Run modes

`update_data.py --mode` selects how posts are stored:

| Mode | When | What happens |
|---|---|---|
| `full` | `data/processed/posts.parquet` exists | New posts are appended to the raw store and every aggregate can be rebuilt from text. |
| `aggregates` | No raw store | New posts are screened, aggregated and folded into `ABSTRACTED_DATA/` as text-free deltas. The raw files stay in `data/raw/`. |
| `auto` (default) | — | Detects which of the above applies. |

Both modes produce the same aggregate tables and the same dashboard. A
fresh clone is an aggregates-mode copy.

### 2.2 Price provider

`config/settings.csv → price_provider` sets the default; `--provider`
overrides it for one run, on `update_data.py` and on `pull_prices.py`.

| Value | Behaviour |
|---|---|
| `auto` (default) | Bloomberg if a Terminal answers, otherwise Yahoo Finance. |
| `bloomberg` | Bloomberg; falls back to Yahoo Finance if the Terminal fails mid-run unless `--no-fallback` is given. |
| `yfinance` | Yahoo Finance only. |

Each symbol is stored from one source at a time (`source` column in
`prices.parquet`). Switching provider re-pulls a symbol's full window
rather than splicing two sources; `--force` re-pulls everything.
Non-US lines need a `yfinance` symbol in
`config/approved_instruments.csv` or they are skipped with a message.

### 2.3 Bot screen

Automated and duplicated posts are excluded from the aggregates before
anything is counted (`ingestion/bot_screen.py`). The screen runs inside
every refresh and prints one line (`bot screen: excluded N of M posts`);
the same summary is stored in `data/reference/bot_screen_last.json`.
Thresholds live in `config/settings.csv` (`bot_screen_*`). To see what a
setting change would exclude before adopting it:

    python tools/bot_screen_report.py --threshold 0.5 --examples 15

## 3. Dashboard

The dashboard has no mode selectors: the production configuration is
fixed in code and read from the frozen record. Branding and the
pipeline buttons come from `config/settings.csv`.

**A code or config edit is not showing.** Restart the server. The file watcher is off
by design (`.streamlit/config.toml`) so a pipeline rewriting parquet in
place cannot trigger a mid-read reload. Stop every running Streamlit
first, or the new one silently takes the next port (8501 → 8502) and a
pinned browser tab keeps serving the old process. On Windows PowerShell:

    Get-NetTCPConnection -LocalPort 8501,8502,8503 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
    python -m streamlit run dashboard.py

The sidebar shows the build time and the serving port.
`config/theme_etfs.csv` is re-read on its own mtime, so a rerun is
enough for it.

**Sidebar pipeline controls.** `show_pipeline_controls` in
`config/settings.csv` is `false` in the committed file so a hosted copy
never offers a button that spends API credit. On a machine that runs the
pipeline, `tools/publish_dashboard.py` writes the git-ignored
`config/settings.local.csv` with the setting on; `RETAILAPOLLO_CONTROLS=1`
in the environment does the same for one process.

**Freshness line.** The masthead shows how many business days old the
data is and, on a hosted copy, whether the deployed bundle is behind the
one last published (`publish_manifest.json`). A viewer sees one plain
sentence; the machine with the pipeline controls sees the diagnosis and
the command. The remedy for either is a refresh and a push (section 4).

**ETF lookup (beta).** The landing page has a search box for any ETF,
configured or not. It resolves the text (configured themes, the local
`config/etf_catalogue.csv`, then the Terminal's security lookup when one
answers, else Yahoo Finance), reads the ETF's top holdings (Bloomberg
first, Yahoo as the fallback, or typed by hand), derives a word list
from the names, assembles attention and sentiment from the aggregates,
pulls the price (store first, then the price provider, cached under
`data/prices/lookups/`), and scores the result two ways: the rule-based
euphoria level with the frozen gauge bands, and the production
INCREASE / CUT EXPOSURE model applied read-only at its frozen cuts. The
model comes from `data/processed/euphoria_desk_model.joblib`, written by
every analytics pass (the fitted ensemble plus the population it
scored, so a new name is ranked on the same scale); nothing is fitted or
re-thresholded for a lookup, and no capture rate is claimed for it.
Self-test: `python -m src.lookup india` (add `--yahoo-only` to skip the
Terminal). Heading text: `config/settings.csv → lookup_section_title`.

## 4. Hosted dashboard

A hosted dashboard (for example Streamlit Community Cloud) is a **view**:
it never fetches, never recomputes and never reaches a Terminal. It
renders the bundle in `DASHBOARD_DATA/`.

`python update_data.py` stages the bundle as its last step, so the
routine refresh needs no second command, only a commit:

    python update_data.py
    git add ABSTRACTED_DATA DASHBOARD_DATA
    git commit -m "publish dashboard"
    git push

| Task | Command |
|---|---|
| Stage the bundle on its own | `python tools/publish_dashboard.py` (`--dry-run` to preview) |
| Refresh without touching the bundle | `python update_data.py --skip-publish` |
| Include the author-level influence board | `python tools/publish_dashboard.py --with-influence` |

The push is the deployment. Deployment settings: main file
`dashboard.py`, branch `main`, Python 3.12. No secrets are needed on the
host: nothing there reads a credential.

Two safeguards are automatic. `publish_dashboard.py` refuses to stage any
frame carrying an identity column (post text, author, id, forum), the
same rule that protects `ABSTRACTED_DATA/`; and the bundle step runs
only after the text-free check has passed. On first load the host copies
the bundle into `data/processed/` — a copy, never a recompute.

## 5. AI layer (optional)

Two providers behind one interface (`src/ai.py`); `AI_PROVIDER` in `.env`
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

    python -m src.ai --selftest       # prints the resolved provider and model
    python update_data.py --ai        # regenerate only the AI panels

## 6. Research and re-validation

| Task | Command |
|---|---|
| Full re-validation (walk-forward + threshold re-freeze) | `python -m analytics.run_analytics --what phases --research` |
| Rebuild aggregates from the raw store, then re-validate | `python update_data.py --full` (full mode only) |
| Operating-point sweep (writes nothing) | `python research/tools/sweep_operating_point.py` (optional folder) |

Research re-opens in exactly three ways: `--research`, `--full`, or
automatically on year rollover or a missing record. **Any backfill or
historical fold must be followed by `--what phases --research`**: a fold
rewrites the history the thresholds were fitted on, and scoring new
history against thresholds fitted on the old one is a silent lookahead.

The model family is pinned (`DESK_MODEL_FAMILY = "ens"` in
`src/config.py`): a research pass fits and judges that family only.
Set it to `None` to re-open the full tournament, which is required if
the feature bank, the labels or the universe change.

## 7. Historical data

| Task | Command |
|---|---|
| Backfill a Reddit gap (resumable, chunked) | `python tools/backfill_reddit.py --start YYYY-MM-DD --end YYYY-MM-DD` (`--status` to inspect, `--estimate` to size) |
| Fold pulled history into the aggregates | `python tools/fold_historical.py --arctic` (`--dry-run` first) |
| Fold dump archives | `python tools/fold_historical.py --dumps FILE...` |

In aggregates mode, `tools/fold_historical.py` is the only path by which
historical posts reach the aggregates: the live fold keeps only posts
dated on or after `LIVE_START` by design. The fold ledger is
per-(file, month), so re-running cannot double count; the backfill ledger
is keyed by window and forum set, so adding forums later correctly
re-opens a finished window. After any fold, run the research pass
(section 6).

## 8. Ingestion internals worth knowing

- **Dedup** is an uncapped, atomically written parquet
  (`data/reference/abstracted_seen_ids.parquet`). Aggregate merges are
  additive with no post ids, so this set is the only guard against
  permanent double counting. `data/reference/` is snapshotted into
  `_backups/` (last 7) after every successful fold. Posts excluded by
  the bot screen are recorded as seen so they are not re-judged.
- **Watermarks** advance only on a clean pagination finish; an
  interrupted forum re-covers its window on the next run.
- **Forum panel**: a monthly review may add at most one forum
  (`exploration` tier) to `config/forums.csv`; every addition is logged
  in `data/reference/subreddit_panel.json` and the review text is
  written to `data/reference/panel_review_latest.md`.
- **Fetch fast mode** (backfills): `limit=auto`, minimal fields,
  header-based pacing. The live fetch uses the conservative path.

## 9. Scheduling

Windows Task Scheduler, twice a week:

```powershell
schtasks /Create /SC WEEKLY /D MON,THU /ST 06:30 /TN "retailAPOLLO refresh" ^
  /TR "<path-to-venv>\Scripts\python.exe <path-to-project>\update_data.py"
```

cron equivalent: `30 6 * * 1,4 cd <project> && .venv/bin/python update_data.py >> logs/cron.log 2>&1`.

## 10. Recovery

| Symptom | Action |
|---|---|
| Aggregates look doubled for a month | `python tools/data_health.py` (double-count scan), then check `data/reference/historical_fold_ledger.json` for two entries covering that month. |
| Ledger lost or corrupted | Restore the newest snapshot from `data/reference/_backups/`. |
| Dashboard shows a stale build | Restart the server (section 3). |
| Hosted dashboard says the bundle is behind | Run a refresh, commit `ABSTRACTED_DATA DASHBOARD_DATA`, push. |
| A backfill finished but coverage did not move | The fold is a separate step: `python tools/fold_historical.py --arctic`, then the research pass. |
| Price pull fails on Bloomberg | It falls back to Yahoo Finance automatically; `--provider yfinance` pins it. Check `data_health.py` for a corporate-action warning if a series jumps. |
| Bot screen excludes too much or too little | `python tools/bot_screen_report.py --examples 20`, adjust `bot_screen_*` in `config/settings.csv`, re-run. |
| A config edit is rejected | `python tools/validate_config.py` prints the exact row and rule. |
