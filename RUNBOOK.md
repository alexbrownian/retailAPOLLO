# Runbook

All commands run from the project root. Routine operation is one
command (`python update_data.py`); everything else here covers setup,
maintenance, and recovery. Parameter provenance: `docs/DECISIONS.md`.

## Routine operation

| Task | Command |
|---|---|
| Refresh everything (live) | `python update_data.py` — parallel fetch, fold, score at frozen thresholds, pull prices |
| Open the dashboard | `python -m streamlit run dashboard.py` |
| Recompute without API calls | `python update_data.py --skip-fetch` |
| Analytics only (seconds) | `python -m analytics.run_analytics` |
| Health check (read-only, safe any time) | `python tools/data_health.py` |
| Project soundness check | `python tools/preflight.py` |
| Dependency / dangling-reference check | `python tools/verify_deps.py` |
| Pull Bloomberg prices (Terminal open) | `python pull_bloomberg_prices.py` (`--dry-run` to preview) |
| See what live data landed | `python check_live_ingestion.py` |
| Full test suite | `python -m pytest tests/ -q` |

Run cadence: about twice a week. The comment fetch is budgeted so that
at that cadence nothing is ever deferred; a longer gap defers the
oldest pages to the next run (deferral, never loss — a capped
subreddit keeps its watermark).

## Dashboard

The dashboard has no mode selectors; the production configuration is
fixed in code (`dashboard.py`, signal-configuration block):

- Trigger: shipped (crowd + price). The crowd-only `*_xp` columns are
  still computed on every run as a research record.
- Operating point: the F1 columns (`get_in` / `get_out`). The F0.5
  precision-weighted `*_strict` columns are what the research pack
  quotes; both are stored on every run.
- GET IN is ungated; GET OUT keeps its phase gate.
- The dial shows % of signal: 100 means the call fires.

**A code or config edit is not showing**: restart the server — the
file watcher is off by design (`.streamlit/config.toml`) so the
pipeline rewriting parquet in place cannot trigger a mid-read reload.
Stop every running Streamlit first, or the new one silently takes the
next port (8501 → 8502) and a pinned browser tab keeps serving the old
process. PowerShell, targeted so a running pipeline is not killed with
it:

    Get-NetTCPConnection -LocalPort 8501,8502,8503 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
    python -m streamlit run dashboard.py

The sidebar shows the build time and the serving port.
`config/theme_etfs.csv` is the exception: it re-reads on its own
mtime, so a rerun is enough.

## Research and re-validation

| Task | Command |
|---|---|
| Full re-validation (walk-forward + threshold re-freeze) | `python -m analytics.run_analytics --what phases --research` |
| Force research from a data rebuild | `python update_data.py --full` (external machine) |
| Operating-point sweep (writes nothing) | `python tools/sweep_operating_point.py` |
| Re-run a research notebook | `python -m jupytext --to ipynb notebooks/<name>.py` then `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` |

Research re-opens in exactly three ways: `--research`, `--full`, or
automatically on year rollover / missing record. **Any backfill or
historical fold must be followed by `--what phases --research`**: a
fold rewrites the history the thresholds were fitted on, and scoring
new history against thresholds fitted on the old one is a silent
lookahead.

The model family is pinned (`DESK_MODEL_FAMILY = "ens"`): a research
pass fits and judges that family only (2 fits, ~2 minutes). Set it to
`None` in `src/config.py` to re-open the full tournament — required if
the feature bank, labels, or universe change.

## Historical data

| Task | Command |
|---|---|
| Backfill a Reddit gap (resumable, chunked) | `python tools/backfill_reddit.py --start YYYY-MM-DD --end YYYY-MM-DD` (`--status` to inspect; `--estimate` to size) |
| Fold pulled history into the aggregates (internal machine) | `python tools/fold_historical.py --arctic` (`--dry-run` first) |
| Fold torrent-style dump archives | `python tools/fold_historical.py --dumps FILE...` |

On the internal machine, `tools/fold_historical.py` is the **only**
path by which historical posts reach the aggregates: the live fold
keeps only posts dated on or after LIVE_START by design. The fold
ledger is per-(file, month), so re-running cannot double count; the
backfill ledger is keyed by window **and** subreddit set, so adding
subreddits later correctly re-opens a finished window. After any fold,
run the research pass (above). Details and the measured recovery:
`docs/RESEARCH_RECORD.md`.

## Ingestion internals worth knowing

- **Dedup** is an uncapped, atomically written parquet
  (`data/reference/abstracted_seen_ids.parquet`). Aggregate merges are
  additive with no post ids, so this set is the only guard against
  permanent double counting. `data/reference/` is snapshotted into
  `_backups/` (last 7) after every successful fold.
- **Watermarks** advance only on a clean pagination finish; an
  interrupted subreddit re-covers its window on the next run.
- **Subreddit panel**: a monthly review may auto-add at most one
  community (exploration tier); every addition is logged in
  `ingestion/subreddit_panel.json`. Review output:
  `docs/panel_review_latest.md`.
- **Fetch fast mode** (backfills): `limit=auto`, minimal fields,
  header-based pacing. The daily live fetch uses the conservative
  path.

## Environment

Credentials load from `.env` (see `ingestion/fetch_all.py --check`):
`FETCHLAYER_KEY` or `X_BEARER_TOKEN` for X; `REDDIT_*` for the OAuth
fallback. Arctic Shift needs no key. `blpapi` installs from Bloomberg's
index with the Terminal running (see `requirements.txt`). The AI poll /
pulse stages require the firm gateway and skip cleanly elsewhere.

## Recovery

| Symptom | Action |
|---|---|
| Aggregates look doubled for a month | `python tools/data_health.py` (double-count scan), then check `data/reference/historical_fold_ledger.json` for two entries covering that month. |
| Ledger lost or corrupted | Restore the newest snapshot from `data/reference/_backups/`. |
| Dashboard shows a stale build | Restart the server (watcher is off by design; see above). |
| A backfill "finished" but coverage did not move | On the internal machine the fold step is separate: run `python tools/fold_historical.py --arctic`, then research. |
