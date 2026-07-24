# RUNBOOK — what to run, when

Every command runs from the project root (`retailAPOLLO/`).
Most days only one command is needed: `python update_data.py` — or none at
all: the dashboard's sidebar buttons run the same pipelines.

## Quick reference

| Task | Command |
|---|---|
| Refresh everything (LIVE, fast) | `python update_data.py` — parallel fetch of all sources, fold, recompute signals, pull prices |
| Open the terminal | `python -m streamlit run dashboard.py` |
| Backtest / view a past window | set the window in `src/config.py` (or `--start/--end`), then `python update_data.py` — instant: the aggregates are window-independent |
| Build aggregates over ALL history | `python update_data.py --full` — external machine, run ONCE (and after changing BUILD_START_DATE or the theme definitions) |
| Recompute without API calls | `python update_data.py --skip-fetch` |
| Recompute ONLY the analytics | `python -m analytics.run_analytics` (seconds) |
| Windowed ticker-signal backtest | `python -m analytics.run_analytics --what signals --start 2026-01-01 --end 2026-06-01` |
| Check the FetchLayer key | `python ingestion/fetch_all.py --check` (no calls) / `--test` (1 credit) |
| See what live data landed | `python check_live_ingestion.py` |
| Pull Bloomberg prices | `python pull_bloomberg_prices.py` (`--dry-run` to preview) |
| Refresh euphoria/onset LIVE data | automatic in every pull - scores at the FROZEN walk-forward thresholds in seconds (intra-year recompute is a no-op: thresholds train on strictly earlier years) |
| Re-run the FULL euphoria/onset validation | `python -m analytics.run_analytics --research` (or `--what euphoria`/`--what phases` with `--research`) - walk-forward + ablation + ML challenger + threshold re-selection. Run after a backfill or rule change; auto-triggers on year rollover or a missing report; `update_data --full` forces it |
| Extend prices to full history | PowerShell: `$env:PIPELINE_START_DATE="2017-01-01"; python pull_bloomberg_prices.py` (incremental — pulls only the missing 2017-2020 spans, then rerun the euphoria stage) |
| Comments + Influence Tracker | `python update_comments.py` — the dedicated runner (desk decision 2026-07-24: comments LEFT the daily pipeline because they are the slow fetch — 10-50x post volume at the API's polite 1s/page). It prints an upfront time estimate (first run ~10-25 min; incremental ~1-4 min), fetches comments (watermarked, Ctrl-C-safe, resumable), then updates the influence board in one go. The store still builds itself from nothing on the first run and re-judges matured calls automatically. `python update_data.py --with-comments` restores the old bundled behaviour for one run. `python -m analytics.influence --top 20` prints the board |
| Dynamic subreddit panel | NOTHING TO RUN — a monthly, watermarked review rides every live pull (`ingestion/discover_subreddits.py --if-due`): it mines collected text for r/NAME referrals, and a candidate with ≥100 unique panel referrers/28d (the A0 floor, reused) that passes the finance screen auto-joins the EXPLORATION tier (max 1/review). Audit trail: `ingestion/subreddit_panel.json` + `docs/panel_review_latest.md`. Force a review: `python ingestion/discover_subreddits.py` (`--report-only` to rank without adding) |
| Rebuild the ONSET detector (Start/End radar data) | `python -m analytics.run_analytics --what phases` — LIVE mode: episode catalog + today's scores/alerts at the frozen threshold (seconds); add `--research` for the full walk-forward scorecard + threshold re-selection |
| Re-run the phases research notebooks | `cd notebooks` then `jupyter nbconvert --to notebook --execute --inplace 01_*.ipynb 02_*.ipynb 03_*.ipynb 04_*.ipynb 06_*.ipynb` — every figure/number re-renders from current data (06 = the full signal-efficacy report: forward returns at 3/10/21/84d, hit rates vs baseline, event study, overlay PnL, per-name tables) |
| Influential-users model (notebook 05) | AFTER the first live pull has seeded the influence store: `jupyter nbconvert --to notebook --execute --inplace notebooks/05_influence_users_model.ipynb` — a standing experiment, re-run any time |
| Run the tests | `python -m pytest tests/ -v` |
| Rebuild the presentation evidence pack | `python helper/research_charts.py` — every validation chart + correlation/calibration test regenerated from CURRENT data into `docs/research/` (figures, `research_stats.json`, README) |

## The one switch: the window

```python
START_DATE = "2021-01-01"   # inclusive
END_DATE   = ""             # "" = LIVE (to today);  "2021-11-01" = backtest window
```

`END_DATE = ""` → live fast path (minutes). A date → a frozen backtest
VIEW — instant, because the aggregates are built once over
`BUILD_START_DATE` → today (`--full`) and every window is just a lens over
them. The window drives the Bloomberg pull and the dashboard;
`--start/--end` override it for a single run. A **WINDOW CHECK** in every
run's output flags, per source, whether the chosen window actually has
data — an empty chart is never a mystery.

Every run prints a **data coverage table** (posts per month, per source) so
gaps are visible immediately, and ends with the text-free **safety check**
on ABSTRACTED_DATA.

## Which pipeline steps run where

| Step | External machine (raw store) | Internal machine |
|---|---|---|
| fetch (3 sources, parallel) | every live run | every live run |
| append | `merge_live.py` → posts.parquet, then tail splice | `append_live_abstracted.py` → ABSTRACTED_DATA |
| `--full` rebuild from raw text | yes (`build_aggregates.py`, all cores) | never — no raw text exists there |
| analytics (conviction + signals) | every run | every run |
| Bloomberg pull + dashboard | yes | yes |

The machine is auto-detected (posts.parquet present = external); force
with `--external` / `--internal`.

## Initial setup (once per machine)

```powershell
pip install -r requirements.txt --user
```

Create `.env` in the project root with the FetchLayer key
(`FETCHLAYER_KEY=...`). `.env` is never committed. StockTwits and Arctic
Shift Reddit need no key.

**Internal machine, first time only:**

```powershell
git pull
python -c "from src import abstracted_data; abstracted_data.hydrate()"
```

`hydrate()` copies the committed aggregates from `ABSTRACTED_DATA/` into
`data/processed/`, where the analytics look. After this one step,
`update_data.py` keeps the two in sync automatically.

For the Bloomberg prices, install blpapi once per machine (Terminal running):

```powershell
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi --user
python -c "import blpapi; print('blpapi', blpapi.__version__)"
```

## Everyday live refresh

The same command on both machines — it auto-detects which one it is on:

```powershell
python update_data.py
```

Then commit the updated aggregates + the influence store (the store is
committed by design — text-free, pseudonymous; the safety check covers
both):

```powershell
git add ABSTRACTED_DATA data/reference/influence
git commit -m "live update"
git push
```

(Only commit when the run's safety line says **PASS**.)

## Backtest / study a past regime

1. Set the window: `--start 2021-01-01 --end 2021-11-01` (or edit
   `src/config.py`).
2. `python update_data.py --start 2021-01-01 --end 2021-11-01` — backtest
   mode skips fetching automatically; nothing rebuilds unless stale.
3. Open the dashboard, set the same window in the sidebar — every tab
   (overlays included) clips itself to it.

## The dashboard tabs (RetailRadar - all interactive, no notebooks)

- **EUPHORIA: Themes** / **EUPHORIA: Singles** — the headline signal,
  one tab per instrument kind (desk decision 2026-07-24: conclusions
  only on the terminal; evidence in the notebooks). Each tab: a sparse
  state strip (names STARTING = onset alert in the last 21d, names
  ENDING = top alert in the last 21d - empty is the radar working), then
  per-instrument charts with the state ON the chart: BLUE vertical line
  = euphoria starting, RED vertical line = euphoria ending, euphoria
  level underneath, "EUPHORIA STARTING/ENDING NOW" badge in the title.
  A single caption states the validated record; the walk-forward
  tables, ablation, ML challenger and tournament live in
  `notebooks/01-04` + `docs/DECISIONS.xlsx`, not on the terminal
- **Trade desk** — the live ledger, scorecard, certainty ranking, signal
  charts with per-trade reasons; INSTRUMENT LOOKUP expander above the
  tabs shows every suggestion + reason for one tradeable instrument
- **Overlays: themes** — first derivative vs anchor ETF, conviction
  crossings, BUY/SELL on the anchor price + report card. (Single-ticker
  overlays were removed — the desk trades themes via anchor ETFs only;
  ticker research lives in `analytics/`, e.g. windowed backtests via
  `run_analytics --what signals`)
- **Top trends / Emerging trends** — most-mentioned and fastest-growing
  tradeable themes. Emerging has a **growth lookback slider (3–30d)**:
  7d = twitchy early-warning list, 21d = sustained build-ups only
- **Conviction** — ranked by an **EWMA of conviction z** (half-life
  slider, default 10d), with the latest z and the old flat 30d average
  shown alongside. The z uses an EWM trailing baseline (validated July
  2026 on real prices: best cross-validated PnL AND self-recentring
  after coverage shocks). Grey open triangles = the signal reverting to
  neutral, the validated early-exit point; the trade desk flags OPEN
  trades whose conviction has REVERTED ("consider exit" instead of
  waiting out the 20d cap). Conviction is computed live on the
  dashboard - no recompute needed to see engine changes
- **Historical checker** — any window, any theme

Every theme list and picker shows the TRADEABLE universe only (themes
with a firm-approved instrument in `THEME_ETFS`).
The sidebar's `items per section` slider controls how many charts each tab
draws; the window controls at the top drive every tab at once. Chart
note: days with under `MIN_TOTAL` total mentions are masked as too thin;
those stretches draw as a dotted, dimmed bridge with the legend key
"not enough posts that day" — visibly different from real data, and no
values are invented.

## Running pipelines from the dashboard

The sidebar buttons launch the pipelines as BACKGROUND processes: the
app stays responsive, a panel shows a progress bar + plain-English stage
checklist (markers parsed from the run log every 2s), and **cancel**
kills the whole process tree (fetchers and analytics included). The raw
output lives in the panel's *technical log* expander — it auto-opens
when a run fails. Buttons grey out while a pipeline runs (one at a
time); hit *dismiss* on the finished panel to refresh the data views.

In a terminal fetch, a `still fetching: Reddit, X` heartbeat line prints
every ~30s — the pauses are deliberate API rate-limit pacing
(StockTwits ~1.5s/symbol, X 5s/request, Arctic Shift 1s/page), so a
quiet minute is normal, not a hang.

**How long should things take?** (also shown on the dashboard buttons)

| Run | Typical time | Where it goes |
|---|---|---|
| LIVE pull, first run of the day | ~3–10 min | X rate-limit pacing dominates; Reddit paginates only what is NEW since the last run (per-subreddit watermark, 1-day overlap) |
| LIVE pull, repeat run | ~2–5 min | mostly the X pass |
| window rebuild (prices + signals) | ~1–3 min | Bloomberg pull is incremental — covered spans are skipped |
| analytics only | ~1 min | pure local compute |
| FULL rebuild | 30 min – hours | raw-text extraction + sentiment over the whole build range (external machine) |

## Transfer: external → internal

External machine (after `update_data.py`):
`git add ABSTRACTED_DATA && git commit && git push`.

Internal machine: `git pull`, then `python update_data.py` (it detects the
fresh aggregates and recomputes the derived outputs once).

## If something looks off

- `python check_live_ingestion.py` — freshness of every layer, in flow order.
- The **DATA COVERAGE** table in the run output shows exactly which months
  have data, per source ('.' = a real gap).
- The safety line at the end of every run must say **PASS** before
  committing ABSTRACTED_DATA.
- Close Excel/viewers before a run (Windows locks the parquet files); if a
  write fails, the script prints the manual rename fix.
- `python -m pytest tests/ -v` — the invariants that would catch silent
  corruption (merge maths, look-ahead, dedup, text-leaks).
