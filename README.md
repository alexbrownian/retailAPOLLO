# retailAPOLLO — retail attention → signals → price

retailAPOLLO measures how much retail attention each stock **ticker** and
**theme** receives across **17 finance subreddits, X (Twitter) and
StockTwits**, detects attention take-offs, scores sentiment, combines the
two into a **conviction score**, and emits **BUY/SELL signals with explicit
reasons** — then overlays everything against **Bloomberg prices** to assess
whether the crowd leads the move.

It is the full re-engineering of the RetailFlow1 project with the same
counting rules, thresholds and data contracts, but **no notebooks anywhere**:
every analysis that used to be a rendered `.ipynb` is now a plain importable
`.py` module, and the charts render interactively in one Streamlit
dashboard. A full recompute of nine years of signals takes **seconds**, not
the minutes the notebook chain needed — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for exactly where the speed
comes from.

Everything runs from one command — `python update_data.py` — which fetches
the latest posts (three sources **in parallel**, with a 30-second heartbeat
so a long rate-limited fetch never looks frozen), folds them into small
**text-free daily aggregates** (`ABSTRACTED_DATA/` — counts + sentiment, no
post text kept), recomputes conviction + signals, snapshots them and pulls
prices. The dashboard then renders every view for whatever window you pick.

## Architecture

```
  17 subreddits ──┐
  X (Twitter) ────┤──► ingestion/fetch_all.py ──► fold (dedup: first seen wins)
  StockTwits ─────┘        (parallel)                  │
                                                       ▼
              EXTERNAL machine                  ABSTRACTED_DATA/          INTERNAL machine
              posts.parquet   ──build──►  6 text-free   ──git──►  hydrate + fold
              (raw text, private)         aggregates              (no raw text ever)
                                                       │
                                                       ▼
                              analytics/  conviction (trailing z)
                              + the 5-check BUY/SELL signal engine
                                                       │
             pull_bloomberg_prices.py ─────────────────┤ (PX_LAST, incremental)
                                                       ▼
                              dashboard.py — every chart interactive:
                              trade desk, overlays, conviction, trends
```

**Two machines, one repository** (identical to RetailFlow1):
- The **external machine** holds the raw post store (`posts.parquet`,
  gitignored) and can rebuild every aggregate from raw text
  (`update_data.py --full`).
- The **internal machine** holds only `ABSTRACTED_DATA/` (committed, ~7 MB,
  text-free). Live posts fold straight into the aggregates; raw text never
  lands on disk there.
- Every run ends with a safety check that the committed aggregates carry no
  text-bearing columns.

> **Note on this repository's starting state:** it was seeded with the
> committed aggregates from RetailFlow1 (2017 → today). The raw post store
> and the historical backfill scripts (Pushshift dumps, HuggingFace X
> archives) deliberately stay in RetailFlow1 — this project consumes their
> abstracted output and carries the live pipeline forward from here.

## The one knob: the window

In `src/config.py` (or per run via `--start` / `--end`):

```python
START_DATE = "2021-01-01"   # inclusive
END_DATE   = ""             # "" = LIVE (to today);  "2021-11-01" = backtest window
PRICE_TOP_N = 150           # how many top-mentioned tickers the price pull covers
```

`END_DATE = ""` → **live fast path**: fetch a week of the most popular posts
(Reddit newest via Arctic Shift; X Top + Latest + broad discovery queries;
StockTwits streams), fold them in, recompute the signals, pull prices.
Minutes — and the fetch itself is parallel across the three sources.

A date → **backtest**: instant. The aggregates are **window-independent**
(built once over `BUILD_START_DATE` → today), so a backtest is just a lens —
nothing is fetched, nothing rebuilt. Command cheat-sheet:
**[RUNBOOK.md](RUNBOOK.md)**.

## Folder layout

```
retailAPOLLO/
├── update_data.py            # THE one command (window in src/config.py)
├── dashboard.py              # "GIC RetailRadar" - all charts + pipeline runner
├── pull_bloomberg_prices.py  # PX_LAST via blpapi (incremental, append-only)
├── check_live_ingestion.py   # freshness check, layer by layer
├── RUNBOOK.md                # scenario cheat-sheet
├── docs/                     # ARCHITECTURE.md + LIVE_INGESTION.md
├── ABSTRACTED_DATA/          # the ONLY committed data: 6 text-free aggregates
├── src/                      # shared logic (config, extraction, themes, sentiment)
├── ingestion/                # live fetchers + fold/merge/rebuild scripts
├── analytics/                # conviction + signals + overlay maths (was nb 08-16)
├── helper/                   # research tools (emerging terms, threshold tuning)
├── data/                     # gitignored except reference/ (processed, prices, raw)
└── tests/                    # pytest invariants for the whole pipeline
```

Key `src/` modules: `config.py` (every path + tunable in ONE place),
`abstracted_data.py` (export/hydrate + the text-free merge maths),
`extract_tickers.py` + `screen_tickers.py` + `ticker_universe.py` (ticker
extraction with data-driven word-ticker screening), `themes.py` (39
tradeable themes, each anchored to a liquid instrument), `sentiment.py`
(VADER + finance lexicon, parallel scoring, permanent id→score store),
`terms.py` (emerging-term vocabulary).

## The analytics (what replaced notebooks 08–16)

| module | replaces | what it does |
|---|---|---|
| `analytics/conviction.py` | nb 08, 09 | bull pressure → 7d roll → trailing 84d z, per ticker & theme; divergence flags, weekly heatmap + snail-trail data |
| `analytics/signals.py` | nb 10 | the 5-check BUY/SELL engine: crossing triggers, sentiment gate, score ≥ 4/5, 21d cooldown, reasons attached |
| `analytics/overlays.py` | nb 11–16 | mention share & first derivative vs price, forward-move deciles, lead/lag scan, direction flips, conviction crossings, the signal report card |
| `analytics/run_analytics.py` | nbconvert | recomputes conviction + signals (in parallel) and writes the same parquet outputs the notebooks wrote |

The dashboard renders all overlay analytics **on demand** from the saved
outputs — "refresh the overlays" is now just moving the window slider.

## The dashboard: GIC RetailRadar

The dashboard shows **themes and their anchor ETFs only** — the desk does
not trade single tickers, so the individual-ticker overlay views were
removed. The ticker analytics remain in `analytics/` for research
(windowed backtests via `run_analytics --what signals --start ... --end ...`).

`dashboard.py` presents everything as **GIC RetailRadar** (Alex Brown —
GIP 2026 Project — MAARS Global Macro). Beyond the charts, it can RUN the
pipelines itself: the sidebar buttons launch them as **background
processes** with a progress bar, a plain-English stage checklist
("Fetching new posts" → "Analysing: conviction + trade signals" → …), the
raw output tucked into a *technical log* expander (auto-opens on
failure), and a working **cancel** button that kills the whole process
tree. Notable chart behaviours, all documented in the code:

- **Masked ≠ missing**: days with under `MIN_TOTAL` total mentions are
  masked as too thin to trust; those stretches draw as a **dotted, dimmed
  bridge** labelled *"not enough posts that day"* in the legend, so a
  filled-in stretch is visibly different from real data and no values are
  invented.
- **Emerging trends** has a growth-lookback slider (3–30d): short = the
  early-warning list, long = the confirmed, sustained-build-up list.
- **Conviction** ranks by an **EWMA** of conviction z (half-life slider,
  default 10d) with the latest z shown alongside — reactive to where
  crowds are *now*, unlike a flat 30-day mean. Negative values are
  information, not errors: the z is relative to each theme's own trailing
  84-day normal.

## Counting rules (the important ones — unchanged from RetailFlow1)

- **One signal only: raw `mention_count`** — the number of distinct posts
  mentioning a ticker that day. A post mentioning NVDA five times counts
  once (breadth of attention, not verbosity). There is deliberately no
  score-based weighting: archived scores are final scores, so weighting
  day-t mentions by them leaks future information into backtests. Tests
  enforce this.
- **Dedup is a contract**: every ingestion path skips ids that already exist
  ("first seen wins"). Id prefixes (`x_`, `st_`, Reddit base36) make
  cross-source collisions impossible.
- **Word-tickers are demoted, not deleted**: symbols that are everyday words
  (EDGE, LOAN, RENT) only count when written as `$cashtags`, decided by a
  measured caps-ratio on the corpus with a wordfreq fallback
  (`src/screen_tickers.py`; the shipped `data/reference/
  ticker_classification.csv` carries the screening measured on the full
  RetailFlow1 corpus).
- **Live vs archive volumes differ hugely.** Charts therefore default to
  share-of-chatter normalisation, and z-scores use trailing baselines. The
  coverage table printed by every `update_data.py` run shows exactly what
  data exists, month by month, per source.

## Data sources

- **Reddit**: 17 finance subreddits, live via the Arctic Shift public API
  (complete per-subreddit coverage, near-real-time, no key). FetchLayer
  remains as a manual fallback (`ingestion/fetch_reddit_live.py`).
- **X (Twitter)**: live via FetchLayer — top-of-week + latest cashtag
  searches plus broad discovery queries that catch names not on any
  watchlist (the extractor finds every valid ticker in post text).
- **StockTwits**: public symbol streams, no key. Users label their own posts
  Bullish/Bearish — ground truth for calibrating the sentiment engine.
- **Bloomberg**: PX_LAST daily closes via blpapi (the prices file stays
  local and gitignored for licensing reasons; the pull is incremental —
  only missing spans are requested).

## Setup

```bash
pip install -r requirements.txt --user
```

Create `.env` in the project root with `FETCHLAYER_KEY=...` (see
`docs/LIVE_INGESTION.md` for all keys; `example.env` is the template).
For prices, install blpapi once per machine (Terminal running):

```powershell
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi --user
```

First run on a fresh machine:

```bash
python -c "from src import abstracted_data; abstracted_data.hydrate()"
python update_data.py                    # fetch + fold + signals + prices
python -m streamlit run dashboard.py     # the terminal
```

## Known limitations

- Live coverage is thinner than the archive; signals in the live era lean on
  the share normalisation and the 28-day z warm-up.
- Sentiment is lexicon-based (VADER + finance slang): robust in aggregate,
  weak on sarcasm. Upgrade path: swap a finance-tuned transformer into
  `src/sentiment.py::score_text()`; everything downstream is unchanged.
- The ticker universe is today's listing plus a curated delisted supplement
  (`src/ticker_universe.py`) — a full point-in-time universe would remove
  the residual survivorship bias.
- Mention spikes measure attention, not direction; the sentiment gate in the
  signal engine addresses this, but levels remain noisier than changes.
- Ticker-level signals use a volume floor that is a MEAN over the engine's
  input window — over the full 2017→today span almost no single name
  clears it (the archive's abstracted volumes are modest), so ticker
  signals are produced by **windowed** runs
  (`python -m analytics.run_analytics --what signals --start ... --end ...`),
  exactly how the old ticker backtests were made.
