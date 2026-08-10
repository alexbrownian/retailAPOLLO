# retailAPOLLO — the retail EUPHORIA detector (calling tops)

> **Maintaining this project?** Start with [`POST_INTERN_HANDOVER.md`](POST_INTERN_HANDOVER.md) — every editable config, the operating cadence, the AI keyword-audit workflow, and the pending items.

**THE AIM (re-set July 2026): detect retail euphoria and use it to call
price TOPS.** retailAPOLLO measures retail attention and sentiment for
**themes, hot single names and retail commodities** across **17 finance
subreddits, X (Twitter) and StockTwits**, condenses them into a 0–100
**EUPHORIA LEVEL** per instrument, and raises a **red euphoria alert**
when the crowd goes euphoric — because something has to go euphoric
before it crashes. **Prediction is Reddit-only by rule: price never
enters the euphoria level or the alert — it only defines and scores the
ground-truth tops**, so the claim stays clean: the crowd alone called
the top. Success is defined precisely: an alert inside
**[peak − 30 days, peak + 1 day]** of a genuine price top (a boom
followed by a ≥12% ETF / ≥25% single-name drawdown — bars lowered
2026-08-07 after a full-walk-forward sweep, PARAMETER_REGISTER Class
14b). Walk-forward on real Bloomberg closes (the only fitted quantity —
the alert threshold — is always learned from PAST years only): **~14% of
coverage-detectable peaks captured, median lead ~9 days before the peak,
0.18 false alarms per instrument-year** under the current ground truth.
Every rule is **ablation-tested**, and the record is re-frozen on every
re-validation event. **The DESK signals go further (2026-08-07): a
tournament-selected learned model** — a logit + monotone-GBM rank
ensemble over 9 crowd measurements + 2 price features, walk-forward,
three surviving constants — **catches 42% of detectable episode tops
(GET OUT) and 57% of starts (GET IN)**; `analytics/ml_detector.py`,
notebook 03 §SS, `docs/research/ml_tournament.md`. Full rules and
research grounding for the crowd-only baseline (attention-reversal
literature, Sornette's LPPLS bubble signature — applied to attention,
not price): `analytics/euphoria.py`.

Alongside it, the **Influence Tracker** (method: Chan, Oxford M.Eng
2026) finds the users whose calls have actually been USEFUL — a
volatility-scaled correctness bar, abnormal-return weighting, Bayesian
shrinkage, a composite usefulness score with a HIGH tier, plus a
bot-filtered reply-graph PageRank and a *loud-but-wrong* flag (the
thesis found the loudest accounts were the least accurate). The store
is **committed to git** (text-free by a hard write-time check;
pseudonymous public identifiers only) and every live pipeline run
extends it incrementally.

The earlier conviction/BUY-SELL machinery remains in `analytics/` for
research; the dashboard leads with euphoria.

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

## Where everything is documented

*One fact, one home. This table says where a thing goes — and if two
documents ever start saying the same thing, one of them is wrong.*

| Document | Answers | Does NOT hold |
|---|---|---|
| `README.md` (this file) | what the project is, what it detects, how to set it up, and where everything is | operating procedure, parameter reasoning |
| `RUNBOOK.md` | what to run, when, on which machine, and what to do when it breaks | why the system is shaped this way |
| `docs/ARCHITECTURE.md` | the shape of a run, the two machines, the AI layer, the text-free boundary, the five invariants | individual parameter values |
| `docs/PARAMETER_REGISTER.md` | every number, its class, and the evidence behind it | how to operate anything |
| `POST_INTERN_HANDOVER.md` | which files a maintainer edits, when, and what to run afterwards | architecture, rationale, open work |
| `OPEN_ITEMS.md` | what is outstanding right now — the only document expected to go out of date | anything permanent |
| `notebooks/00`–`05` | the research record, in PRESENTATION ORDER (data → ground truth → features → tournament → evaluation → influence): every claim with its evidence and its confidence intervals. Retired studies with frozen records: `notebooks/_retired_2026-08-07_presentation_refactor/` | operating instructions |
| `notebooks/07_presentation_pack` | the PRESENTATION notebook (added 2026-08-07; merged with the slide pack 2026-08-09): the eight questions the talk must answer, one PPT-ready figure each — data-quality fix, episode definition, feature bank, model menu, AUROC+ablation, tournament, thresholds+Spearman, the flip-flop fix, worked examples (gold, GME), the record vs the old rules — every figure exported to `docs/figures/slides/` at 200 dpi in the deck palette | new research (it renders the frozen records; the evidence lives in 00–06 and `docs/research/`) |
| `docs/PRESENTATION_BRIEF.md` | the complete brief for building the pitch deck, with every number written out and the figure pack in `docs/figures/deck/` | anything about how the system runs |
| `docs/NOTEBOOK_STYLE.md`, `docs/ETF_RESEARCH.md` | how the notebooks are written; the ETF constituent research behind `theme_tickers.csv` | — |

## Architecture

```
  17 subreddits ──┐
  X (Twitter) ────┤──> ingestion/fetch_all.py ──> fold (dedup: first seen wins)
  StockTwits ─────┘        (parallel)                  │
                                                       v
              EXTERNAL machine                  ABSTRACTED_DATA/          INTERNAL machine
              posts.parquet   ──build──>  6 text-free   ──git──>  hydrate + fold
              (raw text, private)         aggregates              (no raw text ever)
                                                       │
                                                       v
                              analytics/  conviction (trailing z)
                              + the 5-check BUY/SELL signal engine
                                                       │
             pull_bloomberg_prices.py ─────────────────┤ (PX_LAST, incremental)
                                                       v
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
├── dashboard.py              # "RetailRadar" - all charts + pipeline runner
├── pull_bloomberg_prices.py  # PX_LAST via blpapi (incremental, append-only)
├── check_live_ingestion.py   # freshness check, layer by layer
├── RUNBOOK.md                # scenario cheat-sheet
├── docs/                     # ARCHITECTURE, PARAMETER_REGISTER, DECISIONS.xlsx
│   ├── figures/deck/        # the pitch-deck figure pack (build_deck_figures.py)
│   ├── figures/dashboard/   # product screenshots (dashboard_shots.py)
│   └── research/            # the JSON each notebook exports
├── ABSTRACTED_DATA/          # the ONLY committed data: 6 text-free aggregates
├── src/                      # shared logic (config, extraction, themes, sentiment)
├── ingestion/                # live fetchers + fold/merge/rebuild scripts
├── analytics/                # conviction + signals + overlay maths (was nb 08-16)
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
| `analytics/conviction.py` | nb 08, 09 | bull pressure → 7d roll → **EWM-baseline trailing z** (validated on real prices; see ARCHITECTURE §6.1) with grey back-to-neutral exit points; divergence flags, heatmap + snail-trail data |
| `analytics/signals.py` | nb 10 | the 5-check BUY/SELL engine: crossing triggers, sentiment gate, score ≥ 4/5, 21d cooldown, reasons attached |
| `analytics/overlays.py` | nb 11–16 | mention share & first derivative vs price, forward-move deciles, lead/lag scan, direction flips, conviction crossings, the signal report card |
| `analytics/run_analytics.py` | nbconvert | recomputes conviction + signals + **euphoria** + the **influence** live update (in parallel) and writes the parquet outputs |
| `analytics/euphoria.py` | *(new aim)* | the top detector: 4 Reddit-only percentile rules + fade trigger → euphoria level, price-defined ground-truth peaks, walk-forward validation + ablation + ML challenger |
| `analytics/influence.py` | *(new)* | the Influence Tracker: volatility-judged calls, composite usefulness scores (thesis method), reply-graph PageRank, loud-but-wrong flag — committed text-free store, extended live |
| `analytics/euphoria_phases.py` | *(new, July 2026)* | the **phases study**: episode ground truth (trough→peak→bust), the onset feature bank, the walk-forward tournament machinery, and the LIVE desk detector feeding the dashboard's Start/End radar (since 2026-08-07 the shipped desk model is selected by the tournament in `analytics/ml_detector.py` — see below) |
| `analytics/ml_detector.py` | *(new, Aug 2026)* | the **learned desk detectors**: one 11-feature bank (9 crowd measurements + the price pair), four model families (logistic / monotone GBM / MLP / logit+GBM rank ensemble) raced walk-forward against the incumbent rules under a pre-stated criterion (one family for both heads, combined test-AP lift). The winner replaces the gate stack live; only three constants survive — the coverage floor, the 21d cooldown, and a probability cut chosen on past years by F1 |
| `analytics/robust_share.py` | *(new, Aug 2026)* | the **coverage-robust share estimator** behind every attention series: trailing-window ratio-of-sums + per-source stratification + empirical-Bayes shrinkage toward each name's own baseline. Fixes the fake zero-share days and the source-mix dilution documented in `docs/research/DATA_QUALITY_2026-08.md` |
| `analytics/influence_ml.py` | *(new, July 2026)* | the influential-users MODEL (thesis ch. 6 port): can HIGH-tier authors be identified from behaviour + graph position alone? MLP / LabelProp / GraphSAGE-lite vs random, run by notebook 05 on the live store |

**Research notebooks — THE PRESENTATION PIPELINE (reshaped 2026-08-07:
the notebook set now mirrors the final presentation, chapter for
chapter).** Each imports the SAME modules the pipeline runs (a
drift-guard assert enforces it) and re-executes end-to-end from current
data; no ordering constraints between them:

* **00 — the data, and the estimator it deserves**: what the committed
  stores hold, the coverage regimes (Reddit 2017→, StockTwits Feb-26→,
  X Jul-26→), and the coverage-robust share estimator that fixed the
  fake zero-share days (308 → 31 on the top-30 audit; method in
  `analytics/robust_share.py`, full write-up in
  `docs/research/DATA_QUALITY_2026-08.md`).
* **01 — the episode ground truth**: the price-only exam paper (494
  boom→bust arcs under the swept 20/40 boom / 12/25 bust bars; 199
  gradeable starts, 219 gradeable tops).
* **02 — the feature battery**: per-feature evidence for the crowd bank,
  the `source_breadth` rejection, and the §SS August extension
  (`bull_level`, `bull_persist`, the price pair).
* **03 — the model tournament**: the July race (rules won under the old
  frame — kept as provenance) and **§SS, the August selection that
  ships**: logistic / monotone GBM / MLP / ensemble, walk-forward,
  pre-stated criterion, winner = the **logit+GBM rank ensemble**.
* **04 — the deep evaluation**: every constant of the RULES system swept
  and justified — the record of what the August model replaced, and
  still the live evaluation of the crowd-only research detectors.
* **05 — the influential-users model** (thesis ch. 6 port).
* **06 — the worked example: gold, 2025–26** — the whole machine on one
  name: the price-only exam, the raw store rows, the 11 measurements
  through the episode, the ensemble's probability crossing its cut days
  before the 29 Jan 2026 peak, the graded outcome, and the same call on
  the dashboard (figures in `docs/figures/06/`).
* **07 — THE PRESENTATION PACK**: the eight questions the talk must
  answer, one PPT-ready figure each, exported to `docs/figures/slides/`
  in the deck palette (blue / dark blue / grey) — data fix, episode
  definition + sweep, feature bank, model menu, feature AUROC +
  ablation, tournament, thresholds + Spearman + the flip-flop fix,
  worked examples (gold, GME), the record vs the old rules, event
  study, decision log. (The pre-merge figure set survives in
  `docs/figures/07/`.)

Six studies were retired to
`notebooks/_retired_2026-08-07_presentation_refactor/` (files intact,
frozen JSONs still live and still read by the dashboard): the July
method walkthrough (00), strictness (06), index composite (07),
single-state (08), agentic watch (09) and AI sentiment (10 — verdict
still pending a desk-machine run; resurrect from the retirement folder).
That folder's README records why each left the live set.

Headline (walk-forward, the adopted desk model, ground truth 20/40 /
12/25): **GET OUT catches 42% of detectable episode tops** (incumbent
rules: 12%) at 0.35 false alarms per instrument-year with a 16-day
median lead and a flat-to-negative median forward path; **GET IN catches
57% of detectable starts** (incumbent: 9%) at 0.24 FA/instr-yr. Full
per-model table: `docs/research/ml_tournament.md`. The earlier
crowd-only claims and the rejected trading translation remain recorded
in notebook 04 and the frozen JSONs.

The dashboard renders all overlay analytics **on demand** from the saved
outputs — "refresh the overlays" is now just moving the window slider.

## The dashboard: RetailRadar

The dashboard shows **themes and their anchor ETFs only** — the desk does
not trade single tickers, so the individual-ticker overlay views were
removed. The ticker analytics remain in `analytics/` for research
(windowed backtests via `run_analytics --what signals --start ... --end ...`).

`dashboard.py` presents everything as **RetailRadar** (Alex Brown —
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
- **Top/Emerging trends** carry an attention TOGGLE (desk 2026-08-07):
  the attention line as the LEVEL (coverage-robust share of chatter) or
  its FIRST DERIVATIVE (smoothed pp/day — is the crowd arriving or
  leaving, regardless of how big it already is).
- **Tabs removed 2026-08-07** (desk instruction): *Overlays: themes*,
  *Conviction* and *Historical checker*. The conviction ENGINE
  (`analytics/conviction.py`) still runs in the pipeline and its store
  is still written — only the display surface went. **Added:** *[dev]
  Data Stats* — the live snapshot of the data behind everything (store
  freshness, per-source volumes, ingestion ledgers, raw-archive
  inventory, the shipped signal engine).
- **Tradeable universe only**: every theme list, ranking and picker on
  the dashboard is restricted to themes with a firm-approved instrument
  (`THEME_ETFS`); non-tradeable themes stay in the data but off the desk.

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

Create `.env` in the project root with `FETCHLAYER_KEY=...` — the single
`.env` holds every credential, including the Apollo LLM auth.
`POST_INTERN_HANDOVER.md` §1 is the authoritative key list. It is
git-ignored and has no template, so keep a private copy somewhere safe.
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
