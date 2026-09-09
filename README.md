# retailAPOLLO

Retail-attention and sentiment monitoring for tradeable themes and single
names. The pipeline reads public finance forums (Reddit, StockTwits, X),
reduces every post to text-free daily counts and sentiment, and scores
each instrument against a frozen, walk-forward-validated model that
raises **INCREASE EXPOSURE** / **CUT EXPOSURE** calls around the start
and the top of retail manias. A Streamlit dashboard renders the result.

Every figure the dashboard quotes is walk-forward: thresholds are fitted
on strictly earlier years and scored blind on later ones. Success is
defined before the data is looked at (a CUT EXPOSURE call inside
[peak − 30 days, peak + 1 day] of a price-defined top), and the record
that supports each number ships with the code in
`reference/research_record/`.

## Headline results

Walk-forward test years, production model (`reference/KEY_PARAMETERS.md`
explains every term):

| Signal | Episodes caught | Median lead | AUROC | AP vs base rate |
|---|---|---|---|---|
| INCREASE EXPOSURE (episode starts) | 57% | 19 days | 0.73 | 2.5× |
| CUT EXPOSURE (episode tops) | 44% | 14 days | 0.73 | 2.7× |

The model is a rank ensemble of a logistic regression and a monotone
gradient-boosting classifier over nine crowd features and two price
features, chosen by a tournament under a rule stated before the numbers
were computed. A crowd-only variant (no price feature anywhere) is kept
as a research record: AUROC ≈ 0.57, which is what the crowd alone can
see.

## Quick start

```bash
pip install -r requirements.txt
python tools/preflight.py                 # environment, configs, data
python update_data.py                     # fetch, fold, score, pull prices
python -m streamlit run dashboard.py
```

Prices come from a Bloomberg Terminal when one is running and from Yahoo
Finance otherwise (`--provider` overrides; see `RUNBOOK.md`). No other
credential is required for a first run: Reddit and StockTwits are read
without keys, and the AI panels and X fetcher skip cleanly when their
keys are absent.

A copy without the raw post store (`--mode aggregates`) folds live posts
into the committed text-free aggregates and runs everything else
unchanged. That is the normal state of a fresh clone.

## Repository layout

Everything the pipeline needs to run is committed. `research/` is a
git-ignored working folder and `presentations/` holds material about
the project; the pipeline never reads from either.

| Path | Contents |
|---|---|
| `dashboard.py` | The Streamlit application. |
| `update_data.py` | The pipeline entry point: one command refreshes everything. |
| `pull_prices.py` | Price refresh (Bloomberg or Yahoo Finance) for the approved instruments. |
| `config/` | Everything a user edits without touching Python: forums, themes, keywords, tickers, instruments, settings. All CSV; `config/README.md` documents each file. |
| `src/` | Configuration constants, settings loader, text extraction, sentiment, aggregation, price providers. |
| `ingestion/` | Source fetchers, the bot screen, the live fold, the forum panel review. |
| `analytics/` | Episode ground truth, feature bank, detectors, walk-forward machinery, influence tracker. |
| `tools/` | Operations: preflight, config validation, data health, backfill, historical fold, dependency check, bot-screen report, dashboard publish. |
| `tests/` | The test suite (invariants, no-lookahead, display contracts, production hygiene). |
| `reference/` | `KEY_PARAMETERS.md` (every parameter, its test and its rationale) and `research_record/` (the frozen JSON evidence the dashboard quotes). |
| `docs/` | `ARCHITECTURE.md`, `DATA.md`, `CONFIGURATION.md`, `LIVE_INGESTION.md`. |
| `ABSTRACTED_DATA/` | The committed text-free daily aggregates. |
| `DASHBOARD_DATA/` | The published display bundle a hosted dashboard renders. |
| `data/` | Git-ignored runtime data: raw fetches, working stores, prices, ledgers. |
| `research/` | Git-ignored: notebooks, exploratory data, figures, sweeps. Optional. |
| `presentations/` | Slide decks, one-pagers and demo material. Not read by any code. |

## How it works in one paragraph

Each run fetches the trailing week from every enabled forum
(`config/forums.csv`), screens out automated and duplicated posts
(`ingestion/bot_screen.py`), and reduces the rest to daily counts and
sentiment per theme and per ticker. Themes are defined by keyword and
ticker lists in `config/`; each theme is represented by an ETF whose
price defines the ground truth. Analytics turn the aggregates into
trailing features (attention level, acceleration, hype ratio,
bullishness level, persistence, inflection, plus two price features),
score them with the frozen model, and write a per-instrument state the
dashboard reads. Nothing in a live run re-fits: model, thresholds and
feature definitions are frozen by a research pass and pinned by tests.

## Three rules

1. **A live run never re-selects.** Models and thresholds are frozen by an
   explicit research pass; live runs only score.
2. **Research re-opens deliberately.** `analytics.run_analytics --research`
   or `update_data.py --full`. A backfill is a re-validation event because
   it rewrites the history the thresholds were fitted on.
3. **No lookahead.** Every feature is a trailing window; staleness is
   reported, never silently repaired.

## Where to read next

| Question | Document |
|---|---|
| How do I run, schedule, publish or recover it? | `RUNBOOK.md` |
| How is it put together and what may never change? | `docs/ARCHITECTURE.md` |
| What is in each data file and where does it live? | `docs/DATA.md` |
| How do I change forums, themes, tickers, settings? | `docs/CONFIGURATION.md`, `config/README.md` |
| Which sources are read, with what keys and limits? | `docs/LIVE_INGESTION.md` |
| Why is each parameter the value it is? | `reference/KEY_PARAMETERS.md` |
