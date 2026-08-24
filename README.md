# retailAPOLLO

Retail-euphoria detection for tradeable themes and single names: the
system measures retail attention and sentiment across 17 finance
subreddits, X, and StockTwits, condenses them into per-instrument
scores, and raises **GET IN** / **GET OUT** calls around the start and
top of retail manias. Success is defined precisely — a GET OUT inside
[peak − 30 days, peak + 1 day] of a price-defined top — and every
quoted number is walk-forward: thresholds are fitted on strictly
earlier years and scored blind.

## Headline results (walk-forward, frozen record)

| Signal | Capture (of detectable) | Median lead | AUROC | AP vs base |
|---|---|---|---|---|
| GET IN (episode starts) | 57% | 18–19 days | 0.73 | 2.5× |
| GET OUT (episode tops) | 44% | 14 days | 0.73 | 2.7× |

The production model is a logistic-regression + monotone gradient
boosting rank ensemble over nine crowd measurements and two price
features, selected by tournament under a pre-stated criterion and
pinned (`docs/DECISIONS.md`). A crowd-only variant (no price anywhere)
is maintained as a research record: AUROC ~0.57, quantifying what the
crowd alone can see.

An **Influence Tracker** (method: Chan, Oxford M.Eng 2026) scores
individual authors by the measured usefulness of their calls
(volatility-scaled judging, Bayesian shrinkage, reply-graph PageRank,
a loud-but-wrong flag), extended incrementally by every run.

## Architecture

Two machine roles share one codebase; `update_data.py` auto-detects
which it is on:

- **External**: holds raw post text (`data/processed/posts.parquet`)
  and can rebuild every aggregate from scratch.
- **Internal**: holds only `ABSTRACTED_DATA/` — six text-free daily
  aggregate tables (counts and sentiment; no post text) — and folds
  live posts into them incrementally.

The text-free boundary is enforced mechanically: a `FORBIDDEN_COLS`
check refuses to bless any committed aggregate containing raw-text
columns. Price data (Bloomberg daily closes) exists to define and score
ground truth and to render charts.

```
fetchers (Reddit / X / StockTwits)          Bloomberg (Terminal)
        │                                          │
        ▼                                          ▼
data/raw/*.jsonl.zst ──fold──► ABSTRACTED_DATA/*.parquet   data/prices/
        (dedup: uncapped seen-id set; text-free guard)          │
                             │                                  │
                             ▼                                  │
              analytics/  (episodes, walk-forward,  ◄───────────┘
               frozen thresholds, euphoria_desk.parquet)
                             │
                             ▼
                     dashboard.py  (Streamlit)
```

## Quick start

```bash
pip install -r requirements.txt --user
python update_data.py                    # fetch, fold, score, pull prices
python -m streamlit run dashboard.py    # open the terminal
```

`tools/preflight.py` answers "is this project still sound?" in one
command; `tools/data_health.py` (read-only) reports data freshness,
coverage, and ledger integrity.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Configuration, extraction, sentiment, aggregation primitives. |
| `ingestion/` | Source fetchers, the live fold, the subreddit panel review. |
| `analytics/` | Episode ground truth, detectors, walk-forward machinery, influence tracker. |
| `tools/` | Operational tools: preflight, data health, backfill, historical fold, dependency check, operating-point sweep. |
| `config/` | Editable universe definitions (themes, keywords, tickers, ETFs) — all CSV. |
| `tests/` | The fence suite (~220 tests): invariants, no-lookahead, display contracts. |
| `notebooks/` | The research record (jupytext-paired `.py`/`.ipynb`), 00–08. |
| `docs/` | `ARCHITECTURE.md`, `DECISIONS.md`, `docs/RESEARCH_RECORD.md`, `docs/RESEARCH_RECORD.md`, evidence JSONs in `docs/research/`. |
| `ABSTRACTED_DATA/` | The committed text-free aggregates. |
| `data/` | Gitignored runtime data: raw fetches, working stores, prices, ledgers. |

## Operating discipline

Three rules govern every number the system shows (full statement:
`docs/RESEARCH_RECORD.md`):

1. **A live run never re-selects.** Models and thresholds are frozen by
   an explicit research pass; live runs only score.
2. **Research re-opens deliberately**: `--research`, or
   `update_data.py --full`. A backfill is a re-validation event because
   it rewrites the history thresholds were fitted on.
3. **No lookahead.** Trailing windows everywhere; staleness is reported,
   not silently repaired.

See `RUNBOOK.md` for day-to-day operations and `docs/DECISIONS.md` for
the dated decision log with evidence.
