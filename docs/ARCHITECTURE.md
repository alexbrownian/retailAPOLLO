# Architecture

How retailAPOLLO is put together, and which properties may never change.
This page holds the shape of the system and its invariants. Operating
commands are in `RUNBOOK.md`; file formats in `docs/DATA.md`; parameter
rationale in `reference/KEY_PARAMETERS.md`.

## 1. One run

`python update_data.py` is the whole system. Everything else — the
dashboard, the tools, the optional research notebooks — reads what it
leaves on disk.

```
 1. FETCH      every enabled source in parallel (Reddit posts and comments,
               StockTwits, X); a heartbeat keeps a rate-limited pull visible
 2. SCREEN     ingestion/bot_screen.py drops automated and duplicated posts
 3. STORE      full mode: append to posts.parquet
               aggregates mode: aggregate the new posts and fold text-free
               deltas into ABSTRACTED_DATA/
 4. COVERAGE   is the window readable? reported, never silently patched
 5. ANALYSE    features -> frozen model -> per-instrument state; influence board
 5a. PRICES    Bloomberg or Yahoo Finance for the approved instruments
 5b. AI LAYER  optional: agentic scan, poll, pulse (section 5)
 6. SAFETY     verify_abstracted: nothing with text can be committed
 7. PUBLISH    stage DASHBOARD_DATA/ for a hosted copy
 8. SUMMARY    the run's key facts in one glance
```

A full recompute takes seconds because nothing downstream re-reads text:
keyword and ticker extraction happens once, at ingestion, and every
analysis is an importable module operating on wide daily matrices stored
as typed parquet.

## 2. Two storage modes, one contract

| | `full` mode | `aggregates` mode |
|---|---|---|
| Holds | `data/processed/posts.parquet` and the raw `.jsonl.zst` archives | the committed aggregates only |
| Can | rebuild every aggregate from text; run `--full` | fold live posts incrementally |
| Produces | the same aggregate tables | the same aggregate tables |

The contract between them is `ABSTRACTED_DATA/`: **counts and
sentiment only, never text, never identities**. `FORBIDDEN_COLS` in
`src/config.py` names the columns that may never appear in a committed
frame, and `verify_abstracted` in `update_data.py` enforces it on every
run. `tools/publish_dashboard.py` applies the same rule to the display
bundle.

## 3. Ingestion

### 3.1 Sources and the forum panel

`config/forums.csv` is the panel: one row per forum with an `enabled`
flag. Every reader (`fetch_reddit_live.py`, `fetch_reddit_comments.py`,
`fetch_reddit_arctic.py`, `tools/backfill_reddit.py`,
`tools/fold_historical.py`) loads the panel through
`src.settings.load_forums()`, so adding a forum is a one-row edit. A
monthly review (`ingestion/discover_subreddits.py`) may append at most
one forum per review, in the `exploration` tier, when enough referrals
from the existing panel and a sampled ticker-mention rate justify it.

### 3.2 The fetch budget

The design cadence is twice a week with a full refresh under roughly ten
minutes; `PIPELINE_BUDGET_S = 600` is that constraint as a number the
code can enforce. `src/pipeline_budget.py` allocates the ceiling across
sources using two machine-local ledgers (`pipeline_stage_times.json`,
`reddit_comments_cost.json`) that are git-ignored on purpose: they
measure this machine, and committing them would plan one machine's run
with another's numbers.

Running short is a deferral, not data loss. The crawl walks newest-first
and a forum's watermark advances only over ground a run fully covered,
so a forum that hits its cap resumes exactly where it stopped. Every
deferral is printed. Nothing in the budget touches a signal: it decides
how much a run fetches, never how anything is scored.

### 3.3 The bot screen

`ingestion/bot_screen.py` scores every post in `[0, 1]` from five
deterministic signals — near-duplicate text (MinHash LSH over word
3-shingles), posting bursts, self-disclosed bots, low text diversity per
author, and a deleted author — and both aggregation entry points drop
posts at or above `bot_screen_threshold` before anything is counted.
Excluded posts stay in the raw files and are recorded as seen, so the
decision is reversible and never re-judged. The rate is printed on every
run and stored in `data/reference/bot_screen_last.json`.

### 3.4 Dedup

Aggregate merges are additive and carry no post ids, so the seen-id set
(`data/reference/abstracted_seen_ids.parquet`, uncapped, written
atomically) is the only guard against permanent double counting. It is
snapshotted after every successful fold.

## 4. Analytics

### 4.1 Ground truth

An episode is defined from price alone, before any crowd data is
consulted: a run-up of at least `EUPHORIA_BOOM_MIN_*` over the trailing
`EUPHORIA_BOOM_LOOKBACK_D` days into a local peak, followed by a
drawdown of at least `EUPHORIA_CRASH_MIN_*` within
`EUPHORIA_CRASH_WINDOW_D` days. ETFs and single names have different
bars because their unconditional volatility differs. The full
definition, the sweep that chose the bars and the resulting episode
counts are in `reference/KEY_PARAMETERS.md`.

### 4.2 Features and the model

Nine crowd features (attention level, one-month and two-week change,
week-vs-month acceleration, hype ratio, attention convexity, bullishness
level, bullishness persistence, mood inflection) and two price features
(run-up, 21-day return) are computed as trailing windows only. The
production model is a rank ensemble of a logistic regression and a
monotone gradient-boosting classifier, one family for both heads
(INCREASE EXPOSURE at episode starts, CUT EXPOSURE at tops), chosen by a
walk-forward tournament under a rule stated before the numbers were
computed. Each head's threshold is the score percentile that meets a
false-alarm budget on the training years only.

### 4.3 Walk-forward

For each test year the model is fitted on strictly earlier years and
scored blind; no number the dashboard quotes was computed on the year it
is scored against. The record of that procedure —
`data/processed/euphoria_desk_report.json` and the JSON files in
`reference/research_record/` — is what the dashboard reads. It never
recomputes a headline figure.

### 4.4 Frozen means frozen

A live run never re-selects: it does not choose a model, re-fit a
threshold or re-run the walk-forward. Research decides once and the
answer is stored. The reason is traceability rather than speed: a
threshold re-fitted on every run cannot be reconstructed, and a
threshold nobody can reconstruct cannot be defended. Research re-opens
through two typed commands only (`analytics.run_analytics --research`,
`update_data.py --full`) and automatically on year rollover or a
missing record.

`tests/test_production_hygiene.py` pins the ground-truth constants; an
edit to `src/config.py` fails the suite until the research record is
re-run alongside it.

## 5. The AI layer (optional)

`src/ai.py` is the only module that talks to a language model. Two
providers sit behind one interface (an OpenAI-compatible gateway via
`dimsum_lite`, or the Anthropic API); `AI_PROVIDER` selects and `auto`
falls through. Consumers: `analytics/ai_pulse.py` (a written market
read), `analytics/ai_poll.py` (a fixed question set), the keyword-map
auditor, and the agentic-watch scanner (`src/agentic_watch.py`, plain
regex, needs no model).

Two rules shape every AI feature:

* **Numbers come from the stores; words come from the model.** The model
  is handed an evidence pack of already-computed numbers and the pack is
  saved beside the prose, so any sentence can be audited against its
  inputs.
* **Approve-then-apply.** Where a model proposes a config change (the
  keyword auditor), it writes to a CSV with an empty `approved` column
  and cannot touch `config/`; a person approves and runs the apply step.

With no provider configured, `available()` is False and every consumer
returns a reason instead of raising. An AI stage has never been allowed
to fail a run.

## 6. The dashboard

`dashboard.py` reads the stores and the frozen record and renders them.
It has no mode selectors; the production configuration is fixed in code.
Branding, the pipeline buttons and the bot-screen settings come from
`config/settings.csv`; a git-ignored `config/settings.local.csv`
overrides any key on one machine. A hosted copy renders
`DASHBOARD_DATA/` and can neither fetch nor recompute.

## 7. Configuration layers

| Layer | Where | Who changes it | Effect |
|---|---|---|---|
| Universe and vocabulary | `config/*.csv` | any user, in a spreadsheet | next run |
| Settings | `config/settings.csv` (+ `settings.local.csv`) | any user | next run or restart |
| Frozen constants | `src/config.py` | a research pass | re-validation |
| Credentials | `.env` | the machine's owner | next run |

`tools/validate_config.py` checks every CSV and the references between
them; preflight and the test suite run the same check.

## 8. Invariants

1. **Frozen means frozen.** Any change to scoring code or constants is a
   re-validation event: re-run the research pass and compare the stored
   record. Nothing ships as live flags unless it beats the current
   record under the pre-stated rule.
2. **Committed data is text-free.** Section 2. Never weaken it.
3. **The dashboard shows conclusions; the record is the evidence.**
   Performance claims live in `reference/research_record/` with their
   confidence intervals, and the dashboard quotes them.
4. **Every constant carries its provenance.** `reference/KEY_PARAMETERS.md`
   has a row for each.
5. **AI writes words, never numbers, and never edits config.** Section 5.
6. **The pipeline never depends on `research/`.** The folder is
   git-ignored and optional; a test fails if a shipped module imports
   from it.
