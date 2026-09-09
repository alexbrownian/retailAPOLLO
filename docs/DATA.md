# Data

Every file the pipeline reads or writes, where it lives, and whether it
is committed. Three folders hold data:

| Folder | Committed | Contents |
|---|---|---|
| `ABSTRACTED_DATA/` | yes | The text-free daily aggregates. The only data folder in the repository. |
| `DASHBOARD_DATA/` | yes | The display bundle a hosted dashboard renders (aggregates, prices, model outputs, frozen record). |
| `data/` | no | Runtime data: raw fetches, working stores, prices, ledgers, caches. |

`reference/research_record/` (committed) holds the JSON evidence the
dashboard quotes; it is documentation rather than data and is described
in `reference/KEY_PARAMETERS.md`.

## 1. The text-free boundary

Committed data carries **no post text, no authors, no post ids and no
forum names**. `FORBIDDEN_COLS` in `src/config.py` lists the columns
that may never appear in a committed frame; `verify_abstracted`
(`update_data.py`) checks `ABSTRACTED_DATA/` on every run and
`tools/publish_dashboard.py` checks `DASHBOARD_DATA/` before staging.
No individual post can be reconstructed from anything in the repository.

## 2. `ABSTRACTED_DATA/` — committed aggregates

| File | Columns | Grain |
|---|---|---|
| `daily_ticker_counts.parquet` | `date, ticker, mention_count` | one row per ticker-day with at least one mention |
| `daily_ticker_counts_by_source.parquet` | `date, ticker, source, mention_count` | as above, split by `reddit` / `x` / `stocktwits` |
| `daily_ticker_sentiment.parquet` | `date, ticker, n_posts, avg_sentiment, net_bullish` | mean sentiment in [−1, 1] and bullish-minus-bearish share |
| `daily_theme_counts.parquet` | `date, theme, mention_count` | one row per theme-day |
| `daily_theme_sentiment.parquet` | `date, theme, n_posts, avg_sentiment, net_bullish` | as above |
| `daily_term_counts.parquet` | `date, term, mention_count` | vocabulary counts with rolling retention |

Merges are additive: counts add, sentiment means recombine weighted by
`n_posts`. History is never revised, only extended; the seen-id ledger
(section 5) is what prevents a post from being added twice.

`src/abstracted_data.py` owns the folder. `export()` copies the files
from `data/processed/` after a full-mode rebuild; `hydrate()` copies
them back into `data/processed/` on an aggregates-mode copy;
`aggregate_posts()` and `merge_into_abstracted()` implement the live
fold.

## 3. `data/` — runtime (git-ignored)

### 3.1 `data/raw/`

Raw fetches as `.jsonl.zst`, one folder per source (`RedditLive`,
`RedditComments`, `StockTwits`, `X`). Transient: nothing downstream
reads them after the fold, but they are the audit trail for the bot
screen and the input to a full-mode rebuild.

### 3.2 `data/processed/`

Working stores. The aggregates listed in section 2 are mirrored here,
plus:

| File | Columns | Written by |
|---|---|---|
| `posts.parquet` | `id, date, title, selftext, author, subreddit, source, score, …` | full mode only: the raw post store. Its presence is what selects full mode. |
| `daily_ticker_counts_by_subreddit.parquet` | `date, subreddit, ticker, mention_count` | full mode only; feeds the forum panel review. Never committed. |
| `daily_theme_conviction.parquet` | `date, theme, conviction_z` | `analytics.run_analytics` |
| `daily_agentic_counts.parquet` | `date, category, theme, mention_count` | `src/agentic_watch.py` |
| `episodes.parquet` | `name, symbol, kind, trough, peak, bust_date, boom_pct, bust_pct, run_days, onset_lo, onset_hi, year, onset_detectable, top_detectable` | ground truth (`analytics/euphoria_phases.py`) |
| `phase_day_frame.parquet` | `date, name, kind, year` + the crowd features + `y_onset, y_late, y_top` labels | the candidate-day frame the model is fitted and scored on |
| `euphoria_desk.parquet` | `date, name, kind, symbol, in_score, out_score, get_in, get_out, *_strict, *_xp, boom_state, end_stage, inflection_score, inflection, retail_flow, …` | the per-instrument daily state the dashboard reads |
| `euphoria_levels.parquet` | `date, name, symbol, kind, level, e1, e2, e3, e5, fade, hype_ok, alert` | attention levels and the hand-rule baseline |
| `euphoria_onset.parquet` | `date, name, kind, onset_score, alert` + features | the onset head's scores |
| `trade_signals.parquet`, `trade_signals_tickers.parquet` | `signal_date, action_date, action, theme, etf, score, att_z, conv_z, sent_5d_chg, reason` | the conviction-based signal layer |
| `signal_snapshots/<date>_trade_signals.parquet` | as above | one snapshot per run; the forward out-of-sample record. Never rewritten. |
| `euphoria_desk_report.json` | model, selection rule, tournament table, per-year thresholds, headline metrics | the frozen production record the dashboard quotes |
| `euphoria_report.json`, `euphoria_onset_report.json`, `desk_model_insight.json` | supporting records | analytics |
| `readiness_alerts.json`, `ai_pulse.json`, `publish_manifest.json` | dashboard panels and the publish stamp | analytics, `src/ai.py`, `tools/publish_dashboard.py` |

`euphoria_desk.parquet` and `euphoria_desk_report.json` keep their file
names for compatibility with stored snapshots; "desk" in a file name is
historical and carries no meaning.

### 3.3 `data/prices/prices.parquet`

`date, symbol, px_last, source`. Daily closes for every symbol in
`config/approved_instruments.csv` plus the tracked single names. One
source per symbol at a time (`bloomberg` or `yfinance`); a store without
the `source` column is read as all-Bloomberg. Written by
`pull_prices.py`; `tools/data_health.py` warns on single-session moves
that look like unadjusted corporate actions.

### 3.4 `data/reference/` — ledgers and caches

| File | Purpose |
|---|---|
| `abstracted_seen_ids.parquet` | Uncapped set of post ids already folded (aggregates mode). The only guard against double counting; written atomically. |
| `abstracted_live_meta.json` | Fold metadata: `LIVE_START`, per-file skip ledger. |
| `historical_fold_ledger.json` | Per-(file, month) record of historical folds. |
| `reddit_arctic_watermark.json`, `reddit_comments_watermark.json`, `reddit_*_seen.json` | Crawl watermarks and seen sets per source. |
| `reddit_backfill_progress.json` | Resumable backfill state, keyed by window and forum set. |
| `pipeline_stage_times.json`, `reddit_comments_cost.json` | Machine-local timing ledgers for the fetch budget. |
| `bot_screen_last.json` | The last bot-screen summary (rows in, excluded, rate, top reasons). |
| `subreddit_panel.json`, `panel_review_watermark.json`, `panel_review_latest.md`, `subreddit_referrals.parquet` | Forum panel review state and its last report. |
| `keyword_suggestions/` | Proposals from the keyword auditor awaiting approval. Never applied automatically. |
| `influence/` | The influence board's incremental state. |
| `nasdaqlisted.txt`, `otherlisted.txt` | Exchange symbol lists used by the ticker extractor. |
| `_backups/` | The last seven snapshots of this folder, taken after each successful fold. |

### 3.5 `logs/`

One log per run. Git-ignored.

## 4. `DASHBOARD_DATA/` — the display bundle

A copy of the frames a dashboard needs, staged by
`tools/publish_dashboard.py` at the end of every refresh: the aggregates,
`prices.parquet`, the model outputs, the frozen record, the AI panels,
the symbol lists and the signal snapshots. Every frame passes the
text-free check before it is staged. The influence board is opt-in
(`--with-influence`) because it is keyed by author handle. On first
load a hosted dashboard copies the bundle into `data/processed/`; it
never recomputes.

## 5. Retention and recovery

Aggregates are permanent and only ever extended. Raw fetches can be
deleted once folded (full mode keeps `posts.parquet` as the archive).
`data/reference/` is the state that must survive: restore the newest
`_backups/` snapshot if a ledger is lost, then run
`tools/data_health.py`, whose double-count scan confirms the aggregates
are consistent with the ledgers.
