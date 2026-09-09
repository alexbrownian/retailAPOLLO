# Live ingestion

Which sources are read, with what credentials, at what rate, and where
the posts go. The forum list itself is `config/forums.csv`
(`docs/CONFIGURATION.md`).

## 1. Credentials

Credentials go in `.env` at the project root (git-ignored; copy
`example.env`). Scripts parse `.env` directly with `os.environ` as a
fallback, so any value can also be supplied as an environment variable.

    python ingestion/fetch_all.py --check     # which sources will run; prints no secrets

| Credential | Used by | Notes |
|---|---|---|
| none | `fetch_reddit_arctic.py` (Reddit posts and comments) | The default Reddit source is the Arctic Shift public API: no key. |
| none | `fetch_stocktwits.py` | Public read-only symbol streams. |
| `FETCHLAYER_KEY` | `fetch_x_live.py`; optional Reddit path in `fetch_reddit_live.py` | fetchlayer.dev; one credit per request. `python ingestion/test_fetchlayer.py` spends one credit to verify the key. |
| `X_BEARER_TOKEN` | `fetch_x_live.py` (official v2 API fallback) | Paid tier; used only when no FetchLayer key is present. |
| `REDDIT_*` | `fetch_reddit_live.py` (official OAuth fallback) | A *script* app at old.reddit.com/prefs/apps. Unused while Arctic Shift is the source. |

A machine with no credentials still runs: Reddit and StockTwits need
none, X is skipped, and the pipeline recomputes from the aggregates on
disk.

## 2. What each fetcher pulls

`ingestion/fetch_all.py` runs every enabled fetcher concurrently, so a
full fetch takes as long as the slowest source rather than the sum. Each
fetcher writes raw `.jsonl.zst` files into its own folder under
`data/raw/`.

**Reddit posts** (`fetch_reddit_arctic.py`). Per-forum crawl over the
trailing `FETCH_LOOKBACK_DAYS` (7) with a watermark per forum. The crawl
walks newest-first; the watermark advances only over ground a run fully
covered, so an interrupted forum re-covers its window next run. Per
forum the run budget is enforced by `src/pipeline_budget.py`.

**Reddit comments** (`fetch_reddit_comments.py`). Comments feed the
influence board (who said what, how it was received) and the reply
graph. They are the slow species, so their cost is budgeted against
`PIPELINE_BUDGET_S` rather than switched off; `--skip-comments` on
`update_data.py` omits them for one run.

**StockTwits** (`fetch_stocktwits.py`). Public symbol streams for the
approved instruments and the tracked single names. About 200 requests
per hour per IP is the practical cap; the fetcher stops early on HTTP
429. Authors label their own posts Bullish/Bearish; those labels are
kept in the raw files as calibration ground truth for the sentiment
engine.

**X** (`fetch_x_live.py`). A broad discovery pass first (top finance
chatter of the week with engagement floors, so tickers on nobody's
watchlist are still found), then targeted cashtag chunks over the theme
anchors in both Top and Latest products. Backs off on HTTP 429 and
stops immediately on 402 (out of credits).

## 3. From raw files to aggregates

1. **Screen.** `ingestion/bot_screen.py` scores every new post and drops
   those at or above `bot_screen_threshold` (`docs/ARCHITECTURE.md` §3.3).
2. **Extract.** `src/themes.py` and `src/extract_tickers.py` find themes (keyword lists) and
   tickers (`$CASHTAG`, bare capitals on the allow-list, never a
   stop-listed word) once per post; nothing downstream re-reads text.
3. **Score.** `src/sentiment.py` gives each post a sentiment in [−1, 1].
4. **Aggregate and fold.** The destination depends on the mode:
   * full mode: `merge_live.py` appends into `posts.parquet` (first seen
     wins) and `build_aggregates.py` rebuilds the aggregate tables;
   * aggregates mode: `append_live_abstracted.py` aggregates the new posts
     and folds text-free deltas into `ABSTRACTED_DATA/`, tracked by the
     seen-id ledger.

Both paths enforce first-seen-wins against ids already stored, so
re-running never double-counts.

## 4. Caveats about live data

1. Live post scores are near zero at fetch time compared with mature
   archive scores, so the `score` column is never a counting signal. All
   counting uses raw mention counts.
2. Live volume is below archive volume. The dashboard normalises by
   share of discussion, and the coverage table printed by every run
   makes the eras visible.
3. The first `MIN_DAYS` (28) live days of a new forum or name have no
   trailing z-scores (warm-up).

## 5. Historical data

`tools/backfill_reddit.py` pulls a date range from Arctic Shift in
resumable chunks with a ledger keyed by window and forum set;
`tools/fold_historical.py` folds the result (or dump archives) into the
aggregates with a per-(file, month) ledger. Either is a re-validation
event: run `python -m analytics.run_analytics --what phases --research`
afterwards (`RUNBOOK.md` §7).

## 6. Sentiment engine

The engine is VADER with a finance lexicon, fast enough for the full
archive. A finance-tuned transformer is affordable for live volume;
before switching, score a month of StockTwits messages with both engines
and compare against the authors' own Bullish/Bearish labels, which the
raw StockTwits lines preserve (`entities.sentiment.basic`). The swap happens inside
`src/sentiment.py::score_text()`; everything downstream is unchanged,
but it is a re-validation event like any other scoring change.
