# Live data ingestion — keys, limits, cadences

## Where API keys live

All credentials go in `.env` at the project root (git-ignored; keys never
appear in code, notebooks, or commits). Scripts parse `.env` directly, with
`os.environ` as a fallback. One command runs every enabled source:
`python ingestion/fetch_all.py` — it checks `.env` first and calls only the
sources whose keys are filled (`--check` previews without calling).

| Credential | Used by | How to get it |
|---|---|---|
| `FETCHLAYER_KEY` | `fetch_x_live.py` (X); Reddit default is Arctic Shift (no key) | fetchlayer.dev dashboard. 1 credit per request; verify with `python ingestion/test_fetchlayer.py` (1 credit) |
| `REDDIT_*` (app name, id, secret, user, password) | `fetch_reddit_live.py` (official OAuth fallback) | https://old.reddit.com/prefs/apps/ → create a **script** app |
| StockTwits | `fetch_stocktwits.py` | none needed (public read-only streams) |
| `X_BEARER_TOKEN` | `fetch_x_live.py` (official v2 API fallback) | developer.x.com paid tier; used only if no FetchLayer key is present |

## What each fetcher pulls

- **Reddit** (`fetch_reddit_live.py`): two passes per subreddit — `new`
  (everything recent) and `top` of the last week (high-engagement posts) —
  15 subreddits x 2 = ~30 credits/run. Per-request progress lines, split
  connect/read timeouts with one retry, a 300 s run budget, and a
  `--max-credits` cap.
- **X** (`fetch_x_live.py`): a broad DISCOVERY pass first (top finance
  chatter of the week with engagement floors — catches tickers on nobody's
  watchlist, because the extractor finds every valid symbol in post text),
  then targeted cashtag chunks over the theme anchors, in both Top-of-week
  and Latest products. Backs off 30 s / 90 s on HTTP 429 before giving up;
  stops immediately on 402 (out of credits).
- **StockTwits** (`fetch_stocktwits.py`): public symbol streams; ~200
  requests/hour per IP is the practical cap, the fetcher stops early on 429.
  StockTwits users label their own posts Bullish/Bearish — those labels are
  kept in the raw files as calibration ground truth for the sentiment engine.

## Dedup and destinations

Raw posts land in `data/raw/` (transient). The append step depends on the
machine:

- **External machine**: `merge_live.py` appends into `posts.parquet` — a
  fast id pre-check makes the no-new-posts case cost seconds.
- **Internal machine**: `append_live_abstracted.py` aggregates the new posts
  and folds text-free deltas into `ABSTRACTED_DATA/`, tracked by a local
  seen-ids ledger.

Both enforce "first seen wins" against ids already stored, so re-running
never double-counts.

## Live-data caveats

1. Live post scores are near zero at fetch time vs mature archive scores —
   the `score` column is never a counting signal (kept for spam filtering
   only). All counting uses raw mention counts, immune by construction.
2. Live volume is far below archive volume; the overlays default to
   share-of-chatter normalisation and the coverage table in every
   `update_data.py` run makes the eras visible.
3. The first ~28 live days have no trailing z-scores (warm-up).

## Scheduling (Windows Task Scheduler)

```powershell
schtasks /Create /SC DAILY /ST 06:30 /TN "retailAPOLLO daily" ^
  /TR "python <path-to-project>\update_data.py"
```

## Sentiment upgrade path

The historical backfill uses VADER + a finance lexicon (fast enough for
10.8M posts). A finance-tuned transformer (e.g. FinTwitBERT) is affordable
for the live volume; before switching, score a month of StockTwits messages
with both engines and compare against the authors' own Bullish/Bearish
labels (`author_label()` in `src/stocktwits_data.py` extracts them). The
swap happens inside `src/sentiment.py::score_text()` — everything
downstream is unchanged.
