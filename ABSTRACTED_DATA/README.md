# ABSTRACTED_DATA — the only data folder committed to the repository

This folder is the abstraction layer: the only project data allowed onto
GitHub and the internal machine. It holds six small parquet files (~7 MB
total) that carry **no post text, no authors, no post ids, no subreddit
names** — only daily **counts** and **sentiment scores** per ticker / theme.
No individual Reddit / X / StockTwits post can be reconstructed from them.

| file | columns |
|---|---|
| `daily_ticker_counts.parquet` | date, ticker, mention_count |
| `daily_ticker_counts_by_source.parquet` | date, ticker, source, mention_count |
| `daily_ticker_sentiment.parquet` | date, ticker, n_posts, avg_sentiment, net_bullish |
| `daily_theme_counts.parquet` | date, theme, mention_count |
| `daily_theme_sentiment.parquet` | date, theme, n_posts, avg_sentiment, net_bullish |

`source` keeps the readable labels `reddit` / `x` / `stocktwits`; no text is
attached to them.

## Why the split works

The pipeline turns text into numbers at a fixed line:

- **External machine (needs raw text):** `ingestion/build_aggregates.py`
  reads the raw `posts.parquet` (private, gitignored) and writes the
  aggregates above.
- **Internal machine (numbers only):** the whole analytics layer and the
  dashboard read only these aggregates — so they run where no raw post
  ever exists.

## Bootstrap (external machine, where `posts.parquet` lives)

```bash
python update_data.py                     # builds + publishes the aggregates
git add ABSTRACTED_DATA && git commit -m "publish abstracted aggregates"
```

## Internal machine (repeat as often as data is ingested)

```bash
git pull                                  # latest ABSTRACTED_DATA
python update_data.py                     # fetch live -> fold in -> signals
git add ABSTRACTED_DATA && git commit -m "live update"
```

`append_live_abstracted.py` (called by update_data.py) aggregates newly
fetched posts and **merges** them into the files — counts add, sentiment
means recombine weighted by `n_posts`, so history is never revised, only
extended. Raw text is then discarded. A local, gitignored ledger
(`data/reference/abstracted_live_meta.json`) remembers which post ids were
already folded in, so re-running folds nothing twice (first-seen-wins).

## What is NOT here (by design)

`posts.parquet`, `posts_slice.parquet`, raw `*.jsonl` / `*.zst` files, and
the seen-ids ledger. `.gitignore` blocks them from this folder as a safety
net, and every `update_data.py` run ends with a schema check that fails
loudly if a text-bearing column ever appears here.
