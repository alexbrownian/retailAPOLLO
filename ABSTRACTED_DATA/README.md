# ABSTRACTED_DATA — the committed, text-free aggregates

The only data folder in the repository. It holds six small parquet files
that carry **no post text, no authors, no post ids and no forum names**:
only daily counts and sentiment scores per ticker and per theme. No
individual post can be reconstructed from them.

| File | Columns |
|---|---|
| `daily_ticker_counts.parquet` | date, ticker, mention_count |
| `daily_ticker_counts_by_source.parquet` | date, ticker, source, mention_count |
| `daily_ticker_sentiment.parquet` | date, ticker, n_posts, avg_sentiment, net_bullish |
| `daily_theme_counts.parquet` | date, theme, mention_count |
| `daily_theme_sentiment.parquet` | date, theme, n_posts, avg_sentiment, net_bullish |
| `daily_term_counts.parquet` | date, term, mention_count |

`source` keeps the labels `reddit` / `x` / `stocktwits`; no text is
attached to them.

## How it is filled

Text becomes numbers at one fixed line, and only numbers cross it.

- **Full mode** (a copy holding the raw `posts.parquet`):
  `ingestion/build_aggregates.py` rebuilds these files from text and
  `src/abstracted_data.py::export()` copies them here.
- **Aggregates mode** (any other copy): `ingestion/append_live_abstracted.py`
  aggregates each run's new posts and merges the deltas in — counts add,
  sentiment means recombine weighted by `n_posts` — so history is never
  revised, only extended. The seen-id ledger in `data/reference/` makes
  the merge idempotent.

Both modes end a run with `verify_abstracted`, which fails loudly if a
text-bearing column ever appears here. `.gitignore` blocks the raw
stores from this folder as a second net.

Routine use, in either mode:

```bash
python update_data.py                      # fetch, screen, fold, score
git add ABSTRACTED_DATA DASHBOARD_DATA
git commit -m "data refresh"
```

Full schema and the rest of the data layout: `docs/DATA.md`.
