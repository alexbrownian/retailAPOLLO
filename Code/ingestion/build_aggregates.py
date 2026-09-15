"""Build all five aggregate files from ``posts.parquet`` in one pass.

This is the full rebuild used by ``update_data.py --full``::

    python Code/ingestion/build_aggregates.py
    python Code/ingestion/build_aggregates.py --start 2017-01-01

It requires the raw post store, so it runs in full mode only. The pass is
fast because:

* each post's text is processed once: a single extraction pass feeds
  ticker counts, ticker sentiment, theme counts and theme sentiment
  together;
* extraction and theme matching run in parallel across all CPU cores
  (``joblib``);
* sentiment uses the permanent id-to-score store (``src.sentiment``):
  every post is scored exactly once per engine, ever. Rebuilds look scores
  up instead of recomputing them, so only genuinely new posts cost
  anything;
* the store streams in batches, so the raw text is never held in memory
  all at once.

The output matches the schema and counting rules of the live folds (one
post = one mention per ticker/theme; ``n_posts``-weighted sentiment) and is
written through ``src.abstracted_data._safe_write`` into
``Data/processed``.
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import abstracted_data                              # noqa: E402
from src.sentiment import (_score_batch, get_engine_name,    # noqa: E402
                           _store_path, TRUNCATE_CHARS,
                           BULL_CUTOFF, BEAR_CUTOFF)
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402

POSTS_PATH = os.path.join(DATA_DIR, "processed", "posts.parquet")
PROCESSED = os.path.join(DATA_DIR, "processed")
BATCH = 100_000


def _extract_batch(dates, titles, bodies, sources, scores):
    """Worker: extract tickers and themes from one batch of posts.

    One extraction pass per post feeds every aggregate.

    Args:
        dates: Post dates (anything whose ``str()`` starts with
            ``YYYY-MM-DD``).
        titles: Post titles.
        bodies: Post bodies.
        sources: Source label per post.
        scores: Sentiment score per post.

    Returns:
        Tuple of four row lists: ticker mentions ``(date, ticker,
        source)``, ticker sentiment ``(date, ticker, sent, is_bull,
        is_bear)``, theme mentions ``(date, theme)`` and theme sentiment
        ``(date, theme, sent, is_bull, is_bear)``.
    """
    import sys as _sys
    if PROJECT_ROOT not in _sys.path:
        _sys.path.insert(0, PROJECT_ROOT)
    from src.extract_tickers import extract_tickers_from_text
    from src.themes import themes_in_text

    universe = _universe_cache()
    tick_rows, tick_sent, theme_rows, theme_sent = [], [], [], []
    for date, title, body, source, sent in zip(dates, titles, bodies, sources, scores):
        text = (title or "") + " " + (body or "")
        tickers = set(extract_tickers_from_text(text, universe, cashtags_only=False))
        themes = themes_in_text(text)
        is_bull = 1 if sent > BULL_CUTOFF else 0
        is_bear = 1 if sent < BEAR_CUTOFF else 0
        d = str(date)[:10]
        for t in tickers:
            tick_rows.append((d, t, source))
            tick_sent.append((d, t, sent, is_bull, is_bear))
        for th in themes:
            theme_rows.append((d, th))
            theme_sent.append((d, th, sent, is_bull, is_bear))
    return tick_rows, tick_sent, theme_rows, theme_sent


_UNIVERSE = None
def _universe_cache():
    """Return the ticker universe, loading it once per worker process."""
    global _UNIVERSE
    if _UNIVERSE is None:
        from src.abstracted_data import load_universe
        _UNIVERSE = load_universe()
    return _UNIVERSE


def _sent_frame(rows, entity):
    """Aggregate ``(date, entity, sent, bull, bear)`` rows to a daily frame.

    Args:
        rows: Per-post sentiment rows.
        entity: Name of the entity column (``"ticker"`` or ``"theme"``).

    Returns:
        DataFrame with ``date``, ``entity``, ``n_posts``, ``avg_sentiment``
        and ``net_bullish`` columns.
    """
    df = pd.DataFrame(rows, columns=["date", entity, "sentiment", "bull", "bear"])
    daily = (df.groupby(["date", entity])
             .agg(n_posts=("sentiment", "size"),
                  avg_sentiment=("sentiment", "mean"),
                  bull=("bull", "sum"), bear=("bear", "sum"))
             .reset_index())
    daily["net_bullish"] = (daily["bull"] - daily["bear"]) / daily["n_posts"]
    return daily.drop(columns=["bull", "bear"])


def main():
    """Score any unscored posts, extract once, and write the five aggregates.

    Returns:
        ``0`` on success; ``1`` when ``posts.parquet`` is absent.
    """
    ap = argparse.ArgumentParser(description="Build the five aggregates in one fast pass.")
    ap.add_argument("--start", default="2014-01-01",
                    help="build range start (default 2014-01-01)")
    ap.add_argument("--jobs", type=int, default=-1, help="CPU cores (-1 = all)")
    args = ap.parse_args()

    if not os.path.exists(POSTS_PATH):
        print("no posts.parquet - full mode only.")
        return 1

    from joblib import Parallel, delayed
    import time
    t0 = time.time()

    # ---- phase 1: sentiment scores via the permanent store -----------------
    store_path = _store_path()
    known = (pd.read_parquet(store_path).set_index("id")["sentiment"]
             if os.path.exists(store_path) else pd.Series(dtype=float))
    print(f"engine {get_engine_name()} | score store holds {len(known):,} posts")

    pf = pq.ParquetFile(POSTS_PATH)
    batches, new_ids, new_texts = [], [], []
    # author and subreddit are read for the bot screen only; they never
    # reach an aggregate (the text-free boundary is checked after the run)
    _cols = ["id", "date", "title", "selftext", "source"]
    _have = set(pf.schema_arrow.names)
    _cols += [c for c in ("author", "subreddit") if c in _have]
    for b in pf.iter_batches(columns=_cols, batch_size=BATCH):
        df = b.to_pandas()
        df = df[df["date"] >= args.start]
        if not len(df):
            continue
        batches.append(df)
        miss = ~df["id"].astype(str).isin(known.index)
        if miss.any():
            sub = df.loc[miss]
            new_ids.extend(sub["id"].astype(str).tolist())
            texts = (sub["title"].fillna("") + " "
                     + sub["selftext"].fillna("").str.slice(0, TRUNCATE_CHARS))
            new_texts.extend(texts.tolist())
    total = sum(len(d) for d in batches)
    print(f"{total:,} posts in range | {len(new_ids):,} need scoring "
          f"({time.time() - t0:.0f}s to load)")

    # ---- bot screen: drop automated / duplicated posts before they can
    #      reach an aggregate. Scored across all batches at once so a
    #      copy-paste campaign split over two batches is still seen.
    from ingestion.bot_screen import apply_screen, format_report, write_report
    if batches:
        _all = pd.concat(batches, ignore_index=True)
        _kept, _rep = apply_screen(_all)
        print(format_report(_rep) + f" ({time.time() - t0:.0f}s elapsed)")
        write_report(_rep, os.path.join(REPORTS_DIR, "bot_screen_last.json"))
        if _rep["rows_excluded"]:
            batches = [_kept.iloc[i:i + BATCH]
                       for i in range(0, len(_kept), BATCH)]
            total = len(_kept)

    # A pass with no posts in it would write five empty aggregates over a
    # healthy store and report success. Stop while the store is intact.
    if not total:
        print(f"no posts dated >= {args.start} survive in posts.parquet - "
              "nothing to build. The aggregates on disk are untouched; "
              "check --start and the raw store.")
        return 1

    if new_ids:
        chunks = [new_texts[i:i + 20_000] for i in range(0, len(new_texts), 20_000)]
        scored = Parallel(n_jobs=args.jobs)(delayed(_score_batch)(c) for c in chunks)
        flat = [s for part in scored for s in part]
        addition = pd.Series(flat, index=pd.Index(new_ids, name="id"), name="sentiment")
        known = pd.concat([known, addition])
        known = known[~known.index.duplicated(keep="first")]
        known.index.name = "id"          # concat can drop the index name;
        tmp = store_path + ".tmp"        # without it the reload finds no 'id'
        out_df = known.rename("sentiment").reset_index()
        out_df.columns = ["id", "sentiment"]
        out_df.to_parquet(tmp, index=False)
        os.replace(tmp, store_path)
        print(f"scored {len(new_ids):,} new posts "
              f"({time.time() - t0:.0f}s elapsed); store -> {len(known):,}")

    # ---- phase 2: one parallel extraction pass feeds every aggregate -------
    def batch_args(df):
        s = df["id"].astype(str).map(known).fillna(0.0)
        return (df["date"].tolist(), df["title"].fillna("").tolist(),
                df["selftext"].fillna("").tolist(), df["source"].tolist(),
                s.tolist())

    parts = Parallel(n_jobs=args.jobs)(
        delayed(_extract_batch)(*batch_args(df)) for df in batches)
    print(f"extraction done ({time.time() - t0:.0f}s elapsed); aggregating...")

    tick_rows = [r for p in parts for r in p[0]]
    tick_sent = [r for p in parts for r in p[1]]
    theme_rows = [r for p in parts for r in p[2]]
    theme_sent = [r for p in parts for r in p[3]]

    by_source = (pd.DataFrame(tick_rows, columns=["date", "ticker", "source"])
                 .groupby(["date", "ticker", "source"]).size()
                 .rename("mention_count").reset_index())
    counts = (by_source.groupby(["date", "ticker"], as_index=False)["mention_count"].sum())
    theme_counts = (pd.DataFrame(theme_rows, columns=["date", "theme"])
                    .groupby(["date", "theme"]).size()
                    .rename("mention_count").reset_index())
    ticker_sent = _sent_frame(tick_sent, "ticker")
    theme_sent_df = _sent_frame(theme_sent, "theme")

    outputs = {
        abstracted_data.TICKER_COUNTS: counts,
        abstracted_data.TICKER_COUNTS_BY_SOURCE:
            by_source[["date", "ticker", "source", "mention_count"]],
        abstracted_data.TICKER_SENT: ticker_sent,
        abstracted_data.THEME_COUNTS: theme_counts,
        abstracted_data.THEME_SENT: theme_sent_df,
    }
    # The same rule one level down: an extraction pass that matched no
    # ticker or theme is a broken run, not an empty week. Check every
    # frame BEFORE the first write, so a store is never left half
    # rewritten.
    empty = [name for name, df in outputs.items() if df.empty]
    if empty:
        print("ABORT: this pass produced 0 rows for "
              f"{', '.join(empty)} - nothing written, the aggregates on "
              "disk are left as they are.")
        return 1

    for name, df in outputs.items():
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values([c for c in ["date", "ticker", "theme", "source"]
                             if c in df.columns]).reset_index(drop=True)
        abstracted_data._safe_write(df, os.path.join(PROCESSED, name))
        print(f"  wrote {name:<42} {len(df):>9,} rows "
              f"({df['date'].min().date()} -> {df['date'].max().date()})")

    print(f"all five aggregates built in {time.time() - t0:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
