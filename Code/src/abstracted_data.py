"""Text-free daily aggregates: the committed Data/abstracted store.

Data/abstracted/ is the one data folder committed to the repository. It
holds no post text, no authors, no post ids and no subreddit names, only
daily counts and sentiment scores per ticker and per theme. Everything
downstream (the analytics package, the dashboard, the price overlays) runs
from these files with no access to the underlying Reddit / X / StockTwits
posts.

Why the store is safe to commit: the raw stores (posts.parquet,
posts_slice.parquet) carry title, selftext, author, id and subreddit. The
aggregate files are produced after text has been turned into numbers and
carry only (date, ticker/theme, counts, sentiment scores); no individual
post can be reconstructed from them. The source column keeps the readable
labels 'reddit' / 'x' / 'stocktwits' with no text attached.

The six files and their schemas::

    daily_ticker_counts.parquet            date, ticker, mention_count
    daily_ticker_counts_by_source.parquet  date, ticker, source, mention_count
    daily_ticker_sentiment.parquet         date, ticker, n_posts, avg_sentiment, net_bullish
    daily_theme_counts.parquet             date, theme,  mention_count
    daily_theme_sentiment.parquet          date, theme,  n_posts, avg_sentiment, net_bullish
    daily_term_counts.parquet              date, term,   mention_count (rolling retention)

The module does two jobs:

1. Copying. ``export()`` copies the files Data/processed -> Data/abstracted
   (full mode, after the aggregates are built from raw text).
   ``hydrate()`` copies Data/abstracted -> Data/processed (aggregates mode,
   so every consumer finds the files at the path it already reads).

2. Folding. ``aggregate_posts()`` turns a batch of new posts into the six
   aggregate frames and ``merge_into_abstracted()`` folds them into the
   committed store without keeping the posts. Counts add; sentiment means
   recombine weighted by n_posts. Both give the result a one-shot
   aggregation over all posts would.

Why the sentiment merge is weighted: avg_sentiment is a mean over posts and
net_bullish = (bulls - bears) / n is also a per-post mean, so an old day-row
built from n_old posts and a new day-row built from n_new posts cannot be
combined by averaging the two averages. The merge rebuilds the underlying
sums::

    combined_avg = (avg_old * n_old + avg_new * n_new) / (n_old + n_new)

and likewise for net_bullish, so history is extended, never revised.
"""

from __future__ import annotations

import os
import shutil

import pandas as pd

# ---------------------------------------------------------------------------
# Locations.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_DIR  # noqa: E402
ABSTRACTED_DIR = os.path.join(DATA_DIR, "abstracted")      # committed to git
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")     # private, gitignored

# The six canonical filenames; every reader and writer imports these.
TICKER_COUNTS = "daily_ticker_counts.parquet"
TICKER_COUNTS_BY_SOURCE = "daily_ticker_counts_by_source.parquet"
TICKER_SENT = "daily_ticker_sentiment.parquet"
THEME_COUNTS = "daily_theme_counts.parquet"
THEME_SENT = "daily_theme_sentiment.parquet"
TERM_COUNTS = "daily_term_counts.parquet"    # Rolling word/phrase frequencies
                                             # for emerging-term detection;
                                             # text-free like the others.

FILES = [TICKER_COUNTS, TICKER_COUNTS_BY_SOURCE, TICKER_SENT,
         THEME_COUNTS, THEME_SENT, TERM_COUNTS]

# For each file: how a merge combines it, and the columns that make a row
# unique.
#   "counts"    -> mention_count adds up.
#   "sentiment" -> n_posts adds; means recombine weighted by n_posts.
MERGE_RULES = {
    TICKER_COUNTS:            ("counts",    ["date", "ticker"]),
    TICKER_COUNTS_BY_SOURCE:  ("counts",    ["date", "ticker", "source"]),
    TICKER_SENT:              ("sentiment", ["date", "ticker"]),
    THEME_COUNTS:             ("counts",    ["date", "theme"]),
    THEME_SENT:               ("sentiment", ["date", "theme"]),
    TERM_COUNTS:              ("counts",    ["date", "term"]),
}


# ---------------------------------------------------------------------------
# Copy helpers: export (full mode) and hydrate (aggregates mode).
# ---------------------------------------------------------------------------
def _row_count(path):
    """Rows in a parquet file, read from its footer rather than its data.

    Returns -1 when the count cannot be read, so a caller testing for an
    empty source treats an unreadable file as "not known to be empty"
    and leaves the decision to the guards downstream.
    """
    try:
        import pyarrow.parquet as pq                  # noqa: PLC0415
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:                                 # noqa: BLE001
        return -1


def _copy_files(src_dir, dst_dir, verbose, skip_empty=False):
    """Copies whichever of the aggregate files exist from src_dir to dst_dir.

    With skip_empty, a source holding zero rows is left where it is
    instead of being copied over the destination, and the skip is
    reported whatever `verbose` says.

    Each file is copied beside its destination and swapped in with
    os.replace. Copying straight onto the destination truncates it at
    the first byte written, so a Ctrl-C part-way through leaves a
    fragment under the real name - and for `export` that name is a
    committed aggregate, the only copy of a history no rebuild can
    recover. Only a whole file ever appears under the final name.
    """
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in FILES:
        src_path = os.path.join(src_dir, name)
        if os.path.exists(src_path):
            if skip_empty and _row_count(src_path) == 0:
                print(f"  REFUSED {name}: the source holds zero rows; "
                      f"{dst_dir} keeps the copy it has")
                continue
            dst_path = os.path.join(dst_dir, name)
            shutil.copy2(src_path, dst_path + ".tmp")
            os.replace(dst_path + ".tmp", dst_path)
            copied.append(name)
            if verbose:
                size_kb = os.path.getsize(src_path) / 1024
                print(f"  copied {name:<40} ({size_kb:,.0f} KB)")
        elif verbose:
            print(f"  (skip, not found) {name}")
    return copied


def export(src_dir=PROCESSED_DIR, dst_dir=ABSTRACTED_DIR, verbose=True):
    """Publishes the aggregate files from Data/processed to Data/abstracted.

    Used in full mode, after the aggregates are rebuilt from the raw post
    store.

    A source file holding zero rows is refused: the committed store is
    the only copy of the history, callers publish before their own
    row-count checks run, and an empty file copied over it would zero a
    series that cannot be rebuilt from the text-free aggregates. The
    refusal is printed even when quiet, since a publish that skipped a
    file is never routine.

    Args:
        src_dir: Folder holding the freshly built aggregates.
        dst_dir: The committed store to publish into.
        verbose: Print one line per file copied or skipped.

    Returns:
        List of the filenames that were copied.
    """
    if verbose:
        print(f"export: {src_dir} -> {dst_dir}")
    copied = _copy_files(src_dir, dst_dir, verbose, skip_empty=True)
    if verbose:
        print(f"export done: {len(copied)}/{len(FILES)} files in Data/abstracted")
    return copied


def hydrate(src_dir=ABSTRACTED_DIR, dst_dir=PROCESSED_DIR, verbose=True):
    """Copies the committed aggregates into Data/processed.

    Used in aggregates mode, where there is no raw post store: every
    consumer reads from Data/processed, so the committed files are copied
    to the path they already look at.

    Args:
        src_dir: The committed store to read from.
        dst_dir: The working folder to copy into.
        verbose: Print one line per file copied or skipped.

    Returns:
        List of the filenames that were copied.
    """
    if verbose:
        print(f"hydrate: {src_dir} -> {dst_dir}")
    copied = _copy_files(src_dir, dst_dir, verbose)
    if verbose:
        print(f"hydrate done: {len(copied)}/{len(FILES)} files in Data/processed")
    return copied


# ---------------------------------------------------------------------------
# Merge arithmetic: append without revising history.
# ---------------------------------------------------------------------------
def _normalise_date(df):
    """Casts the date column to datetime so grouping never treats the string
    '2021-01-01' and the Timestamp 2021-01-01 as two different days."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def merge_counts(old, new, keys):
    """Additively merges two count frames.

    Rows sharing the same key have their mention_count summed; rows present
    in only one frame are carried through unchanged.

    Args:
        old: Existing aggregate frame.
        new: Frame of newly aggregated rows.
        keys: Columns that identify a row (for example ["date", "ticker"]).

    Returns:
        The merged frame sorted by keys, with a fresh RangeIndex.
    """
    both = pd.concat([_normalise_date(old), _normalise_date(new)], ignore_index=True)
    merged = both.groupby(keys, as_index=False)["mention_count"].sum()
    return merged.sort_values(keys).reset_index(drop=True)


def merge_sentiment(old, new, keys):
    """Merges two sentiment frames weighted by n_posts.

    Rebuilds each row's underlying sums (mean * n_posts), adds them across
    the two frames, then divides back out, so the result equals a one-shot
    aggregation over the union of posts (see the module docstring).

    Args:
        old: Existing aggregate frame.
        new: Frame of newly aggregated rows.
        keys: Columns that identify a row (for example ["date", "theme"]).

    Returns:
        The merged frame in schema column order, sorted by keys.
    """
    def prep(df):
        df = _normalise_date(df)
        # avg_sentiment * n_posts is the day's total compound score.
        df["_sent_sum"] = df["avg_sentiment"] * df["n_posts"]
        # net_bullish * n_posts is the day's (bulls - bears).
        df["_nb_sum"] = df["net_bullish"] * df["n_posts"]
        return df

    both = pd.concat([prep(old), prep(new)], ignore_index=True)
    grouped = both.groupby(keys, as_index=False).agg(
        n_posts=("n_posts", "sum"),
        _sent_sum=("_sent_sum", "sum"),
        _nb_sum=("_nb_sum", "sum"),
    )
    grouped["avg_sentiment"] = grouped["_sent_sum"] / grouped["n_posts"]
    grouped["net_bullish"] = grouped["_nb_sum"] / grouped["n_posts"]
    grouped = grouped.drop(columns=["_sent_sum", "_nb_sum"])
    # Keep the columns in the schema order every reader expects.
    entity = [k for k in keys if k != "date"]
    cols = ["date"] + entity + ["n_posts", "avg_sentiment", "net_bullish"]
    return grouped[cols].sort_values(keys).reset_index(drop=True)


def _stage_write(df, path):
    """Writes the frame beside its target and returns the staged path.

    The staged name is the target plus a .tmp suffix, so it ends in
    '.parquet.tmp' and no consumer's '*.parquet' glob can pick it up; a
    staged file is invisible to every reader until it is swapped in."""
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    return tmp


def _swap_in(tmp, path):
    """Swaps a staged file in over its target.

    os.replace overwrites the target in one step (on Windows too), so
    there is never a moment with no file. If the target is locked by
    another process the manual rename commands are printed and the
    PermissionError is re-raised rather than leaving a half-written
    file."""
    try:
        os.replace(tmp, path)
    except PermissionError:
        print("!" * 68)
        print(f"Could not replace {os.path.basename(path)} - it is open in")
        print("another program (a Jupyter kernel or Excel).")
        print("Close it, then rename by hand:")
        print(f'  del "{path}"')
        print(f'  ren "{tmp}" "{os.path.basename(path)}"')
        print("!" * 68)
        raise


def _safe_write(df, path):
    """Writes a parquet file atomically: stage it beside the target, then
    swap it in."""
    _swap_in(_stage_write(df, path), path)


def merge_into_abstracted(new_aggs, target_dir=ABSTRACTED_DIR, verbose=True):
    """Folds a batch of new aggregates into the committed store.

    Each file named in MERGE_RULES is read and merged by its rule
    (additive for counts, n_posts-weighted for sentiment). The batch is
    written in two passes: every merged frame is staged beside its target
    first, and the staged files are swapped in only once all of them
    exist, so the whole batch lands together. That matters because the
    caller records the folded post ids only after this returns and the
    aggregates carry no post ids: a batch that reached some files and not
    others would be re-folded on the next run into the files that already
    took it, and could never afterwards be detected or undone. Files that
    already exist accumulate; files that do not are created. The term
    file is trimmed to its retention window before it is staged.

    Args:
        new_aggs: Mapping {filename: aggregate frame}, as returned by
            aggregate_posts(). Missing or empty entries are skipped.
        target_dir: The store to merge into.
        verbose: Print one line per file merged.

    Returns:
        Mapping {filename: row count after the merge}.
    """
    os.makedirs(target_dir, exist_ok=True)
    staged = []
    try:
        for name, (kind, keys) in MERGE_RULES.items():
            new = new_aggs.get(name)
            if new is None or len(new) == 0:
                continue
            path = os.path.join(target_dir, name)
            if os.path.exists(path):
                old = pd.read_parquet(path)
                if kind == "counts":
                    merged = merge_counts(old, new, keys)
                else:
                    merged = merge_sentiment(old, new, keys)
            else:
                merged = _normalise_date(new)
            if name == TERM_COUNTS:
                # The term file rolls: old days fall off so it stays small
                # enough to commit; the spike test never looks that far back.
                from src.terms import trim_to_retention
                merged = trim_to_retention(merged)
            staged.append((name, path, _stage_write(merged, path), len(merged)))
    except BaseException:
        # Nothing has been swapped in, so the store still holds the
        # pre-merge numbers; drop the staged files so none is left behind.
        for _, _, tmp, _ in staged:
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    summary = {}
    for name, path, tmp, rows in staged:
        _swap_in(tmp, path)
        summary[name] = rows
        if verbose:
            print(f"  merged {name:<40} -> {rows:,} rows")
    return summary


# ---------------------------------------------------------------------------
# Aggregation: turn a batch of posts into the six aggregate frames. The
# same functions build the live and the historical numbers, so both are
# produced by identical code.
# ---------------------------------------------------------------------------
def _build_daily_theme_counts(posts_df):
    """Builds (date, theme, mention_count). Each post counts once per theme
    it mentions (breadth of attention), the same rule the ticker side uses."""
    from src.themes import themes_in_text

    rows = []
    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str)
    dates = posts_df["date"].astype(str)
    for date, title, body in zip(dates, titles, bodies):
        for theme in set(themes_in_text(title + " " + body)):
            rows.append({"date": date, "theme": theme})
    if not rows:
        return pd.DataFrame(columns=["date", "theme", "mention_count"])
    long_df = pd.DataFrame(rows)
    daily = (long_df.groupby(["date", "theme"], as_index=False)
             .size().rename(columns={"size": "mention_count"}))
    return daily


def load_universe():
    """Loads the valid US ticker set (cached Nasdaq files + delisted supplement).

    max_cache_age_days is set very large so the call never hits the network;
    the cache under Data/reference is sufficient in aggregates mode.

    Returns:
        The ticker universe as returned by
        src.ticker_universe.load_us_ticker_universe().
    """
    from pathlib import Path
    from src.ticker_universe import load_us_ticker_universe
    return load_us_ticker_universe(Path(DATA_DIR) / "reference",
                                   max_cache_age_days=100000)


def aggregate_posts(posts_df, universe=None, cashtags_only=False):
    """Turns a batch of posts into the six aggregate frames.

    Ticker counts are built per source and summed into the combined
    series; sentiment is looked up in the permanent score store and only
    unseen posts are scored; theme counts, theme sentiment and rolling
    term counts are built from the same batch.

    Args:
        posts_df: Standard 9-column posts frame; needs at least date, title,
            selftext and source.
        universe: Ticker universe; loaded via load_universe() when None.
        cashtags_only: Count only $-prefixed ticker mentions.

    Returns:
        Mapping {filename: aggregate frame} keyed by the six canonical
        filenames.
    """
    from src.build_mentions import build_daily_counts
    from src.sentiment import (add_sentiment_cached,
                               build_daily_ticker_sentiment,
                               build_daily_theme_sentiment)

    if universe is None:
        universe = load_universe()

    # Ticker counts: per source first, then sum into the combined signal.
    parts = []
    for source_name in sorted(posts_df["source"].unique()):
        one = posts_df[posts_df["source"] == source_name]
        d = build_daily_counts(one, universe, cashtags_only=cashtags_only)
        d["source"] = source_name
        parts.append(d)
    if parts:
        by_source = pd.concat(parts, ignore_index=True)
    else:
        by_source = pd.DataFrame(columns=["date", "ticker", "mention_count", "source"])
    counts = by_source.groupby(["date", "ticker"], as_index=False)["mention_count"].sum()

    # Sentiment: look up the permanent score store first, score only posts
    # never seen before, then roll up per ticker and per theme.
    posts_scored = add_sentiment_cached(posts_df)
    ticker_sent = build_daily_ticker_sentiment(posts_scored, universe,
                                               cashtags_only=cashtags_only)
    theme_sent = build_daily_theme_sentiment(posts_scored)
    theme_counts = _build_daily_theme_counts(posts_df)

    # Term counts: rolling word/phrase frequencies so emerging-term
    # detection keeps working after the fold, in either mode.
    from src.terms import count_daily_terms
    term_counts = count_daily_terms(posts_df)

    return {
        TICKER_COUNTS: counts,
        TICKER_COUNTS_BY_SOURCE: by_source[["date", "ticker", "source", "mention_count"]],
        TICKER_SENT: ticker_sent,
        THEME_COUNTS: theme_counts,
        THEME_SENT: theme_sent,
        TERM_COUNTS: term_counts,
    }
