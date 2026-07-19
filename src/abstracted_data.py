"""
abstracted_data.py
==================
ABSTRACTED_DATA/ is the one data folder committed to the repository and
shared with the INTERNAL machine. It holds no post text, no authors, no post
ids and no subreddit names - only daily COUNTS and SENTIMENT SCORES per
ticker / theme. Five small parquet files (~2 MB total) from which the signal
notebooks (08/09/10) and the price overlays run with zero access to the
underlying Reddit / X / StockTwits posts.

WHY THIS IS SAFE TO COMMIT
--------------------------
The raw stores (posts.parquet, posts_slice.parquet) reveal everything: title,
selftext, author, id, subreddit. The five aggregate files below are what the
pipeline produces AFTER text has been turned into numbers - they carry only
(date, ticker/theme, counts, sentiment scores). No individual post can be
reconstructed from them. The source column keeps the readable labels
'reddit'/'x'/'stocktwits', with no text attached.

THE FIVE FILES (exact schema the notebooks write / read)
    daily_ticker_counts.parquet            date, ticker, mention_count
    daily_ticker_counts_by_source.parquet  date, ticker, source, mention_count
    daily_ticker_sentiment.parquet         date, ticker, n_posts, avg_sentiment, net_bullish
    daily_theme_counts.parquet             date, theme,  mention_count
    daily_theme_sentiment.parquet          date, theme,  n_posts, avg_sentiment, net_bullish

TWO JOBS THIS MODULE DOES
    1. export()   copy the five files data/processed -> ABSTRACTED_DATA
       (EXTERNAL machine, after the aggregates are built from raw data).
       hydrate()  copy the five files ABSTRACTED_DATA -> data/processed
       (INTERNAL machine, so the unchanged notebooks find them where they
       already look). No notebook edits needed - only a copy.

    2. aggregate_posts() + merge_into_abstracted()  fold a batch of NEW live
       posts into the committed aggregates WITHOUT keeping the posts. Counts
       simply ADD; sentiment means RECOMBINE weighted by n_posts. Both give
       the identical result one-shot aggregation would.

WHY THE SENTIMENT MERGE IS WEIGHTED
    avg_sentiment is a mean over posts, and net_bullish = (bulls - bears)/n
    is also a per-post mean. Combining an OLD day-row (n_old posts) with a
    NEW day-row (n_new posts) cannot average the two averages - a row built
    from 100 posts must count more than a row built from 3. The merge
    rebuilds the underlying sums:
        combined_avg = (avg_old*n_old + avg_new*n_new) / (n_old + n_new)
    and the same for net_bullish - exactly what one-shot aggregation would
    compute, so history never gets revised, only extended.
"""

from __future__ import annotations

import os
import shutil

import pandas as pd

# ---------------------------------------------------------------------------
# WHERE THINGS LIVE
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTRACTED_DIR = os.path.join(ROOT, "ABSTRACTED_DATA")      # committed to git
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")     # private, gitignored

# The six canonical filenames (used everywhere - one source of truth).
TICKER_COUNTS = "daily_ticker_counts.parquet"
TICKER_COUNTS_BY_SOURCE = "daily_ticker_counts_by_source.parquet"
TICKER_SENT = "daily_ticker_sentiment.parquet"
THEME_COUNTS = "daily_theme_counts.parquet"
THEME_SENT = "daily_theme_sentiment.parquet"
TERM_COUNTS = "daily_term_counts.parquet"    # rolling word/phrase frequencies
                                             # for emerging-term detection -
                                             # text-free like everything here

FILES = [TICKER_COUNTS, TICKER_COUNTS_BY_SOURCE, TICKER_SENT,
         THEME_COUNTS, THEME_SENT, TERM_COUNTS]

# For each file: how a merge combines it, and the columns that make a row unique.
#   "counts"    -> mention_count adds up
#   "sentiment" -> n_posts adds, means recombine weighted by n_posts
MERGE_RULES = {
    TICKER_COUNTS:            ("counts",    ["date", "ticker"]),
    TICKER_COUNTS_BY_SOURCE:  ("counts",    ["date", "ticker", "source"]),
    TICKER_SENT:              ("sentiment", ["date", "ticker"]),
    THEME_COUNTS:             ("counts",    ["date", "theme"]),
    THEME_SENT:               ("sentiment", ["date", "theme"]),
    TERM_COUNTS:              ("counts",    ["date", "term"]),
}


# ---------------------------------------------------------------------------
# COPY HELPERS - export (external machine) and hydrate (internal machine)
# ---------------------------------------------------------------------------
def _copy_files(src_dir, dst_dir, verbose):
    """Copy whichever of the five files exist from src_dir to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in FILES:
        src_path = os.path.join(src_dir, name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, name))
            copied.append(name)
            if verbose:
                size_kb = os.path.getsize(src_path) / 1024
                print(f"  copied {name:<40} ({size_kb:,.0f} KB)")
        elif verbose:
            print(f"  (skip, not found) {name}")
    return copied


def export(src_dir=PROCESSED_DIR, dst_dir=ABSTRACTED_DIR, verbose=True):
    """External machine: publish the five aggregates to ABSTRACTED_DATA."""
    if verbose:
        print(f"export: {src_dir} -> {dst_dir}")
    copied = _copy_files(src_dir, dst_dir, verbose)
    if verbose:
        print(f"export done: {len(copied)}/{len(FILES)} files in ABSTRACTED_DATA")
    return copied


def hydrate(src_dir=ABSTRACTED_DIR, dst_dir=PROCESSED_DIR, verbose=True):
    """Internal machine: copy the committed aggregates into data/processed so
    the unchanged notebooks and scripts find them where they already look."""
    if verbose:
        print(f"hydrate: {src_dir} -> {dst_dir}")
    copied = _copy_files(src_dir, dst_dir, verbose)
    if verbose:
        print(f"hydrate done: {len(copied)}/{len(FILES)} files in data/processed")
    return copied


# ---------------------------------------------------------------------------
# MERGE MATHS - append without revising history
# ---------------------------------------------------------------------------
def _normalise_date(df):
    """Make the date column a real datetime so grouping never treats the
    string '2021-01-01' and the Timestamp 2021-01-01 as two different days."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def merge_counts(old, new, keys):
    """Additive merge: same (date, ticker[, source]) rows have their
    mention_count summed. Brand-new rows are carried through."""
    both = pd.concat([_normalise_date(old), _normalise_date(new)], ignore_index=True)
    merged = both.groupby(keys, as_index=False)["mention_count"].sum()
    return merged.sort_values(keys).reset_index(drop=True)


def merge_sentiment(old, new, keys):
    """n_posts-weighted merge (see the module docstring). Rebuilds the
    per-day sums, adds them, then divides back out."""
    def prep(df):
        df = _normalise_date(df)
        # avg_sentiment * n_posts = total compound score that day
        df["_sent_sum"] = df["avg_sentiment"] * df["n_posts"]
        # net_bullish * n_posts = (bulls - bears) that day
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
    # keep the columns in the schema order the notebooks expect
    entity = [k for k in keys if k != "date"]
    cols = ["date"] + entity + ["n_posts", "avg_sentiment", "net_bullish"]
    return grouped[cols].sort_values(keys).reset_index(drop=True)


def _safe_write(df, path):
    """Write a parquet atomically: write a .tmp file, then swap it in with
    os.replace (which overwrites the target in one step, on Windows too, so
    there is never a moment with no file). If the target is locked (open in
    Jupyter/Excel) the manual rename commands are printed instead of leaving
    things half-written."""
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
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


def merge_into_abstracted(new_aggs, target_dir=ABSTRACTED_DIR, verbose=True):
    """Fold a dict of {filename: new_aggregate_df} into ABSTRACTED_DATA.
    Each file is read, merged by its rule, and written back. Files that
    already exist accumulate; files that don't are created."""
    os.makedirs(target_dir, exist_ok=True)
    summary = {}
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
            # the term file rolls - old days fall off so it stays small
            # enough to commit (the spike test never looks that far back)
            from src.terms import trim_to_retention
            merged = trim_to_retention(merged)
        _safe_write(merged, path)
        summary[name] = len(merged)
        if verbose:
            print(f"  merged {name:<40} -> {len(merged):,} rows")
    return summary


# ---------------------------------------------------------------------------
# AGGREGATION - turn a batch of posts into the five aggregate frames.
# Reuses the SAME functions the notebooks use, so live and historical numbers
# are produced by identical code.
# ---------------------------------------------------------------------------
def _build_daily_theme_counts(posts_df):
    """date, theme, mention_count - each post counts once per theme it
    mentions (breadth of attention), the same rule the ticker side uses."""
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
    """The valid US ticker set (cached Nasdaq files + delisted supplement).
    max_cache_age_days is huge so this never hits the network on the
    internal machine - the cache under data/reference is enough."""
    from pathlib import Path
    from src.ticker_universe import load_us_ticker_universe
    return load_us_ticker_universe(Path(ROOT) / "data" / "reference",
                                   max_cache_age_days=100000)


def aggregate_posts(posts_df, universe=None, cashtags_only=False):
    """posts_df: standard 9-column posts (needs date, title, selftext, source).
    Returns {filename: aggregate_df} for the six files.

    Uses build_mentions + sentiment exactly like notebooks 02/06/07 do."""
    from src.build_mentions import build_daily_counts
    from src.sentiment import (add_sentiment_cached,
                               build_daily_ticker_sentiment,
                               build_daily_theme_sentiment)

    if universe is None:
        universe = load_universe()

    # ---- ticker counts: per source first, then sum into the combined signal
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

    # ---- sentiment: look up the permanent score store first, then score
    # only posts never seen before, then roll up per ticker and per theme
    posts_scored = add_sentiment_cached(posts_df)
    ticker_sent = build_daily_ticker_sentiment(posts_scored, universe,
                                               cashtags_only=cashtags_only)
    theme_sent = build_daily_theme_sentiment(posts_scored)
    theme_counts = _build_daily_theme_counts(posts_df)

    # ---- term counts: rolling word/phrase frequencies so emerging-term
    # detection keeps working after the fold, on whichever machine folded
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
