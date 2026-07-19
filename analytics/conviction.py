"""
conviction.py
=============
Conviction = mentions x sentiment combined, per TICKER and per THEME.
This module is the direct replacement for RetailFlow1 notebooks 08 and 09 -
identical mathematics, but pure vectorised pandas with no chart rendering,
so a full recompute over nine years of aggregates takes ~1 second instead
of minutes of notebook execution.

THE IDEA (read this once and the rest of the file is obvious)
-------------------------------------------------------------
Mentions tell you WHERE the crowd is; sentiment tells you WHICH WAY it
leans. Each alone is weak: mentions are direction-blind (GME calls and
puts look identical), and sentiment LEVELS are bullish-biased (retail
skews long). Combined, they say more than either. The construction:

1. BULL PRESSURE per day = n_posts * net_bullish
     = (bullish posts) - (bearish posts) that day.
   One number carrying volume AND direction: 100 posts at +0.2 lean = 20
   net bulls; 10 posts at +0.9 = 9. Attention without direction scores 0;
   direction without a crowd stays small.

2. 7-DAY ROLLING SUM of bull pressure. One loud afternoon is not
   conviction; a sustained week of lean is.

3. TRAILING Z-SCORE: that rolled sum compared against the SAME name's own
   PRECEDING 84 days -  (today - trailing mean) / trailing std.
   Why trailing and not whole-window: a whole-window z uses the future
   (2021's mania would drag every 2019 day negative), which silently
   corrupts backtests. With a trailing baseline, the number you compute
   for day t uses only information available on day t - what a backtest
   shows is exactly what the live run would have produced ("live parity").

   conviction z = +2 reads: "this name is two standard deviations more
   bullish-active than is normal FOR THIS NAME lately." A permanently loud
   name sits near 0; a quiet name that suddenly gains a devoted bullish
   crowd spikes - and that abnormality, not raw loudness, is the signal.

SUPPORTING SERIES built alongside (used by the signal engine, the
divergence flags and the dashboard):

  * attention z    - the same trailing-z trick on POST VOLUME alone
                     (crowd size, direction-blind)
  * sentiment 5d change - the 5-day change of the rolled net-bullish
                     share (is the MOOD improving or deteriorating?)
  * crowded-top    - attention z > 1 while the mood deteriorates: everyone
                     is watching but enthusiasm is fading, i.e. whoever
                     wanted to buy already has. The classic distribution
                     pattern - counts FOR a sell, AGAINST a buy.
  * swarm          - attention z > 1 while the mood improves: a confirmed
                     crowd arriving.

COVERAGE NORMALISATION (an improvement over notebooks 08/09)
  By default the z inputs are expressed as a SHARE of the day's total
  scored posts (see compute_conviction's `normalise` parameter). Raw bull
  pressure scales with how many posts were COLLECTED, and collection
  volume is not stationary (backfilled months run ~30x the live fetch),
  so raw z-scores go systematically negative after every coverage drop.
  Share normalisation removes the collection-volume term - identical in
  spirit to the share-of-chatter rule the mention charts use.

OUTPUT FILES (identical schema to the notebooks they replace)
  daily_ticker_conviction.parquet   date, ticker, conviction_z
  daily_theme_conviction.parquet    date, theme,  conviction_z
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (ROLL, BASELINE, MIN_DAYS, PROCESSED_DIR,
                        SENT_CHANGE_HORIZON, CROWDED_ATT_Z, CROWDED_SENT_DROP)
from analytics.loaders import (load, to_wide, TICKER_SENT, THEME_SENT,
                               TICKER_CONVICTION, THEME_CONVICTION)


def trailing_z(frame: pd.DataFrame, roll: int = ROLL,
               baseline: int = BASELINE, min_days: int = MIN_DAYS) -> pd.DataFrame:
    """The one z-score used everywhere in this project.

    frame : wide daily matrix (dates x entities), NaN-free (zeros for
            quiet days - the caller decides the fill).
    Steps: 7d rolling SUM -> mean/std over the TRAILING 84 days (needs at
    least 28 days of history before a z exists - the 'warm-up') ->
    standardise. A zero trailing std (a name with a perfectly constant
    week) would divide by zero; replacing it with NaN makes those days
    "no z" rather than infinity, which is the honest answer.
    """
    r = frame.rolling(roll, min_periods=1).sum()
    mu = r.rolling(baseline, min_periods=min_days).mean()
    sd = r.rolling(baseline, min_periods=min_days).std().replace(0, np.nan)
    return (r - mu) / sd


@dataclass
class ConvictionSet:
    """Everything the conviction computation produces for one entity type.
    All members are wide daily matrices (dates x entities) sharing one
    calendar-daily index."""
    n_posts: pd.DataFrame        # scored posts per day (crowd size)
    bull_pressure: pd.DataFrame  # n_posts * net_bullish (size x direction)
    conviction_z: pd.DataFrame   # trailing z of 7d-rolled bull pressure
    attention_z: pd.DataFrame    # trailing z of 7d-rolled post volume
    share: pd.DataFrame          # rolled net-bullish share (mood level)
    sent_change: pd.DataFrame    # 5d change of the share (mood direction)
    crowded_top: pd.DataFrame    # bool: crowd up AND mood souring
    swarm: pd.DataFrame          # bool: crowd up AND mood improving

    def tidy(self, entity_name: str) -> pd.DataFrame:
        """conviction_z back to long (date, entity, conviction_z) - the
        on-disk schema the dashboard and the signal snapshots read."""
        out = (self.conviction_z.stack().rename("conviction_z").reset_index())
        out.columns = ["date", entity_name, "conviction_z"]
        return out


MIN_DAY_TOTAL = 10   # a day needs at least this many scored posts overall
                     # before its pressure/attention SHARES mean anything


def compute_conviction(sent_df: pd.DataFrame, entity_col: str,
                       normalise: bool = True) -> ConvictionSet:
    """Build the full conviction set from one daily-sentiment aggregate.

    sent_df    : long frame (date, <entity>, n_posts, avg_sentiment,
                 net_bullish) - i.e. daily_ticker_sentiment.parquet or
                 daily_theme_sentiment.parquet.
    entity_col : "ticker" or "theme".
    normalise  : True (default) = COVERAGE-INVARIANT conviction: bull
                 pressure and attention are divided by the day's TOTAL
                 scored posts before the z (each expressed as % of the
                 day's chatter). WHY THIS MATTERS: collection volume is
                 not stationary - the 2026 backfill runs at ~1,000 scored
                 posts/day while the live fetch collects a fraction of
                 that, so RAW pressure falls off a cliff at the
                 archive->live boundary and every theme's z reads
                 negative for the next 84 days purely because fewer posts
                 were COLLECTED, not because the crowd left. Dividing by
                 the day's total removes the collection-volume term -
                 the same share-of-chatter trick the mention charts use.
                 Days with under MIN_DAY_TOTAL scored posts are treated
                 as zero-evidence (a 3-post day cannot define a share).
                 False = the raw-pressure behaviour of RetailFlow1
                 notebooks 08/09, kept for comparison.
    """
    df = sent_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Bull pressure BEFORE pivoting (row-wise product on the long frame).
    df["bull_pressure"] = df["n_posts"] * df["net_bullish"]

    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    # Days with no posts carry zero pressure and zero volume - filling 0 is
    # semantically correct here (silence IS zero conviction evidence).
    wide_bp = to_wide(df, entity_col, "bull_pressure", all_days, fill=0.0)
    wide_n = to_wide(df, entity_col, "n_posts", all_days, fill=0.0)

    if normalise:
        # coverage-invariant inputs: % of the day's total scored posts
        day_total = wide_n.sum(axis=1)
        denom = day_total.where(day_total >= MIN_DAY_TOTAL)
        z_in_bp = (wide_bp.div(denom, axis=0) * 100).fillna(0.0)
        z_in_n = (wide_n.div(denom, axis=0) * 100).fillna(0.0)
    else:
        z_in_bp, z_in_n = wide_bp, wide_n

    conviction_z = trailing_z(z_in_bp)
    attention_z = trailing_z(z_in_n)

    # Mood level: rolled pressure / rolled volume = the 7-day net-bullish
    # share. Dividing SUMS (not averaging daily ratios) weights every post
    # equally - a 3-post day cannot swing the week like a 300-post day.
    roll_bp = wide_bp.rolling(ROLL, min_periods=1).sum()
    roll_n = wide_n.rolling(ROLL, min_periods=1).sum()
    share = roll_bp / roll_n.replace(0, np.nan)
    sent_change = share.diff(SENT_CHANGE_HORIZON)

    # Divergence flags - the interesting moments are when the ingredients
    # DISAGREE (crowd arriving while the mood sours = crowded top).
    crowded_top = (attention_z > CROWDED_ATT_Z) & (sent_change < CROWDED_SENT_DROP)
    swarm = (attention_z > CROWDED_ATT_Z) & (sent_change > -CROWDED_SENT_DROP)

    return ConvictionSet(n_posts=wide_n, bull_pressure=wide_bp,
                         conviction_z=conviction_z, attention_z=attention_z,
                         share=share, sent_change=sent_change,
                         crowded_top=crowded_top, swarm=swarm)


# ---------------------------------------------------------------------------
# Extra read-outs for the dashboard (the visual summaries notebooks 08/09
# used to draw as static matplotlib - here they return DATA, the dashboard
# renders them as interactive Plotly).
# ---------------------------------------------------------------------------
def weekly_heatmap_frames(cs: ConvictionSet, top_n: int = 20,
                          min_weekly_posts: int = 10):
    """Weekly sentiment share + weekly conviction z for the most-posted
    names. Returns (share_weekly, conv_z_weekly), rows = weeks, columns =
    entities. Weeks with under `min_weekly_posts` scored posts are masked
    (NaN) - too thin to mean anything, shown grey on the dashboard."""
    top = list(cs.n_posts.sum().sort_values(ascending=False).head(top_n).index)
    wk_bp = cs.bull_pressure[top].resample("W").sum()
    wk_n = cs.n_posts[top].resample("W").sum()
    wk_share = (wk_bp / wk_n.replace(0, np.nan)).where(wk_n >= min_weekly_posts)
    wk_cz = (wk_bp - wk_bp.mean()) / wk_bp.std().replace(0, np.nan)
    return wk_share, wk_cz


def snail_trail(cs: ConvictionSet, name: str) -> pd.DataFrame:
    """One name's monthly path through the attention-x-sentiment plane:
    one row per month, columns az (avg attention z) and share (avg net
    bullish share). A healthy swarm walks right AND up; the classic
    blow-off walks right while sliding DOWN (crowd still growing, mood
    already rolling over) - visible weeks before it shows in price."""
    az_m = cs.attention_z[name].resample("ME").mean()
    share_m = cs.share[name].resample("ME").mean()
    return pd.DataFrame({"az": az_m, "share": share_m}).dropna()


# ---------------------------------------------------------------------------
# The pipeline entry point: recompute both conviction files on disk.
# ---------------------------------------------------------------------------
def rebuild_conviction_files(verbose: bool = True) -> dict:
    """Read the two sentiment aggregates, compute conviction, write the two
    conviction parquets (atomic write). Returns {filename: n_rows}."""
    from src.abstracted_data import _safe_write   # atomic parquet swap

    written = {}
    for sent_name, out_name, entity in [
            (TICKER_SENT, TICKER_CONVICTION, "ticker"),
            (THEME_SENT, THEME_CONVICTION, "theme")]:
        sent = load(sent_name)
        if sent is None:
            if verbose:
                print(f"  (skip {out_name} - {sent_name} not found)")
            continue
        cs = compute_conviction(sent, entity)
        tidy = cs.tidy(entity)
        _safe_write(tidy, os.path.join(PROCESSED_DIR, out_name))
        written[out_name] = len(tidy)
        if verbose:
            print(f"  wrote {out_name:<38} {len(tidy):>9,} rows "
                  f"({sent[entity].nunique()} {entity}s)")
    return written
