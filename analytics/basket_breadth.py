"""Read an index or factor basket through its CONSTITUENTS.

Desk instruction 2026-08-05, and it is the whole design in one sentence:
*"you should be MAPPING stuff to these indices and not just looking for
mentions of the ETF"* — with the worked example, *"sk hynix would map to
SMH"*, and the framing that followed: *"momentum is basically the same as
retail bullishness, S&P 500 is just all the names and their individual
bullishness/attention"*.

WHY THE ETF TICKER IS THE WRONG THING TO COUNT
----------------------------------------------
Measured over 176,126 archived comments (2025-07-25 → 2026-07-26):

    MTUM      0 mentions        VTV   0        IVW   0
    VUG       2                 IVE   4 CAPS — against 82 lowercase,
                                      because "IVE" is how people type
                                      "I've" without the apostrophe

Retail does not discuss factors as factors. It discusses MU, AMD, AVGO
and INTC — which is what MTUM currently holds. So the crowd read for a
factor has to be assembled from the holdings, one name at a time, and
never from the fund's own symbol. The same logic is why the keyword map
carries "sk hynix" and "tsmc": a constituent that is not US-listed is
still discussed, just by name rather than by ticker.

WHAT THIS MODULE COMPUTES
-------------------------
Per day, over the constituents of one basket:

    n_live        constituents the crowd mentioned at all
    mentions      their total mention count
    share         those mentions as a fraction of ALL ticker mentions
                  that day — attention RELATIVE to the market, so a
                  quiet news week does not read as a cold basket
    net_bullish   post-weighted mean of the per-name net_bullish, i.e.
                  the basket's own bullishness, weighted by how much
                  each name was actually talked about
    breadth       share of live constituents whose net_bullish > 0 —
                  is the whole basket bullish, or one loud name?

`net_bullish` and `breadth` answer different questions and both are
needed. A basket can be strongly bullish on one mega-cap while every
other holding is being sold; breadth is what separates that from a
broad-based move, and it is the reason this is a "breadth" module.

WEIGHTING: BY CHATTER, NOT BY MARKET CAP
----------------------------------------
The project holds no market-cap or index-weight data, so the fund's
actual weights are NOT used even though `etf_constituents.csv` records
rank. Every constituent enters weighted by how much the crowd discussed
it. That is the honest choice for a *sentiment* read — the question is
what the crowd thinks about this basket, and the crowd does not
cap-weight its opinions — but it means these series are NOT a proxy for
the fund's return, and nothing here should be read as one.

FROZEN-PARAMETER STATUS: none. Every number below is descriptive and
carries no threshold, so this module is outside the re-validation
protocol in docs/ARCHITECTURE.md §6. It informs; it never flags.
"""

from __future__ import annotations

import csv
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTITUENTS_CSV = os.path.join(ROOT, "config", "etf_constituents.csv")

# A basket needs enough live names before its breadth means anything.
# Below this the "share bullish" figure is one or two names wearing a
# percentage sign - notebook 07 hit the same wall on its own composite
# and answered it the same way, with a floor and a printed sensitivity
# rather than a silent minimum.
MIN_LIVE_NAMES = 3


def load_baskets(path: str = CONSTITUENTS_CSV) -> dict[str, list[str]]:
    """{ETF -> [constituent tickers]} from config/etf_constituents.csv.

    One file, one home: the same rows the theme map is built from, so a
    basket read and a theme count can never disagree about what an ETF
    holds."""
    out: dict[str, list[str]] = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            etf = str(row.get("etf", "")).strip().upper()
            tic = str(row.get("ticker", "")).strip().upper()
            if etf and tic and tic not in out.setdefault(etf, []):
                out[etf].append(tic)
    return out


def basket_breadth(basket: list[str],
                   counts: pd.DataFrame,
                   sentiment: pd.DataFrame | None = None,
                   min_live: int = MIN_LIVE_NAMES) -> pd.DataFrame:
    """Daily crowd read for one basket. Empty frame if nothing lands.

    `counts` is daily_ticker_counts (date, ticker, mention_count);
    `sentiment` is daily_ticker_sentiment (date, ticker, n_posts,
    net_bullish). Sentiment is optional so the function still returns
    attention on a machine that has not built it."""
    if not basket or counts is None or not len(counts):
        return pd.DataFrame()
    syms = {s.upper() for s in basket}
    c = counts.copy()
    c["date"] = pd.to_datetime(c["date"])
    c["ticker"] = c["ticker"].astype(str).str.upper()

    market = c.groupby("date")["mention_count"].sum().rename("market")
    sub = c[c["ticker"].isin(syms)]
    if not len(sub):
        return pd.DataFrame()

    out = sub.groupby("date").agg(n_live=("ticker", "nunique"),
                                  mentions=("mention_count", "sum"))
    out = out.join(market)
    out["share"] = out["mentions"] / out["market"].where(out["market"] > 0)

    if sentiment is not None and len(sentiment):
        s = sentiment.copy()
        s["date"] = pd.to_datetime(s["date"])
        s["ticker"] = s["ticker"].astype(str).str.upper()
        s = s[s["ticker"].isin(syms)]
        if len(s):
            s = s.assign(_w=s["net_bullish"] * s["n_posts"])
            agg = s.groupby("date").agg(_wsum=("_w", "sum"),
                                        _n=("n_posts", "sum"))
            # post-weighted, so a name nobody posted about cannot swing it
            out["net_bullish"] = (agg["_wsum"]
                                  / agg["_n"].where(agg["_n"] > 0))
            pos = (s[s["n_posts"] > 0]
                   .assign(_p=lambda d: d["net_bullish"] > 0)
                   .groupby("date")["_p"].mean())
            out["breadth"] = pos

    out = out.drop(columns=["market"])
    # a floor, applied openly: the row stays so the gap is visible on a
    # chart, but the derived figures are blanked rather than quoted from
    # two names.
    thin = out["n_live"] < min_live
    for col in ("share", "net_bullish", "breadth"):
        if col in out.columns:
            out.loc[thin, col] = float("nan")
    return out.sort_index()


def basket_coverage(basket: list[str], counts: pd.DataFrame,
                    days: int = 365) -> dict:
    """How much of a basket the crowd actually talks about.

    The honesty check that decides whether a basket is worth reading at
    all: a fund whose holdings nobody mentions produces a smooth line
    made of nothing."""
    if not basket or counts is None or not len(counts):
        return {"holdings": len(basket or []), "mentioned": 0,
                "strong": 0, "mentions": 0, "share": 0.0}
    c = counts.copy()
    c["date"] = pd.to_datetime(c["date"])
    c = c[c["date"] >= c["date"].max() - pd.Timedelta(days=days)]
    vol = c.groupby(c["ticker"].astype(str).str.upper())["mention_count"].sum()
    syms = [s.upper() for s in basket]
    live = [s for s in syms if s in vol.index]
    total = float(vol.reindex(live).fillna(0).sum())
    return {"holdings": len(syms),
            "mentioned": len(live),
            "strong": int(sum(1 for s in live if vol[s] >= 100)),
            "mentions": int(total),
            "share": total / float(vol.sum()) if vol.sum() else 0.0}
