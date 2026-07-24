"""
influence.py
============
THE INFLUENCE TRACKER - find the users whose calls have actually been
right, what they are saying NOW, and who follows them.

Method ported from Chan (Oxford M.Eng thesis, 2026), which showed on
r/CryptoMarkets that (a) predictive ability is concentrated in a SMALL
subset of users, (b) that subset is NOT the loud hubs - the thesis's
false-positive analysis found the structurally prominent users (3x the
degree, 2x the PageRank) had barely-above-chance accuracy (40.2%) while
the true high-predictors sat in peripheral positions with 79% accuracy -
and (c) HOW users participate (comment-vs-post balance) is more
informative than raw volume. This module implements those findings on
our data:

  1. CALL EXTRACTION - every post/comment by an author that mentions a
     ticker with clearly-signed sentiment is a directional CALL
     (author, date, ticker, direction, stance strength). Neutral chatter
     is not a call.
  2. CALL SCORING, VOLATILITY-AWARE (thesis section 4.5) - a call is
     judged against the next HORIZON days of real closes, but the bar
     scales with the name's own volatility:
         tau = max(MOVE_MIN, 0.5 * sigma)
     where sigma is the trailing 90d std of HORIZON-day moves for THAT
     ticker. A 3% move is a real call on an index ETF and noise on a
     meme stock - one fixed bar would misgrade both. Each judged call
     also gets an ABNORMAL-RETURN Z (how unusual the move was vs the
     name's own recent history), and an ENHANCED outcome (correct AND
     |z| > 1 - the move was direction-right and genuinely significant).
  3. AUTHOR "USEFULNESS" SCORING (thesis section 4.6) - three scores per
     author, each Bayesian-shrunk toward the population mean so nobody
     looks brilliant on three lucky calls:
       s_conf  stance-weighted accuracy      (shrink alpha=10)
       s_z     accuracy weighted by stance AND by the abnormal-return
               factor w(z)=clip(1+|z|, 0.1, 2.0)   (alpha=5 - a big-|z|
               hit is itself strong evidence, so it needs less shrink)
       s_enh   stance-weighted ENHANCED accuracy (alpha=10)
     Each is min-max normalised across authors, then
       COMPOSITE = 0.4*s_conf + 0.4*s_z + 0.2*s_enh
     and authors with composite >= 0.66 get the HIGH tier - the thesis's
     exact weighting and cutoff.
  4. BOOM/BUST RECORD - an author's record around the euphoria
     ground-truth peaks: bearish calls inside [peak-30d, peak+5d]
     = "called the top"; bullish calls there = "bought the top".
  5. THE SOCIAL INTERACTION GRAPH (thesis chapters 4-5) - an undirected
     WEIGHTED graph over authors, an edge when one replies to another,
     weight = number of interactions. From it: degree (distinct
     neighbours), weighted degree, and PAGERANK (the thesis ablation's
     single most beneficial structural feature; raw degree was actually
     HARMFUL there, so the board ranks by usefulness and shows PageRank
     as context, never ranks by degree). Bot filters ported from the
     thesis's cleaning table: edge weights capped at 100, star-topology
     accounts (degree centrality > 0.5) and broadcast accounts
     (> 1000 comments or > 100 posts here) excluded from graph metrics.
  6. LOUD-BUT-WRONG FLAG - the thesis's false-positive profile, made a
     column: top-quartile PageRank AND below-median composite. These are
     the accounts a naive "follow the big names" desk would copy - and
     precisely the ones the evidence says to fade.

STORAGE - COMMITTED, TEXT-FREE (desk decision, July 2026, reversing the
earlier local-only rule): the store now lives in git so both machines
share one leaderboard and every live run extends it. What is committed
is METADATA ONLY - author names (public pseudonymous identifiers),
dates, tickers, directions, outcomes, graph scores. NO post/comment
text ever enters these files (same text-free contract as
ABSTRACTED_DATA; enforced by a FORBIDDEN-column check at write time).

LIVE UPDATES: run_analytics calls update() on every pipeline run. A
per-file ledger (which raw files have been ingested, at what size)
means each run only parses the NEW raw files, appends their calls and
edges, and rescores the whole board (rescoring is cheap; parsing is
not). A full rebuild is just deleting the ledger.

USAGE - normally NOTHING (desk decision, July 2026: live-only). Every
live pipeline run (update_data.py) fetches new comments and calls
update(): the store builds itself from nothing on the first pull, new
calls append on every later pull, and recently-made calls re-judge
automatically once their 20-day windows have prices. No rebuilds, ever.
Manual forms, when wanted:
    python -m analytics.influence --top 20         # print the leaderboard
    python -m analytics.influence --update         # what the pipeline runs
    python -m analytics.influence --build          # force full re-parse
    python ingestion/fetch_reddit_comments.py --backfill START END
    (optional history deepener - the live path never needs it)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RAW_DIR, REFERENCE_DIR, PRICES_PATH   # noqa: E402

INFLUENCE_DIR = os.path.join(REFERENCE_DIR, "influence")     # committed
CALLS_PATH = os.path.join(INFLUENCE_DIR, "calls.parquet")
SCORES_PATH = os.path.join(INFLUENCE_DIR, "author_scores.parquet")
EDGES_PATH = os.path.join(INFLUENCE_DIR, "reply_edges.parquet")
LEDGER_PATH = os.path.join(INFLUENCE_DIR, "ingest_ledger.json")

# --- call rules (fixed, documented) ---
STANCE_MIN = 0.20     # |VADER compound| needed for a post to count as a
                      # directional CALL - mild chatter is not a forecast
HORIZON = 20          # days over which a call is judged (same horizon the
                      # project uses everywhere)
MOVE_MIN = 0.03       # the FLOOR of the correctness bar: tau never drops
                      # below 3% - sub-3% wiggles are never "calls landing"
VOL_HALF = 0.5        # tau = max(MOVE_MIN, VOL_HALF * sigma_90d) - the
                      # thesis's volatility-scaled threshold (its tau0 was
                      # 0.25% on hourly crypto; ours is 3% on 20d equity
                      # moves - same construction, domain-scaled)
ENH_Z = 1.0           # enhanced-correct needs the move >= 1 sigma abnormal
PRIOR_N_CONF = 10     # shrinkage strengths, straight from the thesis:
PRIOR_N_Z = 5         # the z-weighted score shrinks LESS because a big
PRIOR_N_ENH = 10      # abnormal move is itself diagnostic evidence
HIGH_TIER = 0.66      # composite cutoff for the HIGH tier (thesis 4.6.3)
MIN_CALLS_BOARD = 5   # leaderboard entry floor
# --- graph hygiene (thesis table 4.1, adapted) ---
MAX_EDGE_W = 100      # cap pairwise interaction weight (bot-like pairs)
MAX_COMMENTS = 1000   # accounts beyond these volumes are broadcast/bot
MAX_POSTS = 100       # accounts - excluded from GRAPH metrics (their
                      # calls still score; only their centrality is void)
MAX_DEG_CENT = 0.5    # star-topology filter: linked to > half the graph
PAGERANK_D = 0.85     # standard damping
# columns that must NEVER appear in a committed store file (the same
# text-free contract ABSTRACTED_DATA lives under)
FORBIDDEN_COLS = {"text", "body", "title", "selftext", "content"}


# ---------------------------------------------------------------------------
# 1. call extraction from local raw files
# ---------------------------------------------------------------------------
def _raw_files():
    """Every local raw file that carries authors, in a stable order."""
    return (sorted(glob.glob(os.path.join(RAW_DIR, "RedditLive",
                                          "*.jsonl.zst")))
            + sorted(glob.glob(os.path.join(RAW_DIR, "RedditComments",
                                            "*.jsonl.zst"))))


def _iter_raw_records(paths):
    """Yield (author, created, text, kind, link_id, parent_id, rec_id)
    from the given raw files: live Reddit posts and Arctic comment files.
    (X/StockTwits authors could be added the same way; Reddit first - it
    is where the reply graph lives.)"""
    from src.clean_data import read_json_lines
    for path in paths:
        is_comment = os.sep + "RedditComments" + os.sep in path
        for rec in read_json_lines(path):
            a = rec.get("author") or ""
            if not a or a in ("[deleted]", "AutoModerator"):
                continue
            created = rec.get("created_utc", 0)
            if not created:
                continue
            if is_comment:
                yield (a, int(created), rec.get("body") or "", "comment",
                       str(rec.get("link_id", "")),
                       str(rec.get("parent_id", "")),
                       str(rec.get("id", "")))
            else:
                text = ((rec.get("title") or "") + " "
                        + (rec.get("selftext") or ""))
                yield (a, int(created), text, "post",
                       str(rec.get("id", "")), "", str(rec.get("id", "")))


def extract_calls_and_edges(paths):
    """One pass over the given raw files -> (calls_df, edges_df).
    calls: rec_id, author, date, ticker, direction (+1/-1), stance, kind
    edges: replier -> author edges, resolved through BOTH the post map
           (t3_ link ids) and the comment map (t1_ parent ids), so
           comment-on-comment threads count too - the thesis's graph is
           built from exactly these interaction events."""
    from src.abstracted_data import load_universe
    from src.extract_tickers import extract_tickers_from_text
    from src.sentiment import score_text

    universe = load_universe()
    calls, replies = [], []
    post_author, comment_author = {}, {}
    n_seen = 0
    for a, created, text, kind, link_id, parent_id, rid in \
            _iter_raw_records(paths):
        n_seen += 1
        date = pd.Timestamp(created, unit="s").normalize()
        if kind == "post":
            post_author[rid] = a
            post_author["t3_" + rid] = a       # comments use t3_<id> links
        else:
            comment_author["t1_" + rid] = a
            replies.append((a, link_id, parent_id, rid))
        tickers = set(extract_tickers_from_text(text, universe,
                                                cashtags_only=False))
        if not tickers:
            continue
        s = score_text(text)
        if abs(s) < STANCE_MIN:
            continue                            # chatter, not a call
        for t in tickers:
            calls.append({"rec_id": rid, "author": a, "date": date,
                          "ticker": t, "direction": 1 if s > 0 else -1,
                          "stance": round(float(s), 3), "kind": kind})
    calls_df = pd.DataFrame(calls)
    edges = []
    for child, link, parent, rid in replies:
        # prefer the direct parent (comment-on-comment), fall back to the
        # post author (comment-on-post) - one edge per reply event
        target = comment_author.get(parent) or post_author.get(parent) \
            or post_author.get(link)
        if target and target != child:
            edges.append({"rec_id": rid, "replier": child, "author": target})
    edges_df = pd.DataFrame(edges)
    print(f"scanned {n_seen:,} raw items -> {len(calls_df):,} directional "
          f"calls by {calls_df['author'].nunique() if len(calls_df) else 0} "
          f"authors | {len(edges_df):,} reply edges")
    return calls_df, edges_df


# ---------------------------------------------------------------------------
# 2. volatility-aware call scoring (vectorised - comment backfills can
#    push the store past 100k calls, a per-row loop would crawl)
# ---------------------------------------------------------------------------
def score_calls(calls: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Judge every call against the next HORIZON days of real closes,
    with the thesis's volatility-scaled bar and abnormal-return z."""
    out = calls.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["fwd_ret"] = np.nan
    out["z"] = np.nan
    for sym, g in prices.groupby("symbol"):
        mask = out["ticker"] == sym
        if not mask.any():
            continue
        px = (g.sort_values("date").set_index("date")["px_last"]
              .asfreq("D").ffill())
        fwd = px.shift(-HORIZON) / px - 1          # HORIZON-day fwd move
        # the name's own recent distribution of such moves (trailing 90d,
        # shifted so day t only sees moves that COMPLETED before t)
        hist = fwd.shift(HORIZON)
        mu = hist.rolling(90, min_periods=30).mean()
        sd = hist.rolling(90, min_periods=30).std()
        d = out.loc[mask, "date"]
        r = fwd.reindex(d).to_numpy()
        m = mu.reindex(d).to_numpy()
        s = sd.reindex(d).to_numpy()
        out.loc[mask, "fwd_ret"] = r
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[mask, "z"] = np.where(s > 0, (r - m) / s, 0.0)
        out.loc[mask, "tau"] = np.maximum(MOVE_MIN, VOL_HALF * s)
    out["tau"] = out.get("tau", np.nan)
    signed = out["fwd_ret"] * out["direction"]
    out["outcome"] = np.select(
        [signed >= out["tau"], signed <= -out["tau"]],
        ["correct", "wrong"], default="flat")
    out.loc[out["fwd_ret"].isna(), "outcome"] = "unscored"
    # enhanced (thesis eq 4.6): direction right AND the move was >= 1
    # sigma ABNORMAL for this name - significance, not just sign
    out["enhanced"] = ((out["outcome"] == "correct")
                       & (out["z"] * out["direction"] >= ENH_Z))
    return out


# ---------------------------------------------------------------------------
# 3. the interaction graph (thesis chapters 4-5)
# ---------------------------------------------------------------------------
def build_graph_metrics(edges: pd.DataFrame,
                        activity: pd.DataFrame) -> pd.DataFrame:
    """Undirected weighted author graph -> degree, weighted degree,
    PageRank per author, with the thesis's bot filters applied first.
    activity: per-author n_comments / n_posts (for the broadcast filter).
    PageRank by plain power iteration - ~15 lines, no library needed."""
    if not len(edges):
        return pd.DataFrame(columns=["degree", "weighted_degree",
                                     "pagerank"])
    # undirected weighted edge list: count each replier<->author pair
    pair = edges.assign(
        a=np.minimum(edges["replier"], edges["author"]),
        b=np.maximum(edges["replier"], edges["author"]))
    w = (pair.groupby(["a", "b"]).size().rename("w")
         .clip(upper=MAX_EDGE_W)                       # bot-pair cap
         .reset_index())
    # broadcast/bot accounts contribute no graph signal (thesis filters)
    bots = set(activity.index[(activity["n_comments"] > MAX_COMMENTS)
                              | (activity["n_posts"] > MAX_POSTS)])
    w = w[~w["a"].isin(bots) & ~w["b"].isin(bots)]
    if not len(w):
        return pd.DataFrame(columns=["degree", "weighted_degree",
                                     "pagerank"])
    nodes = sorted(set(w["a"]) | set(w["b"]))
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    # star-topology filter: drop nodes linked to > half the graph. Only
    # meaningful once the graph is real-sized - in a 10-user graph an
    # ordinary active member exceeds any centrality cap (the thesis's
    # graph had 2,617 nodes when this filter earned its keep)
    stars = set()
    if n >= 50:
        deg = pd.concat([w.groupby("a").size(), w.groupby("b").size()],
                        axis=1).fillna(0).sum(axis=1)
        stars = set(deg.index[deg / max(n - 1, 1) > MAX_DEG_CENT])
    if stars:
        w = w[~w["a"].isin(stars) & ~w["b"].isin(stars)]
        nodes = sorted(set(w["a"]) | set(w["b"]))
        idx = {n_: i for i, n_ in enumerate(nodes)}
        n = len(nodes)
    if not n:
        return pd.DataFrame(columns=["degree", "weighted_degree",
                                     "pagerank"])
    # both directions of every undirected edge
    src = np.concatenate([w["a"].map(idx).to_numpy(),
                          w["b"].map(idx).to_numpy()])
    dst = np.concatenate([w["b"].map(idx).to_numpy(),
                          w["a"].map(idx).to_numpy()])
    ww = np.concatenate([w["w"].to_numpy(float)] * 2)
    out_w = np.zeros(n)
    np.add.at(out_w, src, ww)
    pr = np.full(n, 1.0 / n)
    for _ in range(50):                       # power iteration
        contrib = pr[src] * ww / out_w[src]
        nxt = np.zeros(n)
        np.add.at(nxt, dst, contrib)
        pr = (1 - PAGERANK_D) / n + PAGERANK_D * nxt
    g = pd.DataFrame(index=pd.Index(nodes, name="author"))
    dcount = np.zeros(n)
    np.add.at(dcount, src, 1.0)
    g["degree"] = dcount
    wdeg = np.zeros(n)
    np.add.at(wdeg, src, ww)
    g["weighted_degree"] = wdeg
    g["pagerank"] = pr
    return g


# ---------------------------------------------------------------------------
# 4. boom/bust record + the usefulness board
# ---------------------------------------------------------------------------
def boom_bust_record(scored: pd.DataFrame) -> pd.DataFrame:
    """Per author: calls inside euphoria peak windows. A BEARISH call in
    [peak-30d, peak+5d] = called the top; a BULLISH one = bought the top."""
    try:
        prices = pd.read_parquet(PRICES_PATH)
        prices["date"] = pd.to_datetime(prices["date"])
        from analytics.euphoria import ground_truth_peaks
        pxmap = {s: g.sort_values("date").set_index("date")["px_last"]
                 .asfreq("D").ffill() for s, g in prices.groupby("symbol")}
        peak_map = {sym: ground_truth_peaks(px, "single")
                    for sym, px in pxmap.items()}
    except Exception:
        return pd.DataFrame(columns=["author", "called_tops", "bought_tops"])
    rows = []
    for _, r in scored.iterrows():
        for p in peak_map.get(r["ticker"], []):
            if p - pd.Timedelta(days=30) <= r["date"] <= p + pd.Timedelta(days=5):
                rows.append({"author": r["author"],
                             "called": r["direction"] < 0})
                break
    if not rows:
        return pd.DataFrame(columns=["author", "called_tops", "bought_tops"])
    df = pd.DataFrame(rows)
    return (df.groupby("author")["called"]
            .agg(called_tops="sum", bought_tops=lambda s: int((~s).sum()))
            .reset_index())


def _shrink(per_author_mean, n, prior_n, global_mean):
    """Empirical-Bayes shrinkage (thesis eq 4.8): each author's mean is
    pulled toward the population mean; the pull fades as evidence (n)
    accumulates."""
    return (n * per_author_mean + prior_n * global_mean) / (n + prior_n)


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def build_author_scores(scored: pd.DataFrame,
                        edges: pd.DataFrame) -> pd.DataFrame:
    """The usefulness board: the thesis's three shrunk scores + composite
    + tier, the interaction-graph metrics, participation style, the
    boom/bust record, and each author's CURRENT stance."""
    judged = scored[scored["outcome"].isin(["correct", "wrong"])].copy()
    judged["y"] = (judged["outcome"] == "correct").astype(float)
    judged["conf"] = judged["stance"].abs()
    judged["s_conf"] = judged["conf"] * judged["y"]
    wz = np.clip(1 + judged["z"].abs(), 0.1, 2.0)     # thesis w(z)
    judged["s_z"] = judged["conf"] * judged["y"] * wz
    judged["s_enh"] = judged["conf"] * judged["enhanced"].astype(float)

    g = judged.groupby("author")
    stats = pd.DataFrame({
        "n_calls": scored.groupby("author").size(),
        "n_judged": g.size(),
        "hits": g["y"].sum().astype(int),
    }).fillna(0)
    stats["n_judged"] = stats["n_judged"].fillna(0)
    stats["hit_rate"] = stats["hits"] / stats["n_judged"].replace(0, np.nan)
    base = judged["y"].mean() if len(judged) else 0.5

    # the three usefulness scores, shrunk then normalised (thesis 4.6)
    for col, prior in (("s_conf", PRIOR_N_CONF), ("s_z", PRIOR_N_Z),
                       ("s_enh", PRIOR_N_ENH)):
        m = g[col].mean()
        glob = judged[col].mean() if len(judged) else 0.0
        stats[col] = _minmax(_shrink(m, stats["n_judged"], prior, glob)
                             .reindex(stats.index))
    stats["composite"] = (0.4 * stats["s_conf"] + 0.4 * stats["s_z"]
                          + 0.2 * stats["s_enh"])
    stats["tier"] = np.where(stats["composite"] >= HIGH_TIER, "HIGH", "low")
    # legacy simple shrunk hit-rate (still shown - easiest to explain)
    stats["score"] = _shrink(stats["hit_rate"].fillna(base),
                             stats["n_judged"], PRIOR_N_CONF, base)

    # participation style (thesis: the most informative behavioural
    # family) + audience
    kinds = scored.groupby(["author", "kind"]).size().unstack(fill_value=0)
    stats["n_comments"] = kinds.get("comment", pd.Series(0, index=kinds.index))
    stats["n_posts"] = kinds.get("post", pd.Series(0, index=kinds.index))
    stats[["n_comments", "n_posts"]] = (
        stats[["n_comments", "n_posts"]].fillna(0))
    stats["comment_post_ratio"] = (stats["n_comments"]
                                   / stats["n_posts"].replace(0, np.nan))
    if len(edges):
        aud = edges.groupby("author")["replier"].agg(
            followers="nunique", replies="count")
        stats = stats.join(aud, how="left")
    else:
        stats["followers"] = np.nan
        stats["replies"] = np.nan

    # the interaction graph (bot-filtered)
    gm = build_graph_metrics(edges, stats[["n_comments", "n_posts"]])
    stats = stats.join(gm, how="left")

    # loud-but-wrong: the thesis's false-positive profile as a column
    if stats["pagerank"].notna().any():
        pr_hi = stats["pagerank"] >= stats["pagerank"].quantile(0.75)
        comp_lo = stats["composite"] < stats["composite"].median()
        stats["loud_but_wrong"] = (pr_hi & comp_lo).fillna(False)
    else:
        stats["loud_but_wrong"] = False

    # current stance: the last 3 calls, newest first
    recent = (scored.sort_values("date", ascending=False)
              .groupby("author").head(3))
    stance = recent.groupby("author").apply(
        lambda g_: " | ".join(f"{'LONG' if d > 0 else 'SHORT'} {t} "
                              f"({dt.date()})"
                              for d, t, dt in zip(g_["direction"],
                                                  g_["ticker"], g_["date"])),
        include_groups=False).rename("latest_calls")
    stats = stats.join(stance, how="left")
    bb = boom_bust_record(scored)
    if len(bb):
        stats = stats.join(bb.set_index("author"), how="left")
    stats[["called_tops", "bought_tops"]] = (
        stats.reindex(columns=["called_tops", "bought_tops"]).fillna(0))
    stats["base_rate"] = round(base, 3)
    return (stats.reset_index().rename(columns={"index": "author"})
            .sort_values("composite", ascending=False))


# ---------------------------------------------------------------------------
# store I/O (committed + text-free, with a hard check)
# ---------------------------------------------------------------------------
def _safe_store_write(df: pd.DataFrame, path: str):
    bad = FORBIDDEN_COLS & set(c.lower() for c in df.columns)
    if bad:
        raise RuntimeError(f"REFUSING to write {path}: text columns {bad} "
                           "would break the text-free git contract")
    df.to_parquet(path, index=False)


def _load_ledger() -> dict:
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {"files": {}}


def _new_files(ledger: dict) -> list:
    out = []
    for p in _raw_files():
        key = os.path.relpath(p, RAW_DIR)
        if ledger["files"].get(key) != os.path.getsize(p):
            out.append(p)
    return out


def _mark_files(ledger: dict, paths: list):
    for p in paths:
        ledger["files"][os.path.relpath(p, RAW_DIR)] = os.path.getsize(p)


# ---------------------------------------------------------------------------
# build / update / CLI
# ---------------------------------------------------------------------------
def _rebuild_board_and_save(calls: pd.DataFrame, edges: pd.DataFrame,
                            ledger: dict) -> int:
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    scored = score_calls(calls, prices)
    board = build_author_scores(scored, edges)
    _safe_store_write(scored, CALLS_PATH)
    _safe_store_write(board, SCORES_PATH)
    if len(edges):
        _safe_store_write(edges, EDGES_PATH)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=1)
    judged = scored["outcome"].isin(["correct", "wrong"]).sum()
    n_high = int((board["tier"] == "HIGH").sum())
    print(f"influence store: {len(board)} authors ({n_high} HIGH tier), "
          f"{len(scored):,} calls ({judged:,} judged, base rate "
          f"{board['base_rate'].iloc[0] if len(board) else '?'}) "
          f"-> {INFLUENCE_DIR} (committed, text-free)")
    return 0


def build(incremental: bool = False):
    """Full rebuild (incremental=False) or extend-with-new-files."""
    os.makedirs(INFLUENCE_DIR, exist_ok=True)
    ledger = _load_ledger() if incremental else {"files": {}}
    paths = _new_files(ledger)
    old_calls = old_edges = None
    if incremental and os.path.exists(CALLS_PATH):
        old_calls = pd.read_parquet(CALLS_PATH)
        if os.path.exists(EDGES_PATH):
            old_edges = pd.read_parquet(EDGES_PATH)
    if not paths and old_calls is None:
        print("no raw author data found - fetch first:\n"
              "  python ingestion/fetch_all.py            (posts)\n"
              "  python ingestion/fetch_reddit_comments.py [--backfill ...]")
        return 1
    if paths:
        calls, edges = extract_calls_and_edges(paths)
    else:
        calls, edges = pd.DataFrame(), pd.DataFrame()
        print("no new raw files - rescoring the existing store")
    if old_calls is not None and len(old_calls):
        keep = [c for c in old_calls.columns if c in
                ("rec_id", "author", "date", "ticker", "direction",
                 "stance", "kind")]
        calls = pd.concat([old_calls[keep], calls], ignore_index=True)
        calls = calls.drop_duplicates(subset=["rec_id", "ticker"],
                                      keep="first")
    if old_edges is not None and len(old_edges):
        edges = pd.concat([old_edges, edges], ignore_index=True)
        if "rec_id" in edges.columns:
            edges = edges.drop_duplicates(subset=["rec_id", "author"],
                                          keep="first")
    if not len(calls):
        print("no directional calls found in the raw data")
        return 1
    _mark_files(ledger, paths)
    return _rebuild_board_and_save(calls, edges, ledger)


def update():
    """The live hook: parse only NEW raw files, extend the store, rescore.
    Silent no-op when there is nothing to do (internal machine before the
    first git pull of the store, or no new raw files)."""
    has_store = os.path.exists(CALLS_PATH)
    has_new = bool(_new_files(_load_ledger()))
    if not has_store and not has_new:
        return 0            # nothing local to work with - not an error
    if not os.path.exists(PRICES_PATH):
        print("influence: skipped (no prices yet)")
        return 0
    return build(incremental=True)


def main():
    p = argparse.ArgumentParser(description="Influence tracker store")
    p.add_argument("--build", action="store_true",
                   help="full rebuild from every raw file")
    p.add_argument("--update", action="store_true",
                   help="incremental: only new raw files (what the live "
                        "pipeline runs)")
    p.add_argument("--top", type=int, default=0)
    args = p.parse_args()
    if args.build:
        return build(incremental=False)
    if args.update:
        return update()
    if args.top:
        if not os.path.exists(SCORES_PATH):
            print("no store yet - run with --build first")
            return 1
        b = pd.read_parquet(SCORES_PATH)
        b = b[b["n_judged"] >= MIN_CALLS_BOARD].head(args.top)
        cols = ["author", "composite", "tier", "hit_rate", "n_judged",
                "pagerank", "followers", "loud_but_wrong",
                "called_tops", "bought_tops", "latest_calls"]
        print(b[[c for c in cols if c in b.columns]].to_string(index=False))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
