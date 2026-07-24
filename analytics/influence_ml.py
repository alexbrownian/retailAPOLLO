"""
influence_ml.py
===============
The influential-users MODEL - the graph-learning half of Chan (2026)
ported to RetailRadar's own live influence store: can HIGH-predictive
authors be identified from their behaviour and their position in the
reply graph, BEFORE reading their track record?

RELATION TO THE THESIS (what is ported, what is adapted)
--------------------------------------------------------
The thesis formulates this as semi-supervised node classification on a
weighted social-interaction graph (r/CryptoMarkets, 2,617 nodes) and
compares seven architectures; GraphSAGE wins with AP 0.140 / AUROC 0.632
(~+61% AP over random). Here the same task runs on the desk's own store
(data/reference/influence/ - live-only, zero-touch, text-free), with the
model family scaled to the data:

  random          Bernoulli scores - the floor (thesis 6.1.8)
  mlp             feature-only MLP (thesis 6.1.1) - "behaviour alone"
  labelprop       structure-only label propagation over the reply graph
                  (thesis 6.1.2) - "position alone"
  sage_lite       GraphSAGE-style neighbourhood aggregation: each node's
                  features are concatenated with the mean of its
                  neighbours' features, then classified. This is exactly
                  one GraphSAGE layer with a mean aggregator and no
                  learned aggregation weights - honest about being the
                  small-data version of the thesis's winner (their own
                  finding: bigger models memorise when positives are
                  scarce; their GNNs reached ~12% precision on 133
                  positives).

DISCIPLINE (thesis 6.2, kept exactly)
-------------------------------------
* labels: HIGH tier = composite >= HIGH_TIER (0.66) among authors with
  >= 1 judged call; every other store author is an unlabelled graph node
  (structural context, never trained on);
* LEAKAGE GUARD: every feature derived from the labelling pipeline
  (composite, the three shrunk scores, hit_rate, tier) is EXCLUDED from
  the feature set - the model must predict the label, not read it;
* stratified 60/20/20 train/val/test split, seeds 42/100/2026,
  class-weighted losses, features z-scored on TRAIN statistics only;
* operating threshold: max positive-class precision s.t. recall >= 0.05,
  chosen on VALIDATION only (thesis 6.2.2);
* headline metrics: AP + AUROC on TEST (threshold-independent);
* robustness: feature-category ablation (7.2.1) + graph perturbation for
  the structural model - random edge rewiring and DICE-style swaps
  (7.2.2).

The module is import-clean (no side effects): notebook 05 drives it and
renders the narrative; nothing here writes to the store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.influence import HIGH_TIER

# The feature banks (text-free, store-only). Behavioural = how the author
# participates; structural = where they sit in the reply graph. The split
# mirrors the thesis's feature categories so the ablation reads the same.
BEHAVIOURAL = ["n_calls", "n_comments", "n_posts", "comment_post_ratio",
               "mean_conf"]
STRUCTURAL = ["degree", "weighted_degree", "pagerank", "followers",
              "replies"]
LEAKAGE = {"composite", "s_conf", "s_z", "s_enh", "hit_rate", "score",
           "tier", "hits", "n_judged"}          # never features


# ---------------------------------------------------------------------------
# node table
# ---------------------------------------------------------------------------
def build_node_table(board: pd.DataFrame,
                     calls: pd.DataFrame) -> pd.DataFrame:
    """One row per author: features + label. `board` is
    author_scores.parquet, `calls` is calls.parquet (for mean |stance| -
    perceived confidence, a legitimate behavioural feature; correctness
    -derived columns are the label's ancestry and stay out)."""
    tab = board.set_index("author").copy()
    conf = calls.assign(conf=calls["stance"].abs()).groupby("author")[
        "conf"].mean().rename("mean_conf")
    tab = tab.join(conf, how="left")
    tab["labelled"] = tab["n_judged"].fillna(0) > 0
    tab["y"] = ((tab["composite"] >= HIGH_TIER)
                & tab["labelled"]).astype(int)
    feats = BEHAVIOURAL + STRUCTURAL
    for c in feats:
        if c not in tab.columns:
            tab[c] = np.nan
    tab[feats] = tab[feats].astype(float).fillna(0.0)
    assert not (set(feats) & LEAKAGE), "leakage feature in the bank"
    return tab[feats + ["labelled", "y"]]


def neighbour_mean(tab: pd.DataFrame, edges: pd.DataFrame,
                   feats: list) -> pd.DataFrame:
    """The GraphSAGE-lite aggregation: for each author, the mean feature
    vector of their reply-graph neighbours (undirected), zeros for
    isolated nodes. Unlabelled nodes contribute here - this is exactly
    the 'unlabelled nodes remain in the graph during training' part of
    the thesis's semi-supervised setup."""
    if not len(edges):
        return pd.DataFrame(0.0, index=tab.index,
                            columns=[f"nb_{c}" for c in feats])
    und = pd.concat([
        edges.rename(columns={"replier": "src", "author": "dst"})[
            ["src", "dst"]],
        edges.rename(columns={"replier": "dst", "author": "src"})[
            ["src", "dst"]]], ignore_index=True)
    und = und[und["src"].isin(tab.index) & und["dst"].isin(tab.index)]
    nb = (und.merge(tab[feats], left_on="dst", right_index=True)
          .groupby("src")[feats].mean())
    nb = nb.reindex(tab.index).fillna(0.0)
    nb.columns = [f"nb_{c}" for c in feats]
    return nb


# ---------------------------------------------------------------------------
# splits + threshold rule (thesis 6.2)
# ---------------------------------------------------------------------------
def stratified_split(y: pd.Series, seed: int,
                     frac=(0.6, 0.2, 0.2)) -> pd.Series:
    """'train'/'val'/'test' per labelled node, class-stratified so the
    minority HIGH class appears proportionally in every part."""
    rng = np.random.default_rng(seed)
    part = pd.Series("", index=y.index, dtype=object)
    for cls in (0, 1):
        idx = y.index[y == cls].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        part.loc[idx[:a]] = "train"
        part.loc[idx[a:b]] = "val"
        part.loc[idx[b:]] = "test"
    return part


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray,
                   min_recall: float = 0.05) -> float:
    """Thesis 6.2.2: sweep candidate thresholds on VALIDATION, reject any
    with positive-class recall < min_recall, keep the highest precision."""
    best_thr, best_prec = 0.5, -1.0
    for thr in np.unique(np.round(p_val, 3)):
        pred = p_val >= thr
        tp = int((pred & (y_val == 1)).sum())
        if y_val.sum() and tp / y_val.sum() < min_recall:
            continue
        prec = tp / pred.sum() if pred.sum() else 0.0
        if prec > best_prec:
            best_prec, best_thr = prec, float(thr)
    return best_thr


# ---------------------------------------------------------------------------
# the four models - each returns P(HIGH) for every labelled node
# ---------------------------------------------------------------------------
def _zscore(train_X: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    mu, sd = train_X.mean(), train_X.std().replace(0, 1.0)
    return (X - mu) / sd


def model_random(tab, edges, part, feats, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(size=len(tab)), index=tab.index)


def model_mlp(tab, edges, part, feats, seed):
    from sklearn.neural_network import MLPClassifier
    train = tab.index[part == "train"]
    X = _zscore(tab.loc[train, feats], tab[feats])
    m = MLPClassifier(hidden_layer_sizes=(16,), early_stopping=True,
                      max_iter=500, random_state=seed)
    m.fit(X.loc[train], tab.loc[train, "y"])
    return pd.Series(m.predict_proba(X)[:, 1], index=tab.index)


def model_labelprop(tab, edges, part, feats, seed,
                    n_iter: int = 30, alpha: float = 0.85):
    """Structure-only: propagate train labels over the undirected reply
    graph (power iteration, the same style as influence.py's PageRank).
    Nodes keep alpha of their neighbours' mean belief + (1-alpha) of
    their clamped prior; unlabelled/val/test start at the train base
    rate. No features are used - that is the point."""
    train = set(tab.index[part == "train"])
    base = tab.loc[list(train), "y"].mean() if train else 0.5
    belief = pd.Series(base, index=tab.index, dtype=float)
    clamp = tab["y"].where(tab.index.isin(train))
    belief.loc[clamp.notna()] = clamp.dropna()
    if not len(edges):
        return belief
    und = pd.concat([
        edges.rename(columns={"replier": "src", "author": "dst"})[
            ["src", "dst"]],
        edges.rename(columns={"replier": "dst", "author": "src"})[
            ["src", "dst"]]], ignore_index=True)
    und = und[und["src"].isin(tab.index) & und["dst"].isin(tab.index)]
    for _ in range(n_iter):
        nb_mean = (und.merge(belief.rename("b"), left_on="dst",
                             right_index=True)
                   .groupby("src")["b"].mean())
        upd = alpha * nb_mean.reindex(tab.index).fillna(belief) \
            + (1 - alpha) * belief
        belief = upd
        belief.loc[clamp.notna()] = clamp.dropna()   # re-clamp trains
    return belief


def model_sage_lite(tab, edges, part, feats, seed):
    """One GraphSAGE layer, small-data version: [self features |
    neighbour-mean features] -> logistic regression (class-weighted).
    The aggregation uses ALL nodes (unlabelled included); only labelled
    train rows fit the classifier."""
    from sklearn.linear_model import LogisticRegression
    nb = neighbour_mean(tab, edges, feats)
    X_all = pd.concat([tab[feats], nb], axis=1)
    train = tab.index[part == "train"]
    X = _zscore(X_all.loc[train], X_all)
    m = LogisticRegression(class_weight="balanced", max_iter=2000)
    m.fit(X.loc[train], tab.loc[train, "y"])
    return pd.Series(m.predict_proba(X)[:, 1], index=tab.index)


MODELS = {"random": model_random, "mlp": model_mlp,
          "labelprop": model_labelprop, "sage_lite": model_sage_lite}


# ---------------------------------------------------------------------------
# evaluation harness (thesis 6.2/7.1) + robustness (7.2)
# ---------------------------------------------------------------------------
def evaluate(tab: pd.DataFrame, edges: pd.DataFrame,
             feats: list | None = None,
             seeds=(42, 100, 2026)) -> pd.DataFrame:
    """Every model x every seed -> AP / AUROC on TEST + the tuned
    threshold's confusion counts. Mean +/- std per model."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    feats = feats or (BEHAVIOURAL + STRUCTURAL)
    lab = tab[tab["labelled"]]
    rows = []
    for seed in seeds:
        part = stratified_split(lab["y"], seed)
        part_full = part.reindex(tab.index).fillna("unlabelled")
        for name, fn in MODELS.items():
            p = fn(tab, edges, part_full, feats, seed)
            val, test = lab.index[part == "val"], lab.index[part == "test"]
            y_t, p_t = lab.loc[test, "y"].values, p.loc[test].values
            thr = tune_threshold(lab.loc[val, "y"].values,
                                 p.loc[val].values)
            pred = p_t >= thr
            rows.append({
                "model": name, "seed": seed,
                "ap": average_precision_score(y_t, p_t)
                if y_t.sum() else np.nan,
                "auroc": roc_auc_score(y_t, p_t)
                if 0 < y_t.sum() < len(y_t) else np.nan,
                "threshold": thr,
                "tp": int((pred & (y_t == 1)).sum()),
                "fp": int((pred & (y_t == 0)).sum()),
                "fn": int((~pred & (y_t == 1)).sum()),
                "test_pos": int(y_t.sum()), "test_n": len(y_t),
            })
    df = pd.DataFrame(rows)
    return (df.groupby("model")
            .agg(ap=("ap", "mean"), ap_std=("ap", "std"),
                 auroc=("auroc", "mean"), auroc_std=("auroc", "std"),
                 tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean"),
                 test_pos=("test_pos", "first"), test_n=("test_n", "first"))
            .sort_values("ap", ascending=False).reset_index())


def ablate_categories(tab, edges, seeds=(42, 100, 2026)) -> pd.DataFrame:
    """Thesis 7.2.1 at category level: run the best-practice model
    (sage_lite) with each feature family removed. Category drops sidestep
    the correlated-single-feature caveat the thesis flags."""
    from sklearn.metrics import average_precision_score
    lab = tab[tab["labelled"]]
    variants = [("full", BEHAVIOURAL + STRUCTURAL),
                ("- behavioural", STRUCTURAL),
                ("- structural", BEHAVIOURAL)]
    rows = []
    for label, feats in variants:
        aps = []
        for seed in seeds:
            part = stratified_split(lab["y"], seed)
            part_full = part.reindex(tab.index).fillna("unlabelled")
            p = model_sage_lite(tab, edges, part_full, feats, seed)
            test = lab.index[part == "test"]
            y_t = lab.loc[test, "y"].values
            if y_t.sum():
                aps.append(average_precision_score(y_t,
                                                   p.loc[test].values))
        rows.append({"variant": label, "ap": np.mean(aps),
                     "ap_std": np.std(aps)})
    base = rows[0]["ap"]
    for r in rows:
        r["d_ap"] = r["ap"] - base
    return pd.DataFrame(rows)


def perturb_graph(edges: pd.DataFrame, rate: float, mode: str,
                  labels: pd.Series, seed: int) -> pd.DataFrame:
    """Thesis 7.2.2: corrupt a fraction of edges. 'random' rewires one
    endpoint to a random node; 'dice' (Disconnect Internally, Connect
    Externally) targets same-label edges and rewires them across labels."""
    if not len(edges):
        return edges
    rng = np.random.default_rng(seed)
    out = edges.copy().reset_index(drop=True)
    nodes = labels.index.to_numpy()
    if mode == "random":
        pick = rng.random(len(out)) < rate
        out.loc[pick, "author"] = rng.choice(nodes, size=int(pick.sum()))
    else:                                          # DICE
        lab = labels.reindex(out["replier"]).values == \
            labels.reindex(out["author"]).values
        same = np.flatnonzero(lab & (rng.random(len(out)) < rate))
        for i in same:
            me = labels.get(out.at[i, "replier"], 0)
            other = nodes[labels.values != me]
            if len(other):
                out.at[i, "author"] = rng.choice(other)
    return out[out["replier"] != out["author"]]


def perturbation_curve(tab, edges, rates=(0.0, 0.1, 0.2, 0.3, 0.5),
                       seeds=(42, 100, 2026)) -> pd.DataFrame:
    """sage_lite's test AP as the graph degrades, both perturbation
    modes. A structure-using model must fall under 'random'; DICE probes
    whether same-label neighbourhoods carry the signal."""
    from sklearn.metrics import average_precision_score
    lab = tab[tab["labelled"]]
    rows = []
    for mode in ("random", "dice"):
        for rate in rates:
            aps = []
            for seed in seeds:
                part = stratified_split(lab["y"], seed)
                part_full = part.reindex(tab.index).fillna("unlabelled")
                e = perturb_graph(edges, rate, mode, tab["y"], seed)
                p = model_sage_lite(tab, e, part_full,
                                    BEHAVIOURAL + STRUCTURAL, seed)
                test = lab.index[part == "test"]
                y_t = lab.loc[test, "y"].values
                if y_t.sum():
                    aps.append(average_precision_score(
                        y_t, p.loc[test].values))
            rows.append({"mode": mode, "rate": rate,
                         "ap": np.mean(aps), "ap_std": np.std(aps)})
    return pd.DataFrame(rows)
