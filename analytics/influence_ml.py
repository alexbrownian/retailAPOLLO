"""
influence_ml.py
===============
The influential-users MODEL, run on RetailRadar's own live influence
store: can HIGH-predictive authors be identified from their behaviour and
their position in the reply graph, BEFORE reading their track record?

This module is INFORMATION ONLY. Nothing here feeds the euphoria START /
END signal; the euphoria engine does not import it. Its output is the
"who is worth listening to, and what are they saying" panel.

THE TASK, AND THE MODEL FAMILY CHOSEN FOR IT
--------------------------------------------
The task is semi-supervised transductive node classification on a
weighted social-interaction graph: unlabelled authors stay in the graph
as structural context, and only labelled nodes are split and scored. It
runs on the desk's own store (data/reference/influence - live-only,
zero-touch, text-free), with the model family deliberately scaled to how
few positives that store contains:

  random        Bernoulli scores - the floor. Its AP is the
                positive-class prevalence, which is why every result below
                is quoted as AP LIFT OVER RANDOM: that is the only figure
                that stays comparable across label regimes and across
                vintages of the store.
  mlp           feature-only MLP - "behaviour alone".
  labelprop     structure-only label propagation - "position alone", no
                features at all.
  sage_lite     GraphSAGE mean-aggregator, one layer:
                [self features | mean of neighbours' features].
  gcn_lite      GCN in its simplified/linear form: two
                symmetric-normalised propagation steps of the features,
                then a linear classifier (Wu et al.'s SGC identity - a
                GCN with the non-linearities removed).
  mixhop_lite   MixHop: CONCATENATE several propagation
                powers [X | SX | S^2X] instead of composing them, so the
                classifier can weight hop distances differently.
  h2gcn_lite    H2GCN: separate the ego features from the
                1-hop and the 2-hop-excluding-1-hop aggregates. This is
                the architecture built for HETEROPHILOUS graphs, and our
                own label analysis (influence_graph.homophily) says the
                positive class here is extremely heterophilous - so this
                is a prediction of the network analysis, not a fishing
                expedition.

The "lite" in every graph model means one thing and it is stated
honestly: the aggregation weights are FIXED (mean / symmetric
normalisation) rather than learned, and the read-out is a
class-weighted logistic regression rather than a deep MLP. That is a
deliberate response to the size of the labelled set - with ~10^2
positives, a model with learned aggregation weights has more parameters
than evidence and memorises. GAT is NOT included for the same reason:
attention IS the learned part, so a fixed-attention GAT would just be
sage_lite under another name.

DISCIPLINE
----------
* LABELS: three pre-stated regimes (see LABEL_REGIMES), because the
  headline needs a sensitivity check and because the production HIGH
  tier (composite >= 0.66) marks only ~25 authors, which leaves ~5 test
  positives - not decision-grade. The headline regime is therefore the
  SOFTENED cut, the strictest of the three that clears the
  pre-registered >= 130-positives maturity bar; the strict cut is still
  reported, labelled under-powered; and a prevalence-matched cut exists
  so that AP stays numerically comparable as the store's balance shifts
  over time. The production board tier is NOT redefined by any of this.
* LEAKAGE GUARD: every column with labelling-pipeline ancestry is banned
  from the feature bank - not just the score itself but anything computed
  from forward returns or from call correctness (see LEAKAGE). The model
  must predict the label, not read it.
* stratified 60/20/20 train/val/test on labelled nodes, unlabelled nodes
  stay in the graph as structural context, features z-scored on TRAIN
  statistics only, class-weighted losses.
* operating threshold: max positive-class precision s.t. recall >=
  min_recall, chosen on VALIDATION only.
* headline metrics: AP + AUROC on TEST (threshold-independent), reported
  as mean +/- std over seeds and as lift over random.
* robustness: per-feature and per-category ablation, graph
  perturbation - random rewiring and DICE - reported on ACCURACY as well
  as AP, because on a 20:1 problem precision is too noisy to read
  reliably and accuracy is kept only to show how uninformative it is.
* ADOPTION RULE: a change is adopted only if `paired_ap_test` shows a
  paired per-seed AP improvement whose confidence interval excludes zero
  over ADOPTION_SEEDS. Ranking tables stay on the three house seeds. When
  several candidates are tested in one round the confidence level is
  Bonferroni-corrected, and the ladder is climbed for ONE re-test round
  only - a ladder climbed until it stops improving is a ladder climbed
  into noise.

WHAT THE EVALUATION ACTUALLY FOUND (NB05, 2026-07-27; 12,528 authors,
38,201 edges, 5,071 labelled, 237 positives under the softened cut)
---------------------------------------------------------------------
1. THE GRAPH DOES NOT EARN ITS COMPLEXITY. sage_lite tops the leaderboard
   on mean AP (0.107 vs logit 0.098) but the 10-seed paired test puts the
   margin at -0.003, CI [-0.010, +0.004]. Every other architecture is at
   or significantly BELOW the linear model. The parsimony rule ships
   `logit`. The ordering the graph-learning literature would predict
   (SAGE > GCN > MLP > LabelProp) does hold - it is the size of the gap
   that does not survive.
2. THE FIRST PASS WAS HALF CIRCULAR. `mean_conf` is one of the two
   multiplicands of the label (see SCORE_ADJACENT) and was worth
   +0.098 AP, CI [+0.083, +0.113] - roughly half of the apparent
   performance. Excluding it moves the headline from AP 0.183 to 0.098
   over a 0.054 random floor: lift 2.87x -> 1.07x.
3. THE SIGNAL IS SMALL BUT REAL. 200-permutation label-shuffle null:
   AP 0.124 vs null mean 0.052, p = 0.005 (floor 0.005).
4. WHY THE GRAPH FAILS IS MEASURABLE, NOT MYSTERIOUS. Positive-class node
   homophily is 0.095 against 0.963 for the negative class: influential
   authors do not sit next to each other. DICE perturbation - deliberately
   rewiring same-label edges ACROSS the label boundary - raises AP from
   0.183 to 0.378 at 50%: deliberately damaging the graph makes the model
   BETTER. The reply graph's structure is actively misleading for
   this label, so a model that leans on it loses.
5. IT DOES NOT GENERALISE TO NEW AUTHORS. On the tenure split (fit on
   established voices, graded on authors who arrived later) every model
   including logit sits at or below the random floor. The standing
   limitation of a transductive setup is therefore measured here rather
   than merely acknowledged, and it is why the dashboard ranks authors by
   their MEASURED record and treats this model as a research exhibit.

The module is import-clean (no side effects): notebook 05 drives it and
renders the narrative; nothing here writes to the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from analytics.influence import HIGH_TIER
from analytics import influence_graph as ig

# --- label regimes -----------------------------------------------------
# SOFT_TIER: the round mid-point of a min-max-normalised composite. A
#   stated convention, chosen a priori and never tuned.
# REFERENCE_PREVALENCE: a FIXED reference prevalence, held constant run to
#   run. Random AP *is* prevalence, so pinning it pins the floor, which
#   keeps AP numbers from different vintages of the store directly
#   comparable rather than only ordinally comparable.
# MIN_POSITIVES: pre-registered maturity bar. A paired test on a
#   rare-positive problem of this shape needs on the order of 130
#   positives before it can reject anything at all; below the bar, no
#   result here is decision-grade.
SOFT_TIER = 0.50
REFERENCE_PREVALENCE = 0.0695
MIN_POSITIVES = 130

# --- feature banks (text-free, correctness-free, store-only) ----------
# BEHAVIOURAL = how much and in what form the author participates.
# STRUCTURAL  = where they sit in the reply graph.
# FORECAST    = the SHAPE of their calls (how spread, how one-sided, how
#               concentrated) - still nothing about whether they were right.
BEHAVIOURAL = ["n_calls", "n_comments", "n_posts", "comment_post_ratio"]
STRUCTURAL = ["degree", "weighted_degree", "pagerank", "followers",
              "replies"]
FORECAST = ["frac_bearish", "dir_entropy", "n_tickers", "ticker_entropy",
            "active_days", "calls_per_active_day", "comment_call_frac",
            "span_days"]
CATEGORIES = {"behavioural": BEHAVIOURAL, "structural": STRUCTURAL,
              "forecast": FORECAST}

# --- SCORE-ADJACENT: admissible-looking, quietly circular --------------
# Every component of the label is built as conf * y, where conf = |stance|
# and y = was-the-call-right (analytics.influence.build_author_scores).
# `mean_conf` is therefore literally one of the two multiplicands of the
# quantity being thresholded, and `stance_sd` is a second moment of the
# same variable. Neither reads the answer sheet - y is absent from both -
# but neither is an independent predictor either, and in a first pass
# `mean_conf` was by a wide margin the single most useful feature in the
# bank (-0.077 AP when removed, three times the next one). A feature that
# is an arithmetic factor of its own target does not belong in the
# headline bank; it is kept here, out of the default, so the notebook can
# price exactly what excluding it costs instead of quietly benefiting.
SCORE_ADJACENT = ["mean_conf", "stance_sd"]
FULL_BANK = BEHAVIOURAL + STRUCTURAL + FORECAST
WIDE_BANK = FULL_BANK + SCORE_ADJACENT      # reported, never shipped

# Banned as features. The first line is the label and its parts; the
# second is everything downstream of FORWARD RETURNS or of call
# correctness - `tau` in particular IS the correctness bar, so a feature
# built from it would be reading the answer sheet.
LEAKAGE = {"composite", "s_conf", "s_z", "s_enh", "hit_rate", "score",
           "tier", "hits", "n_judged",
           "tau", "z", "fwd_ret", "enhanced", "outcome", "called_tops",
           "bought_tops", "base_rate", "loud_but_wrong"}

ADOPTION_SEEDS = tuple(range(10))       # paired test (this project's rule)
HOUSE_SEEDS = (42, 100, 2026)           # ranking tables (house seeds)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def label_standard(board: pd.DataFrame) -> pd.Series:
    """Production HIGH tier: composite >= 0.66 (influence.HIGH_TIER)."""
    return (board["composite"] >= HIGH_TIER).astype(int)


def label_softened(board: pd.DataFrame) -> pd.Series:
    """Softened criterion: a lower composite cut, chosen a priori as the
    round half-way point of the score scale - not tuned to any result."""
    return (board["composite"] >= SOFT_TIER).astype(int)


def label_prevalence(board: pd.DataFrame) -> pd.Series:
    """Prevalence-matched: the top REFERENCE_PREVALENCE of LABELLED authors
    by composite. This regime exists for one reason only - random AP
    equals prevalence, so pinning the reference prevalence pins the
    floor, which is what keeps AP numbers directly comparable across
    vintages of the store rather than merely ordinally comparable."""
    lab = board[board["n_judged"].fillna(0) > 0]
    if not len(lab):
        return pd.Series(0, index=board.index, dtype=int)
    cut = lab["composite"].quantile(1 - REFERENCE_PREVALENCE)
    return (board["composite"] >= cut).astype(int)


LABEL_REGIMES = {"standard": label_standard, "softened": label_softened,
                 "prevalence": label_prevalence}
HEADLINE_REGIME = "softened"


# ---------------------------------------------------------------------------
# node table
# ---------------------------------------------------------------------------
def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, 0 for a degenerate distribution."""
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def call_features(calls: pd.DataFrame) -> pd.DataFrame:
    """The FORECAST bank, computed per author from the calls table.

    HOW each one is derived, and why it is not leakage:
      frac_bearish        share of calls that are short. Which side you
                          take is a style, not an outcome.
      dir_entropy         binary entropy of that share: 0 = always the
                          same side (a permabull or a permabear), ln 2 =
                          perfectly two-sided.
      n_tickers           how many distinct names they talk about.
      ticker_entropy      entropy of their name mix, normalised by
                          ln(n_tickers) so it reads as "spread out"
                          (1) vs "one-name obsessive" (0), independent of
                          how many calls they made.
      active_days         distinct calendar days with at least one call.
      calls_per_active_day intensity when they do show up.
      comment_call_frac   share of calls made in comments rather than
                          posts - where an author does their talking is
                          a style, and the ablation prices whether it
                          helps.
      stance_sd           dispersion of signed conviction - does this
                          author shout the same amplitude every time?
      span_days           first-to-last call gap: tenure.
    """
    if not len(calls):
        return pd.DataFrame(columns=FORECAST)
    c = calls.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c["bear"] = (c["direction"] < 0).astype(float) \
        if pd.api.types.is_numeric_dtype(c["direction"]) \
        else (ig._direction_sign(c["direction"]) < 0).astype(float)
    g = c.groupby("author", sort=False)
    out = pd.DataFrame({
        "frac_bearish": g["bear"].mean(),
        "n_tickers": g["ticker"].nunique(),
        "active_days": g["date"].nunique(),
        "comment_call_frac": g["kind"].apply(lambda s: float(
            (s == "comment").mean())),
        "stance_sd": g["stance"].std(),
        "span_days": (g["date"].max() - g["date"].min()).dt.days,
    })
    p = out["frac_bearish"].clip(1e-9, 1 - 1e-9)
    out["dir_entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    tick = (c.groupby(["author", "ticker"], sort=False).size()
            .groupby(level=0, sort=False)
            .apply(lambda s: _entropy((s / s.sum()).to_numpy())))
    out["ticker_entropy"] = (tick / np.log(out["n_tickers"].clip(lower=2))
                             ).reindex(out.index)
    out["calls_per_active_day"] = (g.size() /
                                   out["active_days"].clip(lower=1))
    out["mean_conf"] = g["stance"].apply(lambda s: float(s.abs().mean()))
    return out


def build_node_table(board: pd.DataFrame, calls: pd.DataFrame,
                     regime: str = HEADLINE_REGIME,
                     feats: list | None = None) -> pd.DataFrame:
    """One row per author: features + label.

    `board` is author_scores.parquet, `calls` is calls.parquet. Only
    authors with at least one JUDGED call carry a label; everyone else
    stays in the table (and therefore in the graph) as unlabelled
    structural context - that is what makes the setup transductive.
    """
    tab = board.set_index("author").copy()
    cf = call_features(calls)
    tab = tab.join(cf[[c for c in cf.columns if c not in tab.columns]],
                   how="left")
    tab["labelled"] = tab["n_judged"].fillna(0) > 0
    y = LABEL_REGIMES[regime](board.set_index("author"))
    tab["y"] = (y.reindex(tab.index).fillna(0).astype(int)
                * tab["labelled"].astype(int))
    feats = list(feats) if feats else FULL_BANK
    for c in feats:
        if c not in tab.columns:
            tab[c] = np.nan
    tab[feats] = tab[feats].astype(float).replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    assert not (set(feats) & LEAKAGE), "leakage feature in the bank"
    return tab[feats + ["labelled", "y"]]


# ---------------------------------------------------------------------------
# propagation context (built once, reused by every graph model)
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    """tab + the graph aligned to it + cached propagation operators.

    Building the operators once and reusing them is what makes the
    tournament, the ablation and the 10-seed adoption test affordable:
    the sparse products are the expensive part, and they do not depend on
    the split or the seed.
    """
    tab: pd.DataFrame
    g: ig.Graph
    _ops: dict = field(default_factory=dict, repr=False)

    def op(self, kind: str) -> sparse.csr_matrix:
        if kind not in self._ops:
            self._ops[kind] = _make_op(self.g, kind)
        return self._ops[kind]

    def propagate(self, X: pd.DataFrame, kind: str,
                  power: int = 1) -> np.ndarray:
        S = self.op(kind)
        M = X.to_numpy(dtype=float)
        for _ in range(power):
            M = S @ M
        return M


def make_ctx(tab: pd.DataFrame, edges: pd.DataFrame) -> Ctx:
    """The graph is built on tab's index ORDER, so every matrix row lines
    up with a table row and no reindexing is needed downstream."""
    return Ctx(tab=tab, g=ig.build_graph(edges, nodes=tab.index))


def _make_op(g: ig.Graph, kind: str) -> sparse.csr_matrix:
    """The three propagation operators used by the graph models.

      mean : row-normalised weighted adjacency. (S X)_i = weighted mean of
             i's neighbours' features. Isolated nodes get a zero row,
             which is the honest answer - "no neighbourhood evidence".
      gcn  : D~^-1/2 (A + I) D~^-1/2, the standard GCN operator. Adding
             the self-loop keeps a node's own features in the mix and
             the symmetric normalisation stops hubs from dominating.
      hop2 : the 2-hop neighbourhood with the 1-hop neighbours and self
             REMOVED, then row-normalised. This is H2GCN's key idea -
             "my friends' friends who are not my friends" is a different
             kind of evidence from "my friends", and mixing them is what
             hurts on a heterophilous graph.
    """
    n = g.n
    if n == 0:
        return sparse.csr_matrix((0, 0))
    A = g.A.tocsr()
    if kind == "mean":
        s = np.asarray(A.sum(axis=1)).ravel()
        inv = np.where(s > 0, 1.0 / np.where(s == 0, 1.0, s), 0.0)
        return sparse.diags(inv) @ A
    if kind == "gcn":
        At = (A + sparse.eye(n, format="csr")).tocsr()
        d = np.asarray(At.sum(axis=1)).ravel()
        dinv = np.where(d > 0, 1.0 / np.sqrt(np.where(d == 0, 1.0, d)), 0.0)
        D = sparse.diags(dinv)
        return (D @ At @ D).tocsr()
    if kind == "hop2":
        B = A.copy()
        B.data = np.ones_like(B.data)
        two = (B @ B).tocsr()
        two.setdiag(0.0)
        two = two - two.multiply(B > 0)      # drop pairs already 1-hop
        two.data = np.where(two.data > 0, 1.0, 0.0)
        two.eliminate_zeros()
        s = np.asarray(two.sum(axis=1)).ravel()
        inv = np.where(s > 0, 1.0 / np.where(s == 0, 1.0, s), 0.0)
        return (sparse.diags(inv) @ two).tocsr()
    raise ValueError(f"unknown operator {kind!r}")


def neighbour_mean(ctx: Ctx, feats: list, kind: str = "mean",
                   power: int = 1, prefix: str | None = None
                   ) -> pd.DataFrame:
    """A propagated copy of the feature block, named so the ablation and
    the coefficient tables stay readable."""
    prefix = prefix or f"{kind}{power}_"
    M = ctx.propagate(ctx.tab[feats], kind, power)
    return pd.DataFrame(M, index=ctx.tab.index,
                        columns=[f"{prefix}{c}" for c in feats])


# ---------------------------------------------------------------------------
# splits + threshold rule
# ---------------------------------------------------------------------------
def stratified_split(y: pd.Series, seed: int,
                     frac=(0.6, 0.2, 0.2)) -> pd.Series:
    """'train'/'val'/'test' per labelled node, class-stratified so the
    minority HIGH class appears proportionally in every part."""
    rng = np.random.default_rng(seed)
    part = pd.Series("", index=y.index, dtype=object)
    for cls in (0, 1):
        # copy=True is not decoration: for some index types .to_numpy()
        # hands back a read-only VIEW of the index's own buffer, and an
        # in-place shuffle of that either raises or (worse, on older
        # numpy) reorders the caller's index.
        idx = y.index[y == cls].to_numpy(copy=True)
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        part.loc[idx[:a]] = "train"
        part.loc[idx[a:b]] = "val"
        part.loc[idx[b:]] = "test"
    return part


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray,
                   min_recall: float = 0.05) -> float:
    """Sweep candidate thresholds on VALIDATION, reject any with
    positive-class recall < min_recall, keep the highest precision."""
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
# the models - each returns P(HIGH) for every node in ctx.tab
# ---------------------------------------------------------------------------
def _zscore(train_X: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    mu, sd = train_X.mean(), train_X.std().replace(0, 1.0)
    return (X - mu) / sd.replace(0, 1.0)


def _fit_linear(ctx: Ctx, X_all: pd.DataFrame, part: pd.Series,
                seed: int) -> pd.Series:
    """Shared read-out: z-score on TRAIN rows only, then a class-weighted
    logistic regression. Every graph model differs from every other only
    in what goes into X_all - which is exactly what makes the tournament
    a comparison of ARCHITECTURES rather than of optimisers."""
    from sklearn.linear_model import LogisticRegression
    train = ctx.tab.index[part == "train"]
    X = _zscore(X_all.loc[train], X_all).fillna(0.0)
    m = LogisticRegression(class_weight="balanced", max_iter=2000,
                           random_state=seed)
    m.fit(X.loc[train], ctx.tab.loc[train, "y"])
    return pd.Series(m.predict_proba(X)[:, 1], index=ctx.tab.index)


def model_random(ctx, part, feats, seed):
    """Uniform scores. Its AP is the prevalence - the floor every other
    number is quoted against."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(size=len(ctx.tab)), index=ctx.tab.index)


def model_mlp(ctx, part, feats, seed):
    """Feature-only MLP: no graph at all. 'Behaviour alone'."""
    from sklearn.neural_network import MLPClassifier
    train = ctx.tab.index[part == "train"]
    X = _zscore(ctx.tab.loc[train, feats], ctx.tab[feats]).fillna(0.0)
    m = MLPClassifier(hidden_layer_sizes=(16,), early_stopping=True,
                      max_iter=500, random_state=seed)
    m.fit(X.loc[train], ctx.tab.loc[train, "y"])
    return pd.Series(m.predict_proba(X)[:, 1], index=ctx.tab.index)


def model_logit(ctx, part, feats, seed):
    """Features only, linear. Not a graph architecture - it is the
    control that tells us whether the MLP's non-linearity earns its keep
    at this sample size."""
    return _fit_linear(ctx, ctx.tab[feats], part, seed)


def model_labelprop(ctx, part, feats, seed, n_iter: int = 30,
                    alpha: float = 0.85):
    """Label propagation: structure only, no features at all. Train
    labels are clamped and beliefs diffuse over the weighted reply graph;
    unlabelled, val and test nodes start at the train base rate. Written
    as sparse power iteration, the same style as influence.py's PageRank.
    """
    tab = ctx.tab
    train = tab.index[part == "train"]
    base = float(tab.loc[train, "y"].mean()) if len(train) else 0.5
    belief = np.full(len(tab), base)
    clamp_pos = tab.index.get_indexer(train)
    clamp_val = tab.loc[train, "y"].to_numpy(dtype=float)
    belief[clamp_pos] = clamp_val
    S = ctx.op("mean")
    isolated = np.asarray(np.abs(S).sum(axis=1)).ravel() == 0
    for _ in range(n_iter):
        nb = S @ belief
        nb = np.where(isolated, belief, nb)
        belief = alpha * nb + (1 - alpha) * belief
        belief[clamp_pos] = clamp_val
    return pd.Series(belief, index=tab.index)


def model_sage_lite(ctx, part, feats, seed):
    """GraphSAGE, one layer with a mean aggregator:
    [self features | mean of neighbours' features]. Unlabelled nodes
    contribute to the aggregation - that IS the semi-supervised part."""
    X = pd.concat([ctx.tab[feats], neighbour_mean(ctx, feats, "mean")],
                  axis=1)
    return _fit_linear(ctx, X, part, seed)


def model_gcn_lite(ctx, part, feats, seed):
    """GCN in linear (SGC) form: propagate features twice with the
    symmetric-normalised self-looped adjacency, then classify. Two hops
    is the standard depth for this form; deeper propagation over-smooths
    on a graph this sparse."""
    X = neighbour_mean(ctx, feats, "gcn", power=2, prefix="gcn2_")
    return _fit_linear(ctx, X, part, seed)


def model_mixhop_lite(ctx, part, feats, seed):
    """MixHop: CONCATENATE hop powers instead of composing them, so
    the classifier can give hop-0, hop-1 and hop-2 evidence different
    (and even opposite-signed) weights."""
    X = pd.concat([ctx.tab[feats],
                   neighbour_mean(ctx, feats, "gcn", 1, "gcn1_"),
                   neighbour_mean(ctx, feats, "gcn", 2, "gcn2_")], axis=1)
    return _fit_linear(ctx, X, part, seed)


def model_h2gcn_lite(ctx, part, feats, seed):
    """H2GCN: ego / 1-hop / 2-hop-excluding-1-hop kept SEPARATE.
    The design target is heterophily - graphs where a node's neighbours
    tend to carry the OTHER label - which is precisely what our own
    label analysis finds for the positive class."""
    X = pd.concat([ctx.tab[feats],
                   neighbour_mean(ctx, feats, "mean", 1, "h1_"),
                   neighbour_mean(ctx, feats, "hop2", 1, "h2_")], axis=1)
    return _fit_linear(ctx, X, part, seed)


def model_sage_unweighted(ctx, part, feats, seed):
    """sage_lite with the class weighting REMOVED. Not an architecture -
    it exists so the choice of class-weighted losses is a MEASURED
    decision here rather than an inherited convention."""
    from sklearn.linear_model import LogisticRegression
    X_all = pd.concat([ctx.tab[feats],
                       neighbour_mean(ctx, feats, "mean")], axis=1)
    train = ctx.tab.index[part == "train"]
    X = _zscore(X_all.loc[train], X_all).fillna(0.0)
    m = LogisticRegression(max_iter=2000, random_state=seed)
    m.fit(X.loc[train], ctx.tab.loc[train, "y"])
    return pd.Series(m.predict_proba(X)[:, 1], index=ctx.tab.index)


MODELS = {"random": model_random, "logit": model_logit, "mlp": model_mlp,
          "labelprop": model_labelprop, "sage_lite": model_sage_lite,
          "gcn_lite": model_gcn_lite, "mixhop_lite": model_mixhop_lite,
          "h2gcn_lite": model_h2gcn_lite,
          "sage_unweighted": model_sage_unweighted}
# the architectures (+ the linear control) that belong in the headline
# tournament; sage_unweighted is a discipline check, not a rival.
TOURNAMENT = ["random", "logit", "mlp", "labelprop", "sage_lite",
              "gcn_lite", "mixhop_lite", "h2gcn_lite"]

# SHIPPED MODEL. sage_lite tops the leaderboard on mean AP, but the
# 10-seed paired test (adoption_ladder, NB05) puts its margin over the
# plain linear model at -0.003 with CI [-0.010, +0.004] - the graph does
# not earn its complexity here. The parsimony rule therefore ships the
# linear model, and the fact that it is the SIMPLEST member of the
# tournament is the finding, not a shortcut.
BEST_MODEL = "logit"
# The best structure-USING model. Only relevant to exhibits that ask what
# the graph is doing (perturbation, DICE): corrupting the graph under a
# model that never reads it would measure nothing.
BEST_GRAPH_MODEL = "sage_lite"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Every headline number for one (y, p, threshold) triple.

    AP (average precision) is the area under the precision-recall curve:
    on a rare-positive problem it is the honest summary, and its random
    baseline is exactly the prevalence - which is why `ap_lift` below is
    the number that travels between datasets. AUROC is prevalence-free
    but flattered by the huge negative class. ACCURACY is reported as a
    cautionary panel only: on a 20:1 problem it barely moves whatever the
    model does, and showing that is the point.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    pos, n = int(y.sum()), len(y)
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prev = pos / n if n else np.nan
    ap = average_precision_score(y, p) if 0 < pos < n else np.nan
    return {"ap": ap, "ap_random": prev,
            "ap_lift": (ap / prev - 1.0) if (prev and np.isfinite(ap))
            else np.nan,
            "auroc": roc_auc_score(y, p) if 0 < pos < n else np.nan,
            "threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / pos if pos else np.nan,
            "accuracy": (tp + tn) / n if n else np.nan,
            "test_pos": pos, "test_n": n}


def roc_points(y: np.ndarray, p: np.ndarray, thr: float) -> pd.DataFrame:
    """ROC curve plus the operating point, for the results figure."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thrs = roc_curve(y, p)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thr": thrs})
    df["operating"] = df["thr"] <= thr
    return df


def pr_points(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_curve
    prec, rec, thrs = precision_recall_curve(y, p)
    return pd.DataFrame({"precision": prec, "recall": rec,
                         "thr": np.append(thrs, np.nan)})


# ---------------------------------------------------------------------------
# evaluation harness
# ---------------------------------------------------------------------------
def _run_one(ctx: Ctx, name: str, feats: list, seed: int,
             ensemble_seeds: tuple | None = None) -> dict:
    """One (model, split-seed) cell: fit, tune the threshold on VAL,
    score TEST. If `ensemble_seeds` is given the SPLIT is held fixed and
    the model is refit under each of those seeds, then the probabilities
    are averaged - that is the single-run vs ensemble comparison.
    """
    lab = ctx.tab[ctx.tab["labelled"]]
    part = stratified_split(lab["y"], seed)
    part_full = part.reindex(ctx.tab.index).fillna("unlabelled")
    fn = MODELS[name]
    if ensemble_seeds:
        ps = [fn(ctx, part_full, feats, s) for s in ensemble_seeds]
        p = sum(ps) / len(ps)
    else:
        p = fn(ctx, part_full, feats, seed)
    val, test = lab.index[part == "val"], lab.index[part == "test"]
    thr = tune_threshold(lab.loc[val, "y"].to_numpy(),
                         p.loc[val].to_numpy())
    out = _metrics(lab.loc[test, "y"].to_numpy(), p.loc[test].to_numpy(),
                   thr)
    out.update(model=name, seed=seed)
    return out


def evaluate(ctx: Ctx, feats: list | None = None,
             seeds=HOUSE_SEEDS, models: list | None = TOURNAMENT,
             ensemble_seeds: tuple | None = None) -> pd.DataFrame:
    """Every model x every seed -> AP / AUROC / lift / confusion on TEST,
    aggregated as mean +/- std per model - the leaderboard table."""
    feats = feats or FULL_BANK
    names = models or list(MODELS)
    rows = [_run_one(ctx, name, feats, seed, ensemble_seeds)
            for seed in seeds for name in names]
    df = pd.DataFrame(rows)
    agg = (df.groupby("model")
           .agg(ap=("ap", "mean"), ap_std=("ap", "std"),
                ap_lift=("ap_lift", "mean"),
                auroc=("auroc", "mean"), auroc_std=("auroc", "std"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                accuracy=("accuracy", "mean"),
                tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean"),
                test_pos=("test_pos", "first"), test_n=("test_n", "first"))
           .sort_values("ap", ascending=False).reset_index())
    return agg


def per_seed_ap(ctx: Ctx, name: str, feats: list, seeds) -> pd.Series:
    """The paired sample the adoption rule works on: one AP per seed."""
    return pd.Series({s: _run_one(ctx, name, feats, s)["ap"]
                      for s in seeds}, name=name)


def paired_ap_test(ctx: Ctx, name_a: str, name_b: str,
                   feats_a: list | None = None, feats_b: list | None = None,
                   seeds=ADOPTION_SEEDS, conf: float = 0.95) -> dict:
    """THE ADOPTION RULE. Same splits for both candidates, one AP each per
    seed, then a paired difference with a normal-approximation CI on the
    mean of the differences (n = len(seeds) paired observations).

    Pairing is what makes this powerful with so few seeds: the split
    lottery is by far the biggest source of variance here, and pairing
    removes it, because both candidates see the identical split.
    'b improves on a' is declared only when the whole CI is above zero.
    """
    from scipy import stats
    fa = feats_a or FULL_BANK
    fb = feats_b or fa
    a = pd.Series({s: _run_one(ctx, name_a, fa, s)["ap"] for s in seeds})
    b = pd.Series({s: _run_one(ctx, name_b, fb, s)["ap"] for s in seeds})
    d = (b - a).dropna()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    t = stats.t.ppf(0.5 + conf / 2, n - 1) if n > 1 else np.nan
    lo, hi = (d.mean() - t * se, d.mean() + t * se) if n > 1 else (
        np.nan, np.nan)
    return {"baseline": name_a, "candidate": name_b, "n_seeds": n,
            "ap_a": float(a.mean()), "ap_b": float(b.mean()),
            "mean_diff": float(d.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "wins": int((d > 0).sum()),
            "adopt": bool(n > 1 and lo > 0)}


# the complexity ladder: each rung is (incumbent, challenger). Rung 0 asks
# whether the FEATURES carry any signal at all; every later rung asks a
# single question - does this extra machinery beat the plain linear model
# on the same splits? Written down before it is run, so the shipped model
# is decided by the rule and not by whichever row happened to top the
# leaderboard.
LADDER = [("random", "logit"),
          ("logit", "mlp"),
          ("logit", "labelprop"),
          ("logit", "sage_lite"),
          ("logit", "gcn_lite"),
          ("logit", "mixhop_lite"),
          ("logit", "h2gcn_lite")]


def adoption_ladder(ctx: Ctx, rungs=LADDER, feats: list | None = None,
                    seeds=ADOPTION_SEEDS) -> pd.DataFrame:
    """Run `paired_ap_test` on every rung and return one row each.

    A mean AP difference is NOT a result: three seeds of a 48-positive
    test fold move AP by more than most of the gaps in the leaderboard.
    This is the table that decides, because it holds the split fixed
    across the two candidates and asks whether the difference survives
    its own confidence interval.
    """
    rows = [paired_ap_test(ctx, a, b, feats_a=feats, feats_b=feats,
                           seeds=seeds) for a, b in rungs]
    return (pd.DataFrame(rows)
            .drop(columns=["n_seeds"])
            .sort_values("mean_diff", ascending=False)
            .reset_index(drop=True))


def choose_model(ladder: pd.DataFrame, floor: str = "logit") -> str:
    """PARSIMONY RULE, applied to the ladder: ship the simplest model the
    evidence actually supports.

    Climb off `floor` only for a challenger whose paired CI clears zero;
    if several do, take the largest improvement; if none does, the extra
    machinery has not earned its place and the linear model ships. This is
    the same rule the euphoria side used to reject GBM ("rules over
    learners") - reused rather than reinvented so the project has one
    standard of proof, not two.
    """
    won = ladder[(ladder["baseline"] == floor) & ladder["adopt"]]
    if not len(won):
        return floor
    return str(won.sort_values("mean_diff", ascending=False)
               .iloc[0]["candidate"])


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------
def ablate_categories(ctx: Ctx, model: str = BEST_MODEL,
                      seeds=HOUSE_SEEDS) -> pd.DataFrame:
    """Ablation at CATEGORY level: drop a whole feature family and
    re-run. Category drops sidestep the correlated-single-feature caveat
    on `ablate_features` (drop one of two twins and the other covers for
    it, so both look harmless).
    """
    full = FULL_BANK
    variants = [("full", full)] + [
        (f"- {k}", [c for c in full if c not in v])
        for k, v in CATEGORIES.items()]
    rows = []
    for label, feats in variants:
        aps = [_run_one(ctx, model, feats, s)["ap"] for s in seeds]
        rows.append({"variant": label, "n_features": len(feats),
                     "ap": float(np.nanmean(aps)),
                     "ap_std": float(np.nanstd(aps))})
    base = rows[0]["ap"]
    for r in rows:
        r["d_ap"] = r["ap"] - base
    return pd.DataFrame(rows).sort_values("d_ap")


def ablate_features(ctx: Ctx, model: str = BEST_MODEL,
                    seeds=HOUSE_SEEDS, feats: list | None = None
                    ) -> pd.DataFrame:
    """Leave-one-out ablation: drop ONE feature at a time.

    Read the sign carefully: d_ap < 0 means removing the feature HURT, so
    the feature was beneficial; d_ap > 0 means removing it HELPED, so the
    feature was actively harmful - a live possibility at this sample
    size, where a noisy column costs the linear read-out more than it
    contributes.
    """
    full = list(feats or FULL_BANK)
    base = float(np.nanmean([_run_one(ctx, model, full, s)["ap"]
                             for s in seeds]))
    rows = []
    for f in full:
        kept = [c for c in full if c != f]
        ap = float(np.nanmean([_run_one(ctx, model, kept, s)["ap"]
                               for s in seeds]))
        cat = next((k for k, v in CATEGORIES.items() if f in v), "other")
        rows.append({"removed": f, "category": cat, "ap": ap,
                     "d_ap": ap - base})
    out = pd.DataFrame(rows).sort_values("d_ap")
    out.attrs["base_ap"] = base
    return out


def perturb_graph(edges: pd.DataFrame, rate: float, mode: str,
                  labels: pd.Series, seed: int) -> pd.DataFrame:
    """Corrupt a fraction of edges.
      'random' rewires one endpoint of each selected edge to a random
               node - pure structural noise.
      'dice'   (Disconnect Internally, Connect Externally) selects
               SAME-label edges and rewires them across the label
               boundary. On a heterophilous graph this can HELP, which is
               exactly what our own homophily numbers predict.
      'swap'   degree-preserving double-edge swap: take two edges
               (a->b) and (c->d) and turn them into (a->d) and (c->b).
               Every node keeps its exact degree, so this isolates the
               value of WHO is connected to WHOM from the value of simply
               being busy. 'random' and 'dice' both disturb the degree
               sequence; this mode leaves it exactly intact, which makes
               it the cleanest test of whether the graph carries
               information beyond degree.
    """
    if not len(edges) or rate <= 0:
        return edges
    rng = np.random.default_rng(seed)
    out = edges.copy().reset_index(drop=True)
    nodes = labels.index.to_numpy()
    y = labels.to_numpy()
    if mode == "swap":
        k = int(rate * len(out)) // 2 * 2
        if k >= 2:
            pick = rng.choice(len(out), size=k, replace=False)
            half = k // 2
            a, b = pick[:half], pick[half:]
            tgt_a = out.loc[a, "author"].to_numpy()
            out.loc[a, "author"] = out.loc[b, "author"].to_numpy()
            out.loc[b, "author"] = tgt_a
    elif mode == "random":
        pick = rng.random(len(out)) < rate
        out.loc[pick, "author"] = rng.choice(nodes, size=int(pick.sum()))
    else:
        la = labels.reindex(out["replier"]).to_numpy()
        lb = labels.reindex(out["author"]).to_numpy()
        same = np.flatnonzero((la == lb) & (rng.random(len(out)) < rate))
        if len(same):
            pools = {c: nodes[y != c] for c in np.unique(y)}
            new = [rng.choice(pools[la[i]]) if len(pools.get(la[i], []))
                   else out.at[i, "author"] for i in same]
            out.loc[same, "author"] = new
    return out[out["replier"] != out["author"]]


def perturbation_curve(tab: pd.DataFrame, edges: pd.DataFrame,
                       model: str = BEST_GRAPH_MODEL,
                       rates=(0.0, 0.1, 0.2, 0.3, 0.5),
                       seeds=HOUSE_SEEDS,
                       modes=("random", "dice", "swap")) -> pd.DataFrame:
    """The model's TEST AP *and accuracy* as the graph degrades, in every
    mode. A structure-using model must fall under 'random' - if it does
    not, it was never using the graph."""
    rows = []
    for mode in modes:
        for rate in rates:
            aps, accs = [], []
            for seed in seeds:
                e = perturb_graph(edges, rate, mode, tab["y"], seed)
                ctx = make_ctx(tab, e)
                r = _run_one(ctx, model,
                             FULL_BANK, seed)
                aps.append(r["ap"])
                accs.append(r["accuracy"])
            rows.append({"mode": mode, "rate": rate,
                         "ap": float(np.nanmean(aps)),
                         "ap_std": float(np.nanstd(aps)),
                         "accuracy": float(np.nanmean(accs)),
                         "accuracy_std": float(np.nanstd(accs))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# label-criteria sensitivity + misclassification analysis
# ---------------------------------------------------------------------------
def label_regime_table(board: pd.DataFrame, calls: pd.DataFrame,
                       edges: pd.DataFrame, model: str = BEST_MODEL,
                       seeds=HOUSE_SEEDS) -> pd.DataFrame:
    """Label-criteria sensitivity: does the conclusion survive a different
    definition of "influential"? One row per regime with its positive
    count, prevalence, AP, lift over random, AUROC and positive-class
    node homophily."""
    rows = []
    for regime in LABEL_REGIMES:
        tab = build_node_table(board, calls, regime=regime)
        ctx = make_ctx(tab, edges)
        res = [_run_one(ctx, model, FULL_BANK, s)
               for s in seeds]
        rnd = [_run_one(ctx, "random", FULL_BANK,
                        s) for s in seeds]
        h = ig.homophily(ctx.g, tab["y"])
        lab = tab[tab["labelled"]]
        rows.append({
            "regime": regime,
            "positives": int(lab["y"].sum()),
            "labelled": int(len(lab)),
            "prevalence": float(lab["y"].mean()),
            "powered": bool(lab["y"].sum() >= MIN_POSITIVES),
            "ap": float(np.nanmean([r["ap"] for r in res])),
            "ap_std": float(np.nanstd([r["ap"] for r in res])),
            "ap_random": float(np.nanmean([r["ap"] for r in rnd])),
            "ap_lift": float(np.nanmean([r["ap_lift"] for r in res])),
            "auroc": float(np.nanmean([r["auroc"] for r in res])),
            "node_homophily_pos": h.get("node_homophily_class_1", np.nan),
            "node_homophily_neg": h.get("node_homophily_class_0", np.nan),
        })
    return pd.DataFrame(rows)


def prediction_frame(ctx: Ctx, model: str = BEST_MODEL,
                     feats: list | None = None,
                     seeds=HOUSE_SEEDS) -> pd.DataFrame:
    """Per-node TEST predictions pooled across seeds, with the confusion
    bucket. Pooling test folds is what makes the misclassification tables
    readable at all when a single fold holds ~40 positives."""
    feats = feats or FULL_BANK
    lab = ctx.tab[ctx.tab["labelled"]]
    frames = []
    for seed in seeds:
        part = stratified_split(lab["y"], seed)
        part_full = part.reindex(ctx.tab.index).fillna("unlabelled")
        p = MODELS[model](ctx, part_full, feats, seed)
        val, test = lab.index[part == "val"], lab.index[part == "test"]
        thr = tune_threshold(lab.loc[val, "y"].to_numpy(),
                             p.loc[val].to_numpy())
        f = pd.DataFrame({"author": test, "seed": seed,
                          "y": lab.loc[test, "y"].to_numpy(),
                          "p": p.loc[test].to_numpy(), "thr": thr})
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["pred"] = (out["p"] >= out["thr"]).astype(int)
    out["bucket"] = np.select(
        [(out["y"] == 1) & (out["pred"] == 1),
         (out["y"] == 0) & (out["pred"] == 1),
         (out["y"] == 1) & (out["pred"] == 0)],
        ["TP", "FP", "FN"], default="TN")
    return out


def confusion_table(pred: pd.DataFrame) -> pd.DataFrame:
    """The 2x2 confusion matrix, pooled over seeds."""
    ct = pd.crosstab(pred["y"], pred["pred"])
    ct.index = ["actual low", "actual HIGH"]
    ct.columns = [f"pred {c}" for c in ct.columns]
    return ct


def bucket_profiles(pred: pd.DataFrame, tab: pd.DataFrame,
                    cols: list | None = None) -> pd.DataFrame:
    """Mean attributes of the TP / FP / FN / TN groups. This is where the
    "what does the model confuse" story lives - e.g. FPs being
    high-degree loud accounts that simply never got a call right."""
    cols = cols or FULL_BANK
    j = pred.join(tab[cols], on="author")
    prof = j.groupby("bucket")[cols].mean().T
    prof["n"] = np.nan
    counts = pred["bucket"].value_counts()
    prof.loc["_n_nodes"] = [counts.get(c, 0) for c in prof.columns[:-1]] \
        + [np.nan]
    return prof


def worst_misses(pred: pd.DataFrame, board: pd.DataFrame,
                 n: int = 5) -> pd.DataFrame:
    """The top-n false negatives: the true HIGH authors the model scored
    LOWEST. Reading their store rows is the most informative single
    exhibit for what the features are blind to."""
    fn = (pred[pred["bucket"] == "FN"]
          .sort_values("p").drop_duplicates("author").head(n))
    keep = ["author", "n_calls", "n_judged", "hit_rate", "composite",
            "degree", "pagerank", "n_comments", "n_posts"]
    keep = [c for c in keep if c in board.columns]
    return fn[["author", "p"]].merge(board[keep], on="author", how="left")


# ---------------------------------------------------------------------------
# GOING FURTHER: label-ingredient sensitivity, a significance test that
# nothing else in the module supplies, and the two standing limitations
# of this design (correlated features, no unseen-author validation)
# ---------------------------------------------------------------------------
def feature_correlation(tab: pd.DataFrame,
                        feats: list | None = None) -> pd.DataFrame:
    """Spearman correlation of the feature bank.

    WHY this exhibit exists: the per-feature ablation carries an
    explicit caveat - when two features are near-duplicates, dropping
    either one looks harmless because the twin covers for it, so a
    single-feature ablation understates both. Printing the correlation
    matrix is what turns that caveat from a sentence into something the
    reader can check, and it is why the CATEGORY ablation is reported
    beside the per-feature one.
    """
    feats = feats or FULL_BANK
    return tab[feats].corr(method="spearman")


def composite_variants(board: pd.DataFrame) -> dict:
    """Alternative recipes for the composite score.

    Production uses 0.4*s_conf + 0.4*s_z + 0.2*s_enh. These variants
    re-mix the SAME three shrunk components, then min-max renormalise, so
    each one is a legitimate alternative definition of "influential"
    rather than a different measurement. They are the ingredient-level
    companion to the cut-level sensitivity in `label_regime_table`:
    together they answer "is the finding about the model, or about one
    particular arithmetic choice in the scoring rule?"
    """
    parts = board.set_index("author")[["s_conf", "s_z", "s_enh"]].fillna(0.0)
    recipes = {
        "production 0.4/0.4/0.2": (0.4, 0.4, 0.2),
        "equal 1/3 each": (1 / 3, 1 / 3, 1 / 3),
        "accuracy only (s_conf)": (1.0, 0.0, 0.0),
        "magnitude only (s_z)": (0.0, 1.0, 0.0),
        "drop enhancement 0.5/0.5": (0.5, 0.5, 0.0),
    }
    out = {}
    for name, (a, b, c) in recipes.items():
        raw = a * parts["s_conf"] + b * parts["s_z"] + c * parts["s_enh"]
        span = raw.max() - raw.min()
        out[name] = (raw - raw.min()) / (span if span else 1.0)
    return out


def label_ingredient_table(board: pd.DataFrame, calls: pd.DataFrame,
                           edges: pd.DataFrame, model: str = BEST_MODEL,
                           seeds=HOUSE_SEEDS,
                           n_positives: int | None = None) -> pd.DataFrame:
    """Re-run the model under each composite recipe, holding PREVALENCE
    fixed (top-`n_positives` labelled authors under each recipe) so the
    AP numbers are comparable - a lower cut would raise AP for free.
    Also reports the overlap with the headline label set, because a recipe
    that reshuffles who is positive but keeps the AP is a much stronger
    robustness result than one that barely changes the label."""
    base_tab = build_node_table(board, calls, regime=HEADLINE_REGIME)
    base_pos = set(base_tab.index[base_tab["y"] == 1])
    n_positives = n_positives or len(base_pos)
    labelled = board.set_index("author")["n_judged"].fillna(0) > 0
    rows = []
    for name, score in composite_variants(board).items():
        s = score[labelled.reindex(score.index).fillna(False)]
        pos = set(s.nlargest(n_positives).index)
        tab = base_tab.copy()
        tab["y"] = tab.index.isin(pos).astype(int)
        ctx = make_ctx(tab, edges)
        res = [_run_one(ctx, model, FULL_BANK, s_)
               for s_ in seeds]
        rows.append({
            "recipe": name, "positives": len(pos),
            "overlap_with_headline": len(pos & base_pos) / max(
                len(base_pos), 1),
            "ap": float(np.nanmean([r["ap"] for r in res])),
            "ap_std": float(np.nanstd([r["ap"] for r in res])),
            "ap_lift": float(np.nanmean([r["ap_lift"] for r in res])),
            "auroc": float(np.nanmean([r["auroc"] for r in res])),
        })
    return pd.DataFrame(rows)


def permutation_test(ctx: Ctx, model: str = BEST_MODEL,
                     feats: list | None = None, n_perm: int = 50,
                     seed: int = 42) -> dict:
    """Is the model's AP distinguishable from luck?

    Mean +/- std across the house seeds says how STABLE a number is but
    not whether it is REAL, and nothing else in the module supplies that
    test. This one does: shuffle the labels among labelled nodes
    (destroying any feature-label and graph-label relationship while
    keeping the class balance and the graph exactly as they are), refit,
    and record the AP.
    Doing that n_perm times builds the null distribution of "AP achievable
    on this data with no signal at all". The p-value is the share of
    permutations that match or beat the real AP; with n_perm draws the
    smallest reportable p is 1/(n_perm+1), and that floor is reported
    rather than rounded to zero.
    """
    feats = feats or FULL_BANK
    real = _run_one(ctx, model, feats, seed)["ap"]
    rng = np.random.default_rng(seed)
    lab_idx = ctx.tab.index[ctx.tab["labelled"]]
    null = []
    for _ in range(n_perm):
        tab = ctx.tab.copy()
        y = tab.loc[lab_idx, "y"].to_numpy().copy()
        rng.shuffle(y)
        tab.loc[lab_idx, "y"] = y
        c = Ctx(tab=tab, g=ctx.g, _ops=ctx._ops)      # same graph + ops
        null.append(_run_one(c, model, feats, seed)["ap"])
    null = np.array([v for v in null if np.isfinite(v)])
    ge = int((null >= real).sum())
    return {"model": model, "ap": float(real), "n_perm": int(len(null)),
            "null_mean": float(null.mean()) if null.size else np.nan,
            "null_p95": float(np.quantile(null, 0.95)) if null.size
            else np.nan,
            "p_value": (ge + 1) / (len(null) + 1) if null.size else np.nan,
            "p_floor": 1 / (len(null) + 1) if null.size else np.nan,
            "null": null}


def cohort_split(tab: pd.DataFrame, calls: pd.DataFrame,
                 train_frac: float = 0.6, val_frac: float = 0.2,
                 seed: int = 42) -> pd.Series:
    """A split by author TENURE instead of at random.

    The transductive, single-snapshot setup is the main limitation of
    this design: every labelled node was visible when the model was
    fitted, so nothing above shows the model working on an author it
    had never seen. Here the labelled authors are ordered by the date of
    their FIRST call and cut chronologically - the model learns on the
    established voices and is graded on authors who arrived later. That is
    the deployment question ("a new name shows up loud this week - is it
    worth listening to?"), and it is a strictly harder test than the
    random split because the late cohort has shorter records.
    """
    first = (calls.assign(date=pd.to_datetime(calls["date"],
                                             errors="coerce"))
             .groupby("author")["date"].min())
    lab = tab.index[tab["labelled"]]
    order = first.reindex(lab).fillna(pd.Timestamp.max).sort_values()
    n = len(order)
    a, b = int(train_frac * n), int((train_frac + val_frac) * n)
    part = pd.Series("test", index=order.index, dtype=object)
    part.iloc[:a] = "train"
    part.iloc[a:b] = "val"
    return part.reindex(tab.index).fillna("unlabelled")


def evaluate_cohort(ctx: Ctx, calls: pd.DataFrame,
                    feats: list | None = None,
                    models: list | None = TOURNAMENT) -> pd.DataFrame:
    """The tournament re-run on the tenure split. A model that survives
    here generalises to authors it has never seen; one that only works on
    the random split was leaning on the population it was fitted to."""
    feats = feats or FULL_BANK
    part = cohort_split(ctx.tab, calls)
    lab = ctx.tab[ctx.tab["labelled"]]
    p_lab = part.reindex(lab.index)
    rows = []
    for name in (models or list(MODELS)):
        p = MODELS[name](ctx, part, feats, 42)
        val = lab.index[p_lab == "val"]
        test = lab.index[p_lab == "test"]
        thr = tune_threshold(lab.loc[val, "y"].to_numpy(),
                             p.loc[val].to_numpy())
        r = _metrics(lab.loc[test, "y"].to_numpy(), p.loc[test].to_numpy(),
                     thr)
        r["model"] = name
        rows.append(r)
    cols = ["model", "ap", "ap_random", "ap_lift", "auroc", "precision",
            "recall", "accuracy", "tp", "fp", "fn", "test_pos", "test_n"]
    return (pd.DataFrame(rows)[cols]
            .sort_values("ap", ascending=False).reset_index(drop=True))


def community_positive_table(g, comm: pd.Series, tab: pd.DataFrame,
                             min_size: int = 25) -> pd.DataFrame:
    """Community structure taken one step further: are the positives
    CONCENTRATED in particular communities, or spread evenly?

    This matters for the model: if HIGH authors clustered into a few
    communities, community id alone would be a strong feature and a
    graph model would have an easy job. An even spread is the harder
    world - and it is the same message as the low positive-class
    homophily, seen at group scale instead of neighbour scale.
    """
    lab = tab[tab["labelled"]]
    c = comm.reindex(lab.index)
    df = pd.DataFrame({"community": c, "y": lab["y"]}).dropna()
    grp = df.groupby("community").agg(labelled=("y", "size"),
                                      positives=("y", "sum"))
    grp = grp[grp["labelled"] >= min_size]
    grp["prevalence"] = grp["positives"] / grp["labelled"]
    grp["vs_overall"] = grp["prevalence"] / (df["y"].mean() or np.nan)
    return grp.sort_values("prevalence", ascending=False)
