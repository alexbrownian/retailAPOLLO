# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 05 — The Influential-Users Model
#
# **This notebook:** the graph-learning half of Chan (2026) — *can
# HIGH-predictive authors be identified from their behaviour and their
# position in the reply graph, before reading their track record?* — run on
# RetailRadar's **own live influence store** (`data/reference/influence/`,
# live-only, zero-touch, text-free, pseudonymous).
#
# > **Run order note:** the store seeds itself on the first live
# > `python update_data.py` after the influence tracker shipped. If the
# > assert below fires, run the pipeline once and re-run this notebook.
# > Everything here is a **forward record from inception** — no backfill,
# > by desk decision.
#
# ## The port, section by section
#
# | Thesis | Theirs | Ours |
# |---|---|---|
# | §3 | semi-supervised node classification, 2,617 nodes, 13:1 imbalance | same task on the live store; unlabelled authors stay in the graph as context |
# | §4.6 | composite score (0.4/0.4/0.2), Bayesian shrinkage, HIGH ≥ 0.66 | already implemented verbatim in `influence.py` — the labels here |
# | §6.1 | MLP / LabelProp / GCN / GAT / GraphSAGE / H₂GCN / MixHop | random, feature-only MLP, structure-only LabelProp, **sage_lite** (one GraphSAGE mean-aggregation layer — the small-data version of their winner; their own GNNs reached only ~12% precision on 133 positives, a warning against model appetite exceeding label supply) |
# | §6.2 | stratified 60/20/20, seeds 42/100/2026, class weights, threshold = max precision s.t. recall ≥ 0.05 on validation, AP+AUROC on test | identical, via `analytics/influence_ml.py` |
# | §6.2 | leakage guard: labelling-pipeline features excluded | identical: composite / shrunk scores / hit-rate are the label's ancestry, never features |
# | §7.2 | ablation + random/DICE graph perturbation | category-level ablation + both perturbation modes |
#
# **Benchmark to beat / compare against (thesis Table 7.1):** GraphSAGE
# AP 0.140 ± 0.002, AUROC 0.632 ± 0.016 — ≈ +61% AP over their random
# baseline. Our store is younger and smaller; the honest expectation is
# *directional agreement* (structure + behaviour beats either alone),
# not their absolute numbers, until the store matures.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

def despine(ax, keep_bottom=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)

# %% [markdown]
# ## Load the live store

# %%
from analytics.influence import CALLS_PATH, SCORES_PATH, EDGES_PATH
from analytics.influence_ml import (build_node_table, evaluate,
                                    ablate_categories, perturbation_curve,
                                    BEHAVIOURAL, STRUCTURAL)

for p, what in [(SCORES_PATH, "author board"), (CALLS_PATH, "calls"),
                (EDGES_PATH, "reply edges")]:
    assert Path(p).exists(), (
        f"influence store missing ({what}: {p}).\n"
        "The store seeds itself on the first live `python update_data.py` "
        "run - do that once, then re-run this notebook. The board is a "
        "forward record from inception (no backfill, desk decision).")

board = pd.read_parquet(SCORES_PATH)
calls = pd.read_parquet(CALLS_PATH)
edges = pd.read_parquet(EDGES_PATH)
tab = build_node_table(board, calls)
n_lab, n_high = int(tab.labelled.sum()), int(tab.y.sum())
print(f"store: {len(tab)} authors ({n_lab} labelled, {n_high} HIGH tier "
      f"= {n_high/max(n_lab,1):.1%} positive class), "
      f"{len(edges):,} reply edges")
print(f"thesis reference: 2,617 nodes, 6.9% positive; interpret our "
      "numbers against store size accordingly")

# %% [markdown]
# ## The tournament (thesis §7.1 shape: every model vs random, AP primary)
#
# Small-store caution, stated before the table: with few HIGH-tier
# authors, test sets contain a handful of positives and AP/AUROC swing
# hard between seeds — read the mean ± std, and treat single-seed
# precision as anecdote (the thesis flags exactly this in §7.1.1).

# %%
res = evaluate(tab, edges)
res.round(3)

# %%
fig, ax = plt.subplots(figsize=(7.2, 3))
colors = {"sage_lite": C1, "mlp": C2, "labelprop": C3, "random": GRID}
sub = res.sort_values("ap")
ax.barh(sub.model, sub.ap, xerr=sub.ap_std.fillna(0), height=0.55,
        color=[colors.get(m, C4) for m in sub.model],
        error_kw=dict(ecolor=INK, lw=1))
rnd_ap = float(res.loc[res.model == "random", "ap"].iloc[0])
ax.axvline(rnd_ap, color=INK, lw=1, ls="--")
ax.text(rnd_ap, len(sub) - 0.3, " random floor", fontsize=8, color=INK)
prev = tab[tab.labelled].y.mean()
ax.axvline(prev, color=MUTED, lw=1, ls=":")
ax.text(prev, -0.45, " prevalence", fontsize=8, color=MUTED)
ax.set_title("Average precision by model (mean ± std over seeds "
             "42/100/2026)")
despine(ax)
plt.show()

# %% [markdown]
# ## Feature-category ablation (thesis §7.2.1, category level)
#
# Removing whole families sidesteps the correlated-single-feature caveat.
# The thesis's finding to compare against (§8.1): behavioural engagement
# + PageRank drive performance; raw degree *hurts* (their FP profile —
# structurally prominent, predictively weak).

# %%
abl = ablate_categories(tab, edges)
abl.round(3)

# %% [markdown]
# ## Graph perturbation (thesis §7.2.2)
#
# `random` rewiring must degrade a structure-using model; `DICE`
# (disconnect internally, connect externally) probes whether same-label
# neighbourhoods carry the signal. The thesis saw random hurt and DICE
# *help* (their graph's homophily was majority-class-driven) — whether
# ours reproduces that is an empirical question the cell answers.

# %%
pert = perturbation_curve(tab, edges)
fig, ax = plt.subplots(figsize=(7, 3.2))
for mode, c in (("random", C1), ("dice", C3)):
    sub = pert[pert["mode"] == mode]
    ax.errorbar(sub.rate, sub.ap, yerr=sub.ap_std, fmt="o-", color=c,
                lw=1.6, ms=4, capsize=2, label=mode)
ax.set_xlabel("fraction of edges perturbed")
ax.set_ylabel("sage_lite test AP")
ax.set_title("Robustness under graph perturbation (mean ± std over seeds)")
ax.legend(frameon=False)
despine(ax)
plt.show()

# %% [markdown]
# ## Cross-check: the model vs the board's own 'loud but wrong' flag
#
# The board flags top-quartile-PageRank / below-median-composite authors
# (the thesis's FP profile: hubs were 40% accurate vs 79% for quiet
# users). A sane model should assign LOWER P(HIGH) to flagged authors.

# %%
if "loud_but_wrong" in board.columns and board["loud_but_wrong"].any():
    from analytics.influence_ml import model_sage_lite, stratified_split
    lab = tab[tab.labelled]
    part = stratified_split(lab["y"], 42).reindex(tab.index).fillna(
        "unlabelled")
    p = model_sage_lite(tab, edges, part, BEHAVIOURAL + STRUCTURAL, 42)
    lbw = board.set_index("author")["loud_but_wrong"].reindex(tab.index)
    print(f"mean P(HIGH): loud-but-wrong {p[lbw == True].mean():.3f} vs "
          f"others {p[lbw != True].mean():.3f} "
          "(lower for flagged = model and board agree on the FP profile)")
else:
    print("no loud-but-wrong flags yet (young graph) - re-run once the "
          "store has a real-sized graph")

# %% [markdown]
# ## Verdict & maturity criterion (pre-stated)
#
# * The leaderboard, ablation and perturbation cells above re-render from
#   the live store on every run — this notebook is a **standing
#   experiment**, not a one-off result. The narrative to present: does
#   structure + behaviour beat either alone (the thesis's core claim),
#   and does the FP cross-check agree with the board's flag?
# * **Maturity criterion (recorded now, before the store is big):** model
#   outputs become decision-grade only once the store holds **≥ 130
#   labelled positives** — the scale at which the thesis's own best model
#   still only reached ~12% operating precision. Until then the model is
#   a research companion to the board, never a ranking the desk acts on.
# * The board itself (shrunk scores, tiers, loud-but-wrong) remains the
#   production surface on the dashboard's Influence tab — this model adds
#   the *"could we have known without the track record?"* claim, which is
#   the thesis's contribution.
