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
# # Notebook 05 — Can we tell who is worth listening to?
#
# **A full study of RetailRadar's own live influence store** — the network,
# eight model architectures, and the four hardest questions anyone would
# put to the result. The question is:
#
# > *Can HIGH-influence authors be identified from how they behave and where
# > they sit in the reply graph — **without** reading their track record?*
#
# And in desk words: **if a name is loud this week, can we tell in advance
# whether it is worth listening to — before we have seen whether they were
# right?**
#
# The store is `data/reference/influence/`: live-only, zero-touch,
# text-free, pseudonymous. No backfill (recorded decision) — it is a forward
# record from inception.
#
# ---
#
# ## How to read this notebook
#
# * **The verdict box below** is the whole answer, including the parts that
#   did not work. If you read nothing else, read that.
# * **Every section is a question**, and each one carries the same three
#   blocks: **WHY THIS** (what we did not know), **HOW IT WORKS** (the
#   mechanism, and where every number came from), **SO WHAT** (the finding
#   and what changed because of it).
# * **`IF ASKED` blocks** are the awkward questions a reviewer would put to
#   this work, with the answers. They are marked so you can find them.
# * **Nothing here is a threshold somebody liked the look of.** Every
#   constant is tagged with where it came from: LEARNED from data, DERIVED
#   from a definition, CONVENTION (a stated convention, and ablated),
#   GROUND TRUTH, or PROJECT DECISION.
# * Section 1 defines every term used, so no jargon is load-bearing.
#
# ---
#
# ## VERDICT BOX — read this first, then read why
#
# | Question | Answer | Evidence in this notebook |
# |---|---|---|
# | Does the feature bank carry signal? | **Yes, small but real.** | §7 rung 0 adopted; §14 permutation p = 0.005 |
# | Does the *graph* earn its complexity? | **No.** Every graph architecture loses to a plain linear model on paired splits. | §7 ladder — all graph rungs rejected |
# | Why does the graph fail? | Positive-class node homophily **0.09** vs 0.96 negative; DICE corruption *improves* AP. The reply graph is actively misleading for this label. | §3.6, §12 |
# | Was the first pass honest? | **No — half of it was circular.** `mean_conf` is an arithmetic factor of the label. Quarantined; the headline drops from AP 0.183 to 0.098. | §5 |
# | Does it generalise to authors it has never seen? | **No.** The shipped model falls *below* the random floor (AP 0.046 vs 0.048). The best of any model, `mlp` at 0.073, is barely clear of what a literally-random scorer got on the same 49-positive split (0.050). | §15 |
# | So what ships? | The **measured** leaderboard (composite score) on the dashboard. This model ships as a *research exhibit*, clearly labelled, and is **not** an input to the euphoria signal. | §16 |
#
# **Shipped configuration:** `FULL_BANK` (17 admissible features) +
# `logit`. Nothing else cleared the adoption rule.
#
# ---
#
# ## What this notebook covers, section by section
#
# | Section | The question it answers |
# |---|---|
# | §2 | Semi-supervised transductive node classification on the live store; unlabelled authors stay in the graph as context |
# | §3.1–3.3 | Network statistics, degree distribution, centralities |
# | §3.4 | Louvain communities (our own implementation, no networkx), modularity, inter-community share, positives-per-community |
# | §3.6 | Homophily, per class — the key diagnostic |
# | §4 | Labelling-criteria sensitivity: three label regimes plus a pre-registered maturity bar |
# | §5 | The leakage guard, and the extension of it that caught a circular feature |
# | §6 | Eight architectures — MLP / LabelProp / GCN / GAT / GraphSAGE / H₂GCN / MixHop / random — in linearised small-data form, plus a linear control. Stratified 60/20/20, seeds 42/100/2026, class-weighted loss, threshold = max precision s.t. recall ≥ 0.05 on val, AP + AUROC on test |
# | §7 | A **paired** adoption test: does any graph architecture beat the linear control on the same splits? |
# | §8 | Single run vs ensemble, and why it is a no-op for linear readouts |
# | §9 | The scoring rule itself: composite 0.4·s_conf + 0.4·s_z + 0.2·s_enh with Bayesian shrinkage, HIGH ≥ 0.66, implemented in `influence.py` — it *is* the label here |
# | §10 | Misclassification analysis, top-5 false negatives |
# | §11 | Feature ablation, correlation matrix, and bank candidates through the adoption rule with Bonferroni |
# | §12 | Random, DICE and degree-preserving-swap perturbation |
# | §13 | Class weighting, measured rather than assumed |
# | §14 | A permutation significance test |
# | §15 | Generalisation to authors the model has never seen |
#
# ### `IF ASKED` — "why is a notebook full of negative results worth keeping?"
#
# * Because each negative result is the reason something is *not* in
#   production, and that reasoning has to be auditable. "We don't use a
#   graph model" is an assertion; §7 plus §3.6 plus §12 is a defence.
# * Because the expensive failure mode in this project is shipping a
#   plausible model, not omitting one. Three of the sections below exist
#   purely to catch that: §5 caught it happening.

# %%
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# %matplotlib inline
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
RESEARCH_DIR = ROOT / "docs" / "research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
T0 = time.time()

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


pd.set_option("display.width", 190)
pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## 1 · Definitions — every term this notebook uses
#
# **WHY THIS**
#
# * A result nobody can define is a result nobody can check. Every term
#   below appears in a conclusion further down, so each one has to be
#   pinned before it is used.
# * These definitions are **imported, not retyped**, from
#   `analytics/plain_english.py` — the same file the dashboard reads. That
#   is deliberate: it makes it impossible for the notebook and the screen to
#   call the same quantity two different things.
#
# **SO WHAT**
#
# * Anywhere below that a chart says "usefulness score" and the code says
#   `composite`, they are the same number. The stored column names never
#   change; only the labels a human reads do.

# %%
from analytics.plain_english import (glossary_md,                 # noqa: E402
                                     censor_series)

# The terms this particular notebook leans on, in the order they first
# matter.  Listing them (rather than dumping the whole glossary) is the
# point: six relevant definitions get read, forty get skipped.
print(glossary_md([
    "average precision",
    "the random floor",
    "how well it sorts good from bad",
    "homophily",
    "DICE",
    "confidence interval",
    "paired seeds",
]))

# %% [markdown]
# ## 2 · What is in the store, and is it big enough to conclude anything?
#
# **WHY THIS**
#
# * Every number later in the notebook is bounded by how much labelled data
#   exists. Reading a result before reading the census is how people
#   over-trust small samples.
# * A study of this kind needs on the order of 130 positives before a paired
#   test can reject anything at all. If our store were smaller than that,
#   the honest move would be to stop here rather than report a number.
#
# **HOW IT WORKS**
#
# * `board` = one row per author with the shrunk composite score;
#   `calls` = one row per extracted (ticker, direction, stance) claim;
#   `edges` = one row per reply, `replier → author`.
# * The node table joins the FORECAST features computed from `calls` onto
#   the board and applies the label regime.
# * It **keeps unlabelled authors**. An author with no *judged* call carries
#   no label but still occupies a graph position and still contributes to
#   every neighbourhood average. That is what makes the setup
#   *transductive* rather than plain supervised — a stated design choice,
#   held fixed across every experiment below (CONVENTION).
# * The table is built on `WIDE_BANK` so the score-adjacent columns *exist*
#   and can be priced in §5. Every experiment names its own feature list,
#   and the shipped one never includes them.
#
# **SO WHAT**
#
# * If the positive count printed below is under 130, nothing in this
#   notebook is decision-grade — that bar is registered in §4 and enforced
#   in code, not left to judgement.

# %%
from analytics.influence import (CALLS_PATH, SCORES_PATH, EDGES_PATH,
                                 HIGH_TIER)                    # noqa: E402
from analytics import influence_graph as ig                     # noqa: E402
from analytics import influence_ml as ml                        # noqa: E402

for p, what in [(SCORES_PATH, "author board"), (CALLS_PATH, "calls"),
                (EDGES_PATH, "reply edges")]:
    assert Path(p).exists(), (
        f"influence store missing ({what}: {p}).\n"
        "The store seeds itself on the first `python update_data.py` run - "
        "comments are fetched on every live run under a measured page "
        "allowance - so do that once, then re-run this notebook. To seed it "
        "in one long sitting instead of across a few runs, use the "
        "unbudgeted runner: `python update_comments.py`. The board is a "
        "forward record from inception.")

board = pd.read_parquet(SCORES_PATH)
calls = pd.read_parquet(CALLS_PATH)
edges = pd.read_parquet(EDGES_PATH)

tab = ml.build_node_table(board, calls, feats=ml.WIDE_BANK)
ctx = ml.make_ctx(tab, edges)
g = ctx.g
lab = tab[tab["labelled"]]

census = pd.Series({
    "authors in store": len(board),
    "authors labelled (>=1 judged call)": int(tab["labelled"].sum()),
    f"positives (regime '{ml.HEADLINE_REGIME}')": int(tab["y"].sum()),
    "prevalence among labelled": round(float(lab["y"].mean()), 4),
    "calls extracted": len(calls),
    "reply records": len(edges),
    "unique undirected edges in graph": g.m,
    "graph nodes": g.n,
    "first call": str(pd.to_datetime(calls["date"]).min().date()),
    "last call": str(pd.to_datetime(calls["date"]).max().date()),
}, name="value")
print(census.to_string())
print(f"\nmaturity bar: >= {ml.MIN_POSITIVES} positives before any result "
      "here is decision-grade (registered in section 4)")

# %% [markdown]
# **SO WHAT — read the scale before reading any result.**
#
# * The store is large in nodes and thin in labels, and that cuts both ways.
# * **In our favour:** enough positives that the paired tests below have real
#   power to reject a challenger, rather than merely failing to detect a
#   difference.
# * **Against us:** a sparse, noisy graph, assembled from a live zero-touch
#   pull rather than a curated scrape. Density is the thing a graph model
#   most needs and the thing we have least of.
# * So every model claim below is stated as **ordering first, magnitude
#   second** — the ordering is the part that would survive on a different
#   slice of the same crowd; the magnitude is not.

# %% [markdown]
# ## 3 · What does the network look like — and can a graph model possibly work?
#
# **WHY THIS**
#
# * This is the section that *explains* the model results in §7, which is
#   why it comes before them rather than being skipped as description.
# * A graph whose useful people are surrounded by ordinary people cannot be
#   exploited by neighbourhood averaging — and every graph neural network is
#   neighbourhood averaging. We can know that **before fitting anything**.
# * Predicting a model failure from the data's structure, in advance, is the
#   difference between a negative result and an explanation.
#
# **SO WHAT (stated up front, then evidenced)**
#
# * The positives are spread evenly, not clustered (§3.4).
# * The positives' neighbours look nothing like them (§3.6).
# * Therefore §7's rejection of every graph architecture is predicted here,
#   not discovered there.
#
# ### 3.1 What shape is this network?
#
# **HOW IT WORKS**
#
# * Two columns are **sampled** rather than exact, for complexity reasons,
#   not convenience: average path length and betweenness both need
#   single-source shortest paths from every node — O(n·m), which on a
#   12k-node / 38k-edge graph is minutes, per run.
# * Brandes' algorithm from a random sample of `PIVOT_SAMPLE = 400` source
#   nodes estimates both at about 1/30 of the cost, without bias. Sampled
#   columns are named `_sampled` so no reader mistakes them for exact.
#   *(400 = PROJECT DECISION on runtime; the estimate is unbiased at any pivot
#   count, so this trades precision for minutes, not correctness for
#   convenience.)*
# * `small_world_sigma` = (C/C_rand) / (L/L_rand), against Erdős–Rényi
#   references C_rand = ⟨k⟩/n and L_rand = ln n / ln ⟨k⟩ (DERIVED — these
#   are the standard references, not fitted). σ ≫ 1 is the small-world
#   signature: tight local clustering *plus* global shortcuts.
#
# **SO WHAT**
#
# * This table is the reference the rest of section 3 reads against: density
#   tells us how much neighbourhood there is to average over, modularity
#   tells us whether the crowd splits into groups at all, and σ tells us
#   whether the graph has the small-world structure that makes message
#   passing worth attempting in the first place.

# %%
t = time.time()
bc = ig.betweenness(g, sample=ig.PIVOT_SAMPLE, seed=ig.GRAPH_SEED)
comm = ig.louvain(g)
stats = ig.network_stats(g, comm=comm, bc=bc)
print(f"(computed in {time.time() - t:.0f}s)\n")

side = pd.DataFrame({"ours": stats})
print(side.to_string())

# %% [markdown]
# ### 3.2 Do hubs even exist?
#
# **WHY THIS**
#
# * The whole exercise assumes some people are far more connected than
#   others. If they are not, there is nothing to find and the word
#   "influence" is empty.
# * This is the cheapest possible sanity check on the premise, so it goes
#   first.
#
# **HOW IT WORKS**
#
# * Two panels, because they answer different questions.
# * The **linear** panel shows where the mass is: almost everyone has a tiny
#   degree.
# * The **log-log CCDF** shows whether the *tail* is heavy. A heavy tail is
#   what makes hubs exist. If degree were Poisson there would be no hubs at
#   all.
# * The dashed line is straight in log-log space, i.e. a pure power law. It
#   is a **visual reference only**.
#
# **SO WHAT**
#
# * A tail this heavy means a small number of accounts absorb most of the
#   replies — so "who is influential" is a real question with a real answer,
#   and §3.3 onward is worth running.
#
# ### `IF ASKED` — "why not quote a power-law exponent?"
#
# * Because a defensible exponent needs a Clauset-style maximum-likelihood
#   fit with a fitted lower cutoff, plus a Kolmogorov–Smirnov
#   goodness-of-fit test to check a power law is even the right family.
# * A slope read off a chart by eye is precisely the kind of number this
#   project does not report. The picture supports the claim "the tail is
#   heavy"; it does not support "the exponent is 2.3", so only the first
#   claim is made.

# %%
dd = ig.degree_distribution(g)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
ax = axes[0]
ax.plot(dd["degree"], dd["p"], color=C1, lw=1.4)
ax.set_xlim(0, min(60, dd["degree"].max()))
ax.set_xlabel("degree k (replies received + given)")
ax.set_ylabel("P(k)")
ax.set_title("Degree distribution — linear")
despine(ax)

ax = axes[1]
pos = dd[(dd["degree"] > 0) & (dd["ccdf"] > 0)]
ax.loglog(pos["degree"], pos["ccdf"], ".", color=C1, ms=4)
ref_k = pos["degree"].to_numpy()
ref = pos["ccdf"].iloc[0] * (ref_k / ref_k[0]) ** -1.5
ax.loglog(ref_k, ref, ls="--", lw=1, color=MUTED,
          label="slope −1.5 (visual reference only)")
ax.set_xlabel("degree k (log)")
ax.set_ylabel("P(K ≥ k) (log)")
ax.set_title(f"Heavy tail — max degree {int(dd['degree'].max()):,}, "
             f"median {int(np.median(g.degree))}")
ax.legend(frameon=False, fontsize=8)
despine(ax, keep_bottom=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Are influential authors actually more central?
#
# **WHY THIS**
#
# - Every graph feature we could hand a model — degree, PageRank,
#   betweenness, k-core — is a *centrality*. If influential authors are
#   not more central than ordinary ones, none of those columns carry
#   information and the whole graph approach is dead before it is fitted.
# - The decision hanging on it: whether §6's tournament should expect the
#   graph models to beat the plain table model, or expect them to tie.
#
# **HOW IT WORKS**
#
# - Compute each centrality for every author, split the authors by label,
#   take the class means, and divide: `ratio_1_over_0` = mean(HIGH) ÷
#   mean(ordinary). One number per feature, and its null value is 1.
# - The 1.0 reference line is DERIVED, not chosen: a ratio of exactly 1
#   *is* "the two classes have the same mean", so there is nothing to
#   tune. Above 1, HIGH authors are more central; below 1, less.
# - `closeness` needs distances from a node to *everything*, so it is only
#   defined on the 400 sampled pivots (PROJECT DECISION, see §3.1) and is NaN
#   elsewhere. Its class means come from whichever pivots happened to be
#   labelled — read that row as indicative, not measured.
#
# **SO WHAT**
#
# - The ratios sit close to 1, which says the separation a classifier can
#   extract from pure position in the graph is small.
# - Nothing is dropped on the strength of this alone — a weak feature is
#   still a feature, and §11's ablation is where columns actually get
#   removed. What changes is the *expectation*: we go into §6 predicting
#   the graph models will not clear the plain table model, and §6 is then
#   a test of that prediction rather than a fishing trip.

# %%
cent = ig.centrality_table(g, bc=bc)
byc = ig.by_class(cent, lab["y"])
print(byc.round(6).to_string())

fig, ax = plt.subplots(figsize=(7.4, 3.2))
plot_c = byc.dropna(subset=["ratio_1_over_0"]).sort_values("ratio_1_over_0")
cols = [C1 if v >= 1 else C3 for v in plot_c["ratio_1_over_0"]]
ax.barh(plot_c.index, plot_c["ratio_1_over_0"], color=cols, height=0.6)
ax.axvline(1.0, color=INK, lw=1, ls="--")
ax.text(1.0, len(plot_c) - 0.4, "  no separation", fontsize=8, color=INK)
ax.set_xlabel("mean(HIGH) / mean(ordinary)")
ax.set_title("How much more central is an influential author? (ratio > 1 "
             "= HIGH authors are more central)")
despine(ax)
plt.show()

# %% [markdown]
# ### 3.4 Does the crowd split into groups, and do the influential
# authors bunch up inside them?
#
# **WHY THIS**
#
# - §3.3 asked whether influential authors stand out *individually*. This
#   asks whether they stand out *as a group*: if HIGH authors clustered
#   into two or three communities, then "which community are you in" would
#   be a strong feature on its own and any graph model would have an easy
#   job.
# - We also need to know the shape of the crowd for its own sake. Tight
#   silos and loose overlapping interest groups imply different things
#   about how a call spreads.
#
# **HOW IT WORKS**
#
# - Communities come from Louvain: start with every author alone, keep
#   moving authors into the neighbouring group that raises modularity
#   most, collapse each group into a single node, repeat. Modularity Q is
#   "edges inside groups, minus how many you would expect by chance" —
#   DERIVED, no threshold anywhere in it.
# - Louvain is implemented in `influence_graph` directly (multi-level,
#   self-loops preserved through the aggregation) rather than pulled from
#   networkx, so the *dashboard* needs no graph library at runtime. That
#   was licensed by an exact numerical agreement check against a reference
#   implementation in an earlier validation pass, not by taste.
# - The concentration test is the step that turns a description into a
#   decision. For every
#   community with ≥ 25 labelled authors (PROJECT DECISION — below ~25 the
#   prevalence estimate is one or two authors wide and the ratio is noise)
#   we compute its positive rate ÷ the overall positive rate. The 1.0 line
#   is DERIVED again: 1.0 *is* "this community is exactly average".
#
# **SO WHAT**
#
# - Read Q and the inter-community edge share together: a moderate Q with a
#   majority of edges *crossing* groups means loose interest clusters rather
#   than silos, which is what the printed numbers below show.
# - The positives are **spread, not concentrated** — the best community is
#   only a small multiple of the overall rate. So community membership is
#   not a shortcut feature, and this is the same message as low
#   positive-class homophily (§3.6), seen at group scale instead of
#   neighbour scale. Two independent measurements agreeing is why §7 can
#   state the diagnosis rather than guess at it.

# %%
crep = ig.community_report(g, comm)
print({k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in crep.items()})

cpt = ml.community_positive_table(g, comm, tab, min_size=25)
print(f"\ncommunities with >= 25 labelled authors: {len(cpt)}")
print(cpt.head(10).round(4).to_string())

sizes = comm.value_counts()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
ax = axes[0]
ax.loglog(np.arange(1, len(sizes) + 1), sizes.to_numpy(), ".", color=C1,
          ms=4)
ax.set_xlabel("community rank (log)")
ax.set_ylabel("members (log)")
ax.set_title(f"Community sizes — {comm.nunique():,} communities, "
             f"largest {crep['largest_community_share']:.1%} of authors")
despine(ax)

ax = axes[1]
if len(cpt):
    top = cpt.head(15).sort_values("vs_overall")
    ax.barh([str(i) for i in top.index], top["vs_overall"], color=C1,
            height=0.6)
    ax.axvline(1.0, color=INK, lw=1, ls="--")
    ax.set_xlabel("positive prevalence ÷ overall prevalence")
    ax.set_title(f"Positives are spread, not concentrated — best "
                 f"community only {cpt['vs_overall'].max():.2f}× overall")
else:
    ax.text(0.5, 0.5, "no community holds >= 25 labelled authors yet",
            ha="center", transform=ax.transAxes, color=MUTED)
    ax.set_title("Positive concentration by community")
despine(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.5 How do you draw a 12,000-node graph without lying?
#
# **WHY THIS**
#
# - Everything so far has been a summary statistic. A reader is entitled
#   to *see* the object those statistics describe before believing claims
#   about its shape.
# - The trap: you cannot draw this graph. Any layout of 12k nodes is a
#   hairball, and thinning it at random produces a picture of a structure
#   that does not exist. So the question is not "how do we draw it" but
#   "which sub-picture is a statement rather than a sample".
#
# **HOW IT WORKS**
#
# - **The k-core backbone** (left). The k-core is what remains after
#   repeatedly deleting every author with fewer than k neighbours. Every
#   author in a k-core *provably* has ≥ k neighbours inside it, so the
#   picture is a claim: "here is the densely interconnected heart of the
#   conversation." We take the deepest core that still has ≥ 40 members —
#   PROJECT DECISION, and the only judgement in the section: it trades depth
#   for a picture with enough nodes to read.
# - **One ego network** (right). The 1-hop neighbourhood of the single
#   highest-composite author. If the ball exceeds the 120-node cap the
#   *highest-strength* neighbours are kept, so it shows the busy part of
#   their neighbourhood — and the chart title says so rather than
#   pretending it is complete.
# - Layout is Fruchterman–Reingold: every pair of nodes repels with k²/d,
#   every edge attracts with d²/k, and the step size cools linearly to
#   zero so the picture settles instead of oscillating. Seeded, so it is
#   the same picture every run — a layout that moves between runs cannot
#   be cited.
#
# **SO WHAT**
#
# - Nothing is adopted or rejected here; this section changes no number.
#   It exists so that the structural claims in §3.4 and §3.6 have a
#   picture attached, and so a reviewer can see that the "hub with a crowd
#   around it" story is literally what the ego network looks like.
# - Read the left panel for *colour*: several distinct communities in the
#   core, not one blob, which is §3.4's modularity made visible. Read the
#   right panel for *shape*: one large node, many small ones, few edges
#   between the small ones — that is §3.6's homophily finding in advance.

# %%
sub, kcore = ig.kcore_subgraph(g)
nodes_f, links_f = ig.map_frames(sub, board=board, comm=comm)
top_author = board.nlargest(1, "composite")["author"].iloc[0]
ego = ig.ego_subgraph(g, top_author, radius=1, max_nodes=120)
ego_pos = ig.spring_layout(ego)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

ax = axes[0]
seg = links_f[["x0", "y0", "x1", "y1"]].to_numpy().reshape(-1, 2, 2)
ax.add_collection(LineCollection(seg, colors=GRID, linewidths=0.4,
                                 alpha=0.7))
sz = 12 + 90 * (nodes_f["strength_here"] /
                max(nodes_f["strength_here"].max(), 1))
pal = [C1, C2, C3, C4, "#7b4fa8", "#00838f", "#b25c00"]
cid = nodes_f["community"].astype("category").cat.codes
ax.scatter(nodes_f["x"], nodes_f["y"], s=sz,
           c=[pal[i % len(pal)] for i in cid], alpha=0.85,
           edgecolors="white", linewidths=0.4, zorder=3)
ax.set_title(f"The {kcore}-core backbone — {sub.n} authors, {sub.m} "
             f"replies\n(colour = Louvain community, size = reply volume)")
ax.set_xticks([]), ax.set_yticks([])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)

ax = axes[1]
el = ig.edge_list(ego)
if len(el):
    e = (el.join(ego_pos.rename(columns={"x": "x0", "y": "y0"}), on="u")
         .join(ego_pos.rename(columns={"x": "x1", "y": "y1"}), on="v"))
    seg = e[["x0", "y0", "x1", "y1"]].to_numpy().reshape(-1, 2, 2)
    ax.add_collection(LineCollection(seg, colors=GRID, linewidths=0.5,
                                     alpha=0.8))
is_c = ego_pos.index == top_author
ax.scatter(ego_pos.loc[~is_c, "x"], ego_pos.loc[~is_c, "y"], s=26,
           color=C1, alpha=0.85, edgecolors="white", linewidths=0.4,
           zorder=3, label="repliers")
ax.scatter(ego_pos.loc[is_c, "x"], ego_pos.loc[is_c, "y"], s=170,
           color=C4, edgecolors=INK, linewidths=0.9, zorder=4,
           label="the author")
ax.set_title(f"Ego network of the top-composite author\n{ego.n} nodes, "
             f"{ego.m} edges (top-strength neighbours if capped)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_xticks([]), ax.set_yticks([])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.6 Do influential authors have influential neighbours? — **the
# single most important number in this notebook**
#
# **WHY THIS**
#
# - Every graph neural network works by averaging a node's neighbours into
#   that node. That only helps if neighbours resemble each other on the
#   thing being predicted. The name for "neighbours resemble each other"
#   is **homophily**, and measuring it tells us in advance whether a graph
#   model *can* work here — before a single model is fitted.
# - This is the section that makes §7's rejection of every graph
#   architecture an explanation instead of an empirical shrug.
#
# **HOW IT WORKS**
#
# - **Edge homophily** = the share of edges whose two endpoints share a
#   label. On a 20:1 imbalanced problem this is near 1 no matter what,
#   because almost every edge joins two negatives. It is close to
#   uninformative, and is reported only so that the per-class figure below
#   cannot be accused of being the flattering half of a pair.
# - **Node homophily, per class** = for each author, the fraction of their
#   neighbours sharing their label; then averaged within each class.
#   Splitting by class is what rescues the measure from the imbalance, and
#   *this* is where the finding lives.
# - Both are DERIVED counts — there is no parameter in either, which is
#   part of why the number is worth leaning on.
#
# **SO WHAT**
#
# - The two classes come out at opposite ends of the scale: ordinary
#   authors sit near 1 (their neighbours are ordinary too), influential
#   authors sit near 0 (their neighbours are not influential). Read the
#   printed numbers below for the exact values.
# - That is a diagnosis, not a defect. Influential users are surrounded by
#   ordinary users *essentially by definition* — being replied to **by** the
#   crowd is what makes someone influential in the first place.
# - A neighbourhood-averaging architecture therefore averages a HIGH
#   author's signal *away*: the more aggressively it uses the graph, the
#   more it destroys the thing it is looking for.
# - What changes: §7 stops treating "graph model lost" as a tuning
#   failure. **Predicting the model result from the graph structure before
#   fitting anything is what turns a negative result into an
#   explanation** — and it is the reason we do not go back and try a
#   ninth architecture.
#
# **IF ASKED — "you are explaining away a bad result after the fact."**
#
# Fair challenge, and the ordering is the answer: this section runs before
# §6, its two measurements (neighbour-scale homophily here, group-scale
# concentration in §3.4) were specified together, and they agree. An
# after-the-fact excuse would have to be invented once the models had
# already lost; this one predicts the loss and is corroborated twice.

# %%
h = ig.homophily(g, lab["y"])
node_h = h.pop("node_homophily_series")
print({k: round(v, 4) for k, v in h.items()})

fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
ax = axes[0]
keys = ["edge_homophily", "node_homophily", "node_homophily_class_0",
        "node_homophily_class_1"]
nice = ["edge\n(all)", "node\n(all)", "node\nordinary", "node\nHIGH"]
vals = [h.get(k, np.nan) for k in keys]
ax.bar(nice, vals, color=[GRID, MUTED, C1, C3], width=0.6)
for i, v in enumerate(vals):
    if np.isfinite(v):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8,
                color=INK)
ax.set_ylim(0, 1.12)
ax.set_ylabel("share of neighbours sharing my label")
ax.set_title("Homophily — HIGH authors sit among ordinary users")
despine(ax)

ax = axes[1]
y_ser = lab["y"].reindex(node_h.index)
for cls, c, name in ((0, C1, "ordinary"), (1, C3, "HIGH")):
    v = node_h[(y_ser == cls) & node_h.notna()]
    if len(v):
        ax.hist(v, bins=np.linspace(0, 1, 26), color=c, alpha=0.65,
                density=True, label=f"{name} (n={len(v):,})")
ax.set_xlabel("node homophily")
ax.set_ylabel("density")
ax.set_title("The two classes live in different neighbourhoods")
ax.legend(frameon=False, fontsize=8)
despine(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4 · Does the answer depend on where we drew the line for
# "influential"?
#
# **WHY THIS**
#
# - "Influential" is not a fact in the data, it is a line we drew. Any
#   result that only holds at one line is an artefact of the line.
# - The decision hanging on it: which regime the rest of the notebook
#   reports as the headline — and therefore which AP number goes in the
#   verdict box.
#
# **HOW IT WORKS**
#
# - Three regimes, all fixed **before any of them was run** (this is the
#   part that makes it a test rather than a search):
#
# | Regime | Rule | Why this one |
# |---|---|---|
# | `standard` | composite ≥ 0.66 (`HIGH_TIER`) | the production tier the dashboard already uses |
# | `softened` | composite ≥ 0.50 | the round mid-point of a min-max-normalised score — a stated convention, chosen a priori and never tuned |
# | `prevalence` | top 6.95% of labelled authors | a **fixed reference prevalence**, held constant run to run. Because random AP *is* prevalence, pinning it pins the floor, so AP numbers from different vintages of the store stay directly comparable instead of only ordinally comparable |
#
# - **Why `softened` is the headline regime.** Not because it scores best
#   — because of a maturity bar registered before the store was large:
#   model output is only decision-grade at **≥ 130 labelled positives**.
#   `softened` is the regime that clears that
#   bar. Regimes that do not clear it are shown anyway, flagged
#   `powered = False`, and are used for no decision. Picking the headline
#   by *statistical power* rather than by *score* is the whole point: the
#   other rule would be choosing the winner after seeing the results.
#
# **SO WHAT**
#
# - Read the middle panel first: in every regime the model's AP sits above
#   its own random floor, and the floor moves with prevalence, so the two
#   bars have to be compared *within* a regime and never across.
# - Read the right panel second: the homophily gap of §3.6 survives all
#   three definitions. The core finding is therefore a property of the
#   crowd, not of where we drew the line.
# - What changes: `softened` is fixed as the reporting regime for the rest
#   of the notebook and for `docs/research/nb05_influence.json (written by this notebook; absent until it runs)`. Nothing
#   else in the pipeline changes — the dashboard's leaderboard still uses
#   the production `HIGH_TIER` of 0.66.
#
# **IF ASKED — "you report the regime with the most positives; isn't that
# just the one most likely to look good?"**
#
# More positives raise power, which makes a *true* effect easier to see —
# and equally makes a false one easier to rule out. The bar was set at
# ≥ 130 positives before we knew which of the three regimes would clear it,
# and the two that miss it are printed rather than hidden, with their scores
# visible next to the headline's.

# %%
t = time.time()
regimes = ml.label_regime_table(board, calls, edges)
print(f"(computed in {time.time() - t:.0f}s)\n")
print(regimes.round(4).to_string(index=False))
print(f"\nmaturity bar: >= {ml.MIN_POSITIVES} labelled positives")
print(f"headline regime: '{ml.HEADLINE_REGIME}'  |  HIGH_TIER={HIGH_TIER}, "
      f"SOFT_TIER={ml.SOFT_TIER}, reference prevalence="
      f"{ml.REFERENCE_PREVALENCE}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
r = regimes.set_index("regime")
ax = axes[0]
ax.bar(r.index, r["positives"], color=[C2 if p else C3
                                       for p in r["powered"]], width=0.55)
ax.axhline(ml.MIN_POSITIVES, color=INK, lw=1, ls="--")
ax.text(-0.4, ml.MIN_POSITIVES * 1.04, f" maturity bar "
        f"({ml.MIN_POSITIVES})", fontsize=8, color=INK)
ax.set_ylabel("labelled positives")
ax.set_title("Label supply (green = powered)")
despine(ax)

ax = axes[1]
x = np.arange(len(r))
ax.bar(x - 0.18, r["ap"], width=0.34, color=C1, label="model AP")
ax.bar(x + 0.18, r["ap_random"], width=0.34, color=GRID,
       label="random floor")
ax.set_xticks(x), ax.set_xticklabels(r.index)
ax.set_title("AP against its own floor")
ax.legend(frameon=False, fontsize=8)
despine(ax)

ax = axes[2]
ax.bar(x - 0.18, r["node_homophily_pos"], width=0.34, color=C3,
       label="HIGH")
ax.bar(x + 0.18, r["node_homophily_neg"], width=0.34, color=C1,
       label="ordinary")
ax.set_xticks(x), ax.set_xticklabels(r.index)
ax.set_ylim(0, 1.05)
ax.set_title("Homophily gap holds in every regime")
ax.legend(frameon=False, fontsize=8)
despine(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5 · **THE CIRCULARITY AUDIT** — is the model predicting the label, or
# is it just re-reading it?
#
# **WHY THIS**
#
# - A first pass produced a suspiciously good number. "Suspiciously good"
#   is not a feeling to be argued away, it is a hypothesis to be tested.
# - The specific worry: one of our features is an arithmetic *ingredient*
#   of the thing it is predicting. If so, the score is partly the model
#   reading its own answer back, and every number downstream is inflated.
# - It is the most important methodological check in the notebook, and it
#   moved the headline down.
#
# ### How the label is built
#
# In `analytics/influence.build_author_scores`, for every judged call:
#
# ```python
# conf   = |stance|                         # how loudly they said it
# y      = was the call right               # forward-return outcome
# s_conf = conf * y
# s_z    = conf * y * clip(1 + |z|, 0.1, 2.0)
# s_enh  = conf * enhanced
# composite = 0.4*s_conf + 0.4*s_z + 0.2*s_enh      # then min-max scaled
# tier      = composite >= HIGH_TIER
# ```
#
# Every component of the label is `conf × (something about correctness)`.
#
# ### Why `mean_conf` is not an admissible feature
#
# **HOW IT WORKS**
#
# - `mean_conf` is the author's mean `|stance|` — **literally one of the
#   two multiplicands of the quantity being thresholded**.
# - It never touches `y`, so it passes a naive leakage check: nothing
#   about forward returns enters it. That is exactly what makes it
#   dangerous — the standard test clears it.
# - But it is an *arithmetic factor of its own target*. A feature like
#   that will look predictive on **any** dataset, including one with no
#   signal whatsoever, because varying it moves the label mechanically.
# - `stance_sd` is the second moment of the same variable and inherits the
#   same defect in weaker form, so it is quarantined with it.
# - The tell: in the first pass `mean_conf` was, by a wide margin, the
#   single most useful feature in the bank — −0.077 AP when removed, three
#   times the next one.
#
# ### The response: quarantine, then **price** it
#
# - The two columns move to a named `SCORE_ADJACENT` bank and are excluded
#   from the default feature set.
# - The *cost* of excluding them is then measured with the same paired
#   10-seed test used for every other decision in this notebook, and
#   printed.
#
# **SO WHAT**
#
# - About half of the first pass's apparent skill was label circularity.
#   The headline AP falls, and stays fallen.
# - What changes: everything downstream runs on `FULL_BANK` (17 admissible
#   features). `WIDE_BANK` (19) is reported once, here, and never ships.
# - Deleting the two columns silently would have been indistinguishable
#   from never having noticed. Quoting the price is what makes the lower
#   headline defensible rather than merely modest.
#
# **IF ASKED — "you refused a feature that genuinely improves the score.
# Isn't that leaving performance on the table?"**
#
# It improves the *measured* score, which is not the same thing. The
# quantity we care about is skill on an author whose composite we have not
# computed yet; a factor of the composite cannot supply that, it can only
# reconstruct it. The right-hand panel shows the gain is consistent across
# splits — consistency is what you would expect from an identity, not
# evidence against circularity.

# %%
print("admissible bank (FULL_BANK, %d features):" % len(ml.FULL_BANK))
for name, blk in ml.CATEGORIES.items():
    print(f"  {name:12s} {blk}")
print(f"\nquarantined  SCORE_ADJACENT {ml.SCORE_ADJACENT}")
print(f"banned       LEAKAGE        {sorted(ml.LEAKAGE)}")

t = time.time()
circ = ml.paired_ap_test(ctx, "logit", "logit", feats_a=ml.FULL_BANK,
                         feats_b=ml.WIDE_BANK)
print(f"\n(computed in {time.time() - t:.0f}s)")
print("=== price of the two score-adjacent features (10 paired seeds) ===")
print({k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in circ.items()})

wide_seeds = ml.per_seed_ap(ctx, "logit", ml.WIDE_BANK, ml.ADOPTION_SEEDS)
full_seeds = ml.per_seed_ap(ctx, "logit", ml.FULL_BANK, ml.ADOPTION_SEEDS)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
ax = axes[0]
ax.bar(["admissible\n(FULL_BANK, 17)", "+ score-adjacent\n(WIDE_BANK, 19)"],
       [circ["ap_a"], circ["ap_b"]], color=[C1, C3], width=0.5)
floor = float(lab["y"].mean())
ax.axhline(floor, color=INK, lw=1, ls="--")
ax.text(-0.45, floor * 1.06, f" random floor = prevalence {floor:.3f}",
        fontsize=8, color=INK)
for i, v in enumerate([circ["ap_a"], circ["ap_b"]]):
    ax.text(i, v + 0.004, f"{v:.3f}  (lift {v / floor - 1:+.0%})",
            ha="center", fontsize=8, color=INK)
ax.set_ylabel("test AP")
ax.set_title("About half the first pass was label circularity")
despine(ax)

ax = axes[1]
sd = (wide_seeds - full_seeds).sort_values()
ax.barh([str(s) for s in sd.index], sd.to_numpy(), color=C3, height=0.6)
ax.axvline(0, color=INK, lw=1)
ax.axvline(circ["mean_diff"], color=C2, lw=1.4, ls="--",
           label=f"mean {circ['mean_diff']:+.3f}")
ax.axvspan(circ["ci_lo"], circ["ci_hi"], color=C2, alpha=0.12,
           label=f"95% CI [{circ['ci_lo']:+.3f}, {circ['ci_hi']:+.3f}]")
ax.set_xlabel("AP gain from adding the score-adjacent pair")
ax.set_ylabel("split seed")
ax.set_title(f"Same direction on {circ['wins']}/10 paired splits")
ax.legend(frameon=False, fontsize=8, loc="lower right")
despine(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# **The honest headline, therefore:** everything from here on uses
# `FULL_BANK`. `WIDE_BANK` is reported once, here, and never shipped.

# %% [markdown]
# ## 6 · Which architecture actually wins?
#
# **WHY THIS**
#
# - We now know what the graph looks like and which features are
#   admissible. The open question is whether *using the graph* beats
#   *ignoring it* — i.e. whether any of the eight architectures clears a
#   plain logistic regression on the author's own behaviour.
# - The decision hanging on it: what ships. If the graph adds nothing, the
#   dashboard's leaderboard should be computed the cheap way.
#
# **HOW IT WORKS**
#
# - Eight architectures, in the linearised small-data forms that make
#   sense on this label supply. Each differs from the others **only in
#   what goes into the feature matrix**; the read-out is the same
#   class-weighted logistic regression throughout.
# - That constraint is deliberate and is the section's main design choice:
#   it makes the tournament a comparison of *architectures* rather than of
#   optimisers, tuning budgets or luck. A win here cannot be bought with a
#   better learning rate.
#
# | Model | Role | What it sees |
# |---|---|---|
# | `random` | floor | uniform noise. Its AP **is** the prevalence — the floor every other number is quoted against |
# | `logit` | control | the author's own features, no graph. The simplest thing that could work |
# | `mlp` | no graph | own features, one 16-unit hidden layer. "Behaviour alone, non-linearly" |
# | `labelprop` | graph only | no features at all. "Structure alone" |
# | `sage_lite` | graph | GraphSAGE, one mean-aggregation layer: `[self ‖ mean(neighbours)]` |
# | `gcn_lite` | graph | GCN in its linear (SGC) form: propagate twice through the symmetric-normalised self-looped adjacency |
# | `mixhop_lite` | graph | *concatenates* hop powers instead of composing them, so hop-0/1/2 can get different — even opposite-signed — weights |
# | `h2gcn_lite` | graph | ego / 1-hop / 2-hop-excluding-1-hop kept separate. Designed for **heterophily**, which §3.6 says we have |
#
# - **GAT is deliberately absent.** Attention has to *learn* edge weights,
#   and with ~250 positives that is more parameters than evidence — it
#   would be fitted noise wearing an architecture's name. `mixhop_lite`
#   covers the same ground (hops weighted differently) at a fraction of the
#   parameter count, and the substitution is named here rather than left
#   for a reader to notice.
# - **Protocol, identical for every entrant:** stratified 60/20/20 over
#   labelled nodes; seeds 42/100/2026 (CONVENTION — the house seeds used
#   throughout this project, fixed so results are reproducible);
#   class-weighted loss;
#   threshold chosen on **validation** as max precision subject to recall
#   ≥ 5%; AP and AUROC reported on **test**, never on validation.
# - AP is the primary metric because on a rare-positive problem its random
#   baseline is exactly the prevalence — which is what lets `ap_lift`
#   stay meaningful as the store's balance shifts over time.
#
# **SO WHAT**
#
# - Read the leaderboard for *order*, not for gaps: SAGE on top, GCN under
#   it, feature-only MLP under that, structure-only LabelProp at the bottom.
#   That ordering is the one the graph-learning literature would predict on
#   a graph of this kind, which is a reassurance the harness is sound — not
#   yet evidence that any of it beats the control.
# - Nothing is adopted on the strength of this chart. The error bars are a
#   spread over three seeds, and a spread is not a test — §7 runs the
#   paired test that actually decides.

# %%
t = time.time()
res = ml.evaluate(ctx, feats=ml.FULL_BANK)
print(f"(computed in {time.time() - t:.0f}s)\n")
print(res.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.6, 3.4))
sub = res.sort_values("ap")
colours = {"random": GRID, "logit": C2, "sage_lite": C1}
ax.barh(sub["model"], sub["ap"], xerr=sub["ap_std"].fillna(0), height=0.6,
        color=[colours.get(m, C4) for m in sub["model"]],
        error_kw=dict(ecolor=INK, lw=1, capsize=2))
rnd = float(res.loc[res["model"] == "random", "ap"].iloc[0])
ax.axvline(rnd, color=INK, lw=1, ls="--")
ax.text(rnd, len(sub) - 0.35, "  random floor", fontsize=8, color=INK)
ax.set_xlabel("test AP (mean ± std over seeds 42/100/2026)")
ax.set_title("Leaderboard — read the ORDERING; §7 asks whether the "
             "MARGINS are real")
despine(ax)
plt.show()

# %% [markdown]
# **What this table does and does not say**
#
# - It **does** say the models rank in the expected order: SAGE on top, GCN
#   below it, feature-only MLP below that, structure-only LabelProp at the
#   bottom.
# - It does **not** say SAGE is better than the linear control. A mean
#   over three seeds with a ~50-positive test fold moves AP by more than
#   the gaps in this table — the differences are inside the noise.
# - A leaderboard is a hypothesis generator. §7 is the test.

# %% [markdown]
# ### 6.1 At the threshold we would actually use, how wrong is it?
#
# **WHY THIS**
#
# - AP is a single number summarising a whole curve. A desk does not
#   operate on a curve, it operates at *one point* — so the curve and that
#   point both have to be shown.
# - It is also the honesty check on AUROC: a respectable-sounding AUROC
#   and a poor top-of-ranking can coexist, and only the PR curve shows it.
#
# **HOW IT WORKS**
#
# - **ROC** plots true-positive rate against false-positive rate. It is
#   prevalence-free, which makes it comparable across datasets — and
#   flattered by a huge negative class, which makes it optimistic here.
# - **PR** plots precision against recall: at any recall you care about,
#   what fraction of the flagged authors are real? That is the quantity a
#   desk experiences. Its no-skill baseline is the prevalence (DERIVED),
#   drawn as the dashed line.
# - The marked point is the *chosen operating threshold* — max precision
#   subject to recall ≥ 5%, picked on **validation** and never on test.
#   Choosing it on test is the single easiest way to inflate a result, so
#   the split is stated rather than assumed.
#
# **SO WHAT**
#
# - AUROC lands around 0.67: the model sorts meaningfully better than
#   chance.
# - The PR curve is the sobering one: precision at the operating point is
#   ~0.12. Roughly one flagged author in eight is genuinely influential —
#   a real lift over the ~0.047 floor, and nowhere near a signal you would
#   trade off.
# - What changes: this is the number the dashboard's Influence tab is
#   captioned with, and the reason the tab is labelled *information, not a
#   signal*.

# %%
pred = ml.prediction_frame(ctx, model=ml.BEST_MODEL, feats=ml.FULL_BANK)
one = pred[pred["seed"] == ml.HOUSE_SEEDS[0]]
roc = ml.roc_points(one["y"].to_numpy(), one["p"].to_numpy(),
                    float(one["thr"].iloc[0]))
prc = ml.pr_points(one["y"].to_numpy(), one["p"].to_numpy())
op = roc[roc["operating"]].iloc[0] if roc["operating"].any() else None

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
ax.plot(roc["fpr"], roc["tpr"], color=C1, lw=1.6)
ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
if op is not None:
    ax.plot(op["fpr"], op["tpr"], "o", color=C4, ms=8, zorder=5,
            label=f"operating thr {float(one['thr'].iloc[0]):.3f}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_xlabel("false positive rate"), ax.set_ylabel("true positive rate")
auroc = float(res.loc[res["model"] == ml.BEST_MODEL, "auroc"].iloc[0])
ax.set_title(f"ROC — AUROC {auroc:.3f}")
despine(ax)

ax = axes[1]
ax.plot(prc["recall"], prc["precision"], color=C1, lw=1.6)
ax.axhline(floor, color=INK, lw=1, ls="--")
ax.text(0.55, floor * 1.08, f"no-skill = prevalence {floor:.3f}",
        fontsize=8, color=INK)
ap_best = float(res.loc[res["model"] == ml.BEST_MODEL, "ap"].iloc[0])
ax.set_xlabel("recall"), ax.set_ylabel("precision")
ax.set_title(f"Precision–recall — AP {ap_best:.3f} "
             f"(lift {ap_best / floor - 1:+.0%})")
despine(ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7 · **THE ADOPTION LADDER** — is any gap in §6 real?
#
# **WHY THIS**
#
# - A leaderboard reports mean ± std over three seeds. That tells you how
#   *stable* a number is. It does not tell you whether one model is
#   *better* than another, because the two numbers came from different
#   random splits — and the split lottery is by far the largest source of
#   variance on a fold this small.
# - This is the section that decides what ships. Everything before it
#   describes; this one commits.
#
# **HOW IT WORKS**
#
# - The rule, **written down before it was run**: for a rung
#   `(incumbent, challenger)`, run both on the **identical** split for
#   each of 10 seeds, take the 10 paired AP differences, and form a
#   t-based 95% CI on the mean difference. **Adopt only if the whole CI is
#   above zero.**
# - Pairing is what makes 10 seeds enough. Unpaired, the split variance
#   swamps every margin in §6's table. Paired, the split variance cancels
#   *exactly*, because both candidates see the same split — the only
#   difference left between the two runs is the model.
# - 10 seeds is a PROJECT DECISION and the cost is stated: it fixes the
#   smallest difference we can resolve at roughly ±0.02 AP. A gap smaller
#   than that is not "absent", it is "not measurable here", and the
#   notebook says so rather than claiming a null.
# - The ladder itself (fixed in `influence_ml.LADDER`):
#   - Rung 0 — `random → logit`: do the features carry *any* signal?
#   - Rungs 1–6 — `logit → {mlp, labelprop, sage_lite, gcn_lite,
#     mixhop_lite, h2gcn_lite}`: does the extra machinery beat the plain
#     linear model on the same splits?
# - Every later rung is measured against the *linear control*, not against
#   the previous winner. That is the **parsimony rule**: ship the simplest
#   model the evidence supports. It is the same rule the euphoria side
#   used to reject gradient boosting — reused rather than reinvented, so
#   the project has one standard of proof and not two.
#
# **SO WHAT** — spelled out in the cell below the chart.

# %%
t = time.time()
ladder = ml.adoption_ladder(ctx, feats=ml.FULL_BANK)
shipped = ml.choose_model(ladder)
print(f"(computed in {time.time() - t:.0f}s)\n")
print(ladder.round(4).to_string(index=False))
print(f"\nPARSIMONY RULE ships: {shipped!r}   "
      f"(module binding BEST_MODEL={ml.BEST_MODEL!r})")
assert shipped == ml.BEST_MODEL, (
    "the ladder no longer agrees with the shipped binding - re-read the "
    "table before changing anything downstream")

fig, ax = plt.subplots(figsize=(8, 3.6))
lp = ladder.iloc[::-1].reset_index(drop=True)
ypos = np.arange(len(lp))
for i, row in lp.iterrows():
    c = C2 if row["adopt"] else C3
    ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=c, lw=2.4,
            solid_capstyle="round")
    ax.plot(row["mean_diff"], i, "o", color=c, ms=7, zorder=4)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(ypos)
ax.set_yticklabels([f"{r.baseline} → {r.candidate}"
                    for r in lp.itertuples()])
ax.set_xlabel("paired AP difference, 95% CI (10 seeds)")
ax.set_title("Only rung 0 clears zero — the graph does not earn its "
             "complexity\n(green = adopted, pink = rejected)")
despine(ax)
plt.show()

# %% [markdown]
# ### SO WHAT — what just happened, in words
#
# * **`random → logit` is adopted.** The features do carry signal. This is
#   the result that keeps the whole exercise alive.
# * **Every graph rung is rejected.** `sage_lite` — the model that *tops*
#   the §6 leaderboard — has a paired margin over `logit` whose CI
#   straddles zero. `h2gcn_lite` is significantly *worse*. `labelprop`
#   (structure with no features) is worst of all.
# * **What changes:** the shipped model is the **linear one**. The
#   dashboard computes its leaderboard without a graph library at
#   runtime, and the fact that the winner is the simplest member of the
#   tournament is the finding, not a shortcut.
#
# **IF ASKED — "the leaderboard in §6 said the graph won. Which is it?"**
#
# Both tables are correct; they answer different questions. §6 ranks means,
# §7 tests differences. The gap is real in the mean and absent under
# pairing, which is the classic signature of a difference that is smaller
# than the split lottery. §3.6 predicted exactly this — on a graph where the
# positive class has node homophily ~0.09, neighbourhood averaging destroys
# the signal it is meant to aggregate. An unpaired leaderboard, read alone,
# is how a project ends up shipping a graph library it does not need.

# %% [markdown]
# ## 8 · Would averaging several re-fits help?
#
# **WHY THIS**
#
# - Averaging predictions over re-fits is the standard cheap way to damp
#   seed variance. If it helps here, it costs almost nothing and we should
#   ship it; if it does not, we should be able to say *why* rather than
#   just reporting a null.
#
# **HOW IT WORKS**
#
# - Run each model once, then run it again averaging predictions over
#   seeds 42/100/2026, and difference the test AP.
# - One structural point first, because it stops the result being
#   over-read: **holding the split fixed, every model here except `mlp` is
#   deterministic.** The linear read-outs are convex problems solved to
#   convergence — refitting under a different `random_state` returns the
#   same coefficients, so averaging them is arithmetically the identity.
# - `mlp` is the only genuinely stochastic member (random weight init,
#   early stopping on a random inner split), so it is the only row where
#   ensembling can do anything at all.
#
# **SO WHAT**
#
# - `logit`'s `d_ap` is exactly 0.0000, which is not a weak result — it is
#   the determinism argument above, confirmed numerically. A table of nine
#   identical rows presented as a "robustness check" would have been
#   theatre; printing two rows and explaining why seven are uninformative
#   is the honest version.
# - **Nothing changes.** The shipped model is deterministic given the
#   split, so there is no ensemble to adopt.

# %%
t = time.time()
single = ml.evaluate(ctx, feats=ml.FULL_BANK, models=["mlp", ml.BEST_MODEL])
ens = ml.evaluate(ctx, feats=ml.FULL_BANK, models=["mlp", ml.BEST_MODEL],
                  ensemble_seeds=ml.HOUSE_SEEDS)
cmp = (single.set_index("model")[["ap", "ap_std", "auroc"]]
       .join(ens.set_index("model")[["ap", "ap_std", "auroc"]],
             lsuffix="_single", rsuffix="_ensemble"))
cmp["d_ap"] = cmp["ap_ensemble"] - cmp["ap_single"]
print(f"(computed in {time.time() - t:.0f}s)\n")
print(cmp.round(4).to_string())
print("\nlogit d_ap == 0 confirms the determinism argument above; the mlp "
      "row is the only informative one.")

# %% [markdown]
# ## 9 · Is the finding about the model, or about one arithmetic choice in
# the scoring rule?
#
# **WHY THIS**
#
# - §4 varied *where the line was drawn*. It did not vary the **recipe**
#   that produces the score being cut — the 0.4 / 0.4 / 0.2 mix. If the
#   result only holds for that one weighting, it is a property of an
#   arbitrary formula rather than of the crowd.
#
# **HOW IT WORKS**
#
# - Re-mix the same three shrunk components into five alternative
#   definitions of "influential", renormalise, re-label, re-run.
# - Prevalence is **held fixed** (top-N labelled authors under each
#   recipe). This matters: without it, a recipe with a lower effective cut
#   would raise AP for free — because the random floor *is* prevalence —
#   and the comparison would be meaningless.
# - The column that decides it is `overlap_with_headline`: the share of
#   the headline's positives that this recipe also calls positive. A
#   recipe that reshuffles *who* is positive and still lands on a similar
#   AP is a much stronger robustness result than one that barely changes
#   the label set — because the second kind **cannot fail**.
#
# **SO WHAT**
#
# - AP holds above the floor even for recipes with low overlap, so the
#   skill is not an artefact of the specific 0.4 / 0.4 / 0.2 weighting.
# - **Nothing changes.** The production recipe keeps its 0.4 / 0.4 / 0.2
#   mix — this section buys the right to keep it, not a reason to alter it.
#   Read the chart left-to-right: the interesting points are the ones far
#   to the *left* (a different set of people) that still sit well above
#   the floor.

# %%
t = time.time()
ing = ml.label_ingredient_table(board, calls, edges)
print(f"(computed in {time.time() - t:.0f}s)\n")
print(ing.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.scatter(ing["overlap_with_headline"], ing["ap"], s=90, color=C1,
           edgecolors="white", zorder=3)
for _, r_ in ing.iterrows():
    ax.annotate(r_["recipe"], (r_["overlap_with_headline"], r_["ap"]),
                textcoords="offset points", xytext=(7, 4), fontsize=8,
                color=MUTED)
ax.axhline(floor, color=INK, lw=1, ls="--")
ax.text(0.02, floor * 1.05, "random floor", fontsize=8, color=INK)
ax.set_xlim(0, 1.25)
ax.set_xlabel("share of headline positives this recipe also calls positive")
ax.set_ylabel("test AP")
ax.set_title("Label-ingredient sensitivity — AP holds even where the "
             "recipe reshuffles who is positive")
despine(ax)
plt.show()

# %% [markdown]
# ## 10 · When the model is wrong, *how* is it wrong?
#
# **WHY THIS**
#
# - A precision of ~0.12 says the model is wrong most of the time. It does
#   not say whether the mistakes are random or systematic — and a
#   systematic mistake is the one you can either fix or warn about.
# - The specific hypothesis, stated before looking: false positives are
#   *structurally prominent* accounts — high degree, high PageRank — who
#   simply were not right about anything.
#
# **HOW IT WORKS**
#
# - Test folds hold only ~50 positives, so one fold's confusion matrix is
#   unreadable. Predictions are pooled across the three seeds (each
#   labelled author appears once per seed in which it landed in test),
#   which is what makes the bucket profiles stable enough to interpret.
# - Each confusion bucket (TP / FP / FN / TN) gets a mean feature profile,
#   divided through by the TN group's mean so the chart reads as "how many
#   times a *typical* author is this bucket". The 1.0 line is DERIVED —
#   it is the typical author by construction.
# - The independent cross-check: our board already carries a
#   `loud_but_wrong` flag built from a different calculation entirely, so
#   the model's FPs and the board's flag can be compared without either
#   having been fitted to the other.
#
# **SO WHAT**
#
# - The FP profile is loud: false positives sit well above the typical
#   author on degree, PageRank and comment volume, and *not* above on the
#   forecast-quality columns. The model is confusing **prominence** with
#   **skill** — and that is the crowd's property, not the model's: the
#   loudest accounts in this store are not the most accurate ones.
# - The cross-check agrees: mean P(HIGH) is lower for `loud_but_wrong`
#   authors than for the rest, which is the direction it should be.
# - **What changes:** nothing in the model — this is a property of the
#   crowd, not a bug. It changes the *dashboard copy*: the Influence tab
#   says outright that a big reply count is not evidence of being right,
#   because that is the specific mistake a reader of that tab would
#   otherwise make.

# %%
ct = ml.confusion_table(pred)
print(ct.to_string())
tp, fp = int(ct.iloc[1, 1]), int(ct.iloc[0, 1])
fn = int(ct.iloc[1, 0])
print(f"\npooled precision {tp / max(tp + fp, 1):.3f}, "
      f"recall {tp / max(tp + fn, 1):.3f}")

prof = ml.bucket_profiles(pred, tab)
print("\nbucket profiles (mean feature value per confusion bucket):")
print(prof.round(3).to_string())

show = [c for c in ["degree", "pagerank", "weighted_degree", "n_comments",
                    "n_calls", "replies", "followers", "active_days"]
        if c in prof.index]
fig, ax = plt.subplots(figsize=(8.4, 3.6))
x = np.arange(len(show))
w = 0.2
for i, (bkt, c) in enumerate((("TP", C2), ("FP", C3), ("FN", C4),
                              ("TN", GRID))):
    if bkt not in prof.columns:
        continue
    v = prof.loc[show, bkt] / prof.loc[show, "TN"].replace(0, np.nan)
    ax.bar(x + (i - 1.5) * w, v, width=w, color=c, label=bkt)
ax.axhline(1.0, color=INK, lw=1, ls="--")
ax.set_xticks(x), ax.set_xticklabels(show, rotation=30, ha="right")
ax.set_ylabel("mean ÷ mean of TN group")
ax.set_title("What the model confuses — FPs are structurally loud, not "
             "predictively good")
ax.legend(frameon=False, fontsize=8, ncol=4)
despine(ax)
plt.show()

# %%
# Handles are masked for display only (`censor_series`): the offending
# spans collapse to ** and everything else survives, so two authors stay
# distinguishable on the page. The join keys upstream are the true handles.
_miss = ml.worst_misses(pred, board, n=5).round(4)
_miss["author"] = censor_series(_miss["author"])
print("worst false negatives — the true HIGH authors the model ranked "
      "LOWEST:")
print(_miss.to_string(index=False))

if "loud_but_wrong" in board.columns and board["loud_but_wrong"].any():
    lbw = board.set_index("author")["loud_but_wrong"]
    j = pred.assign(flag=lbw.reindex(pred["author"]).to_numpy())
    m_flag = j.loc[j["flag"] == True, "p"].mean()          # noqa: E712
    m_rest = j.loc[j["flag"] != True, "p"].mean()          # noqa: E712
    print(f"\nboard cross-check — mean P(HIGH): loud-but-wrong "
          f"{m_flag:.3f} vs others {m_rest:.3f}  "
          f"({'AGREE' if m_flag < m_rest else 'DISAGREE'}: the model should "
          "score flagged hubs lower)")
else:
    print("\nno loud_but_wrong flags in the board yet")

# %% [markdown]
# ## 11 · Which features are actually pulling their weight?
#
# **WHY THIS**
#
# - 17 features on ~250 positives is a lot of columns per example. If some
#   are useless — or actively harmful — dropping them is free
#   performance, and a leaner bank is easier to defend.
#
# **HOW IT WORKS**
#
# - Leave-one-out ablation: refit without each feature and record the
#   change in AP. Read the sign carefully — `d_ap < 0` means removing it
#   **hurt**, so it was beneficial; `d_ap > 0` means removing it
#   **helped**, so it was actively harmful. Raw degree is the column to
#   watch: on a graph like this one it is expected to fall on the harmful
#   side.
# - Single-feature ablation has a known blind spot: when two features are
#   near-duplicates, dropping either looks harmless because the twin
#   covers for it, so the table *understates both*. Rather
#   than just note that, we do two things about it:
#   1. the **Spearman correlation matrix** is printed, so a reader can see
#      which pairs are twins (|ρ| ≥ 0.95 — CONVENTION, and the printed
#      pair list makes the cut auditable rather than load-bearing);
#   2. the **category-level** ablation runs beside it, because dropping a
#      whole family removes both twins at once.
#
# **SO WHAT**
#
# - The ablation *ranks* the features and identifies the twins. It adopts
#   nothing — §11.1 is where the ranking is put to a test that can fail.

# %%
t = time.time()
abl_f = ml.ablate_features(ctx, model=ml.BEST_MODEL, feats=ml.FULL_BANK)
abl_c = ml.ablate_categories(ctx, model=ml.BEST_MODEL)
print(f"(computed in {time.time() - t:.0f}s)")
print(f"base AP {abl_f.attrs['base_ap']:.4f}\n")
print(abl_f.round(4).to_string(index=False))
print("\ncategory level:")
print(abl_c.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0),
                         gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
cat_c = {"behavioural": C2, "structural": C1, "forecast": C4}
a = abl_f.sort_values("d_ap")
ax.barh(a["removed"], a["d_ap"],
        color=[cat_c.get(c, MUTED) for c in a["category"]], height=0.65)
ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("Δ AP when this feature is REMOVED  "
              "(left = it helped, right = it hurt)")
ax.set_title("Leave-one-out ablation")
handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in cat_c.values()]
ax.legend(handles, cat_c.keys(), frameon=False, fontsize=8,
          loc="lower right")
despine(ax)

ax = axes[1]
fc = ml.feature_correlation(tab, ml.FULL_BANK)
im = ax.imshow(fc.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(fc))), ax.set_yticks(range(len(fc)))
ax.set_xticklabels(fc.columns, rotation=90, fontsize=7)
ax.set_yticklabels(fc.index, fontsize=7)
ax.grid(False)
ax.set_title("Spearman correlation — where the twins are")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

pairs = (fc.abs().where(~np.eye(len(fc), dtype=bool)).stack().dropna()
         .sort_values(ascending=False))
pairs = pairs[pairs >= 0.95]
seen, twins = set(), []
for (a_, b_), v in pairs.items():
    if a_ in seen or b_ in seen:
        continue
    seen.update({a_, b_})
    twins.append((a_, b_, float(v)))
print(f"near-duplicate pairs (|rho| >= 0.95): "
      f"{twins if twins else 'none'}")

# %% [markdown]
# ### 11.1 Should we actually drop anything? (ablation ranks; only the
# adoption rule decides)
#
# **WHY THIS**
#
# - A negative `d_ap` in the table above is **not** a licence to delete a
#   feature. It is a three-seed mean on a ~50-positive fold — exactly the
#   kind of number the adoption rule exists to discipline.
#
# **HOW IT WORKS**
#
# - Six candidate banks, **all specified before running**:
#   1. drop the worst single feature in the ablation
#   2. drop the three ablation-harmful features
#   3. drop the whole behavioural family
#   4. drop the whole structural family
#   5. the de-duplicated bank — one of each twin pair, and *which* twin
#      goes is decided by the ablation, not by taste
#   6. forecast family only
# - Each is put through the same paired 10-seed test as §7.
# - **The multiple-comparison correction.** Six candidates each tested at
#   95% gives a family-wise error of 1 − 0.95⁶ ≈ **26%**, not 5%. So the
#   rule has a second clause, also pre-stated: the winner must clear zero
#   at Bonferroni confidence 1 − 0.05/6 = 0.9917 (DERIVED from the count
#   of candidates — it is not a number we chose, it is a number the design
#   forced).
# - And a third clause: **one re-test round only**. A ladder climbed until
#   it stops improving is a ladder climbed into noise.
#
# **SO WHAT**
#
# - Some candidates clear zero at 95%. The best of them does **not**
#   survive the Bonferroni re-test. So **no bank change ships** and
#   `FULL_BANK` (17 features) stands.
# - That is the correction doing its job in the direction that costs us
#   something. Had we stopped at 95% we would have shipped a leaner bank
#   on a one-in-four chance of it being noise.
#
# **IF ASKED — "isn't Bonferroni too conservative? You may have thrown
# away a real improvement."**
#
# Possibly — Bonferroni controls the family-wise error rate and pays for
# it in power, and we say so rather than pretending the test is free. Two
# things make it the right trade here: the candidate that lost was a
# *simplification*, so rejecting it leaves the more conservative object in
# place; and the alternative — an uncorrected 26% family-wise error on a
# result we would then write into the docs — is the failure mode this
# whole notebook is built to avoid.

# %%
t = time.time()
hurt = abl_f.set_index("removed")["d_ap"]
drop_twins = [b_ if hurt[b_] >= hurt[a_] else a_ for a_, b_, _ in twins]
lean = [c for c in ml.FULL_BANK if c not in drop_twins]
worst = abl_f.iloc[-1]["removed"]
harmful = abl_f[abl_f["d_ap"] > 0]["removed"].tolist()

CANDS = {
    f"drop {worst} (worst in ablation)":
        [c for c in ml.FULL_BANK if c != worst],
    f"drop the {len(harmful)} ablation-harmful":
        [c for c in ml.FULL_BANK if c not in harmful],
    "drop behavioural family": ml.STRUCTURAL + ml.FORECAST,
    "drop structural family": ml.BEHAVIOURAL + ml.FORECAST,
    f"de-duplicated bank (drop {drop_twins or 'nothing'})": lean,
    "forecast family only": ml.FORECAST,
}
N_CAND = len(CANDS)
BONF = 1 - 0.05 / N_CAND

rows = []
for name, feats in CANDS.items():
    r_ = ml.paired_ap_test(ctx, "logit", "logit", feats_a=ml.FULL_BANK,
                           feats_b=feats)
    r_["candidate"], r_["n_feats"] = name, len(feats)
    rows.append(r_)
cand = pd.DataFrame(rows)[["candidate", "n_feats", "ap_a", "ap_b",
                           "mean_diff", "ci_lo", "ci_hi", "wins", "adopt"]]
print(f"(computed in {time.time() - t:.0f}s)")
print("=== bank candidates vs FULL_BANK, 10 paired seeds, 95% ===")
print(cand.round(4).to_string(index=False))

winners = cand[cand["adopt"]]
print(f"\ncleared zero at 95%: {len(winners)} of {N_CAND}")
print(f"family-wise error if we stopped here: "
      f"{1 - 0.95 ** N_CAND:.0%}  ->  re-test at Bonferroni "
      f"confidence {BONF:.4f}")

bank_change = None
if len(winners):
    w = winners.sort_values("mean_diff", ascending=False).iloc[0]
    conf_r = ml.paired_ap_test(ctx, "logit", "logit",
                               feats_a=ml.FULL_BANK,
                               feats_b=CANDS[w["candidate"]], conf=BONF)
    print(f"\nBonferroni re-test of {w['candidate']!r}:")
    print({k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in conf_r.items()})
    if conf_r["adopt"]:
        bank_change = w["candidate"]
        print("-> ADOPTED, and the ONE permitted re-test round follows.")
    else:
        print("-> REJECTED once the correction is applied. No bank change "
              "ships; the shipped bank stays FULL_BANK (17 features).")
else:
    print("\nno candidate cleared zero even uncorrected - FULL_BANK stands.")

SHIP_FEATS = CANDS[bank_change] if bank_change else ml.FULL_BANK
print(f"\nSHIPPED BANK: {len(SHIP_FEATS)} features")

# %%
fig, ax = plt.subplots(figsize=(8.6, 3.6))
cp = cand.iloc[::-1].reset_index(drop=True)
for i, r_ in cp.iterrows():
    c = C2 if r_["adopt"] else C3
    ax.plot([r_["ci_lo"], r_["ci_hi"]], [i, i], color=c, lw=2.4,
            solid_capstyle="round")
    ax.plot(r_["mean_diff"], i, "o", color=c, ms=7, zorder=4)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(range(len(cp)))
ax.set_yticklabels([f"{c_[:38]} ({n})" for c_, n in
                    zip(cp["candidate"], cp["n_feats"])], fontsize=8)
ax.set_xlabel("paired AP difference vs FULL_BANK, 95% CI (10 seeds)")
ax.set_title("Feature-bank candidates — green cleared zero at 95%, and "
             "then had to survive Bonferroni")
despine(ax)
plt.show()

# %% [markdown]
# ## 12 · If we vandalise the graph, does the graph model even notice?
#
# **WHY THIS**
#
# - §7 rejected every graph model, and §3.6 offered an explanation. This
#   is the section that *tests* the explanation instead of asserting it:
#   if the graph is genuinely misleading, then damaging it in specific
#   ways should have specific, predictable effects.
# - A prediction that could fail is what separates a diagnosis from a
#   story.
#
# **HOW IT WORKS**
#
# - Deliberately corrupt the graph at increasing rates and watch a
#   structure-*using* model. This runs on `sage_lite`
#   (`BEST_GRAPH_MODEL`), **not** on the shipped `logit`: corrupting the
#   graph under a model that never reads it would measure nothing at all.
# - Three corruption modes, each isolating a different thing:
#
# | Mode | What it does | What it tests |
# |---|---|---|
# | `random` | rewire one endpoint of a fraction of edges to a random node | pure structural noise. A model that truly uses the graph **must** degrade |
# | `dice` | Disconnect Internally, Connect Externally — select **same-label** edges and rewire them *across* the label boundary | whether same-label neighbourhoods are helping or hurting. On a heterophilous graph this can *help*, and that is the outcome §3.6 predicts |
# | `swap` | degree-preserving double-edge swap: (a→b),(c→d) become (a→d),(c→b) | isolates *who is connected to whom* from *being busy*. Every node keeps its exact degree, so any change is attributable to wiring alone |
#
# - Accuracy is plotted alongside AP as a **cautionary panel**, not as a
#   result: it barely moves under any corruption, because predicting
#   "ordinary" for everyone is already ~95% accurate on a 20:1 problem.
#   **AP is the panel to read**; the accuracy panel is kept to show exactly
#   how misleading accuracy is on data shaped like this.
#
# **SO WHAT** — in the cell below the chart, because the DICE result is
# the whole point of the section.

# %%
t = time.time()
pert = ml.perturbation_curve(tab, edges, model=ml.BEST_GRAPH_MODEL)
print(f"(computed in {time.time() - t:.0f}s)\n")
print(pert.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))
styles = {"random": (C1, "o-"), "dice": (C3, "s-"), "swap": (C4, "^-")}
for metric, sd_col, ax, name in (("ap", "ap_std", axes[0], "test AP"),
                                 ("accuracy", "accuracy_std", axes[1],
                                  "test accuracy")):
    for mode, (c, fmt) in styles.items():
        s = pert[pert["mode"] == mode]
        ax.errorbar(s["rate"], s[metric], yerr=s[sd_col], fmt=fmt, color=c,
                    lw=1.5, ms=4, capsize=2, label=mode)
    ax.set_xlabel("fraction of edges perturbed")
    ax.set_ylabel(name)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)
axes[0].set_title(f"{ml.BEST_GRAPH_MODEL} AP under corruption — DICE "
                  "IMPROVES it")
axes[1].set_title("Accuracy is nearly flat — which is why the accuracy "
                  "panel is uninformative here")
plt.tight_layout()
plt.show()

d0 = pert[(pert["mode"] == "dice")].set_index("rate")["ap"]
r0 = pert[(pert["mode"] == "random")].set_index("rate")["ap"]
print(f"DICE: AP {d0.iloc[0]:.4f} at 0% -> {d0.iloc[-1]:.4f} at "
      f"{d0.index[-1]:.0%}  ({d0.iloc[-1] / d0.iloc[0] - 1:+.0%})")
print(f"random: AP {r0.iloc[0]:.4f} -> {r0.iloc[-1]:.4f}  "
      f"({r0.iloc[-1] / r0.iloc[0] - 1:+.0%})")

# %% [markdown]
# ### SO WHAT — the DICE result is the diagnosis, stated arithmetically
#
# * **DICE roughly doubles AP.** Rewiring same-label edges *across* the
#   label boundary is vandalism on a homophilous graph, and it **improves**
#   the model substantially here.
# * **Random rewiring degrades it only mildly** (≈ −5%), and
#   **degree-preserving swap is essentially flat.**
# * Read together, the three say: the model gets almost nothing from *who
#   is connected to whom* (swap flat), a little from *how busy a node is*
#   (random degrades mildly), and is being actively **misled** by
#   same-label adjacency (DICE helps a lot).
# * **What changes:** nothing new is adopted — but §7's rejection stops
#   being an empirical shrug. Read next to §3.6's homophily numbers, the
#   DICE result is not a curiosity, it is the mechanism: it names *why*
#   the graph fails rather than only recording *that* it failed.
#
# **IF ASKED — "if breaking the graph helps, why not ship the broken
# graph?"**
#
# Because DICE is not a transformation you can apply at prediction time:
# it needs the labels to decide which edges are same-label, and the labels
# are what we are trying to predict. It is a diagnostic that proves the
# adjacency is misleading, not a feature engineering step. The shippable
# response to "the adjacency is misleading" is the one §7 already took —
# do not use the adjacency.

# %% [markdown]
# ## 13 · Was the class-weighted loss the right call?
#
# **WHY THIS**
#
# - Class-weighted loss is the standard default on an imbalanced problem,
#   and §6 adopted it without argument. An unargued default is still an
#   assumption, and on a 20:1 problem this one is load-bearing.
#
# **HOW IT WORKS**
#
# - `sage_unweighted` is `sage_lite` with the weighting removed and
#   **nothing else changed**, run through the same paired 10-seed test as
#   every other decision. That turns an assumed choice into a measured
#   one.
#
# **SO WHAT**
#
# - `adopt = False`: the weighting does not earn its place on this data at
#   the 95% bar. So it is **defensible convention, not a measured gain**,
#   and the notebook labels it that way rather than quietly presenting it
#   as validated.
# - **What changes:** nothing ships differently — the weighting stays,
#   because dropping a convention on a null result is as unjustified as
#   keeping it on one. What changes is the *claim* we are entitled to make
#   about it.

# %%
t = time.time()
cw = ml.paired_ap_test(ctx, "sage_unweighted", "sage_lite",
                       feats_a=ml.FULL_BANK, feats_b=ml.FULL_BANK)
print(f"(computed in {time.time() - t:.0f}s)")
print({k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in cw.items()})
print("\nreading: 'candidate' is the class-WEIGHTED version. adopt=True "
      "means the weighting earns its place on this data; adopt=False "
      "means it is defensible convention, not a measured gain.")

# %% [markdown]
# ## 14 · Is the result distinguishable from luck?
#
# **WHY THIS**
#
# - An AP of ~0.10 against a ~0.047 floor is a 2× lift, which sounds like
#   something. On ~250 positives it could also be what a signal-free
#   dataset produces on a good day. Nothing else in the notebook settles
#   that.
# - Everything the notebook claims rests on this being non-zero. It is the
#   load-bearing test.
#
# **HOW IT WORKS**
#
# - Shuffle the labels among labelled nodes — destroying every
#   feature–label and graph–label relationship while keeping the class
#   balance and the graph **exactly** as they are — refit, record the AP.
#   Repeat 200 times to build the null distribution of "AP achievable on
#   this data with no signal at all".
# - Shuffling the labels rather than the features is what makes the null
#   honest: every other property of the data (imbalance, degree
#   distribution, feature correlations) is preserved, so the only thing
#   removed is the relationship we are testing.
# - The p-value is the share of permutations matching or beating the real
#   AP. With 200 draws the smallest reportable p is 1/201 ≈ 0.005
#   (DERIVED — it is the resolution of the test, not a choice), and that
#   **floor is reported rather than rounded to zero**.
#
# **SO WHAT**
#
# - p = 0.005, i.e. no permutation out of 200 beat the real AP. The real
#   AP also sits clear of the null's 95th percentile.
# - **The signal is small but it is not luck.** That is the sentence the
#   whole notebook is entitled to, and it is deliberately not a stronger
#   one — a p at the resolution floor means "at the limit of what 200
#   permutations can show", not "one in a thousand".

# %%
t = time.time()
perm = ml.permutation_test(ctx, model=ml.BEST_MODEL, feats=SHIP_FEATS,
                           n_perm=200)
null = perm.pop("null")
print(f"(computed in {time.time() - t:.0f}s)")
print({k: (round(v, 4) if isinstance(v, float) else v)
       for k, v in perm.items()})

fig, ax = plt.subplots(figsize=(7.4, 3.3))
ax.hist(null, bins=32, color=GRID, edgecolor="white")
ax.axvline(perm["null_mean"], color=MUTED, lw=1.2, ls=":",
           label=f"null mean {perm['null_mean']:.3f}")
ax.axvline(perm["null_p95"], color=C4, lw=1.2, ls="--",
           label=f"null 95th pct {perm['null_p95']:.3f}")
ax.axvline(perm["ap"], color=C2, lw=2,
           label=f"real AP {perm['ap']:.3f}  (p = {perm['p_value']:.3f}, "
                 f"floor {perm['p_floor']:.3f})")
ax.set_xlabel("test AP under shuffled labels")
ax.set_ylabel("permutations")
ax.set_title("The signal is small but it is not luck")
ax.legend(frameon=False, fontsize=8)
despine(ax)
plt.show()

# %% [markdown]
# ## 15 · Does it work on authors it has never seen?
#
# **WHY THIS**
#
# - The transductive single-snapshot setup used everywhere above is the
#   main limitation of the whole exercise: every labelled node was present
#   when the model was fitted, so nothing so far shows the model working
#   on a *new* user.
# - That is the **only question a desk actually asks** — *"a new name is
#   loud this week; is it worth listening to?"* Every number before this
#   section answers a different, easier question.
# - This is the section that decides whether the influence work is a
#   signal or an exhibit.
#
# **HOW IT WORKS**
#
# - Split by **tenure** instead of at random: order labelled authors by
#   the date of their first call, train on the established voices, grade
#   on the authors who arrived later.
# - This is strictly harder than a random split, and not only because of
#   novelty — the late cohort also has shorter records, so their labels
#   are noisier. Both effects push the same way, and neither is a reason
#   to soften the reading.
#
# **SO WHAT**
#
# - **The shipped model fails outright.** `logit` scores AP 0.046 against
#   a floor of 0.048 — *below* the floor, with 0 true positives out of 49.
# - The best of any model is `mlp` at 0.073. That is 1.5× the floor and
#   looks like something until you read the `random` row on the same
#   split: a literally-random scorer got 0.050. One split, 49 positives,
#   no paired CI — this is not a result, it is a number inside the noise,
#   and the notebook declines to dress it up as a partial success.
# - **Nothing here generalises to authors it has never seen.** Stated as
#   loudly as any positive result in this notebook, because it is the one
#   that constrains what we are allowed to build.
# - **What changes, and it is the biggest change in the notebook:** the
#   dashboard's Influence tab surfaces the *measured* leaderboard (what
#   authors have actually got right) and **never** a model prediction; and
#   no author-quality feature is permitted into the euphoria signal.

# %%
t = time.time()
coh = ml.evaluate_cohort(ctx, calls, feats=SHIP_FEATS)
print(f"(computed in {time.time() - t:.0f}s)\n")
print(coh.round(4).to_string(index=False))

rand_ap = res.set_index("model")["ap"]
fig, ax = plt.subplots(figsize=(8, 3.4))
mods = [m for m in coh["model"] if m in rand_ap.index]
x = np.arange(len(mods))
ax.bar(x - 0.19, rand_ap.reindex(mods), width=0.36, color=C1,
       label="random split (§6)")
ax.bar(x + 0.19, coh.set_index("model")["ap"].reindex(mods), width=0.36,
       color=C3, label="tenure split (unseen authors)")
ax.axhline(float(coh["ap_random"].iloc[0]), color=INK, lw=1, ls="--")
ax.text(len(mods) - 0.6, float(coh["ap_random"].iloc[0]) * 1.05,
        "random floor", fontsize=8, color=INK, ha="right")
ax.set_xticks(x), ax.set_xticklabels(mods, rotation=20, ha="right")
ax.set_ylabel("test AP")
ax.set_title("Unseen authors — the shipped model drops to the random "
             "floor\n(the transductive limitation, measured rather than "
             "noted)")
ax.legend(frameon=False, fontsize=8)
despine(ax)
plt.show()

best_coh = coh.iloc[0]
_shp = coh.set_index("model").loc[ml.BEST_MODEL]
_rnd = coh.set_index("model").loc["random"]
print(f"best on the tenure split: {best_coh['model']} at AP "
      f"{best_coh['ap']:.4f} vs floor {best_coh['ap_random']:.4f} "
      f"(lift {best_coh['ap_lift']:+.4f})")
print(f"the SHIPPED model {ml.BEST_MODEL!r}: AP {_shp['ap']:.4f} "
      f"(lift {_shp['ap_lift']:+.4f}) - BELOW the floor")
print(f"reality check - the literally-random scorer on this same split "
      f"got AP {_rnd['ap']:.4f} on {int(best_coh['test_pos'])} positives, "
      f"so read the best row as noise, not as partial success")

# %% [markdown]
# ## 16 · What ships, what does not, and what that means for the desk
#
# ### Ships
#
# * **The measured leaderboard.** The dashboard's influence panel ranks
#   authors by the `composite` score — their *realised* record, shrunk
#   toward the population mean so a 2-for-2 newcomer cannot outrank a
#   40-for-60 veteran. That is a measurement, not a prediction, and §15 is
#   the reason it is the surface the desk sees.
# * **What they are saying.** A trailing-window digest of the panel's
#   calls (ticker, direction, conviction), and an influence map drawn from
#   the k-core backbone. Information only.
# * **The model, as a labelled research exhibit.** `FULL_BANK` (17
#   admissible features) + `logit`, headline AP ≈ 0.10 against a 0.054
#   floor, permutation p = 0.005.
#
# ### Does not ship
#
# * **Any graph architecture.** §7: no rung clears zero.
# * **`mean_conf` / `stance_sd`.** §5: circular. Priced at +0.098 AP and
#   left out anyway.
# * **Any feature-bank change.** §11.1: the one candidate that cleared 95%
#   did not survive Bonferroni.
# * **Any influence input to the euphoria signal.** This is an explicit
#   separation, not an oversight. The euphoria detector's inputs are
#   frozen and validated in notebooks 06–07; adding an author-quality
#   feature to it would invalidate that validation, and §15 says the
#   feature would not generalise anyway.
#
# ### Limitations, stated plainly
#
# 1. **Transductive.** The random-split numbers assume the author was in
#    the graph at fit time. §15 shows what happens when they are not.
# 2. **The label is a proxy.** "Influential" here means "was right, with
#    conviction, on moves that mattered" — not reach or persuasion. An
#    author who moves the crowd and is wrong is a negative by
#    construction.
# 3. **One venue, one language, one snapshot.** No cross-platform
#    validation.
# 4. **Sampled network statistics.** Path length and betweenness are
#    400-pivot estimates; they carry sampling error the table does not
#    show.
# 5. **The forward record is short.** Judged calls need forward returns to
#    resolve, so the labelled population lags the store and the earliest
#    cohort is over-represented among positives.
#
# ### For the PM, in one sentence
#
# Rank the loud accounts by what they have actually got right, read what
# they are saying now, and do not let a model that cannot beat a
# coin-flip on new names into the euphoria signal.

# %%
verdict = {
    "notebook": "05_influence_users_model",
    "date": str(pd.Timestamp.today().date()),
    "store": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
              for k, v in census.items()},
    "shipped": {
        "model": ml.BEST_MODEL,
        "graph_model_for_exhibits": ml.BEST_GRAPH_MODEL,
        "feature_bank": SHIP_FEATS,
        "n_features": len(SHIP_FEATS),
        "label_regime": ml.HEADLINE_REGIME,
        "bank_change_adopted": bank_change,
    },
    "headline": {
        "ap": float(res.loc[res["model"] == ml.BEST_MODEL, "ap"].iloc[0]),
        "ap_std": float(res.loc[res["model"] == ml.BEST_MODEL,
                                "ap_std"].iloc[0]),
        "ap_random": floor,
        "auroc": auroc,
        "pooled_precision": tp / max(tp + fp, 1),
    },
    "circularity_audit": {k: v for k, v in circ.items()},
    "adoption_ladder": ladder.to_dict("records"),
    "bank_candidates": cand.to_dict("records"),
    "bonferroni": {"n_candidates": N_CAND, "confidence": BONF,
                   "adopted": bank_change},
    "significance": perm,
    "class_weighting_check": cw,
    "network_stats": side["ours"].to_dict(),
    "homophily": {k: round(float(v), 4) for k, v in h.items()},
    "perturbation": pert.to_dict("records"),
    "cohort_generalisation": coh.to_dict("records"),
    "label_regimes": regimes.to_dict("records"),
    "label_ingredients": ing.to_dict("records"),
    "not_shipped": ["any graph architecture (no ladder rung clears zero)",
                    "mean_conf / stance_sd (circular; priced at "
                    f"{circ['mean_diff']:+.3f} AP)",
                    "any feature-bank change (Bonferroni)",
                    "any influence input to the euphoria signal"],
    "limitations": ["transductive: random-split numbers assume the author "
                    "was present at fit time",
                    "label is a correctness proxy, not reach or persuasion",
                    "single venue / language / snapshot",
                    "path length and betweenness are 400-pivot estimates",
                    "forward record is short; earliest cohort "
                    "over-represented among positives"],
    "maturity_bar": {"min_positives": ml.MIN_POSITIVES,
                     "met": bool(int(tab["y"].sum()) >=
                                 ml.MIN_POSITIVES)},
}
with open(RESEARCH_DIR / "nb05_influence.json", "w") as f:
    json.dump(verdict, f, indent=1, default=str)
print("saved nb05_influence.json")
print(f"total runtime {time.time() - T0:.0f}s")

# %% [markdown]
# ## References
#
# * Blondel, V.D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. (2008).
#   "Fast unfolding of communities in large networks." *J. Stat. Mech.* —
#   the Louvain method implemented in `influence_graph`.
# * Brandes, U. (2001). "A faster algorithm for betweenness centrality."
#   *J. Math. Sociol.* 25(2) — the sampled-pivot betweenness.
# * Fruchterman, T.M.J. & Reingold, E.M. (1991). "Graph drawing by
#   force-directed placement." *Software: Practice and Experience* 21(11).
# * Watts, D.J. & Strogatz, D.H. (1998). "Collective dynamics of
#   'small-world' networks." *Nature* 393 — the σ statistic in §3.1.
# * Hamilton, W., Ying, Z. & Leskovec, J. (2017). "Inductive
#   Representation Learning on Large Graphs." *NeurIPS* — GraphSAGE.
# * Kipf, T.N. & Welling, M. (2017). "Semi-Supervised Classification with
#   Graph Convolutional Networks." *ICLR*; Wu, F. et al. (2019).
#   "Simplifying Graph Convolutional Networks." *ICML* — the linear (SGC)
#   form used by `gcn_lite`.
# * Zhu, J. et al. (2020). "Beyond Homophily in Graph Neural Networks."
#   *NeurIPS* — H₂GCN and the heterophily framing of §3.6.
# * Abu-El-Haija, S. et al. (2019). "MixHop." *ICML* — hop-power
#   concatenation.
# * Zhu, X. & Ghahramani, Z. (2002). *Learning from Labeled and Unlabeled
#   Data with Label Propagation.* CMU tech report — `labelprop`.
# * Zügner, D. & Günnemann, S. (2019). "Adversarial Attacks on Graph
#   Neural Networks via Meta Learning." *ICLR*; Waniek, M. et al. (2018),
#   *Nature Human Behaviour* 2 — DICE-style perturbation.
# * Maslov, S. & Sneppen, K. (2002). "Specificity and stability in
#   topology of protein networks." *Science* 296 — degree-preserving
#   double-edge swap.
# * Davis, J. & Goadrich, M. (2006). "The Relationship Between
#   Precision-Recall and ROC Curves." *ICML*; Saito, T. & Rehmsmeier, M.
#   (2015). *PLOS ONE* 10(3) — why AP and not AUROC under imbalance.
# * Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of
#   Hypotheses.* Springer — §14.
# * Dunn, O.J. (1961). "Multiple Comparisons Among Means." *JASA* 56 —
#   the Bonferroni correction in §11.1.
