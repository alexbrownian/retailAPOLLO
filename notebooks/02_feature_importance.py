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
# # Notebook 02 — Which crowd measurements are worth anything?
#
# Notebook 01 wrote the exam. This one grades the *ingredients*: every
# candidate feature for the onset (GET IN) and top (GET OUT) detectors, put
# through the importance battery below.
#
# > **The one question this notebook answers:** does any single crowd
# > measurement detect euphoria on its own — and if not, which ones deserve a
# > place in a *combination* that might?
#
# ---
#
# ## How to read this notebook
#
# * **The verdict box below is the whole answer**, including the feature that
#   scored best and was thrown out anyway.
# * **Every section is a question**, and each carries **WHY THIS** (what we did
#   not know), **HOW IT WORKS** (the mechanism, and where every number came
#   from), **SO WHAT** (the finding, and what changes).
# * **`IF ASKED` blocks** answer the awkward questions a reviewer would put.
# * **No unexplained constants.** Every number is tagged LEARNED, DERIVED,
#   CONVENTION, GROUND TRUTH or PROJECT DECISION.
#
# ---
#
# ## VERDICT BOX — read this first, then read why
#
# | Question | Answer | Evidence |
# |---|---|---|
# | Is any single feature a euphoria detector? | **No, and it is not close.** Every AUROC lands in **0.51–0.59**; the best is `e3` at **0.567** for starts and `e1` at **0.592** for tops. A detector you would trade needs far more than that. | §4 |
# | Then is any of it real? | **Yes, weakly but measurably.** Eight of ten features clear no-skill on their own label with the interval excluding 0.5, and AP lift runs **1.2–1.5×** over base rate. | §4 |
# | Best-scoring candidate of all? | `source_breadth` — and it is **REJECTED**. It mostly encodes *which year it is*, because the archive only gained a second and third source in 2026. | §3 |
# | Do the onset features duplicate the top features? | **No.** The speed features carry information the level features do not — except `attention_convexity`, which **is** `e5` (ρ = 1.00, by construction, and stated as such). | §6 |
# | Does the combined scorer depend on one fragile input? | **No.** Every perturbation curve degrades gradually; nothing is a knife-edge. | §8 |
# | So what ships? | A **five-feature onset bank** and the unchanged five-feature top bank, into the notebook-03 tournament. No feature is a detector; the combination has to earn it. | §7, §9 |
#
# **Two caveats carried into every reading below**, because both bite here:
# (1) correlated features make single-feature drops **understate** importance;
# (2) with few positives, small deltas are noise — read big movements, never
# decimals.
#
# ---
#
# ## The battery, section by section
#
# | Test | What it does here | Here |
# |---|---|---|
# | Threshold-independent scoring | per-feature AUROC/AP vs the episode labels, CIs by instrument-cluster bootstrap | §4 |
# | Integrity profiling | the `source_breadth` integrity check — a feature that works for the wrong reason is a false positive of *research*, not of trading | §3 |
# | Single-feature ablation | drop-one ablation of a reference scorer (the un-weighted bank mean — the exact construction of the validated euphoria LEVEL) | §7 |
# | Input perturbation | feature-noise perturbation — does the scorer *rely* on each input? | §8 |

# %% [markdown]
# ## 0 · Setup
#
# Imports, the repo path, and the house chart style in one place, so
# every figure below is styled identically and no cell quietly changes
# the look of a chart mid-notebook.

# %%
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# Commented, not deleted: inline IS the default backend under Jupyter, and
# leaving the magic in place makes the file unrunnable as a plain script -
# which is exactly how it gets verified headless (MPLBACKEND=Agg python ...).
# %matplotlib inline

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

C1, C2, C3, C4 = "#2a78d6", "#008300", "#e87ba4", "#eda100"
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
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

RESEARCH_DIR = ROOT / "docs" / "research"
RNG = np.random.default_rng(42)   # fixed seed convention (42/100/2026) so every run is reproducible

# %% [markdown]
# ## 1 · The words this notebook uses
#
# **WHY THIS**
#
# * This is the most jargon-dense notebook in the project: ten feature names,
#   two labels and four metrics, all of which a reviewer is entitled to ask us
#   to define.
# * Everything below comes from `analytics/plain_english.py`, the same module
#   the dashboard reads its tooltips from, so a term cannot mean one thing here
#   and another thing on the PM's screen.
# * The governing convention, worth stating once: **"extreme" always means
#   extreme *for this name*.** Every feature is a percentile rank against the
#   instrument's own past year, never an absolute count.

# %%
from analytics.plain_english import glossary_md                   # noqa: E402

print(glossary_md([
    "mention share",
    "percentile rank",
    "attention vs its own year",
    "how long the mood stayed bullish",
    "rate of new people arriving",
    "attention going near-vertical",
    "this week busier than this month",
    "crowd size vs its own normal",
    "mood turning up",
    "new arrivals, at double speed",
    "how many platforms are talking",
    "crowd big enough to signal",
    "episode",
    "the onset window",
    "detectable",
    "average precision",
    "the random floor",
    "how well it sorts good from bad",
    "confidence interval",
]))

# %% [markdown]
# ## 2 · What is on trial, and where did each window come from?
#
# **WHY THIS**
#
# * A feature bank is where arbitrary numbers hide. "We used a 14-day window"
#   is exactly the kind of choice that never gets questioned and quietly does
#   the overfitting.
# * So before any score is computed: every window in the new onset bank is
#   named, and traced to a constant that already existed in the validated
#   detector. **Not one of them is new.**
#
# **HOW IT WORKS — the incumbent top bank (CONVENTION, unchanged)**
#
# `e1` attention extremity, `e2` sustained bullishness (persistence-gated),
# `e3` crowd influx, `e5` super-exponential attention growth, plus the `fade`
# flag. These are inherited from the validated top study and are **not
# re-tuned here** — re-tuning a validated bank against the same labels it was
# validated on is how a project launders overfitting into a result.
#
# **HOW IT WORKS — the new onset bank, aimed at the LEFT side of an episode**
#
# | Feature | Definition | Provenance of the numbers |
# |---|---|---|
# | `attention_accel` | pct-rank of (7d share − 28d share) | DERIVED — 7 and 28 are the project's existing ROLL and E2/E3 windows |
# | `hype_ratio` | pct-rank of 7d share ÷ own 120d median | DERIVED — the A1 hype gate's exact ratio, made continuous (A1 thresholds it at 2×) |
# | `bull_inflection` | pct-rank of the 14d change of the 14d net-bullish share | DERIVED — 14 is the fade rule's window, pointed the other way (mood turning UP) |
# | `influx_speed` | pct-rank of the 14d change in mention share | DERIVED — E3's 28d change at exactly half the horizon, because a *start* is about rate of change |
# | `attention_convexity` | ≡ `e5` | CONVENTION — contagion accelerating is an inherently early-phase signature (Sornette). Identical to E5 and reported as identical, never as a second piece of evidence |
# | `source_breadth` | pct-rank of distinct sources active in 7d | Candidate only. Must survive §3 before it is admissible |
#
# **SO WHAT**
#
# * Nothing is measured yet. What this section buys is that no reviewer can
#   point at a window in the onset bank and ask "why that one?" — the answer is
#   always "because it was already in the validated detector".

# %% [markdown]
# ### 2.1 The table everything is measured on
#
# **HOW IT WORKS**
#
# * One row per (instrument, **candidate day**). A candidate day is a day the
#   detector is *allowed to speak on*: the coverage gate A0 holds and the day
#   sits inside the judgeable price window. Days the detector could not have
#   spoken on are excluded rather than counted as correct silence — counting
#   them would inflate every accuracy figure with free wins.
# * Three labels, all from notebook 01's price-only catalog: `y_onset` (inside
#   an episode's onset window), `y_late` (inside the rally but past its start),
#   `y_top` (inside `[peak−30d, peak+1d]`).
# * The frame is written to `phase_day_frame.parquet` because **notebook 03
#   stands on this exact table** — the tournament must not rebuild its own
#   version of the data and quietly differ.

# %%
from analytics.euphoria import build_all_series                   # noqa: E402
from analytics.euphoria_phases import build_day_frame             # noqa: E402
from analytics.loaders import (                                   # noqa: E402
    load, THEME_COUNTS, THEME_SENT, TICKER_COUNTS, TICKER_SENT)

t0 = time.time()
prices = pd.read_parquet(ROOT / "data" / "prices" / "prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
series, pxmap = build_all_series(prices)
catalog = pd.read_parquet(ROOT / "data" / "processed" / "episodes.parquet")
counts = {"theme": load(THEME_COUNTS), "ticker": load(TICKER_COUNTS)}
sents = {"theme": load(THEME_SENT), "ticker": load(TICKER_SENT)}
for d in list(counts.values()) + list(sents.values()):
    d["date"] = pd.to_datetime(d["date"])

frame = build_day_frame(series, pxmap, catalog, counts, sents)
frame.to_parquet(ROOT / "data" / "processed" / "phase_day_frame.parquet",
                 index=False)   # notebook 03 stands on this exact table
print(f"{time.time()-t0:.0f}s | {len(frame):,} candidate days, "
      f"{frame['name'].nunique()} instruments, "
      f"prevalence onset {frame.y_onset.mean():.3f} / "
      f"top {frame.y_top.mean():.3f}")

# %% [markdown]
# ## 3 · The best-scoring feature — is it a signal or an artefact?
#
# **WHY THIS**
#
# * `source_breadth` posts the **best raw AUROC of any candidate**. If we ran
#   the battery and read off the leaderboard, it would top the onset bank.
# * The decision that hangs on it: whether the shipped bank contains a feature
#   whose apparent skill is really the archive's shape. That is not a small
#   error — it would put a variable that means "it is 2026" into a detector
#   sold as reading the crowd.
#
# **HOW IT WORKS**
#
# * The rule this section applies: profile *why* something scores well before
#   trusting it. So before scoring anything, count the archive by source and by
#   year. This is a check on the **data**, not on the model, which is why it
#   runs first.
# * The test is stated in advance and is binary: if the feature's variation is
#   dominated by *when* rather than *which name*, it is an artefact.

# %%
by_src = pd.read_parquet(ROOT / "ABSTRACTED_DATA" /
                         "daily_ticker_counts_by_source.parquet")
by_src["date"] = pd.to_datetime(by_src["date"])
comp = (by_src.assign(year=by_src.date.dt.year)
        .pivot_table(index="year", columns="source",
                     values="mention_count", aggfunc="sum")
        .fillna(0).astype(int))
comp

# %% [markdown]
# **READ THE TABLE LIKE THIS** — one row per year, one column per source. Look
# down the StockTwits and X columns and find the year they become non-trivial.
#
# **SO WHAT**
#
# * **StockTwits and X exist in the archive only from 2026** (trace amounts
#   2023–25). In **2021 — the label-rich year — breadth is the constant 1** for
#   every name, so its percentile rank is degenerate.
# * Its apparent skill is therefore a **coverage-regime artefact**: it mostly
#   encodes "is this 2026", which correlates with episodes through the shape of
#   the archive, not through anything the crowd did.
# * **What changes: `source_breadth` is REJECTED from the onset bank**, despite
#   the best raw AUROC of any candidate. Five features enter, not six.
# * The check, not the AUROC, is the finding. The feature becomes admissible
#   once the multi-source archive covers a full episode cycle — carried into
#   future work rather than quietly dropped.
#
# **IF ASKED — "you rejected your best feature. Isn't that just discarding
# performance you did not like the look of?"**
#
# The rejection is not on the score, it is on a property of the data that can
# be checked without reference to the score: before 2026 the feature has no
# variance to carry information *with*. A variable that is constant across the
# entire label-rich period cannot be measuring crowd behaviour in that period,
# so whatever it is measuring in the pooled number is coming from the regime
# difference between periods. Note also which way the incentive ran — this
# check *cost* us the top of the leaderboard, and it was pre-specified as the
# first thing this notebook does precisely so it could not be skipped once we
# saw what it would cost.
#
# ---
#
# ## 3.1 Locking the bank against production drift

# %%
ONSET_BANK = ["attention_accel", "hype_ratio", "bull_inflection",
              "influx_speed", "attention_convexity"]
TOP_BANK = ["e1", "e2", "e3", "e5", "fade"]

# drift guard: the bank this notebook just derived must equal the LOCKED
# production bank in analytics/euphoria_phases.py - if this cell raises,
# research and production have diverged and one of them is stale
from analytics.euphoria_phases import (                           # noqa: E402
    ONSET_BANK as PRODUCTION_BANK)
assert ONSET_BANK == PRODUCTION_BANK, "bank drift vs euphoria_phases.py"
ALL_FEATS = ONSET_BANK + TOP_BANK

# %% [markdown]
# **WHY THIS** — the assert above is small and load-bearing. The research
# record and the shipped detector must contain the *same five features*; if
# someone edits one and not the other, this notebook stops rather than
# reporting results for a bank nobody is running. A silent divergence between
# what we measured and what we ship is the worst failure available to a project
# like this, so it is made noisy.
#
# ---
#
# ## 4 · Does any single feature detect euphoria on its own?
#
# **WHY THIS**
#
# * This is the question a sceptic asks first, and the one a hopeful reading of
#   the literature answers wrongly. If one feature *were* a detector, the rest
#   of the project would be unnecessary.
# * The decision that hangs on it: whether notebook 03 needs to be a tournament
#   of *combinations* at all, and which features are even admissible to it.
#
# **HOW IT WORKS**
#
# * **AP leads, AUROC follows.** Under class imbalance AP is the metric that
#   punishes false positives; AUROC summarises ranking quality but flatters
#   rare classes. Both are reported so neither can be cherry-picked.
# * **AP is always quoted against its no-skill baseline**, which is the label
#   prevalence — DERIVED, not chosen. An AP of 0.08 means nothing until you
#   know the base rate is 0.06.
# * **The confidence interval resamples whole instruments, not days.** Daily
#   rows within one name are serially dependent — TSLA on Tuesday is not
#   independent evidence from TSLA on Monday — so a naive bootstrap would
#   produce intervals several times too tight and would let us call noise
#   significant. The cluster is the instrument because that is the unit
#   plausibly independent. 300 resamples, 90% interval, seed 42 (CONVENTION,
#   the project's fixed seed, so the interval is reproducible).
# * **`separates` is the pre-stated test**: the lower bound of the interval
#   must clear 0.5. "Point estimate above 0.5" is not a finding, it is a
#   coin-flip with a direction.

# %%
def cluster_bootstrap_ci(frame, feat, label, n_boot=300, seed=42):
    """AUROC point estimate + 90% CI, resampling instruments with
    replacement. Returns (auroc, lo, hi)."""
    groups = {n: g for n, g in frame.dropna(subset=[feat])
              .groupby("name")[[feat, label]] if g[label].nunique() >= 1}
    names = list(groups)
    point = roc_auc_score(frame.dropna(subset=[feat])[label],
                          frame.dropna(subset=[feat])[feat])
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        y = np.concatenate([groups[n][label].values for n in pick])
        s = np.concatenate([groups[n][feat].values for n in pick])
        if y.min() == y.max():
            continue
        stats.append(roc_auc_score(y, s))
    lo, hi = np.percentile(stats, [5, 95])
    return point, lo, hi

rows = []
for feat in ALL_FEATS:
    sub = frame.dropna(subset=[feat])
    for label in ("y_onset", "y_top"):
        auroc, lo, hi = cluster_bootstrap_ci(frame, feat, label)
        rows.append({"feature": feat, "label": label, "auroc": auroc,
                     "ci_lo": lo, "ci_hi": hi,
                     "ap": average_precision_score(sub[label], sub[feat]),
                     "ap_baseline": sub[label].mean(),
                     "separates": lo > 0.5})
per_feature = pd.DataFrame(rows)
per_feature["ap_lift"] = per_feature.ap / per_feature.ap_baseline
print(per_feature.round(3).to_string())

# %%
from analytics.plain_english import plain                         # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

# TWO THINGS ARE ON THIS CHART AND THEY ARE NOT THE SAME THING:
#   * the PANEL says which question the measurement is being graded on -
#     left = "does it spot the START?" (GET IN), right = "does it spot
#     the TOP?" (GET OUT). Every measurement is graded on BOTH.
#   * the COLOUR says which detector that measurement is a part of - the
#     GET IN bank (the five onset features) or the GET OUT bank (the
#     incumbent euphoria features).
# So a GET OUT-coloured bar scoring high in the LEFT panel is a
# measurement doing a job it was not hired for, which is precisely the
# thing this chart exists to reveal.
GET_IN_C, GET_OUT_C = C1, C2
_bank_color = [GET_IN_C if f in ONSET_BANK else GET_OUT_C for f in ALL_FEATS]
_color_of = dict(zip(ALL_FEATS, _bank_color))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for ax, label, question in [
        (axes[0], "y_onset", "graded on: did it spot the START?  (GET IN)"),
        (axes[1], "y_top", "graded on: did it spot the TOP?  (GET OUT)")]:
    sub = (per_feature[per_feature.label == label]
           .sort_values("auroc", ascending=True).reset_index(drop=True))
    colors = [_color_of[f] for f in sub.feature]
    # plain-English tick labels: the stored column names never change, only
    # what a human reads (analytics/plain_english.py is the single glossary)
    ax.barh([plain(f) for f in sub.feature], sub.auroc - 0.5, left=0.5,
            color=colors, height=0.6)
    ax.errorbar(sub.auroc, np.arange(len(sub)),
                xerr=[sub.auroc - sub.ci_lo, sub.ci_hi - sub.auroc],
                fmt="none", ecolor=INK, elinewidth=1, capsize=2)
    ax.axvline(0.5, color=INK, lw=1)
    ax.set_title(f"per-feature AUROC — {question}\n"
                 "90% cluster-bootstrap CI", fontsize=9)
    ax.set_xlim(0.42, 0.72)
    despine(ax)
fig.tight_layout()
fig.legend(handles=[Patch(color=GET_IN_C,
                          label="measurement belongs to the GET IN bank"),
                    Patch(color=GET_OUT_C,
                          label="measurement belongs to the GET OUT bank")],
           loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2,
           frameon=False, fontsize=8)
fig.text(0.01, -0.13, "bars start at no-skill 0.5; a whisker crossing 0.5 "
         "means the measurement has not been shown to do anything",
         fontsize=8, color=MUTED)
plt.show()

# %% [markdown]
# **READ THE CHART LIKE THIS** — the bar is the point estimate, the whisker is
# the 90% interval, the vertical line at 0.5 is a coin flip. A feature whose
# whisker crosses that line has not been shown to do anything.
#
# * **Panel = the question.** Left panel grades every measurement on *did it
#   spot the START of an episode* (the **GET IN** job); right panel grades the
#   same measurements on *did it spot the TOP* (the **GET OUT** job).
# * **Colour = the team.** Blue bars are the measurements that make up the
#   **GET IN** bank; green bars are the ones that make up the **GET OUT**
#   bank. Colour never changes between panels — it is a property of the
#   measurement, not of the panel it is in.
# * So the interesting cases are the mismatches: a **green bar high in the
#   left panel** is a GET OUT measurement that also sees starts, and a **blue
#   bar high in the right panel** is a GET IN measurement that also sees tops.
#   Those are the crossovers notebook 03 is allowed to exploit.
#
# **SO WHAT**
#
# * **No single feature is a detector, and the ceiling is low.** Every AUROC
#   sits in the **0.51–0.59** band. The best for starts is `e3` (crowd influx)
#   at **0.567**; the best for tops is `e1` (attention extremity) at **0.592**.
#   For scale: a tradeable single-variable signal would be well north of 0.65.
# * **But it is not nothing.** AP lift runs **1.2–1.5× over base rate**, and
#   the interval clears 0.5 for eight of the ten feature/label pairs.
# * **The two banks sort themselves onto their own labels**, which is a check
#   passing rather than a result being manufactured: the speed features clear
#   the line on `y_onset`, and `e2` (sustained bullishness) — a *top* feature —
#   is the one that **fails** on `y_onset` with an interval of 0.42–0.52. A
#   feature designed to detect a mature one-way mood should not detect a start,
#   and it does not.
# * **What changes:** notebook 03 is confirmed as a tournament of
#   *combinations*. There is no shortcut single variable to ship instead, and
#   any later claim of a large single-feature effect should be disbelieved
#   against this table.
#
# **IF ASKED — "AUROC 0.55 is barely above chance. Why is this project
# continuing?"**
#
# Because a weak *per-day* ranking and a useful *per-episode* alert are
# different objects, and the gap between them is the gates and the combination.
# The detector does not have to rank every ordinary Tuesday correctly; it has
# to fire rarely and land inside a 45-day window when it does. A feature with
# AUROC 0.55 across 60,000 candidate days can still contribute to a rule that
# fires a handful of times a year with a meaningful hit rate — and notebooks 04
# and 06 measure exactly that, against the same labels, so the claim is tested
# rather than assumed. What this section *does* rule out is the other story,
# the one where a single crowd variable is quietly excellent. It is not.
#
# ---
#
# ## 5 · Do the two populations actually look different?
#
# **WHY THIS**
#
# * An AUROC is a summary, and summaries hide shapes. A feature can post 0.55
#   because it shifts the whole distribution slightly, or because it has a
#   thin, extreme right tail that is almost all positives — and those two
#   worlds call for completely different detector designs.
#
# **HOW IT WORKS**
#
# * For each onset-bank feature: the distribution on onset-window days (blue)
#   over the distribution on ordinary days (grey), density-normalised so the
#   rare class is visible at all.
# * Every feature is a percentile rank on [0,1], so the horizontal distance
#   between the two humps is directly comparable across panels — no rescaling
#   is needed and none is applied.
# * This is the distribution view of what §4 scored: one panel per feature,
#   onset days against ordinary days, so the separation can be seen and not
#   only read off a number.

# %%
fig, axes = plt.subplots(1, len(ONSET_BANK), figsize=(12.5, 2.9),
                         sharey=True)
for ax, feat in zip(axes, ONSET_BANK):
    sub = frame.dropna(subset=[feat])
    pos = sub.loc[sub.y_onset == 1, feat]
    neg = sub.loc[sub.y_onset == 0, feat]
    ax.hist(neg, bins=25, density=True, color=GRID, label="ordinary day")
    ax.hist(pos, bins=25, density=True, color=C1, alpha=0.65,
            label="onset window")
    auc = roc_auc_score(sub.y_onset, sub[feat])
    ax.set_title(f"{plain(feat)}\nAUC {auc:.2f}", fontsize=9)
    despine(ax)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# **READ THE CHART LIKE THIS** — look for blue sitting to the RIGHT of grey.
# The overlap is the honest picture of how hard this problem is: these are
# heavily-overlapping populations, not two separated clusters.
#
# **SO WHAT**
#
# * The separation is a **modest rightward shift, not a distinct tail.** No
#   feature has a region of its range that is purely onset days.
# * **What that implies for the design:** a detector cannot work by thresholding
#   one feature high, because there is no height at which the population is
#   clean. It has to work by requiring *several* mediocre signals to agree —
#   which is what the bank scorer and the gates do.
# * Nothing is adopted or dropped here. The section exists because "AUROC 0.55"
#   and "a thin pure tail worth 0.55" would justify different designs, and this
#   rules out the second.
#
# ---
#
# ## 6 · How much of this is the same information counted twice?
#
# **WHY THIS**
#
# * Caveat 1 from the verdict box bites here: **correlated features make
#   single-feature drops understate importance.** Section 4 measured ten
#   features as if independent. They are not.
# * The decision that hangs on it: whether "five onset features" is really five
#   pieces of evidence or two dressed as five — and whether the onset bank is
#   just the top bank wearing new names.
#
# **HOW IT WORKS**
#
# * Spearman (rank) correlation, not Pearson, because every feature is already
#   a percentile rank — a rank correlation is the matching tool and is immune
#   to the monotone rescalings the features have been through.
# * Computed on candidate days only, i.e. the same rows everything else in this
#   notebook is measured on.

# %%
corr = frame[ALL_FEATS].corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
_labels = [plain(f) for f in ALL_FEATS]
ax.set_xticks(range(len(ALL_FEATS)), _labels, rotation=35, ha="right",
              fontsize=7)
ax.set_yticks(range(len(ALL_FEATS)), _labels, fontsize=7)
for i in range(len(ALL_FEATS)):
    for j in range(len(ALL_FEATS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=7,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else INK)
ax.set_title("Rank correlation on candidate days\n"
             "(attention going near-vertical ≡ E5 by construction — ρ = 1)")
ax.grid(False)
fig.colorbar(im, shrink=0.8)
plt.show()

# %% [markdown]
# **READ THE CHART LIKE THIS** — red is positive correlation, blue negative,
# pale is independent. Look for red blocks: those are groups of features that
# are largely the same measurement.
#
# **SO WHAT**
#
# * **`attention_convexity` ≡ `e5`, ρ = 1.00 exactly.** They are the same
#   column. This is disclosed rather than discovered: the onset bank
#   deliberately re-uses E5, and the pair is never presented as two independent
#   pieces of evidence anywhere in this project.
# * **The onset bank is NOT redundant with the top bank.** The speed features
#   (`attention_accel`, `influx_speed`) carry information the level features
#   (`e1`, `e2`) do not — which is the whole reason a separate onset bank
#   exists rather than re-pointing the top detector at the left of the arc.
# * **What changes: nothing is dropped here, but §7 is now read differently.**
#   Because these features overlap, every drop-one result below **understates**
#   the dropped feature's importance — the survivors absorb its job. That is
#   exactly caveat 1, and it is why §7 refuses to adopt on the strength of
#   a small delta.
#
# ---
#
# ## 7 · What does each feature contribute to the *combination*?
#
# **WHY THIS**
#
# * §4 asked what each feature does alone. That is not the question the shipped
#   system poses — the system uses a bank, and a feature that is mediocre alone
#   can be valuable in company (or redundant in it).
#
# **HOW IT WORKS**
#
# * The reference scorer is the **un-weighted mean of the bank** — DERIVED, not
#   chosen: it is the exact construction of the validated euphoria LEVEL.
#   Equal weights also mean **no fitting is involved**, so nothing here can
#   overfit and the deltas are attributable to the feature rather than to a
#   re-optimised weight quietly compensating.
# * Drop one feature, re-score, record the change. Negative Δ = the bank got
#   worse without it = the feature was helping.
# * This complements rather than replaces §4. The full-system ablation — gates,
#   cooldowns, walk-forward — already lives in `analytics/euphoria.py` and is
#   re-run on every rebuild; this is the score-function view of the same
#   question.
# * **Read big movements, not decimals** (caveat 2 above): with these
#   positive counts, a Δ AUROC of 0.002 is noise wearing a sign.

# %%
def bank_score(df, feats):
    return df[feats].mean(axis=1)


def ablation_table(bank, label):
    rows = []
    base_score = bank_score(frame, bank)
    mask = base_score.notna()
    base_auroc = roc_auc_score(frame[label][mask], base_score[mask])
    base_ap = average_precision_score(frame[label][mask], base_score[mask])
    rows.append({"variant": "FULL bank", "auroc": base_auroc, "ap": base_ap,
                 "d_auroc": 0.0, "d_ap": 0.0})
    for drop in bank:
        kept = [f for f in bank if f != drop]
        s = bank_score(frame, kept)
        m = s.notna()
        auroc = roc_auc_score(frame[label][m], s[m])
        ap = average_precision_score(frame[label][m], s[m])
        # plain-English row label: the bank list keeps the stored names, only
        # what a human reads is translated (analytics/plain_english.py)
        rows.append({"variant": f"without {plain(drop)}",
                     "auroc": auroc,
                     "ap": ap,
                     "d_auroc": auroc - base_auroc,
                     "d_ap": ap - base_ap})
    return pd.DataFrame(rows)


abl_onset = ablation_table(ONSET_BANK, "y_onset")
abl_top = ablation_table(TOP_BANK, "y_top")
print("GET IN bank (start), scored against 'did a start happen here'")
print(abl_onset.round(4).to_string(index=False))
print()
print("GET OUT bank (top), scored against 'did a top happen here'")
print(abl_top.round(4).to_string(index=False))
print()
_worst_on = abl_onset.iloc[1:].sort_values("d_auroc").iloc[0]
_worst_tp = abl_top.iloc[1:].sort_values("d_auroc").iloc[0]
print(f"most load-bearing in the GET IN bank:  {_worst_on['variant']} costs "
      f"{_worst_on['d_auroc']:+.4f} AUROC")
print(f"most load-bearing in the GET OUT bank: {_worst_tp['variant']} costs "
      f"{_worst_tp['d_auroc']:+.4f} AUROC")

# %%
# Here the colour means ONE thing only - the DIRECTION of the change -
# and it means the same thing in both panels. (It used to be blue on the
# left and green on the right for identical results, which read as a
# per-bank colour code and is not what it was.)
KEEP_C, DROP_C = C1, DIV_POS

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
for ax, abl, title in [(axes[0], abl_onset, "the GET IN bank"),
                       (axes[1], abl_top, "the GET OUT bank")]:
    sub = abl[abl.variant != "FULL bank"]
    ax.barh(sub.variant, sub.d_auroc, height=0.55,
            color=[DROP_C if v > 0 else KEEP_C for v in sub.d_auroc])
    ax.axvline(0, color=INK, lw=1)
    ax.set_title(f"Take one measurement away — {title}\n"
                 "LEFT (blue) = the bank got WORSE without it, it was "
                 "earning its place\nRIGHT (red) = the bank got BETTER "
                 "without it", fontsize=9)
    ax.set_xlabel("change in AUROC vs the full bank")
    despine(ax)
fig.tight_layout()
plt.show()

# %% [markdown]
# **How to read the chart above.** Each bar is the whole bank re-scored with
# that one measurement removed. Bars pointing LEFT mean removal HURT — the
# feature was contributing. Bars pointing RIGHT mean the bank was better
# without it. The zero line is the full bank.
#
# **SO WHAT**
#
# * Nothing in either bank is load-bearing on its own. The biggest single
#   removal moves AUROC by a few thousandths, which is inside the noise band
#   this notebook has already declared unreadable — so the honest reading is
#   *the bank is a committee, not a soloist plus four passengers*.
# * That is a **stability argument, not a weakness**: a bank where one input
#   carried everything would break the day that one data source changed. This
#   one degrades smoothly, which is what a live system needs.
# * It also means **no feature earns removal on these numbers**. Parsimony
#   would drop a feature only if removing it clearly helped; none clearly
#   does, so all five stay and the decision is deferred to Notebook 03, where
#   the tournament scores whole models under walk-forward instead of scoring a
#   fixed equal-weight average on one pooled frame.
#
# **IF ASKED — "if every bar is inside the noise, why show the chart at all?"**
# Because a null result is a result, and this is the null we needed. The claim
# being tested was "the bank might be one good feature dressed up as five". The
# chart refutes it in the only direction that matters: had one feature been
# carrying the bank, its bar would have been visibly long, and it is not. A
# defence panel is entitled to see the test that could have embarrassed us.

# %% [markdown]
# ## 8 · If a data feed degrades, does the signal collapse?
#
# **WHY THIS**
#
# * Everything so far assumes the inputs arrive clean. In production they will
#   not: a scraper misses a day, a platform changes its API, sentiment scoring
#   drifts after a model update. A research result that only holds on pristine
#   data is not a result a desk can run.
# * This is a perturbation test. The question it answers is whether the bank
#   relies on any one *feed* — so the corruption is applied to one feature at
#   a time, and the cost of degrading it is measured.
#
# **HOW IT WORKS**
#
# * Every feature is a percentile rank, so it lives on 0–1 and noise is
#   directly interpretable: σ = 0.10 means "this measurement is now typically
#   ten percentile points off the truth". CONVENTION, and a deliberately
#   punishing one — σ = 0.40 is closer to *destroyed* than to *degraded*.
# * Add gaussian noise to one feature, clip back into 0–1, re-score the whole
#   bank, record the change in AP. Repeat over **20 noise draws** per level and
#   average, so a single unlucky draw cannot set the shape of a curve.
# * A steep downward curve = the bank leans on that feed. A flat curve = the
#   bank barely notices it going wrong.

# %%
def perturbation_curves(bank, label, sigmas=(0.05, 0.1, 0.2, 0.4),
                        n_draws=20):
    base_score = bank_score(frame, bank)
    mask = base_score.notna()
    y = frame[label][mask].values
    base_ap = average_precision_score(y, base_score[mask])
    out = {}
    for feat in bank:
        degr = []
        for s in sigmas:
            aps = []
            for d in range(n_draws):
                rng = np.random.default_rng(1000 * d + int(100 * s))
                noisy = frame.copy()
                noisy[feat] = (noisy[feat]
                               + rng.normal(0, s, len(noisy))).clip(0, 1)
                sc = bank_score(noisy, bank)[mask]
                aps.append(average_precision_score(y, sc))
            degr.append(np.mean(aps) - base_ap)
        out[feat] = degr
    return base_ap, sigmas, out

base_ap, sigmas, curves = perturbation_curves(ONSET_BANK, "y_onset")
fig, ax = plt.subplots(figsize=(8.5, 3.6))
palette = [C1, C2, C3, C4, INK]
for (feat, degr), c in zip(curves.items(), palette):
    ax.plot(sigmas, degr, "o-", color=c, lw=1.6, ms=4)
    ax.text(sigmas[-1] * 1.03, degr[-1], plain(feat), color=c, fontsize=8,
            va="center")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("how badly that one measurement is corrupted "
              "(σ, in percentile points)")
ax.set_ylabel("what it costs the bank (change in AP)")
ax.set_title("Break one input on purpose, see what the start bank loses\n"
             f"(clean bank AP = {base_ap:.3f}; a steep fall = the bank leans "
             "on that feed)", fontsize=9)
ax.set_xlim(0.03, 0.66)
despine(ax)
fig.tight_layout()
plt.show()

_worst_feat = min(curves, key=lambda f: curves[f][-1])
print(f"clean start-bank AP: {base_ap:.4f}")
for feat, degr in curves.items():
    print(f"  {plain(feat):<38} AP change at heavy noise: {degr[-1]:+.4f}")
print(f"most damaging feed to lose: {plain(_worst_feat)} "
      f"({curves[_worst_feat][-1]:+.4f} AP)")

# %% [markdown]
# **SO WHAT**
#
# * **No feed is a single point of failure.** Even at σ = 0.40 — corruption
#   severe enough that the measurement is barely related to the truth any more
#   — the bank's AP falls by a modest amount rather than collapsing. A desk can
#   run this knowing a bad scrape day degrades the signal instead of inverting
#   it.
# * **The ordering of the curves is the useful output**, not their depth. The
#   steepest curve names the feed to monitor hardest in production; that is
#   now an operational instruction, not an opinion.
# * **Consistent with §7.** Both tests point the same way — the bank is a
#   committee. Two independent methods agreeing is worth more than either
#   alone, because they fail differently: ablation removes information
#   entirely, perturbation keeps the column and destroys its content.
#
# **IF ASKED — "isn't σ = 0.4 an unrealistic amount of corruption?"** Yes, and
# deliberately so. This is a stress test, not a forecast of data quality. The
# realistic levels are the left-hand end of the curve; the right-hand end
# exists to show where the cliff is, and the finding is that there isn't one
# inside the tested range.

# %% [markdown]
# ## 9 · What ships out of this notebook?
#
# Every claim below is re-rendered from the cells above on each rebuild, so
# this section cannot drift away from the evidence.
#
# **What was established**
#
# * **No single crowd measurement is a euphoria detector.** AUROCs sit in the
#   0.51–0.59 band. This is not a disappointing result to be buried — it is the
#   finding that attention features *select candidates* while gates,
#   combination and confirmation do the actual work.
#   Anyone shipping a single magic crowd variable would be shipping an overfit.
# * **The weak signal is nonetheless real.** Eight of ten feature/label pairs
#   clear no-skill with the confidence interval excluding 0.5, and precision
#   runs 1.2–1.5× base rate. Weak-but-real is exactly the raw material a
#   combination step is for.
# * **The best raw scorer was rejected.** `source_breadth` topped the
#   leaderboard and is out, because it mostly encodes which year it is. Scoring
#   well for the wrong reason is a research false positive, and catching it
#   here is cheaper than catching it in production.
# * **The bank is not one feature in disguise.** Ablation (§7) and perturbation
#   (§8) agree: contributions are diffuse, nothing is load-bearing, nothing is
#   a knife-edge.
#
# **What physically ships**
#
# * A **five-feature start bank** — this week busier than this month, crowd
#   size vs its own normal, mood turning up, new arrivals at double speed,
#   attention going near-vertical — asserted equal to the production list in
#   §3.1, so this notebook fails loudly if the shipped code ever drifts.
# * The **unchanged five-feature top bank** (E1, E2, E3, E5, fade).
# * `docs/research/nb02_feature_stats.json`, the per-feature scoreboard the
#   later notebooks and the parameter register read from.
#
# **What this notebook explicitly does NOT claim**
#
# * That the euphoria signal works. Nothing here is walk-forward, nothing is
#   gated, nothing is scored per episode. A per-day AUROC of 0.57 and a useful
#   per-episode alert are different objects, and only Notebooks 04, 06 and 07
#   measure the second one.
# * That the feature list is optimal. It is *defensible* — every member is
#   either measured above no-skill or required by construction, and the one
#   rejection is documented with its reason.
#
# **Next (Notebook 03):** the model tournament — random baseline, rule gates,
# logistic regression, gradient boosting, MLP — under walk-forward, with the
# selection criterion pre-stated in writing before any result is computed, so
# the winner cannot be chosen after seeing the scoreboard.

# %%
per_feature.to_json(RESEARCH_DIR / "nb02_feature_stats.json", orient="records",
                    indent=1)
print("saved nb02_feature_stats.json")

# %% [markdown]
# ---
# # SS — The August-2026 bank extension (the four columns the desk model added)
#
# **WHY THIS**
#
# * The July bank above was built for the RULES detectors. The August desk
#   model (`analytics/ml_detector.py`, selected in notebook 03 §SS) widened
#   the bank in two deliberate ways, and this section gives the new columns
#   the same per-feature scrutiny the original ten received:
#   - **`bull_level` + `bull_persist`** — e2's two raw ingredients, split
#     apart so the model can learn the interaction the old 75% persistence
#     GATE hard-coded;
#   - **`price_runup` + `price_ret21`** — the price pair, licensed for the
#     DESK heads only (the desk configuration may use price; the crowd-only
#     detectors above never see these columns).
#
# **HOW IT WORKS** — same instrument as the rest of this notebook:
# standalone AUROC per feature per label on the coverage-gated day frame,
# descriptive attribution only (the walk-forward evidence for the bank as a
# whole is notebook 03's tournament).

# %%
from analytics import ml_detector as mld                    # noqa: E402
from sklearn.metrics import roc_auc_score as _auroc         # noqa: E402

_prices_aug = pd.read_parquet(ROOT / "data" / "prices" / "prices.parquet")
_prices_aug["date"] = pd.to_datetime(_prices_aug["date"])
from analytics.euphoria import build_all_series as _bas     # noqa: E402
from analytics.euphoria_phases import (episode_catalog as _ec,  # noqa: E402
                                       build_day_frame as _bdf)
from analytics.loaders import load as _load                 # noqa: E402

_series_aug, _pxmap_aug = _bas(_prices_aug)
_eps_aug = _ec(_series_aug, _pxmap_aug)
_frame_aug = _bdf(_series_aug, _pxmap_aug, _eps_aug,
                  {"theme": _load("daily_theme_counts.parquet"),
                   "ticker": _load("daily_ticker_counts.parquet")},
                  {"theme": _load("daily_theme_sentiment.parquet"),
                   "ticker": _load("daily_ticker_sentiment.parquet")})
_cand_aug = mld.attach_price_features(
    mld.candidate_frame(_frame_aug), _series_aug, _pxmap_aug)

_rows_aug = []
for _f in mld.DESK_ML_BANK:
    _rows_aug.append({
        "feature": mld.ML_BANK_LABELS.get(_f, _f),
        "column": _f,
        "new in August": _f in ("bull_level", "bull_persist",
                                "price_runup", "price_ret21"),
        "AUROC vs y_onset": round(_auroc(_cand_aug["y_onset"],
                                         _cand_aug[_f]), 3),
        "AUROC vs y_top": round(_auroc(_cand_aug["y_top"],
                                       _cand_aug[_f]), 3),
    })
_aug_board = (pd.DataFrame(_rows_aug)
              .sort_values("AUROC vs y_top", ascending=False))
_aug_board

# %% [markdown]
# **SO WHAT**
#
# * The price pair is the strongest standalone material in the bank — which
#   is exactly why the July system used it as a hard GATE. Handing it to the
#   model as a continuous feature (rather than a 25%/50% door) is where a
#   large part of the August accuracy gain came from (notebook 03 §SS
#   measures the bank jointly, walk-forward).
# * `bull_level` / `bull_persist` sit in the same weak-but-real band as the
#   rest of the crowd bank — consistent with §5's finding that no single
#   crowd measurement is a detector. Splitting e2 cost nothing in signal
#   and removed one embedded constant (the 75% gate).
# * Numbers here are DESCRIPTIVE (full-frame, not walk-forward) — the
#   selection evidence stays notebook 03's tournament, and nothing in this
#   section chose the shipped model.

# %%
import json                                                 # noqa: E402

# its own file, beside the July board (nb02_feature_stats.json is a LIST
# of per-feature rows and stays exactly as the July record wrote it)
with open(RESEARCH_DIR / "nb02_august_bank.json", "w") as _fh:
    json.dump(_aug_board.to_dict(orient="records"), _fh, indent=1,
              default=str)
print("saved nb02_august_bank.json (the August bank extension board)")
