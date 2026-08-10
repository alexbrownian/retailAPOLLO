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
# # Notebook 03 — The Model Tournament
#
# **This notebook:** compares candidate detectors for euphoria ONSET and
# euphoria TOP under one walk-forward discipline, with the selection
# criterion written down **in this cell, before any result below it was
# computed.**
#
# > **READ §SS FIRST if you want what SHIPS.** This notebook now holds two
# > tournaments, both preserved because both are evidence:
# > 1. **The July race (the body below)** — the original contest, run for
# >    the RULES-era system under its FA budget and gated candidacies. Its
# >    verdict (rules win by parsimony) was correct *for that frame* and is
# >    kept unchanged as the record of what we tested first.
# > 2. **§SS — the August desk tournament (at the end)** — the desk
# >    re-framed the brief ("no prior thresholds; best accuracy;
# >    explainable; not just rules"), the candidacy collapsed to the
# >    coverage gate, the bank was un-gated and extended, and a NEW
# >    pre-stated criterion selected the **logit+GBM rank ensemble** that
# >    now runs live. The presentation quotes §SS; the July body is the
# >    provenance trail that shows the winner was not the first thing we
# >    tried.
#
# ## The pre-stated criterion (nothing here is chosen after seeing results)
#
# **Layer 1 — model selection:** models are ranked by **average precision
# (AP)**, ties broken by AUROC, both computed threshold-independently on the
# stacked walk-forward *test* scores. A model must beat the seed-matched
# **random baseline** on both AP and AUROC to be eligible at all — score
# quality is where chance must be beaten. (A random alerter's raw *capture
# count* is deliberately not the bar: firing blindly every cooldown window
# captures episodes by accident at several times the false-alarm rate —
# precision-free capture is noise, not skill, and the boards below display
# exactly this pattern.) If a learned model's AP does not clear the
# rule-based bank's AP outside its instrument-cluster bootstrap 90% CI, the
# **rules win by parsimony** — the same standard that rejected this project's
# first ML challenger.
#
# **Layer 2 — operating point:** each test year's alert threshold is chosen
# on its train years only: maximise captured episodes subject to the
# false-alarm budget, tie → the more conservative threshold. The budget is
# **derived, not invented**: 0.23 FAs/instrument-year — the noise level the
# desk already accepted from the validated top detector (read from
# `euphoria_report.json`). Out-of-sample FA rates are reported against the
# same budget, whatever they turn out to be (the same validation/test
# separation).
#
# **Layer 3 — prerequisites (identical for every contestant, so the race is
# fair):** onset models only speak on days when 7d attention ≥ its own 120d
# median (the A1 hype-gate construction with multiplier 1 — parameter-free);
# top models inherit the incumbent's A1 gate unchanged (2× median). Both
# inherit the coverage gate A0 and the judgeable-window clip.
#
# **Sparsity** (the desk's "only a few names at once" rule of thumb) is
# reported for the winner as a descriptive check, not fitted.
#
# ## The contestants
#
# | Model | What it is | Why it is here |
# |---|---|---|
# | `random` | uniform scores, seeds 42/100/2026 | the floor any claim must clear |
# | `rules` | un-weighted mean of the bank (the euphoria-LEVEL construction; for TOP, with the incumbent A2/A2b gates baked in) | the incumbent design — parsimony reference |
# | `logreg` | logistic regression, class-weighted | the simplest learner; its coefficients are readable |
# | `gbm` | small gradient-boosted trees (depth ≤ 3), class-weighted, seeds ×3 | non-linear interactions, still small enough for ~130 positives |
# | `mlp` | one hidden layer (16), early stopping, seeds ×3 | a feature-only neural baseline |
#
# Every learner sees exactly the bank features the rules see — no learner
# gets information the rules lack, so any win is attributable to *weighting*,
# not *inputs*.

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import roc_auc_score, average_precision_score

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

RESEARCH_DIR = ROOT / "docs" / "research"
SEEDS = [42, 100, 2026]          # fixed seeds so the tournament is reproducible

# %% [markdown]
# ## Data: the labelled day frame + the derived FA budget

# %%
from analytics.euphoria import build_all_series
from analytics.euphoria_phases import (build_day_frame, run_tournament_entry,
                                       ONSET_WINDOW_DAYS)
from analytics.loaders import load, THEME_COUNTS, THEME_SENT, \
    TICKER_COUNTS, TICKER_SENT

t0 = time.time()
prices = pd.read_parquet(ROOT / "data" / "prices" / "prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
series, pxmap = build_all_series(prices)
episodes = pd.read_parquet(ROOT / "data" / "processed" / "episodes.parquet")
counts = {"theme": load(THEME_COUNTS), "ticker": load(TICKER_COUNTS)}
sents = {"theme": load(THEME_SENT), "ticker": load(TICKER_SENT)}
for d in list(counts.values()) + list(sents.values()):
    d["date"] = pd.to_datetime(d["date"])
frame = build_day_frame(series, pxmap, episodes, counts, sents)

FA_BUDGET = json.load(open(ROOT / "data" / "processed" /
                           "euphoria_report.json"))["overall"][
                               "fa_per_instrument_year"]
from analytics.euphoria_phases import ONSET_BANK  # the LOCKED bank
from analytics.euphoria_phases import TOP_FEATURES as TOP_BANK

# Layer-3 prerequisites, applied identically to every contestant
onset_frame = frame[frame.hype_raw >= 1].copy()
top_frame = frame[frame.hype_ok].copy()
print(f"{time.time()-t0:.0f}s | onset candidates {len(onset_frame):,} "
      f"(prev {onset_frame.y_onset.mean():.3f}) | top candidates "
      f"{len(top_frame):,} (prev {top_frame.y_top.mean():.3f}) | "
      f"FA budget {FA_BUDGET}/instr-yr (incumbent's accepted level)")

# %% [markdown]
# ## The scorer factories
#
# Each returns `f(train_df, apply_df, feats) -> score array`. The rules
# scorer ignores `train_df` entirely — that is the point of rules.

# %%
def make_rules(label):
    def f(train, apply, feats):
        score = apply[feats].mean(axis=1).values
        if label == "y_top":
            # the incumbent's non-fitted gates A2/A2b, baked into the score
            gate = (apply["e1"] >= 0.90) & (apply["e2"] > 0)
            score = np.where(gate, score, 0.0)
        return score
    return f

def make_random(seed):
    def f(train, apply, feats):
        return np.random.default_rng(seed + len(apply)).uniform(
            size=len(apply))
    return f

def make_logreg(label):
    from sklearn.linear_model import LogisticRegression
    def f(train, apply, feats):
        m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(train[feats], train[label])
        return m.predict_proba(apply[feats])[:, 1]
    return f

def make_gbm(label, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    def f(train, apply, feats):
        m = HistGradientBoostingClassifier(
            max_depth=3, class_weight="balanced", random_state=seed)
        m.fit(train[feats], train[label])
        return m.predict_proba(apply[feats])[:, 1]
    return f

def make_mlp(label, seed):
    from sklearn.neural_network import MLPClassifier
    def f(train, apply, feats):
        m = MLPClassifier(hidden_layer_sizes=(16,), early_stopping=True,
                          max_iter=400, random_state=seed)
        m.fit(train[feats], train[label])
        return m.predict_proba(apply[feats])[:, 1]
    return f

# %% [markdown]
# ## Run the tournament

# %%
def run_task(task_frame, feats, label, mode):
    entries = {}
    entries["rules"] = [run_tournament_entry(
        task_frame, episodes, feats, label, mode, make_rules(label),
        FA_BUDGET)]
    entries["logreg"] = [run_tournament_entry(
        task_frame, episodes, feats, label, mode, make_logreg(label),
        FA_BUDGET)]
    for name, maker in [("random", make_random),
                        ("gbm", lambda s: make_gbm(label, s)),
                        ("mlp", lambda s: make_mlp(label, s))]:
        entries[name] = [run_tournament_entry(
            task_frame, episodes, feats, label, mode,
            maker(seed) if name != "random" else make_random(seed),
            FA_BUDGET) for seed in SEEDS]
    return entries

def leaderboard(entries):
    rows = []
    for model, runs in entries.items():
        rows.append({
            "model": model, "runs": len(runs),
            "captured": np.mean([r["captured"] for r in runs]),
            "capture_rate": np.mean([r["capture_rate"] for r in runs]),
            "late": np.mean([r["late"] for r in runs]),
            "false_alarms": np.mean([r["false_alarms"] for r in runs]),
            "fa_per_iy": np.mean([r["fa_per_iy"] for r in runs]),
            "ap": np.mean([r["ap"] for r in runs]),
            "ap_std": np.std([r["ap"] for r in runs]),
            "auroc": np.mean([r["auroc"] for r in runs]),
            "detectable": runs[0]["detectable"],
        })
    return (pd.DataFrame(rows)
            .sort_values("ap", ascending=False).reset_index(drop=True))

t0 = time.time()
onset_entries = run_task(onset_frame, ONSET_BANK, "y_onset", "onset")
onset_board = leaderboard(onset_entries)
print(f"onset tournament: {time.time()-t0:.0f}s")
onset_board.round(3)

# %%
t0 = time.time()
top_entries = run_task(top_frame, TOP_BANK, "y_top", "top")
top_board = leaderboard(top_entries)
print(f"top tournament: {time.time()-t0:.0f}s")
top_board.round(3)

# %% [markdown]
# ## Is the best learner's AP edge real? (cluster-bootstrap, parsimony rule)
#
# The rules can only be dethroned if the challenger's AP clears the rules'
# AP outside a 90% instrument-cluster bootstrap CI of the *difference* —
# computed on identical test rows, so the comparison is paired.

# %%
def paired_ap_bootstrap(task_frame, feats, label, mode, scorer_a, scorer_b,
                        n_boot=200, seed=42):
    """90% CI of AP(model_a) - AP(model_b) on the stacked walk-forward
    test scores, resampling instruments (clusters) with replacement."""
    from analytics.euphoria_phases import walk_forward_scores
    sa = walk_forward_scores(task_frame, feats, label, scorer_a)
    sb = walk_forward_scores(task_frame, feats, label, scorer_b)
    key = ["name", "date"]
    merged = sa[key + [label, "score"]].merge(
        sb[key + ["score"]], on=key, suffixes=("_a", "_b"))
    groups = dict(tuple(merged.groupby("name")))
    names = list(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        sub = pd.concat([groups[n] for n in pick])
        if sub[label].nunique() < 2:
            continue
        diffs.append(average_precision_score(sub[label], sub["score_a"])
                     - average_precision_score(sub[label], sub["score_b"]))
    return float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))

best_learner_name = onset_board[~onset_board.model.isin(["rules", "random"])
                                ].iloc[0]["model"]
maker = {"logreg": lambda: make_logreg("y_onset"),
         "gbm": lambda: make_gbm("y_onset", 42),
         "mlp": lambda: make_mlp("y_onset", 42)}[best_learner_name]
lo, hi = paired_ap_bootstrap(onset_frame, ONSET_BANK, "y_onset", "onset",
                             maker(), make_rules("y_onset"))
print(f"ONSET: AP({best_learner_name}) − AP(rules) 90% CI = "
      f"[{lo:+.4f}, {hi:+.4f}] -> "
      f"{'learner clears rules' if lo > 0 else 'PARSIMONY: rules win'}")

best_top_name = top_board[~top_board.model.isin(["rules", "random"])
                          ].iloc[0]["model"]
maker_t = {"logreg": lambda: make_logreg("y_top"),
           "gbm": lambda: make_gbm("y_top", 42),
           "mlp": lambda: make_mlp("y_top", 42)}[best_top_name]
lo_t, hi_t = paired_ap_bootstrap(top_frame, TOP_BANK, "y_top", "top",
                                 maker_t(), make_rules("y_top"))
print(f"TOP:   AP({best_top_name}) − AP(rules) 90% CI = "
      f"[{lo_t:+.4f}, {hi_t:+.4f}] -> "
      f"{'learner clears rules' if lo_t > 0 else 'PARSIMONY: rules win'}")

# %% [markdown]
# ## Apply the criterion mechanically

# %%
def apply_criterion(board, ci_lo, best_learner):
    """Layer 1, as pre-stated: eligibility vs random (AP and AUROC - the
    score-quality floor), AP ranking, and the parsimony rule for learners
    vs rules."""
    rnd = board[board.model == "random"].iloc[0]
    eligible = board[(board.model != "random")
                     & (board.ap > rnd.ap)
                     & (board.auroc > rnd.auroc)]
    if eligible.empty:
        raise RuntimeError("NO MODEL beats random on score quality - "
                           "nothing ships; stop and investigate")
    top_row = eligible.iloc[0]
    if top_row.model == best_learner and ci_lo <= 0:
        # learner leads on point AP but not outside noise -> rules
        if "rules" in eligible.model.values:
            return "rules", (f"{best_learner} leads on AP but not outside "
                             "the bootstrap CI - rules win by parsimony")
    return top_row.model, f"{top_row.model} wins on AP {top_row.ap:.3f}"

onset_winner, onset_why = apply_criterion(onset_board, lo, best_learner_name)
top_winner, top_why = apply_criterion(top_board, lo_t, best_top_name)
print("ONSET verdict:", onset_winner, "-", onset_why)
print("TOP   verdict:", top_winner, "-", top_why)

# %% [markdown]
# ## The winners' scorecards (per year, both denominators, budget line)

# %%
def winner_entry(entries, name):
    return entries[name][0]      # seed-42 run for stochastic models

ow = winner_entry(onset_entries, onset_winner)
tw = winner_entry(top_entries, top_winner)

def per_year_table(entry, mode):
    det_col = ("onset_detectable" if mode == "onset" else "top_detectable")
    rows = []
    alerts = [(n, a) for n, al in entry["alerts_by_name"].items()
              for a in al]
    for y in entry["test_years"]:
        det = episodes[(episodes.year == y) & episodes[det_col]]
        allp = episodes[episodes.year == y]
        rows.append({"year": y, "episodes": len(allp),
                     "detectable": len(det),
                     "alerts": sum(1 for _, a in alerts if a.year == y),
                     "threshold": round(entry["thresholds"][y], 3)})
    return pd.DataFrame(rows)

print(f"ONSET winner ({onset_winner}): captured {ow['captured']}/"
      f"{ow['detectable']} detectable ({ow['capture_rate']:.0%}), "
      f"late {ow['late']}, FA {ow['false_alarms']} "
      f"({ow['fa_per_iy']}/instr-yr vs budget {FA_BUDGET})")
display(per_year_table(ow, "onset"))
print(f"TOP winner ({top_winner}): captured {tw['captured']}/"
      f"{tw['detectable']} detectable ({tw['capture_rate']:.0%}), "
      f"FA {tw['false_alarms']} ({tw['fa_per_iy']}/instr-yr)")
display(per_year_table(tw, "top"))

# %% [markdown]
# ## Sparsity check (the desk's rule of thumb, reported not fitted)
#
# For each calendar day: how many instruments issued an alert in the
# trailing 7 days? "Euphoria should be rare" means this line should hug
# zero and spike only in genuine mania seasons.

# %%
def sparsity_series(entry):
    all_alerts = [a for al in entry["alerts_by_name"].values() for a in al]
    if not all_alerts:
        return pd.Series(dtype=float)
    idx = pd.date_range(min(all_alerts), max(all_alerts), freq="D")
    s = pd.Series(0, index=idx)
    for a in all_alerts:
        s.loc[a] += 1
    return s.rolling(7).sum()

fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
for ax, entry, ttl, c in [(axes[0], ow, f"ONSET ({onset_winner})", C1),
                          (axes[1], tw, f"TOP ({top_winner})", C2)]:
    s = sparsity_series(entry)
    ax.plot(s.index, s.values, color=c, lw=1.2)
    ax.set_title(f"{ttl} — instruments alerted in trailing 7d "
                 f"(mean {s.mean():.2f}, max {int(s.max())})")
    despine(ax)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Why the threshold is the number it is (nothing hand-picked)
#
# The alert threshold is the ONE fitted quantity in each detector — a
# **learned output, not a chosen input**. Two selection rules exist in
# this project and both are drawn below, on the exact training evidence
# they see (all years before the newest; the candidate grid is the
# percentiles of the training scores, so even the GRID is data-derived,
# not hand units):
#
# * **Utility rule** (the incumbent top detector's original):
#   maximise `captures − 1.0 × FAs`. The curves show its honest
#   weakness in an FA-rich environment: FAs shrink faster than captures
#   as the threshold rises, so utility keeps improving toward the
#   conservative end and the selection saturates at the top of whatever
#   grid it is given — the incumbent's historical "85" is partly "the
#   most conservative candidate on its 50–85 grid". **Recorded as a
#   limitation, not hidden.**
# * **Budget rule** (Layer 2 of this tournament — the selection actually
#   used for the onset detector): maximise captures SUBJECT TO the
#   false-alarm budget the desk already accepted (0.23/instr-yr). This
#   picks an INTERIOR point — the loosest threshold that still respects
#   the accepted noise level — and cannot saturate at a grid edge.
#
# Either way: "why not 80? why not 90?" is answered by these curves, and
# if new data reshapes them, the next research pass moves the number
# automatically. No human ever picks it.

# %%
from analytics.euphoria_phases import _pregroup, _tally

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
for ax, (task_frame, feats, label, mode, ttl) in zip(axes, [
        (top_frame, TOP_BANK, "y_top", "top", "TOP detector"),
        (onset_frame, ONSET_BANK, "y_onset", "onset", "ONSET detector")]):
    train = task_frame[task_frame.year < task_frame.year.max()]
    scorer = make_rules(label)
    scored = train.assign(score=scorer(train, train, feats))
    groups = _pregroup(scored, episodes)
    yrs = sorted(train.year.unique())
    in_years = (lambda ys: (lambda e: e.year.isin(ys)))(yrs)
    n_iy = train["name"].nunique() * len(yrs)
    grid = np.unique(np.percentile(scored["score"].dropna(),
                                   np.arange(40, 100, 2)))
    rows = []
    for thr in grid:
        r = _tally(groups, episodes, float(thr), mode, in_years)
        rows.append({"thr": thr, "cap": r["captured"],
                     "fa": r["false_alarms"],
                     "fa_rate": r["false_alarms"] / n_iy,
                     "u": r["captured"] - r["false_alarms"]})
    cur = pd.DataFrame(rows)
    ax.plot(cur.thr, cur.cap, color=C2, lw=1.8, label="captures (train)")
    ax.plot(cur.thr, cur.fa, color=C3, lw=1.8, label="false alarms")
    ax.plot(cur.thr, cur.u, color=C1, lw=2.0,
            label="utility = captures − FAs")
    feas = cur[cur.fa_rate <= FA_BUDGET]
    if len(feas):
        budget_pick = feas.sort_values(["cap", "thr"]).iloc[-1]
        ax.axvspan(feas.thr.min(), cur.thr.max(), color=C2, alpha=0.08,
                   lw=0)
        ax.axvline(budget_pick.thr, color=C2, lw=1.6, ls="--")
        ax.text(budget_pick.thr, ax.get_ylim()[1] * 0.86,
                f"  budget rule: {budget_pick.thr:.2f}\n"
                f"  (cap {int(budget_pick.cap)}, FA within "
                f"{FA_BUDGET}/iy)", fontsize=8, color=C2)
    upick = cur.loc[cur.u.idxmax()]
    ax.axvline(upick.thr, color=C1, lw=1.2, ls=":")
    ax.text(upick.thr, ax.get_ylim()[0] * 0.5 if ax.get_ylim()[0] < 0
            else ax.get_ylim()[1] * 0.5,
            f"  utility rule: {upick.thr:.2f}\n  (saturates toward the "
            "grid edge)", fontsize=8, color=C1)
    ax.set_title(f"{ttl} — shaded = within the accepted FA budget")
    ax.set_xlabel("candidate threshold (training-score percentile grid)")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    despine(ax)
fig.suptitle("The training evidence behind each threshold — two selection "
             "rules, both drawn, neither hand-picked", y=1.03,
             fontsize=11, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Tested variant: PHASE-AWARE onset candidacy (verdict recorded)
#
# **Hypothesis (motivated by NB02's finding that E2 is anti-predictive at
# onset):** a name whose END-stage state is already fully active (all
# three top gates jointly: E1 >= 0.90, E2 > 0, 2x hype) cannot be
# "starting" - exclude those days from onset candidacy. No new constant:
# the exclusion reuses the top detector's own gate set.
#
# **Adoption rule, pre-stated:** adopt only if utility (captures - FAs)
# improves AND LATE alerts drop.

# %%
end_stage = (frame.e1 >= 0.90) & (frame.e2 > 0) & frame.hype_ok
pa_frame = frame[(frame.hype_raw >= 1) & ~end_stage].copy()
r_pa = run_tournament_entry(pa_frame, episodes, ONSET_BANK, "y_onset",
                            "onset", make_rules("y_onset"), FA_BUDGET)
r_inc = onset_entries["rules"][0]
cmp_tbl = pd.DataFrame([
    {"variant": "incumbent", "captured": r_inc["captured"],
     "late": r_inc["late"], "FA": r_inc["false_alarms"],
     "AP": r_inc["ap"],
     "utility": r_inc["captured"] - r_inc["false_alarms"]},
    {"variant": "phase-aware", "captured": r_pa["captured"],
     "late": r_pa["late"], "FA": r_pa["false_alarms"], "AP": r_pa["ap"],
     "utility": r_pa["captured"] - r_pa["false_alarms"]}])
display(cmp_tbl)
adopted_pa = (cmp_tbl.utility.iloc[1] > cmp_tbl.utility.iloc[0]
              and cmp_tbl.late.iloc[1] < cmp_tbl.late.iloc[0])
print("VERDICT:", "ADOPTED" if adopted_pa else
      "REJECTED under the pre-stated rule - it halves start/end "
      "adjacency and cuts LATE alerts, but costs "
      f"{cmp_tbl.captured.iloc[0]-cmp_tbl.captured.iloc[1]} genuine "
      "captures with no utility gain. The display-side fix (episode-span "
      "shading) addresses the adjacency readability instead, at zero "
      "capture cost. The desk may overrule with a recorded decision if "
      "message cleanliness is worth the captures.")

# %% [markdown]
# ## Labelling-criteria sensitivity
#
# Conclusions that only hold under one labelling rule are not conclusions,
# so the labelling constant is varied and the models re-run. Here the boom
# thresholds are probed one step softer (20% ETF / 40% single) and one step
# harder (30% / 60%) around the desk's 25/50, and the WINNING onset model is
# re-run under the full discipline against each variant truth. The claim
# being tested: the capture/FA profile moves smoothly — no cliff, no
# sign flip.

# %%
import analytics.euphoria as euph
from analytics.euphoria_phases import episode_catalog, label_days

def relabel(task_frame, eps_variant):
    """Swap the label columns for a variant episode truth (features are
    label-independent, so only y_* changes)."""
    parts = []
    for name, g in task_frame.groupby("name"):
        idx = pd.DatetimeIndex(g["date"])
        lab = label_days(idx, eps_variant, name)
        h = g.drop(columns=["y_onset", "y_late", "y_top"]).reset_index(
            drop=True)
        parts.append(pd.concat([h, lab.reset_index(drop=True)], axis=1))
    return pd.concat(parts, ignore_index=True)

sens_rows = []
VARIANTS = [("softened (20/40)", 0.20, 0.40),
            ("standard (25/50)", 0.25, 0.50),
            ("hardened (30/60)", 0.30, 0.60)]
orig = (euph.EUPHORIA_BOOM_MIN_ETF, euph.EUPHORIA_BOOM_MIN_SINGLE)
maker_w = {"rules": lambda: make_rules("y_onset"),
           "logreg": lambda: make_logreg("y_onset"),
           "gbm": lambda: make_gbm("y_onset", 42),
           "mlp": lambda: make_mlp("y_onset", 42)}[onset_winner]
for label_name, etf, single in VARIANTS:
    euph.EUPHORIA_BOOM_MIN_ETF, euph.EUPHORIA_BOOM_MIN_SINGLE = etf, single
    eps_v = episode_catalog(series, pxmap)
    fr_v = relabel(onset_frame, eps_v)
    r = run_tournament_entry(fr_v, eps_v, ONSET_BANK, "y_onset", "onset",
                             maker_w(), FA_BUDGET)
    sens_rows.append({"labelling": label_name,
                      "detectable": r["detectable"],
                      "captured": r["captured"],
                      "capture_rate": r["capture_rate"],
                      "false_alarms": r["false_alarms"],
                      "fa_per_iy": r["fa_per_iy"], "ap": r["ap"]})
euph.EUPHORIA_BOOM_MIN_ETF, euph.EUPHORIA_BOOM_MIN_SINGLE = orig
pd.DataFrame(sens_rows)

# %% [markdown]
# ## Commissioned test (desk, 2026-07-24): a PRICE-ASSISTED END gate
#
# **The desk's question:** does requiring an actual price boom before an
# END alert can fire ("it has to rise X% first, so we don't flag random
# fluctuations") improve accuracy?
#
# **Design, pre-stated:**
# * The gate is **G2's own boom thresholds** — price ≥ 25% (ETF) / 50%
#   (single) above its trailing 120d low, computed from PAST prices only
#   (no look-ahead; no new constant). A half-strength gate is included as
#   a dose-response check: a real mechanism should show a gradient.
# * **The claim changes.** This variant is NOT crowd-only: it defends
#   "the crowd called it and the chart confirmed a boom", not "the crowd
#   alone called it". It can therefore only ship as a clearly-labelled
#   SECOND signal — never silently replace the crowd-only detector. (The
#   project's earlier price-assisted variant captured 46% vs 23%; that
#   delta was recorded as the cost of the crowd-only claim. This test
#   re-measures it under the current discipline.)
# * **Adoption rule:** offer as a labelled overlay if, on MATCHED test
#   years, it improves BOTH capture and utility with absolute FAs not
#   worse, and the capture gain's cluster-bootstrap CI excludes zero.

# %%
from src.config import EUPHORIA_BOOM_MIN_ETF, EUPHORIA_BOOM_MIN_SINGLE
from analytics.euphoria_phases import (classify_top_alerts, _eps_arrays,
                                       _day_ints)

def boom_state_col(mult):
    rows = []
    for es in series:
        px = pxmap[es.symbol].dropna().asfreq("D").ffill()
        low120 = px.rolling(120, min_periods=60).min()
        bm = (EUPHORIA_BOOM_MIN_SINGLE if es.kind == "single"
              else EUPHORIA_BOOM_MIN_ETF) * mult
        st = (px / low120 - 1) >= bm
        rows.append(pd.DataFrame({"name": es.name, "date": st.index,
                                  "st": st.values}))
    return pd.concat(rows, ignore_index=True)

fpx = frame.merge(boom_state_col(1.0).rename(columns={"st": "boom_full"}),
                  on=["name", "date"], how="left")
fpx = fpx.merge(boom_state_col(0.5).rename(columns={"st": "boom_half"}),
                on=["name", "date"], how="left")
fpx[["boom_full", "boom_half"]] = fpx[["boom_full", "boom_half"]].fillna(False)

variants = {
    "crowd-only (incumbent)": fpx[fpx.hype_ok].copy(),
    "+ half boom gate": fpx[fpx.hype_ok & fpx.boom_half].copy(),
    "+ FULL boom gate (G2)": fpx[fpx.hype_ok & fpx.boom_full].copy(),
}
entries_px = {lbl: run_tournament_entry(fr, episodes, TOP_BANK, "y_top",
                                        "top", make_rules("y_top"),
                                        FA_BUDGET)
              for lbl, fr in variants.items()}

# matched test years: per-iy rates are only comparable on the SAME years
common_years = set.intersection(*[set(e["test_years"])
                                  for e in entries_px.values()])
def matched_scores(entry):
    captured, fa = set(), 0
    eps_by = dict(tuple(episodes.groupby("name")))
    for name, alerts in entry["alerts_by_name"].items():
        al = [a for a in alerts if a.year in common_years]
        if not al:
            continue
        res = classify_top_alerts(
            _day_ints(pd.DatetimeIndex(sorted(al))),
            _eps_arrays(eps_by.get(name, episodes.iloc[0:0])))
        captured |= {(name, p) for p in res["captured"]}
        fa += len(res["fa"])
    det = episodes[episodes.year.isin(common_years) & episodes.top_detectable]
    return len(captured), len(det), fa

rows = []
for lbl, e in entries_px.items():
    cap, det, fa = matched_scores(e)
    rows.append({"variant": lbl, "captured": cap, "detectable": det,
                 "capture_rate": round(cap / det, 3),
                 "false_alarms": fa, "utility": cap - fa,
                 "AP": e["ap"], "AUROC": e["auroc"]})
px_table = pd.DataFrame(rows)
display(px_table)

# cluster-bootstrap CI of the capture-rate GAIN (full gate vs incumbent)
def per_name_caps(entry):
    eps_by = dict(tuple(episodes.groupby("name")))
    out = {}
    for es in series:
        name = es.name
        al = [a for a in entry["alerts_by_name"].get(name, [])
              if a.year in common_years]
        res = classify_top_alerts(
            _day_ints(pd.DatetimeIndex(sorted(al))),
            _eps_arrays(eps_by.get(name, episodes.iloc[0:0]))) if al             else {"captured": set()}
        det = episodes[(episodes.name == name)
                       & episodes.year.isin(common_years)
                       & episodes.top_detectable]
        out[name] = (len(res["captured"]), len(det))
    return out

inc_c = per_name_caps(entries_px["crowd-only (incumbent)"])
ful_c = per_name_caps(entries_px["+ FULL boom gate (G2)"])
rng = np.random.default_rng(42)
names_ = list(inc_c)
gains = []
for _ in range(500):
    pick = rng.choice(names_, size=len(names_), replace=True)
    ci_, cf_, d_ = 0, 0, 0
    for n in pick:
        ci_ += inc_c[n][0]; cf_ += ful_c[n][0]; d_ += inc_c[n][1]
    if d_:
        gains.append((cf_ - ci_) / d_)
lo_g, hi_g = np.percentile(gains, [5, 95])
print(f"capture-rate GAIN (full gate − incumbent), 90% cluster CI: "
      f"[{lo_g:+.3f}, {hi_g:+.3f}] -> "
      f"{'CI excludes zero' if lo_g > 0 else 'CI includes zero'}")

inc_row = px_table.iloc[0]; ful_row = px_table.iloc[2]
adopt = (ful_row.captured > inc_row.captured
         and ful_row.utility > inc_row.utility
         and ful_row.false_alarms <= inc_row.false_alarms
         and lo_g > 0)
print("VERDICT:", ("OFFER AS LABELLED SECOND SIGNAL - all pre-stated "
                   "conditions met. The crowd-only detector remains the "
                   "headline (its claim is different, not worse); the "
                   "price-assisted variant is the desk overlay for "
                   "risk-timing accuracy.") if adopt else
      "NOT offered - pre-stated conditions not met.")

# %% [markdown]
# ## Verdict & what ships
#
# The verdict cells above are mechanical applications of the criterion in
# the first cell; the narrative below is written to the *structure*, and the
# numbers re-render on every rebuild:
#
# * The winner of each task is whatever survived Layer 1 (beat random,
#   best AP, parsimony rule applied). The sensitivity table must show a
#   smooth capture/FA profile across labelling variants for the winner to
#   be trusted.
# * The onset detector's out-of-sample FA rate is reported against the
#   derived budget. If it exceeds the budget, that is the documented cost
#   of onset detection at this coverage level — the dashboard pane ships
#   with the measured rate on its face, and the desk decides with open
#   eyes.
# * **Notebook 04** takes the winners to the full evaluation: lead-time
#   distributions, FA anatomy, calibration, case studies, and the
#   trading-translation study (pre-stated criterion of its own).

# %%
out = {
    "criterion": {
        "layer1": "AP ranking, must beat random on AP+capture, parsimony "
                  "CI rule for learners vs rules",
        "layer2": f"train-time constrained max capture s.t. FA <= "
                  f"{FA_BUDGET}/instr-yr (incumbent's accepted level)",
        "layer3": "onset gate: 7d share >= own 120d median (A1 with "
                  "multiplier 1); top gate: incumbent A1 unchanged",
    },
    "onset": {"winner": onset_winner, "why": onset_why,
              "board": onset_board.to_dict(orient="records"),
              "winner_scorecard": {k: v for k, v in ow.items()
                                   if k not in ("leads", "alerts_by_name")}},
    "top": {"winner": top_winner, "why": top_why,
            "board": top_board.to_dict(orient="records"),
            "winner_scorecard": {k: v for k, v in tw.items()
                                 if k not in ("leads", "alerts_by_name")}},
    "sensitivity": sens_rows,
}
with open(RESEARCH_DIR / "nb03_tournament.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("saved nb03_tournament.json")

# %% [markdown]
# ---
# # SS — The 2026-08 desk tournament (the shipped selection)
#
# Everything above is the July record: it selected the RULES bank under the
# 0.23 FA budget and the gated candidacies. In August the desk re-framed
# the question ("no prior thresholds — best accuracy, explainable, and not
# just rules"), and the selection below is what now ships
# (`analytics/ml_detector.py`, called by the SAME production path the
# dashboard reads — this notebook imports that module, it does not
# re-implement it).
#
# **What changed relative to the July race, and why each change is safe:**
#
# 1. **Candidacy collapses to the coverage gate.** The hype gate, the boom
#    gate and the end-stage exclusion become FEATURES (continuous versions
#    of the same measurements), so the model learns the cut instead of a
#    constant hard-coding it. Three numbers survive: the coverage floor,
#    the 21d cooldown, and a probability cut chosen walk-forward by F1.
# 2. **The bank is un-gated.** e2's embedded 75% persistence gate is
#    split into its two raw ingredients (`bull_level`, `bull_persist`);
#    the desk heads also receive the price pair (`price_runup`,
#    `price_ret21`) — the desk configuration is licensed to use price,
#    and the July incumbent already spent that licence on a hard gate.
# 3. **The selection rule** (stated before computing): ONE model family
#    for both heads; highest COMBINED test AP lift (AP over its own
#    frame's base rate — raw AP is not comparable across candidacy
#    frames); ties by AUROC, then fewer false alarms.
#
# The crowd-only variants of every learner are run alongside as the
# clean-claim record ("the crowd alone"), and the incumbent rules run
# unchanged as the baseline row.

# %%
from analytics import ml_detector as mld                    # noqa: E402
from analytics.euphoria_phases import boom_state_frame      # noqa: E402

_prices, _series, _pxmap, _counts, _sents = mld._load_all()
_sym_by = {es.name: es.symbol for es in _series}
_eps_aug = episode_catalog(_series, _pxmap)
_frame_aug = build_day_frame(_series, _pxmap, _eps_aug, _counts, _sents)
_boom_aug = boom_state_frame(_series, _pxmap)
print(f"universe {len(_series)} | episodes {len(_eps_aug)} "
      f"(ground truth as frozen in src/config.py) | "
      f"candidate days {len(_frame_aug):,}")

# %%
_tour = mld.run_ml_tournament(_frame_aug, _eps_aug, _pxmap, _sym_by,
                              series=_series, boom=_boom_aug)
_winner = mld.pick_winner(_tour)
print("winner (one family, both heads, combined AP lift):", _winner)

# %%
# the thesis-style board, one per head
_rows = []
for _head in ("get_out", "get_in"):
    for _m, _r in _tour[_head].items():
        if "error" in _r:
            continue
        _rows.append({
            "head": _head, "model": _m,
            "AP": _r.get("ap"), "AP base": _r.get("ap_baseline"),
            "AP lift": (round(_r["ap"] / _r["ap_baseline"], 2)
                        if _r.get("ap") and _r.get("ap_baseline")
                        else None),
            "AUROC": _r.get("auroc"),
            "captured": f"{_r.get('captured')}/{_r.get('detectable')}",
            "capture %": (round(100 * (_r.get("capture_rate") or 0)))
            if _r.get("capture_rate") is not None else None,
            "FA/instr-yr": _r.get("fa_per_iy"),
            "precision": _r.get("precision"),
            "median lead (d)": _r.get("median_lead_days"),
        })
_board_aug = pd.DataFrame(_rows)
_board_aug

# %%
# forward price moves after the winner's alerts — the finance-audience
# row: what did price DO in the 1 week / 1 month / 1 quarter after a call
for _head in ("get_out", "get_in"):
    _fr = _tour[_head][_winner].get("forward_returns", {})
    print(_head, {k: v for k, v in _fr.items()})

# %%
# persist alongside the July record (additive — the July board above is
# unchanged evidence, this is the August selection that ships)
out_aug = {
    "criterion": "one family both heads; combined test AP lift; "
                 "ties AUROC then fewer FAs; operating point = "
                 "train-year episode F1 (no FA budget)",
    "winner": _winner,
    "board": _board_aug.to_dict(orient="records"),
}
with open(RESEARCH_DIR / "nb03_tournament.json") as f:
    _prev = json.load(f)
_prev["august_desk_tournament"] = out_aug
with open(RESEARCH_DIR / "nb03_tournament.json", "w") as f:
    json.dump(_prev, f, indent=1, default=str)
print("nb03_tournament.json updated with the August selection")

# %% [markdown]
# ---
# # SS1 — Head-to-head: the NEW model vs the PREVIOUS rules system
#
# **WHY THIS** — the single question the desk asks first: *is the new
# approach actually better than what we ran before?* This section answers
# it on one table and one figure, with the comparison rules stated first
# so the answer cannot be an artifact of how it was framed:
#
# * **Same episode truth** — both systems are graded against the identical
#   frozen ground-truth catalogue (`src/config.py` bars, notebook 01).
# * **Same discipline** — both run through the identical walk-forward
#   machinery (`run_tournament_entry`): every alert in year Y comes from a
#   configuration that saw only years before Y.
# * **Each on its own candidacy** — the rules keep their gates (that IS
#   the incumbent); the ML runs coverage-only. Because the two frames have
#   different base rates, raw AP is NOT comparable — **AP lift** ("how
#   many times better than guessing on your own frame") is the
#   like-for-like score-quality number, and capture/precision/FA are
#   graded on the same episode catalogue either way.

# %%
_HH_METRICS = [
    ("episodes caught", lambda r: f"{r['captured']}/{r['detectable']}"
                                  f" ({r['capture_rate']:.0%})"),
    ("precision (caught / all alerts graded)",
     lambda r: f"{r['precision']:.0%}"),
    ("false alarms per instrument-year", lambda r: f"{r['fa_per_iy']:.2f}"),
    ("median lead (days)", lambda r: r["median_lead_days"]),
    ("AUROC (0.5 = coin flip)", lambda r: f"{r['auroc']:.2f}"),
    ("AP lift (1.0x = guessing)",
     lambda r: f"{r['ap'] / r['ap_baseline']:.2f}x"),
]
_hh_tbl = {}
for _head, _hlbl in (("get_out", "GET OUT"), ("get_in", "GET IN")):
    for _m, _mlbl in (("rules", "previous rules"), (_winner, "NEW model")):
        _r = _tour[_head][_m]
        _hh_tbl[(_hlbl, _mlbl)] = {k: f(_r) for k, f in _HH_METRICS}
_hh = pd.DataFrame(_hh_tbl)
display(_hh)

# %%
# the finance-audience clincher: what price actually DID after each
# system's calls (median % move; a GET OUT call done well precedes
# flat-to-falling prices, a GET IN call rising ones)
print("median price move AFTER each call (5d / 21d / 84d):")
for _head, _want in (("get_out", "flat-to-NEGATIVE if calls are real"),
                     ("get_in", "POSITIVE if calls are real")):
    print(f"  {_head.upper():8s} (want: {_want})")
    for _m, _mlbl in (("rules", "previous rules"), (_winner, "NEW model")):
        _fr = _tour[_head][_m]["forward_returns"]
        print(f"    {_mlbl:15s} " + "  ".join(
            f"{_fr[f'fwd_{h}d']['median_pct']:+.2f}%" for h in (5, 21, 84)))

# %%
# one figure: the two comparable axes, side by side, saved for the deck
_fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6))
_heads = ["GET OUT", "GET IN"]
_x = np.arange(2)
for _ax, _metric, _fmt, _ttl in (
        (_axes[0], lambda r: r["capture_rate"], "{:.0%}",
         "episodes caught (share of detectable)"),
        (_axes[1], lambda r: r["ap"] / r["ap_baseline"], "{:.2f}x",
         "AP lift — score quality vs guessing\non its own frame")):
    _rv = [_metric(_tour[h][("rules")]) for h in ("get_out", "get_in")]
    _mv = [_metric(_tour[h][_winner]) for h in ("get_out", "get_in")]
    _b1 = _ax.bar(_x - 0.18, _rv, 0.32, color=MUTED,
                  label="previous rules")
    _b2 = _ax.bar(_x + 0.18, _mv, 0.32, color=C1, label="NEW model")
    for _bars in (_b1, _b2):
        for _b in _bars:
            _ax.text(_b.get_x() + _b.get_width() / 2, _b.get_height(),
                     _fmt.format(_b.get_height()), ha="center",
                     va="bottom", fontsize=8.5)
    _ax.set_xticks(_x, _heads)
    _ax.set_title(_ttl, fontsize=9.5)
    _ax.legend(frameon=False, fontsize=8)
    despine(_ax)
_fig.suptitle("New model vs previous rules — same episode truth, same "
              "walk-forward discipline", fontweight="bold", fontsize=11)
_fig.tight_layout()
for _p in (ROOT / "docs" / "figures" / "03" / "F22_rules_vs_ml.png",
           ROOT / "docs" / "figures" / "deck" / "F22_rules_vs_ml.png"):
    _p.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_p, dpi=180)
    print(f"saved {_p.relative_to(ROOT)}")
plt.show()

# %%
# persist the head-to-head beside the August board (one fact, one home)
with open(RESEARCH_DIR / "nb03_tournament.json") as f:
    _prev = json.load(f)
_prev["august_desk_tournament"]["head_to_head"] = {
    _head: {_m: {k: _tour[_head][_m].get(k) for k in
                 ("captured", "detectable", "capture_rate", "precision",
                  "fa_per_iy", "median_lead_days", "auroc", "ap",
                  "ap_baseline", "forward_returns")}
            for _m in ("rules", _winner)}
    for _head in ("get_out", "get_in")}
with open(RESEARCH_DIR / "nb03_tournament.json", "w") as f:
    json.dump(_prev, f, indent=1, default=str)
print("head-to-head persisted into nb03_tournament.json")

# %% [markdown]
# **SO WHAT — the verdict, stated plainly**
#
# * **GET IN: the new model is better on every axis.** ~6× the episodes
#   caught (57% vs 9% of detectable), ~3× the precision (50% vs 17%),
#   comparable false-alarm rate, twice the AP lift, and it arrives 10
#   days earlier in the episode. There is no dimension on which the old
#   onset rules win.
# * **GET OUT: the new model catches ~4× the tops (42% vs 12%)** at
#   double the AP lift, and — the decisive evidence — **price actually
#   behaved like a top after its calls** (median −0.1% / −0.2% / −2.6%
#   over 1wk/1mo/1qtr) whereas after the old rules' calls price kept
#   RISING (+0.5% / +0.4% / +4.0%): the old system's few calls were, on
#   the tape's own testimony, systematically early-or-wrong. The cost is
#   more false alarms (0.35 vs 0.09 per instrument-year) — exactly the
#   trade the desk authorised ("more false alarms is ok; catch the
#   episodes").
# * **The old system's headline precision (60%) is a base-rate artifact,
#   not skill**: its gates pre-filter to days where ~half are already
#   inside an episode window, so its AP lift — skill over guessing on its
#   own frame — is only 1.13×, barely above chance, and its AUROC (0.55)
#   says its *score* barely ranks. The new model's 2.45×/2.55× lift and
#   0.72/0.75 AUROC are the like-for-like comparison, and they are not
#   close.
# * And structurally: the old system needed ~7 hand-set constants
#   defended one by one; the new one needs three (coverage floor,
#   cooldown, and a cut that is *learned*, not chosen).

# %% [markdown]
# ---
# # SS2 — Inside the winner: what the ensemble is, and what it looks at
#
# **WHY THIS**
#
# * "We use an ensemble" is not an explanation. This section opens the
#   box: the exact training protocol, the two members' hyperparameters
#   (all stated, none tuned per-year), and — the part a PM actually asks
#   for — **which measurements the fitted model leans on**, read two
#   independent ways.
#
# **HOW IT WORKS — the protocol, in the order the code runs it**
#
# 1. **Candidacy**: every (instrument, day) with ≥100 tagged posts (posts
#    naming the instrument) in the
#    trailing 28d. No other doors — the old gates enter as features.
# 2. **Features**: the 11-column bank (9 crowd percentiles + the price
#    pair). All trailing; day t sees nothing after t.
# 3. **Fit, per test year Y**: on years < Y only —
#    * *logistic regression*: L2, class-balanced, max_iter=2000 — one
#      weight per feature, readable as a formula;
#    * *monotone GBM*: HistGradientBoosting, max_depth 3, learning rate
#      0.1, 200 rounds, balanced sample weights, and **every feature
#      constrained monotonically non-decreasing** — more crowd-heat can
#      only raise the score, so the tree ensemble cannot learn "high
#      attention is sometimes safe";
#    * *the ensemble*: average of the two models' probability RANKS
#      (rank space, so neither model's calibration dominates).
# 4. **Operating point, per test year Y**: the probability cut that
#    maximises episode-level F1 on the TRAIN years. No FA budget, no
#    penalty constant.
# 5. **Alerts**: cross the cut → fire; then one alert per name per 21d.
#
# Every number above is stated once, fixed for every year, and identical
# for both heads. `random_state=0` throughout: same data → same model.

# %% [markdown]
# ## SS2.0 — The training, run once in the open (GET OUT head, newest fold)
#
# The protocol above says what the training does; this section **does it,
# step by step, printing every intermediate** — the maximum-explainability
# walkthrough. One head (GET OUT), one fold (the newest test year), the
# exact same code path production uses (`analytics/ml_detector.py`
# builders — nothing re-implemented here). Every other fold and the GET IN
# head repeat this recipe verbatim with different years/labels.

# %%
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.inspection import permutation_importance        # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from analytics.ml_detector import _balanced_weights          # noqa: E402

# STEP 1 - CANDIDACY: the coverage gate only (>=100 tagged posts in the
# trailing 28d). No hype gate, no boom gate - those are features now.
_cand_px = mld.attach_price_features(
    mld.candidate_frame(_frame_aug), _series, _pxmap)
print(f"STEP 1  candidacy: {len(_cand_px):,} (instrument, day) rows over "
      f"{_cand_px['name'].nunique()} instruments, years "
      f"{int(_cand_px.year.min())}-{int(_cand_px.year.max())}")

# STEP 2 - FEATURES: the 11-column bank. Five real rows, exactly as the
# model sees them (crowd columns are percentiles vs the name's OWN
# history; the price pair is raw returns; all trailing - day t sees
# nothing after t):
print(f"STEP 2  the bank: {mld.DESK_ML_BANK}")
display(_cand_px.sample(5, random_state=0)
        .set_index(["name", "date"])[mld.DESK_ML_BANK].round(2))

# STEP 3 - THE FOLD: fit on every year before the newest, test on the
# newest. The model NEVER sees a test-year row while fitting.
_ymax = int(_cand_px["year"].max())
_tr = _cand_px[_cand_px["year"] < _ymax]
_te = _cand_px[_cand_px["year"] == _ymax]
print(f"STEP 3  fold: fit on {sorted(_tr.year.unique())} "
      f"({len(_tr):,} rows, {int(_tr.y_top.sum()):,} inside a top window "
      f"= {_tr.y_top.mean():.1%}) | test on {_ymax} ({len(_te):,} rows)")

# %%
# STEP 4a - MEMBER 1, the logistic regression: ONE signed weight per
# measurement. The fitted model IS this formula - nothing more:
_lg_w = LogisticRegression(class_weight="balanced", max_iter=2000)
_lg_w.fit(_tr[mld.DESK_ML_BANK], _tr["y_top"])
print("STEP 4a  P(top is near) = sigmoid(")
for _f, _w in sorted(zip(mld.DESK_ML_BANK, _lg_w.coef_[0]),
                     key=lambda t: -abs(t[1])):
    print(f"    {_w:+6.2f} x {mld.ML_BANK_LABELS.get(_f, _f)}")
print(f"    {_lg_w.intercept_[0]:+6.2f}  )   "
      "[positive weight = more of this, more risk]")

# STEP 4b - MEMBER 2, the monotone GBM: 200 boosting rounds of depth<=3
# trees, every feature constrained non-decreasing - the trees learn the
# interactions but CANNOT invert a direction ("high attention is
# sometimes safe" is unlearnable by construction):
_gb_w = HistGradientBoostingClassifier(
    max_depth=3, learning_rate=0.1, max_iter=200,
    monotonic_cst=[1] * len(mld.DESK_ML_BANK), random_state=0)
_gb_w.fit(_tr[mld.DESK_ML_BANK], _tr["y_top"],
          sample_weight=_balanced_weights(_tr["y_top"].values))
print(f"STEP 4b  monotone GBM fitted: {_gb_w.n_iter_} rounds kept, "
      f"depth<=3, all {len(mld.DESK_ML_BANK)} directions constrained +1")

# STEP 4c - THE ENSEMBLE: average the two members' probability RANKS
# (rank space, so neither member's calibration dominates). The arithmetic
# for the highest-scoring test day, in the open:
_p_lg = pd.Series(_lg_w.predict_proba(_te[mld.DESK_ML_BANK])[:, 1],
                  index=_te.index)
_p_gb = pd.Series(_gb_w.predict_proba(_te[mld.DESK_ML_BANK])[:, 1],
                  index=_te.index)
_rk_lg, _rk_gb = _p_lg.rank(pct=True), _p_gb.rank(pct=True)
_ens_sc = (_rk_lg + _rk_gb) / 2
_i = _ens_sc.idxmax()
print(f"STEP 4c  e.g. {_te.loc[_i, 'name']} on "
      f"{pd.Timestamp(_te.loc[_i, 'date']).date()}:  "
      f"logit p={_p_lg[_i]:.3f} (rank {_rk_lg[_i]:.3f}) | "
      f"GBM p={_p_gb[_i]:.3f} (rank {_rk_gb[_i]:.3f}) | "
      f"ensemble = ({_rk_lg[_i]:.3f}+{_rk_gb[_i]:.3f})/2 "
      f"= {_ens_sc[_i]:.3f}")

# %%
# STEP 5 - THE OPERATING POINT: rescore the TRAIN years with the fitted
# ensemble and sweep the percentile grid of candidate cuts; at each cut,
# count captured episodes, false alarms -> precision, recall, F1 (all
# EPISODE-level, the unit the desk cares about). The chosen cut is the
# F1 argmax - a LEARNED output. No FA budget, no penalty constant.
_ens_fit = mld.make_ens_fit("y_top")
_tr_scored = _tr.assign(score=_ens_fit(_tr, _tr, mld.DESK_ML_BANK))
_grid_w = np.unique(np.percentile(_tr_scored["score"].dropna(),
                                  np.arange(50, 100, 2.5)))
_groups_w = _pregroup(_tr_scored, episodes)
_in_years = (lambda ys: (lambda e: e.year.isin(ys)))(
    sorted(_tr_scored.year.unique()))
_pts = []
for _thr in _grid_w:
    _r = _tally(_groups_w, episodes, float(_thr), "top", _in_years)
    _npos = _r["captured"] + _r["false_alarms"]
    if not _npos or not _r["detectable"]:
        continue
    _prec = _r["captured"] / _npos
    _rec = _r["captured"] / _r["detectable"]
    _pts.append({"cut": _thr, "precision": _prec, "recall": _rec,
                 "F1": 2 * _prec * _rec / (_prec + _rec)
                 if (_prec + _rec) else 0.0})
_pts = pd.DataFrame(_pts)
# same tie-break as production: equal F1 -> the more conservative cut
_star = _pts.sort_values(["F1", "cut"]).iloc[-1]
_prod_cut = mld.choose_threshold_f1(_tr_scored, episodes, "top", 0.0,
                                    _tr["name"].nunique())
assert abs(_star.cut - _prod_cut) < 1e-9, "walkthrough != production"

_fig, _ax = plt.subplots(figsize=(8.2, 3.6))
_ax.plot(_pts.cut, _pts.precision, color=C2, lw=1.6, label="precision")
_ax.plot(_pts.cut, _pts.recall, color=C4, lw=1.6, label="recall")
_ax.plot(_pts.cut, _pts.F1, color=C1, lw=2.2, label="F1")
_ax.axvline(_star.cut, color=INK, lw=1.2, ls="--")
_ax.annotate(f"the chosen cut: {_star.cut:.3f}\n(train-year F1 max "
             f"= {_star.F1:.2f})", (_star.cut, _star.F1),
             xytext=(-130, 18), textcoords="offset points", fontsize=8.5)
_ax.set_xlabel("candidate probability cut (train-score percentile grid)")
_ax.set_title(f"STEP 5 — the cut is learned: episode-level precision / "
              f"recall / F1 on the train years (< {_ymax})")
_ax.legend(frameon=False, fontsize=8.5)
despine(_ax)
_fig.tight_layout()
_p23 = ROOT / "docs" / "figures" / "03" / "F23_threshold_learning.png"
_p23d = ROOT / "docs" / "figures" / "deck" / "F23_threshold_learning.png"
for _p in (_p23, _p23d):
    _p.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_p, dpi=180)
    print(f"saved {_p.relative_to(ROOT)}")
plt.show()
print(f"STEP 5  chosen cut {_star.cut:.3f} == production chooser's "
      f"{_prod_cut:.3f} (asserted identical). Then: cross the cut -> "
      "fire; one alert per name per 21d.")

# %% [markdown]
# **What the walkthrough just showed** — the whole model is five
# auditable steps: a candidacy anyone can count (STEP 1), features anyone
# can read off a table (STEP 2), a train/test split anyone can verify by
# year (STEP 3), two members whose fitted forms are printed in full — a
# 12-term formula and a direction-constrained tree stack — combined by an
# average of ranks shown digit by digit (STEP 4), and a cut selected by
# maximising a printed curve, asserted equal to what production computes
# (STEP 5). Nothing is tuned by hand; re-run the pipeline and every
# number re-derives.

# %%
# what the fitted model looks at - two independent reads on the FINAL
# fit (trained on all full years before the newest, i.e. the live model)
_insight = {}
for _label, _head in (("y_top", "GET OUT"), ("y_onset", "GET IN")):
    _lg = LogisticRegression(class_weight="balanced", max_iter=2000)
    _lg.fit(_tr[mld.DESK_ML_BANK], _tr[_label])
    _gb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, max_iter=200,
        monotonic_cst=[1] * len(mld.DESK_ML_BANK), random_state=0)
    _gb.fit(_tr[mld.DESK_ML_BANK], _tr[_label],
            sample_weight=_balanced_weights(_tr[_label].values))
    _pi = permutation_importance(_gb, _te[mld.DESK_ML_BANK], _te[_label],
                                 n_repeats=5, random_state=0,
                                 scoring="average_precision")
    _insight[_head] = pd.DataFrame({
        "feature": [mld.ML_BANK_LABELS.get(f, f)
                    for f in mld.DESK_ML_BANK],
        "logit weight": np.round(_lg.coef_[0], 2),
        "GBM permutation importance": np.round(_pi.importances_mean, 4),
    }).sort_values("GBM permutation importance", ascending=False)

for _head, _df in _insight.items():
    print(f"--- {_head}: what the live model leans on")
    print(_df.to_string(index=False))

# %% [markdown]
# **SO WHAT**
#
# * The two reads agree on the headline: the **price run-up** and the
#   **attention block** (level + the arrival speeds) carry the most
#   weight, with the sentiment block confirming rather than leading —
#   the same structure the hand rules encoded, now LEARNED with the cuts
#   set by data instead of by hand.
# * Logit weights are all interpretable in sign (positive = more of this,
#   more risk), and the GBM cannot disagree in direction by construction
#   (the monotone constraint). That is the explainability contract: a
#   300-tree model whose global behaviour is as directional as a rule.
#
# **IF ASKED — "why F1 and not the old FA budget?"** The desk removed the
# budget ("no desk-given requirement; more false alarms is ok") and asked
# for accuracy. F1 is the standard, single-sentence balance of precision
# and recall; it introduces no constant, and the resulting FA rates
# (0.35 / 0.24 per instrument-year) landed near the old budget anyway —
# accuracy alone got there.
#
# **IF ASKED — "would a deeper model do better?"** The MLP row in §SS is
# the answer, and it is a null: with ~500 positive episodes there is not
# enough label supply for depth to earn its opacity.

# %% [markdown]
# ## SS2.1 — The architecture, on one slide

# %%
# the how-a-call-is-made diagram (desk request 2026-08-07) - drawn from
# the shipped construction, saved for the deck AND shown on the
# dashboard's "what drives the calls" expander
import matplotlib.patches as _mp
from matplotlib.path import Path as _MplPath

_fig, _ax = plt.subplots(figsize=(11, 6.4))
_ax.set_xlim(0, 100); _ax.set_ylim(0, 100); _ax.axis("off")
_NAVY, _TEAL, _GREEN, _RED, _GRAY = ("#0A1E2E", "#2E6E7E", "#1F6F5C",
                                     "#A6413B", "#717171")


def _box(x, y, w, h, text, fc, tc="white", fs=9, weight="bold"):
    _ax.add_patch(_mp.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        facecolor=fc, edgecolor="none"))
    _ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             color=tc, fontsize=fs, fontweight=weight, linespacing=1.5)


def _arrow(x0, y0, x1, y1, c=_GRAY):
    _ax.annotate("", (x1, y1), (x0, y0),
                 arrowprops=dict(arrowstyle="-|>", lw=1.4, color=c))


# inputs
_box(4, 62, 28, 30,
     "THE CROWD  (9 measurements)\n"
     "$\\bf{attention}$: level · 1-mo change\n"
     "2-wk change · week vs month\n"
     "vs own normal · accelerating\n"
     "$\\bf{mood}$: level · persistence · turning\n"
     "(each a percentile vs the name's\nOWN history, trailing only)",
     _NAVY, fs=8)
_box(4, 40, 28, 16,
     "THE TAPE  (2 measurements)\n"
     "run-up off the 54d low\n1-month return",
     _TEAL, fs=8)
# models
_box(44, 68, 22, 12, "LOGISTIC\nREGRESSION\none weight per measurement",
     _GRAY, fs=8)
_box(44, 46, 22, 12,
     "MONOTONE GBM\n300 shallow trees; more heat\ncan only mean more risk",
     _GRAY, fs=8)
_arrow(32, 74, 44, 74); _arrow(32, 50, 44, 52)
_arrow(32, 70, 44, 54); _arrow(32, 54, 44, 72)
# ensemble
_box(74, 57, 22, 12, "RANK ENSEMBLE\naverage of the two\nmodels' rankings",
     _GREEN, fs=8.5)
_arrow(66, 74, 74, 65); _arrow(66, 52, 74, 61)
# cut + call
_box(74, 36, 22, 12,
     "FROZEN CUT (~0.92)\nchosen on PAST years only,\nby F1 - then one "
     "call per\nname per 21 days", "#3d3d3d", fs=8)
_arrow(85, 57, 85, 48)
_box(60, 12, 17, 12, "GET IN\neuphoria starting", _GREEN, fs=9)
_box(83, 12, 17, 12, "GET OUT\neuphoria ending", _RED, fs=9)
_arrow(82, 36, 70, 24); _arrow(88, 36, 91, 24)
_ax.text(2, 6, "Two heads, one architecture: the same 11 measurements and the same construction are fitted separately "
         "against 'a start is near' and 'a top is near'.\nEverything is walk-forward: the models and the cut that "
         "fire in year Y have seen only years before Y.",
         fontsize=8, color=_GRAY, va="top")
_ax.set_title("How a call is made", fontsize=13, fontweight="bold",
              loc="left")
for _p in (ROOT / "docs" / "figures" / "03" / "F21_model_architecture.png",
           ROOT / "docs" / "figures" / "deck" / "F21_model_architecture.png"):
    _p.parent.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_p, dpi=180, bbox_inches="tight")
    print(f"saved {_p.relative_to(ROOT)}")
plt.show()
