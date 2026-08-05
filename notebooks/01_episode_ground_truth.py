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
# # Notebook 01 — What are we even trying to catch?
#
# **Study:** detecting the START and the END of retail euphoria episodes from
# the crowd alone (RetailRadar, July 2026).
#
# This notebook writes the exam paper. It builds the **episode catalog** — the
# price-defined list of what actually happened — *before* a single feature or
# model is looked at. Every hit rate quoted anywhere else in this project is
# scored against this file.
#
# > **The one question this notebook answers:** what counts as a euphoria
# > episode, how many of them are there, and on how many of them could a
# > crowd-only detector *fairly* be graded?
#
# The project question it serves, for context:
#
# > Can crowd-only features (mentions + scored sentiment; **price is never an
# > input to the signal**) detect both the START and the END of extreme retail
# > euphoria episodes — GME-scale rallies only — sparingly, with hit rate
# > maximised across all judgeable periods?
#
# ---
#
# ## How to read this notebook
#
# * **The verdict box below is the whole answer.** If you read nothing else,
#   read that.
# * **Every section is a question**, and each carries the same three blocks:
#   **WHY THIS** (what we did not know), **HOW IT WORKS** (the mechanism, and
#   where every number came from), **SO WHAT** (the finding, and what changes).
# * **`IF ASKED` blocks** are the awkward questions a reviewer would put to
#   this work, answered. They are marked so you can find them.
# * **No unexplained constants.** Every number is tagged LEARNED (from data),
#   DERIVED (forced by a definition), CONVENTION (inherited and tested),
#   GROUND TRUTH, or DESK DECISION (a judgement, with its reason).
# * Section 1 defines every term, so no jargon is load-bearing.
#
# ---
#
# ## VERDICT BOX — read this first, then read why
#
# | Question | Answer | Evidence |
# |---|---|---|
# | How many euphoria episodes are there? | **333** boom→bust arcs across 59 instruments, 2017–2026. | §3 |
# | How many can a crowd detector fairly be graded on? | **126** for the start, **136** for the top. The rest happened while the post archive was too thin to judge. | §3 |
# | Is the archive evenly spread? | **No, and this is the single biggest limitation in the project.** 2021 alone supplies 52 of the 126 gradeable starts; 2024 and 2025 supply **zero**. | §3, §7 |
# | How big is a typical episode? | Median boom **+64%**, median bust **−36%**, median run **86 days** low→top. | §4 |
# | Is the 45-day onset window a fair one? | **Yes, and it is not a new number** — it is the false-alarm horizon the scorer already used. It covers the first **52%** of a median run-up. | §4 |
# | Does anything here look at the crowd? | **No. Nothing.** The catalog is built from prices only, which is what makes it usable as an exam. | §2 |
#
# **What ships:** `data/processed/episodes.parquet` — the single source of
# truth for notebooks 02–07 and for every hit rate on the dashboard.
#
# ---
#
# ## The rule sheet — decided BEFORE any result was seen
#
# This is the "no arbitrary numbers" register for this notebook. Every row was
# fixed before the catalog was built, so none of them could have been tuned to
# flatter a result.
#
# | Quantity | Value | Class | Where it comes from |
# |---|---|---|---|
# | G1 local max | highest close in a ±21d window | CONVENTION | Inherited unchanged from the validated euphoria detector (`analytics/euphoria.py`). |
# | G2 boom | peak ≥ **25%** (theme ETF) / **50%** (single name) above the minimum close of the prior 120d | CONVENTION | Same source. Single names carry the harder bar because they move further on less. |
# | G3 bust | drawdown ≥ **15%** / **30%** within the next 90d | CONVENTION | Same source. The bust requirement is what makes an episode *worth having warned about*. |
# | trough | the argmin of the **same** 120d window G2 already measures the boom from | DERIVED | Re-using G2's window means no new lookback is introduced, so no new number can be tuned. |
# | bust date | the first day G3's drawdown condition is met | DERIVED | Forced by G3; nothing is chosen. |
# | Onset hit window | `[trough, trough+45d]`, **capped at the peak** | DESK DECISION | 45d mirrors the false-alarm horizon already in `score_alerts` (an alert is false if no peak follows within 45d), so it is not a new constant. The cap exists because on a fast rally an uncapped window would let an alert fired *after* the top count as "caught the start". |
# | LATE ≠ FALSE | alerts in `(onset end, peak]` are reported in their own bucket | DESK DECISION | Calling a mid-rally alert a *hit* inflates the onset claim; calling it *false* punishes an alert fired inside a genuine episode. A third bucket avoids both distortions. |
# | Top hit window | `[peak−30d, peak+1d]` | CONVENTION | The stated aim of the existing detector — "a few weeks before, or one day after" — unchanged. |
# | Coverage gate | ≥100 scored posts in 28d, on at least one day of the window | CONVENTION | The existing A0 rule. It decides *detectability*, never *truth*. |

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Commented, not deleted: inline IS the default backend under Jupyter, and
# leaving the magic in place makes the file unrunnable as a plain script -
# which is exactly how it gets verified headless (MPLBACKEND=Agg python ...).
# %matplotlib inline

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

# House chart rules (they were shared with helper/research_charts.py, which
# is not in this repo - audited 2026-08-05): one axis per
# panel, fixed validated categorical palette, direct labels, recessive grid.
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
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1 · The words this notebook uses
#
# **WHY THIS**
#
# * Five terms below carry the entire argument. If a reader guesses at what
#   *detectable* means, they will read the headline count wrong by a factor of
#   two and a half.
# * The definitions come from `analytics/plain_english.py`, the same module the
#   dashboard reads its tooltips from — so a term cannot mean one thing here
#   and another thing on the screen the PM is looking at.

# %%
from analytics.plain_english import glossary_md                   # noqa: E402

print(glossary_md([
    "episode",
    "boom and bust thresholds",
    "the onset window",
    "detectable",
    "lead time",
]))

# %% [markdown]
# ## 2 · How is an episode defined — and why does price define it?
#
# **WHY THIS**
#
# * Before anything can be scored, something has to be *right*. This section
#   fixes what "right" means, and fixes it in a way the thing being graded
#   cannot influence.
# * The decision that hangs on it: whether any hit rate in this project means
#   anything at all. A detector graded against labels its own features helped
#   draw is not being tested, it is being asked to agree with itself.
#
# **HOW IT WORKS**
#
# * An **episode** is the full boom→bust arc around a *confirmed top*. The top
#   test is inherited unchanged from the validated detector
#   (`analytics/euphoria.py`, rules G1–G3), all three tagged CONVENTION in the
#   rule sheet above:
#     * **G1** the highest close in a ±21d window;
#     * **G2** that peak sits ≥25% (theme) / ≥50% (single) above the lowest
#       close of the prior 120d;
#     * **G3** a fall of ≥15% / ≥30% follows within 90d.
# * The arc is then extended using **only quantities G1–G3 already measure** —
#   the trough is the argmin of G2's own 120d window, the bust date is the
#   first day G3's condition is met. Both DERIVED. This is deliberate: a new
#   lookback would be a new number, and a new number is a new thing to tune.
# * **Price is the only input.** No mention count, no sentiment score, and no
#   model output touches the catalog.
# * The universe is the detector's own: 34 theme anchor ETFs (rates/bonds and
#   real estate excluded — DESK DECISION, they are macro instruments the retail
#   crowd does not trade in size) plus the top-25 single names chosen by data
#   (≥3,000 scored posts, and priced). 59 instruments.
# * The builder lives in `analytics/euphoria_phases.py` — **the same module the
#   production pipeline imports**, so the research record and the dashboard
#   cannot drift apart.

# %%
from analytics.euphoria import build_all_series                   # noqa: E402
from analytics.euphoria_phases import (                           # noqa: E402
    episode_catalog, ONSET_WINDOW_DAYS)

t0 = time.time()
prices = pd.read_parquet(ROOT / "data" / "prices" / "prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
series, pxmap = build_all_series(prices)
catalog = episode_catalog(series, pxmap)
print(f"built in {time.time()-t0:.1f}s | universe: {len(series)} instruments "
      f"({sum(1 for s in series if s.kind=='theme')} themes, "
      f"{sum(1 for s in series if s.kind=='single')} single names)")
print(f"episodes: {len(catalog)} | onset-detectable: "
      f"{int(catalog.onset_detectable.sum())} | top-detectable: "
      f"{int(catalog.top_detectable.sum())}")

# %% [markdown]
# **SO WHAT**
#
# * **333 episodes** across 59 instruments and ten years, of which **126** are
#   gradeable on the start and **136** on the top.
# * Nothing is adopted or rejected here — this section produces the ruler, not
#   a reading. What it buys is the right to quote every later number as
#   out-of-sample against labels no feature helped draw.
#
# ---
#
# ## 3 · How many episodes can we *fairly* be graded on?
#
# **WHY THIS**
#
# * 333 arcs exist in the price data. The crowd archive does not cover all of
#   them evenly, and a detector cannot be marked wrong on a stretch where it
#   had nothing to read.
# * The decision that hangs on it: which denominator every hit rate in
#   notebooks 04, 06 and 07 is quoted against. Get this wrong in the generous
#   direction and the project overstates itself by 2.6×.
#
# **HOW IT WORKS**
#
# * `onset_detectable` / `top_detectable` ask whether the **coverage gate** —
#   ≥100 scored posts in 28 days, the existing A0 rule, CONVENTION — held on at
#   least one day of the respective window.
# * An episode that failed the gate **stays in the catalog** and stays in the
#   *all* column. It is real; it happened. It is simply one the detector was
#   **blind** to rather than **wrong** about, and merging those two states is
#   the most common way a backtest flatters itself.
# * So every rate in this study is reported against **both** bars, always. The
#   grey bar is what happened; the blue bar is what could be judged.

# %%
per_year = (catalog.groupby("year")
            .agg(episodes=("name", "count"),
                 onset_detectable=("onset_detectable", "sum"),
                 top_detectable=("top_detectable", "sum"),
                 themes=("kind", lambda k: int((k == "theme").sum())),
                 singles=("kind", lambda k: int((k == "single").sum()))))
per_year

# %%
fig, ax = plt.subplots(figsize=(9, 3.4))
x = np.arange(len(per_year))
ax.bar(x - 0.22, per_year["episodes"], 0.42, color=GRID,
       label="all episodes (price truth)")
ax.bar(x + 0.22, per_year["onset_detectable"], 0.42, color=C1,
       label="onset-detectable (coverage held)")
for i, (a, b) in enumerate(zip(per_year["episodes"],
                               per_year["onset_detectable"])):
    ax.text(i - 0.22, a + 0.8, str(a), ha="center", fontsize=8, color=MUTED)
    ax.text(i + 0.22, b + 0.8, str(b), ha="center", fontsize=8, color=C1)
ax.set_xticks(x, per_year.index)
ax.set_title("Euphoria episodes per year — every rate in this study is "
             "reported against BOTH bars")
ax.legend(frameon=False, loc="upper right")
despine(ax)
plt.show()

# %% [markdown]
# **READ THE CHART LIKE THIS** — grey is what the market did, blue is what the
# archive lets us grade. Where the two bars are far apart, the detector was
# blind, not wrong. 2024 and 2025 have **no blue bar at all**.
#
# **SO WHAT**
#
# * **126 of 333 starts are gradeable — 38%.** Every onset hit rate in this
#   project is a rate out of 126, and it is labelled as such wherever it
#   appears.
# * **The sample is concentrated in one year.** 2021 alone supplies 52 of the
#   126 gradeable starts — 41% of the evidence comes from the meme-stock year.
#   That is the most important fact in this notebook and it is why notebooks 06
#   and 07 quote confidence intervals rather than point estimates.
# * **2024–25 are a coverage desert**, zero gradeable episodes in either year.
#   Nothing can be claimed about them in either direction.
# * What changes because of it: the walk-forward design downstream is *forced*.
#   With the evidence this unevenly spread, a single train/test split would be
#   a statement about 2021 wearing a lab coat.
#
# ---
#
# ## 4 · How big is a typical episode, and is a 45-day window fair?
#
# **WHY THIS**
#
# * The onset window is the one genuinely discretionary number in this
#   notebook. If 45 days turns out to cover 5% of a typical run-up it is too
#   strict to be meaningful; if it covers 95% it is so loose that "caught the
#   start" means "noticed at some point".
# * We also do not yet know the *stakes*: whether the busts these episodes end
#   in are large enough to be worth warning a PM about at all.
#
# **HOW IT WORKS**
#
# * **Boom size** on a log10 scale, because GME 2021 at roughly +3,700% would
#   otherwise flatten every other episode into the first bin. Log is a
#   readability choice, not a transform applied to any calculation.
# * **Bust depth** is the fall inside 90 days of the peak — what the party
#   costs, and the reason calling tops is worth doing.
# * **Run length** is trough→peak in days. The dashed line is the 45-day onset
#   window, so the fraction of the histogram to its right is the fraction of
#   run-ups where 45 days is a genuinely *early* call.
# * The window itself is **not a new constant** — 45d is the false-alarm
#   horizon `score_alerts` already used (DESK DECISION, recorded above). The
#   test here is whether that inherited number is defensible, not what it
#   should be.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].hist(np.log10(1 + catalog.boom_pct), bins=30, color=C1)
axes[0].set_title("Boom size, log10(1 + gain)")
axes[0].set_xlabel("0.1≈+26%   0.5≈+216%   1.5≈+3,000%")
axes[1].hist(100 * catalog.bust_pct, bins=30, color=C3)
axes[1].set_title("Bust depth within 90d (%)")
axes[2].hist(catalog.run_days, bins=24, color=C2)
axes[2].axvline(ONSET_WINDOW_DAYS, color=INK, lw=1.2, ls="--")
axes[2].text(ONSET_WINDOW_DAYS + 2, axes[2].get_ylim()[1] * 0.92,
             f"onset window = {ONSET_WINDOW_DAYS}d", fontsize=8, color=INK)
axes[2].set_title("Run length trough→peak (days)\n(≤120 by construction)")
for ax in axes:
    despine(ax)
fig.tight_layout()
plt.show()

med = catalog.run_days.median()
print(f"median run {med:.0f}d -> the 45d onset window covers the first "
      f"{100*ONSET_WINDOW_DAYS/med:.0f}% of a median run-up")

# %% [markdown]
# **READ THE CHART LIKE THIS** — left: almost every episode is a genuine
# multiple, not a 26% wobble. Middle: the mass sits well past −20%, so these
# are falls a desk would have wanted warning of. Right: the dashed line is the
# onset window, and most of the distribution lies to its right, which is what
# makes "caught the start" a meaningful claim rather than a free one.
#
# **SO WHAT**
#
# * **Median boom +64%, median bust −36%, median run 86 days.** These are not
#   marginal moves; the bust distribution is the entire commercial case for
#   calling tops.
# * **The 45-day window covers the first 52% of a median run-up** — roughly the
#   first half. Strict enough that an alert inside it is genuinely early; loose
#   enough that a real detector can land in it. The inherited number survives.
# * Nothing is adopted or changed here. What this section buys is the right to
#   quote the onset window without a reviewer being able to ask "why 45?".
#
# **IF ASKED — "run lengths cap out at 120 days. Isn't that suspicious?"**
#
# It is not suspicious, it is arithmetic: the trough is defined as the argmin
# of G2's 120-day lookback, so `run_days ≤ 120` **by construction**. The honest
# reading of the third panel is therefore "the length of the final measured
# run-up", not "the age of the rally". For a slow multi-quarter theme the
# narrative start is earlier than our trough, and any onset claim we make about
# such a name is correspondingly *later* than a human would call it — that is a
# conservative error, and it is carried into the limitations in §7 rather than
# corrected by inventing a longer lookback nobody has validated.
#
# ---
#
# ## 5 · What does an episode actually look like?
#
# **WHY THIS**
#
# * Everything above is a distribution. Distributions hide whether the labels
#   are placed sensibly on a real chart, and a mis-placed label would corrupt
#   every downstream result silently.
# * These three panels are also the canvas every later notebook draws its
#   alerts onto, so it is worth knowing what the blank canvas looks like.
#
# **HOW IT WORKS**
#
# * Shaded **blue = the onset window** `[trough, min(trough+45d, peak)]`;
#   shaded **pink = peak→bust**. Dots mark the trough and the peak.
# * Two panels are named in the project aim (GME 2021, gold 2026) — chosen
#   because they are the cases the desk asked about, and stated as such.
# * The **third panel is chosen by the data**, not by hand: the largest
#   onset-detectable theme boom excluding gold. Hand-picking all three would
#   make this a gallery of successes; letting the data pick one keeps it a
#   sample. DESK DECISION.
# * Log price scale where the boom demands it (GME), linear elsewhere. A
#   display choice; no number is transformed.

# %%
import matplotlib.dates as mdates                                 # noqa: E402

def case_panel(ax, name, symbol, trough, peak, onset_hi, bust, log=False):
    px = pxmap[symbol].dropna()
    lo = trough - pd.Timedelta(days=90)
    hi = peak + pd.Timedelta(days=120)
    win = px.loc[lo:hi]
    ax.plot(win.index, win.values, color=INK, lw=1.4)
    ax.axvspan(trough, onset_hi, color=C1, alpha=0.18, lw=0)
    if bust is not None and not pd.isna(bust):
        ax.axvspan(peak, bust, color=C3, alpha=0.25, lw=0)
    ax.plot([trough], [px.loc[trough]], "o", color=C1, ms=7)
    ax.plot([peak], [px.loc[peak]], "o", color=C3, ms=7)
    if log:
        ax.set_yscale("log")
    ax.set_title(f"{name} ({symbol})\ntrough {trough.date()} → "
                 f"peak {peak.date()}", fontsize=10)
    loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    despine(ax)

# GME 2021 + gold 2026 are named in the project aim; the third panel is the
# largest onset-detectable THEME boom, chosen by the data, not by hand.
gme = catalog[(catalog.name == "GME") & (catalog.year == 2021)].iloc[0]
gld = catalog[(catalog.name == "gold_metals") & (catalog.year == 2026)].iloc[0]
theme_pool = catalog[(catalog.kind == "theme") & catalog.onset_detectable
                     & ~catalog.name.isin(["gold_metals"])]
third = theme_pool.sort_values("boom_pct", ascending=False).iloc[0]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
case_panel(axes[0], "GME", "GME", gme.trough, gme.peak, gme.onset_hi,
           gme.bust_date, log=True)
case_panel(axes[1], "gold_metals", "GLD", gld.trough, gld.peak, gld.onset_hi,
           gld.bust_date)
case_panel(axes[2], third["name"], third.symbol, third.trough, third.peak,
           third.onset_hi, third.bust_date)
fig.tight_layout()
plt.show()
print(f"third panel (data-chosen): {third['name']} {third.peak.date()}, "
      f"boom +{100*third.boom_pct:.0f}%, bust {100*third.bust_pct:.0f}%")

# %% [markdown]
# **READ THE CHART LIKE THIS** — check that the blue band starts at the bottom
# of the run and the pink band covers the fall. If the labels look right here,
# they are right everywhere, because all 333 were built by the same code path.
#
# **SO WHAT**
#
# * The labels land where a human would put them: blue at the base of the
#   run-up, pink over the collapse. The catalog passes an eyeball test as well
#   as an arithmetic one.
# * GME 2021 shows the **overlapping-arc** case honestly — consecutive peaks
#   30–120 days apart produce arcs that overlap. Carried forward as limitation
#   4 in §7 rather than smoothed over.
# * Nothing changes as a result of this section. It is a sanity check, and it
#   passed.
#
# ---
#
# ## 6 · What ships
#
# **WHY THIS**
#
# * One file has to be the single source of truth, or notebooks 02–07 will
#   quietly diverge on what "an episode" was.
#
# **HOW IT WORKS**
#
# * `data/processed/episodes.parquet` — the catalog itself, read by notebooks
#   02, 03, 04, 06 and 07 and by the dashboard's hit-rate panel.
# * `docs/research/nb01_episode_stats.json` — the headline counts, so a doc or
#   a chart elsewhere can quote them without re-running this notebook and
#   without anyone re-typing a number by hand.

# %%
out_path = ROOT / "data" / "processed" / "episodes.parquet"
catalog.to_parquet(out_path, index=False)

stats = {
    "episodes_total": int(len(catalog)),
    "onset_detectable": int(catalog.onset_detectable.sum()),
    "top_detectable": int(catalog.top_detectable.sum()),
    "median_run_days": float(catalog.run_days.median()),
    "median_boom_pct": float(catalog.boom_pct.median()),
    "median_bust_pct": float(catalog.bust_pct.median()),
    "onset_window_days": int(ONSET_WINDOW_DAYS),
    "per_year": {str(y): {"episodes": int(r.episodes),
                          "onset_detectable": int(r.onset_detectable),
                          "top_detectable": int(r.top_detectable)}
                 for y, r in per_year.iterrows()},
}
with open(RESEARCH_DIR / "nb01_episode_stats.json", "w") as f:
    json.dump(stats, f, indent=1)
print(f"saved {out_path.name} ({len(catalog)} episodes) "
      f"+ nb01_episode_stats.json")

# %% [markdown]
# ## 7 · What this catalog cannot do
#
# **WHY THIS**
#
# * A ground truth with unstated limits is worse than no ground truth, because
#   every number downstream inherits them silently. These five are stated here
#   once and referred back to wherever they bite.
#
# 1. **Coverage deserts.** 2024–25 have zero detectable episodes. Every
#    downstream rate is reported against both denominators; the detector is
#    *blind* there, not *validated* there — and 41% of the gradeable evidence
#    comes from 2021 alone.
# 2. **Trough truncation.** `run_days ≤ 120` by construction. "Onset" means
#    "the start of the final measured run-up", which for a slow multi-quarter
#    rally is later than the narrative start. The error is conservative: it
#    makes our early calls look *less* early than a human would judge them.
# 3. **Price-defined truth.** An episode requires a *confirmed* bust. Euphoria
#    that deflated slowly — no ≥15/30% fall inside 90 days — never enters the
#    catalog at all. So the detectors are graded only on arcs a desk would
#    actually have wanted flagged, which is the right exam, but it means we
#    measure nothing about gentle unwinds.
# 4. **Overlapping episodes.** Peaks closer than 30d are collapsed upstream;
#    distinct peaks 30–120d apart can still produce overlapping arcs (visible
#    in the GME panel). Labels treat each arc independently; the 21d alert
#    cooldown is what prevents double-alerting in practice.
# 5. **V-recoveries qualify.** "A boom off a 120d low" admits crash-rebound
#    rallies — the data-chosen third case study is energy's post-COVID
#    V-recovery. The catalog's semantics are "extreme rally that then broke",
#    which is broader than "mania".
#
# **IF ASKED — "point 5 means your labels aren't really euphoria labels."**
#
# Correct, and deliberately so. G2 is inherited from the validated detector and
# we did not add a hand-written "but not if it was a recovery" clause, because
# that clause would be exactly the kind of arbitrary rule this project refuses
# — it would encode our opinion of which rallies are manias into the answer
# sheet. Instead the distinction is left for the **crowd features to earn** in
# the tournament: if new-entrant rate and near-vertical attention separate
# manias from recoveries, notebooks 02–03 will measure them doing it. Assuming
# the distinction gives you a better-looking hit rate; making the features earn
# it gives you a result you can defend.
#
# **IF ASKED — "with 41% of the evidence from 2021, is any of this general?"**
#
# Not as a single number, no — and the project does not present it as one. That
# concentration is why every downstream evaluation is **walk-forward** rather
# than a single split, why hit rates carry confidence intervals, and why
# notebook 07 reports performance by year rather than only in aggregate. The
# honest claim this catalog supports is "measured across 126 gradeable starts,
# heavily weighted to 2021, with intervals wide enough to show it" — not "works
# in all regimes". Stating the weakness in the ground truth is what stops it
# being discovered later in the results.
#
# ---
#
# **Next (Notebook 02):** the six-feature onset bank vs the incumbent E1–E5,
# put through the importance battery — per-feature AUROC/AP,
# leave-one-out ablation, perturbation robustness, and the correlation
# matrix — all scored against these labels.
