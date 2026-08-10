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
# # Notebook 07 — The Index Composite: what was tried at index level, and why none of it ships
#
# > **The one question this notebook answers:** can the per-name GET IN /
# > GET OUT machinery say anything honest about the S&P 500 or a factor
# > basket — and the answer, twice arrived at and twice acted on, is no.
#
# **RETIRED BY DESK DECISION, 2026-08-05.** The index and factor work was
# built, measured, and removed the same week. This notebook is kept as the
# record of what was tried, because the question will be asked again and
# the measurements are the reason it should not be re-attempted the same
# way. Nothing described below is live: there is no `sp500` theme, no
# `momentum_factor` / `growth_factor`, no basket-breadth module (`analytics/basket_breadth.py` was removed),
# no index-scale ground-truth constants and no Index & factors tab.
#
# ---
#
# ## WHAT WAS TRIED, AND WHAT KILLED IT
#
# | Attempt | Measurement | Outcome |
# |---|---|---|
# | Count crowd mentions of the FUND (MTUM, VTV, IVW, IVE) | Over 176,126 archived comments: MTUM **0**, VTV **0**, IVW **0**, and the four apparent "IVE" hits are people typing "I've" against 82 lowercase. Retail does not discuss factors as factors. | Dead on arrival — and it is the reason every later attempt went through constituents instead. |
# | Read a basket through its CONSTITUENTS | Coverage was genuinely good: MTUM's 25 holdings all discussed, 8 above 100 mentions/yr, 5.0% of all chatter; VUG 19 of 25 strong, 18.2%; a 269-name S&P union 22.5%. | The read worked. It measured mega-cap tech under three different names — momentum overlapped `memory` 12/25 and `semiconductors` 10/25 — so it added exposure, not information. |
# | Give the S&P its own GET IN / GET OUT | At the frozen theme bars (boom 25% / crash 15%) `ground_truth_peaks` returns **0 episodes for SPY — and 0 for XLI**, the example the desk cited as working. Volatility scaling does not rescue it: SPY's 18.3% vol against the 25.8% median theme anchor implies 17.8% / 10.7%, which also returns **0**. An index's post-run-up drawdowns are shallower AND slower, so the rule misses them by shape, not size. | Only 12% / 8% produced usable ground truth (SPY 6). At those bars a "top" is a 12% run-up and an 8% fall — an ordinary market wiggle. The words stop meaning what they mean everywhere else on the dashboard. **Not shipped.** |
# | Market-wide flag counts over the momentum price | Built and rendered; MTUM was never priced, so it drew against QQQ — a Nasdaq-100 concentration bet, not a momentum screen. | Descriptive only, never judged against forward returns. **Not shipped.** |
#
# ## The one durable conclusion
#
# The detector earns its keep on **narratives the crowd argues about**,
# where attention concentrates and then breaks. An index is the opposite
# object by construction: diversification is precisely what stops it
# staging the run-up-then-bust arc the ground truth is built to find. That
# is not a threshold problem to be tuned around — lowering the bars until
# episodes appear buys ground truth made of milder events, and every
# capture rate in the record would move for reasons unrelated to the
# detector getting better.
#
# §1 below still contains the breadth composite as originally written. It
# is left intact as the least-bad honest mapping if anyone revisits this.

# %% [markdown]
# ## Definitions — every term in desk English
#
# (From `analytics/plain_english.py`, the shared glossary.)

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.dates as mdates

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
SEED = 42                      # the project's standard seed set is {42, 100, 2026}
NB_T0 = time.time()

from analytics.plain_english import glossary_md  # noqa: E402
print(glossary_md([
    "episode", "detectable", "walk-forward", "percentile rank",
    "boom and bust thresholds", "confidence interval",
    "crowd big enough to signal",
]))

# %% [markdown]
# Three terms this notebook adds:
#
# * **breadth** — the share of tracked names in a given state on a day
#   (in end-stage, in GET OUT candidacy, …). A cross-sectional average,
#   with the denominator = the names MEASURABLE that day (coverage gate
#   held), never the full roster.
# * **exit / entry readiness** — the mean over measurable names of
#   (production score ÷ its frozen trigger), taking 0 where the name is
#   not in candidacy. 1.0 would mean "the average tracked name is AT its
#   trigger". Scores and triggers are the production ones from
#   `euphoria_desk.parquet` / `euphoria_desk_report.json` — nothing is
#   re-derived.
# * **coverage floor** — breadth over a handful of names moves in giant
#   steps (1 name in 5 = 20 points), so composite series are masked on
#   days with fewer than 10 measurable names. CONVENTION: the
#   cross-sectional analogue of the A0 coverage gate ("percentile
#   extremes on a handful of posts are noise wearing a costume");
#   sensitivity at floors {5, 10, 15} is shown before anything is read.

# %% [markdown]
# ## Setup — the production stores, nothing re-derived

# %%
from analytics.euphoria import ground_truth_peaks, trailing_pct_rank  # noqa: E402
from analytics.euphoria_phases import find_episodes                    # noqa: E402
from src.config import (ROLL, EUPHORIA_ATT_GATE, EUPHORIA_PCT_WINDOW,
                        EUPHORIA_COOLDOWN_DAYS)                        # noqa: E402

t0 = time.time()
desk = pd.read_parquet(ROOT / "data" / "processed" /
                       "euphoria_desk.parquet")
desk["date"] = pd.to_datetime(desk["date"])
prices = pd.read_parquet(ROOT / "data" / "prices" / "prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
PRICED = set(prices["symbol"].unique())
pxb = {s: g.sort_values("date").set_index("date")["px_last"].dropna()
       for s, g in prices.groupby("symbol")
       if s in ("SPY", "QQQ", "MTUM")}
pxd = {s: p.asfreq("D").ffill() for s, p in pxb.items()}

desk_rep = json.load(open(ROOT / "data" / "processed" /
                          "euphoria_desk_report.json"))
THR_OUT = float(desk_rep["get_out"]["live_threshold"])
THR_IN = float(desk_rep["get_in"]["live_threshold"])
FA_BUDGET = float(desk_rep["fa_budget_per_iy"])

n_theme = desk[desk.kind == "theme"]["name"].nunique()
n_single = desk[desk.kind == "single"]["name"].nunique()
print(f"{time.time()-t0:.0f}s | {n_theme + n_single} tracked names "
      f"({n_theme} themes, {n_single} singles) | desk store "
      f"{desk['date'].min().date()} → {desk['date'].max().date()} | "
      f"frozen triggers out {THR_OUT:.3f} / in {THR_IN:.3f}")
print(f"MTUM priced: {'MTUM' in PRICED} | SPY priced: {'SPY' in PRICED} "
      f"| QQQ priced: {'QQQ' in PRICED}")

# %% [markdown]
# ---
# # §1 — THE COMPOSITE: the constituents' signals as a whole
#
# **WHY THIS**
#
# * The desk said *"use the constituent components and their individual
#   signals as a whole"*. Our constituents are NOT the S&P 500's: they
#   are the project's tracked universe — 31 retail-attention theme ETFs
#   plus 23 crowd-favourite single names. **That is a sentiment
#   cross-section of the market, not its cap-weighted membership**, and
#   the mapping is stated up front because it is the honest one AND the
#   right one: this project measures where retail attention concentrates,
#   and the index-level question it can answer is *"is the whole tracked
#   crowd euphoric at once?"* — which is a statement about the market's
#   speculative temperature, not about AAPL's index weight. Building a
#   fake cap-weighted composite from names chosen for Reddit chatter
#   would claim S&P coverage this data does not have.
#
# **HOW IT WORKS** — from `euphoria_desk.parquet` (production scores and
# flags, live through the newest pull), per day:
#
# * **share of names in end-stage** (`end_stage` — every GET OUT gate
#   already open: crowd 2×, attention ≥ 0.90, persistence);
# * **share with GET OUT gates open** = share in GET OUT candidacy
#   (`out_score` present: crowd swollen AND price boom underway) — and
#   the GET IN mirror (`in_score` present);
# * **exit / entry readiness** — mean of (score ÷ frozen trigger),
#   0 where absent (definition above);
# * **trailing-21d flag counts** (21d = the house cooldown window, so
#   one name can contribute at most one flag per window);
# * everything masked below the coverage floor, whose sensitivity is
#   shown first.

# %%
g = desk.groupby("date")
b = pd.DataFrame({
    "n_meas": g.size(),
    "end_share": g["end_stage"].mean(),
    "out_open": g["out_score"].apply(lambda x: x.notna().mean()),
    "in_open": g["in_score"].apply(lambda x: x.notna().mean()),
    "exit_ready": g["out_score"].apply(
        lambda x: (x / THR_OUT).fillna(0).mean()),
    "entry_ready": g["in_score"].apply(
        lambda x: (x / THR_IN).fillna(0).mean()),
    "out_flags": g["get_out"].sum(),
    "in_flags": g["get_in"].sum(),
}).asfreq("D")
b[["out_flags", "in_flags"]] = b[["out_flags", "in_flags"]].fillna(0)
b["n_meas"] = b["n_meas"].fillna(0)
b["out_21d"] = b["out_flags"].rolling(21, min_periods=1).sum()
b["in_21d"] = b["in_flags"].rolling(21, min_periods=1).sum()

# the coverage floor, with its sensitivity shown BEFORE it is used
sens = pd.DataFrame([
    {"floor (names)": fl,
     "days kept": int((b["n_meas"] >= fl).sum()),
     "share of days kept": round(float((b["n_meas"] >= fl).mean()), 2)}
    for fl in (5, 10, 15)])
display(sens)
FLOOR = 10
ok = (b["n_meas"] >= FLOOR).fillna(False)
keep_by_year = ok.groupby(ok.index.year).mean().round(2)
print(f"coverage floor {FLOOR}: days kept by year "
      f"{keep_by_year.to_dict()}")
print("LOUDLY: 2024 and 2025 are almost entirely BELOW the floor - the "
      "archive's known thin era (NB04's standing limitation). Every "
      "composite series is blank there because the crowd was "
      "unmeasurable, not calm.")

# %%
spy = pxd["SPY"]
fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.4, 1.4,
                                                        1.2, 0.8]})
ax = axes[0]
ax.plot(spy.index, spy.values, color=INK, lw=1.1)
ax.set_yscale("log")
ax.set_ylabel("SPY (log)")
ax.set_title("SPY against the tracked universe's breadth composite "
             "(grey bands = below the coverage floor: the crowd was "
             "unmeasurable, not calm)", fontsize=9)
ax = axes[1]
ax.plot(b.index, b["end_share"].where(ok) * 100, color=C3, lw=1.1,
        label="share of names in END-STAGE (all GET OUT gates open)")
ax.plot(b.index, b["out_open"].where(ok) * 100, color=C4, lw=1.1,
        label="share in GET OUT candidacy (crowd 2x AND boom)")
ax.set_ylabel("% of names")
ax.legend(frameon=False, fontsize=8)
ax = axes[2]
ax.plot(b.index, b["exit_ready"].where(ok), color=C3, lw=1.1,
        label="EXIT readiness (mean out_score / trigger)")
ax.plot(b.index, b["entry_ready"].where(ok), color=C1, lw=1.1,
        label="ENTRY readiness (mean in_score / trigger)")
ax.set_ylabel("readiness")
ax.legend(frameon=False, fontsize=8)
ax = axes[3]
ax.plot(b.index, b["out_21d"], color=C3, lw=1.1,
        label="GET OUT flags, trailing 21d (all names)")
ax.plot(b.index, b["in_21d"], color=C1, lw=1.1,
        label="GET IN flags, trailing 21d")
ax.set_ylabel("flags / 21d")
ax.legend(frameon=False, fontsize=8)
ax = axes[4]
ax.plot(b.index, b["n_meas"], color=MUTED, lw=1)
ax.axhline(FLOOR, color=C4, ls=":", lw=1)
ax.set_ylabel("names\nmeasurable")
for ax in axes:
    for lo, hi in [(g.index.min(), g.index.max())
                   for _, g in (~ok).groupby((ok != ok.shift()).cumsum())
                   if not g.empty and not ok.loc[g.index[0]]]:
        ax.axvspan(lo, hi, color=GRID, alpha=0.45, lw=0)
    despine(ax)
loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
axes[-1].xaxis.set_major_locator(loc)
axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
fig.tight_layout()
plt.show()

peak_day = b["exit_ready"].where(ok).idxmax()
print(f"the composite's all-time exit-readiness maximum: "
      f"{b['exit_ready'].where(ok).max():.3f} on {peak_day.date()} - the "
      f"meme mania. Trailing-21d GET OUT flags peaked at "
      f"{int(b['out_21d'].max())} on {b['out_21d'].idxmax().date()}.")

# %% [markdown]
# **SO WHAT**
#
# * **The composite is a speculative-temperature gauge and it reads like
#   one**: its all-time maximum is the January-2021 meme mania (exit
#   readiness ≈ 0.16, 8 GET OUT flags in 21 days), with secondary ridges
#   in late-2019/early-2020 and the 2021 echo waves. SPY itself barely
#   flinched at the 2021 peak — the first hint (confirmed in §2) that
#   crowd euphoria in OUR universe is not an S&P timing signal.
# * Even at the mania peak the average name sits at ~16% of its own
#   trigger — breadth composites live near zero because most names are
#   quiet most days. The information is in the RIDGES, not the level.
# * The grey bands are the honest cost of the archive: 2024–25 are
#   unmeasurable (coverage floor), so no index statement of any kind is
#   made there.

# %% [markdown]
# ---
# # §2 — THE SAME PROCESS AT INDEX LEVEL, AND ITS JUDGE
#
# **WHY THIS** — "the same process as before" has a precise meaning here:
# composite score → gate → 7d smoothing → walk-forward per-test-year
# frozen threshold → 21d cooldown → judged against price-only ground
# truth. Every stage is reused; the only new object is the composite
# itself.
#
# **HOW IT WORKS — each stage, with provenance**
#
# * **Score**: exit readiness (entry readiness for GET IN) — production
#   scores over frozen triggers, defined in §1.
# * **Gate (the breadth condition)**: share of names in end-stage must be
#   in its own top decile — trailing percentile rank ≥ 0.90 over 365d.
#   CONVENTION, reused: the same top-decile-of-own-history definition as
#   the A2 attention gate (`EUPHORIA_ATT_GATE`), on the same window
#   (`EUPHORIA_PCT_WINDOW`); "the market cannot be euphoric quietly"
#   either. GET IN mirror: share in GET IN candidacy, same rank, same
#   cut.
# * **Smoothing**: the house 7d (`ROLL`). **Cooldown**: 21d
#   (`EUPHORIA_COOLDOWN_DAYS`).
# * **The judge**: `ground_truth_peaks(SPY, kind="theme")` — the ETF exam
#   bars (+25% boom off the 120d low, −15% bust within 90d), unchanged.
# * **The threshold**: this is where the process must be allowed to
#   refuse. The walk-forward can only freeze a threshold if the exam
#   contains episodes; the cell below runs the attempt and reports the
#   episode count LOUDLY before anything descriptive is shown.

# %%
spy_eps = find_episodes(pxb["SPY"], "sp500_composite", "SPY", "theme")
spy_peaks = ground_truth_peaks(pxb["SPY"], "theme")
run120 = (pxd["SPY"] / pxd["SPY"].rolling(120, min_periods=60).min()
          - 1)
fall7 = pxd["SPY"].shift(-7) / pxd["SPY"] - 1
print(f"ground_truth_peaks(SPY, kind='theme'): {len(spy_peaks)} peaks | "
      f"find_episodes: {len(spy_eps)} episodes  <- THE NUMBER, LOUDLY: "
      "ZERO")
print(f"  why: SPY's biggest 120d run-up on record is "
      f"{run120.max():+.0%} ({run120.idxmax().date()}) - the +25% boom "
      "bar HAS been cleared - and SPY has crashed >=10%-in-7d on "
      f"{int((fall7 <= -0.10).sum())} days "
      f"({sorted(set(fall7.index[fall7 <= -0.10].year))}) - but never "
      "BOTH in one arc: no confirmed local max that followed a >=25% "
      "boom was then followed by a >=15% drawdown within 90d.")
print("  => there are NO positives to train on, so the walk-forward "
      "threshold selection CANNOT run honestly. The confirmatory test "
      "is specified below and parked; what follows is DESCRIPTIVE.")

# %% [markdown]
# **The pre-specified confirmatory test (parked, not run):** the day
# SPY's history contains ≥ 3 qualifying episodes in the training years,
# run this composite through `run_tournament_entry` exactly as every
# name-level signal is (per-test-year thresholds from train years only,
# FA budget 0.23, 21d cooldown) and adopt the index flag only if it
# captures ≥ 1 episode inside budget. Until the exam exists, no
# index-level threshold is frozen and no index flag ships — a threshold
# fitted to zero positives would be a number wearing a costume.
#
# **The descriptive stand-in below** replaces only the un-freezable
# stage (the threshold) with a parameter-free convention — the score's
# own trailing top decile (365d, ≥ 0.90), the same convention as the
# gate — and keeps every other stage of the process unchanged. Its
# output is an EVENT STUDY, not a record: n is tiny and is printed on
# everything.

# %%
comp = {"OUT": b["exit_ready"].where(ok), "IN": b["entry_ready"].where(ok)}
gate_src = {"OUT": b["end_share"].where(ok), "IN": b["in_open"].where(ok)}
flags, sm_keep = {}, {}
for side in ("OUT", "IN"):
    sm = comp[side].rolling(ROLL, min_periods=1).mean()
    gate = trailing_pct_rank(gate_src[side]) >= EUPHORIA_ATT_GATE
    thr_s = sm.rolling(EUPHORIA_PCT_WINDOW, min_periods=180).quantile(
        EUPHORIA_ATT_GATE)
    fire = (sm >= thr_s) & gate
    al, last = [], None
    for d in fire.index[fire.fillna(False)]:
        if last is None or (d - last).days >= EUPHORIA_COOLDOWN_DAYS:
            al.append(d)
            last = d
    flags[side] = al
    sm_keep[side] = sm
    print(f"index-level {side}: {len(al)} descriptive flags "
          f"({al[0].date()} → {al[-1].date()})")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(spy.index, spy.values, color=INK, lw=1.1)
ax.set_yscale("log")
for d in flags["OUT"]:
    ax.axvline(d, color="#b3002d", lw=1.4, alpha=0.85)
for d in flags["IN"]:
    ax.axvline(d, color=C2, lw=1.4, alpha=0.85)
for lo, hi in [(gg.index.min(), gg.index.max())
               for _, gg in (~ok).groupby((ok != ok.shift()).cumsum())
               if not gg.empty and not ok.loc[gg.index[0]]]:
    ax.axvspan(lo, hi, color=GRID, alpha=0.45, lw=0)
ax.set_ylabel("SPY (log)")
ax.set_title("SPY with the DESCRIPTIVE index-level flags (dark red = "
             "index GET OUT, green = index GET IN; grey = crowd "
             "unmeasurable)\nread it like this: the dark-red cluster "
             "before the 2020 crash is the pattern the desk hopes for; "
             "the 2018-2019 reds with no crash behind them are the "
             "false-alarm cost", fontsize=9)
despine(ax)
fig.tight_layout()
plt.show()

# %%
TD_HORIZONS = [5, 20, 84]
rng = np.random.default_rng(SEED)

def fwd_td(px, dates, n_td):
    idx = px.index.searchsorted(pd.DatetimeIndex(dates))
    out = np.full(len(idx), np.nan)
    j = idx + n_td
    okk = j < len(px)
    out[okk] = px.values[j[okk]] / px.values[idx[okk]] - 1
    return out

def hit_ci(hits, n_boot=2000):
    hits = np.asarray(hits, dtype=float)
    if len(hits) == 0:
        return (np.nan, np.nan)
    bs = [rng.choice(hits, size=len(hits), replace=True).mean()
          for _ in range(n_boot)]
    return (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))

def event_table(px, side_flags, baseline_days, down_is_hit):
    rows = []
    for h in TD_HORIZONS:
        v = pd.Series(fwd_td(px, side_flags, h)).dropna()
        base = pd.Series(fwd_td(px, baseline_days, h)).dropna()
        hits = (v < 0) if down_is_hit else (v > 0)
        bhits = (base < 0) if down_is_hit else (base > 0)
        lo, hi = hit_ci(hits.values)
        rows.append({"horizon (td)": h, "n flags": len(v),
                     "hit rate": round(float(hits.mean()), 2),
                     "hit 90% CI": f"[{lo:.2f}, {hi:.2f}]",
                     "baseline hit": round(float(bhits.mean()), 2),
                     "median move": round(float(v.median()), 4),
                     "baseline median": round(float(base.median()), 4)})
    return pd.DataFrame(rows)

base_days = sm_keep["OUT"].dropna().index      # every day the composite spoke
spyb = pxb["SPY"]
ev_out = event_table(spyb, flags["OUT"], base_days, down_is_hit=True)
ev_in = event_table(spyb, flags["IN"], base_days, down_is_hit=False)
print("index GET OUT vs SPY (hit = SPY fell):")
display(ev_out)
print("index GET IN vs SPY (hit = SPY rose):")
display(ev_in)

# the desk's cliff: days to a >=10%-in-7d SPY fall after an OUT flag
def days_to_fall(d, horizon=90):
    win = fall7.loc[d:d + pd.Timedelta(days=horizon)]
    hits = win[win <= -0.10]
    return float((hits.index[0] - d).days) if len(hits) else np.nan

d2f = pd.Series([days_to_fall(d) for d in flags["OUT"]],
                index=flags["OUT"])
base_fall = pd.Series([days_to_fall(d) for d in base_days],
                      index=base_days)
print(f"\n>=10%-in-7d SPY fall within 90d: after {int(d2f.notna().sum())}"
      f" of {len(d2f)} OUT flags ({d2f.notna().mean():.0%}) vs "
      f"{base_fall.notna().mean():.0%} of all composite days; when it "
      f"came, median {d2f.dropna().median():.0f} days (the flags that "
      "saw one: "
      + ", ".join(f"{d.date()} (+{v:.0f}d)"
                  for d, v in d2f.dropna().items()) + ")")

# %% [markdown]
# **SO WHAT — the nulls first, because they are the record**
#
# * **The index-level GET OUT is NOT a validated SPY signal.** At +5td
#   and +20td the down-hit rate after a flag is *below* the all-days
#   baseline; at +84td it is above (0.45 vs 0.32) but on 20 flags the
#   bootstrap CI spans roughly [0.25, 0.65] — indistinguishable from
#   baseline. Twenty flags is an anecdote budget, not a record.
# * **The pattern worth keeping is the 2019-12 → 2020-01 cluster**: three
#   consecutive index GET OUT flags 72, 49 and 28 days before the COVID
#   crash. The full ledger is less kind: the June-2022 fall had no flag
#   within the 90 days before it (the Aug-2022 flag came after), the
#   April-2025 fall sat in the unmeasurable hole, and 2018–2019 supplied
#   a run of flags with no crash behind them. One true cluster, several
#   false ones, one miss: exactly what "suggestive, unproven" looks like.
# * **The index GET IN's +5td edge (0.72 vs 0.61) decays with horizon
#   and is null by +84td** — consistent with what it mechanically is: a
#   burst of name-level entry candidacies, i.e. short-term market-wide
#   momentum.
# * The deep reason the judge is empty: **diversification is exactly the
#   property that keeps SPY out of this project's exam.** A 500-name
#   cap-weighted index does not run +25% in 120 days and then give back
#   15% in 90 — single themes do. The per-name process survives intact;
#   the EXAM is what does not transfer to the index.
#
# **IF ASKED — "so lower the exam bars for the index until episodes
# exist."** That is tuning the exam to manufacture a gradeable record —
# the one optimisation the project refuses (NB04 §1.6 states the
# principle). If the desk wants an index-risk product, the honest routes
# are (a) wait for the confirmatory test to become runnable, or (b)
# pre-register a DIFFERENT, index-appropriate outcome (e.g. forward
# drawdown quantiles) as a new exam — forward-only, with its own budget.

# %% [markdown]
# ---
# # §3 — MTUM: the signal runs today, the judge is pending
#
# **WHY THIS** — the desk asked for MTUM explicitly, and momentum
# crowding is arguably what this universe measures BEST: our names are
# wherever retail attention is piling in, which is the crowded-momentum
# cross-section by construction. The composite from §1/§2 IS the MTUM
# signal side. What cannot run today is the JUDGING: **MTUM has no rows
# in `prices.parquet`** (added to `config/approved_instruments.csv`
# 2026-07-31; the next Bloomberg pull fetches it). The cell below
# detects the symbol automatically — when the pull lands, re-running
# this notebook judges MTUM with no edit.

# %%
def judge_index_flags(symbol: str, kind: str = "theme"):
    """The MTUM-ready judge: episodes from the project's exam, the
    descriptive flags from §2, the event study from §2 - all reused.
    Runs for ANY priced symbol; returns None when the symbol is not
    priced yet."""
    if symbol not in PRICED:
        return None
    px = (prices[prices.symbol == symbol].sort_values("date")
          .set_index("date")["px_last"].dropna())
    eps = find_episodes(px, f"index_{symbol}", symbol, kind)
    return {"symbol": symbol,
            "episodes": len(eps),
            "peaks": [str(e.peak.date()) for e in eps],
            "event_out": event_table(px, flags["OUT"], base_days,
                                     down_is_hit=True),
            "event_in": event_table(px, flags["IN"], base_days,
                                    down_is_hit=False)}

mtum = judge_index_flags("MTUM")
if mtum is None:
    print("=" * 72)
    print("MTUM JUDGING: **PENDING** - MTUM has no rows in "
          "prices.parquet yet.")
    print("It was added to config/approved_instruments.csv on "
          "2026-07-31; the next `pull_bloomberg_prices.py` run fetches "
          "it, and re-executing this notebook then judges MTUM "
          "automatically through judge_index_flags('MTUM') - no code "
          "change needed.")
    print("=" * 72)
else:
    print(f"MTUM priced - judging. Episodes under the exam: "
          f"{mtum['episodes']} ({mtum['peaks']})")
    print("index GET OUT vs MTUM (hit = MTUM fell):")
    display(mtum["event_out"])
    print("index GET IN vs MTUM (hit = MTUM rose):")
    display(mtum["event_in"])

# %% [markdown]
# **Meanwhile, the stated substitution: the composite is shown against
# QQQ — the closest PRICED crowded-momentum line — as an interim proxy
# for MTUM. QQQ is not MTUM** (Nasdaq-100 concentration vs a momentum
# factor screen), but it is the nearest priced expression of "the
# crowded trade", and every number below is replaced by the real MTUM
# judgement at the next price pull.

# %%
qqq_res = judge_index_flags("QQQ")
print(f"QQQ episodes under the exam: {qqq_res['episodes']} - like SPY, "
      "ZERO (diversified indices do not pass the theme bars), so this "
      "too is an event study, not a record.")
print("index GET OUT vs QQQ (hit = QQQ fell):")
display(qqq_res["event_out"])
print("index GET IN vs QQQ (hit = QQQ rose):")
display(qqq_res["event_in"])

qqq = pxd["QQQ"]
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.4]})
ax = axes[0]
ax.plot(qqq.index, qqq.values, color=INK, lw=1.1)
ax.set_yscale("log")
for d in flags["OUT"]:
    ax.axvline(d, color="#b3002d", lw=1.4, alpha=0.85)
for d in flags["IN"]:
    ax.axvline(d, color=C2, lw=1.4, alpha=0.85)
ax.set_ylabel("QQQ (log)")
ax.set_title("QQQ (interim MTUM proxy - substitution stated in bold "
             "above) with the same descriptive index flags\nlook at "
             "whether the dark-red flags sit nearer QQQ's crests than "
             "SPY's - the crowded line should feel the crowd first",
             fontsize=9)
ax = axes[1]
ax.plot(b.index, sm_keep["OUT"], color=C3, lw=1.2,
        label="smoothed exit readiness (the composite)")
ax.plot(b.index, sm_keep["IN"], color=C1, lw=1.2,
        label="smoothed entry readiness")
ax.set_ylabel("readiness")
ax.legend(frameon=False, fontsize=8)
for ax in axes:
    for lo, hi in [(gg.index.min(), gg.index.max())
                   for _, gg in (~ok).groupby((ok != ok.shift()).cumsum())
                   if not gg.empty and not ok.loc[gg.index[0]]]:
        ax.axvspan(lo, hi, color=GRID, alpha=0.45, lw=0)
    despine(ax)
loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
axes[-1].xaxis.set_major_locator(loc)
axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
fig.tight_layout()
plt.show()

# %% [markdown]
# **SO WHAT**
#
# * **Against QQQ the picture is the same shape as against SPY, not
#   better — stated plainly.** The +84td down-hit edge over baseline is
#   +14pp on both lines (QQQ 0.40 vs 0.26; SPY 0.45 vs 0.31), with CIs
#   that overlap the baselines; the short horizons are below baseline on
#   both. The one QQQ-specific reading: the median +84td move after a
#   flag lags QQQ's own baseline drift by more than on SPY (−5.5pp vs
#   −2.1pp) — the crowded line pays more forgone upside around these
#   flags — a directional hint, not a result, on 20 flags. **No
#   MTUM-specific claim exists until MTUM itself is judged.**
# * **The MTUM confirmatory run is fully specified and costless**: next
#   price pull → re-execute → `judge_index_flags("MTUM")` fills §3. If
#   MTUM (a 100-plus-name factor sleeve) also shows zero exam episodes —
#   likely, for the same diversification reason — the event study is the
#   deliverable there too, and the pre-registered §2 criterion governs
#   anything stronger.

# %% [markdown]
# ---
# # §4 — Closing: what ships, what does not, one line for the PM
#
# **WHAT SHIPS**
#
# * The §1 **breadth composite as dashboard CONTEXT** (five series, all
#   descriptive, all from production stores, no new thresholds): share
#   in end-stage, candidacy shares, exit/entry readiness, trailing flag
#   counts, with the coverage floor and grey bands exactly as drawn
#   here.
# * The **MTUM auto-judge hook** (§3): zero-maintenance; activates on
#   the next price pull.
#
# **WHAT DOES NOT SHIP, loudly**
#
# * **No index-level GET OUT / GET IN flag.** The exam contains zero SPY
#   (and zero QQQ) episodes, so no threshold can be frozen and no
#   capture record exists; the descriptive flags are an event study
#   whose +5/+20td GET OUT numbers sit BELOW baseline. Shipping a flag
#   on the strength of one good COVID cluster would be selling one
#   anecdote.
# * No cap-weighted S&P claim of any kind — our universe is the
#   sentiment cross-section, and every §1 chart says so.
#
# **LIMITATIONS, plainly:** the composite's denominator moves with
# coverage (54 names at best, 0 in the 2024–25 hole); readiness mixes
# themes and singles with equal weight (a DESK-style convention — no
# market-cap data enters this project); the descriptive trigger reuses
# the top-decile convention rather than a fitted value, which is the
# point but also means its operating level was never optimised; and 18–20
# flags support no inference stronger than the CIs printed on them.
#
# **One line for the PM:** *we can tell you when the whole tracked crowd
# is euphoric at once — the composite nailed Jan-2021 and flagged ahead
# of COVID — but the S&P itself never stages the boom-bust arcs our exam
# grades, so there is no honest walk-forward index flag yet: use the
# breadth strip as context, and MTUM gets its real judgement at the next
# price pull.*

# %%
out = {
    "as_of": pd.Timestamp.now().isoformat(timespec="seconds"),
    "universe": {"themes": int(n_theme), "singles": int(n_single)},
    "coverage_floor": {"floor": FLOOR,
                       "sensitivity": sens.to_dict(orient="records"),
                       "days_kept_by_year": {int(k): float(v) for k, v
                                             in keep_by_year.items()}},
    "composite": {
        "exit_ready_max": float(b["exit_ready"].where(ok).max()),
        "exit_ready_argmax": str(peak_day.date()),
        "out_flags_21d_max": int(b["out_21d"].max()),
        "out_flags_21d_argmax": str(b["out_21d"].idxmax().date())},
    "spy_exam": {
        "episodes": len(spy_eps),
        "max_120d_runup": float(run120.max()),
        "n_days_10pct_7d_fall": int((fall7 <= -0.10).sum()),
        "verdict": "walk-forward threshold CANNOT be frozen (0 episodes)"
                   " - descriptive only; confirmatory test pre-specified"},
    "descriptive_flags": {
        "out": [str(d.date()) for d in flags["OUT"]],
        "in": [str(d.date()) for d in flags["IN"]],
        "rule": "smoothed readiness >= own trailing 365d q0.90 AND "
                "breadth pct-rank >= 0.90 AND 21d cooldown "
                "(all reused conventions, no fitted numbers)"},
    "event_study_spy": {"get_out": ev_out.to_dict(orient="records"),
                        "get_in": ev_in.to_dict(orient="records"),
                        "cliff": {
                            "flags_with_10in7_fall_90d":
                                int(d2f.notna().sum()),
                            "n_flags": int(len(d2f)),
                            "baseline_share":
                                float(base_fall.notna().mean()),
                            "median_days_when_it_came":
                                (float(d2f.dropna().median())
                                 if d2f.notna().any() else None)}},
    "event_study_qqq_proxy": {
        "note": "QQQ stands in for MTUM until priced (stated in bold "
                "in the notebook)",
        "episodes": qqq_res["episodes"],
        "get_out": qqq_res["event_out"].to_dict(orient="records"),
        "get_in": qqq_res["event_in"].to_dict(orient="records")},
    "mtum": {"priced": "MTUM" in PRICED,
             "status": ("judged" if mtum is not None else
                        "PENDING - auto-judges at next price pull")},
    "ships": ["breadth composite as dashboard context",
              "MTUM auto-judge hook"],
    "does_not_ship": ["index-level GET OUT/GET IN flags (no exam "
                      "episodes; descriptive record null at short "
                      "horizons)"],
}
with open(RESEARCH_DIR / "nb07_index_composite.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print(f"saved nb07_index_composite.json | total runtime "
      f"{time.time()-NB_T0:.0f}s")
