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
# # Notebook 07 — The Presentation Pack
#
# **One notebook, the whole talk** (merged 2026-08-09 from the former
# presentation pack + the slide pack, desk instruction: "if there is
# overlap just combine the two"). It answers the eight questions the
# presentation must answer — **one PPT-ready figure (or two) per
# question**, for a NON-TECHNICAL audience: big fonts, one message per
# chart, plain-English titles, a professional blue / dark-blue / grey
# palette. Every figure saves to `docs/figures/slides/` at 200 dpi,
# sized for a 16:9 slide.
#
# | § | The question | Figures |
# |---|---|---|
# | 0 | The pipeline on one page — and which figure answers what | S00 |
# | P | Was the data trustworthy? (the defect we fixed first) | S0a, S0b |
# | 1 | What cases are we optimising for? (the bubbles; the objective) | S1a, S1b, S1c |
# | 2 | What features come from the raw data? | S2a, S2b |
# | 3 | What choices did we have for the modelling? | S3 |
# | 4 | Which features were good? (AUROC + ablation) | S4a, S4b |
# | 5 | Why this particular ML model? | S5 |
# | 5.5 | What are the weights, and how do they become a call? | S5b, S5c |
# | 6 | Why these thresholds? (score distributions, yearly cuts, Spearman) | S6a, S6b |
# | 6.5 | One clear call per boom: phase gates, re-arm, spacing, separation | S6c |
# | 7 | How do the pieces combine into a GET IN / GET OUT? | S7a, S7b |
# | 8 | How well does it perform — and is it better than before? | S8a, S8b, S8c |
#
# Everything quotes the SAME frozen artifacts the pipeline uses
# (`docs/research/ml_tournament.json`, `nb02_august_bank.json`,
# `conditioning_sweep.json`, `data/processed/episodes.parquet`) or
# recomputes through the live modules (`analytics/ml_detector.py`,
# `analytics/euphoria_phases.py`) — nothing is re-implemented here.
#
# ## VERDICT BOX — read this and you have the talk
#
# | Question | Answer | Evidence |
# |---|---|---|
# | Was the data broken? | The DATA was fine; the ESTIMATOR was broken. Fake zero-share days (308 across the top-30 names, May–Aug) came from averaging daily ratios over an uneven pull cadence and a shifting source mix. Fixed with three standard techniques (ratio-of-sums, per-source stratification, empirical-Bayes shrinkage), no data invented. | §P |
# | What counts as a euphoria episode? | Price-only ground truth: local max + boom + bust. Bars LOWERED 25/50→**20/40** boom, 15/30→**12/25** bust after a three-level sweep: episodes 292→494, every named desk episode kept, accuracy improved. | §1 |
# | What predicts it? | One bank of **11 plain-English measurements** (9 crowd + 2 price), each a percentile against the name's own history — no embedded gate constants left. | §2, §4 |
# | What model ships? | The **logit + monotone-GBM rank ensemble** — winner of a pre-stated walk-forward tournament over 5 families. | §3, §5 |
# | Is it better than the old rules? | Yes. The model ranks episode-days 2.4–2.6× better than chance (the rules: 1.1–1.2×); the shipped shaped calls catch 2.5× the tops (29% vs 12%, median 6 days before the peak) at one false alarm per name per ~9 years, and GET IN precision is 1.7× the old rules'. | §8 |
# | Does it flip-flop? | Fixed (2026-08-09, two rounds of evidence): calls are PHASE-GATED by the episode definition itself — GET IN only before a name's boom completes, GET OUT only after — with deep re-arm, one call per name per quarter per side, and no GET IN within 21d of a GET OUT in either direction. Adjacent IN/OUT pairs: zero, by construction. Score smoothing was tested and REJECTED. | §6.5 |
# | Is it explainable? | Three surviving numbers: 100-tagged-post coverage floor, 21-day cooldown, a probability cut chosen on past years (F1 standard / F0.5 strict). Everything else is learned, walk-forward, from earlier years only. | §6, notebook 03 §SS2.0 |
#
# ---
# ## DEFINITIONS (desk English, before any result)
#
# - **Tagged post** — a post that names a ticker or theme we track (and
#   therefore carries a sentiment read). "TOTAL posts" = everything
#   pulled, tagged or not.
# - **Share of chatter** — the fraction of everything retail posted that
#   day that mentions this name. Era-safe: comparable between the 2021
#   archive and a live week.
# - **Episode** — a full boom-bust arc trough → peak → bust, defined by
#   price alone (so the crowd detector is graded by an independent judge).
# - **GET IN / GET OUT** — the two desk signals: euphoria *starting* /
#   euphoria *ending*. Fired on an upward crossing of the cut, one alert
#   per name per 21 days.
# - **Standard / Strict** — the two operating points on the dashboard:
#   the F1 cut (balanced) and the F0.5 cut (precision weighted twice —
#   about half the calls and half the false alarms).
# - **Walk-forward** — every fit and every threshold is chosen on years
#   strictly BEFORE the year being scored, then frozen. Nothing ever sees
#   its own test year.
# - **AP lift** — threshold-free score quality over the base rate of the
#   model's own candidate frame — the number that compares fairly across
#   models with different candidacies. 1.0× = guessing.

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
# %matplotlib inline
from sklearn.metrics import average_precision_score

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

# THE SLIDE PALETTE (professional blue / dark blue / grey)
NAVY = "#0A2540"     # dark blue: ink, emphasis, price, GET OUT
BLUE = "#2E6FDB"     # primary: the model, GET IN
SKY = "#9DB9DC"      # light blue: fills, shading, secondary bars
GREY = "#6B7280"     # the old rules, context, baselines
LIGHT = "#E8EBEF"    # grid, soft fills
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12.5,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.edgecolor": LIGHT, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.labelcolor": NAVY, "xtick.color": GREY, "ytick.color": GREY,
    "text.color": NAVY,
    "figure.dpi": 100, "savefig.bbox": "tight",
})

SLIDE_DIR = ROOT / "docs" / "figures" / "slides"
SLIDE_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_DIR = ROOT / "docs" / "research"
SAVED = []


def save(fig, name):
    p = SLIDE_DIR / name
    fig.savefig(p, dpi=200, facecolor="white")
    SAVED.append(name)
    print(f"saved {p.relative_to(ROOT)}")


def despine(ax, keep_bottom=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(keep_bottom)


# %% [markdown]
# ## Load — the same live modules and frozen records as the pipeline

# %%
from analytics import ml_detector as mld
from analytics.euphoria import build_all_series
from analytics.euphoria_phases import (build_day_frame, walk_forward_scores,
                                       _alerts_int, _day_ints)
from analytics.loaders import (load, THEME_COUNTS, THEME_SENT,
                               TICKER_COUNTS, TICKER_SENT,
                               TICKER_COUNTS_BY_SOURCE)
from analytics.robust_share import robust_share
from src.config import PROCESSED_DIR, EUPHORIA_COOLDOWN_DAYS

t0 = time.time()
prices = pd.read_parquet(ROOT / "data" / "prices" / "prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
series, pxmap = build_all_series(prices)
sym_by = {es.name: es.symbol for es in series}
episodes = pd.read_parquet(ROOT / "data" / "processed" / "episodes.parquet")
for c in ("trough", "peak", "bust_date"):
    episodes[c] = pd.to_datetime(episodes[c])
counts = {"theme": load(THEME_COUNTS), "ticker": load(TICKER_COUNTS)}
sents = {"theme": load(THEME_SENT), "ticker": load(TICKER_SENT)}
for d in list(counts.values()) + list(sents.values()):
    d["date"] = pd.to_datetime(d["date"])
frame = build_day_frame(series, pxmap, episodes, counts, sents)
cand_px = mld.attach_price_features(mld.candidate_frame(frame), series,
                                    pxmap)

T = json.load(open(RESEARCH_DIR / "ml_tournament.json"))
RES = T["results"]
BANK_BOARD = json.load(open(RESEARCH_DIR / "nb02_august_bank.json"))
GT_SWEEP = T.get("ground_truth_sweep", {})
COND = json.load(open(RESEARCH_DIR / "conditioning_sweep.json"))
SHAPE = json.load(open(RESEARCH_DIR / "alert_shape_sweep.json"))
desk_rep = json.load(open(Path(PROCESSED_DIR) /
                          "euphoria_desk_report.json"))
print(f"{time.time()-t0:.0f}s | {len(series)} instruments | "
      f"{len(episodes)} episodes | {len(cand_px):,} candidate days | "
      f"model on file: {desk_rep['model']} | trigger: "
      f"{desk_rep.get('conditioning', {}).get('trigger', 'level')}")

# %% [markdown]
# **The walk-forward scores, computed once here and reused everywhere
# below** (cached to `docs/research/nb07_wf_scores.parquet`): the SAME
# construction production uses — fit on years < Y, score year Y,
# rank-average the two members within the year.

# %%
CACHE = RESEARCH_DIR / "nb07_wf_scores.parquet"
FP = {"rows": len(cand_px), "max_date": str(cand_px["date"].max().date())}
WF = None
if CACHE.exists():
    _c = pd.read_parquet(CACHE)
    if ("fingerprint" in _c.columns
            and json.loads(_c["fingerprint"].iloc[0]) == FP):
        WF = _c.drop(columns=["fingerprint"])
        print("reusing cached walk-forward scores")
if WF is None:
    parts = []
    for label, head in (("y_top", "GET OUT"), ("y_onset", "GET IN")):
        t1 = time.time()
        key = ["name", "date", "test_year"]
        lg = walk_forward_scores(cand_px, mld.DESK_ML_BANK, label,
                                 mld.make_logit_fit(label))
        gb = walk_forward_scores(cand_px, mld.DESK_ML_BANK, label,
                                 mld.make_gbm_fit(label))
        m = (lg[key + [label, "score"]].rename(columns={"score": "logit"})
             .merge(gb[key + ["score"]].rename(columns={"score": "gbm"}),
                    on=key))
        m["logit_rank"] = m.groupby("test_year")["logit"].rank(pct=True)
        m["gbm_rank"] = m.groupby("test_year")["gbm"].rank(pct=True)
        m["score"] = (m["logit_rank"] + m["gbm_rank"]) / 2
        m["head"] = head
        m = m.rename(columns={label: "y"})
        parts.append(m)
        print(f"{head}: walk-forward members {time.time()-t1:.0f}s")
    WF = pd.concat(parts, ignore_index=True)
    out = WF.copy()
    out["fingerprint"] = json.dumps(FP)
    out.to_parquet(CACHE, index=False)
    print(f"cached -> {CACHE.relative_to(ROOT)}")

THR = {h: {int(k): v for k, v in RES[hk]["ens"]["thresholds"].items()}
       for h, hk in (("GET OUT", "get_out"), ("GET IN", "get_in"))}


# %% [markdown]
# ---
# # S00 — The pipeline on one page (and which figure answers what)
#
# Read this first. It is the whole machine end to end, and it names the
# figure that answers each question — so "where is the evidence for X?"
# never needs a search.

# %%
_fig, _ax = plt.subplots(figsize=(13.4, 6.6))
_ax.set_xlim(0, 100)
_ax.set_ylim(0, 100)
_ax.axis("off")

_STAGE = [
    (2, "THE CROWD", "Reddit · StockTwits · X\nevery post pulled",
     NAVY),
    (21.5, "READ & DISCARD", "tickers · themes · mood\nthe text is thrown "
     "away", NAVY),
    (41, "11 MEASUREMENTS", "9 crowd + 2 price\neach vs the name's OWN "
     "history", BLUE),
    (60.5, "TWO MODELS", "logistic + monotone GBM\nfitted on PAST years "
     "only", BLUE),
    (80, "ONE CALL", "rank-average → frozen cut\nGET IN / GET OUT",
     "#1F6F5C"),
]
for _x, _t, _sub, _c in _STAGE:
    _ax.add_patch(mpatches.FancyBboxPatch(
        (_x, 52), 17, 22, boxstyle="round,pad=0.6,rounding_size=1.2",
        facecolor=_c, edgecolor="none"))
    _ax.text(_x + 8.5, 68, _t, ha="center", va="center", color="white",
             fontsize=10.5, fontweight="bold")
    _ax.text(_x + 8.5, 60, _sub, ha="center", va="center", color="white",
             fontsize=8.5, linespacing=1.5)
for _x in (19, 38.5, 58, 77.5):
    _ax.annotate("", (_x + 2.6, 63), (_x, 63),
                 arrowprops=dict(arrowstyle="-|>", lw=1.8, color=GREY))

_Q = [
    (2, "How is the raw data\nturned into numbers?", "S0a · S0b · S2a\nnb06 W0"),
    (21.5, "What is the target,\nand why those bars?", "S1a · S1b · S1c"),
    (41, "Which measurements\nactually matter?", "S4a · S4b · S5b"),
    (60.5, "Why this model, and\nhow does it work?", "S3 · S5 · S5b · S5c"),
    (80, "When does it fire, and\nis it any good?", "S6a–S6d · S7 · S8"),
]
for _x, _q, _figs in _Q:
    _ax.annotate("", (_x + 8.5, 44), (_x + 8.5, 51),
                 arrowprops=dict(arrowstyle="-|>", lw=1.3, color=GREY))
    _ax.add_patch(mpatches.FancyBboxPatch(
        (_x, 20), 17, 23, boxstyle="round,pad=0.5,rounding_size=1.0",
        facecolor="#F2F5FA", edgecolor="none"))
    _ax.text(_x + 8.5, 37, _q, ha="center", va="center", color=NAVY,
             fontsize=9.5, fontweight="bold", linespacing=1.5)
    _ax.text(_x + 8.5, 26, _figs, ha="center", va="center", color=BLUE,
             fontsize=9, linespacing=1.6)

_ax.text(2, 12,
         "Everything is WALK-FORWARD: the measurements, the models and the cut that fire in year Y were built from years before Y "
         "only.\nThe text-free boundary sits between box 2 and box 3 — no post text is ever committed or read by the desk machine.",
         fontsize=9.5, color=GREY, va="top")
_ax.set_title("The pipeline, end to end — and where each question is "
              "answered", fontsize=15, fontweight="bold", loc="left")
save(_fig, "S00_pipeline_map.png")
plt.show()

# %% [markdown]
# ---
# # P — Before trusting anything: the data, and the defect we fixed
#
# Every signal downstream is built on "share of chatter". The desk saw
# names' share "just go to zero, which doesn't make sense" — so before
# any modelling, the question was: bad data, or bad estimator? It was
# the estimator: the pipeline pulls ~2×/week (daily totals swing
# 246 → 4,033 posts between neighbouring days) and the source mix is a
# regime (Reddit since 2017, StockTwits from Feb-2026, X from Jul-2026).
# Averaging daily ratios against that denominator measures the puller,
# not the crowd. The fix (`analytics/robust_share.py`) is three textbook
# techniques — ratio-of-sums, per-source stratification, empirical-Bayes
# shrinkage — and no invented data.

# %%
# S0a — the denominator is a regime, not a constant
by_src = load(TICKER_COUNTS_BY_SOURCE)
by_src["date"] = pd.to_datetime(by_src["date"])
wk = (by_src.groupby([pd.Grouper(key="date", freq="W"), "source"])
      ["mention_count"].sum().unstack(fill_value=0)).loc["2025-01-01":]
SRC_C = {"reddit": NAVY, "stocktwits": BLUE, "x": SKY}

fig, ax = plt.subplots(figsize=(12.8, 4.2))
bottom = np.zeros(len(wk))
for s in ["reddit", "stocktwits", "x"]:
    if s in wk.columns:
        ax.bar(wk.index, wk[s].values, width=6, bottom=bottom,
               color=SRC_C[s], label=s, linewidth=0)
        bottom += wk[s].values
ax.set_title("The data problem we fixed first: the denominator is a "
             "regime, not a constant", fontsize=15)
ax.set_ylabel("tagged posts per week")
ax.legend(frameon=False, ncols=3)
ax.margins(x=0.01)
fig.text(0.01, -0.06,
         "StockTwits appears in bulk from spring 2026 and X only in July — any 'share of the whole forum' computed day-by-day\n"
         "against this total measures the puller, not the crowd.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S0a_source_regime.png")
plt.show()

# %%
# S0b — the estimator fix, on the name the desk complained about
tc = counts["ticker"]
all_days = pd.date_range(tc["date"].min(), tc["date"].max(), freq="D")
day_tot = (tc.groupby("date")["mention_count"].sum()
           .reindex(all_days).fillna(0.0))
NAME0 = "AMC"
m0 = (tc[tc.ticker == NAME0].groupby("date")["mention_count"].sum()
      .reindex(all_days).fillna(0.0))
old = ((m0 / day_tot.where(day_tot > 0)) * 100).rolling(
    7, min_periods=1).mean().loc["2026-05-01":"2026-08-05"]
new = robust_share(tc, "ticker", NAME0, all_days,
                   by_source=by_src).loc["2026-05-01":"2026-08-05"]

fig, axes = plt.subplots(2, 1, figsize=(12.8, 5.2), sharex=True)
axes[0].plot(old.index, old.values, color=GREY, lw=2.0)
for d in old.index[old.fillna(0) == 0]:
    axes[0].axvspan(d, d + pd.Timedelta(days=1), color=GREY, alpha=0.25,
                    lw=0)
axes[0].set_title(f"{NAME0}, share of chatter — OLD estimator "
                  "(average of daily ratios): stripes are fake zero-days",
                  fontsize=12.5, loc="left")
axes[0].set_ylabel("%")
axes[1].plot(new.index, new.values, color=BLUE, lw=2.0)
axes[1].set_title("NEW estimator (ratio-of-sums, source-stratified, "
                  "shrunk): the crowd never actually left", fontsize=12.5,
                  loc="left")
axes[1].set_ylabel("%")
for ax in axes:
    despine(ax)
fig.suptitle("The fix: estimate shares like a statistician, invent "
             "nothing", fontsize=16, fontweight="bold", y=1.02)
fig.text(0.01, -0.05,
         "Top-30 names, May–Aug 2026: 308 fake zero-days before the fix, 31 after (all genuine no-coverage stretches). Only observed\n"
         "counts are reweighted — the days aren't missing, the posts were never pulled, and inventing them would poison every count.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S0b_estimator_fix.png")
plt.show()

# %% [markdown]
# ---
# # Q1 — What are we optimising for?
#
# The target is the **crowd-driven boom→bust episode**: a name the retail
# crowd piles into, the price runs, and then it breaks. S1a shows three
# the desk names by heart; S1b shows why the definition's bars sit where
# they do; S1c anatomises one episode and the exact windows a call is
# graded against.

# %%
# S1a — the anatomy of what we hunt, on three famous cases
CASES1 = [("GME — the meme mania", "GME", "2020-08-01", "2021-09-30"),
          ("SMCI — the AI-semis run", "SMCI", "2023-09-01", "2024-09-30"),
          ("Gold — the 2026 rush", "GLD", "2025-08-01", "2026-05-01")]

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
for ax, (title, sym, w0, w1) in zip(axes, CASES1):
    px = pxmap[sym].dropna().loc[w0:w1]
    ax.plot(px.index, px.values, color=NAVY, lw=2.0)
    eps_n = episodes[(episodes.symbol == sym)
                     & (episodes.peak >= w0) & (episodes.peak <= w1)]
    for r in eps_n.itertuples():
        ax.axvspan(r.trough, r.peak, color=SKY, alpha=0.35, lw=0)
    if len(eps_n):
        big = eps_n.loc[eps_n.boom_pct.idxmax()]
        pk = px.asof(big.peak)
        ax.plot([big.peak], [pk], "o", color=NAVY, ms=8, zorder=5)
        ax.annotate(f"+{big.boom_pct*100:,.0f}%\nthen "
                    f"{big.bust_pct*100:,.0f}%",
                    (big.peak, pk), xytext=(8, -6),
                    textcoords="offset points", fontsize=11,
                    fontweight="bold", color=NAVY, va="top")
    if px.max() / max(px.min(), 1e-9) > 4:
        ax.set_yscale("log")
        ax.set_ylabel("price (log scale)")
    else:
        ax.set_ylabel("price")
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    despine(ax)
fig.suptitle("What we are optimising for: crowd-driven boom → bust "
             "episodes", fontsize=16, fontweight="bold", y=1.04)
fig.text(0.01, -0.10,
         "An episode = a rise of ≥40% (single stock) or ≥20% (theme/ETF) off its recent low, followed by a fall of ≥25% / ≥12%\n"
         "within 90 days (shaded = the run-up we must call). The objective: flag the START (GET IN) and the TOP (GET OUT) of episodes\n"
         "like these — catching as many as possible while keeping false calls rare (the two are balanced by F1; no other objective).",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S1a_what_we_hunt.png")
plt.show()

# %%
# S1b — why the definition's bars sit where they do (the sweep)
order = ["GT-old (25/50, 15/30)", "GT-ADOPTED (20/40, 12/25)",
         "GT-loosest (15/30, 10/20)"]
labels1 = ["stricter bars\n(25/50 boom)", "ADOPTED\n(20/40 boom)",
           "looser bars\n(15/30 boom)"]
fig, ax = plt.subplots(figsize=(12.8, 4.6))
xs = np.arange(len(order))
eps_counts = [GT_SWEEP[k]["episodes"] for k in order]
bars = ax.bar(xs, eps_counts, 0.55, color=[GREY, BLUE, SKY])
for x, b, k in zip(xs, bars, order):
    ap = GT_SWEEP[k]["get_out"]["ap"]
    gold = "gold ✓" if any("gold" in t for t in
                                GT_SWEEP[k]["themes_with_episode"]) \
        else "gold ✗"
    ax.text(x, b.get_height() + 6, f"{int(b.get_height())} episodes",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(x, b.get_height() / 2,
            f"model AP {ap:.2f}\n{gold}", ha="center", fontsize=11,
            color="white", fontweight="bold")
ax.set_xticks(xs, labels1)
ax.set_ylabel("episodes in 2017–2026")
ax.set_title("Why these bars: strict enough to mean something, loose "
             "enough to keep the cases that matter", fontsize=15)
fig.text(0.01, -0.06,
         "Looser bars mint many weak 'episodes' that dilute what the model learns; stricter bars shrink the sample toward only the\n"
         "monsters. The adopted 20/40 row keeps gold, the meme names and the semis in the catalogue while the model's accuracy holds.",
         fontsize=11, color=GREY, va="top")
despine(ax)
fig.tight_layout()
save(fig, "S1b_why_these_bars.png")
plt.show()

# %%
# S1c — one episode anatomised: the exact windows a call is graded on
EP_NAME = "gold_metals"
ep = episodes[episodes["name"] == EP_NAME].sort_values("boom_pct").iloc[-1]
px1 = pxmap[ep["symbol"]].dropna()
win = px1.loc[ep["trough"] - pd.Timedelta(days=60):
              (ep["bust_date"] if pd.notna(ep["bust_date"]) else ep["peak"])
              + pd.Timedelta(days=90)]

fig, ax = plt.subplots(figsize=(12.8, 4.6))
ax.plot(win.index, win.values, color=NAVY, lw=2.0)
ax.axvspan(ep["onset_lo"], ep["onset_hi"], color=BLUE, alpha=0.18, lw=0)
ax.axvspan(ep["peak"] - pd.Timedelta(days=30), ep["peak"], color=GREY,
           alpha=0.22, lw=0)
for d, lbl, c in ((ep["trough"], "trough", BLUE),
                  (ep["peak"], "peak", NAVY),
                  (ep["bust_date"], "bust confirmed", GREY)):
    if d is None or pd.isna(d):
        continue
    ax.axvline(d, color=c, lw=1.2, ls=":")
    ax.annotate(f" {lbl}", (d, win.max()), rotation=90, fontsize=10,
                color=c, va="top", ha="right")
ax.set_title(f"One episode, anatomised — gold ({ep['symbol']}), "
             f"{ep['peak'].year}: +{ep['boom_pct']:.0%} boom, "
             f"{ep['bust_pct']:.0%} bust", fontsize=14)
ax.set_ylabel("price (USD)")
despine(ax)
fig.text(0.01, -0.07,
         "Blue band = the GET IN scoring window (trough to trough+45d, capped at the peak); grey band = the GET OUT window\n"
         "(peak−30d to peak). A call inside its band is a hit; outside any episode it is a false alarm. The judge is price alone —\n"
         "the crowd never grades itself.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S1c_episode_anatomy.png")
plt.show()

# %% [markdown]
# ---
# # Q2 — What features come from the raw data?
#
# Three raw streams — **how many are talking, what they are saying, and
# what price is doing** — become eleven trailing measurements. S2a shows
# the raw streams on the GME mania; S2b lists all eleven in plain English.

# %%
# S2a — the three raw streams, on the case everyone knows
name2, sym2 = "GME", "GME"
w0, w1 = "2020-08-01", "2021-06-30"
m = (counts["ticker"][counts["ticker"].ticker == name2]
     .set_index("date")["mention_count"].asfreq("D").fillna(0).loc[w0:w1])
nb = (sents["ticker"][sents["ticker"].ticker == name2]
      .set_index("date")["net_bullish"].asfreq("D").loc[w0:w1])
px2 = pxmap[sym2].dropna().loc[w0:w1]

fig, axes = plt.subplots(3, 1, figsize=(12.8, 6.4), sharex=True,
                         height_ratios=[1, 1, 1.2])
axes[0].bar(m.index, m.values, color=SKY, width=1.0,
            label="tagged posts per day")
axes[0].plot(m.rolling(7).mean(), color=BLUE, lw=2.2,
             label="7-day average")
axes[0].set_title("Stream 1 — ATTENTION: how many posts name it",
                  fontsize=12.5, loc="left")
axes[0].legend(frameon=False, loc="upper left")
axes[1].axhline(0, color=GREY, lw=1)
axes[1].plot(nb.rolling(7, min_periods=3).mean(), color=BLUE, lw=2.2)
axes[1].set_title("Stream 2 — MOOD: net bullish-minus-bearish tone of "
                  "those posts (7-day average)", fontsize=12.5, loc="left")
axes[2].plot(px2.index, px2.values, color=NAVY, lw=2.2)
axes[2].set_yscale("log")
axes[2].set_title("Stream 3 — THE TAPE: price (log scale)",
                  fontsize=12.5, loc="left")
for ax in axes:
    despine(ax)
fig.suptitle("From raw data to features: the three streams, on GME "
             "2020–21", fontsize=16, fontweight="bold", y=1.01)
fig.text(0.01, -0.045,
         "The desk machine only ever sees counts and scores like these — never the post text itself (the text-free boundary).\n"
         "All 11 model features are trailing transforms of these three streams: day t uses nothing after day t.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S2a_three_streams.png")
plt.show()

# %%
# S2b — the 11 measurements, in plain English
ROWS2 = [
    ("THE CROWD — how many are talking", None),
    ("Attention level", "How loud is the name vs its own last year?"),
    ("Attention change, 1 month", "Is the crowd bigger than a month ago?"),
    ("Attention change, 2 weeks", "Is the crowd bigger than two weeks ago?"),
    ("Week vs month", "Is this week busier than this month?"),
    ("Week vs own normal", "Is this week above the name's own normal?"),
    ("Attention accelerating", "Is the growth itself speeding up? "
                               "(the classic bubble signature)"),
    ("THE MOOD — what they are saying", None),
    ("Bullishness level", "How bullish is the mood vs its own year?"),
    ("Bullishness persistence", "How long has the mood stayed bullish?"),
    ("Mood turning", "Is the mood starting to roll over?"),
    ("THE TAPE — what price is doing", None),
    ("Price run-up", "How far has price run off its recent low?"),
    ("Price return, 1 month", "What did price do over the last month?"),
]
fig, ax = plt.subplots(figsize=(12.8, 6.2))
ax.axis("off")
y = 1.0
for label, q in ROWS2:
    if q is None:
        y -= 0.055
        ax.text(0.01, y, label, fontsize=14, fontweight="bold",
                color=NAVY, va="top")
        y -= 0.065
    else:
        ax.text(0.05, y, f"•  {label}", fontsize=12.5, color=NAVY,
                va="top", fontweight="bold")
        ax.text(0.42, y, q, fontsize=12.5, color=GREY, va="top")
        y -= 0.062
ax.text(0.01, y - 0.02,
        "Every crowd/mood measurement is a percentile against the SAME "
        "name's own history — so 'loud' means loud for THAT name,\nand no "
        "hand-set constant is embedded in any feature.",
        fontsize=11.5, color=GREY, va="top", style="italic")
ax.set_title("The 11 measurements the model sees", fontsize=16,
             loc="left", pad=16)
save(fig, "S2b_the_11_measurements.png")
plt.show()

# %% [markdown]
# ---
# # Q3 — What choices did we have?

# %%
# S3 — the menu of approaches, and the verdict on each
import textwrap
MENU = [
    ("Hand rules  (the old system)",
     "fixed IF-THEN thresholds set by people",
     "fully — but every number must be defended by hand",
     "baseline — beaten"),
    ("Logistic regression",
     "one learned weight per measurement",
     "fully — the fitted model IS a printable formula",
     "strong and simple"),
    ("Monotone gradient-boosted trees",
     "hundreds of small decision trees; constrained so more crowd-heat "
     "can only mean MORE risk",
     "globally directional by construction",
     "strong; learns interactions"),
    ("Small neural network",
     "layered weighted sums (16-8 hidden units)",
     "weak — a black box",
     "no better here — rejected"),
    ("ENSEMBLE:  logistic + monotone GBM",
     "average of the two models' rankings (equal weight)",
     "inherits both members' readability",
     "ADOPTED"),
]
fig, ax = plt.subplots(figsize=(12.8, 5.6))
ax.axis("off")
cols_x = [0.01, 0.30, 0.60, 0.86]
for cx, h in zip(cols_x, ["Approach", "How it decides",
                          "Can we explain it?", "Verdict"]):
    ax.text(cx, 0.98, h, fontsize=13, fontweight="bold", color=GREY,
            va="top")
y = 0.88
for row in MENU:
    if row[3] == "ADOPTED":
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.005, y - 0.135), 1.005, 0.16,
            boxstyle="round,pad=0.005", facecolor=SKY, alpha=0.35,
            edgecolor="none", transform=ax.transAxes))
    for cx, txt, w in zip(cols_x, row, [0.27, 0.28, 0.24, 0.14]):
        fw = "bold" if cx in (cols_x[0], cols_x[3]) else "normal"
        c = BLUE if (row[3] == "ADOPTED" and cx == cols_x[3]) else \
            (NAVY if fw == "bold" else GREY)
        ax.text(cx, y, "\n".join(textwrap.wrap(txt, int(w * 105))),
                fontsize=12, fontweight=fw, color=c, va="top")
    y -= 0.185
ax.set_title("The modelling choices — every option was actually run, "
             "under the same rules", fontsize=16, loc="left", pad=14)
fig.text(0.01, -0.02,
         "All five contestants ran through the identical walk-forward tournament (fit on past years only, judged on unseen years).\n"
         "The selection rule was written down BEFORE the results were computed.",
         fontsize=11, color=GREY, va="top")
save(fig, "S3_the_menu.png")
plt.show()

# %% [markdown]
# ---
# # Q4 — Which features were good?
#
# Two independent reads. **S4a**: each measurement ALONE as a detector —
# AUROC, where 0.50 is a coin flip. **S4b**: remove one measurement from
# the adopted model and measure the accuracy (AP) it loses — the
# ablation. A feature can look modest alone but matter in combination
# (and vice versa), which is why both views are shown.

# %%
# S4a — every measurement alone (single-feature AUROC, walk-forward)
bb = pd.DataFrame(BANK_BOARD)
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)
for ax, col, ttl in ((axes[0], "AUROC vs y_top",
                      "Calling the TOP (GET OUT)"),
                     (axes[1], "AUROC vs y_onset",
                      "Calling the START (GET IN)")):
    d = bb.sort_values("AUROC vs y_top")
    cs = [NAVY if c in ("price_runup", "price_ret21") else BLUE
          for c in d["column"]]
    ax.barh(d["feature"], d[col], color=cs, height=0.62)
    ax.axvline(0.5, color=GREY, lw=1.4, ls="--")
    for i, v in enumerate(d[col]):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=10,
                color=NAVY)
    ax.set_xlim(0.42, 0.85)
    ax.set_title(ttl, fontsize=13)
    despine(ax, keep_bottom=True)
axes[0].legend(handles=[mpatches.Patch(color=BLUE, label="crowd / mood "
                                       "measurement"),
                        mpatches.Patch(color=NAVY,
                                       label="price measurement"),
                        Line2D([0], [0], color=GREY, lw=1.4, ls="--",
                               label="0.50 = coin flip")],
               frameon=False, loc="lower right")
fig.suptitle("Each measurement alone, as a detector (AUROC — higher is "
             "better)", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "S4a_feature_auroc.png")
plt.show()

# %%
# S4b — the ablation: drop one measurement, refit the adopted model,
# measure the accuracy lost (final fold: fit < newest year, test on it)
ymax = int(cand_px["year"].max())
tr_f = cand_px[cand_px["year"] < ymax]
te_f = cand_px[cand_px["year"] == ymax]
PLAIN = {r["column"]: r["feature"] for r in BANK_BOARD}


def fold_ap(label, feats):
    sc = mld.make_ens_fit(label)(tr_f, te_f, feats)
    return average_precision_score(te_f[label], sc)


abl = {}
for label, head in (("y_top", "GET OUT"), ("y_onset", "GET IN")):
    t1 = time.time()
    full = fold_ap(label, mld.DESK_ML_BANK)
    rows = []
    for f0 in mld.DESK_ML_BANK:
        rest = [f for f in mld.DESK_ML_BANK if f != f0]
        rows.append({"column": f0, "feature": PLAIN.get(f0, f0),
                     "delta": full - fold_ap(label, rest)})
    abl[head] = (full, pd.DataFrame(rows))
    print(f"{head}: full-model AP {full:.3f} | ablation "
          f"{time.time()-t1:.0f}s")

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)
sort_order = abl["GET OUT"][1].sort_values("delta")["feature"]
for ax, head in ((axes[0], "GET OUT"), (axes[1], "GET IN")):
    full, d = abl[head]
    d = d.set_index("feature").loc[sort_order].reset_index()
    cs = [NAVY if c in ("price_runup", "price_ret21") else BLUE
          for c in d["column"]]
    ax.barh(d["feature"], d["delta"], color=cs, height=0.62)
    ax.axvline(0, color=GREY, lw=1.2)
    for i, v in enumerate(d["delta"]):
        ax.text(v + (0.001 if v >= 0 else -0.001), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=10, color=NAVY)
    ax.set_title(f"{head} — accuracy lost when removed "
                 f"(model AP with all 11: {full:.2f})", fontsize=12.5)
    ax.set_xlabel("drop in AP when this measurement is removed")
    despine(ax)
fig.suptitle("The ablation: what the adopted model actually relies on",
             fontsize=16, fontweight="bold", y=1.02)
fig.text(0.01, -0.04,
         "Bars to the right = the model gets worse without that measurement. The price pair and the attention block carry the most;\n"
         "near-zero bars are measurements whose information the others already cover (kept: they cost nothing and add robustness).",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S4b_ablation.png")
plt.show()

# %% [markdown]
# ---
# # Q5 — Why this particular model?

# %%
# S5 — the tournament, one bar per contestant (AP lift: how many times
# better than guessing, each on its own candidate frame)
MODELS = [("rules", "hand rules\n(old system)"), ("logit", "logistic"),
          ("gbm", "monotone\nGBM"), ("mlp", "neural net"),
          ("ens", "ENSEMBLE\nlogit + GBM")]
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
for ax, head, ttl in ((axes[0], "get_out", "Calling the TOP (GET OUT)"),
                      (axes[1], "get_in", "Calling the START (GET IN)")):
    lifts = [RES[head][m]["ap"] / RES[head][m]["ap_baseline"]
             for m, _ in MODELS]
    cs = [GREY if m == "rules" else (BLUE if m == "ens" else SKY)
          for m, _ in MODELS]
    bars = ax.bar(range(len(MODELS)), lifts, 0.6, color=cs)
    ax.axhline(1.0, color=NAVY, lw=1.2, ls="--")
    ax.text(-0.42, 1.06, "1.0× = no better than guessing",
            fontsize=10, color=NAVY)
    for i, v in enumerate(lifts):
        ax.text(i, v + 0.05, f"{v:.2f}×", ha="center", fontsize=12,
                fontweight="bold", color=NAVY)
    wi = [m for m, _ in MODELS].index("ens")
    ax.text(wi, lifts[wi] + 0.32, "ADOPTED", ha="center", fontsize=11,
            fontweight="bold", color=BLUE)
    ax.set_xticks(range(len(MODELS)), [l for _, l in MODELS], fontsize=11)
    ax.set_title(ttl, fontsize=13)
    ax.set_ylabel("times better than guessing (AP lift)")
    despine(ax)
fig.suptitle("The tournament: every approach, same data, same "
             "walk-forward rules", fontsize=16, fontweight="bold", y=1.04)
fig.text(0.01, -0.06,
         "Selection rule (written before computing): ONE model family for both calls; highest combined lift wins; ties broken by AUROC.\n"
         "The neural net does NOT beat the ensemble — with ~500 episodes there isn't enough data for a deeper model to earn its opacity.\n"
         "The ensemble also posts the best hit-rate-per-false-alarm of any learner, and both of its members are individually readable.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S5_why_this_model.png")
plt.show()

# %% [markdown]
# **IF ASKED — "why does the old system's raw precision look high?"**
# Its gates pre-filter the candidate frame to days where ~half are
# already inside an episode window (base rate ~0.48 vs 0.10 open). Raw
# AP rewards the doorman, not the judge; lift is the comparable number,
# and on lift the old rules are 1.1–1.2× — barely above chance.
#
# **Deep-dive pointers:** notebook 03 §SS2.0 executes the full training
# recipe step by step (the printed logit formula, the rank-average
# arithmetic, the F1 curve the cut is read from); 03 §SS1 is the formal
# head-to-head vs the previous rules; 04 §SS.1 is the plain-language
# report card.
#

# %% [markdown]
# ---
# # Q5.5 — Inside the model: the weights, and how they become a call
#
# §5 says *which* model won. This section opens it: **the fitted
# weights themselves**, what the two members disagree about, and then
# one real day carried end-to-end — feature values → weighted sum →
# probability → rank → ensemble → the call. Nothing here is a
# description of the code; it is the code's own fitted objects,
# printed.

# %%
# fit the LIVE model once (all full years before the newest - the same
# fit the pipeline freezes), then read it two ways
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.inspection import permutation_importance         # noqa: E402
from analytics.ml_detector import _balanced_weights           # noqa: E402

_ymax5 = int(cand_px["year"].max())
_tr5 = cand_px[cand_px["year"] < _ymax5]
_te5 = cand_px[cand_px["year"] == _ymax5]
BANK = mld.DESK_ML_BANK
LBL = {f: mld.ML_BANK_LABELS.get(f, f) for f in BANK}

_fits = {}
for _label5, _head5 in (("y_top", "GET OUT"), ("y_onset", "GET IN")):
    _lg5 = LogisticRegression(class_weight="balanced", max_iter=2000)
    _lg5.fit(_tr5[BANK], _tr5[_label5])
    _gb5 = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, max_iter=200,
        monotonic_cst=[1] * len(BANK), random_state=0)
    _gb5.fit(_tr5[BANK], _tr5[_label5],
             sample_weight=_balanced_weights(_tr5[_label5].values))
    _pi5 = permutation_importance(
        _gb5, _te5[BANK], _te5[_label5], n_repeats=5, random_state=0,
        scoring="average_precision")
    _fits[_head5] = {"logit": _lg5, "gbm": _gb5,
                     "imp": dict(zip(BANK, _pi5.importances_mean))}
    print(f"{_head5}: fitted on {len(_tr5):,} days from years "
          f"{int(_tr5.year.min())}-{int(_tr5.year.max())}, "
          f"tested on {_ymax5}")

print()
print("THE GET OUT MODEL, WRITTEN OUT IN FULL:")
print("  P(a top is near) = sigmoid(")
_lg_out = _fits["GET OUT"]["logit"]
for _f5, _w5 in sorted(zip(BANK, _lg_out.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"      {_w5:+6.2f} × {LBL[_f5]}")
print(f"      {_lg_out.intercept_[0]:+6.2f}   )")

# %%
# S5b — the fitted weights, both reads, on one slide
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
_w_out = dict(zip(BANK, _fits["GET OUT"]["logit"].coef_[0]))
_order5 = sorted(BANK, key=lambda f: _w_out[f])
_names5 = [LBL[f] for f in _order5]

_vals = [_w_out[f] for f in _order5]
_cols = [BLUE if v >= 0 else GREY for v in _vals]
axes[0].barh(_names5, _vals, color=_cols, height=0.62)
axes[0].axvline(0, color=NAVY, lw=1.2)
for i, v in enumerate(_vals):
    axes[0].text(v + (0.06 if v >= 0 else -0.06), i, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=10, color=NAVY)
axes[0].set_title("MEMBER 1 — the logistic regression\n"
                  "one weight per measurement (+ = raises top risk)",
                  fontsize=12.5)
axes[0].set_xlabel("fitted weight (all features on the same 0–1 scale)")
despine(axes[0])

_imp = _fits["GET OUT"]["imp"]
axes[1].barh(_names5, [max(_imp[f], 0) for f in _order5], color=NAVY,
             height=0.62)
for i, f in enumerate(_order5):
    axes[1].text(max(_imp[f], 0) + 0.0007, i, f"{_imp[f]:.3f}",
                 va="center", fontsize=10, color=NAVY)
axes[1].set_title("MEMBER 2 — the monotone GBM\n"
                  "how much accuracy is lost if you shuffle it",
                  fontsize=12.5)
axes[1].set_xlabel("drop in average precision when shuffled")
despine(axes[1])
fig.suptitle("What the GET OUT model actually weighs — two independent "
             "reads of the same fit", fontsize=16, fontweight="bold",
             y=1.02)
fig.text(0.01, -0.05,
         "LEFT is the model itself: the logistic member IS this list of numbers, and a positive weight means 'more of this,\n"
         "more top risk'. RIGHT is a behavioural test on days the model never saw: shuffle one measurement and watch accuracy\n"
         "fall. They agree on the headline — the price run-up and the attention block carry the call, the mood block confirms it.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S5b_the_weights.png")
plt.show()

# %%
# S5c — ONE REAL DAY, carried end to end: values -> contributions ->
# log-odds -> probability -> rank -> ensemble -> the call
_case_name = "gold_metals"
_wf_out5 = WF[(WF["head"] == "GET OUT") & (WF["name"] == _case_name)]
_alert_day = None
_shape_alerts = SHAPE["adopted"]["standard"]["alerts"].get(_case_name, {})
if _shape_alerts.get("out"):
    _alert_day = pd.Timestamp(_shape_alerts["out"][-1])
if _alert_day is None or _alert_day not in set(_wf_out5["date"]):
    _alert_day = _wf_out5.sort_values("score")["date"].iloc[-1]

_row5 = cand_px[(cand_px["name"] == _case_name)
                & (cand_px["date"] == _alert_day)]
if _row5.empty:                    # fall back to the nearest scored day
    _cn = cand_px[cand_px["name"] == _case_name].copy()
    _cn["gap"] = (_cn["date"] - _alert_day).abs()
    _row5 = _cn.sort_values("gap").head(1)
    _alert_day = pd.Timestamp(_row5["date"].iloc[0])
_x5 = _row5[BANK].iloc[0]

_contrib = {f: float(_w_out[f]) * float(_x5[f]) for f in BANK}
_b0 = float(_fits["GET OUT"]["logit"].intercept_[0])
_logodds = _b0 + sum(_contrib.values())
_p_lg = 1 / (1 + np.exp(-_logodds))
_p_gb = float(_fits["GET OUT"]["gbm"].predict_proba(_row5[BANK])[0, 1])

_day_all = WF[(WF["head"] == "GET OUT") & (WF["date"] == _alert_day)]
_r_lg = float((_day_all["logit"] <= _day_all.loc[
    _day_all["name"] == _case_name, "logit"].iloc[0]).mean()) \
    if len(_day_all) and (_day_all["name"] == _case_name).any() else np.nan
_ens_row = _wf_out5[_wf_out5["date"] == _alert_day]
_ens_score = float(_ens_row["score"].iloc[0]) if len(_ens_row) else np.nan
_cut5 = THR["GET OUT"].get(int(pd.Timestamp(_alert_day).year), np.nan)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8),
                         width_ratios=[1.35, 1])

# left: the contribution waterfall
_ord_c = sorted(BANK, key=lambda f: _contrib[f])
axes[0].barh([LBL[f] for f in _ord_c], [_contrib[f] for f in _ord_c],
             color=[BLUE if _contrib[f] >= 0 else GREY for f in _ord_c],
             height=0.62)
axes[0].axvline(0, color=NAVY, lw=1.2)
for i, f in enumerate(_ord_c):
    v = _contrib[f]
    axes[0].text(v + (0.03 if v >= 0 else -0.03), i,
                 f"{_x5[f]:.2f} × {_w_out[f]:+.2f} = {v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=9, color=NAVY)
axes[0].set_title(f"Gold on {pd.Timestamp(_alert_day):%d %b %Y} — each "
                  "measurement's contribution\n(its value × its weight)",
                  fontsize=12.5)
axes[0].set_xlabel("contribution to the log-odds")
axes[0].margins(x=0.28)
despine(axes[0])

# right: the ladder from log-odds to the call
axes[1].axis("off")
_steps = [
    ("1  add them up", f"{_b0:+.2f} (base) + contributions  =  "
     f"{_logodds:+.2f} log-odds"),
    ("2  squash to a probability",
     f"sigmoid({_logodds:+.2f})  =  {_p_lg:.3f}"),
    ("3  the other member", f"monotone GBM says  {_p_gb:.3f}"),
    ("4  rank each across the market",
     "both probabilities become percentile ranks that day"),
    ("5  average the two ranks",
     f"the ensemble score  =  {_ens_score:.3f}"
     if np.isfinite(_ens_score) else "the ensemble score"),
    ("6  compare with the frozen cut",
     f"{_ens_score:.3f}  vs  {_cut5:.3f}   →   "
     f"{'FIRE — GET OUT' if np.isfinite(_ens_score) and _ens_score >= _cut5 else 'no call'}"
     if np.isfinite(_ens_score) and np.isfinite(_cut5) else "the cut"),
]
_y5 = 0.95
for _k, (_t1, _t2) in enumerate(_steps):
    _fc = "#1F6F5C" if _k == len(_steps) - 1 else "#F2F5FA"
    _tc = "white" if _k == len(_steps) - 1 else NAVY
    axes[1].add_patch(mpatches.FancyBboxPatch(
        (0.01, _y5 - 0.125), 0.97, 0.115,
        boxstyle="round,pad=0.006", facecolor=_fc, edgecolor="none",
        transform=axes[1].transAxes))
    axes[1].text(0.04, _y5 - 0.045, _t1, fontsize=11, fontweight="bold",
                 color=_tc, va="center", transform=axes[1].transAxes)
    axes[1].text(0.04, _y5 - 0.093, _t2, fontsize=10.5, color=_tc,
                 va="center", transform=axes[1].transAxes)
    if _k < len(_steps) - 1:
        axes[1].annotate("", (0.5, _y5 - 0.135), (0.5, _y5 - 0.125),
                         xycoords=axes[1].transAxes,
                         textcoords=axes[1].transAxes,
                         arrowprops=dict(arrowstyle="-|>", lw=1.3,
                                         color=GREY))
    _y5 -= 0.155
axes[1].set_title("…and how that becomes a call", fontsize=12.5,
                  loc="left")
fig.suptitle("How one day becomes one signal — the arithmetic, in full",
             fontsize=16, fontweight="bold", y=1.0)
fig.text(0.01, -0.04,
         "Read left to right. Every measurement is a percentile (0–1) against gold's OWN history, so a contribution is just\n"
         "'how extreme is this, times how much the model cares'. The log-odds are turned into a probability, the second member\n"
         "votes independently, the two are averaged in RANK space, and only then is the frozen cut applied. No step is hidden.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S5c_one_day_one_signal.png")
plt.show()

# %% [markdown]
# **SO WHAT**
#
# * **The model is a readable object.** Member 1 is eleven signed
#   numbers and an intercept — you can evaluate it on paper. Member 2 is
#   300 shallow trees, but every feature is constrained to push one way,
#   so it cannot contradict member 1's direction; the only thing it adds
#   is *interactions* (how loud AND how fast, together).
# * **A call is six steps, none of them discretionary:** value × weight,
#   sum, sigmoid, rank, average, compare with a cut chosen on past years.
#   The only quantity a human ever chose is which measurements to offer.
# * **Both reads agree on what matters**, which is the honest test of an
#   explanation: if the linear weights and the shuffle test disagreed
#   about a feature, we could not claim to know why the model fires.
#
# **IF ASKED — "why average RANKS instead of probabilities?"** Because
# the two members are calibrated differently: the GBM's probabilities
# cluster near the extremes and would dominate a plain average. Ranking
# each member across the market on the day puts them on one scale and
# makes the ensemble a vote about ORDER — which is exactly what an
# alert threshold consumes.

# %% [markdown]
# ---
# # Q6 — Why these thresholds? (and the final feature set)
#
# The walk-forward scores loaded at the top of this notebook are what
# every panel below reads: fit on years < Y, score year Y,
# rank-average the two members within the year.

# %%
# S6a — where the trigger sits: the two score distributions and the cut
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
for ax, head in ((axes[0], "GET OUT"), (axes[1], "GET IN")):
    d = WF[WF["head"] == head]
    bins = np.linspace(0, 1, 41)
    ax.hist(d.loc[d.y == 0, "score"], bins=bins, density=True,
            color=GREY, alpha=0.55, label="ordinary days")
    ax.hist(d.loc[d.y == 1, "score"], bins=bins, density=True,
            color=BLUE, alpha=0.70, label="days inside a real episode "
            "window")
    thr = THR[head][max(THR[head])]
    ax.axvline(thr, color=NAVY, lw=2.2, ls="--")
    ax.text(thr - 0.02, ax.get_ylim()[1] * 0.97,
            f"the trigger ({thr:.2f})\nlearned from past years only",
            ha="right", va="top", fontsize=11, fontweight="bold",
            color=NAVY)
    ax.set_title(head, fontsize=13)
    ax.set_xlabel("the model's daily score (rank across the market, "
                  "0 → 1)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left")
    despine(ax)
fig.suptitle("Why the threshold is where it is: episode days pile up at "
             "the top of the ranking", fontsize=16, fontweight="bold",
             y=1.04)
fig.text(0.01, -0.06,
         "Grey: the score on ordinary days. Blue: the score on days that were genuinely inside an episode window (out-of-sample).\n"
         "The trigger is NOT hand-picked: each year it is set where precision and recall balance (max F1) on the years before —\n"
         "then frozen and judged on the unseen year. Fired on upward crossings only, one call per name per 21 days.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S6a_score_distributions.png")
plt.show()

# %%
# S6b — the final feature set holds together (Spearman) + the learned
# cut has been stable year after year
plain_short = {
    "e1": "attention level", "e3": "attn chg 1m",
    "influx_speed": "attn chg 2w", "attention_accel": "week vs month",
    "hype_ratio": "week vs normal", "attention_convexity": "accelerating",
    "bull_level": "bullish level", "bull_persist": "bullish persist",
    "bull_inflection": "mood turning", "price_runup": "price run-up",
    "price_ret21": "price ret 1m"}
corr = cand_px[mld.DESK_ML_BANK].corr(method="spearman")
cmap = LinearSegmentedColormap.from_list("bg", [GREY, "#FFFFFF", BLUE])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.4),
                               width_ratios=[1.15, 1])
im = ax1.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax1.set_xticks(range(len(corr)),
               [plain_short[c] for c in corr.columns], rotation=45,
               ha="right", fontsize=9.5)
ax1.set_yticks(range(len(corr)),
               [plain_short[c] for c in corr.columns], fontsize=9.5)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i, j]
        ax1.text(j, i, f"{v:.1f}".replace("0.", "."), ha="center",
                 va="center", fontsize=7.5,
                 color="white" if abs(v) > 0.6 else NAVY)
ax1.set_title("The 11 selected features: related but not redundant\n"
              "(Spearman rank correlation)", fontsize=12.5)
ax1.grid(False)
fig.colorbar(im, ax=ax1, shrink=0.8)

for head, c, mk in (("GET OUT", NAVY, "o"), ("GET IN", BLUE, "s")):
    yrs = sorted(THR[head])
    ax2.plot(yrs, [THR[head][y] for y in yrs], color=c, lw=2.0,
             marker=mk, ms=7, label=head)
ax2.set_ylim(0.85, 1.0)
ax2.set_title("The learned trigger, year by year:\nre-fitted annually, "
              "never hand-moved", fontsize=12.5)
ax2.set_ylabel("probability-rank cut")
ax2.legend(frameon=False, loc="lower left")
despine(ax2)
fig.text(0.01, -0.05,
         "Left: the attention measurements correlate with each other (by design — they view one phenomenon at different speeds) but the\n"
         "mood and price blocks bring genuinely new information. Right: the data has kept choosing nearly the same cut for nine years —\n"
         "the trigger is a stable property of the phenomenon, not a fragile tuning.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S6b_spearman_and_cuts.png")
plt.show()

# %% [markdown]
# ---
# # Q6.5 — One clear call per boom: the alert shape (desk orders 2026-08-09)
#
# **The complaints, verbatim:** GET OUT calls "all the way up a price
# graph"; "we CANNOT have a get in and a get out so close together";
# "I don't want many many get ins — a clear GET IN at the start of the
# boom and a GET OUT quite near the top."
#
# **Two rounds of evidence, both frozen.** Round 1
# (`conditioning_sweep.json`) tested score smoothing (the desk's own
# suggestion — REJECTED: it cost 10 points of GET OUT capture), a
# stricter cut, and a crossing trigger. Round 2
# (`alert_shape_sweep.json`) found the deeper cause: the two heads'
# scores are near-twins, so they cross together — no trigger tweak can
# fix that. The adopted design makes the heads STRUCTURALLY disjoint by
# reusing the episode definition itself as a phase gate:
#
# * **GET OUT may only fire after the boom** — price ≥ the ground
#   truth's own 120d boom bar (20% themes / 40% singles). Re-arms at
#   the cut.
# * **GET IN may only fire before the boom completes.** Re-arms only
#   when the crowd fully cools (the train-median score).
# * **One call per name per quarter per side** (63d spacing), and **no
#   GET IN stands within 21 days of a GET OUT, either direction** (GET
#   OUT, the risk signal, is never suppressed).
#
# No new constants beyond one tolerance: the bar, the windows, the
# median and the 21d separation are all reused quantities; 63d = three
# cooldowns.
#
# **Round 3 — "closer to the actual peak" (2026-08-10,
# `alert_shape_sweep.json → timing_sweep`):** seven timing variants
# were tested, including the desk's own price-escalation idea ("+25%
# above the last flag to re-fire" — REJECTED: busts reset prices
# between episodes, so it cost 14–20 captures), a crest-call (closest
# to the peak but precision collapses to 0.38), and a near-high gate
# (measured best: capture 63→65, FA 55→47, precision 0.53→0.58). The
# desk reviewed the round and **kept the round-2 shape unchanged** —
# the measurements are kept as the record of what was tried.

# %%
_sh_rows = []
for _k, _r in SHAPE["results"].items():
    for _head in ("GET IN", "GET OUT"):
        _h = _r[_head]
        _sh_rows.append({
            "config": _k, "head": _head,
            "caught": f"{_h['captured']}/{_h['detectable']}",
            "FA": _h["false_alarms"],
            "precision": _h["precision"],
            "calls per boom": _h["calls_per_alerted_episode"],
            "alerts": _h["n_alerts"]})
display(pd.DataFrame(_sh_rows).set_index(["config", "head"]).head(24))

AD = SHAPE["adopted"]
print("ADOPTED (walk-forward record of what now ships):")
for _lbl in ("standard", "strict"):
    for _head in ("GET IN", "GET OUT"):
        _r = AD[_lbl][_head]
        print(f"  {_lbl:8s} {_head:8s} caught {_r['captured']}/"
              f"{_r['detectable']} ({_r['capture_rate']:.0%}) | "
              f"FA {_r['false_alarms']} ({_r['fa_per_iy']}/instr-yr) | "
              f"precision {_r['precision']:.0%} | alerts "
              f"{_r['n_alerts']} | median lead {_r['median_lead_days']}d")
print("Config:", json.dumps(AD["config"], indent=1))

# %%
# S6c — before / after, on the name with the worst repeated-fire problem
def _level_alerts(head, name):
    """The ORIGINAL level trigger (the 'before' picture)."""
    d = WF[(WF["head"] == head) & (WF["name"] == name)]
    out = []
    for y, g in d.groupby("test_year"):
        thr = THR[head].get(int(y))
        if thr is None:
            continue
        g = g.sort_values("date")
        al = _alerts_int(_day_ints(g["date"]),
                         g["score"].to_numpy(float), thr)
        out += list(np.asarray(al, dtype="datetime64[D]"))
    return pd.DatetimeIndex(out)


def _adopted_alerts(head, name, setting="standard"):
    """The SHIPPED alert set (frozen in alert_shape_sweep.json)."""
    a = SHAPE["adopted"][setting]["alerts"].get(name, {})
    key = "in" if head == "GET IN" else "out"
    return pd.DatetimeIndex([pd.Timestamp(x) for x in a.get(key, [])])


_worst = max(WF["name"].unique(),
             key=lambda n: len(_level_alerts("GET OUT", n)))
_sym_w = sym_by[_worst]
_pxw = pxmap[_sym_w].dropna()
_a_old = _level_alerts("GET OUT", _worst)
_a_new = _adopted_alerts("GET OUT", _worst, "standard")
_a_str = _adopted_alerts("GET OUT", _worst, "strict")
_w0 = max(_a_old, key=lambda a: sum(
    1 for b in _a_old if a <= b <= a + pd.Timedelta(days=540)))
_w1 = _w0 + pd.Timedelta(days=540 + 60)
_w0 = _w0 - pd.Timedelta(days=60)
_pxw = _pxw.loc[_w0:_w1]

fig, axes = plt.subplots(3, 1, figsize=(12.8, 7.6), sharex=True)


def _inwin(al):
    return [a for a in al if _pxw.index.min() <= a <= _pxw.index.max()]


for ax, al, ttl, c in (
        (axes[0], _a_old, f"BEFORE — level trigger: {len(_inwin(_a_old))}"
         " GET OUT calls in this window, re-firing all the way up the "
         "rally", GREY),
        (axes[1], _a_new, f"AFTER — shaped calls (Standard): "
         f"{len(_inwin(_a_new))} calls, only after the boom bar", BLUE),
        (axes[2], _a_str, f"AFTER — shaped calls (Strict): "
         f"{len(_inwin(_a_str))} calls", NAVY)):
    ax.plot(_pxw.index, _pxw.values, color=NAVY, lw=1.8)
    if _pxw.max() / max(_pxw.min(), 1e-9) > 4:
        ax.set_yscale("log")
    for r in episodes[(episodes.name == _worst)].itertuples():
        if r.peak >= _pxw.index.min() and r.trough <= _pxw.index.max():
            ax.axvspan(max(r.trough, _pxw.index.min()),
                       min(r.peak, _pxw.index.max()), color=SKY,
                       alpha=0.28, lw=0)
    _al_w = _inwin(al)
    ax.plot(_al_w, [_pxw.asof(a) for a in _al_w], "v", color=c, ms=12,
            mec="white", mew=1.2, zorder=5)
    ax.set_xlim(_pxw.index.min(), _pxw.index.max())
    ax.set_title(ttl, fontsize=12.5, loc="left")
    despine(ax)
fig.suptitle(f"One clear call per boom — {_worst} ({_sym_w})",
             fontsize=16, fontweight="bold", y=1.0)
fig.text(0.01, -0.03,
         "BEFORE: the level trigger re-fired every cooldown for as long as the score stayed hot. AFTER: a GET OUT exists only once\n"
         "the name has actually boomed (the ground truth's own 120d bar), re-arms only below the cut, and fires at most once a\n"
         "quarter — and a GET IN can never stand within 21 days of a GET OUT, in either direction.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S6c_flip_flop_fix.png")
plt.show()

# %%
# S6d — the desk's four named cases, exactly as the dashboard shows them
# (the SHIPPED store's replay - the frozen final cut, shaped)
_desk_ds = pd.read_parquet(ROOT / "data" / "processed" /
                           "euphoria_desk.parquet")
_desk_ds["date"] = pd.to_datetime(_desk_ds["date"])
CASES4 = [("Gold (GLD) — the 2026 rush", "gold_metals",
           "2025-04-01", None),
          ("Meme stocks (ARKK) — the 2020–21 mania", "meme_stocks",
           "2020-01-01", "2022-12-31"),
          ("Semiconductors (SMH)", "semiconductors", "2024-06-01", None),
          ("AI megacaps (MSFT basket)", "ai_megacap",
           "2021-06-01", "2023-12-31")]
fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6))
for ax, (title4, name4, w40, w41) in zip(axes.flat, CASES4):
    px4 = pxmap[sym_by[name4]].dropna()
    px4 = px4.loc[w40:w41] if w41 else px4.loc[w40:]
    g4 = _desk_ds[_desk_ds["name"] == name4]
    ins4 = [x for x in g4.loc[g4["get_in"], "date"]
            if px4.index.min() <= x <= px4.index.max()]
    outs4 = [x for x in g4.loc[g4["get_out"], "date"]
             if px4.index.min() <= x <= px4.index.max()]
    ax.plot(px4.index, px4.values, color=NAVY, lw=1.8)
    if px4.max() / max(px4.min(), 1e-9) > 4:
        ax.set_yscale("log")
    for r in episodes[episodes.name == name4].itertuples():
        if r.peak >= px4.index.min() and r.trough <= px4.index.max():
            ax.axvspan(max(r.trough, px4.index.min()),
                       min(r.peak, px4.index.max()), color=SKY,
                       alpha=0.3, lw=0)
    ax.plot(ins4, [px4.asof(x) for x in ins4], "^", color=BLUE, ms=14,
            mec="white", mew=1.4, zorder=5, label="GET IN")
    ax.plot(outs4, [px4.asof(x) for x in outs4], "v", color=NAVY, ms=14,
            mec="white", mew=1.4, zorder=5, label="GET OUT")
    ax.set_xlim(px4.index.min(), px4.index.max())
    ax.set_title(f"{title4} — {len(ins4)} IN / {len(outs4)} OUT",
                 loc="left", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    despine(ax)
    ax.tick_params(axis="x", rotation=25)
fig.suptitle("The shaped calls on the desk's four cases — as the "
             "dashboard now shows them (Standard setting)",
             fontsize=15, fontweight="bold", y=1.0)
fig.text(0.01, -0.035,
         "Shaded = real episode run-ups (the price-only ground truth). GET IN can only fire before a name's boom completes;\n"
         "GET OUT only after (the ground truth's own 120d bar), re-armed, at most one call per name per quarter per side,\n"
         "and never a GET IN within 21 days of a GET OUT. Gold: one OUT, 7 days before the 29 Jan 2026 peak.",
         fontsize=10.5, color=GREY, va="top")
fig.tight_layout()
save(fig, "S6d_four_cases.png")
plt.show()

# %% [markdown]
# ---
# # Q7 — How do the pieces combine into a GET IN or a GET OUT?
#
# The chain on one chart: the two members each rank every (name, day) in
# the market; the ensemble is their **equal-weight average of rankings**
# (no tuned mixing weight); a SHAPED crossing of the learned cut fires
# the call — GET IN only before the boom completes, GET OUT only after,
# one call per name per quarter per side. Shown for the two cases the
# desk asks about first: gold and GME.

# %%
def slide_case(disp, name, sym, w0, w1, fname):
    d_out = WF[(WF["head"] == "GET OUT") & (WF["name"] == name)
               ].set_index("date").sort_index().loc[w0:w1]
    d_in = WF[(WF["head"] == "GET IN") & (WF["name"] == name)
              ].set_index("date").sort_index().loc[w0:w1]
    px = pxmap[sym].dropna().loc[w0:w1]
    a_out = [a for a in _adopted_alerts("GET OUT", name)
             if w0 <= str(a.date()) <= w1]
    a_in = [a for a in _adopted_alerts("GET IN", name)
            if w0 <= str(a.date()) <= w1]

    fig, axes = plt.subplots(4, 1, figsize=(12.8, 8.6), sharex=True,
                             height_ratios=[2.1, 1, 1, 1])
    ax = axes[0]
    ax.plot(px.index, px.values, color=NAVY, lw=2.0)
    if px.max() / max(px.min(), 1e-9) > 4:
        ax.set_yscale("log")
    _x0, _x1 = px.index.min(), px.index.max()
    for r in episodes[(episodes.name == name) & (episodes.peak >= w0)
                      & (episodes.peak <= w1)].itertuples():
        ax.axvspan(max(r.trough, _x0), min(r.peak, _x1), color=SKY,
                   alpha=0.3, lw=0)
    ax.set_xlim(_x0, _x1)
    ax.plot(a_in, [px.asof(a) for a in a_in], "^", color=BLUE, ms=13,
            mec="white", mew=1.2, label="GET IN call", zorder=5)
    ax.plot(a_out, [px.asof(a) for a in a_out], "v", color=NAVY, ms=13,
            mec="white", mew=1.2, label="GET OUT call", zorder=5)
    ax.set_title(f"{disp} — price, with the model's calls "
                 "(shaded = real episode run-ups)", fontsize=12.5,
                 loc="left")
    ax.legend(frameon=False, loc="upper left")
    despine(ax)

    ax = axes[1]
    ax.plot(d_out.index, d_out["logit_rank"], color=GREY, lw=1.6,
            label="member 1: logistic (rank)")
    ax.plot(d_out.index, d_out["gbm_rank"], color=SKY, lw=1.6,
            label="member 2: monotone GBM (rank)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Step 1 — each member ranks this name against the whole "
                 "market, every day (GET OUT head)", fontsize=12.5,
                 loc="left")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    despine(ax)

    for ax, d, head, c, al in ((axes[2], d_out, "GET OUT", NAVY, a_out),
                               (axes[3], d_in, "GET IN", BLUE, a_in)):
        ax.plot(d.index, d["score"], color=c, lw=2.0,
                label="ensemble = the simple average of the two ranks")
        thr_line = d["test_year"].map(lambda y: THR[head].get(int(y)))
        ax.plot(d.index, thr_line, color=GREY, lw=1.6, ls="--",
                label="the learned trigger")
        ax.plot(al, [1.02] * len(al), "v" if head == "GET OUT" else "^",
                color=c, ms=10, clip_on=False)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Step 2 — combine equally; a SHAPED crossing of "
                     f"the trigger fires the call: {head}",
                     fontsize=12.5, loc="left")
        ax.legend(frameon=False, loc="lower left", ncols=2)
        despine(ax)
    fig.suptitle(f"How a call is made — {disp}", fontsize=16,
                 fontweight="bold", y=1.005)
    fig.text(0.01, -0.03,
             "Everything is out-of-sample: the members, the ensemble and the trigger that fire in year Y saw only years before Y.\n"
             "The combination is an EQUAL-WEIGHT average of the two members' rankings — no extra fitted weight to defend.",
             fontsize=11, color=GREY, va="top")
    fig.tight_layout()
    save(fig, fname)
    plt.show()


# gold only enters the model's candidate frame in 2026 (the theme is
# new) - window the case to where the scores exist, so every panel fills
slide_case("Gold (GLD)", "gold_metals", "GLD", "2026-01-01", "2026-05-01",
           "S7a_how_a_call_is_made_gold.png")

# %%
slide_case("GME", "GME", "GME", "2020-09-01", "2021-07-31",
           "S7b_how_a_call_is_made_gme.png")

# %% [markdown]
# ---
# # Q8 — How well does it perform, and is it better than what we ran
# before?
#
# The comparison rules first, so the answer cannot be an artifact of
# framing: same frozen episode truth, same walk-forward machinery, each
# system on its own candidacy (which is why score quality is compared on
# **AP lift**, not raw AP). "The model" below is the SHIPPED
# configuration — the ensemble with the SHAPED alerts of §6.5
# (phase-gated, re-armed, quarterly-spaced, separated), Standard cut.

# %%
# the shipped configuration's walk-forward record (the ADOPTED shaped
# alerts frozen in alert_shape_sweep.json)
SHIP = {h: SHAPE["adopted"]["standard"][h] for h in ("GET OUT", "GET IN")}
STRICT = {h: SHAPE["adopted"]["strict"][h] for h in ("GET OUT", "GET IN")}
RULES = {"GET OUT": RES["get_out"]["rules"], "GET IN": RES["get_in"]["rules"]}

_hh_rows = []
for _head in ("GET OUT", "GET IN"):
    for _sys, _r in (("previous rules", RULES[_head]),
                     ("MODEL — Standard", SHIP[_head]),
                     ("MODEL — Strict", STRICT[_head])):
        _hh_rows.append({
            "head": _head, "system": _sys,
            "caught": f"{_r['captured']}/{_r['detectable']} "
                      f"({_r['capture_rate']:.0%})",
            "FA/instr-yr": round(_r["fa_per_iy"], 2),
            "precision": round(_r["precision"], 2),
        })
display(pd.DataFrame(_hh_rows).set_index(["head", "system"]))
for _head, _hk in (("GET OUT", "get_out"), ("GET IN", "get_in")):
    _ru, _mlr = RULES[_head], SHIP[_head]
    _lift = RES[_hk]["ens"]["ap"] / RES[_hk]["ens"]["ap_baseline"]
    _rl = _ru["ap"] / _ru["ap_baseline"]
    print(f"{_head}: {_mlr['capture_rate']/_ru['capture_rate']:.1f}x the "
          f"episodes caught ({_mlr['capture_rate']:.0%} vs "
          f"{_ru['capture_rate']:.0%}), score quality {_lift:.1f}x vs "
          f"{_rl:.1f}x better-than-guessing (rules: barely above chance)")

# %%
# S8a — the operational record vs the old system
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
heads = ["GET OUT", "GET IN"]
for ax, metric, ttl, fmt in (
        (axes[0], "capture_rate", "episodes caught\n(share of all "
         "catchable episodes)", "{:.0%}"),
        (axes[1], "precision", "calls that were right\n(precision)",
         "{:.0%}")):
    xs = np.arange(2)
    rv = [RULES[h][metric] for h in heads]
    mv = [SHIP[h][metric] for h in heads]
    b1 = ax.bar(xs - 0.18, rv, 0.32, color=GREY, label="old rules")
    b2 = ax.bar(xs + 0.18, mv, 0.32, color=BLUE, label="the model")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    fmt.format(b.get_height()), ha="center", fontsize=12,
                    fontweight="bold", color=NAVY)
    ax.set_xticks(xs, heads, fontsize=12)
    ax.set_title(ttl, fontsize=13)
    ax.legend(frameon=False, loc="upper left")
    despine(ax)
fig.suptitle("The record, out of sample — every year judged by a model "
             "that never saw it", fontsize=16, fontweight="bold", y=1.05)
_o, _i = SHIP["GET OUT"], SHIP["GET IN"]
fig.text(0.01, -0.07,
         f"GET OUT: {_o['captured']}/{_o['detectable']} tops caught at {_o['fa_per_iy']:.2f} false alarms per name per year "
         f"(≈ one every 3 years); GET IN: {_i['captured']}/{_i['detectable']} starts at {_i['fa_per_iy']:.2f}. "
         "The old rules caught 12% / 9% under the identical test.\n"
         f"The Strict setting: {STRICT['GET OUT']['captured']}/{STRICT['GET OUT']['detectable']} and "
         f"{STRICT['GET IN']['captured']}/{STRICT['GET IN']['detectable']} at roughly HALF the false alarms "
         f"({STRICT['GET OUT']['fa_per_iy']:.2f} / {STRICT['GET IN']['fa_per_iy']:.2f} per name-year).",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S8a_the_record.png")
plt.show()

# %%
# S8b — the tape's own verdict: what price did after each call
# (the shipped alert sets and their forward returns, from the frozen
# adopted record)
_ship_alerts = {h: {n: list(_adopted_alerts(h, n))
                    for n in WF["name"].unique()}
                for h in ("GET OUT", "GET IN")}
FWD_SHIP = SHAPE["adopted"]["standard"]["forward_returns"]

H = [("fwd_5d", "1 week"), ("fwd_21d", "1 month"), ("fwd_84d", "1 quarter")]
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
for ax, head, ttl, want in ((axes[0], "GET OUT",
                             "after a GET OUT call", "want: price DOWN"),
                            (axes[1], "GET IN",
                             "after a GET IN call", "want: price UP")):
    xs = np.arange(len(H))
    rv = [RULES[head]["forward_returns"][k]["median_pct"] for k, _ in H]
    mv = [FWD_SHIP[head][k]["median_pct"] for k, _ in H]
    b1 = ax.bar(xs - 0.18, rv, 0.32, color=GREY, label="old rules")
    b2 = ax.bar(xs + 0.18, mv, 0.32, color=BLUE, label="the model")
    ax.axhline(0, color=NAVY, lw=1.2)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.12 if v >= 0 else -0.12),
                    f"{v:+.1f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=11,
                    fontweight="bold", color=NAVY)
    ax.set_xticks(xs, [t for _, t in H], fontsize=12)
    ax.set_title(f"Median price move {ttl}   ({want})", fontsize=13)
    ax.set_ylabel("median % move")
    ax.legend(frameon=False, loc="upper right")
    despine(ax)
fig.suptitle("The tape agrees with the model — and disagreed with the "
             "old rules", fontsize=16, fontweight="bold", y=1.05)
fig.text(0.01, -0.06,
         "After the model's GET OUT calls, price drifted DOWN over the following quarter (a top call doing its job). After the old\n"
         "rules' GET OUT calls, price kept RISING (+4% a quarter later) — its few calls were systematically early or wrong.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S8b_the_tape_agrees.png")
plt.show()

# %%
# S8c — the event study: the median price path around the shipped calls
HORIZON_PRE, HORIZON_POST = 30, 90


def paths_around(dates_by_name):
    paths = []
    for name, dts in dates_by_name.items():
        px = pxmap.get(sym_by.get(name))
        if px is None:
            continue
        px = px.dropna()
        for d in dts:
            w = px.loc[d - pd.Timedelta(days=HORIZON_PRE):
                       d + pd.Timedelta(days=HORIZON_POST)]
            p0 = px.asof(d)
            if pd.isna(p0) or p0 == 0 or len(w) < 30:
                continue
            rel = (w / p0) * 100
            rel.index = (rel.index - d).days
            paths.append(rel)
    if not paths:
        return None
    grid = pd.DataFrame(paths).T.sort_index()
    return grid.loc[-HORIZON_PRE:HORIZON_POST]


fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharex=True)
for ax, head, title, c in ((axes[0], "GET OUT",
                            "around a GET OUT call", NAVY),
                           (axes[1], "GET IN",
                            "around a GET IN call", BLUE)):
    grid = paths_around(_ship_alerts[head])
    med = grid.median(axis=1)
    q25, q75 = grid.quantile(0.25, axis=1), grid.quantile(0.75, axis=1)
    ax.fill_between(med.index, q25, q75, color=c, alpha=0.13, lw=0)
    ax.plot(med.index, med.values, color=c, lw=2.4,
            label=f"median of {grid.shape[1]} calls")
    ax.axvline(0, color=GREY, lw=1.2, ls="--")
    ax.axhline(100, color=GREY, lw=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("days from the call")
    ax.set_ylabel("price (call day = 100)")
    ax.legend(frameon=False, fontsize=10)
    despine(ax)
fig.suptitle("What price does around the model's calls (walk-forward, "
             "shipped configuration)", fontsize=16, fontweight="bold",
             y=1.05)
fig.text(0.01, -0.07,
         "Shaded = the middle half of outcomes. GET OUT calls land after a run-up and ahead of a topping median path — which is why\n"
         "GET OUT reads 'reduce', never 'short': the upper-quartile overshoots are real. GET IN calls confirm a crowd arriving after\n"
         "the first leg; the upside is fat-tailed — a minority of starts become the manias that pay for the rest.",
         fontsize=11, color=GREY, va="top")
fig.tight_layout()
save(fig, "S8c_event_study.png")
plt.show()

# %% [markdown]
# ---
# # The decision log (what we tested → what we decided → why)
#
# | # | We tested | We decided | Because |
# |---|---|---|---|
# | 1 | Was the data broken or the estimator? (per-source coverage audit, §P) | Fix the ESTIMATOR: ratio-of-sums + stratification + shrinkage; no imputation | 308 fake zero-days traced to pull cadence + source mix; inventing posts would poison the counts |
# | 2 | Ground-truth bars at 25/50, 20/40, 15/30 — full walk-forward per level (§1) | **20/40 boom, 12/25 bust** | Doubles episodes, keeps gold/meme/semis, accuracy improves; 15/30 makes "euphoria" = any correction |
# | 3 | 5 model families × 2 feature sets, walk-forward, pre-stated rule (§5) | **logit + monotone-GBM rank ensemble**, crowd+price bank | Highest combined AP lift (2.45× / 2.55×); best hit-rate-to-FA trade of any learner |
# | 4 | Keep the gate stack or hand gates to the model? | Gates become FEATURES; 3 numbers survive (coverage floor, cooldown, learned cut) | Same information, no per-constant defence needed; capture 12%→~41% and 9%→~55% |
# | 5 | MLP as the ceiling check | NOT adopted — reported as a null | ~500 positives cannot fund a deeper model; it never beat the ensemble |
# | 6 | Crowd-only versions of every learner | Kept as the parallel record, not shipped for the desk | The desk signals are licensed to use price; the "crowd alone" claim keeps its own honest (weaker) record |
# | 7 | START-on-top-of-END coherence | Kept as a display-layer rule under every model, in BOTH signal settings | Desk stated three times this is the error that destroys PM trust; one sentence to explain |
# | 8 | Alert shape, two rounds: smoothing / stricter cut / crossing (8 configs), then phase gates × re-arm depth × spacing (24 configs) | **Shaped calls**: GET OUT only post-boom (120d G2 bar, re-arm at cut), GET IN only pre-boom (re-arm at train-median), 63d spacing, 21d IN/OUT separation both directions; smoothing REJECTED | Desk order ("one clear call per boom; never IN and OUT together"); calls-per-boom 1.6→1.2, GET OUT precision 0.36→0.53, adjacency zero by construction; evidence `conditioning_sweep.json` + `alert_shape_sweep.json` |
#
# # Limitations, said plainly, then the PM line
#
# - **The frozen cuts are young.** Selected by train-year F-measures
#   under a ground truth adopted the same week; their real out-of-sample
#   test is the forward signal snapshots, which accumulate from now.
# - **2026 dominates the multi-source era.** StockTwits/X exist in the
#   live archive only from 2026; earlier years are effectively
#   Reddit-only, and the stratified estimator handles but cannot erase
#   that asymmetry.
# - **Median GET OUT forward moves are flat-to-negative; means are
#   positive.** Manias overshoot: a top-caller's misses keep rallying.
#   That is why the desk reads GET OUT as "reduce", never "short".
# - **Parsimony was bought with capture** (DESK DECISION 2026-08-09):
#   the un-shaped model caught 41%/55% of episodes; the shaped calls
#   catch 29%/11% — the desk chose clean, phase-correct, one-per-boom
#   calls over raw coverage, and the un-shaped record remains frozen in
#   the tournament table for comparison.
# - **Late GET INs are now suppressed by design** — a start may only be
#   called before the boom completes, so a crowd that arrives late
#   (gold 2026) yields no GET IN at all; the GET OUT still protects the
#   exit.
#
# **The PM line:** *the crowd, plus the tape, read by a two-model
# ensemble that only ever trains on the past, calls roughly three in
# ten episode tops a median of six days before the peak — two and a
# half times what the hand rules caught, at better than one false alarm
# per name per eight years — and flags fresh crowd arrivals only while
# the boom is still young; at most one call per name per quarter per
# side, never a GET IN within three weeks of a GET OUT, a Strict
# setting that thins it further, and a comparison table for every
# alternative we declined.*

# %% [markdown]
# ## The manifest — every number the slides quote, frozen in one place

# %%
manifest = {
    "figures": SAVED,
    "palette": {"navy": NAVY, "blue": BLUE, "sky": SKY, "grey": GREY},
    "winner": RES["winner"],
    "conditioning": desk_rep.get("conditioning"),
    "headline": {
        h: {"rules": {k: RULES[h].get(k) for k in
                      ("captured", "detectable", "capture_rate",
                       "precision", "fa_per_iy", "forward_returns")},
            "shipped_standard": SHIP[h],
            "shipped_strict": STRICT[h],
            "forward_returns_shipped": FWD_SHIP[h]}
        for h in ("GET OUT", "GET IN")},
    "score_quality": {hk: {"ap": RES[hk]["ens"]["ap"],
                           "ap_baseline": RES[hk]["ens"]["ap_baseline"],
                           "auroc": RES[hk]["ens"]["auroc"]}
                      for hk in ("get_out", "get_in")},
    "thresholds": {h: THR[h] for h in ("GET OUT", "GET IN")},
    "strict_thresholds_final": {
        "GET IN": desk_rep["get_in"].get("strict_threshold"),
        "GET OUT": desk_rep["get_out"].get("strict_threshold")},
    "ablation": {h: abl[h][1].to_dict(orient="records") for h in abl},
    "ablation_note": "final fold (fit < newest year, tested on it); "
                     "delta = AP(all 11) - AP(without the feature)",
    "alert_shape": SHAPE["adopted"]["config"],
}
with open(RESEARCH_DIR / "nb07_slide_pack.json", "w") as f:
    json.dump(manifest, f, indent=1, default=str)
print(f"saved nb07_slide_pack.json | {len(SAVED)} slide figures:")
for s in SAVED:
    print("  ", s)
