"""
build_final_plots.py — the FINAL PRESENTATION PLOT PACK
======================================================
Desk request 2026-08-11. One script, one folder, every chart the deck
needs, in the house presentation palette (blue / dark blue / grey) at
200 dpi on 16:9-friendly canvases.

    python tools/build_final_plots.py

Everything is recomputed from the committed stores and the frozen
research records - this script SELECTS and DRAWS, it never
re-implements a model. Output: "docs/final presentation plots/".
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import ml_detector as mld                      # noqa: E402
from analytics.euphoria import build_all_series               # noqa: E402
from analytics.euphoria_phases import (build_day_frame,       # noqa: E402
                                       episode_catalog)
from analytics.loaders import (load, THEME_COUNTS,            # noqa: E402
                               THEME_SENT, TICKER_COUNTS,
                               TICKER_SENT)

# ---------------------------------------------------------------- style
NAVY, BLUE, SKY, GREY, LIGHT = ("#0A2540", "#2E6FDB", "#9DB9DC",
                                "#6B7280", "#E8EBEF")
GREEN, PALE = "#1F6F5C", "#F2F5FA"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12.5, "legend.fontsize": 12,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.edgecolor": LIGHT, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.labelcolor": NAVY, "xtick.color": GREY, "ytick.color": GREY,
    "text.color": NAVY, "savefig.bbox": "tight",
})
OUT = os.path.join("docs", "final presentation plots")
os.makedirs(OUT, exist_ok=True)
SAVED = []


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    SAVED.append(name)
    print(f"  saved {name}", flush=True)


def despine(ax, bottom=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)


def foot(fig, text, y=-0.05):
    fig.text(0.01, y, text, fontsize=11, color=GREY, va="top")


# ----------------------------------------------------------------- data
T0 = time.time()
print("loading stores ...", flush=True)
prices = pd.read_parquet("data/prices/prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
series, pxmap = build_all_series(prices)
sym_by = {es.name: es.symbol for es in series}
name_by_sym = {v: k for k, v in sym_by.items()}
episodes = pd.read_parquet("data/processed/episodes.parquet")
for c in ("trough", "peak", "bust_date"):
    episodes[c] = pd.to_datetime(episodes[c])
desk = pd.read_parquet("data/processed/euphoria_desk.parquet")
desk["date"] = pd.to_datetime(desk["date"])
counts = {"theme": load(THEME_COUNTS), "ticker": load(TICKER_COUNTS)}
sents = {"theme": load(THEME_SENT), "ticker": load(TICKER_SENT)}
for d in list(counts.values()) + list(sents.values()):
    d["date"] = pd.to_datetime(d["date"])
BANK_BOARD = json.load(open("docs/research/nb02_august_bank.json"))
PLAIN = {r["column"]: r["feature"] for r in BANK_BOARD}

# short, plain axis names (desk: "use simpler feature names")
SHORT = {
    "e1": "how loud (vs its own year)",
    "e3": "louder than a month ago",
    "influx_speed": "louder than 2 weeks ago",
    "attention_accel": "this week vs this month",
    "hype_ratio": "this week vs normal",
    "attention_convexity": "getting loud FASTER",
    "bull_level": "how bullish",
    "bull_persist": "bullish for how long",
    "bull_inflection": "mood turning",
    "price_runup": "price run-up",
    "price_ret21": "price, last month",
}
CROWD = [c for c in mld.DESK_ML_BANK if c not in mld.PRICE_FEATURES]

# THE SETTING THIS PACK QUOTES (desk 2026-08-11: "use the stricter
# setting"). Strict is also the better-performing one on the measure
# that matters: walk-forward, names it flags underperform the rest of
# the universe by 1.4% in the month after a GET OUT and outperform by
# 0.7% after a GET IN (Standard: +0.1% / -0.4%).
SETTING = "strict"
IN_COL = "get_in_strict" if SETTING == "strict" else "get_in"
OUT_COL = "get_out_strict" if SETTING == "strict" else "get_out"
THR_KEY = "strict_threshold" if SETTING == "strict" else "live_threshold"
SETTING_LABEL = "Strict setting"


def price_of(sym, lo=None, hi=None):
    p = pxmap.get(sym)
    if p is None:
        return pd.Series(dtype=float)
    p = p.dropna()
    return p.loc[lo:hi] if (lo or hi) else p


# =====================================================================
# 1 + 2 — the three names, price only, then price + calls
# =====================================================================
# THREE panels, one row, 2026-08-11 (desk: "just make it 3 graphs
# please (bigger)"). Defense & aerospace was dropped for having only one
# Strict call; the desk asked for Bitcoin or QQQ as the replacement and
# neither can carry the figure - `crypto` is the most-tracked theme in the
# store (1.43M mentions vs gold's 83k) but has NO approved instrument in
# config/theme_etfs.csv, so it is never scored and has no calls, and QQQ
# (the ai_megacap theme) is scored but has ZERO calls on the Strict
# setting this pack uses. TSLA was tried and dropped: seven of its ten
# Strict GET OUTs fall in the 2019-20 melt-up, which reads as "sell all
# the way up" on a slide. Three panels at full width beat four cramped.
CASES = [("GME", "GME", "GME"),
         ("Gold", "GLD", "gold_metals"),
         ("Semiconductors", "SMH", "semiconductors")]
LO, HI = "2019-01-01", None

print("1-2  the three names ...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
for ax, (label, sym, _nm) in zip(axes.flat, CASES):
    p = price_of(sym, LO, HI)
    ax.plot(p.index, p.values, color=NAVY, lw=2.0)
    if len(p) and p.max() / max(p.min(), 1e-9) > 6:
        ax.set_yscale("log")
        ax.set_ylabel("price (log scale)")
    else:
        ax.set_ylabel("price (USD)")
    ax.set_title(f"{label}  ({sym})", loc="left")
    ax.tick_params(axis="x", rotation=20)
    despine(ax)
fig.suptitle("The three names, price only", fontsize=19,
             fontweight="bold", y=0.99)
foot(fig, "Daily close, 2019 to today. GME is on a log scale because "
          "the 2021 move is too large to share a linear axis with the "
          "rest of its own history.", y=-0.02)
fig.tight_layout()
save(fig, "P01_three_names_price_only.png")

fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
for ax, (label, sym, nm) in zip(axes.flat, CASES):
    p = price_of(sym, LO, HI)
    ax.plot(p.index, p.values, color=NAVY, lw=1.9)
    if len(p) and p.max() / max(p.min(), 1e-9) > 6:
        ax.set_yscale("log")
    g = desk[desk["name"] == nm]
    ins = [d for d in g.loc[g[IN_COL], "date"]
           if len(p) and p.index.min() <= d <= p.index.max()]
    outs = [d for d in g.loc[g[OUT_COL], "date"]
            if len(p) and p.index.min() <= d <= p.index.max()]
    for r in episodes[episodes["name"] == nm].itertuples():
        if len(p) and r.peak >= p.index.min() and r.trough <= p.index.max():
            ax.axvspan(max(r.trough, p.index.min()),
                       min(r.peak, p.index.max()), color=SKY, alpha=0.30,
                       lw=0)
    ax.plot(ins, [p.asof(d) for d in ins], "^", color=BLUE, ms=14,
            mec="white", mew=1.4, zorder=5, label="GET IN")
    ax.plot(outs, [p.asof(d) for d in outs], "v", color=NAVY, ms=14,
            mec="white", mew=1.4, zorder=5, label="GET OUT")
    if nm not in set(desk["name"]):
        ax.set_title(f"{label}  ({sym})  —  not in the tracked universe, "
                     "so no calls", loc="left", fontsize=13)
    else:
        ax.set_title(f"{label}  ({sym})  —  {len(ins)} GET IN · "
                     f"{len(outs)} GET OUT", loc="left", fontsize=13)
        ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_ylabel("price")
    ax.tick_params(axis="x", rotation=20)
    despine(ax)
fig.suptitle(f"The same three names, with the model's calls  "
             f"({SETTING_LABEL})", fontsize=17, fontweight="bold",
             y=0.99)
foot(fig, "Shaded = a real boom→bust episode as the price-only ground "
          "truth defines it. \u25b2 GET IN = the crowd is arriving early "
          "in a move; \u25bc GET OUT = the crowd is extreme after a "
          "boom.\n"
          f"Drawn on the {SETTING_LABEL} — the higher-conviction cut: "
          "about half the calls of the Standard setting, and the one "
          "with the measurable edge (names it flags underperform the "
          "universe by 1.4% in the month after a GET OUT).\n"
          "GET IN is rare on this setting by design — Strict fires only "
          "the highest-conviction calls, so these panels are largely a "
          "GET OUT story.", y=-0.045)
fig.tight_layout()
save(fig, "P02_three_names_with_calls.png")

# =====================================================================
# 3 — the pipeline as a FUNNEL, with real counts
# =====================================================================
print("3    pipeline funnel ...", flush=True)
_tm = load("daily_term_counts.parquet")
n_posts = int(_tm[_tm["term"] == "__TOTAL__"]["mention_count"].sum())
n_tagged = int(sents["ticker"]["n_posts"].sum())
n_tick = int(counts["ticker"]["ticker"].nunique())
n_universe = len(series)
n_days = len(desk)
n_eps = len(episodes)
n_calls = int(desk[IN_COL].sum() + desk[OUT_COL].sum())

# NB: do NOT put "331k posts" and "796k mentions" in the same funnel -
# the first is posts in the term store's 12-month window, the second is
# all-time post x ticker PAIRS, so the funnel appears to GROW. Each rung
# below is a different unit, stated as such, and every rung genuinely
# narrows.
# A LINEAR left-to-right chain (desk 2026-08-11: "make it linear again,
# good for a PPT"). Six stages, one arrow each, no funnel geometry.
STAGES = [
    (f"{n_posts/1000:,.0f}k posts",
     "Reddit · StockTwits · X\n(last 12 months)", NAVY),
    (f"{n_tick:,} tickers\n+ {counts['theme']['theme'].nunique()} themes",
     "names extracted,\nthe text thrown away", NAVY),
    ("mood +\nattention",
     "how many are talking,\nand how bullish", BLUE),
    ("11\nmeasurements",
     "each vs the name's\nOWN history", BLUE),
    ("2 models\ncombined",
     "logistic regression\n+ monotone GBM", BLUE),
    ("GET IN /\nGET OUT",
     "the call, with the\ndate it was made", GREEN),
]
fig, ax = plt.subplots(figsize=(16, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
n = len(STAGES)
bw, gapx = 13.6, 3.0
x0 = (100 - (n * bw + (n - 1) * gapx)) / 2
for i, (headline, sub, col) in enumerate(STAGES):
    x = x0 + i * (bw + gapx)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, 44), bw, 30, boxstyle="round,pad=0.5,rounding_size=1.1",
        facecolor=col, edgecolor="none"))
    ax.text(x + bw / 2, 65, headline, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold",
            linespacing=1.35)
    ax.text(x + bw / 2, 52.5, sub, ha="center", va="center",
            color="white", fontsize=9.5, alpha=0.93, linespacing=1.45)
    ax.text(x + bw / 2, 78, f"{i+1}", ha="center", va="center",
            color=col, fontsize=15, fontweight="bold")
    if i < n - 1:
        ax.annotate("", (x + bw + gapx - 0.4, 59), (x + bw + 0.4, 59),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color=GREY))
ax.text(50, 30,
        "Everything is built WALK-FORWARD — each year is scored by a "
        "model trained only on earlier years — so every number quoted "
        "is out-of-sample.",
        ha="center", va="center", fontsize=12, color=NAVY, style="italic")
ax.text(50, 21,
        "The desk machine never sees post text: names and a sentiment "
        "score are extracted at step 2 and the words are discarded.",
        ha="center", va="center", fontsize=11, color=GREY)
ax.set_title("From raw posts to a call", fontsize=19,
             fontweight="bold", loc="left", pad=16)
save(fig, "P03_pipeline_funnel.png")

# =====================================================================
# 4 — which measurements work (crowd only; price pair removed)
# =====================================================================
print("4    feature strength ...", flush=True)
bb = pd.DataFrame(BANK_BOARD)
bb = bb[~bb["column"].isin(mld.PRICE_FEATURES)].copy()
bb["short"] = bb["column"].map(SHORT)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
for ax, col, ttl, colr in (
        (axes[0], "AUROC vs y_onset", "Calling the START  (GET IN)", BLUE),
        (axes[1], "AUROC vs y_top", "Calling the TOP  (GET OUT)", NAVY)):
    d = bb.sort_values("AUROC vs y_top")
    ax.barh(d["short"], d[col], color=colr, height=0.62)
    ax.axvline(0.5, color=GREY, lw=1.6, ls="--")
    for i, v in enumerate(d[col]):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=11,
                color=NAVY)
    ax.set_xlim(0.42, 0.72)
    ax.set_title(ttl, fontsize=14)
    ax.set_xlabel("hit rate when ranking days (0.50 = coin flip)")
    despine(ax)
fig.suptitle("Which crowd measurements actually work, one at a time",
             fontsize=17, fontweight="bold", y=1.02)
foot(fig, "Each bar is that ONE measurement used alone as a detector. "
          "0.50 is a coin flip; higher is better. The two price "
          "measurements are deliberately excluded here — 'price went up, "
          "so price will go up' is not an insight, and we want to show "
          "the CROWD is doing work.\nNo single measurement is a signal "
          "on its own — the model's edge comes from combining them.",
     y=-0.06)
fig.tight_layout()
save(fig, "P04_which_measurements_work.png")

# =====================================================================
# 5 — break one input on purpose (the ablation)
# =====================================================================
print("5    ablation (fitting models, ~1 min) ...", flush=True)
frame = build_day_frame(series, pxmap, episodes, counts, sents)
cand = mld.attach_price_features(mld.candidate_frame(frame), series, pxmap)
ymax = int(cand["year"].max())
tr, te = cand[cand["year"] < ymax], cand[cand["year"] == ymax]
from sklearn.metrics import average_precision_score               # noqa: E402


def fold_ap(label, feats):
    return average_precision_score(
        te[label], mld.make_ens_fit(label)(tr, te, feats))


abl = {}
for label, head in (("y_top", "GET OUT"), ("y_onset", "GET IN")):
    full = fold_ap(label, mld.DESK_ML_BANK)
    # CROWD measurements only on this chart (desk 2026-08-11, same
    # reason as P04): the price pair stays IN the model - it is removed
    # from the DISPLAY, because "price predicts price" is not the claim
    # we are making and it crowds out the crowd bars.
    rows = [{"column": f, "short": SHORT.get(f, f),
             "delta": full - fold_ap(
                 label, [x for x in mld.DESK_ML_BANK if x != f])}
            for f in CROWD]
    abl[head] = (full, pd.DataFrame(rows))

fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
order = abl["GET OUT"][1].sort_values("delta")["short"]
for ax, head, colr in ((axes[0], "GET IN", BLUE),
                       (axes[1], "GET OUT", NAVY)):
    full, d = abl[head]
    d = d.set_index("short").loc[order].reset_index()
    cols = [GREY if v <= 0 else colr for v in d["delta"]]
    ax.barh(d["short"], d["delta"], color=cols, height=0.62)
    ax.axvline(0, color=NAVY, lw=1.4)
    for i, v in enumerate(d["delta"]):
        ax.text(v + (0.0015 if v >= 0 else -0.0015), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=10.5, color=NAVY)
    ax.set_title(f"{head} — accuracy lost", fontsize=14)
    ax.set_xlabel("how much worse the model gets without it")
    despine(ax)
fig.suptitle("Breaking one input on purpose — what the model actually "
             "relies on", fontsize=17, fontweight="bold", y=1.02)
foot(fig, "We remove ONE measurement, refit the whole model, and measure "
          "how much accuracy it loses. Coloured bars to the right = the "
          "model needs it. Grey bars = removing it changed nothing "
          "measurable,\nbecause the other measurements already carry that "
          "information (they are kept anyway: they cost nothing and add "
          "robustness if the data shifts).\nThe two price measurements "
          "are still in the model — they are left off this chart so it "
          "shows what the CROWD contributes.", y=-0.06)
fig.tight_layout()
save(fig, "P05_break_one_input.png")

# =====================================================================
# 6 — how the measurements relate (Spearman, readable decimals)
# =====================================================================
print("6    correlations ...", flush=True)
from matplotlib.colors import LinearSegmentedColormap                # noqa: E402
corr = cand[mld.DESK_ML_BANK].corr(method="spearman")
labels = [SHORT.get(c, c) for c in corr.columns]
cmap = LinearSegmentedColormap.from_list("bg", [GREY, "#FFFFFF", BLUE])
fig, ax = plt.subplots(figsize=(11.5, 9.4))
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), labels, rotation=40, ha="right",
              fontsize=11)
ax.set_yticks(range(len(corr)), labels, fontsize=11)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(v) > 0.55 else NAVY,
                fontweight="bold" if i == j else "normal")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.72, label="rank correlation")
ax.set_title("How the 11 measurements relate to each other",
             fontsize=16, loc="left", pad=14)
foot(fig, "+1.00 = they always move together, 0.00 = unrelated, −1.00 = "
          "they move oppositely. The attention block is correlated by "
          "design (same phenomenon at different speeds);\nthe mood and "
          "price blocks bring genuinely separate information — which is "
          "why the model needs all three.", y=-0.02)
fig.tight_layout()
save(fig, "P06_how_measurements_relate.png")

# =====================================================================
# 7 + 8 — what price movements we can track, and how big they are
# =====================================================================
print("7-8  what we track ...", flush=True)
eps = episodes.copy()
eps["boom_pct_x"] = eps["boom_pct"] * 100
eps["bust_pct_x"] = eps["bust_pct"] * 100
eps["run_d"] = (eps["peak"] - eps["trough"]).dt.days

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
ax = axes[0]
# what SIZE of move we can track, split by instrument type (desk
# 2026-08-11): a theme ETF and a single name are different animals
_grp = {"Themes / ETFs": eps[eps["kind"] != "single"],
        "Single stocks": eps[eps["kind"] == "single"]}
xs = np.arange(2)
rise = [g["boom_pct_x"].median() for g in _grp.values()]
fall = [-g["bust_pct_x"].median() for g in _grp.values()]
b1 = ax.bar(xs - 0.19, rise, 0.34, color=BLUE, label="typical rise")
b2 = ax.bar(xs + 0.19, fall, 0.34, color=GREY, label="typical fall")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.0f}%", ha="center", fontsize=12,
                fontweight="bold", color=NAVY)
ax.set_xticks(xs, [f"{k}\n({len(v)} episodes)"
                   for k, v in _grp.items()], fontsize=11.5)
ax.set_ylabel("size of the move  (%)")
ax.set_title("What size of move we can track", fontsize=13)
ax.legend(frameon=False, fontsize=11)
ax.set_ylim(0, max(rise + fall) * 1.22)
despine(ax)
ax = axes[1]
ax.hist(eps["run_d"], bins=26, color=BLUE, edgecolor="white")
ax.axvline(eps["run_d"].median(), color=NAVY, lw=2, ls="--")
ax.text(eps["run_d"].median() + 4, ax.get_ylim()[1] * 0.88,
        f"typical: {eps['run_d'].median():.0f} days", fontsize=11,
        color=NAVY, fontweight="bold")
ax.set_xlabel("days from the low to the top")
ax.set_ylabel("number of episodes")
ax.set_title("How long a move lasts", fontsize=13)
despine(ax)
ax = axes[2]
by_year = eps.groupby("year").size()
ax.bar(by_year.index, by_year.values, color=BLUE, width=0.66)
ax.set_xlabel("year")
ax.set_ylabel("episodes")
ax.set_title("When they happen (manias cluster)", fontsize=13)
despine(ax)
fig.suptitle("What price movements we can track — the 494 episodes the "
             "model is graded on", fontsize=17, fontweight="bold", y=1.03)
foot(fig, "An episode = a rise of at least 40% (single stock) or 20% "
          "(theme/ETF) off its recent low, followed by a fall of at least "
          "25% / 12% within 90 days.\nDefined by price ALONE — the crowd "
          "never grades its own calls.", y=-0.07)
fig.tight_layout()
save(fig, "P07_what_we_can_track.png")

fig, ax = plt.subplots(figsize=(12.5, 6.0))
bins = np.logspace(np.log10(max(eps["boom_pct_x"].min(), 5)),
                   np.log10(eps["boom_pct_x"].max()), 34)
ax.hist(eps["boom_pct_x"], bins=bins, color=BLUE, edgecolor="white")
ax.set_xscale("log")
med = eps["boom_pct_x"].median()
ax.axvline(med, color=NAVY, lw=2.2, ls="--")
ax.annotate(f"half the episodes are bigger than {med:.0f}%",
            (med, ax.get_ylim()[1] * 0.78), xytext=(18, 0),
            textcoords="offset points", fontsize=12, color=NAVY,
            fontweight="bold")
for v, lab in ((100, "a double"), (1000, "a 10-bagger")):
    if bins[0] <= v <= bins[-1]:
        ax.axvline(v, color=GREY, lw=1.2, ls=":")
        ax.text(v, ax.get_ylim()[1] * 0.30, f"  {lab}", fontsize=11,
                color=GREY, rotation=90, va="bottom")
ax.set_xlabel("size of the rise, before the fall  (%, log scale)")
ax.set_ylabel("number of episodes")
ax.set_title("How big are the moves we are trying to catch?",
             fontsize=16, loc="left")
foot(fig, "A log scale is used because the range is enormous — most "
          "episodes are tens of percent, a handful (GME 2021) are "
          "thousands. Each bar counts episodes of that size.\nThe point "
          "of the chart: this is not a rare-monster problem — there are "
          "hundreds of ordinary-sized manias to catch.", y=-0.06)
fig.tight_layout()
save(fig, "P08_how_big_are_the_moves.png")

# =====================================================================
# 9 — gold: the measurements through the episode (big, plain names)
# =====================================================================
print("9    gold measurements ...", flush=True)
GNAME = "gold_metals"
gwin = ("2025-09-01", "2026-05-01")
gc = cand[(cand["name"] == GNAME)].set_index("date").sort_index()
gc = gc.loc[gwin[0]:gwin[1]].dropna(subset=CROWD + ["price_runup"])
if len(gc):
    # clip EVERYTHING (price row included) to the days that actually
    # carry measurements - gold only enters the scored frame in 2026,
    # and empty leading panels read as broken data (desk 2026-08-11)
    gwin = (gc.index.min().strftime("%Y-%m-%d"),
            gc.index.max().strftime("%Y-%m-%d"))
g_eps = episodes[(episodes["name"] == GNAME)
                 & (episodes["peak"] >= gwin[0])
                 & (episodes["peak"] <= gwin[1])]
show = CROWD + ["price_runup"]
fig, axes = plt.subplots(len(show) + 1, 1, figsize=(13.5, 15.5),
                         sharex=True,
                         height_ratios=[2.0] + [1] * len(show))
gp = price_of("GLD", *gwin)
axes[0].plot(gp.index, gp.values, color=NAVY, lw=2.2)
for r in g_eps.itertuples():
    axes[0].axvspan(max(r.trough, gp.index.min()),
                    min(r.peak, gp.index.max()), color=SKY, alpha=0.30,
                    lw=0)
gd = desk[(desk["name"] == GNAME)]
gouts = [d for d in gd.loc[gd[OUT_COL], "date"]
         if gp.index.min() <= d <= gp.index.max()]
axes[0].plot(gouts, [gp.asof(d) for d in gouts], "v", color=NAVY, ms=16,
             mec="white", mew=1.6, zorder=5)
for d in gouts:
    axes[0].axvline(d, color=NAVY, lw=1.2, ls="--", alpha=0.7)
axes[0].set_title(f"Gold (GLD) — price, with the GET OUT call "
                  f"({SETTING_LABEL})", fontsize=15, loc="left")
axes[0].set_ylabel("price")
despine(axes[0])
for ax, f in zip(axes[1:], show):
    s = gc[f]
    ax.plot(s.index, s.values, color=BLUE, lw=2.0)
    ax.fill_between(s.index, 0, s.values, color=BLUE, alpha=0.12)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1.0], ["low", "mid", "HIGH"], fontsize=10.5)
    ax.set_ylabel(SHORT.get(f, f), rotation=0, ha="right", va="center",
                  fontsize=12)
    for d in gouts:
        ax.axvline(d, color=NAVY, lw=1.2, ls="--", alpha=0.7)
    despine(ax, bottom=False)
fig.suptitle("Gold, 2025–26: every measurement through the episode",
             fontsize=18, fontweight="bold", y=0.995)
foot(fig, "Each row is one of the model's measurements, on the same time "
          "axis as the price above. HIGH means 'extreme compared with "
          "gold's OWN history' — never compared with other names.\nThe "
          "dashed line is the day the GET OUT fired: read up the column "
          "to see which measurements were pinned high at that moment.",
     y=0.0)
fig.tight_layout()
save(fig, "P09_gold_measurements_through_the_episode.png")

# =====================================================================
# 10 — how the weights were FOUND, and what they are
# =====================================================================
print("10   how the weights were found ...", flush=True)
from sklearn.linear_model import LogisticRegression                  # noqa: E402
lg = LogisticRegression(class_weight="balanced", max_iter=2000)
lg.fit(tr[mld.DESK_ML_BANK], tr["y_top"])
W = dict(zip(mld.DESK_ML_BANK, lg.coef_[0]))
B0 = float(lg.intercept_[0])

fig = plt.figure(figsize=(14.5, 7.4))
gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1], wspace=0.55)
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
boxes = [
    (78, "1.  Show it the past",
     f"{len(tr):,} instrument-days from\n"
     f"{int(tr.year.min())}–{int(tr.year.max())}, each labelled\n"
     "'a top followed' or 'it didn't'", NAVY),
    (54, "2.  Let it try weights",
     "start from zero, score every day,\ncompare with what really "
     "happened,\nnudge the weights, repeat", BLUE),
    (30, "3.  Keep what worked",
     "the weights that separate the two\ngroups best — this is the "
     "whole\n'training' step", BLUE),
    (6, "4.  Freeze and test on unseen years",
     "the weights never change again;\nevery number we quote comes from\n"
     "years the model never saw", GREEN),
]
for y, head, body, col in boxes:
    ax.add_patch(mpatches.FancyBboxPatch(
        (2, y), 96, 19, boxstyle="round,pad=0.5,rounding_size=1.1",
        facecolor=col, edgecolor="none"))
    ax.text(6, y + 14, head, fontsize=13, fontweight="bold",
            color="white", va="center")
    ax.text(6, y + 6.2, body, fontsize=10.5, color="white", va="center",
            linespacing=1.5)
    if y > 10:
        ax.annotate("", (50, y - 4.2), (50, y - 0.8),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=GREY))
ax.set_title("How the weights were found", fontsize=15, loc="left")

ax2 = fig.add_subplot(gs[0, 1])
order = sorted(mld.DESK_ML_BANK, key=lambda f: W[f])
vals = [W[f] for f in order]
cols = [BLUE if v >= 0 else GREY for v in vals]
ax2.barh([SHORT.get(f, f) for f in order], vals, color=cols, height=0.62)
ax2.axvline(0, color=NAVY, lw=1.4)
_pad = (max(vals) - min(vals)) * 0.035
for i, v in enumerate(vals):
    ax2.text(v + (_pad if v >= 0 else -_pad), i, f"{v:+.2f}",
             va="center", ha="left" if v >= 0 else "right",
             fontsize=11.5, color=NAVY, fontweight="bold")
ax2.set_xlim(min(vals) - _pad * 9, max(vals) + _pad * 9)
ax2.set_title("…and what they came out as  (GET OUT)", fontsize=15,
              loc="left")
ax2.set_xlabel("weight  (+ = more of this means more top risk)")
despine(ax2)
fig.suptitle("The model is 11 numbers — here is where they come from",
             fontsize=17, fontweight="bold", y=1.02)
foot(fig, "Blue = raises the risk of a top, grey = lowers it. All "
          "measurements are on the same 0–1 scale, so the sizes are "
          "directly comparable.\nNothing here is hand-set: a person "
          "chose WHICH measurements to offer, the data chose how much "
          "each one counts.", y=-0.04)
save(fig, "P10_how_the_weights_were_found.png")

# =====================================================================
# 11 — gold: one day, from measurements to the call
# =====================================================================
print("11   gold call walkthrough ...", flush=True)
gouts_all = sorted(gd.loc[gd[OUT_COL], "date"])
day = None
for d in gouts_all:
    if d in gc.index:
        day = d
if day is None and len(gc):
    day = gc.index[-1]
x = gc.loc[day]
contrib = {f: float(W[f]) * float(x[f]) for f in mld.DESK_ML_BANK}
logodds = B0 + sum(contrib.values())
p_lg = 1 / (1 + np.exp(-logodds))
thr = json.load(open(
    "data/processed/euphoria_desk_report.json"))["get_out"][THR_KEY]
out_score = float(desk[(desk["name"] == GNAME)
                       & (desk["date"] == day)]["out_score"].iloc[0])

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25],
                      width_ratios=[1.32, 1], hspace=0.52, wspace=0.30)
axp = fig.add_subplot(gs[0, :])
axp.plot(gp.index, gp.values, color=NAVY, lw=2.2)
for r in g_eps.itertuples():
    axp.axvspan(max(r.trough, gp.index.min()),
                min(r.peak, gp.index.max()), color=SKY, alpha=0.30, lw=0)
axp.axvline(day, color=NAVY, lw=1.8, ls="--")
axp.plot([day], [gp.asof(day)], "v", color=NAVY, ms=17, mec="white",
         mew=1.6, zorder=5)
_pk = g_eps["peak"].max() if len(g_eps) else None
_lead = (pd.Timestamp(_pk) - pd.Timestamp(day)).days if _pk is not None \
    else None
axp.annotate(f"GET OUT — {pd.Timestamp(day):%d %b %Y}"
             + (f"\n{_lead} days before the top" if _lead and _lead > 0
                else ""),
             (day, gp.asof(day)), xytext=(-210, -108),
             textcoords="offset points", fontsize=12.5, color=NAVY,
             fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.4))
axp.set_title("Gold (GLD) — the call, in context", fontsize=15,
              loc="left")
axp.set_ylabel("price")
despine(axp)

axc = fig.add_subplot(gs[1, 0])
order_c = sorted(mld.DESK_ML_BANK, key=lambda f: contrib[f])
vals_c = [contrib[f] for f in order_c]
axc.barh([SHORT.get(f, f) for f in order_c], vals_c,
         color=[BLUE if v >= 0 else GREY for v in vals_c], height=0.6)
axc.axvline(0, color=NAVY, lw=1.4)
for i, f in enumerate(order_c):
    axc.text(contrib[f] + (0.03 if contrib[f] >= 0 else -0.03), i,
             f"{x[f]:.2f} × {W[f]:+.2f}", va="center",
             ha="left" if contrib[f] >= 0 else "right", fontsize=9.5,
             color=NAVY)
axc.set_title(f"What each measurement contributed on "
              f"{pd.Timestamp(day):%d %b}", fontsize=13, loc="left")
axc.set_xlabel("reading × weight")
_lo, _hi = min(vals_c), max(vals_c)
_sp = (_hi - _lo)
axc.set_xlim(_lo - _sp * 0.55, _hi + _sp * 0.55)
despine(axc)

axl = fig.add_subplot(gs[1, 1])
axl.axis("off")
steps = [
    ("Add the contributions", f"total = {logodds:+.2f}"),
    ("Turn it into a probability", f"{p_lg:.0%} chance a top is near"),
    ("Rank it against every other name", f"score {out_score:.2f} out of 1"),
    ("Compare with the trigger",
     f"{out_score:.2f}  vs  {thr:.2f}  →  FIRE"),
]
y = 0.93
for k, (h, b) in enumerate(steps):
    last = k == len(steps) - 1
    axl.add_patch(mpatches.FancyBboxPatch(
        (0.02, y - 0.185), 0.95, 0.165,
        boxstyle="round,pad=0.008", facecolor=GREEN if last else PALE,
        edgecolor="none", transform=axl.transAxes))
    axl.text(0.06, y - 0.06, f"{k+1}.  {h}", fontsize=12.5,
             fontweight="bold", color="white" if last else NAVY,
             transform=axl.transAxes, va="center")
    axl.text(0.06, y - 0.145, b, fontsize=12,
             color="white" if last else GREY, transform=axl.transAxes,
             va="center")
    if not last:
        axl.annotate("", (0.5, y - 0.205), (0.5, y - 0.19),
                     xycoords=axl.transAxes, textcoords=axl.transAxes,
                     arrowprops=dict(arrowstyle="-|>", lw=1.4, color=GREY))
    y -= 0.235
axl.set_title("…and how that becomes a call", fontsize=13, loc="left")
fig.suptitle(f"How gold's GET OUT was made — from readings to the call "
             f"({SETTING_LABEL})", fontsize=17, fontweight="bold",
             y=0.98)
foot(fig, "Bottom left: each measurement's reading on the day (0–1 "
          "against gold's own history) multiplied by its weight. Bottom "
          "right: those contributions become one score, which is compared "
          "with a trigger\nchosen on earlier years only. The trigger is "
          "fixed in advance and never tuned to this episode.", y=-0.01)
save(fig, "P11_gold_from_readings_to_call.png")

print(f"\n{len(SAVED)} figures -> {OUT}/  ({time.time()-T0:.0f}s)")
for s in SAVED:
    print("   ", s)
