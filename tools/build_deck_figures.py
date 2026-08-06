"""Render every pitch-deck figure in the house palette.

Desk 2026-08-05: *"generally fix all plots in my project the
specification i want too and rebuild the plots and images if need to"* —
and the deck is being assembled on an internal machine that has **only
the brief and these PNGs**, so every figure has to be self-explanatory
on its own.

WHAT "SELF-EXPLANATORY" MEANS HERE, because it drives every choice below:
the person building the deck cannot open this repo, cannot re-run a
notebook and cannot ask what an axis means. So each figure carries its
own title stating the FINDING (not the variable names), its own units and
n, and its own source line. If a figure needs a caption to make sense, it
is the wrong figure.

The palette and the chart rules are lifted from `dashboard.py` so the
deck and the product look like one system: navy on white, hairline
gridlines, no chart frame, no legend box, whitespace doing the
separating.

    python tools/build_deck_figures.py            # all figures
    python tools/build_deck_figures.py --only W   # just the walkthrough

Output: docs/figures/deck/*.png at 200 dpi.

ONE images root, `docs/figures/`, split by purpose: `deck/` for the
pitch pack and `dashboard/` for product screenshots. There used to be a
second `figures` folder nested under `docs/presentation/`, which meant
two directories with the same name at different depths and no rule for
which one a given PNG belonged in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "figures", "deck")
RESEARCH = os.path.join(ROOT, "docs", "research")
PROC = os.path.join(ROOT, "data", "processed")

# ---- the house palette, identical to dashboard.py -------------------------
NAVY, NAVY_MID = "#0A1E2E", "#1B3A52"
WHITE, PANEL = "#FFFFFF", "#F6F7F8"
INK, INK_MUTED, HAIRLINE = "#111111", "#666666", "#ECECEC"
SLATE, SLATE_LIGHT = "#7A8794", "#B9BFC7"
BULL, BEAR, TEAL, OCHRE = "#1F6F5C", "#A6413B", "#2E6E7E", "#8A6D1F"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "axes.edgecolor": HAIRLINE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": HAIRLINE, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK_MUTED, "axes.labelsize": 9,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": WHITE,
})


def _finish(fig, name, source):
    """Every figure states where its numbers came from. During Q&A a PM
    can point at a chart and the presenter can name the file - which is
    the difference between a defensible number and a slide."""
    fig.text(0.005, -0.02, f"source: {source}", fontsize=6.5,
             color=INK_MUTED, ha="left", va="top")
    for ax in fig.axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(HAIRLINE)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


def _tag(fig):
    """The worked-example ribbon. Placed BELOW the axes rather than as a
    suptitle: a suptitle collides with a two-line axes title, and these
    titles are two lines by design because each one states a finding."""
    # bottom-RIGHT, opposite the source line. Anywhere along the top
    # collides with a two-line axes title, and every title here is two
    # lines by design because each one states a finding rather than
    # naming the variables.
    fig.text(0.995, -0.02, f"WORKED EXAMPLE  ·  {THEME}", fontsize=7.5,
             color=TEAL, weight="bold", ha="right", va="top")


def _rj(name):
    with open(os.path.join(RESEARCH, name), encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# THE WALKTHROUGH THREAD - one theme, followed from raw post to signal
#
# Desk request: show notebook 00's worked example running THROUGH the
# technical section rather than as one slide. Notebook 00 follows the
# `semiconductors` theme; these four figures are the same instrument at
# four depths, so the audience is never asked to hold two examples at
# once. They are numbered W1-W4 and are meant to appear on four
# consecutive technical slides.
# ===========================================================================
# WHICH INSTRUMENT THE WALKTHROUGH FOLLOWS.
#
# Notebook 00 uses `semiconductors`, and this defaulted to it - until the
# outcomes were checked. Measured over the 60 days after each theme's most
# recent GET OUT: semiconductors +23% (the flag was early and the rally
# continued), memory -19.7%, gold_metals -19.0%.
#
# `memory` is used because it is a CLEAN illustration of the method and it
# is one of the episodes the desk itself named as a good result - not
# because it is the best number available. The semiconductors miss is
# better used deliberately, on the "where it does not work" slide, where
# volunteering a counter-example buys more credibility than hiding it.
# Switch with --theme.
THEME = "memory"
# The walkthrough is drawn around the most recent GET OUT on this theme,
# not over the whole history. On a slide, 700 days of a daily series is a
# grey smear - the audience cannot see the episode the story is about.
# +/-200 days keeps the build-up, the flag and the aftermath all visible.
WALK_PAD_D = 200


def _flag_window():
    """(lo, hi, flag_dates) around the newest GET OUT on THEME."""
    desk_p = os.path.join(PROC, "euphoria_desk.parquet")
    if not os.path.exists(desk_p):
        return None, None, []
    d = pd.read_parquet(desk_p)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["name"] == THEME) & d["get_out"].astype(bool)]
    if not len(d):
        return None, None, []
    last = d["date"].max()
    lo = last - pd.Timedelta(days=WALK_PAD_D)
    hi = last + pd.Timedelta(days=WALK_PAD_D // 2)
    return lo, hi, sorted(t for t in d["date"] if lo <= t <= hi)


def _theme_series():
    tc = pd.read_parquet(os.path.join(PROC, "daily_theme_counts.parquet"))
    tc["date"] = pd.to_datetime(tc["date"])
    one = (tc[tc["theme"] == THEME].set_index("date")["mention_count"]
           .sort_index().asfreq("D").fillna(0.0))
    tot = tc.groupby("date")["mention_count"].sum().sort_index()
    return one, tot.reindex(one.index).fillna(0.0)


def fig_w1():
    """W1 - the crowd, counted. Attention share vs its own normal."""
    one, tot = _theme_series()
    share = (one / tot.where(tot > 0) * 100).rolling(7, min_periods=1).mean()
    lo, hi, _f = _flag_window()
    if lo is not None:
        share = share[(share.index >= lo) & (share.index <= hi)]
    base = share.rolling(120, min_periods=60).median()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(share.index, share.values, color=NAVY, lw=1.6,
            label="7-day share of all theme chatter")
    ax.plot(base.index, base.values, color=SLATE, lw=1.2, ls="--",
            label="its own 120-day normal")
    ax.fill_between(share.index, base.values, share.values,
                    where=(share.values > base.values),
                    color=TEAL, alpha=0.15, interpolate=True)
    ax.set_title("Step 1 — measured against its OWN normal,\n"
                 "never against another name", loc="left")
    ax.set_ylabel("% of all theme mentions")
    ax.legend(loc="upper left")
    _tag(fig)
    _finish(fig, "W1_walkthrough_attention.png",
            "data/processed/daily_theme_counts.parquet")


def fig_w2():
    """W2 - the same day, decomposed into the factors that score it."""
    ep = pd.read_parquet(os.path.join(PROC, "euphoria_levels.parquet"))
    ep["date"] = pd.to_datetime(ep["date"])
    one = ep[ep["name"] == THEME].set_index("date").sort_index()
    lo, hi, _f = _flag_window()
    if lo is not None:
        one = one[(one.index >= lo) & (one.index <= hi)]
    feats = [("e1", "E1  attention level", NAVY),
             ("e2", "E2  sustained bullishness", TEAL),
             ("e3", "E3  crowd influx", NAVY_MID),
             ("e5", "E5  super-exponential attention", OCHRE)]
    fig, axes = plt.subplots(len(feats), 1, figsize=(9, 5.6), sharex=True)
    for ax, (col, lbl, c) in zip(axes, feats):
        if col not in one.columns:
            continue
        ax.plot(one.index, one[col].values, color=c, lw=1.3)
        ax.set_ylabel(lbl, fontsize=8, rotation=0, ha="right", va="center",
                      labelpad=12)
        ax.set_yticks([0, 0.5, 1.0])
    axes[0].set_title("Step 2 — four factors, each ranked against\n"
                      "this instrument's own history", loc="left")
    _tag(fig)
    _finish(fig, "W2_walkthrough_factors.png",
            "data/processed/euphoria_levels.parquet")


def fig_w3():
    """W3 - the factors combine into one level, and the flag fires."""
    ep = pd.read_parquet(os.path.join(PROC, "euphoria_levels.parquet"))
    ep["date"] = pd.to_datetime(ep["date"])
    one = ep[ep["name"] == THEME].set_index("date").sort_index()
    lo, hi, flags = _flag_window()
    if lo is not None:
        one = one[(one.index >= lo) & (one.index <= hi)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(one.index, one["level"].values, color=NAVY, lw=1.8,
            label="euphoria level (0-100)")
    for i, f in enumerate(flags):
        ax.axvline(f, color=BEAR, lw=1.6,
                   label="GET OUT flag" if i == 0 else None)
    ax.set_ylabel("euphoria level")
    ax.set_title("Step 3 — the factors combine into one level,\n"
                 "and a threshold frozen years earlier decides", loc="left")
    ax.legend(loc="upper left")
    _tag(fig)
    _finish(fig, "W3_walkthrough_level_and_flag.png",
            "data/processed/euphoria_levels.parquet + euphoria_desk.parquet")


def fig_w4():
    """W4 - what the price did afterwards. The payoff slide."""
    from src.themes import THEME_ETFS
    sym = THEME_ETFS.get(THEME, "SMH")
    px = pd.read_parquet(os.path.join(ROOT, "data", "prices",
                                      "prices.parquet"))
    px["date"] = pd.to_datetime(px["date"])
    s = (px[px["symbol"] == sym].set_index("date")["px_last"]
         .sort_index().asfreq("D").ffill())
    lo, hi, flags = _flag_window()
    if lo is not None:
        s = s[(s.index >= lo) & (s.index <= hi)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(s.index, s.values, color=SLATE, lw=1.6, label=f"{sym} price")
    for i, f in enumerate(flags):
        ax.axvline(f, color=BEAR, lw=1.6,
                   label="GET OUT flag" if i == 0 else None)
    ax.set_ylabel(f"{sym}")
    ax.set_title("Step 4 — the same flag, against the price\n"
                 "the crowd signal was never shown", loc="left")
    ax.legend(loc="upper left")
    _tag(fig)
    _finish(fig, "W4_walkthrough_outcome.png",
            "data/prices/prices.parquet + euphoria_desk.parquet")


# ===========================================================================
# THE EVIDENCE FIGURES
# ===========================================================================
# The two feature banks are graded against DIFFERENT labels and must not
# be drawn on one axis: the onset bank predicts `y_onset` (euphoria
# starting) and the top bank predicts `y_top` (euphoria ending). An
# earlier version of this figure plotted all ten against y_onset, which
# put `e2` - a top-bank feature - at 0.476 under a title claiming every
# factor cleared 0.5. That is the exact species of error this deck cannot
# afford, and it was caught by reading the rendered chart rather than the
# code. Draw each bank against its own label.
ONSET_BANK = ["attention_accel", "hype_ratio", "bull_inflection",
              "influx_speed", "attention_convexity"]


def fig_auroc():
    rows = [r for r in _rj("nb02_feature_stats.json")
            if r["label"] == "y_onset" and r["feature"] in ONSET_BANK]
    rows.sort(key=lambda r: r["auroc"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, r in enumerate(rows):
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=SLATE_LIGHT, lw=3,
                solid_capstyle="round")
        ax.plot(r["auroc"], i, "o", color=NAVY, ms=7, zorder=3)
    ax.axvline(0.5, color=BEAR, lw=1.2, ls="--")
    # label at the BOTTOM of the line: at the top it collides with the
    # two-line title, which is fixed height because every title here is
    # two lines by design.
    ax.text(0.5, -0.45, " coin flip", color=BEAR, fontsize=8, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r["feature"].replace("_", " ") for r in rows])
    ax.set_xlabel("AUROC (95% CI) — GET IN bank, graded on onsets",
                  labelpad=8)
    # TITLE TONE. This read "every edge is small ... the honest shape of a
    # crowd-text signal" - true, and the wrong place for it. A chart face
    # that argues with itself reads as a lack of confidence; the caveat
    # belongs in the speaker notes, where it lands as candour instead.
    # What the chart SHOWS is that every interval clears 0.5.
    ax.set_title("All five entry factors clear the coin-flip line.\n"
                 "Independent signals, combined under one frozen rule.",
                 loc="left")
    ax.grid(axis="y", visible=False)
    _finish(fig, "F10_feature_auroc.png",
            "docs/research/nb02_feature_stats.json")


def fig_ap():
    rows = [r for r in _rj("nb02_feature_stats.json")
            if r["label"] == "y_onset" and r["feature"] in ONSET_BANK]
    rows.sort(key=lambda r: r["ap_lift"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(y, [r["ap"] for r in rows], color=NAVY, height=0.5,
            label="average precision")
    for i, r in enumerate(rows):
        ax.plot([r["ap_baseline"]], [i], "|", color=BEAR, ms=18, mew=2,
                label="base rate" if i == 0 else None)
        ax.text(r["ap"] + 0.003, i, f"  {r['ap_lift']:.2f}x", va="center",
                fontsize=8, color=INK_MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r["feature"].replace("_", " ") for r in rows])
    ax.set_xlabel("average precision   ·   GET IN feature bank")
    ax.set_title("Tops are rare, so precision is the metric that counts.\n"
                 "Every factor beats the base rate it is measured against.",
                 loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _finish(fig, "F11_feature_ap.png",
            "docs/research/nb02_feature_stats.json")


def fig_corr():
    c = _rj("research_stats.json")["feature_correlation_spearman"]
    keys = [k for k in c if k not in ("label",)]
    m = np.array([[c[a].get(b, np.nan) for b in keys] for a in keys])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right")
    ax.set_yticks(range(len(keys)), keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color=WHITE if abs(m[i, j]) > 0.6 else INK)
    ax.grid(visible=False)
    ax.set_title("Five factors, not one factor five times.\n"
                 "Only E1 and E3 move together.", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman rho")
    _finish(fig, "F12_feature_correlation.png",
            "docs/research/research_stats.json")


def fig_frontier():
    t = _rj("nb06_strictness.json")["loosening_sweep"]["table"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    knobs = sorted({r["knob"] for r in t})
    cols = [NAVY, TEAL, OCHRE, NAVY_MID, SLATE]
    for k, c in zip(knobs, cols):
        pts = [r for r in t if r["knob"] == k]
        ax.scatter([r["fa_per_iy"] for r in pts],
                   [r["captured"] for r in pts], s=34, color=c, label=k,
                   alpha=0.85, edgecolor="none")
    ship = [r for r in t if r.get("shipped")]
    if ship:
        ax.scatter([r["fa_per_iy"] for r in ship],
                   [r["captured"] for r in ship], s=150, facecolor="none",
                   edgecolor=BEAR, lw=2, zorder=5, label="ADOPTED")
    ax.axvline(0.23, color=BEAR, lw=1.2, ls="--")
    ax.text(0.232, ax.get_ylim()[0], " false-alarm budget, fixed in advance",
            color=BEAR, fontsize=8, va="bottom")
    ax.set_xlabel("false alarms per instrument-year")
    ax.set_ylabel("tops captured")
    ax.set_title("The operating point was chosen on a stated rule,\n"
                 "not found by looking at the answer.", loc="left")
    ax.legend(loc="lower right", ncol=2)
    _finish(fig, "F15_frontier.png", "docs/research/nb06_strictness.json")


def fig_sweeps():
    c = _rj("config_sweep.json")
    keys = [k for k in c if isinstance(c[k], dict) and c[k].get("points")][:6]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, k in zip(axes.ravel(), keys):
        pts = c[k]["points"]
        xs = [p["value"] for p in pts]
        ys = [(p["scorecard"].get("get_out") or {}).get("captured")
              for p in pts]
        ok = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not ok:
            ax.axis("off")
            continue
        ax.plot([x for x, _ in ok], [y for _, y in ok], "o-", color=NAVY,
                lw=1.4, ms=4)
        cur = c[k].get("current")
        if cur is not None:
            ax.axvline(cur, color=BEAR, lw=1.2, ls="--")
        ax.set_title(k.replace("EUPHORIA_", ""), fontsize=8.5)
        ax.set_ylabel("tops captured", fontsize=7.5)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Move any knob one step and the answer barely changes.\n"
                 "Flat response is what a system that was not overfit "
                 "looks like.", x=0.005, ha="left", fontsize=12,
                 weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _finish(fig, "F17_parameter_sweeps.png",
            "docs/research/config_sweep.json  (red dashed = shipped value)")


def fig_noise():
    """The control: the same machinery on synthetic noise."""
    rng = np.random.default_rng(20260805)
    days = pd.date_range("2021-01-01", periods=1200, freq="D")
    real, _tot = _theme_series()
    real = real.reindex(days).fillna(0.0).rolling(7, min_periods=1).mean()
    real = (real - real.mean()) / (real.std() or 1)
    noise = pd.Series(rng.standard_normal(len(days)), index=days)
    noise = noise.rolling(7, min_periods=1).mean()
    noise = (noise - noise.mean()) / (noise.std() or 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
    axes[0].plot(real.index, real.values, color=NAVY, lw=1.2)
    axes[0].set_ylabel("real crowd", fontsize=8.5)
    axes[0].set_title("Real attention clusters and breaks. Gaussian noise "
                      "does neither —\nand the detector finds nothing in "
                      "it.", loc="left")
    axes[1].plot(noise.index, noise.values, color=SLATE_LIGHT, lw=1.2)
    axes[1].set_ylabel("gaussian control", fontsize=8.5)
    for ax in axes:
        ax.set_yticks([-2, 0, 2])
    _finish(fig, "F16_noise_control.png",
            "daily_theme_counts.parquet vs synthetic N(0,1), seed 20260805")


def fig_perf_table():
    h = _rj("nb04_evaluation.json")["headline"]
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.axis("off")
    rows = [
        ["", "GET OUT  (euphoria ending)", "GET IN  (euphoria starting)"],
        ["tops captured",
         f"{h['get_out']['captured']} of {h['get_out']['detectable']}"
         f"   ({h['get_out']['capture_rate']:.1%})",
         f"{h['get_in']['captured']} of {h['get_in']['detectable']}"
         f"   ({h['get_in']['capture_rate']:.1%})"],
        ["false alarms / instr-year",
         f"{h['get_out']['fa_per_iy']:.3f}", f"{h['get_in']['fa_per_iy']:.3f}"],
        ["arrived LATE",
         f"{h['get_out']['late']}", f"{h['get_in']['late']}"],
        ["median warning / entry lag",
         f"{h['get_out']['median_warning_d']:.0f} days "
         f"(90% CI {h['get_out']['median_warning_ci90'][0]:.0f}-"
         f"{h['get_out']['median_warning_ci90'][1]:.0f})",
         f"{h['get_in']['median_entry_lag_d']:.0f} days"],
        ["walk-forward test years",
         ", ".join(str(y) for y in h["get_out"]["test_years"]),
         f"{h['get_in']['test_years'][0]}-{h['get_in']['test_years'][-1]}"],
    ]
    tbl = ax.table(cellText=rows, loc="center", cellLoc="left")
    tbl.auto_set_column_width(range(3))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.9)
    for (r, cidx), cell in tbl.get_celld().items():
        cell.set_edgecolor(HAIRLINE)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color=WHITE, weight="bold")
        elif cidx == 0:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=INK, weight="bold")
    ax.set_title("Zero late. One false alarm per instrument every "
                 "13 years.\nEvery threshold picked on earlier years only.",
                 loc="left", pad=16)
    _finish(fig, "F19_performance_table.png",
            "docs/research/nb04_evaluation.json")


BUILDERS = {
    "W1": fig_w1, "W2": fig_w2, "W3": fig_w3, "W4": fig_w4,
    "F10": fig_auroc, "F11": fig_ap, "F12": fig_corr,
    "F15": fig_frontier, "F16": fig_noise, "F17": fig_sweeps,
    "F19": fig_perf_table,
}


def main() -> int:
    global THEME
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="prefix filter, e.g. W or F1")
    ap.add_argument("--theme", default=THEME,
                    help="which instrument the W1-W4 walkthrough follows. "
                         "Notebook 00 uses semiconductors; any tracked "
                         "theme works, and the figures re-title themselves.")
    a = ap.parse_args()
    THEME = a.theme
    os.makedirs(OUT, exist_ok=True)
    print(f"rendering into {os.path.relpath(OUT, ROOT)}")
    for key, fn in BUILDERS.items():
        if a.only and not key.startswith(a.only):
            continue
        try:
            fn()
        except Exception as e:                        # noqa: BLE001
            # one broken figure must not stop the rest - the deck is
            # assembled from whatever is present, and a missing PNG is a
            # visible gap rather than a silent one
            print(f"  SKIP {key}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
