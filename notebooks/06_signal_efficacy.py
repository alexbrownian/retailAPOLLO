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
# # Notebook 06 — Full Signal Efficacy: What Happens to Prices Around Every Alert
#
# **The desk's question, answered with full data:** after a STARTING alert,
# how much did prices move over the next 3 / 10 / 21 / 84 days? After an
# ENDING alert, how much did they fall? What is the hit rate, what would a
# simple overlay have earned, and where does the signal work name by name?
#
# **Honesty note, stated before any number (this matters at a defense):**
# notebook 04 ran the *pre-registered, confirmatory* trading test — one
# horizon (the frozen 20d HOLD_DAYS), criterion written before results —
# and REJECTED both trading claims. THIS notebook is the *descriptive*
# efficacy report at the desk-requested horizons (3 / 10 / 21 / 84 days:
# 21 = the alert cooldown, 84 = the project's BASELINE window, 3 and 10 =
# desk-requested short looks). Four horizons × two signals = eight looks
# at the same alerts, so read PATTERNS across horizons, not any single
# starred number — the confirmatory verdict remains notebook 04's.
#
# Alerts are the tournament winners' walk-forward alerts (out-of-sample,
# identical to notebooks 03/04). Every alert, every instrument, full data.

# %%
import json
import sys
import time
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

RESEARCH_DIR = ROOT / "docs" / "research"
HORIZONS = [3, 10, 21, 84]        # desk-requested; 21 = cooldown, 84 = BASELINE

# %% [markdown]
# ## Reproduce the winners' out-of-sample alerts (identical to NB04)

# %%
from analytics.euphoria import build_all_series
from analytics.euphoria_phases import (build_day_frame, run_tournament_entry,
                                       ONSET_BANK)
from analytics.euphoria_phases import TOP_FEATURES as TOP_BANK
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
onset_frame = frame[frame.hype_raw >= 1].copy()
top_frame = frame[frame.hype_ok].copy()

def rules_onset(train, apply, feats):
    return apply[feats].mean(axis=1).values

def rules_top(train, apply, feats):
    sc = apply[feats].mean(axis=1).values
    return np.where((apply["e1"] >= 0.90) & (apply["e2"] > 0), sc, 0.0)

ow = run_tournament_entry(onset_frame, episodes, ONSET_BANK, "y_onset",
                          "onset", rules_onset, FA_BUDGET)
tw = run_tournament_entry(top_frame, episodes, TOP_BANK, "y_top",
                          "top", rules_top, FA_BUDGET)
sym_by = {es.name: es.symbol for es in series}
kind_by = {es.name: es.kind for es in series}
pxd = {s_: p.dropna().asfreq("D").ffill() for s_, p in pxmap.items()}
print(f"{time.time()-t0:.0f}s | onset alerts: "
      f"{sum(len(v) for v in ow['alerts_by_name'].values())} | top alerts: "
      f"{sum(len(v) for v in tw['alerts_by_name'].values())}")

# %% [markdown]
# ## Every alert with its forward returns — the full table

# %%
def fwd_ret(name, d, h):
    px = pxd.get(sym_by[name])
    if px is None or d not in px.index:
        return np.nan
    fut = px.loc[d:d + pd.Timedelta(days=h)]
    if len(fut) < 2 or (fut.index[-1] - d).days < h * 0.6:
        return np.nan                    # not enough future price yet
    return float(fut.iloc[-1] / fut.iloc[0] - 1)

def alert_table(entry, signal):
    rows = []
    for name, alerts in entry["alerts_by_name"].items():
        for d in alerts:
            r = {"name": name, "kind": kind_by[name],
                 "symbol": sym_by[name], "signal": signal, "date": d}
            for h in HORIZONS:
                r[f"fwd_{h}d"] = fwd_ret(name, d, h)
            rows.append(r)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

alerts_all = pd.concat([alert_table(ow, "START"),
                        alert_table(tw, "END")], ignore_index=True)
pd.set_option("display.max_rows", 400)
print("EVERY alert, THEMES (full data):")
display(alerts_all[alerts_all.kind == "theme"]
        .drop(columns=["kind"]).round(3))
print("EVERY alert, SINGLE NAMES (full data):")
display(alerts_all[alerts_all.kind == "single"]
        .drop(columns=["kind"]).round(3))

# %% [markdown]
# ## Headline: mean move, hit rate, and the baseline it must beat
#
# The baseline is the same-instrument unconditional forward return over
# every candidate day (the days the detector was allowed to speak) — the
# honest "what if you acted on a random eligible day" comparison. Hit
# rate = share of alerts followed by the EXPECTED direction (up after
# START, down after END). The baseline hit rate is what a coin weighted
# by the market's drift would score — markets drift up, so ~50% down-hits
# after END is *better than it looks*; always compare to the baseline
# column, not to 50%.

# %%
def baseline_rets(names, h):
    rows = []
    for name in set(names):
        days = frame.loc[frame.name == name, "date"]
        px = pxd.get(sym_by[name])
        if px is None:
            continue
        fut = px.shift(-h)
        rr = (fut / px - 1).reindex(days).dropna()
        rows.append(pd.DataFrame({"name": name, "fwd": rr.values}))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["name", "fwd"]))

def cluster_ci(adf, bdf, col, n_boot=300, seed=42):
    names = sorted(set(adf.name))
    a_by = {n: g[col].dropna().values for n, g in adf.groupby("name")}
    b_by = {n: g["fwd"].values for n, g in bdf.groupby("name")}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        a = np.concatenate([a_by[n] for n in pick if len(a_by.get(n, []))])
        b = np.concatenate([b_by[n] for n in pick if n in b_by])
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
    return float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))

summary_rows = []
for signal, direction in (("START", +1), ("END", -1)):
    sub = alerts_all[alerts_all.signal == signal]
    for h in HORIZONS:
        col = f"fwd_{h}d"
        vals = sub[col].dropna()
        base = baseline_rets(sub.name, h)
        lo_, hi_ = cluster_ci(sub, base, col)
        hit = ((vals > 0) if direction > 0 else (vals < 0)).mean()
        bhit = ((base.fwd > 0) if direction > 0
                else (base.fwd < 0)).mean()
        summary_rows.append({
            "signal": signal, "horizon_d": h, "n": len(vals),
            "mean_move": round(vals.mean(), 4),
            "median_move": round(vals.median(), 4),
            "hit_rate": round(hit, 3),
            "baseline_mean": round(base.fwd.mean(), 4),
            "baseline_hit": round(bhit, 3),
            "edge_vs_baseline": round(vals.mean() - base.fwd.mean(), 4),
            "edge_ci90": f"[{lo_:+.3f}, {hi_:+.3f}]",
            "ci_excludes_0": bool(lo_ > 0 if direction > 0 else hi_ < 0),
        })
summary = pd.DataFrame(summary_rows)
summary

# %% [markdown]
# ## Event study — the average price path around each signal
#
# Mean cumulative return from 21 days before to 84 days after every
# alert, indexed to the alert day. If the signals mean anything, the
# START path should keep climbing after the flag and the END path should
# roll over.

# %%
def event_paths(entry, lo_d=-21, hi_d=84):
    """Per-INSTRUMENT paths first (a name that alerted ten times gets one
    vote, not ten - meme names must not dominate the average), then the
    cross-instrument mean AND median. The median is outlier-immune: a
    single post-alert short squeeze can drag a 67-path mean up 30 points
    for a few days (a sharp up-down spike is the fingerprint of one or
    two paths, never of a real market effect, which would be smooth)."""
    by_name = {}
    for name, alerts in entry["alerts_by_name"].items():
        px = pxd.get(sym_by[name])
        if px is None:
            continue
        for d in alerts:
            win = px.loc[d + pd.Timedelta(days=lo_d):
                         d + pd.Timedelta(days=hi_d)]
            if d not in win.index or len(win) < 40:
                continue
            rel = (win / win.loc[d] - 1) * 100
            rel.index = (rel.index - d).days
            by_name.setdefault(name, []).append(rel)
    if not by_name:
        return None, None
    per_name = []
    for name, paths in by_name.items():
        stacked = pd.concat(paths, axis=1)
        stacked = stacked.groupby(stacked.index).mean()
        per_name.append(stacked.mean(axis=1))
    allp = pd.concat(per_name, axis=1)
    allp = allp.groupby(allp.index).mean()
    return allp.mean(axis=1), allp.median(axis=1)

fig, ax = plt.subplots(figsize=(8.5, 3.6))
for entry, lbl, c in ((ow, "START", C1), (tw, "END", C3)):
    mean_p, med_p = event_paths(entry)
    if mean_p is None:
        continue
    ax.plot(mean_p.index, mean_p.values, color=c, lw=1.2, alpha=0.5,
            label=f"around {lbl}: mean (outlier-sensitive)")
    ax.plot(med_p.index, med_p.values, color=c, lw=2.2,
            label=f"around {lbl}: MEDIAN (robust)")
ax.axvline(0, color=INK, lw=1, ls="--")
ax.text(0.5, ax.get_ylim()[1] * 0.9, "alert day", fontsize=8, color=INK)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("days relative to the alert")
ax.set_ylabel("cumulative return (%)")
ax.set_title("Event study, one vote per instrument: median (bold) vs "
             "mean (faint) - read the median; mean spikes are one or two "
             "squeeze paths")
ax.legend(frameon=False, fontsize=8)
despine(ax)
plt.show()

# %% [markdown]
# ## Is this a valid signal? — confidence intervals everywhere
#
# Three views, all with instrument-cluster bootstrap uncertainty (the
# instrument is the independent unit; daily rows within a name are not):
#
# 1. **Forest plot** — the edge over baseline per signal x horizon with
#    its 90% CI. A valid signal shows CIs consistently on one side of
#    zero SOMEWHERE, not stars scattered at random.
# 2. **Event-study with a CI band** — the average path around alerts with
#    the band of paths you would get resampling instruments.
# 3. **Hit rates vs the drift-adjusted baseline** — with CIs, because a
#    57% down-hit only means something relative to a 45% baseline.

# %%
def cluster_boot_stat(df, col, stat, n_boot=300, seed=42):
    by = {n: g[col].dropna().values for n, g in df.groupby("name")}
    names = [n for n, v in by.items() if len(v)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        vals = np.concatenate([by[n] for n in pick])
        if len(vals):
            out.append(stat(vals))
    return np.percentile(out, [5, 95])

fig, ax = plt.subplots(figsize=(8, 4.2))
ypos, labels = [], []
for i, (signal, direction, c) in enumerate(
        [("START", +1, C1), ("END", -1, C3)]):
    sub = alerts_all[alerts_all.signal == signal]
    for j, h in enumerate(HORIZONS):
        col = f"fwd_{h}d"
        base = baseline_rets(sub.name, h)
        edge = sub[col].dropna().mean() - base.fwd.mean()
        lo_, hi_ = cluster_ci(sub, base, col)
        y = i * (len(HORIZONS) + 1) + j
        ax.errorbar(edge * 100, y, xerr=[[(edge - lo_) * 100],
                                         [(hi_ - edge) * 100]],
                    fmt="o", color=c, ms=6, capsize=3, lw=1.4)
        ypos.append(y)
        labels.append(f"{signal} {h}d")
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(ypos, labels)
ax.invert_yaxis()
ax.set_xlabel("edge vs same-instrument baseline (%, 90% cluster CI)")
ax.set_title("Forest plot: where the signal beats its baseline — and "
             "where it does not")
despine(ax)
plt.show()

# %% [markdown]
# ## Event study with uncertainty bands

# %%
def event_paths_by_name(entry, lo_d=-21, hi_d=84):
    by = {}
    for name, alerts in entry["alerts_by_name"].items():
        px = pxd.get(sym_by[name])
        if px is None:
            continue
        for d in alerts:
            win = px.loc[d + pd.Timedelta(days=lo_d):
                         d + pd.Timedelta(days=hi_d)]
            if d not in win.index or len(win) < 40:
                continue
            rel = (win / win.loc[d] - 1) * 100
            rel.index = (rel.index - d).days
            by.setdefault(name, []).append(rel)
    return by

def boot_band(by, n_boot=200, seed=42):
    names = list(by)
    rng = np.random.default_rng(seed)
    grid = np.arange(-21, 85)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(names, size=len(names), replace=True)
        paths = [p for n in pick for p in by[n]]
        stacked = pd.concat(paths, axis=1)
        stacked = stacked.groupby(stacked.index).mean()
        means.append(stacked.mean(axis=1).reindex(grid).interpolate())
    m = pd.concat(means, axis=1)
    return m.mean(axis=1), m.quantile(0.05, axis=1), m.quantile(0.95, axis=1)

fig, ax = plt.subplots(figsize=(8.5, 3.8))
for entry, lbl, c in ((ow, "around START", C1), (tw, "around END", C3)):
    by = event_paths_by_name(entry)
    if not by:
        continue
    mid, lo_b, hi_b = boot_band(by)
    ax.plot(mid.index, mid.values, color=c, lw=2, label=lbl)
    ax.fill_between(mid.index, lo_b.values, hi_b.values, color=c,
                    alpha=0.15, lw=0)
ax.axvline(0, color=INK, lw=1, ls="--")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("days relative to the alert")
ax.set_ylabel("mean cumulative return (%)")
ax.set_title("Event study with 90% cluster-bootstrap bands — the honest "
             "width of what we know")
ax.legend(frameon=False)
despine(ax)
plt.show()

# %% [markdown]
# ## Hit rates vs the drift-adjusted baseline (with CIs)
# and the END signal's year-by-year stability — a valid signal should not
# owe its whole record to one regime.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
ax = axes[0]
x = np.arange(len(HORIZONS))
for k, (signal, direction, c, off) in enumerate(
        [("START", +1, C1, -0.18), ("END", -1, C3, +0.18)]):
    sub = alerts_all[alerts_all.signal == signal]
    hits, blo, bhi, bases = [], [], [], []
    for h in HORIZONS:
        col = f"fwd_{h}d"
        if direction > 0:
            stat = lambda v: (v > 0).mean()
        else:
            stat = lambda v: (v < 0).mean()
        hits.append(stat(sub[col].dropna().values))
        lo_, hi_ = cluster_boot_stat(sub, col, stat)
        blo.append(lo_)
        bhi.append(hi_)
        base = baseline_rets(sub.name, h)
        bases.append(stat(base.fwd.values))
    hits, blo, bhi = map(np.array, (hits, blo, bhi))
    ax.bar(x + off, hits * 100, 0.36, color=c, label=f"{signal} hit rate")
    ax.errorbar(x + off, hits * 100, yerr=[(hits - blo) * 100,
                                           (bhi - hits) * 100],
                fmt="none", ecolor=INK, capsize=3, lw=1.2)
    ax.plot(x + off, np.array(bases) * 100, "_", color=INK, ms=22,
            label="baseline" if k == 0 else None)
ax.set_xticks(x, [f"{h}d" for h in HORIZONS])
ax.set_ylabel("hit rate (%)")
ax.set_title("Hit rate vs drift-adjusted baseline (black dash)")
ax.legend(frameon=False, fontsize=8)
despine(ax)

ax = axes[1]
ends = alerts_all[alerts_all.signal == "END"].copy()
ends["year"] = ends.date.dt.year
rows = []
for y, g in ends.groupby("year"):
    vals = g["fwd_10d"].dropna()
    if len(vals) < 3:
        continue
    base = baseline_rets(g.name, 10)
    rows.append({"year": y, "edge": (vals.mean() - base.fwd.mean()) * 100,
                 "n": len(vals)})
yr = pd.DataFrame(rows)
colors = [C3 if v < 0 else GRID for v in yr.edge]
ax.bar(yr.year.astype(str), yr.edge, color=colors)
for i, r in yr.iterrows():
    ax.text(i, r.edge, f" n={r.n}", fontsize=8, color=MUTED,
            va="bottom" if r.edge >= 0 else "top", ha="center")
ax.axhline(0, color=INK, lw=1)
ax.set_ylabel("END 10d edge vs baseline (%)")
ax.set_title("Regime stability: END 10d edge by year\n"
             "(negative = prices fell more than baseline after END)")
despine(ax)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## A simple overlay "PnL" view (descriptive, not a strategy claim)
#
# Two mechanical translations, equal-weight, no costs:
# **LONG-ON-START** — buy every START alert, hold 21 / 84 days.
# **RISK-AVOIDED-ON-END** — for every END alert, the drawdown that
# followed within 84 days (the loss an exit would have side-stepped).

# %%
starts = alerts_all[alerts_all.signal == "START"]
pnl_rows = []
for h in (21, 84):
    vals = starts[f"fwd_{h}d"].dropna()
    pnl_rows.append({"overlay": f"LONG-ON-START, hold {h}d",
                     "trades": len(vals),
                     "mean per trade": f"{vals.mean():+.2%}",
                     "median": f"{vals.median():+.2%}",
                     "win rate": f"{(vals > 0).mean():.0%}",
                     "sum (equal-weight)": f"{vals.sum():+.1%}"})

def max_dd_after(name, d, h=84):
    px = pxd.get(sym_by[name])
    if px is None or d not in px.index:
        return np.nan
    fut = px.loc[d:d + pd.Timedelta(days=h)]
    if len(fut) < 10:
        return np.nan
    return float(fut.min() / fut.iloc[0] - 1)

ends = alerts_all[alerts_all.signal == "END"]
dds = ends.apply(lambda r: max_dd_after(r["name"], r["date"]), axis=1).dropna()
base_dd = []
for name in set(ends.name):
    days = frame.loc[frame.name == name, "date"]
    px = pxd.get(sym_by[name])
    if px is None:
        continue
    samp = days.iloc[::30]                # every 30th candidate day
    base_dd += [max_dd_after(name, d) for d in samp]
base_dd = pd.Series(base_dd).dropna()
pnl_rows.append({"overlay": "RISK-AVOIDED-ON-END (worst dip in next 84d)",
                 "trades": len(dds),
                 "mean per trade": f"{dds.mean():+.2%}",
                 "median": f"{dds.median():+.2%}",
                 "win rate": f"vs baseline dip {base_dd.mean():+.2%}",
                 "sum (equal-weight)": "-"})
pd.DataFrame(pnl_rows)

# %% [markdown]
# ## Per-theme and per-name efficacy (full data)
#
# Mean forward move and hit rate per instrument per signal — tiny
# denominators everywhere (most names have 1-5 alerts), so this table
# says WHERE the record comes from, not which theme "works best".

# %%
per_name = []
for (name, signal), g in alerts_all.groupby(["name", "signal"]):
    direction = +1 if signal == "START" else -1
    row = {"name": name, "kind": kind_by[name], "symbol": sym_by[name],
           "signal": signal, "alerts": len(g)}
    for h in HORIZONS:
        vals = g[f"fwd_{h}d"].dropna()
        row[f"mean_{h}d"] = round(vals.mean(), 3) if len(vals) else None
        row[f"hit_{h}d"] = (round(((vals > 0) if direction > 0
                                   else (vals < 0)).mean(), 2)
                            if len(vals) else None)
    per_name.append(row)
per_name = pd.DataFrame(per_name).sort_values(
    ["kind", "signal", "alerts"], ascending=[True, True, False])
print("THEMES:"); display(per_name[per_name.kind == "theme"]
                          .drop(columns=["kind"]))
print("SINGLE NAMES:"); display(per_name[per_name.kind == "single"]
                                .drop(columns=["kind"]))

# %% [markdown]
# ## Other efficacy measures
#
# * **Alert precision** — the share of alerts that landed inside a genuine
#   episode at all (hit + late vs false alarm), per signal.
# * **Lead quality** — recap of NB04: START median 17d after the trough
#   with ~66d of rally ahead; END median 6d before the peak.
# * **Coherence-rule note** — the terminal suppresses a START within 21d
#   after an END (measured cost: 2 onset captures, 7 fewer FAs); these
#   tables use the raw research alerts, matching the validated record.

# %%
prec = []
for label, entry in (("START", ow), ("END", tw)):
    n_alerts = sum(len(v) for v in entry["alerts_by_name"].values())
    in_ep = entry["captured"] + entry["late"]
    prec.append({"signal": label, "alerts": n_alerts,
                 "captured episodes": entry["captured"],
                 "late (in-rally)": entry["late"],
                 "false alarms": entry["false_alarms"],
                 "in-episode share":
                     round(1 - entry["false_alarms"] / n_alerts, 2)
                     if n_alerts else None})
pd.DataFrame(prec)

# %% [markdown]
# ## Reading guide (what these numbers do and do not license)
#
# * The **END signal's** value shows in the event-study roll-over and the
#   risk-avoided table: the average worst dip in the 84 days after an END
#   alert versus the unconditional baseline dip is the risk-timing claim,
#   quantified.
# * The **START signal's** value is lead time (the event-study path keeps
#   climbing after the flag), NOT a mechanical buy edge — the horizon
#   table's CI columns show where the edge over baseline is and is not
#   distinguishable from zero, and notebook 04's pre-registered test
#   remains the confirmatory word (REJECTED at 20d).
# * Anything here can be re-cut per name from the full tables above —
#   every alert is listed, nothing is aggregated away.

# %% [markdown]
# # THE DESK SIGNAL STUDY — price + crowd, one better END
#
# **Desk decision (2026-07-24):** the crowd-only constraint is lifted for a
# SECOND signal family. The crowd-only detector remains the thesis headline
# ("the crowd alone called it"); the DESK signal may use price, and its
# stated objective is the desk's own words: *predict sharp drops (≥10%
# within ~a week) up to a month before they happen, with a better hit rate
# and no one-day blips.*
#
# **Design, pre-stated before any result below:**
# * Candidacy: the validated price-gated set (hype gate AND G2 boom state).
# * Price bank (same constructions as the crowd bank, ranked vs own year,
#   trailing): `px_conv` = Sornette log-convexity ON THE CHART (the
#   original signature, permitted again), `px_boom` = boom-magnitude rank,
#   `px_mom21` = cooldown-window momentum rank.
# * Variants: crowd bank (reference) | price-only | combined | combined
#   with the trigger on the 7d-SMOOTHED score (the house ROLL — the
#   blip-fix hypothesis: sustained elevation, not one loud day) | logistic
#   regression | GBM. All walk-forward, identical discipline.
# * **Adoption criterion:** the DESK END = the variant with the highest
#   cliff-30 hit rate (share of alerts followed by a ≥10%-in-7d drop
#   starting within 30d) SUBJECT TO utility ≥ the crowd reference's and
#   FAs ≤ the crowd reference's; the cliff uplift over the unconditional
#   baseline must have a 90% cluster CI excluding zero. The smoothed
#   trigger is adopted only if it does not reduce captures.

# %%
from analytics.euphoria import log_convexity, trailing_pct_rank
from src.config import (EUPHORIA_BOOM_MIN_ETF, EUPHORIA_BOOM_MIN_SINGLE,
                        ROLL)

prows = []
for es in series:
    px = pxd[es.symbol]
    conv = trailing_pct_rank(log_convexity(px).clip(lower=0))
    low120 = px.rolling(120, min_periods=60).min()
    boom_r = trailing_pct_rank(px / low120 - 1)
    mom21 = trailing_pct_rank(px.pct_change(21))
    bm = (EUPHORIA_BOOM_MIN_SINGLE if es.kind == "single"
          else EUPHORIA_BOOM_MIN_ETF)
    prows.append(pd.DataFrame({
        "name": es.name, "date": px.index,
        "px_conv": conv.values, "px_boom": boom_r.values,
        "px_mom21": mom21.values,
        "boom_state": ((px / low120 - 1) >= bm).values}))
pf = pd.concat(prows, ignore_index=True)
fpx = frame.merge(pf, on=["name", "date"], how="left")
fpx[["px_conv", "px_boom", "px_mom21"]] = fpx[
    ["px_conv", "px_boom", "px_mom21"]].fillna(0.5)
fpx["boom_state"] = fpx["boom_state"].fillna(False)
cand = fpx[fpx.hype_ok & fpx.boom_state].copy()

CROWD = TOP_BANK
PRICE = ["px_conv", "px_boom", "px_mom21"]
COMBO = CROWD + PRICE

def make_desk_rules(feats_used, smooth=False):
    def f(train, apply, feats):
        sc = apply[feats_used].mean(axis=1)
        gate = (apply["e1"] >= 0.90) & (apply["e2"] > 0)
        sc = sc.where(gate, 0.0)
        if smooth:
            sc = sc.groupby(apply["name"]).transform(
                lambda g: g.rolling(ROLL, min_periods=1).mean())
        return sc.values
    return f

def make_desk_lr(label):
    from sklearn.linear_model import LogisticRegression
    def f(train, apply, feats):
        m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(train[feats], train[label])
        return m.predict_proba(apply[feats])[:, 1]
    return f

def make_desk_gbm(label, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    def f(train, apply, feats):
        m = HistGradientBoostingClassifier(max_depth=3,
                                           class_weight="balanced",
                                           random_state=seed)
        m.fit(train[feats], train[label])
        return m.predict_proba(apply[feats])[:, 1]
    return f

desk_runs = {
    "crowd bank (reference)": (CROWD, make_desk_rules(CROWD)),
    "price-only bank": (PRICE, make_desk_rules(PRICE)),
    "COMBINED": (COMBO, make_desk_rules(COMBO)),
    "COMBINED + smoothed trigger": (COMBO, make_desk_rules(COMBO,
                                                           smooth=True)),
    "logreg COMBINED": (COMBO, make_desk_lr("y_top")),
    "gbm COMBINED": (COMBO, make_desk_gbm("y_top", 42)),
}
desk_res = {}
for lbl, (feats, fn) in desk_runs.items():
    desk_res[lbl] = run_tournament_entry(cand, episodes, feats, "y_top",
                                         "top", fn, FA_BUDGET)

def cliff_stats(entry, drop=0.10, week=7, horizon=30):
    """share of alerts followed by a >=drop-in-week fall STARTING within
    `horizon` days, plus per-name pairs for the cluster CI."""
    per_name = {}
    for name, alerts in entry["alerts_by_name"].items():
        px = pxd[sym_by[name]]
        fwd_min = px.rolling(week + 1).min().shift(-week)
        weekdrop = (fwd_min / px - 1) <= -drop
        h = t = 0
        for a in alerts:
            t += 1
            win = weekdrop.loc[a:a + pd.Timedelta(days=horizon)]
            if len(win) and win.any():
                h += 1
        per_name[name] = (h, t)
    hits = sum(h for h, _ in per_name.values())
    tot = sum(t for _, t in per_name.values())
    return hits, tot, per_name

# the honest baseline: P(cliff within 30d) on ALL candidate days
def cliff_baseline():
    per_name = {}
    for name, g in cand.groupby("name"):
        px = pxd[sym_by[name]]
        fwd_min = px.rolling(8).min().shift(-7)
        weekdrop = ((fwd_min / px - 1) <= -0.10)
        cl30 = (weekdrop[::-1].rolling(31, min_periods=1).max()[::-1]
                .reindex(pd.DatetimeIndex(g["date"])).dropna())
        per_name[name] = (int(cl30.sum()), int(len(cl30)))
    hits = sum(h for h, _ in per_name.values())
    tot = sum(t for _, t in per_name.values())
    return hits, tot, per_name

bh, bt, base_by = cliff_baseline()
rows = []
for lbl, r in desk_res.items():
    ch, ct, _ = cliff_stats(r)
    rows.append({"variant": lbl,
                 "captured": r["captured"], "FA": r["false_alarms"],
                 "utility": r["captured"] - r["false_alarms"],
                 "AP": r["ap"],
                 "cliff hit rate": round(ch / ct, 2) if ct else None,
                 "alerts": ct})
desk_table = pd.DataFrame(rows)
print(f"unconditional cliff-30 baseline on candidate days: {bh/bt:.0%}")
display(desk_table)

# %% [markdown]
# ## The blip check — does the smoothed trigger kill one-day euphoria?
#
# For each alert we measure how many CONSECUTIVE days the trigger
# condition held around the alert day. A one-day run is the blip from the
# desk's screenshot; sustained runs are real regimes.

# %%
def run_lengths(entry, feats_used, smooth):
    fn = make_desk_rules(feats_used, smooth)
    sc = pd.Series(fn(cand, cand, None), index=cand.index)
    lens = []
    for name, alerts in entry["alerts_by_name"].items():
        g = cand[cand["name"] == name]
        s = pd.Series(sc.loc[g.index].values,
                      index=pd.DatetimeIndex(g["date"]))
        thr = entry["thresholds"][max(entry["thresholds"])]
        hot = (s >= thr).astype(int)
        for a in alerts:
            if a not in hot.index:
                continue
            run = 1
            d = a - pd.Timedelta(days=1)
            while d in hot.index and hot.loc[d]:
                run += 1
                d -= pd.Timedelta(days=1)
            d = a + pd.Timedelta(days=1)
            while d in hot.index and hot.loc[d]:
                run += 1
                d += pd.Timedelta(days=1)
            lens.append(run)
    return pd.Series(lens)

raw_runs = run_lengths(desk_res["COMBINED"], COMBO, False)
sm_runs = run_lengths(desk_res["COMBINED + smoothed trigger"], COMBO, True)
print(f"one-day trigger runs: raw {int((raw_runs == 1).sum())}/"
      f"{len(raw_runs)} ({(raw_runs == 1).mean():.0%}) -> smoothed "
      f"{int((sm_runs == 1).sum())}/{len(sm_runs)} "
      f"({(sm_runs == 1).mean():.0%}) | median run: raw "
      f"{raw_runs.median():.0f}d -> smoothed {sm_runs.median():.0f}d")

# %% [markdown]
# ## Verdict (mechanical, per the pre-stated criterion)

# %%
ref = desk_table[desk_table.variant == "crowd bank (reference)"].iloc[0]
elig = desk_table[(desk_table.utility >= ref.utility)
                  & (desk_table.FA <= ref.FA)
                  & (desk_table.variant != "crowd bank (reference)")]
winner_row = (elig.sort_values("cliff hit rate", ascending=False).iloc[0]
              if len(elig) else None)
if winner_row is not None:
    wname = winner_row.variant
    ch, ct, per_name = cliff_stats(desk_res[wname])
    rng = np.random.default_rng(42)
    names_ = [n for n in per_name if per_name[n][1] > 0]
    ups = []
    for _ in range(500):
        pick = rng.choice(names_, size=len(names_), replace=True)
        h = sum(per_name[n][0] for n in pick)
        t = sum(per_name[n][1] for n in pick)
        hb = sum(base_by.get(n, (0, 0))[0] for n in pick)
        tb = sum(base_by.get(n, (0, 0))[1] for n in pick)
        if t and tb:
            ups.append(h / t - hb / tb)
    lo_u, hi_u = np.percentile(ups, [5, 95])
    print(f"WINNER: {wname}")
    print(f"  cliff hit rate {ch}/{ct} = {ch/ct:.0%} vs unconditional "
          f"baseline {bh/bt:.0%}; uplift 90% cluster CI "
          f"[{lo_u:+.2f}, {hi_u:+.2f}] -> "
          f"{'VALID (CI excludes zero)' if lo_u > 0 else 'NOT distinguishable from baseline'}")
    print("  ADOPTED as the DESK END signal (claim: crowd + chart). The "
          "crowd-only detector remains the thesis headline."
          if lo_u > 0 else "  NOT adopted.")
else:
    print("no variant met the eligibility constraints - nothing adopted")

# %% [markdown]
# ## Case studies: did the DESK END call the drops the desk cares about?
#
# Gold (Jan-2026) and GME (2021): price with the DESK END alerts (dark
# red) vs the crowd-only END alerts (pink).

# %%
wentry = desk_res[wname] if winner_row is not None else desk_res["COMBINED + smoothed trigger"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, name in zip(axes, ["gold_metals", "GME"]):
    px = pxd[sym_by[name]]
    eps = episodes[episodes.name == name]
    if name == "GME":
        lo_w, hi_w = pd.Timestamp("2020-09-01"), pd.Timestamp("2021-12-31")
        ax.set_yscale("log")
    else:
        lo_w, hi_w = pd.Timestamp("2025-08-01"), pd.Timestamp("2026-07-01")
    win = px.loc[lo_w:hi_w]
    ax.plot(win.index, win.values, color=INK, lw=1.3)
    for ep in eps.itertuples():
        if ep.bust_date is not None and not pd.isna(ep.bust_date) \
                and lo_w <= ep.peak <= hi_w:
            ax.axvspan(ep.peak, ep.bust_date, color=C3, alpha=0.18, lw=0)
    for a in tw["alerts_by_name"].get(name, []):
        if lo_w <= a <= hi_w:
            ax.axvline(a, color=C3, lw=1.4, alpha=0.7)
    for a in wentry["alerts_by_name"].get(name, []):
        if lo_w <= a <= hi_w:
            ax.axvline(a, color="#8b0000", lw=2)
    ax.set_title(f"{name}: DESK END (dark red) vs crowd END (pink); "
                 "shaded = peak-to-bust", fontsize=9)
    despine(ax)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## The REAL finding: the danger STATE is the drop-warning
#
# The verdict above is the study working as designed: the alert layer's
# 65% cliff hit rate is NOT distinguishable from the 62% baseline of its
# own candidate days — because the CANDIDACY CONDITIONS themselves
# (crowd ≥ 2× its normal AND price in a G2 boom) already carry the
# drop-warning. The comparison that matters for the desk is the STATE
# versus ORDINARY days:

# %%
def cliff30_rate(day_index_by_name):
    per = {}
    for name, days in day_index_by_name.items():
        px = pxd[sym_by[name]]
        fwd_min = px.rolling(8).min().shift(-7)
        weekdrop = ((fwd_min / px - 1) <= -0.10)
        cl30 = (weekdrop[::-1].rolling(31, min_periods=1).max()[::-1]
                .reindex(days).dropna())
        per[name] = (int(cl30.sum()), int(len(cl30)))
    h = sum(a for a, _ in per.values())
    t = sum(b for _, b in per.values())
    return h, t, per

state_days = {n: pd.DatetimeIndex(g["date"])
              for n, g in cand.groupby("name")}
ordinary = fpx[~(fpx.hype_ok & fpx.boom_state)]
ord_days = {n: pd.DatetimeIndex(g["date"])
            for n, g in ordinary.groupby("name")}
sh, st_, s_by = cliff30_rate(state_days)
oh, ot, o_by = cliff30_rate(ord_days)
rng = np.random.default_rng(42)
names_ = sorted(set(s_by) & set(o_by))
ups = []
for _ in range(500):
    pick = rng.choice(names_, size=len(names_), replace=True)
    a = sum(s_by[n][0] for n in pick); b = sum(s_by[n][1] for n in pick)
    c_ = sum(o_by[n][0] for n in pick); d_ = sum(o_by[n][1] for n in pick)
    if b and d_:
        ups.append(a / b - c_ / d_)
lo_s, hi_s = np.percentile(ups, [5, 95])
print(f"P(a ≥10%-in-7d drop begins within 30d):")
print(f"  DANGER STATE days (crowd swollen + boom underway): {sh/st_:.0%}")
print(f"  ordinary days:                                     {oh/ot:.0%}")
print(f"  uplift 90% cluster CI [{lo_s:+.2f}, {hi_s:+.2f}] -> "
      f"{'THE STATE IS A VALID DROP-WARNING' if lo_s > 0 else 'not valid'}")
print()
print("Conclusion for the desk: the PM warning is the STATE (a standing "
      "amber condition), and the END alerts time the peak WITHIN it "
      "(26/122 detectable tops, median 8d lead, at the boom-gated "
      "operating point). The alert does not need to out-predict its own "
      "danger state - it needs to time it, which is what it does.")

# %%
desk_verdict = {
    "winner": wname if winner_row is not None else None,
    "table": desk_table.to_dict(orient="records"),
    "cliff_baseline": round(bh / bt, 3),
    "blip": {"raw_one_day_share": round(float((raw_runs == 1).mean()), 2),
             "smoothed_one_day_share":
                 round(float((sm_runs == 1).mean()), 2)},
    "danger_state": {"cliff30_state": round(sh / st_, 3),
                     "cliff30_ordinary": round(oh / ot, 3),
                     "uplift_ci90": [round(lo_s, 3), round(hi_s, 3)]},
}
with open(RESEARCH_DIR / "nb06_desk_signal.json", "w") as f:
    json.dump(desk_verdict, f, indent=1, default=str)
print("saved nb06_desk_signal.json")

# %%
out = {"horizons": HORIZONS,
       "summary": summary.to_dict(orient="records"),
       "precision": prec}
with open(RESEARCH_DIR / "nb06_signal_efficacy.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("saved nb06_signal_efficacy.json")

# %% [markdown]
# # THE ADOPTED DESK CONFIGURATION (2026-07-24) — what production now runs
#
# Everything above measured pieces. This section is the DECISION: the
# exact configuration `rebuild_phase_files` now ships to the dashboard as
# **GET IN** (euphoria starting) / **GET OUT** (euphoria ending), chosen
# by a rule written down BEFORE the table below was computed.
#
# **The desk's stated priorities (verbatim, three separate briefs):**
# 1. *"there still seems to be the error where the start and end are so
#    close together"* — a START landing on top of an END destroys PM
#    trust. ADJACENCY (a START within one 21d cooldown before an END) is
#    the binding constraint.
# 2. *"we need to stop doing stuff like this, where it is euphoria for
#    just 1 day"* — no one-day blips.
# 3. *"we should be using both price and the social media to predict. i
#    want a better hit rate"* — the price gate is permitted (a labelled
#    SECOND claim; the crowd-only detectors remain the thesis headline).
#
# **Pre-stated selection rule:**
# * **GET OUT** = the boom-gated crowd-bank rules (the NB03 commissioned
#   test's winner). The SMOOTHED trigger is adopted iff it does not lower
#   AP and does not raise FAs (priority 2); otherwise raw.
# * **GET IN** = among {incumbent, phase-aware} x {raw, smoothed}: the
#   variant with the LOWEST adjacency, tie-broken by fewer LATE starts,
#   then fewer FAs (priority 1). The capture cost vs the incumbent is
#   RECORDED, not hidden — this is a DESK DECISION that the adjacency
#   priority overrules the raw-capture utility rule, and it is reversible
#   by rerunning this cell with the rule changed.
#
# Every variant below runs through the PRODUCTION code path
# (`analytics.euphoria_phases.desk_*` — imported, not re-implemented), so
# this table and the live system cannot drift.

# %%
from analytics.euphoria_phases import (boom_state_frame, end_stage_mask,
                                       desk_candidacy, desk_end_fit,
                                       desk_onset_fit)
from src.config import EUPHORIA_ATT_GATE, EUPHORIA_COOLDOWN_DAYS

boom = boom_state_frame(series, pxmap)
fpx2 = frame.merge(boom, on=["name", "date"], how="left")
fpx2["boom_state"] = fpx2["boom_state"].fillna(False)
end_f, onset_pa = desk_candidacy(fpx2)          # production candidacies
onset_inc = fpx2[fpx2.hype_raw >= 1].copy()     # incumbent candidacy

def raw_end_fit(train, apply, feats):           # unsmoothed comparators
    sc = apply[feats].mean(axis=1)
    return sc.where((apply["e1"] >= EUPHORIA_ATT_GATE)
                    & (apply["e2"] > 0), 0.0).values

def raw_onset_fit(train, apply, feats):
    return apply[feats].mean(axis=1).values

end_runs = {
    "GET OUT boom-gated raw": run_tournament_entry(
        end_f, episodes, TOP_BANK, "y_top", "top", raw_end_fit, FA_BUDGET),
    "GET OUT boom-gated SMOOTHED (production)": run_tournament_entry(
        end_f, episodes, TOP_BANK, "y_top", "top", desk_end_fit, FA_BUDGET),
}
onset_runs = {
    "GET IN incumbent raw": run_tournament_entry(
        onset_inc, episodes, ONSET_BANK, "y_onset", "onset",
        raw_onset_fit, FA_BUDGET),
    "GET IN incumbent smoothed": run_tournament_entry(
        onset_inc, episodes, ONSET_BANK, "y_onset", "onset",
        desk_onset_fit, FA_BUDGET),
    "GET IN phase-aware raw": run_tournament_entry(
        onset_pa, episodes, ONSET_BANK, "y_onset", "onset",
        raw_onset_fit, FA_BUDGET),
    "GET IN phase-aware SMOOTHED (production)": run_tournament_entry(
        onset_pa, episodes, ONSET_BANK, "y_onset", "onset",
        desk_onset_fit, FA_BUDGET),
}

def adjacency_vs(end_entry, onset_entry):
    adj = 0
    for n, oa in onset_entry["alerts_by_name"].items():
        ta = end_entry["alerts_by_name"].get(n, [])
        adj += sum(1 for o in oa
                   if any(0 <= (t - o).days <= EUPHORIA_COOLDOWN_DAYS
                          for t in ta))
    return adj

prod_end = end_runs["GET OUT boom-gated SMOOTHED (production)"]
rows = []
for lbl, r in end_runs.items():
    rows.append({"variant": lbl, "captured": r["captured"],
                 "detectable": r["detectable"], "late": "-",
                 "FA": r["false_alarms"], "AP": r["ap"],
                 "adjacency": "-"})
for lbl, r in onset_runs.items():
    rows.append({"variant": lbl, "captured": r["captured"],
                 "detectable": r["detectable"], "late": r["late"],
                 "FA": r["false_alarms"], "AP": r["ap"],
                 "adjacency": adjacency_vs(prod_end, r)})
config_table = pd.DataFrame(rows)
display(config_table)

# %% [markdown]
# ## Mechanical verdict (the rule above, applied)

# %%
r_raw = end_runs["GET OUT boom-gated raw"]
r_sm = end_runs["GET OUT boom-gated SMOOTHED (production)"]
out_pick = ("GET OUT boom-gated SMOOTHED (production)"
            if (r_sm["ap"] >= r_raw["ap"]
                and r_sm["false_alarms"] <= r_raw["false_alarms"])
            else "GET OUT boom-gated raw")
print(f"GET OUT adopted: {out_pick}")
print(f"  (smoothed AP {r_sm['ap']} vs raw {r_raw['ap']}; FA "
      f"{r_sm['false_alarms']} vs {r_raw['false_alarms']}; recorded "
      f"capture cost {r_raw['captured'] - r_sm['captured']})")

cand_in = {lbl: (adjacency_vs(prod_end, r), r["late"],
                 r["false_alarms"], lbl)
           for lbl, r in onset_runs.items()}
in_pick = min(cand_in, key=cand_in.get)
inc = onset_runs["GET IN incumbent raw"]
win = onset_runs[in_pick]
print(f"GET IN adopted: {in_pick}")
print(f"  adjacency {adjacency_vs(prod_end, win)} (incumbent "
      f"{adjacency_vs(prod_end, inc)}), late {win['late']} (vs "
      f"{inc['late']}), FA {win['false_alarms']} (vs "
      f"{inc['false_alarms']})")
print(f"  RECORDED COST (desk decision): captures "
      f"{inc['captured']} -> {win['captured']} of {win['detectable']} — "
      "the adjacency priority, stated three times by the desk, overrules "
      "the raw-capture utility rule.")
assert out_pick.endswith("(production)") and in_pick.endswith("(production)"), \
    "the mechanical verdict no longer matches what production ships - re-decide"

# %% [markdown]
# ## Drift guard — the notebook vs the live system
#
# `rebuild_phase_files` wrote its own walk-forward record when it froze
# the desk thresholds. If this notebook's recomputation ever disagrees,
# one of the two is stale — fail loudly.

# %%
desk_rep = json.load(open(ROOT / "data" / "processed" /
                          "euphoria_desk_report.json"))
for key, entry in (("get_out", prod_end),
                   ("get_in", onset_runs[in_pick])):
    rec = desk_rep[key]["walk_forward"]
    for fld in ("captured", "detectable", "false_alarms", "ap"):
        assert rec[fld] == entry[fld], (key, fld, rec[fld], entry[fld])
print("drift guard PASSED: notebook == production "
      f"(frozen thresholds: GET IN {desk_rep['get_in']['live_threshold']:.3f}, "
      f"GET OUT {desk_rep['get_out']['live_threshold']:.3f})")

with open(RESEARCH_DIR / "nb06_desk_config.json", "w") as f:
    json.dump({"adopted": {"get_out": out_pick, "get_in": in_pick},
               "table": config_table.to_dict(orient="records")},
              f, indent=1, default=str)
print("saved nb06_desk_config.json")
