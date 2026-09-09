# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Key parameters: the evidence behind each value
#
# This notebook re-derives the tables in `reference/KEY_PARAMETERS.md`
# from the frozen record in `reference/research_record/` and from the
# price store. It writes nothing. Run it after any research pass so the
# document and the record cannot drift apart.
#
# Sections follow the document: ground truth, candidacy, features,
# model and selection, operating point, inflection, display, bot screen.

# %%
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(
    globals().get("__file__", os.path.join(os.getcwd(), "x")))))
if not os.path.exists(os.path.join(ROOT, "src", "config.py")):
    ROOT = os.getcwd()
sys.path.insert(0, ROOT)

from src import config as C                                   # noqa: E402

RECORD = os.path.join(ROOT, "reference", "research_record")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def rec(name):
    """Load one frozen JSON record by stem."""
    with open(os.path.join(RECORD, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


# %% [markdown]
# ## 1. Ground truth
#
# The frozen bars, as `src/config.py` holds them.

# %%
pd.Series({
    "EUPHORIA_BOOM_MIN_ETF": C.EUPHORIA_BOOM_MIN_ETF,
    "EUPHORIA_BOOM_MIN_SINGLE": C.EUPHORIA_BOOM_MIN_SINGLE,
    "EUPHORIA_CRASH_MIN_ETF": C.EUPHORIA_CRASH_MIN_ETF,
    "EUPHORIA_CRASH_MIN_SINGLE": C.EUPHORIA_CRASH_MIN_SINGLE,
    "EUPHORIA_BOOM_LOOKBACK_D": C.EUPHORIA_BOOM_LOOKBACK_D,
    "EUPHORIA_CRASH_WINDOW_D": C.EUPHORIA_CRASH_WINDOW_D,
    "EUPHORIA_PEAK_LOCAL_MAX_D": C.EUPHORIA_PEAK_LOCAL_MAX_D,
    "EUPHORIA_PEAK_MERGE_D": C.EUPHORIA_PEAK_MERGE_D,
}, name="value")

# %% [markdown]
# ### 1a. The bar sweep
#
# All four bars scaled together. The adopted row is the 0.8x setting
# (0.20 / 0.12 / 0.40 / 0.24, with the single-name bust bar rounded to
# 0.25 in config).

# %%
gt = pd.DataFrame(rec("sweep_ground_truth")["rows"])
gt[["setting", "boom_etf", "crash_etf", "boom_single", "crash_single",
    "peaks", "detectable", "captured", "rate_of_detectable", "fa_per_iy",
    "instruments_with_any_episode"]]

# %% [markdown]
# ### 1b. The lookback sweep

# %%
pd.DataFrame(rec("sweep_lookback")["lookback_sweep"])

# %% [markdown]
# ### 1c. What the record holds

# %%
e = rec("nb01_episode_stats")
pd.Series({k: e[k] for k in ("episodes_total", "onset_detectable",
                              "top_detectable", "median_run_days",
                              "median_boom_pct", "median_bust_pct")})

# %%
pd.DataFrame(e["per_year"]).T

# %% [markdown]
# ### 1d. Unconditional baselines: why single names carry a higher bar
#
# For every instrument that has at least one episode, the trailing
# 120-day run-up from the window low and the forward 90-day maximum
# drawdown on every day of history. The bars are read against these.

# %%
prices_path = os.path.join(ROOT, "data", "prices", "prices.parquet")
episodes_path = os.path.join(ROOT, "data", "processed", "episodes.parquet")
if os.path.exists(prices_path) and os.path.exists(episodes_path):
    px = pd.read_parquet(prices_path)
    ep = pd.read_parquet(episodes_path)
    kinds = ep.drop_duplicates("symbol").set_index("symbol")["kind"]
    wide = (px.pivot(index="date", columns="symbol", values="px_last")
              .sort_index())
    wide = wide[[c for c in wide.columns if c in kinds.index]]
    runup = wide / wide.rolling(C.EUPHORIA_BOOM_LOOKBACK_D,
                                min_periods=60).min() - 1
    fwd_min = wide[::-1].rolling(C.EUPHORIA_CRASH_WINDOW_D,
                                 min_periods=30).min()[::-1]
    drawdown = fwd_min / wide - 1
    rows = []
    for kind, boom_bar, bust_bar in (
            ("theme", C.EUPHORIA_BOOM_MIN_ETF, C.EUPHORIA_CRASH_MIN_ETF),
            ("single", C.EUPHORIA_BOOM_MIN_SINGLE, C.EUPHORIA_CRASH_MIN_SINGLE)):
        cols = [c for c in wide.columns if kinds[c] == kind]
        r, d = runup[cols].stack(), drawdown[cols].stack()
        rows.append({
            "kind": kind, "instruments": len(cols),
            "runup_median": r.median(), "runup_p75": r.quantile(0.75),
            "share_days_over_boom_bar": (r >= boom_bar).mean(),
            "drawdown_median": d.median(),
            "share_days_under_bust_bar": (d <= -bust_bar).mean(),
            "boom_bar_over_median": boom_bar / r.median(),
            "bust_bar_over_median": bust_bar / -d.median(),
        })
    baselines = pd.DataFrame(rows).set_index("kind").round(3)
else:
    baselines = "price store or episodes not present on this copy"
baselines

# %% [markdown]
# ## 2. Candidacy gates

# %%
pd.Series({
    "EUPHORIA_MIN_COVERAGE (A0)": C.EUPHORIA_MIN_COVERAGE,
    "EUPHORIA_HYPE_MULT (A1)": C.EUPHORIA_HYPE_MULT,
    "EUPHORIA_ONSET_HYPE_MIN": C.EUPHORIA_ONSET_HYPE_MIN,
    "EUPHORIA_ATT_GATE (A2)": C.EUPHORIA_ATT_GATE,
    "EUPHORIA_BOOM_WINDOW_D": C.EUPHORIA_BOOM_WINDOW_D,
    "EUPHORIA_PCT_WINDOW": C.EUPHORIA_PCT_WINDOW,
    "EUPHORIA_MIN_HISTORY": C.EUPHORIA_MIN_HISTORY,
    "EUPHORIA_EXCLUDED_THEMES": ", ".join(sorted(C.EUPHORIA_EXCLUDED_THEMES)),
}, name="value")

# %% [markdown]
# Why each near-miss day was not called (the census), and the loosening
# sweep with the shipped values marked.

# %%
st = rec("nb06_strictness")
pd.DataFrame({h: st["near_miss_census"][h]["by_cause"]
              for h in st["near_miss_census"]}).fillna(0).astype(int)

# %%
pd.DataFrame(st["loosening_sweep"]["table"])

# %% [markdown]
# ## 3. Features
#
# Per-feature separation (walk-forward, with bootstrap intervals). Every
# feature is weak alone; the bank is judged in combination.

# %%
fs = pd.DataFrame(rec("nb02_feature_stats"))
(fs.pivot(index="feature", columns="label", values=["auroc", "ap_lift"])
   .round(3))

# %%
pd.DataFrame(rec("research_stats")["feature_correlation_spearman"]).round(2)

# %% [markdown]
# ## 4. Model and selection
#
# The tournament, both heads, walk-forward test years only.

# %%
t = rec("ml_tournament")
print("winner:", t["results"].get("winner"))
rows = []
for head, fams in t["results"].items():
    if not isinstance(fams, dict):
        continue
    for fam, r in fams.items():
        rows.append({"head": head, "family": fam,
                     "capture_rate": r.get("capture_rate"),
                     "auroc": r.get("auroc"),
                     "ap_lift": (r["ap"] / r["ap_baseline"]
                                 if r.get("ap") and r.get("ap_baseline") else None),
                     "fa_per_iy": r.get("fa_per_iy"),
                     "precision": r.get("precision"),
                     "median_lead_days": r.get("median_lead_days")})
pd.DataFrame(rows).round(3)

# %% [markdown]
# The production record the dashboard quotes.

# %%
report_path = os.path.join(ROOT, "data", "processed", "euphoria_desk_report.json")
if not os.path.exists(report_path):
    report_path = os.path.join(ROOT, "DASHBOARD_DATA", "euphoria_desk_report.json")
if os.path.exists(report_path):
    with open(report_path, encoding="utf-8") as f:
        rep = json.load(f)
    print("model:", rep.get("model"))
    print("selection rule:", rep.get("selection_rule"))
    prod = {h: {k: rep["tournament"][h][rep["model"]].get(k)
                for k in ("capture_rate", "median_lead_days", "auroc", "ap",
                          "ap_baseline", "fa_per_iy")}
            for h in rep.get("tournament", {})}
    display(pd.DataFrame(prod).T)   # noqa: F821  (notebook display)
else:
    print("no production record on this copy")

# %% [markdown]
# What the price features add: crowd-only vs crowd + price.

# %%
pb = rec("nb08_price_blind")
pd.DataFrame({k: {m: v.get(m) for m in ("auroc", "ap", "ap_baseline",
                                        "captured", "false_alarms")}
              for k, v in pb["results"]["get_in"].items()}).T

# %% [markdown]
# ## 5. Operating point and alert shape

# %%
pd.DataFrame(rec("operating_point_sweep"))[
    ["head", "beta", "capture_rate", "precision", "fa_per_iy", "auroc", "ap"]]

# %%
cs = rec("conditioning_sweep")
print("adoption rule:", cs["adoption_rule"])
pd.DataFrame({k: v for k, v in cs["results"]["GET OUT"].items()}).T

# %%
cfg = rec("nb06_desk_config")
print("adopted:", cfg["adopted"])
pd.DataFrame(cfg["table"])

# %%
pd.Series({
    "EUPHORIA_FA_BUDGET_PER_IY": C.EUPHORIA_FA_BUDGET_PER_IY,
    "EUPHORIA_FA_PENALTY": C.EUPHORIA_FA_PENALTY,
    "EUPHORIA_STRICT_BETA": C.EUPHORIA_STRICT_BETA,
    "EUPHORIA_COOLDOWN_DAYS": C.EUPHORIA_COOLDOWN_DAYS,
    "EUPHORIA_ALERT_SPACING_D": C.EUPHORIA_ALERT_SPACING_D,
    "HOLD_DAYS": C.HOLD_DAYS,
}, name="value")

# %% [markdown]
# ## 6. Inflection marker

# %%
inf = rec("inflection_trigger_sweep")
print("label:", inf["label"])
pd.DataFrame({"adopted": {"config": inf["adopted"]["config"],
                          "flags": inf["adopted"]["flags"],
                          "hit_rate": inf["adopted"]["hit_rate"],
                          "lift": inf["adopted"]["lift"]},
              "inherited": {"config": inf["inherited_from_euphoria_heads"]["config"],
                            "flags": inf["inherited_from_euphoria_heads"]["flags"],
                            "hit_rate": inf["inherited_from_euphoria_heads"]["hit_rate"],
                            "lift": inf["inherited_from_euphoria_heads"]["lift"]}})

# %%
pd.DataFrame(inf["adopted"]["by_year"])

# %% [markdown]
# ## 7. Display: the gauge zones

# %%
gz = rec("gauge_zones")
print(gz["outcome"])
print("amber edge:", gz["amber_edge"], "| red edge:", gz["red_edge"],
      "| base rate:", round(gz["base_rate"], 3))
pd.DataFrame(gz["edge_grid"])

# %% [markdown]
# ## 8. Bot screen
#
# The settings and the last stored run, if any.

# %%
from src import settings                                        # noqa: E402

pd.Series({k: settings.get(k) for k in (
    "bot_screen_enabled", "bot_screen_threshold",
    "bot_screen_duplicate_jaccard", "bot_screen_burst_posts_per_day")},
    name="value")

# %%
last = os.path.join(ROOT, "data", "reference", "bot_screen_last.json")
if os.path.exists(last):
    with open(last, encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=1))
else:
    print("no bot-screen report stored yet (runs with the next refresh)")
