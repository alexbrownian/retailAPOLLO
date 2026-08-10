"""
build_ml_table.py
=================
Render the 2026-08 model-tournament record
(`docs/research/ml_tournament.json`, written by
`python -m analytics.ml_detector --sweep`) as the two presentation
artifacts the deck uses:

  docs/research/ml_tournament.md            the thesis-style tables, in
                                            markdown (paste-ready)
  docs/figures/deck/F20_model_tournament.png the same table as one
                                            figure for the slides

Run after any research pass that rewrites the JSON:
  python tools/build_ml_table.py
"""

from __future__ import annotations

import json
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "docs", "research", "ml_tournament.json")
OUT_MD = os.path.join(ROOT, "docs", "research", "ml_tournament.md")
OUT_PNG = os.path.join(ROOT, "docs", "figures", "deck",
                       "F20_model_tournament.png")

LABEL = {"rules": "Hand rules (incumbent)",
         "logit": "Logistic regression",
         "logit_crowd": "Logistic regression (crowd-only)",
         "gbm": "Monotone gradient boosting",
         "gbm_crowd": "Monotone GBM (crowd-only)",
         "mlp": "Neural network (MLP 16-8)",
         "mlp_crowd": "MLP (crowd-only)",
         "ens": "Ensemble: logit + monotone GBM",
         "ens_crowd": "Ensemble (crowd-only)"}
ORDER = ["rules", "logit_crowd", "gbm_crowd", "mlp_crowd", "ens_crowd",
         "logit", "gbm", "mlp", "ens"]
HEAD_TITLE = {"get_out": "GET OUT — calling the top",
              "get_in": "GET IN — calling the start"}


def _rows(head_results, winner):
    rows = []
    names = [m for m in ORDER if m in head_results] + \
            [m for m in head_results if m not in ORDER and m != "winner"]
    for m in names:
        r = head_results[m]
        if not isinstance(r, dict) or "error" in r:
            continue
        ap, base = r.get("ap"), r.get("ap_baseline")
        fwd21 = (r.get("forward_returns", {})
                 .get("fwd_21d", {}).get("median_pct"))
        rows.append({
            "Model": LABEL.get(m, m) + (" *" if m == winner else ""),
            "AP": f"{ap:.3f}" if ap is not None else "—",
            "AP lift": (f"{ap / base:.2f}×" if ap and base else "—"),
            "AUROC": (f"{r['auroc']:.3f}"
                      if r.get("auroc") is not None else "—"),
            "Episodes caught": (f"{r.get('captured')}/{r.get('detectable')}"
                                f" ({100 * (r.get('capture_rate') or 0):.0f}%)"),
            "Hit rate": (f"{r['precision']:.0%}"
                         if r.get("precision") is not None else "—"),
            "FA / instr-yr": (f"{r['fa_per_iy']:.2f}"
                              if r.get("fa_per_iy") is not None else "—"),
            "Lead (d)": (str(r["median_lead_days"])
                         if r.get("median_lead_days") is not None else "—"),
            "Move 21d after": (f"{fwd21:+.1f}%" if fwd21 is not None
                               else "—"),
        })
    return rows


def build() -> int:
    with open(SRC) as f:
        doc = json.load(f)
    res = doc["results"]
    winner = res.get("winner", "?")

    md = ["# The model tournament — every approach, one table",
          "",
          f"*Walk-forward test years only; winner (\\*): **{LABEL.get(winner, winner)}**, "
          "selected by the pre-stated rule (one model family for both "
          "heads, highest combined AP lift; ties AUROC, then fewer "
          "false alarms). Raw AP is not comparable across candidacy "
          "frames — the incumbent's gates give it a frame where roughly "
          "half the candidate days are already positive — so **AP lift** "
          "(AP over its own frame's base rate) is the column that "
          "compares. 'Hit rate' = share of alerts that caught a real "
          "episode; 'Move 21d after' = median price change in the month "
          "after an alert (a good GET OUT is flat-to-negative, a good "
          "GET IN positive).*", ""]

    tables = {}
    for head in ("get_out", "get_in"):
        rows = _rows(res[head], winner)
        tables[head] = rows
        cols = list(rows[0].keys())
        md.append(f"## {HEAD_TITLE[head]}")
        md.append("")
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "---|" * len(cols))
        for r in rows:
            md.append("| " + " | ".join(r[c] for c in cols) + " |")
        md.append("")

    sw = doc.get("ground_truth_sweep")
    if sw:
        md += ["## Ground-truth sensitivity (the episode definition sweep)",
               "",
               "| definition (boom ETF/single, crash ETF/single) | "
               "episodes | GET OUT AP | GET OUT capture | GET IN AP | "
               "GET IN capture |", "|---|---|---|---|---|---|"]
        for k, v in sw.items():
            go, gi = v["get_out"], v["get_in"]
            md.append(
                f"| {k} | {v['episodes']} | {go['ap']} | "
                f"{go['captured']}/{go['detectable']} "
                f"({100 * (go['capture_rate'] or 0):.0f}%) | {gi['ap']} | "
                f"{gi['captured']}/{gi['detectable']} "
                f"({100 * (gi['capture_rate'] or 0):.0f}%) |")
        md.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {OUT_MD}")

    # ---- the deck figure ------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.6))
    fig.suptitle("Model tournament — walk-forward test years, "
                 "one family for both heads (* = adopted)",
                 fontsize=13, y=0.99)
    for ax, head in zip(axes, ("get_out", "get_in")):
        rows = tables[head]
        cols = list(rows[0].keys())
        cell = [[r[c] for c in cols] for r in rows]
        ax.axis("off")
        t = ax.table(cellText=cell, colLabels=cols, loc="center",
                     cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(8.3)
        t.scale(1, 1.35)
        for j in range(len(cols)):
            t[0, j].set_facecolor("#1F3A5F")
            t[0, j].set_text_props(color="white", weight="bold")
        for i, r in enumerate(rows, start=1):
            if r["Model"].endswith("*"):
                for j in range(len(cols)):
                    t[i, j].set_facecolor("#E8F0E9")
        ax.set_title(HEAD_TITLE[head], fontsize=11, loc="left", pad=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
