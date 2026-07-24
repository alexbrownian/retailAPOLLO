# Research evidence pack

Regenerate everything (figures + stats) from the CURRENT data with:

    python helper/research_charts.py

Every figure traces to this one script - nothing is hand-drawn, so after
the comment backfill or the 2017 price extension the whole pack refreshes
in one command. Numbers below were computed at build time and live in
`research_stats.json`.

| Figure | What it shows | The one-line takeaway |
|---|---|---|
| fig1_walkforward | Per-year detectable peaks / captured / false alarms | The signal lives where the coverage lives (2021, 2026); 2023-25 the detector is blind, not wrong - the denominator says so |
| fig2_ablation | Metric change when each rule is removed | Hype gate = the precision lever; fade trigger = the capture lever; every rule has a measured job |
| fig3_lead_times | Distribution of days-before-peak for captured alerts | Median 6d before the peak - inside the aim's window, on the early side |
| fig4_correlation | Spearman correlation of features + label (candidate days) | E1 and E3 are heavily correlated (rho 0.79) - which is exactly why single-feature ablation understates them; the label column shows no single feature is a magic bullet within the gated population |
| fig5_feature_separation | Each feature's separation on TWO populations: all coverage-OK days (top) vs candidate days (bottom), with point-biserial r and AUC | THE division-of-labour finding: attention (E1) FINDS the dangerous population (AUC 0.61 unconditionally) but cannot rank within it (AUC<0.5 gated) - a selection feature, correctly a GATE. Sustained bullishness (E2) is the reverse (no unconditional edge, best in-gate ranker, AUC 0.60) - correctly in the LEVEL. Independent corroboration of the ablation |
| fig6_calibration | P(detectable peak within 30d) by euphoria-level bucket | Honest read: NOT monotone within candidates - the alert zone (80-100) runs ~1.3-1.5x the 18% base rate; the 40-50 bump is thin-sample structure. The tested object is the ALERT (gates+fade+threshold), not the raw dial |
| fig7_ml_challenger | Rules vs walk-forward logistic regression + its coefficients | Rules kept under a pre-stated criterion; the ML ranks the same features top - independent evidence the rules are not arbitrary |
| fig8_case_studies | Three captured tops: price, level, the red alert line | What the headline metric looks like on real names |

Method notes for the deck:
- All tests run on **candidate days** (days passing the non-fitted hype +
  coverage gates) so they evaluate the fitted part of the system against
  the same population it operates on.
- "Pre-peak" label = a coverage-detectable ground-truth peak within the
  next 30 days (the aim's own window).
- Point-biserial r is the Pearson correlation of a feature with the 0/1
  label; AUC is threshold-free ranking power (0.5 = chance).
- Price appears in figures ONLY as ground truth / illustration - it is
  never an input to the euphoria level or the alert (desk rule, enforced
  by a unit test).
