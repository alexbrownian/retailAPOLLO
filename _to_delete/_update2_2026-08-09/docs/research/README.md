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


---

## August-2026 addendum — the rebuilt system's evidence pack

The figures and method notes above are the JULY (rules-era) pack and
remain the frozen record of that system. The AUGUST rebuild (robust
share estimator · ground truth 20/40, 12/25 · the tournament-selected
ensemble desk model) has its own artifacts:

| Artifact | What it holds |
|---|---|
| `ml_tournament.json` / `ml_tournament.md` | the full model tournament (5 families x 2 feature sets x 2 heads, walk-forward) + the ground-truth sweep; the md is the paste-ready thesis-style table |
| `nb02_august_bank.json` | per-feature AUROC for the four August bank additions |
| `nb03_tournament.json -> august_desk_tournament` | the selection record notebook 03 §SS writes |
| `DATA_QUALITY_2026-08.md` | the share-estimator investigation and fix |
| `conditioning_sweep.json` | round 1 of the 2026-08-09 alert study (smoothing vs F0.5 cut vs crossing trigger; smoothing rejected) |
| `alert_shape_sweep.json` | round 2 and the ADOPTED alert shape (phase gates from the 120d G2 bar, re-arm depths, 63d spacing, 21d IN/OUT separation; the `adopted` block is the shipped walk-forward record incl. per-name alert dates and forward returns) |
| `nb07_wf_scores.parquet` | notebook 07's cached walk-forward member/ensemble scores (fingerprinted; auto-recomputed when stale) |
| `nb07_slide_pack.json` | every number the slide figures quote, frozen in one place |
| `../figures/00, 06` | notebook 00's data figures, notebook 06's gold worked example |
| `../figures/slides` | notebook 07's PPT-ready slide pack (S0a–S8c, deck palette); the pre-merge pack survives in `../figures/07` |
| `../figures/deck/F20_model_tournament.png` | the comparison table as one slide |

Regenerate: `python -m analytics.ml_detector --sweep` (tournament +
sweep), then re-execute notebooks 00-04, 06 and 07. The July regenerator
(`helper/research_charts.py`) still rebuilds only the July figures.
