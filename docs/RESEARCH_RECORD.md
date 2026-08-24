# Research Record

The consolidated research record for retailAPOLLO: aim, data, ground
truth, feature studies, model selection, the production configuration
and its walk-forward numbers, negative results, and the parameter
register. This file supersedes the former `RESEARCH_REPORT.md`,
`PARAMETER_REGISTER.md`, and the narrative files in `docs/research/`;
the machine-readable evidence (every JSON cited below) remains in
`docs/research/`. Dated decision provenance: `docs/DECISIONS.md`.

Conventions. "Walk-forward" always means: thresholds and models fitted
on strictly earlier years, scored blind on the held-out year.
"Detectable" means the coverage gate (A0, ≥100 tagged posts in the
trailing 28 days) held during the relevant window — the detector is
blind where coverage is absent, not wrong, and every rate is reported
against both denominators where they differ.

---

## 1. Aim and problem formulation

Detect the start and end of retail-euphoria episodes from crowd data
(mention counts and scored sentiment across 17 finance subreddits, X,
and StockTwits). Success is defined before validation: a GET OUT inside
[peak − 30d, peak + 1d] of a price-defined top; a GET IN inside
[trough, min(trough + 45d, peak)]. LATE (an onset alert after the
window but before the peak) is its own bucket — neither hit nor false.
Alerts are judged only where price truth exists at the alert and for
45 days after; recent alerts are pending, not false.

Two eras of the predictive contract:

- **Crowd-only era (July 2026).** Price never entered any predictive
  input; it defined ground truth and scored the detectors only
  (enforced by unit test). This era's frozen record is §6.1.
- **Desk era (August 2026, production).** Price admitted as two model
  features and a phase gate in a second, clearly labelled signal
  family. The crowd-only pair is still computed on every run as the
  "what can the crowd alone see" record (§6.3).

## 2. Data

- **Crowd**: daily text-free aggregates (committed `ABSTRACTED_DATA/`
  contract; a write-time FORBIDDEN_COLS check blocks raw text), 2017 →
  present. Sentiment VADER-scored at ingestion.
- **Prices**: Bloomberg daily closes, ~447 symbols, 2017 → present;
  ground truth and scoring only.
- **Universe**: 59 instruments — 34 theme-anchor ETFs (rates/bonds and
  real-estate excluded) plus the top-25 most-mentioned single names,
  ranked over the trailing 365 days (data-chosen: the aim is catching
  the next mania, not the last one's list).
- **Dynamic panel**: a monthly review can add at most one subreddit per
  month into an exploration tier (≥100 unique referring authors/28d,
  finance-content screen); the founding 17 are frozen core; every
  addition is logged because a panel change steps the share
  denominator.
- **Coverage**: deep 2020–2022 and 2026; the 2023–2025 archive gap was
  closed in August 2026 (§9). Post-restoration, every quarter holds
  ~10,000–17,500 ticker-mention rows.

### 2.1 The robust share estimator (data-quality fix, 2026-08)

Ticker share-of-forum lines intermittently printed zero on days the
name was demonstrably discussed (308 zero-share days across the top 30
tickers over three months). Diagnosis from the stores: (1) the pipeline
pulls ~2×/week, so daily totals swing 246 → 4,033 posts between
adjacent days, and a mean-of-daily-ratios estimator weighted a 10-post
day equally with a 4,000-post day and printed hard zeros on skipped
days; (2) StockTwits and X only ramp from 2026, and a pooled
denominator diluted Reddit-history names wholesale on big pull days.
The replacement (`analytics/robust_share.py`) uses ratio-of-sums over
the trailing window (Cochran's ratio estimator), stratified by source,
with shrinkage — no invented data; masking applies only when the whole
trailing window carries fewer than MIN_TOTAL posts. Every attention
series, chart and feature reads it. Evidence: the estimator module and
`docs/research/config_sweep.json`'s companion checks.

## 3. Ground truth

A peak is a local 21-day maximum whose close sits above the boom bar
relative to the minimum of the preceding 120 days, followed by a bust
within 90 days; peaks under 30 days apart merge (higher close wins).
An episode extends the peak back to the trough (argmin of the same
120-day window) and forward to the bust date. Final bars (sweep,
2026-08-07): boom ≥ 20% ETF / 40% single name; bust ≥ 12% / 25%. The
sweep compared 25/50, 20/40, 15/30, each through the full walk-forward:
20/40 yields 494 episodes, improves AP on both heads (GET OUT 0.168 →
0.201, GET IN 0.211 → 0.290), and keeps every named reference episode;
15/30 was rejected because at ~1.5 episodes per instrument-year an
"episode" is any ordinary correction. Evidence:
`docs/research/boom_min_sweep.json`, `sweep_ground_truth.json`.

Grading windows are deliberately separate from live gates: the 120-day
run-up lookback is the yardstick and lives beside the episode
definition; the live boom gate uses 54 days (§7, Class 4).

## 4. Features

**Design rules.** All features trailing; all percentile ranks against
the instrument's own trailing 365 days; every window inherited from an
already-validated constant. The bank is hand-specified, not searched:
the independent unit of evidence is the instrument (59), so a search
scoring thousands of candidate transforms on 183k daily rows would
select on noise. Cost stated openly: per-feature AUROCs are only
0.51–0.60, and a search would likely have posted better in-sample
numbers.

**Top bank**: E1 attention extremity; E2 sustained bullishness
(persistence-gated); E3 crowd influx; E5 super-exponential attention
growth (LPPLS-style convexity applied to attention, not price); E4 fade
flag (attention maximal while mood rolls over).

**Onset bank**: attention_accel (7d − 28d share), hype_ratio (7d share
÷ own 120d median — the A1 gate made continuous), bull_inflection,
influx_speed, attention_convexity (≡ E5).

**Findings.** No single feature is a detector; gates and combination do
the work. E2 is anti-predictive at onset (AUROC 0.48) — sustained
bullishness has not built when a rally starts, direct evidence the
onset task needs its own bank. The division of labour is measured:
attention (E1) finds the dangerous population (AUC 0.61 unconditional)
but cannot rank within it — correctly a gate; E2 is the reverse —
correctly in the level. A sixth candidate, `source_breadth`, posted the
best raw AUROC (0.65) and was **rejected**: X/StockTwits exist only
from 2026, so it encodes the coverage regime, not crowd behaviour.
Evidence: `docs/research/nb02_feature_stats.json`,
`nb02_august_bank.json`.

## 5. Model selection

### 5.1 Crowd-only tournament (July 2026)

Rules vs logistic regression vs GBM vs MLP vs random, identical inputs
and gates, criterion pre-stated (AP primary, AUROC ties, parsimony CI
rule). The rules won both tasks: GBM tied on AP inside the paired
bootstrap CI ([−0.017, +0.019] → parsimony), the MLP fell below random
(the small-label warning reproduced). Random's 43 "captures" at 508
false alarms is the recorded caution: capture without precision is
spraying. Evidence: `docs/research/nb03_tournament.json`.

### 5.2 Desk tournament (August 2026, production)

With price admitted (two features: price_runup, price_ret21), four
families × two heads × crowd-only/crowd+price, walk-forward, selection
rule pre-stated: one family serves both heads; highest combined AP
lift (AP over own frame's base rate — raw AP is not comparable across
candidacy frames); ties by AUROC, then fewer false alarms. Key rows
(full table: `docs/research/ml_tournament.json`):

| Head | Model | AP lift | AUROC | Captured | FA/instr-yr |
|---|---|---|---|---|---|
| GET OUT | Hand rules | 1.13× | 0.549 | 12% | 0.09 |
| GET OUT | Logit (crowd+price) | 1.93× | 0.688 | 47% | 0.47 |
| GET OUT | GBM (crowd+price) | 2.04× | 0.715 | 45% | 0.57 |
| GET OUT | **Ensemble logit+GBM** | **2.45×** | 0.717 | 42% | 0.35 |
| GET IN | Hand rules | 1.20× | 0.550 | 9% | 0.18 |
| GET IN | Logit (crowd+price) | 2.61× | 0.748 | 69% | 0.52 |
| GET IN | **Ensemble logit+GBM** | 2.5× | 0.734 | 57% | — |

Winner: the logistic + monotone-GBM rank ensemble ("ens"), both heads.
It selected identically on every subsequent research pass and is pinned
(`DESK_MODEL_FAMILY`); re-opening the tournament is one config change
and is required if the bank, labels, or universe change.

## 6. Production configuration and operating record

### 6.1 Crowd-only era record (frozen)

Onset: 23.2% of detectable starts captured, median entry 17 days after
the trough with a median 66 days of rally ahead, 0.35 FA/instr-yr
(50% above the 0.23 budget — stated, not hidden). Top: ~23% of
detectable peaks, median 6-day lead, 0.13 FA/instr-yr. A
price-assisted variant captured 46% — the documented price of the
crowd-only claim. Calibration is honest: non-monotone in low deciles,
steep where the gates live.

### 6.2 Desk era record (current frozen record, post-restoration)

| | GET IN | GET OUT |
|---|---|---|
| Cut / strict cut | 0.945 / 0.971 | 0.945 / 0.970 |
| Captured / detectable | 173/304 (57%) | 137/308 (44%) |
| False alarms | 207 | 295 |
| AUROC | 0.734 | 0.731 |
| AP / base | 0.257 / 0.105 | 0.260 / 0.095 |
| Median lead | 19 d | 14 d |

Shaping (both operating points): GET IN pre-boom only, re-arms at the
train-median score; GET OUT post-boom only (the ground truth's own
120-day bar), re-arms at the cut; one call per name per quarter per
side (63d spacing); no GET IN within 21d of a GET OUT. Evidence:
`docs/research/alert_shape_sweep.json`. GET IN is ungated in the desk
view: the phase gate blocked ~3/4 of correct GET IN calls (28 → 116
captured episodes ungated at under double the false alarms); GET OUT
keeps its gate — ungated its false alarms double.

Two operating points are stored on every run: F1 (`get_in`/`get_out`)
and F0.5 precision-weighted (`*_strict`). The dashboard reads F1 (a
top visible on the chart should produce a call on the chart — the F0.5
cut declines calls missing it by ~0.01 of probability); this record
quotes F0.5-consistent walk-forward figures where noted. Measured
trade, F1 vs F0.5 (`docs/research/operating_point_sweep.json`): GET
OUT capture 34→45% at precision 37→32%; GET IN 39→57% at 58→46%;
FA/instr-yr roughly doubles. The beta sweep also established that
under the pre-stated FA budget no loosening of the strict beta is
admissible on the OUT side, and only 0.5 is admissible on the IN side
— the operating point, not any single instrument, decides.

Signed readiness: one displayed number in [−1, +1] whose sign follows
the phase routing, so GET IN and GET OUT can never both read 100% —
impossible by construction, fenced by test
(`docs/research/nb08_single_dial.json`).

Retail-flow dial: a −1..+1 crowd-flow gauge (anchored fast half + slow
tilt, steady-state Kalman with the gain chosen per fold on train-year
IC), measured to lead price by 1–3 weeks; display context only, cannot
fire or gate a call (`docs/research/nb08_retail_flow.json`).

### 6.3 The price-blind pair (standing research record)

Thirteen crowd measurements, no price features, no phase gate: AUROC
0.55–0.575 both heads, AP 1.3–1.4× base — real but modest signal, and
roughly half the achievable lift comes from combining crowd with
price. Computed on every run (columns `*_xp`); not exposed on the
dashboard. Evidence: `docs/research/nb08_price_blind.json`.

### 6.4 Inflection marker

Context, never a call: fires at tops and bottoms with no direction;
hit rate 9.6% vs 5.6% base (lift 1.72×); beat its base rate in 4 of 6
years, failed 2022–23 (recorded weakness). Swept over cut × re-arm ×
spacing: only the cut separates (0.95); the others follow the heads'
convention. Evidence: `docs/research/inflection_trigger_sweep.json`.

### 6.5 The danger state

On days when the crowd is ≥2× its own normal AND price is in a boom, a
≥10%-in-7-days drop begins within 30 days 62% of the time versus 19%
on ordinary days (cluster CI [+28pp, +50pp]). Shipped as the standing
amber band — pure composition of two existing constants.

### 6.6 Negative results (tested and rejected, kept on the record)

- **Trading translation** (onset→buy, top→sell) at the frozen 20-day
  hold: both rejected under a pre-stated criterion (onset→buy −0.63%
  vs baseline; top→sell CI spans zero at n=67). The detectors'
  demonstrated value is risk timing and monitoring, not standalone
  alpha. Forward-return medians at the strict operating point are
  directionally right but small: post-GET-OUT excess −1.4%/21d,
  post-GET-IN +0.7% (`docs/research/max_performance.json`, which also
  records the rejected "max performance" operating point whose
  in-sample edge did not survive out of sample).
- **Influence convergence as an indicator**: pre-registered, null.
- **A persistence rule for the watch track**: proposed, measured,
  rejected on its own evidence.
- **The crowd-only ML challenger (July)**: rejected by parsimony.

### 6.7 Influence tracker

Chan (2026) method end-to-end on the live text-free store (~100k
authors, 326k calls, 114k judged, base rate 0.52): volatility-scaled
call judging, Bayesian shrinkage, composite usefulness tiers,
reply-graph PageRank, loud-but-wrong flag. The companion model
(`docs/research/nb05_influence.json`): logit AP 0.098 ± 0.024 against
a 0.047 floor (permutation p = 0.005); every graph rung rejected under
the paired 10-seed CI rule — measured node homophily 0.095 positive
class vs 0.963 negative explains why; and on a tenure split the model
sits at the random floor for unseen authors, so the dashboard tab is
information-only and ranks by the measured record. Display units are
rescalings, not new quantities: influence_index = 100·composite/max;
backing_share = share of summed conviction, read against the derived
even-share line 100/n.

## 7. Parameter register (condensed)

Every number in the system belongs to one class. Full values live in
`src/config.py` (all final); dated provenance in `docs/DECISIONS.md`.

**Class 1 — Learned (walk-forward only).** The alert thresholds and
model cuts. Chosen per test year on strictly earlier years by the
budget rule (maximise captures subject to the FA budget); current
frozen values in `data/processed/euphoria_desk_report.json`.

**Class 2 — Derived.** The FA budget (0.23/instr-yr — the incumbent's
accepted rate, frozen as a constant so the bar can neither race nor
ratchet); the onset hit window (45d, mirrors the FA horizon); the
panel qualification bar (= the A0 coverage floor, asserted equal by
test); the danger state (composition of A1 and G2); the end-stage mask
(composition of the END gates).

**Class 3 — Convention.** Round a-priori units whose contribution is
ablation-measured: ROLL 7, the 21d cooldown, 28d windows, the 63d
spacing (3× cooldown), equal feature weights in the level.

**Class 4 — Ground truth.** The episode definition (§3). The live boom
gate (54d) is deliberately a different window from the grading
lookback (120d): the sweep showed capture flat at 21–22 across 52–60d
while false alarms rise monotonically, a capture cliff below 52d, and
that "harmonising" the grading window to 54d empties the ground truth
entirely. Changing the gate invalidates the frozen thresholds.

**Class 5 — Recorded decisions.** See `docs/DECISIONS.md`.

**Class 6 — Influence tracker.** Information-only surface; nothing in
it feeds the euphoria level, the calls, or any price claim.

**Class 7 — Ingestion budget.** One human-chosen number
(PIPELINE_BUDGET_S = 600); every other number measured or derived
(§`src/config.py` 4b). The page allowance self-corrects from the
machine's own stage-time ledger.

**Class 8/9 — Display and cadence contracts.** A live run never
re-selects; staleness is reported, not repaired; no performance
numbers on operational surfaces; display fixes may never move a
threshold (each display change is classed and fenced).

**The knob audit** (all chosen numbers swept; verdicts): boom lookback
120→54d (adopted); GET IN floor 1.0→1.10 (adopted — the only setting
inside budget); A1 2.0× kept (it is the capture maximum); A2 0.90 kept
(no in-budget improvement); cooldown 21d kept (7d double-fires the
same peak); smoothing ROLL 7 kept. Five inherited knobs (ROLL,
BASELINE, MIN_DAYS, DERIV_SMOOTH, MIN_TOTAL) measured to not reach the
live flags at all — display and legacy paths only
(`docs/research/config_sweep.json`).

## 8. Display integrity (selected findings)

Recorded because each was a measured defect or a measured refusal:

- The level chart once drew a threshold that had not fired anything:
  83% of GET OUT alerts sat below the drawn line because the flags
  come from the desk score, not the level. Rule adopted: draw the
  threshold that gated the flags being drawn, plot the series that
  crossed it. No threshold moved.
- The gauge "always shows calm" because pages rank by most recent
  signal while the needle reports today — both correct at once. Fix:
  the window's high-water mark drawn behind the needle, dated. No edge
  moved.
- Handle masking is a convention (no ground truth for "offensive"):
  97 of 12,528 handles masked, zero collisions, display-layer only
  (the join key is never rewritten), deliberately under-masking.

## 9. Historical coverage restoration (2026-08)

The 2023–2025 archive gap (~1,000 mention rows/quarter vs a healthy
10–16k; 49 of 55 instruments with zero scored days across 2024–25) was
closed by a dedicated fold path: `tools/backfill_reddit.py` (resumable,
chunked; ledger keyed by window and subreddit set) plus
`tools/fold_historical.py` (per-(file, month) ledger; text-free; hard
refusal to touch days ≥ LIVE_START). The rewritten fetcher (limit=auto,
minimal fields, header pacing) pulled 1.82M posts in 19.9 minutes;
the two folds restored every drought quarter with the overlap block
adding +60 rows (dedup held). Every fold was followed by the mandatory
research re-freeze. Quarterly restoration tables: `docs/DECISIONS.md` and the fold
ledger.

## 10. Limitations

1. Small event counts: three test years carry most episodes; one
   episode moves a capture rate by a point.
2. The onset FA rate has run above the 0.23 budget in every
   configuration that captures meaningfully; the current record states
   0.207–0.35 depending on era. Carried as a stated limitation, never
   silently re-budgeted.
3. VADER label noise; sentiment is a coarse instrument.
4. Run length is right-truncated at 120 days by construction.
5. The gate does most of the work at short boom windows; detector
   skill (AUROC ≈ 0.5 within-gate for the crowd-only score) is a
   standing open question, addressed in production by the crowd+price
   ensemble (AUROC 0.73).
6. The influence model does not generalise to unseen authors.
7. X and StockTwits history is short (2026-heavy); robust_share
   stratifies by source so this cannot dilute shares, but their
   history is indicative only.

## 11. Evidence index

All machine-readable evidence remains in `docs/research/`:
`nb01_episode_stats`, `nb02_feature_stats`, `nb02_august_bank`,
`nb03_tournament`, `nb04_evaluation`, `nb04_final_eval`,
`nb05_influence`, `nb06_desk_config`, `nb06_desk_signal`,
`nb06_signal_efficacy`, `nb06_strictness`, `nb07_index_composite`,
`nb07_performance_battery`, `nb07_wf_scores.parquet`,
`nb08_inflection`, `nb08_price_blind`, `nb08_single_dial`,
`nb08_single_state`, `nb08_retail_flow`, `nb08_crossings`,
`ml_tournament`, `alert_shape_sweep`, `boom_min_sweep`,
`config_sweep`, `conditioning_sweep`, `gauge_zones`,
`inflection_trigger_sweep`, `max_performance`,
`operating_point_sweep`, `sweep_ground_truth`, `sweep_lookback`,
`research_stats`, plus the agentic/AI records (`nb09`, `nb10`).
The research notebooks (00–08) regenerate these from data.

## 12. References

Barber & Odean (2008); Barber, Huang, Odean & Schwarz (2022, JFE) —
attention-induced buying and reversal. Sornette et al. — LPPLS
super-exponential bubble signature (applied here to attention).
Chan, J. J. J. (2026), *Informed Trading Decisions via Social Network
Analysis*, Oxford M.Eng — influence method, evaluation discipline, and
the graph-learning benchmark. r/WallStreetBets literature (Jame et
al.) — herding and per-name normalisation. Hanley & McNeil (1982);
van Rijsbergen (1979); Matthews (1975) — metric foundations.
