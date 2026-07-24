# PARAMETER REGISTER — every number, its class, its reason

*The rule of this project: a number is either LEARNED by a pre-stated
procedure, DERIVED from an already-accepted quantity, a round CONVENTION
whose contribution is measured (ablation), part of the price-side
GROUND TRUTH (testing only, never prediction), or a dated DESK DECISION
with its rationale recorded. Anything that fits none of those classes
does not ship. Updated with every change (see RESEARCH_REPORT changelog).*

## Class 1 — LEARNED (fitted by walk-forward; no human picks the value)

| Number | Current value | How it is learned | Evidence |
|---|---|---|---|
| TOP alert threshold | 85 (level units) | per test year: maximise `captures − 1.0×FAs` on strictly earlier years; ties → most conservative | NB03 "why the threshold" exhibit; per-year table in `euphoria_report.json`. **Recorded limitation:** under this utility in an FA-rich regime the selection saturates toward the conservative end of its 50–85 grid — the budget rule below is the successor selection and is interior |
| ONSET alert threshold | ~0.90 (score units) | per test year: maximise captures SUBJECT TO the FA budget on strictly earlier years; grid = percentiles of the training scores (data-derived, not hand units) | NB03 exhibit (budget rule marked inside the feasible band); `euphoria_onset_report.json` |
| ML challenger cutoffs | (rejected models) | same walk-forward discipline, probability grid from training percentiles | NB03 tournament; euphoria.py `ml_walk_forward` |
| GET OUT threshold (desk) | 0.630 (score units) | budget rule on full prior years, boom-gated smoothed score | NB06 adopted-configuration section; `euphoria_desk_report.json` |
| GET IN threshold (desk) | 0.848 (score units) | budget rule on full prior years, phase-aware smoothed score | NB06 adopted-configuration section; `euphoria_desk_report.json` |

## Class 2 — DERIVED (computed from an already-accepted quantity)

| Number | Value | Derivation |
|---|---|---|
| FA budget | 0.23 /instr-yr | the incumbent top detector's documented, desk-accepted walk-forward FA rate — a new detector may not be noisier than the noise already accepted (read live from `euphoria_report.json`) |
| Onset prerequisite gate | 1× own 120d median | the A1 hype-gate construction with multiplier **one** — "attention above its own normal", parameter-free |
| Onset hit window | 45 days | mirrors the existing 45d false-alarm horizon in `score_alerts` (an alert is false if no peak follows within 45d) |
| Panel qualification bar | 100 unique referrers / 28d | literally `EUPHORIA_MIN_COVERAGE` reused — the same floor that makes a name measurable (a unit test asserts the equality) |
| Singles display bar | 2× hype at alert | the existing A1 constant applied at display time — no new number |
| DANGER STATE (amber band) | A1 2× hype AND G2 boom state | pure composition of two existing constants; measured: cliff-30 = 62% in-state vs 19% ordinary (CI [+28pp,+50pp], NB06) — shipped as the standing PM warning |
| Price-assisted END gate | G2's boom thresholds (25%/50% above trailing 120d low) | the ground-truth boom definition applied as a LIVE gate (past prices only, no look-ahead, no new number) — changes the claim to 'crowd + chart'; offered as a labelled second signal (NB03) and ADOPTED as GET OUT candidacy (2026-07-24) |
| End-stage mask (phase-aware GET IN) | A1 ∧ A2 ∧ A3-persistence | pure composition of the END detector's own frozen gates — a day satisfying every ending gate cannot host a "start"; no new constant (unit-tested) |

## Class 3 — CONVENTION (round a-priori units; contribution measured by ablation)

| Number | Value | Why this unit | What the ablation says |
|---|---|---|---|
| E1/A2 attention gate | 0.90 pct | "top decile = extreme" — decile convention | removing it: capture unchanged, +3 FAs → not load-bearing; kept as a belt |
| E2 persistence | 75% of posting days | "¾ of days" — the desk's *super bullish AND for a long time* hypothesis | removing the gate: +3 captures, +12 FAs → mild precision help |
| A1 hype multiple | 2× own 120d median | "genuinely swollen = at least double its normal" | **the precision lever**: without it +227 FAs for +13 captures |
| Fade discount | 10 level-points | the fade is historically the last pre-top stage; a round tenth of the scale | fade off: capture −0.067 (the biggest capture lever) |
| Cooldown | 21 days | ~one trading month = one episode, and identical to the signal engine's pre-existing `SIG_COOLDOWN` | enforced by tests; also the coherence-rule and display windows |
| Rolling windows | 7 / 14 / 28 d | week / fortnight / 4 weeks — calendar units (ROLL=7 predates this study) | feature battery (NB02) evaluates each feature built on them |
| Desk trigger smoothing | 7 d (ROLL, trailing) | the house one-week window, reused | measured raw vs smoothed (NB06): GET OUT AP 0.435→0.449, FA 41→39, −2 captures; GET IN adjacency 8→2 on the phase-aware frame; one-day blips structurally removed |
| LPPLS fit window | 60 d | ~one quarter of trading days for a stable quadratic fit | E5's contribution: −0.008 capture if dropped |
| Percentile window | 365 d (min 180) | "extreme for this name" = vs its own last year; half-year minimum before speaking | trailing-rank no-look-ahead test |
| Hype baseline | 120 d median | ~half a year of "normal" to compare a week against | inside A1 (see above) |
| Z baseline | 84 d | pre-existing project constant (~4 trading months) | conviction study (legacy, validated) |

## Class 4 — GROUND TRUTH (price side; used only to grade, never to predict)

| Number | Value | Why |
|---|---|---|
| Peak local-max window | ±21 d (43 d) | month-scale "the highest close around here" |
| Boom minimum | +25% ETF / +50% single | dual thresholds are a recorded desk decision — singles are structurally more volatile |
| Bust minimum | −15% ETF / −30% single, within 90 d | same dual-threshold decision; a quarter to confirm the break |
| Boom lookback | 120 d | the window the trough is measured in (this right-truncates run-length at 120 — a recorded caveat, NB01) |
| Top hit window | [peak−30d, peak+1d] | the stated aim of the project |
| Judgeable horizon | 45 d of future price | an unjudgeable alert is PENDING, not false |
| Label sensitivity | 20/40 and 30/60 probes | robustness sweep values (not fitted — they test that conclusions survive ±1 step) |

## Class 5 — DESK DECISIONS (dated; rationale + evidence in DECISIONS.xlsx)

| Decision | Date | One-line reason |
|---|---|---|
| Crowd-only prediction (price never an input) | Jul 2026 | the claim defended is "the crowd alone called it"; enforced by unit test |
| LATE ≠ FALSE bucket | 2026-07-24 | mid-rally onset alerts are neither hits nor false alarms |
| Onset window capped at peak | 2026-07-24 | a post-top alert must not count as "caught the start" |
| `source_breadth` rejected | 2026-07-24 | best raw AUROC but a coverage-regime artifact (X/StockTwits exist only from 2026) |
| Rules over learners (parsimony CI rule) | 2026-07-24 | GBM tied inside bootstrap noise; MLP below random |
| Trading translation rejected | 2026-07-24 | pre-stated criterion not met at the frozen 20d horizon (NB04); NB06 quantifies the descriptive END 10d edge |
| Research/live split | 2026-07-24 | thresholds train on strictly earlier years → intra-year recompute is a no-op |
| Episode coherence, ASYMMETRIC | 2026-07-24 | symmetric rule tested and rejected (cost half the top captures); only START-after-END is suppressed |
| Comments decoupled; dynamic panel (cap 1/month) | 2026-07-24 | slow fetch out of the daily path; panel expands where the crowd points, one denominator step per month |
| Phase-aware onset tested & REJECTED | 2026-07-24 | halves start/end adjacency and LATE alerts but costs 5 captures with no utility gain (NB03); adjacency readability solved by episode-span shading instead (display, zero capture cost) — **superseded the same day by the desk configuration (below)** |
| DESK CONFIGURATION adopted (GET IN / GET OUT) | 2026-07-24 | the desk stated three times that a START landing on an END destroys PM trust → adjacency overrules the raw-capture utility rule. GET OUT = boom-gated + 7d-smoothed END (cap 24/122, FA 39, AP 0.449); GET IN = phase-aware + 7d-smoothed onset (adjacency 20→2, LATE 21→10, FA 169→124, capture cost 29→20 RECORDED). Selection rule pre-stated; NB06 drift guard pins notebook == production |

*Full derivations: `analytics/euphoria.py` and `analytics/euphoria_phases.py`
docstrings, `src/config.py` inline comments, `docs/DECISIONS.xlsx`,
notebooks 01–06.*
