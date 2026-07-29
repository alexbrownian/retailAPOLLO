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
| GET OUT threshold (desk) | **re-fit pending** at the 54 d gate (offline projection ≈0.6186); 0.617489 at 60 d, 0.630231 at 120 d | budget rule on full prior years, boom-gated smoothed score | NB06 adopted-configuration section; `euphoria_desk_report.json` |
| GET IN threshold (desk) | 0.848 (score units) | budget rule on full prior years, phase-aware smoothed score | NB06 adopted-configuration section; `euphoria_desk_report.json` |

## Class 1b — THE GAUGE'S TWO EDGES (added 2026-07-27, notebook 06)

*The dial on every chart needs a green/amber/red boundary, and "why 76?"
is the first question a speedometer invites. It introduces exactly ONE new
number: the red edge is a number the project already froze, and the needle
is a curve the project already plots.*

| Number | Value | Class | How it is obtained |
|---|---|---|---|
| Gauge NEEDLE | the 7d-smoothed level, 0-100 | **DERIVED — no new quantity** | literally `lvl = lvl_raw.rolling(ROLL).mean()`, the same object the chart's lower panel draws, read at its last day. The dial is a *readout of the curve beneath it*, not a second calculation, so the two cannot disagree — the defect that would destroy a PM's trust fastest. A unit test asserts `fig.data[0].value == lvl.iloc[-1]` |
| Gauge DELTA reference | `ROLL` = 7 days back | **DERIVED** | the same window the curve is smoothed over, so "the change" is a change in the plotted quantity and not a change in daily noise. No new constant |
| Gauge RED edge | 85 (level units) | **LEARNED — the SAME number, reused** | read at import time out of `data/processed/euphoria_report.json`'s `thresholds` dict, i.e. the level the walk-forward already selected for the END alert (85 in every test year 2018–2026). The gauge is forbidden from inventing a second, softer threshold; a unit test asserts equality with the report, and the notebook asserts the walk-forward agreed across years — if it ever disagrees, a single red edge would be a fiction and the code says so rather than averaging it |
| Gauge AMBER edge | **76** (level units) | **LEARNED by a pre-stated rule** — *the one new number* | the lowest cut on a 2-point grid from 68 whose 95% paired cluster-bootstrap lower bound on `P(drop \| level>=L) − P(drop \| level<L)` excludes zero **under all 5 seeds**. Measured: cut 74 flips sign across seeds (−0.0004, +0.0009, −0.0012, −0.0007, +0.0004) and cut 76 does not (+0.0045, +0.0060, +0.0049, +0.0034, +0.0046). **A cut that changes sign with the random seed is not a parameter.** Grid, per-seed lower bounds and the rule are all in `docs/research/gauge_zones.json` (`edge_grid`), written by notebook 06 |
| Gauge outcome definition | ">=10% fall over 7d STARTS within 30d" | **GROUND TRUTH (unchanged)** | the same outcome NB06 already grades the desk signal on, i.e. the desk's own "<1 month" horizon. Implemented as a reversed rolling max (a forward-looking `any()`); the last `30+7 = 37` days are set NaN because scoring an incomplete look-ahead as "no drop" would bias the base rate down *exactly at the live edge* |
| Measured meaning of each band | see below | **MEASURED (183,394 name-days, 59 instruments, 2017-06-29 → 2026-06-15)** | base rate **23%**. `level>=76`: **27%** (n=11,235, diff +0.039 CI [+0.006,+0.074]). `level>=85`: **30%** (n=4,203, +0.073 [+0.021,+0.128]). Danger state alone: **54%** (n=3,819, +0.312 [+0.222,+0.401]). `level>=85` AND danger state: **71%** (n=454, +0.482 [+0.343,+0.595]). Every percentage the dial's caption prints is read out of this JSON — none is typed into `dashboard.py`, and a unit test greps the function body to prove it |

**The honest limitation, stated on the dial's own face.** The level ALONE
is a weak read: at the red edge it is 30% against a 23% base rate, i.e.
**~1.3x**. The strong read is the level **together with** the danger
state, 71% against 23%, i.e. **~3.1x**. The caption therefore quotes the
band the needle is actually in rather than a generic legend, and says so;
a unit test pins the ordering (`P(red AND danger) > P(red)`) so the dial
can never be re-worded into implying the needle is sufficient.

**REJECTED, and why (no code left behind).** The obvious test — compute a
95% CI for `P(drop \| level>=L)` and a 95% CI for the base rate, and take
the lowest cut where they stop overlapping — found **nothing significant
at any cut from 40 to 95**. That is not a finding, it is the
overlapping-confidence-interval fallacy: with 59 instruments both
intervals are wide, and two overlapping intervals do not mean the
difference is zero. Bootstrapping the **DIFFERENCE** (state minus
not-state) on the *same* resampled instruments cancels the shared
instrument-level noise and recovers a significant effect at every cut
from 76 up. Recorded here and in NB06's prose; the first method ships
nowhere.

**Why the bootstrap resamples NAMES, not DAYS.** Adjacent days on one
instrument are the same episode. A day-level bootstrap would treat 4,000
days of one mania as 4,000 independent facts and report an interval
several times too tight. Instruments are the independent unit here, so
they are the resampling unit — the same cluster convention NB04/NB06
already use.

**The dial is a STATE, never an instruction.** GET IN and GET OUT come
from the walk-forward detector and can fire with the needle anywhere. A
unit test asserts the words "get in" and "get out" can never appear in a
band label, and the caption under every dial says it in words.

## Class 2 — DERIVED (computed from an already-accepted quantity)

| Number | Value | Derivation |
|---|---|---|
| FA budget | 0.23 /instr-yr | the incumbent top detector's documented, desk-accepted walk-forward FA rate — a new detector may not be noisier than the noise already accepted (read live from `euphoria_report.json`) |
| Onset prerequisite gate | 1× own 120d median | the A1 hype-gate construction with multiplier **one** — "attention above its own normal", parameter-free |
| Onset hit window | 45 days | mirrors the existing 45d false-alarm horizon in `score_alerts` (an alert is false if no peak follows within 45d) |
| Panel qualification bar | 100 unique referrers / 28d | literally `EUPHORIA_MIN_COVERAGE` reused — the same floor that makes a name measurable (a unit test asserts the equality) |
| Singles display bar | 2× hype at alert | the existing A1 constant applied at display time — no new number |
| DANGER STATE (amber band) | A1 2× hype AND G2 boom state | pure composition of two existing constants; measured: cliff-30 = 62% in-state vs 19% ordinary (CI [+28pp,+50pp], NB06) — shipped as the standing PM warning |
| Price-assisted END gate | G2's boom SIZE thresholds (25%/50%) over a trailing **60d** low (`EUPHORIA_BOOM_WINDOW_D`, was 120d until 2026-07-29) | the ground-truth boom definition applied as a LIVE gate (past prices only, no look-ahead, no new number) — changes the claim to 'crowd + chart'; offered as a labelled second signal (NB03) and ADOPTED as GET OUT candidacy (2026-07-24) |
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
| Desk trigger smoothing | 7 d (ROLL, trailing) | the house one-week window, reused | measured raw vs smoothed (NB06, measured under the previous 120d boom gate — NOT re-measured since): GET OUT AP 0.435→0.449, FA 41→39, −2 captures; GET IN adjacency 8→2 on the phase-aware frame; one-day blips structurally removed. Full sweep w∈{1,3,5,7,10,14} (NB07): no window strictly dominates w=7; run-rule triggers k>1 also dominated by the smoothed single crossing |
| LPPLS fit window | 60 d | ~one quarter of trading days for a stable quadratic fit | E5's contribution: −0.008 capture if dropped |
| Percentile window | 365 d (min 180) | "extreme for this name" = vs its own last year; half-year minimum before speaking | trailing-rank no-look-ahead test |
| Hype baseline | 120 d median | ~half a year of "normal" to compare a week against | inside A1 (see above) |
| Z baseline | 84 d | pre-existing project constant (~4 trading months) | conviction study (legacy, validated) |
| Bootstrap replications | 300 (bands: 200) | enough that the 5th/95th percentile is stable to the third decimal; the *cluster* count (59 instruments), not the replication count, is what actually bounds the interval width | re-running any interval at 300 vs 1000 moves it by less than the third decimal, so the number is not a lever. 200 is used only where a band is drawn at every event-window offset (`03:250` tournament pairing, `06:346` event-study band) — i.e. hundreds of intervals per figure, where 300 buys nothing visible and costs a third more time. 300 elsewhere (`02:353`, `04:428`, `06:174`, `06:289`) |

### Class 3b — HANDLE CENSORING ON SCREEN (added 2026-07-28)

*Desk request 2026-07-28: "maybe censor the innapropriate stuff with \*\*".
This is honestly a **CONVENTION** and cannot be anything else: there is no
ground truth for "offensive", so no bootstrap and no walk-forward can make a
word list evidence-backed. What IS evidence — and what is registered below —
is the **measured behaviour of the list over the real corpus of 12,528
handles**. Every tier assignment was decided by counting hits and reading
them, not by intuition. Implementation: `analytics/plain_english.py`
(`censor`, `censor_series`, `is_obscene`); six tests in
`tests/test_pipeline.py::TestHandleCensoring`.*

| Number / choice | Value | Class | Why / what the evidence says |
|---|---|---|---|
| Where masking happens | **display layer only**, never the store | INVARIANT (unit-tested) | `author` is the join key shared by `author_scores.parquet`, `calls.parquet` and `reply_edges.parquet`. Rewriting it would collide distinct authors, break every join, and destroy the ability to re-judge an author against new prices. `test_the_store_is_never_rewritten` asserts no `**` ever reaches the parquet; the map's `centre=`, the leaderboard's `_push` merge key and the ego selectbox's return value all keep the TRUE handle, and only the rendered string is masked (the selector uses `format_func`, not a censored option list) |
| Masked span | the offending span **only**, not the whole handle | DESK DECISION 2026-07-28 | a plotly category axis merges duplicate y-values into ONE bar and a leaderboard would merge rows, so wholesale replacement would silently destroy people. **MEASURED: 12,528 unique true handles → 12,528 unique censored strings, ZERO collisions.** `test_masking_keeps_handles_distinguishable` fences it |
| Tier A — substring stems | 21 stems, matched as substrings **inside one token** | CONVENTION, tier assigned by MEASUREMENT | needed because obfuscated handles run the stem into other characters with no separator (`fucktheredditapp15`). The containment rule is a free win with no word list: it alone kills `SatoshiTrails` / `TheSatoshiTimes` (*shit* spanning `...oshi\|Trails`) and `MeridianAllocation` (*anal* spanning `Meridian\|Allocation`) |
| Tier B — whole-token words | 30 words, matched only as a complete token | CONVENTION, tier assigned by MEASUREMENT | six stems were **DEMOTED** from Tier A on counted evidence: *anal* 3 innocent v 2 genuine, *rape* 7 innocent v 0 genuine, *cock* 3 innocent, *boob* 2 innocent, *piss* 1 v 0, *wank* 1 v 0. Price of the demotion, stated openly: six genuine handles now escape (`Cockballzz`, `redditsuckscockss`, `TRASHTALKINGCOCKSTAR`, `3boobsarenice`, `eskimoboob`, `analbuttlick`) |
| Promotions tested | *retard* ✓, *boobs* ✓, *tits* ✗ | **MEASURED** | *retard* 10 hits / 10 genuine and it additionally catches `Retardation-Syndrome`, which whole-token matching misses. *boobs* 3 hits / 3 genuine. *tits* REJECTED: 2 hits, one genuine (`Murrrtits`) and one not (`Iplayminecraftitsfun` = "minecraft its fun"), and the innocent one is a single token so the containment rule cannot separate them — **a 50% error rate is not worth one handle** |
| Words deliberately NOT listed | *suck, kill, damn, hell, crap, ball* | DESK DECISION 2026-07-28 | mild enough that masking them costs more identity than it buys decorum, and each is a heavy false-positive source (`Feb17Sucks`, `TheRedditModsSuck`, `BallSmashingForever` all read fine on a board everyone knows is scraped from Reddit) |
| Tokeniser | `[A-Z]+(?![a-z])\|[A-Z][a-z]+\|[a-z]+\|[0-9]+` | DERIVED from the notations actually present | must honour four at once: underscores (`just_lick_my_ass`), hyphens (`dick-knuckle`), camelCase (`AssumptionPretty`) and letter/digit boundaries (`Painkiller_830`). The **acronym alternative comes first** so `RHfuckedup` tokenises as `[RH, fuckedup]` rather than `[R, Hfuck...]` |
| Fixed-point iteration | up to 4 passes, mask runs collapsed | **DERIVED from a real failure**, not defensiveness | `Buttslut69696969` tokenises as `[Buttslut, 69696969]`; *butt* is not a whole token there, so pass 1 removes only *slut* and yields `Butt**69696969` — where `Butt` IS now a whole token. A single pass would leave a crude word on screen having "censored" the handle. Two passes reach `**69696969`. Each pass strictly shortens the letters available to match, so 4 is slack over the deepest case observed (2). This exact case failed `test_censoring_is_idempotent_and_total_on_the_real_store` before the fix and is now pinned by `test_masking_runs_to_a_fixed_point` |
| Direction of error | **under-mask, never over-mask** | DESK DECISION 2026-07-28 | asymmetric costs: a missed handle is one embarrassing name on a board everyone knows is scraped from Reddit; an over-masked handle corrupts identity for every reader and can merge two people. Every tier decision above resolves ties in this direction |
| Measured masking rate | **97 / 12,528 = 0.774%** | **MEASURED** | the naive one-list substring scan flagged 345 and was dominated by false positives; the containment rule plus the six demotions plus the six droppings took it to 97 with **one** residual false positive, `sashitadesol` → `sa**adesol`. On-screen relevance is real, not theoretical: **4 of the top 120 by composite** are masked (`RetardedChimpanzee`, `YuckGootyHole`, `BigBoiBenis`, `just_lick_my_ass`) and **1 of the 25 HIGH-tier authors** |
| Known limitation | **English only** | RECORDED LIMITATION | `fickdichdock` appears on the loud-but-wrong board and is NOT caught. Extending to other languages would multiply the false-positive surface across every language's innocent vocabulary, so it is recorded rather than attempted |

## Class 4 — GROUND TRUTH (price side; used only to grade, never to predict)

| Number | Value | Why |
|---|---|---|
| Peak local-max window | ±21 d (43 d) | month-scale "the highest close around here" |
| Boom minimum | +25% ETF / +50% single | dual thresholds are a recorded desk decision — singles are structurally more volatile |
| Bust minimum | −15% ETF / −30% single, within 90 d | same dual-threshold decision; a quarter to confirm the break |
| Boom lookback — GROUND TRUTH | 120 d | the window the trough is measured in, i.e. what COUNTS as a peak (this right-truncates run-length at 120 — a recorded caveat, NB01). **Unchanged**: moving it changes the yardstick, not the answer |
| Boom lookback — LIVE GATE (`EUPHORIA_BOOM_WINDOW_D`) | **54 d** (120 d → 60 d → 54 d, 2026-07-29) | a prediction-time choice, chosen on the NB07 §A3b frontier by the project's own rule (inside the FA budget, maximise capture), judged on the years all configurations share (2021/2022/2026, 93 peaks): 54 d captures **22** at **0.092** FA/inst-yr vs 60 d's 21 at 0.115 and 120 d's 20 at 0.115 — **120 d is dominated on both axes**. Capture is flat at 21–22 across 52–60 d while FAs rise monotonically; below 52 d it collapses to 15. 52 d is the lower-FA alternative. **Recorded cost:** the walk-forward loses its 2020 test year, denominator 122 → 98. **Correction:** an earlier note rejected short windows on AP lift; that lift was read off each window's own years and is an artefact — AUROC is ≈0.50 at every window from 40–100 d (§8.5) |
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
| Comments decoupled; dynamic panel (cap 1/month) | 2026-07-24 | slow fetch out of the daily path; panel expands where the crowd points, one denominator step per month. **The decoupling half was REVERSED on 2026-07-27** — see the next row and Class 7; the dynamic-panel half stands |
| Comments BUDGETED back INTO the live pipeline | 2026-07-27 | an influence board that rescores month-old comments is not a live board, so the fetch returns to every run — but under a measured page allowance rather than uncapped, because a 7-day gap costs ~16.5 min against a 10-min ceiling. Hitting the cap defers (the watermark does not advance), so the board is never silently partial, only ever less fresh. Every number behind the allowance is in Class 7 |
| Phase-aware onset tested & REJECTED | 2026-07-24 | halves start/end adjacency and LATE alerts but costs 5 captures with no utility gain (NB03); adjacency readability solved by episode-span shading instead (display, zero capture cost) — **superseded the same day by the desk configuration (below)** |
| DESK CONFIGURATION adopted (GET IN / GET OUT) | 2026-07-24 | the desk stated three times that a START landing on an END destroys PM trust → adjacency overrules the raw-capture utility rule. GET OUT = boom-gated + 7d-smoothed END (cap 24/122, FA 39, AP 0.449); GET IN = phase-aware + 7d-smoothed onset (adjacency 20→2, LATE 21→10, FA 169→124, capture cost 29→20 RECORDED). Selection rule pre-stated; NB06 drift guard pins notebook == production |
| Feature bank hand-specified, NOT searched | 2026-07-24 | recorded explicitly 2026-07-27 (it had only ever been implicit). Stepwise / L1-path / genetic search over the daily aggregates was considered and refused: the independent unit of evidence is the INSTRUMENT (59 of them), so a search scoring candidates on 183k daily rows picks among thousands of options on an effective n of 59. Every feature window is instead inherited from a constant already validated elsewhere, so no feature adds a degree of freedom a test year has not seen. Cost stated openly: per-feature AUROCs are only 0.51-0.60 and a search would very likely post better IN-SAMPLE numbers. Full argument: RESEARCH_REPORT §5.2 |
| Three stores kept, none removed (headline vs fallback vs gauge) | 2026-07-27 | audit question was "which euphoria report is headline, remove the loser". Answer: there is no loser, all three are load-bearing on different jobs. `euphoria_desk.parquet` + `euphoria_desk_report.json` = THE headline GET IN / GET OUT the tabs draw (`dashboard.py:1732`). `euphoria_onset.parquet` = the documented FALLBACK when the desk store is absent (`dashboard.py:1750`), which is what makes a fresh clone degrade instead of crash. `euphoria_report.json` = the frozen WALK-FORWARD thresholds, and it is the source of the gauge's red edge 85 (`dashboard.py:1887`) and the scorecard fallback (`:2211`) - deleting it would turn a derived number back into a hard-coded one. Recorded so the question is not re-opened |
| `notebooks/PROJECT_SUMMARY.md` DELETED | 2026-07-27 | 28KB unreferenced reading-guide that had become a CONTRADICTION SOURCE against the report in the same repo: 19 of its headline numbers were stale or wrong (episodes 211 vs 333; period 2021-2026 vs 2017-2026; onset capture 66% vs 23.2%; top capture 68% vs 13.1%; FA "0.23 budget maintained" vs the measured 0.348 = 50% OVER, which the report lists as Limitation 4 "not hidden, not excused"; lead time "17 days BEFORE starts" vs the measured 17 days AFTER the trough - sign-flipped; AP 0.0062 vs 0.098, off by ~16x; test years "2025-2026" vs 2018-2026; precision "~75%" vs 0.21/0.34). It also invented a ground-truth rule that exists nowhere in the code ("G3 duration >= 10 days") and preserved a FALSE record of why phase-aware gating was rejected. Three items were unique and were MIGRATED before deletion: the hand-specified-vs-searched rejection (row above / §5.2), the bootstrap replication count (Class 3), and three primary citations Hanley & McNeil 1982 / van Rijsbergen 1979 / Matthews 1975 (References 15-17). It was also the ONLY file in the repo carrying true emoji, plus a U+FFFD corruption |
| Euphoria GAUGE shipped on every chart | 2026-07-27 | the desk asked for "a very clear speedometer thing for each graph ... for each theme / ticker". Shipped as a dial whose needle IS the plotted curve, whose red edge IS the already-frozen walk-forward END level, and which therefore introduces exactly one new number (the amber edge 76, Class 1b). It reports a STATE and never an instruction, and its caption quotes the MEASURED risk of the band the needle is in — including the admission that the level alone is only ~1.3x base rate |
| Euphoria panel quotes the threshold that FIRED | 2026-07-28 | the desk read "flat and then suddenly a get out flag" off a chart that drew the level-detector's 85 while the flags came from the desk score's own frozen threshold. Measured: the plotted level was under the drawn line on 79 of 95 GET OUT alerts (83%). Corrected by drawing the gating threshold and overlaying the score that crossed it; the window's peak is marked and dated. No threshold moved — the chart was wrong, not the signal. Class 8 |
| Gauge answers "now" AND "this window" | 2026-07-28 | "the gauge seems to alwahys show calm" was a true observation of a correct dial: the page is ordered by most recent signal, so 50 of 59 names read calm today while 17 of them touched red inside the window. Fixed by adding the dated window high behind the needle rather than by softening an edge, and the standing explanation moved into an info hover so it is not repeated on every chart. Class 8 |
| Offensive handles masked ON SCREEN ONLY | 2026-07-28 | "maybe censor the innapropriate stuff with \*\*". Span-level, display-layer masking: 97 of 12,528 handles (0.774%), zero collisions, one residual false positive. Tier assignment measured rather than assumed, error direction deliberately under-mask, store provably untouched. Honestly a CONVENTION — Class 3b |

## Class 6 — INFLUENCE TRACKER (information only; nothing below feeds the euphoria level, the GET IN / GET OUT alerts, or any price claim)

*Added 2026-07-27 with notebook 05. The tracker's job is to say who has
actually been right and what they are saying now — it is a reading aid, not a
predictor, and the reason it is only a reading aid is itself a measured
finding (see the last three rows). Evidence for every number:
`docs/research/nb05_influence.json` and `notebooks/05_influence_users_model.py`.*

| Number | Value | Class | Why / what the evidence says |
|---|---|---|---|
| Composite usefulness score | 0.4·s_conf + 0.4·s_z + 0.2·s_enh | CONVENTION (ported) | Chan (2026) §4.6 weights, adopted unchanged so the port is a replication rather than a re-fit; a re-fit on ~250 positives would be fitting noise. Bayesian-shrunk toward the crowd base rate, then min-max normalised |
| HIGH tier cut | ≥ 0.66 | CONVENTION (ported) | Chan §4.6. Display grouping only — nothing downstream branches on it. **No longer shown on the dashboard (2026-07-27)**: a tier boundary printed next to a min-max-normalised score invites the reader to treat 0.66 as an accuracy. Still computed, still in the notebook |
| Board minimum record (`INFL_MIN_JUDGED`) | 5 judged calls | DESK DECISION 2026-07-27 | a record needs a record: below five graded calls the hit rate is one coin-flip wide, so those authors are shown in the map but never ranked. The dashboard says so in the caption |
| Label regime shipped | `softened` | DESK DECISION 2026-07-27 | the strict regime yields 25 positives — not powered. Softened yields 237, prevalence 354. Softened is the headline because it clears the maturity bar with the strictest definition that does |
| Maturity bar | ≥ 130 positives | CONVENTION | the thesis's own positive count (133) — we refuse to report a model trained on less than the work we are replicating |
| Adoption rule | paired 10-seed AP, `ci_lo > 0` | DESK DECISION | a prettier mean is not an improvement; the floor ships unless the interval clears zero. Seeds fixed and paired (`ADOPTION_SEEDS = 0..9`), ranking tables on {42, 100, 2026} |
| Multiple-comparison correction | Bonferroni, 6 candidates → conf 0.99167 | DERIVED | 1 − 0.05/6. One re-test round only, declared before the round ran |
| Shipped model | `logit` on the 17-feature FULL_BANK | LEARNED-then-parsimony | AP 0.0977 ± 0.0242 vs random floor 0.0467; AUROC 0.6681. `random→logit` adopted (+0.0464, CI [+0.0334, +0.0593], 10/10 seeds). Every graph rung REJECTED: mixhop +0.0011 CI [−0.0063, +0.0085], sage_lite −0.0035, h2gcn −0.0078 CI [−0.0130, −0.0026] (significantly WORSE), gcn −0.0132, mlp −0.0326 |
| `mean_conf`, `stance_sd` excluded (`SCORE_ADJACENT`) | excluded | DESK DECISION 2026-07-27 | they are arithmetic factors of their own target. Including them lifts AP 0.1004 → 0.1987 (mean_diff +0.0983, CI [+0.0834, +0.1132], 10/10) — the price of honesty is recorded, and paid |
| Bank change adopted this round | none | LEARNED (null result) | of six pre-stated candidates only "drop behavioural family" cleared zero at 95% (+0.0076, CI [+0.0023, +0.0128]); at Bonferroni 0.99167 it did not → adopted: null |
| Permutation significance | p = 0.005 | LEARNED | 200 label shuffles: AP 0.1257 vs null mean 0.0521, null p95 0.0705. 0.005 is the reportable floor at 200 draws, not a rounded number |
| Positive-class node homophily | 0.0948 (neg 0.9628) | MEASURED | sharper than Chan's 0.08 / 0.93. Useful callers do NOT cluster — which is why message-passing cannot help |
| DICE perturbation | AP 0.1031 → 0.2063 at 50% | MEASURED | corrupting the graph IMPROVES ranking (0/10/20/30/50% → 0.1031/0.1528/0.1671/0.1884/0.2063) while accuracy falls 0.9343 → 0.8883. Random rewiring only mildly degrades (→0.0978). Chan saw the same direction; the graph is an active liability here, not a weak feature |
| Cohort (unseen-author) split | lift −0.046 | MEASURED | on authors held out by tenure the shipped model sits AT the random floor (logit 0.0461 vs 0.0483; sage_lite 0.0410, lift −0.150). **This is why the dashboard ranks by the measured record and files the model as a research exhibit** |
| Map node pool | authors with ≥1 judged call (5,071) | DESK DECISION 2026-07-27 | the raw reply store holds ~107k accounts of whom only 5,071 have a usefulness colour; drawing the unrestricted graph gave 11.3% colour coverage and a 39.5s layout. Restricting to scored authors gives 100% coverage at 0.3s |
| `INFL_BACKBONE_MIN` / `_MAX` | 150 / 320 nodes | CONVENTION (display) | k is walked DOWN until the k-core has at least 150 people (→ k=8, 224 nodes, 1,798 links, 0.9s); the cap exists because the layout is O(n²). The SLICE is principled (a k-core, not a sample); only its size is a display choice |
| `INFL_EGO_MAX` | 120 nodes | CONVENTION (display) | one screen's worth of neighbourhood; the caption states that busiest neighbours are kept when it binds |
| Label placement geometry | 46px lead, 5.6px/char, 15px line | CONVENTION (display) | derived from the 10px label font on a ~900px canvas, not tuned: labels park outward with a leader arrow and are dropped when no slot is free. Affects which names are printed, no reported number |
| Betweenness on the dashboard | never computed | DESK DECISION | 21s for a number the tab does not use. It lives in the notebook, where a research reader can wait |
| ~~Consensus bar opacity~~ | *removed 2026-07-27* | — | `fig_consensus` plotted the same per-ticker consensus the bubble chart already carries on its x-axis, with the bubble chart's y-axis folded into bar OPACITY — its own docstring admitted it. Removed as a duplicate exhibit; the opacity ramp (0.30 + 0.70·√unit) died with it. Kept in this register because it was a recorded encoding decision |

### Class 6b — THE UNITS THE INFLUENCE TAB IS READ IN (added 2026-07-27)

*Every number on this tab used to be a bare sum with no unit, which is the
concrete reason the desk said "I still don't get it". The three rows below are
the fix, and all three are DERIVED — each is a rescaling of a quantity already
accepted above, so no ranking anywhere on the tab changes. Evidence:
`analytics/influence_graph.py` block comments and `TestInfluenceGraph` /
`TestChartLabelLayout` in `tests/test_pipeline.py`.*

| Number | Value | Class | Why / what the evidence says |
|---|---|---|---|
| Influence index | `100 · composite / max(composite)` | DERIVED | the composite is already min-max normalised, so the top author scores ~1.0 **by construction**, not because they were right 100% of the time. Printed as "usefulness 0.987" it read as an accuracy. On the real store `max(composite) = 0.98547`, so the index is a positive rescaling — rank-identical, but the reader can now say what it means: position in the field, 100 = the strongest measured record in the store. Implemented once, in `influence_index`, so no chart can drift onto a second scale |
| Backing, as SHARE | `100 · weighted / Σ weighted` (`backing_share`) | DERIVED | `weighted_voices` is a unitless Σ(influence × conviction), so "3.42" means nothing and cannot be compared across windows. Share is bounded 0–100, additive (crowding **is** concentration), and a long tail of one-off tickers barely moves the denominator. Measured 30-day cross-section: MSFT 26.10%, ADBE 7.05, FICO 4.82, INTU 4.04, MELI 3.58, tail STRC 0.17 |
| ~~Backing, as a MULTIPLE of the median name~~ | *rejected on data, 2026-07-27* | REJECTED | `weighted / median(weighted)` was implemented first, on the argument that it mirrors A1's self-anchoring ("2× its own 120d median"). **The analogy is false and is withdrawn**: A1 divides a name by *its own history*, a stable reference; the median NAME divides by the middle of a long-tailed cross-section, which here is a ticker one person mentioned once. Week to 2026-06-28: 163 names, median backing **0.24**, MSFT **141×**. Weekly maxima 141× / 41× / 2.7× / 14× / 26× (30d) and **171×** (90d) — unspeakable numbers that move with how many one-off tickers a fetch happened to catch. With `authors=None` the median is exactly 0 and every ratio returned **NaN** |
| Even split (the reference line) | `100 / n` per cent (`even_share`) | DERIVED | what every name would show if attention were spread equally — the line the bubble chart draws in place of the retired 1.0× hairline. **Derived from the window, not chosen**: 2.0% across the 51 names in the 30-day view, against MSFT's 26.1%, i.e. thirteen times an even share. Drawn per-week in the time chart, because a week touching 14 names splits differently from one touching 238 |
| Time-panel population | ALL authors with a record (340), not the top-N slider | **MEASURED** decision 2026-07-27 | cut the top 25 into weeks and the weeks hold **124, 25, 1, 23, 9** calls — the 1-call week has one name, which is 100% of that week's conviction *by definition*, and a "share of the room" built from four people is not a fact about the room. The same weeks over all 340 recorded voices hold **733, 404, 23, 98, 155** calls across 14–171 names. The slider exists to keep NAME-level exhibits legible; it was never meant to define the population whose mood is measured |
| Weekly evidence | tilt-dot AREA ∝ `n_calls` | CONVENTION (display) — **encode, never gate** | weeks are wildly unequal under the ingestion budget (23 vs 7,260 calls across the store), and a 23-call week can read tilt = +1.00 off four people. The tempting fix is a minimum-calls cut-off, which would be exactly the arbitrary threshold this project refuses. Sizing the dot is the honest alternative: no week is dropped, no number is invented, and a thin week **looks** thin. `n_calls` and `n_authors` also come back on every row and every hover |
| `INFL_BUBBLE_MAX` | 30 bubbles | CONVENTION (display) | where ticker labels stop overlapping at the chart's 520px height. A DISPLAY limit only: the dropped tail still counts in the share denominator (that is the point of a share) and still appears in full in the exact-numbers table |
| Label de-collision (`_thin_labels`) | 13px vertical, 30px horizontal | CONVENTION (display), **derived from geometry** | `consensus` is bounded at ±1 and hits +1.00 exactly whenever every call on a name was long, so a dozen names pile into one column and their tickers print through each other — measured: INTU 4.04% and MELI 3.58% are 0.46pp apart, ≈9px on a 35% axis, and a 10pt label needs 13. The gaps are one line box (10pt × 1.3) and a four-character ticker's width, converted to data units from the plot's own size. **This is the only rule on the project that is about pixels rather than data, and it decides where there is room for ink — never which names matter**: every un-labelled point is still drawn, still hovers, still tabulated. Suppression requires BOTH axes tight, so GOOG (x=0.53) and MELI (x=1.00) at the same height both keep their labels. Five unit tests fence it in |
| Label width, when labels are NOT all one size (`label_w_px`, added 2026-07-28) | `30px / 4 chars = 7.5px` per character | DERIVED (from the row above) | the 30px gap is only correct because every ticker is about four characters. A theme label is not — "Robotics automation" is nineteen — so the theme view would have reproduced the exact overlap the function exists to prevent. Labels are drawn CENTRED, so two collide when the gap between their centres is under the **mean of their two widths**; with every width equal to 30px that expression *is* the scalar rule, so **the ticker view is unchanged by construction, not by re-tuning** — verified directly: on the live 30-day and 90-day cross-sections the old and new rules keep the identical 11 and 17 labels, nothing gained, nothing lost. The per-character figure is read off the accepted 30px rather than introduced as a new constant |
| Per-author hit rate on the dashboard | never displayed | DESK DECISION 2026-07-27 | still computed, still stored in `author_scores.parquet`, still an input to the composite — simply not on this screen. (i) A raw hit rate and a shrunk composite answer different questions and disagree **by design**: shrinkage exists precisely to stop 3-from-3 outranking 28-from-40, so a reader reconciling the two columns is fighting the method. (ii) An unqualified per-person accuracy invites position sizing off a sample of five, which the cohort split says nothing here supports. Reported in notebook 05 **with sample size and CI attached**, the only defensible form. The stored column name is unchanged — a schema test asserts it |

### Class 6c — CAN INFLUENCE CONVERGENCE PREDICT? (tested 2026-07-28 — **REJECTED**) and the THEME view that shipped instead

*The desk asked: "lets try and use the follower monitoring (influence) for
some sort of bullish / euphoria indicator? anyway to make that clearer?
perhaps if we have lots of influential accounts converging on a theme?"*

**That is two questions, and they get two different answers.** "Use it as an
indicator" is a PREDICTIVE claim and had to clear the same bar as everything
else on the project — it did not, and nothing from it ships. "Make it
clearer" is a LEGIBILITY request, and the answer to that shipped as an
information-only theme view. The two are kept apart on purpose: the chart
that shipped carries the rejection in its own caption, so it cannot be
mistaken for the thing that was rejected.

**No code from the rejected half is left anywhere in the repo** (standing
rule: if it is not the most efficient thing, document it and leave no
traces). This section IS the trace.

#### The design, fixed before any number was seen

| Choice | Value | Why it is not a knob |
|---|---|---|
| Outcome | a **>10% fall inside a week**, occurring any time in the next **30 days** | copied verbatim from the accepted gauge outcome (`DROP = −0.10`, `FWD = 7`, `HORIZON = 30`). Reusing it is what makes the comparison against the accepted signal like-for-like |
| Candidates | exactly **3**, pre-registered | C1 **breadth** = weighted count of distinct voices; C2 **convergence** = breadth × agreement; C3 **backing share** |
| Agreement | `\|Σ w·s\| / Σ w·\|s\| ∈ [0,1]` | the same arithmetic as the already-accepted `consensus`. One voice, or many saying the same thing, scores 1; a room split down the middle scores 0. **Many voices that DISAGREE is not convergence** |
| Author weight | `influence_index(author) / 100` ∈ [0,1] | **invents no threshold at all**: a nobody contributes ~0, the strongest measured record contributes 1. The obvious alternative — cut the board at the HIGH tier — is unusable here: the store holds **25 HIGH against 12,503 low**, and only **234 of 27,881** live calls come from a HIGH author |
| Window | `ROLL = 7` days trailing | the project's existing roll, not a new one |
| Live window | from **2026-04-01** | the call history is two disjoint blocks: 2021-06 holds 5,521 calls across only **2 distinct days** (an archive snapshot, not history), then a five-year hole, then 2026-04 → 2026-07. A trailing window cannot be computed across the hole |
| Significance | paired bootstrap on INSTRUMENT, 5 seeds × 300 reps, Bonferroni **n = 3** → conf **0.98333** | resampling the instrument, not the day, because name-days inside one ticker are not independent. Adoption requires the **worst** seed's lower bound to clear zero |

#### Result: ADOPTED — NONE

Panel: **15,615 name-days, 233 instruments, 2026-04-08 → 2026-06-15**, base rate 0.539.

| Candidate | top-decile vs rest | diff | worst-seed 98.33% lower bound | Verdict |
|---|---|---|---|---|
| C1 breadth | — | **−0.0291** | −0.1701 | REJECTED |
| C2 convergence | — | **−0.0153** | −0.1513 | REJECTED |
| C3 backing share | — | **−0.0103** | −0.1706 | REJECTED |

#### Why the null is informative: the positive control

A null result only means something if the harness could have found a
signal. The **accepted euphoria level**, pushed through the identical
harness on the identical days, detects easily:

| Control | positive vs rest | diff | worst-seed lower bound | Verdict |
|---|---|---|---|---|
| Euphoria level, top decile (cut 78.91) | 0.846 vs 0.414 | **+0.4321** | +0.0623 | **DETECTED** |
| Euphoria level ≥ 85 (the frozen RED edge) | 0.925 vs 0.428 | **+0.4970** | +0.2515 | **DETECTED** |

#### The decisive exhibit: like-for-like on the control's own days

Same **1,552 name-days**, same **23 instruments**, same **156-day** state
size, base rate 0.457. The only thing that differs is the FEATURE — so
"the window was too short" is not available as an explanation.

| Candidate | positive vs rest | diff | worst-seed lower bound |
|---|---|---|---|
| C1 breadth | 0.237 vs 0.482 | **−0.2449** | −0.5380 |
| C2 convergence | 0.250 vs 0.481 | **−0.2307** | −0.4876 |
| C3 backing share | 0.282 vs 0.477 | **−0.1950** | −0.5142 |

**All three are wrong-signed.** They are not underpowered; they point the
other way.

#### The mechanism, diagnosed rather than asserted

The negative sign is a **composition effect, not a signal.** The
top-decile-breadth names by state-days are MSFT 64, NVDA 59, TSLA 56, AMZN
46, RDDT 45, INTC 44, AAPL 44, SNDK 38, GOOGL 37, MSTR 37 — the influence
board converges on the **most-discussed liquid mega-caps**, and those cliff
less often than the small-cap tail. Splitting BETWEEN names confirms it:
"ever top-decile" 0.534 (n = 6,576) against "never" 0.542 (n = 9,039) —
nearly flat. The effect is **cross-sectional (which names) and not temporal
(when)**, which is precisely what disqualifies it as a timing indicator.

#### What shipped instead (information only)

| Number | Value | Class | Why / what the evidence says |
|---|---|---|---|
| Theme grouping key | the SAME membership as the euphoria Themes tab (`src/themes.py`) | INVARIANT | a theme must mean one thing across the whole app, or the two tabs quietly disagree in front of a PM |
| Shared arithmetic | one `_digest_frame(c, key)` for both views | INVARIANT (unit-tested) | consensus and backing are the desk's accepted formulas; a second copy written for themes would be a second place for them to be wrong, and the two views sit side by side on one toggle where any disagreement would be visible and unexplainable |
| A ticker in several themes | counts in EVERY one | DESK DECISION 2026-07-28 | NVDA is semiconductors AND ai AND ai_megacap. A PM asking "is the panel crowded into AI" must see the NVDA call; assigning each ticker one primary theme would answer a different question and under-count the theme that matters. Consequence, stated on the control: theme shares are shares of the **theme-mapped room**, a different denominator from the ticker room, so the two views are **not expected to agree name-for-name** |
| Calls on tickers in NO theme | dropped, not bucketed as "other" | DESK DECISION 2026-07-28 | "other" is not a theme a PM can position in, and on the live store it is **58.5% of live calls** — it would be the largest bar on the chart purely by being a residue |
| The exhibit's own caption | carries the rejection above, with the numbers | INVARIANT | "we checked" is not defensible; "−0.245 on the same days the accepted signal reads +0.497" is. The caption is what stops a reader inferring a forecast from a picture of crowding |
| Live reading (90 days, sanity check) | AI megacap **26.6%** of the room's conviction at consensus **+0.807**; semiconductors **10.2%** at consensus **+0.222** | MEASURED | the view earns its place by making a distinction the ticker view could not: AI megacap is a genuine convergence, semiconductors at +0.222 is visibly an **argument** — which is exactly what the desk asked to be made clear |

## Class 7 — INGESTION BUDGET (how much comment data one live run may buy)

*Added 2026-07-27 with `src/pipeline_budget.py`. Nothing here touches any
signal: these numbers decide how much data a run FETCHES, never how it is
scored. They exist because "just fetch everything" and "~10 minutes" are
incompatible at the panel's real volume, and the project does not resolve
that with a typed-in cap. Exactly ONE number below was chosen by a human;
the rest are measured or derived from it. Evidence: `src/config.py`'s
COMMENT INGESTION BUDGET block, `src/pipeline_budget.py` docstrings,
ARCHITECTURE §3.1b and §3.1b-i.*

| Number | Value | Class | Why / how it is obtained |
|---|---|---|---|
| `PIPELINE_BUDGET_S` | 600 s | **DESK DECISION 2026-07-27** | the desk's own sentence: "a full update of the dashboard (like weekly) running shouldn't take more than ~10 minutes". The only human-chosen number in this class; everything else is measured against it |
| `COMMENT_RATE_PER_S` | 1.0 req/s | CONVENTION (contract, not a knob) | what the project committed to when it chose a free public archive API. Raising it, or splitting the panel across W workers each pausing a second (an aggregate W req/s), breaks that contract — so it is **not** available as a speedup. Recorded as rejected, with the reason |
| `COMMENT_PAGE` | 100 rows/page | GROUND TRUTH (the API's own page size) | not ours to choose; it is what one request returns |
| Comment volume | ~14,000 comments/day ≈ 140 pages/day (17-sub panel) | **MEASURED** | intersecting comment-call `rec_id`s with `reply_edges` gives 12,010 / 417,208 = **2.879%** call rate among comments; against 403 comment-calls/day (2026-06) and 414/day (2026-07) that implies ~14k comments/day. Caveat: the rate is measured on the edge-covered subpopulation (48% of comment-calls) |
| `COMMENT_PAGES_PER_DAY_PRIOR` | 140 | DERIVED (bootstrap prior) | the row above, per panel member, used only until the machine has its own ledger — then per-subreddit measurement replaces it |
| Non-fetch stage cost | measured per stage, EWMA | **MEASURED** (`pipeline_stage_times.json`) | first real run on this data: analytics 73.3 s (reproduced exactly twice), fold 0.44 s, coverage 0.31 s, hydrate 0.012 s, prices ~60 s. The allowance is the residual of `PIPELINE_BUDGET_S` after these |
| Page allowance | 465 pages (this machine, after 1 run) | DERIVED | `(PIPELINE_BUDGET_S − measured non-fetch seconds) × COMMENT_RATE_PER_S`. Self-corrected 449 → 465 on the first real run, i.e. the loop demonstrably closes |
| `EWMA_ALPHA` / `EWMA_RUNS` | 0.2 / 9 runs | DERIVED | span identity `α = 2/(N+1)` with `N = round(PANEL_REFERRAL_WINDOW / COMMENT_CADENCE_DAYS) = round(28/3.2) = 9`: remember about as much history as the panel's own review window. Not tuned |
| `COMMENT_CADENCE_DAYS` | 3.2 d (measured 3.32 on the real panel) | DERIVED | allowance ÷ panel pages-per-day = how many days of volume one run can buy. This is what makes "~2×/week" a consequence rather than a preference — and the desk chose 2×/week with this number in front of it |
| Per-subreddit allocation | proportional to owed days, one-page floor | CONVENTION | proportional so a stale subreddit is not starved by a fresh one; the one-page floor is anti-starvation (the trim loop stops while any sub is at 1), so a tiny allowance spreads thin rather than dropping subreddits entirely. Surplus is deliberately NOT redistributed mid-run — that would make coverage depend on crawl order, the exact position-dependence the allocation removes |
| `LATE_ARRIVAL_DAYS` | 1 d | CONVENTION | one day of overlap so comments posted just behind the watermark are not missed; the seen-file dedups the re-read |
| Live lookback window | 5 d (derived, not typed) | DERIVED | `ceil(COMMENT_CADENCE_DAYS) + LATE_ARRIVAL_DAYS = ceil(3.2) + 1` — the window is a function of how often the desk actually runs, so it moves if the cadence moves |
| Pacer dead time removed | 28% at unchanged request rate | **MEASURED** | the old code slept a flat 1 s *after* each round-trip, so the true period was RTT + 1 s (≈1.3–1.4× slower than the contract permits). Sleeping the *remainder* of the second measured 261 ms vs 360 ms over 6 requests. The only legitimate speedup found |
| Deferral on cap | watermark does not advance | INVARIANT (unit-tested) | hitting a page cap sets `completed = False`, so the sub keeps its old watermark and the next run resumes exactly there. Without this the watermark would jump past ground never crawled — silent data loss. Asserted directly in a no-network stub test |

## Class 8 — READING THE EUPHORIA PANEL (added 2026-07-28)

*Two desk bug reports, both about legibility, and **neither fixed by moving a
threshold**: "its quite unclear to see WHEN is the actual change / or get out
flag - i want it to be like a clear peak or something (as right now its like
flat and then suddently a get out flag)" and "the gauge, it seems to alwahys
show calm? dont need to explain it fully all the time, maybe an info icon
hover or something". One of the two turned out to be a genuine
**self-contradiction in the chart**, not a display preference. No number in
Classes 1–7 changed. Evidence: `dashboard.py` block comments at the
`draw_chart` threshold block, the peak-marker block and
`fig_euphoria_gauge`.*

| Change | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| Threshold line CORRECTED | draw the threshold that gated the flags on screen, and plot the series that crossed it | **BUG FIX, measured** | the old panel drew ONE dotted line at the level-detector's walk-forward threshold (85) and plotted the euphoria LEVEL against it — but when the desk store is present (the normal case) the flags are produced by the **desk score** crossing **its own** frozen threshold. **MEASURED over all 95 GET OUT alerts in the store: the plotted level was BELOW the drawn 85 line on 79 of them (83%), median plotted level at a GET OUT 74.8.** So the chart showed a curve sitting comfortably under the line it said mattered, and then a flag appeared anyway. The desk's "flat and then suddenly a get out flag" was a correct reading of a wrong chart. `thr_now` (85) survives only on the fallback path, where the level really is the decider |
| GET OUT / GET IN score overlay | the desk score, ×100 onto the panel's existing axis | **DERIVED — no new quantity** | the score is read straight from `euphoria_desk_report.json`; ×100 keeps ONE axis on the panel instead of a second y-axis a reader must notice before they can read it. Only kinds that ACTUALLY FIRED in the window are drawn, so every element explains a flag the reader can see. `lines+markers` with `connectgaps=False` because the score is **sparse by construction** — `out_score` exists on **2.5% of name-days**, in runs as short as one day; a plain line would draw nothing visible for a one-day run and would falsely bridge gaps. The gaps are information: no score means "not judgeable here" |
| Peak marker | dot + dated label on the window maximum of the display curve | CONVENTION (display) — **a label, not a quantity** | the desk's own words, "i want it to be like a clear peak or something". It marks a value already plotted, so nothing about the signal changes if it is removed. Passed as `pd.DatetimeIndex([d])`, **not** a bare `[Timestamp]`: the latter survives the live app but is not JSON-serialisable by kaleido, so it silently breaks static PNG export — and these panels are exported for the decks |
| Gauge "always calm" | the dial now also carries the window's high-water mark | **BUG FIX by explanation, no edge moved** | the dial was not stuck. **MEASURED over the default window (2026-01-01 → latest): 50 of 59 instruments read calm at the last day while 17 of those same names touched the RED ZONE inside the window.** Both are true at once because the page is ordered by MOST RECENT SIGNAL, so a name earns its place with an episode that may have peaked months ago while the needle — correctly — reports today. A dial answering "how hot is it now?" on a name selected for "it was hot recently" reads calm almost always and looks broken while being right. Fix: make the dial answer both questions. Big needle stays TODAY (the desk asked for a *current* percentage); the window high is a dated line behind it, so a calm reading carries its own explanation. **No edge moved, no number invented** |
| Window high drawn as TEXT, not a second needle | one dated line under the dial | CONVENTION (display) | plotly's Indicator has a single `threshold` slot and it already carries the red edge as a hard line (a colour change alone does not survive greyscale or a projector). A second needle would mean drawing shapes against the arc's internal geometry in paper space — fragile across plotly versions — and two needles on a 268px dial reads worse than one sentence. The frame grows 268→300px with a 74px bottom margin, which is what keeps the line inside the canvas |
| Gauge explanation moved to `help=` | one headline on the page, the evidence one hover away | DESK DECISION 2026-07-28 | the dial printed a measured-percentage paragraph plus a four-sentence caption on EVERY chart, so a page of six names carried the same ~90 words six times and the reading that mattered was buried in its own footnotes. **The evidence is not deleted or weakened** — it moved into the tooltip, available when someone challenges the band and silent when nobody is asking |
| Gauge `valueformat` | `".0f"` on both number and delta | DERIVED | the level is a 0-100 index built from percentile ranks, so a tenth of a point is below the resolution of the thing being measured and "28.4" invites a precision the input does not have. Display only; the stored value is untouched |
| Delta arrow inverted | rising = `BEAR`, falling = `BULL` | CONVENTION (display) | euphoria rising is the RISK direction, so "up" must not be painted green on this dial |

### Class 8 continued — the second legibility pass (2026-07-28, same day, after the desk read the first one)

*Four further desk instructions, all about what the panel SAYS rather than what
the detector DOES: "the euphoria charts are still way too messy. i dont get
what is activating a signal, is it a crossing? inflection? i just want one
line"; "remove the hit rate etc stuff (the perfomance metrics) in the dashboard
i will just keep this for the notebooks only"; "arrange the dashboard better /
like title of ticker / guage + some elements next to it / then chart /
essentially using less white space"; and "keep the what is euphoria- start here
- plan english on the dasboard, i like it". **Again no number in Classes 1–7
moved.** Evidence: the block comments in `dashboard.py` at the readiness-series
block, the header/dial row, and `decisions_simple()`.*

| Change | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| ONE readiness line replaces five overlays | `readiness = deciding score / that rule's frozen threshold × 100`, one series per firing rule, one dotted line at 100 | **DERIVED — no new quantity** | the desk could not tell what fired a signal. It IS a crossing (`alerts_from_scores` fires on `score >= threshold`, then a 21-day cooldown; there is no inflection test anywhere in the firing path) but the panel could not show that, because it drew a faint raw level, a bold smoothed level, an eligibility ribbon, a peak marker and the score, at equal weight — and the two thresholds (GET IN 0.848141, GET OUT 0.630231) sat at DIFFERENT heights, so neither line meant "the line". Dividing each score by its own frozen threshold puts the firing line at **100 for every name, every rule, every window**, which is what lets both rules share ONE line. Same stored score, same frozen threshold, divided: **alert dates are bit-identical** and the vertical signal lines still come from `coherent` |
| Near-misses drawn, not just firings | every rule whose score EXISTS in the window gets a line | CONVENTION (display) | the old panel drew a score only when it produced a flag. With the level curve gone that would leave the name-lookup box — the one place a PM checks a name that never alerted — showing an empty panel. A line that climbs to 80 and turns over IS the answer to "why did nothing fire here?" |
| Sparsity kept visible (`connectgaps=False`) | gaps are drawn as gaps | CONVENTION (display), **cost measured** | `desk_candidacy` only scores a name on days the gates allow judgement. Over the live store (63,345 name-days) that is **2.5% of days for GET OUT and 48.8% for GET IN**, so the GET IN line is near-continuous and GET OUT is a set of arcs. Those arcs are drawable: the 1,600 GET OUT scored days form **160 runs, median length 7 days** (mean 10, max 91, only 28 single days). A blank day means nothing CAN fire there, whatever the crowd is doing — bridging it would invent a reading |
| The 0-100 level is NOT lost | it is what the dial reads | — | the definition Alex asked not to change is untouched in the stores (`level`, `hype_ok`, both raw scores). The two questions are separated: the dial answers "how hot is this name", the panel answers "how close is it to firing" |
| Performance metrics REMOVED from the dashboard | no hit rate, lead time, FA count or CI anywhere on the page | DESK DECISION 2026-07-28 | "i will just keep this for the notebooks only". The notebooks are the research record; the dashboard states conclusions. **Scope, stated so it can be challenged:** headline performance reporting is gone everywhere, but the justification numbers inside `decisions_simple()` / `DECISIONS_DOC` and the band-meaning percentages in `gauge_caption()` were KEPT — strip those and every remaining choice on the page looks arbitrary, which is the failure mode this whole register exists to prevent |
| Header → dial+facts row → chart | one compact name line; dial shares a row with five facts; figure follows at `title=None`, 8px top margin | CONVENTION (display) | the old stack paid for a half-used dial row plus a second title strip inside the figure's own 55px top margin, per name — most of a screen on a six-name page. WHICH five facts: state, today's reading, the 7-day change, the window peak, the last signal. Every one describes WHERE THIS NAME IS; **none describes how well the detector has done** — that is the row above, honoured in the one place there was now room to break it |
| Plain-English opener KEPT | "what is euphoria — start here" | DESK DECISION 2026-07-28 | explicitly retained on the desk's instruction ("i like it") during a pass whose whole direction was removal. Recorded so a later cleanup does not read it as leftover explanation and delete it |

### Class 8 continued — the third pass: two screen bugs, a ghost line, and four expanders removed (2026-07-28)

*Four desk items, all about the panel and **none of them about the
detector**: "the graph is not formatted correctly"; "can we make the
euphoria still have like a continuos line? but maybe make it like not as
prominant"; "how does it actually work? the signal for get in / get out?";
and "for the explanations, just keep the 'what is euphoria' one … dont
change the current what is euphoria? (start here - plain English) just
remove the rest". **No number in Classes 1–7 moved, and the alert dates
are bit-identical to the previous pass.** Two of the four were real
rendering bugs that could only be found in the live DOM, not in the
source. Evidence: `dashboard.py` block comments at the `fig.update_layout`
title/margin block, the ghost-line loop, the lower-panel caption and the
two expander-removal comment blocks; live verification via Playwright
against the running app.*

| Change | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| `title=dict(text="")`, never left unset | an explicit EMPTY STRING for the figure title | **BUG FIX, root-caused in the DOM** | a bold **undefined** was printing over every euphoria chart. The word appears in **no Python file in this repo**, which is why four source-level hypotheses were formed and discarded before the DOM was dumped: the decisive step was walking the tspan's ancestor chain and reading `data-unformatted` on `text.gtitle`, which gave `"<b><b>undefined</b></b>"` and therefore named the layer (`g-gtitle`) and the mechanism. **Streamlit's plotly theming rewrites the title as `"<b>" + spec.layout.title.text + "</b>"`**; with no title in the spec that inner value is the JavaScript `undefined`, so the literal string reached the browser. Removing the title strip was itself deliberate (it cost 55px of dead vertical space per name) — the fix is to keep it removed but make it a REAL empty string, so the same rewrite yields `"<b></b>"` and renders nothing. Verified after: rendered `_fullLayout.title.text` = `"<b><b></b></b>"`, occurrences of "undefined" on the page **6 → 0** |
| Top margin `t=8` → `t=38` | 30px more headroom above the price panel | **BUG FIX, derived from the geometry** | the `GET OUT <date>` labels were being cut in half by the canvas edge. Derived, not tuned: the annotations sit at `y=1.0, yref="y domain", yanchor="bottom"` with alternating `yshift = 4 + 14*(i%2)` (so up to 18) to stop adjacent labels colliding, and 9.5px text is ~13px tall — the tallest label therefore reaches **~31px above the panel**. 38 clears 31 with a little air and is still **less than the 55px the deleted title strip used to cost**, so the space saving that motivated removing the title survives |
| THE GHOST LINE | the same readiness series, time-interpolated across gaps, drawn UNDERNEATH at 1px / dotted / 30% opacity / out of the legend | CONVENTION (display) — **and a knowing reversal, so the guards are the entry** | "can we make the euphoria still have like a continuos line? but maybe make it like not as prominant". The arcs are correct but leave the eye nothing to follow, so a reader reconstructs the episode shape themselves. This **reverses** the same-day rejection of a dense line (recorded above: it "invents a reading on ungated days"), so it is only defensible with the reason the old version was unsafe identified and neutralised. **(1) `hoverinfo="skip"` — the ghost NEVER reports a number.** Hover still comes only from the bold trace, so no ungated day can be read off the screen as a score. This is the load-bearing guard: the earlier objection was to a dense line that could be *queried*, not one that could be *seen*. **(2) `limit_area="inside"`** — interpolation happens only BETWEEN two real scored days, never past the first or last, so the ghost cannot imply a reading in a stretch the detector never judged at all. **(3) 1px dotted, 30% opacity, no legend entry** — it reads as a construction line; the solid 2.4px markered trace is still the only thing on the panel that looks like a measurement. Verified live: two `(shape only)` traces present with `hoverinfo: "skip"`, `connectgaps: true`, `opacity: 0.3`, added before the bold traces so they render behind. **Nothing here feeds the model; alert dates untouched** |
| Four model-evidence expanders REMOVED from the page | page-level *"WHY THE MODEL DOES WHAT IT DOES"* (`decisions_simple()`) and the long-form evidence log (`DECISIONS_DOC`); euphoria-tab *"full method & measured record"* (`EUPHORIA_DEF_FULL`) and the deep archive that printed this register off disk | DESK DECISION 2026-07-28 | "just remove the rest". Consistent with the standing split — the notebooks and this register are the research record, the terminal states conclusions — and the content is not lost, because all four were second copies of `DECISIONS.xlsx`, this file, and the notebooks. **What it costs, recorded because the argument went the other way and was overruled: a PM who challenges a threshold live can no longer answer it from this page.** The answer now requires the register or a notebook. Mitigation: `decisions_simple()`, `DECISIONS_DOC` and `EUPHORIA_DEF_FULL` are left **DEFINED BUT UNREFERENCED** in `dashboard.py` rather than deleted, so restoring any of them is a two-line change — **they must not be swept by the no-dead-code pass** |
| The surviving explainer's label is FROZEN | *"what is euphoria?  (start here - plain English)"* — lower case, two spaces, that wording | DESK DECISION 2026-07-28 | a rename to *"What is Euphoria"* was proposed by the desk and then withdrawn in the same exchange ("dont change the current what is euphoria? … just remove the rest"). Recorded so the withdrawal is not re-litigated and so a later tidy-up does not "fix" the capitalisation |
| Lower-panel caption extended | one clause describing the ghost and stating that it carries no reading | CONVENTION (display) | a faint line that cannot be hovered is only honest if the panel says so. The caption now states that the solid dotted-marker line is the measured score drawn on gated days, and that the faint line joins those stretches, carries no reading of its own, and does not respond to hover "because on those days nothing could fire however loud the crowd got" |

### Class 8 continued — the fourth pass: the watch track replaces the ghost, plain-English component names, and the chart cap stops being silent (2026-07-28)

*Three desk items, again **none of them about the detector**: "why are there
so little charts displayed? like only 8 charts - also how do you select what
do to show?"; "also dont use stuff like e1 e2 e3 e4 e5, just use the name of
the term as with the other terms"; and "i liked it before when it had the 7
day smoothed as well … i dont want the red / blue dots to appead just
suddenly, perhaps plot it when it is not eligable (not boom_state) but still
hype_ok, e.g. tracks it until boom_state and then becoems colored and becomes
an alert". **No number in Classes 1–7 moved; the frozen thresholds
(0.630231 / 0.848141) and the alert dates are bit-identical.** Two ambiguities
in the third item were put back to the desk rather than guessed — which curve
("the deciding score", not the 0-100 level, which would have resurrected the
27-Jul bug) and how the pre-eligible stretch should behave ("grey is faint +
un-hoverable"). Evidence: `dashboard.py` block comments at the import, the
watch-track block, the caption, the explainer and the slider; offline
reproduction of the wide-frame score against the live stores; live
verification via Playwright.*

| Change | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| THE WATCH TRACK **replaces** the ghost line | the score the *production* scorer produces on the WIDER candidate set, drawn grey/dashed ahead of the coloured stretch, masked off every judged day | CONVENTION (display) — **built on production functions, so it cannot drift** | the ghost was `interpolate()` — a straight line between the two nearest scored days, i.e. a guess about a number nobody computed. A real number was available instead: `desk_end_fit` / `desk_onset_fit` called with `train=None` (legitimate — the desk family is a rules family, so fitting is a no-op and both functions ignore `train`) over the wide domain. **GET OUT** wide domain = `hype_ok` only, price/boom gate dropped: **2.5% → 24.1%** of all name-days (15,244). **GET IN** wide domain = every onset row (`hype_raw ≥ 1` holds on 100% of them), end-stage veto dropped: **48.8% → 52.6%** (33,306). Computed from the **UNCLIPPED** store, never the window-clipped frame, so a trailing 7-day mean cannot shift when the sidebar window moves. If the scorer changes, the track changes with it |
| The grey is MASKED OFF every day the stored score exists | `_tr.where(_s.isna())` + `connectgaps=False` | **the load-bearing guard — derived from a measured disagreement** | `_smooth_by_name` rolls `ROLL` days over each name's **candidate-day SEQUENCE, not the calendar**, so widening the candidate set changes which days fall inside each window and the wide recomputation is **not** the stored number. Measured over the 1,600 overlapping name-days: median absolute difference **0.0000**, p95 **0.2234 (= 35 threshold-points)**, max **0.7505 (= 119 points)**, and the two **disagree about whether the threshold was crossed on 66 days (4.1%)**. Un-masked, the panel would re-open the exact self-contradiction fixed on 27 Jul — a curve under the line while a flag flies. Masking makes them **incapable** of disagreeing: the bold trace owns every judged day, the grey owns only days with no verdict. `connectgaps=False` is part of the same guard — with `True` plotly bridges straight across the masked days and undoes it. Verified live: **zero overlapping non-null points** across every drawn name |
| The track may exceed 100 without an alert | no clipping, no suppression | **kept deliberately — it is the exhibit, not a defect** | on ineligible days the track is at/above the trigger on **1,891 GET OUT name-days (13.9% of the ineligible wide domain)** and **425 GET IN name-days (17.6%)**. That is the visible answer to "why did nothing fire here?" — the crowd score alone was there and the price gate held it back, which is precisely the eligibility argument in §6.8 made legible. Clipping it would hide the one thing the grey exists to show |
| Grey may sit FLAT ON ZERO | not masked, not hidden | **the production number, stated in the caption** | `desk_end_fit` zeroes the score where the attention gate (`e1 ≥ EUPHORIA_ATT_GATE ∧ e2 > 0`) fails, so on a busy-but-ungated day the score genuinely **is** zero — measured on names such as AAPL (177 watch-days, all zero) and AMZN. A flat line on the axis reads instantly as "nowhere near firing", which is true; suppressing it would have required a rule keyed on an exact-zero sentinel and would have hidden a real reading. The caption says so explicitly so it cannot be read as missing data |
| Grey is `INK_MUTED`, not the rule colour | 1px, dashed, 45% opacity, `hoverinfo="skip"`, no legend entry | CONVENTION (display) | colour on this panel already means "this can fire" (red GET OUT / teal GET IN), so a grey line reads as pre-signal **by construction** rather than by convention. `hoverinfo="skip"` and the legend omission are carried straight over from the ghost and still load-bearing: the track never reports a number, so no watch-day can be read off the screen as a score the detector stood behind. `limit_area="inside"` likewise survives — the fill between watch days never extends past the first or last one |
| No `E1`/`E2`/`E3`/`E5` on screen | every component label in the GET OUT explainer now comes from `PLAIN` in `analytics/plain_english.py` | DESK DECISION 2026-07-28 | "just use the name of the term as with the other terms". The GET IN paragraph already named its features in words, so the **same panel used two notations for the same kind of quantity** — the inconsistency was the complaint. **The STORED COLUMN NAMES are untouched and stay `e1..e5`**: that module's docstring records why (the parquet schema is an interface; renaming would invalidate every cached frame, schema test and saved notebook output for a cosmetic gain), so translation stays at the display layer and the screen and notebooks cannot drift apart. Verified live: `(E1 `/`(E2 ` occurrences on the rendered page **2 → 0** |
| `items per section` max `15` → `60`, plus an explicit count line | "**N of M themes alerted in this window** … Showing the K most recent; raise *items per section* to see the other N−K" | **BUG FIX (silent truncation) + the standing no-silent-caps principle** | answering "why are there so little charts?" required tracing three stacked filters, and only the last is a preference: (1) the window, (2) **the name must have ALERTED in the window** — deliberate, and the standing instruction ("just show all the themes / tickers that have had a euphoria detected in the time frame selected"), (3) the slider. **Filter (2) is what binds**: in the default window 8 of 34 themes and 3 of the 8 present singles alerted, while over all history 30/31 themes and 22/23 singles have alerted — so "only 8" is a statement about the window, not the cap. But the cap was **silently** truncating whenever a wide window pushed the alerting set past 15, and a silent cap reads as "that is all there is". The ceiling now exceeds the largest set the filters can produce, and the count is printed. Verified live: "8 of 34 themes alerted in this window (2026-01-01 to newest) … Showing the 6 most recent; raise "items per section" in the sidebar to see the other 2." |
| Lower-panel caption rewritten again | grey = watching / coloured = live, plus the two counter-intuitive cases | CONVENTION (display) | a two-state line is only honest if the panel says which state is which and what each state licenses. The caption now states that grey is a 7-day-smoothed reading **with no verdict attached**, that it does not respond to hover, that it **can sit above 100 with nothing happening** and that **flat on the axis means the attention gate was shut, not missing data**; and that the coloured stretch is "the one the detector actually acted on, and a crossing here is an alert" |

### Class 8 continued — the fifth pass: one solid line, no interpolation, no state change; a persistence rule rejected on evidence; coverage stated; the influence tab split (2026-07-28)

*Four desk items, again **none of them about the detector**: "why does euphoria
singles only show 3 graphs?"; "i liked the original 7 day on the graph actually
instead of the dotted lines (interpolated lines)"; "it feels quite random, just
based on 1 dot it fires? why is this ok?"; "influence tracker why is everything
crowded long?"; "i liked the euphoria graph back how it was before" — narrowed
mid-pass to **"but same as nbefore but 0 to 100% now"** and, on the one-day
question, to **"if it is a 1 day window does that make it inaccurate?"**, which
is a question about accuracy and was therefore answered with a measurement
rather than a code change. **No number in Classes 1–7 moved; the frozen
thresholds (0.630231 / 0.848141) and the alert dates are bit-identical.** This
pass **knowingly reverses the fourth pass above, on the same day**: the desk
does not want two visual states, and the reversal is recorded rather than
silently overwritten because the sequence is the argument (research report
§6.16). One item was put back to the desk rather than guessed — which line to
draw ("one solid line, % of trigger"; the accepted preview stated in terms
"eligibility is no longer visible on the line") and how to answer the chart
count ("say so on the page", i.e. no filler charts). Evidence: `dashboard.py`
readiness-panel block and its "WHY NO INTERPOLATION IS NEEDED NOW" comment;
`analytics/euphoria_phases.py:683` traced for the gap meaning; offline
measurement against the live stores; Playwright DOM verification.*

| Change | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| **ONE SOLID LINE PER RULE** replaces the two-state watch track | `_s.combine_first(_tr)` — stored score where it exists, tracking score elsewhere — 7-day smoothed, drawn `mode="lines"`, solid, rule colour, one legend entry, as **% of that rule's own frozen threshold** | **DESK DECISION 2026-07-28 (display)**, reversing the fourth-pass row above the same day | the desk read grey-then-coloured and asked for the earlier line back on a percentage axis. Two visual states cost a decoding step on every glance; eligibility is already carried by the flags. 100 is the trigger **by construction** (`score / threshold × 100`), so the axis needs no caption to explain it, and no new quantity is introduced — same scorer, same frozen threshold, divided |
| **NO INTERPOLATION ANYWHERE ON THE PANEL** | no `.interpolate()`, no `limit_area`, `connectgaps=False` | **CONVENTION (display), and the reason the third pass was rejected** | an interpolated point is a guess about a number nobody computed. Every point on the shipped line is a real evaluation of the production scorer |
| The stored score WINS on every judged day | `combine_first`, stored first | **the load-bearing guard, carried over from the fourth pass and strengthened** | `_smooth_by_name` rolls over each name's candidate-day **sequence, not the calendar**, so the wide recomputation is not the stored number: over 1,600 overlapping name-days median abs diff **0.0000**, p95 **0.2234 (35 threshold-points)**, max **0.7505 (119 points)**, **disagreeing about the crossing on 66 days (4.1%)**. Priority to the stored value makes the panel **incapable** of printing a number that contradicts a flag. **The honest cost: the divergence becomes unobservable rather than displayed, so this panel cannot audit the tracking scorer** — that is a notebook's job |
| `INK_MUTED` / dashed / `hoverinfo="skip"` / `limit_area` **retired from this panel** | there is no second trace left for them to style | consequence of the row above | recorded so a future reader does not look for a grey trace the fourth-pass rows describe |
| **WHAT A GAP IN THE GET IN LINE MEANS** | GET OUT never breaks (DOM: 1 segment, 0 nulls, every chart); a GET IN break means the crowd was not building | **GROUND TRUTH — verified against the store's definition, not assumed** | **a wrong claim in my own caption was caught here before delivery.** The draft said a break meant "no crowd at all to score". The onset store is literally `frame_live[frame_live.hype_raw >= 1]` (`euphoria_phases.py:683`) and store membership agrees with that test on **61,872 of 61,872** day-frame rows — zero exceptions. Over the 140,858 calendar days inside names' onset spans: **23.1%** carry a row, **20.5%** were measured but the 7-day chatter share sat **at or below that name's own 120-day median**, **56.5%** never reached the day frame (`build_day_frame`: coverage gate `es.coverage_ok`, day before the judgeable window `j0`, or too little history for the percentiles). One fifth of gaps DO have chatter, so neither the caption nor the docs may say "no crowd at all"; the measurement is written into the source comment beside the code |
| `phase_day_frame.parquet` **REJECTED** as the GET IN base | 61,872 rows × 20 cols, every `ONSET_BANK` feature, 1,505 rows/name, 98.5% one-day dense | **REJECTED on data** | it is the only available way to make GET IN unbroken, and it ends **2026-06-06** while the live window runs to **2026-07-21**. Continuity bought with six weeks of the most recent signal is a pure loss on a desk tool. No code from this attempt remains |
| **THE PERSISTENCE RULE — REJECTED** ("fire only on a full 7-day window") | proposed by the desk's own "just based on 1 dot it fires?"; built as a measurement, never shipped | **REJECTED on evidence — this row is its only trace** | the picture is accurate: **109** above-threshold GET OUT runs, median length **4**, **29 of length 1 (26.6%)**; firing-day depth **69.5% full window vs 25.3% one day**. But thin alerts are **not** less accurate — **11/29 = 37.9%** hit vs full-window **24/66 = 36.4%**, **Fisher exact p = 1.0**. The cost is real: the rule deletes **29 of 95 alerts, 11 hits, and 10 captured episodes outright** (BBBY 2021-02-11, CLOV 2021-09-07, MVIS 2020-12-22, NOK 2021-01-27, SNDL 2021-06-03, biotech_pharma 2019-12-24, gold_metals 2026-01-29, meme_stocks 2020-02-19, semiconductors 2019-04-24, space 2020-06-08), taking episode capture **26 → 18**. Paying ten episodes for a 1.5pp precision change a Fisher test cannot separate from zero is not a trade. **The signal was not band-aided; the explanation was improved** — the panel now shows the smoothed score every day, so a thin fire is visibly a fast climb through 100 |
| **COVERAGE REPORTED SEPARATELY FROM ALERTING** | a line above the charts: "**3 of 25 single names with data in this window alerted** (2026-01-01 to newest). Only names that actually alerted are charted" | **DESK DECISION 2026-07-28 (display)** — chosen over adding filler charts | "why only 3 graphs?" has two different answers (how much of the universe was watchable, how much of it alerted) and printing one number invited inferring the other. Denominator = names with rows **in the window**, because a name whose history ends earlier is *absent*, not *calm*; measured on the 2026-07-21 store all 25 singles reach the last day, so the absence clause does not fire and the three that alerted are AAPL, MSFT, PLTR. The standing "no filler names" instruction forbids charting the other 22 |
| **INFLUENCE TAB SPLIT INTO FOUR SUB-TABS** | *What they are pushing* / *Building or fading?* / *The names* / *The map*; shared controls **above** the sub-tabs; the §6.12 no-model footnote left at page level | **DESK DECISION 2026-07-28 (layout only — nothing removed)** | *"why is everything crowded long?"* was a legibility complaint about one scrolling column in which four exhibits competed. Controls stay above so **one population feeds every view** — a view whose population changed with the tab could not be compared with its neighbours — and the no-model footnote governs all four, so it cannot live inside one of them |

### Class 8 continued — the sixth pass: the level curve returns beside the readiness line on ONE shared 0-100 axis, and a window-dependent smoothing artefact is fixed (2026-07-29)

*One desk item, delivered as a photograph of the third-pass chart taken off the
screen: **"i like this original graph more BUT i want it to be 0 to 100% signal
fires like what you did in these newer ones, can you help me implement that? as
in amix of the two (i like the continous line of this image but i like the 0 to
100% of the new one)"**. It is a request for a **mixture**, not a reversal: the
continuous level curve of the earlier panel, on the percentage-of-trigger axis
of the newer one. Two design choices were put back to the desk rather than
guessed, and both answers are binding here: **what returns** = "smoothed level
+ readiness only", so the raw daily level, the ochre eligibility ribbon and the
dated hottest-day marker stay retired; **the axis** = "one shared axis, like the
photo", so the dual-axis option was rejected. **No number in Classes 1-7 moved;
the frozen thresholds (0.630231 / 0.848141) and the alert dates are again
bit-identical — across all six passes.** This pass is the only one of the six
that also **fixed a measurable defect**, found while implementing the cosmetic
request: the display smoothing was being applied AFTER the sidebar window clip.
Evidence: `dashboard.py` level-series block and the readiness-panel block;
offline recomputation across all 59 charted names against the live stores;
Playwright DOM verification of the rendered panel.*

| Change | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| **THE LEVEL CURVE RETURNS TO THE LOWER PANEL** | `go.Scatter` of the 7-day-smoothed `level`, `ACCENT` navy, width 1.8, `connectgaps=False`, **drawn first** so the readiness lines sit on top of it, legend entry "euphoria level (7d smooth)", hover "euphoria level: N/100" | **DESK DECISION 2026-07-29 (display)** | the fifth pass argued that the dial above the chart carried the level, so the panel did not need it. That argument was **right about the QUESTIONS and wrong about the READING, and it is withdrawn here**: a dial is a single number, so it can say how hot today is but it cannot show that the crowd had been building for three weeks — which is the thing a PM asks about the moment the readiness line moves. Drawing order encodes the hierarchy: the level is **context**, the readiness lines are **the decision** |
| **ONE SHARED 0-100 AXIS; the dual axis REJECTED** | both series on the row-2 y-axis, `range=[0, 125]`, title `"0-100: level, and % of trigger"` | **DESK DECISION 2026-07-29 (display)** — the rejected option is recorded, not left as code | one axis is defensible **because `level` is bounded 0-100 by construction** (live store max exactly `100.0`, min `0.2577`) and readiness is a percentage, so the two share a range without rescaling and 125 already clears both. A second axis would have let the **100 rule be placed anywhere relative to the level curve**, reintroducing precisely the "why is the threshold HERE?" ambiguity that the percentage axis was adopted to kill in the fifth pass |
| **THE LEVEL IS SMOOTHED ON THE UNCLIPPED HISTORY, THEN CLIPPED** | `ek[ek["name"] == name]` → drop duplicate dates → `rolling(ROLL, min_periods=1).mean()` → `.reindex(one_i.index)` | **BUG FIX (window-dependent reading)** | the mean used to be taken **after** the sidebar clip with `min_periods=1`, so the first six days of any window were the average of one, two, … six days — a ramp-up artefact — and **the same calendar day read differently depending on how far back the reader happened to be looking**. Measured across all **59** charted names: the **last day is identical for all 59** (so the dial and the "today" fact were never wrong), **at most 6 days move** per name (median 6, i.e. exactly the ramp), the largest single-day correction is **22.08 level points**, and **5 of 59** names had a stated *window peak* that was an artefact of where the window started. **No alert changed, and none could**: `level` is not an input to either rule (the Reddit-only hard rule keeps price out; the level is a display quantity) |
| The readiness line needed no change | it was already built on the unclipped frame (`_base`) | consistency, not a fix | the readiness line has always smoothed before clipping — the defect was that the **level did not**, so two series on one panel were built two different ways. This row exists so a reader does not "fix" the readiness line to match the old level behaviour |
| **THE LEVEL IS SKIPPED IN THE NO-DESK-STORE FALLBACK** | guard `_lvl_on_panel = bool(_ready) and _ready[0][0] != "SIGNAL"` | **LOAD-BEARING GUARD (display honesty)** | in the fallback path the readiness line **is** the level divided by a threshold. Drawing both would put **the same series on the panel twice at two different scales**, which reads as two independent confirmations of one reading. One series, once |
| The caption states that the two lines are DIFFERENT QUANTITIES | "Both run 0-100 and higher is hotter in both, but they are not the same quantity — a level of 70 is not 70% of the way to a signal, so read each line against its own legend entry" | **CONVENTION (display)** — the cost of the shared axis, paid explicitly | a shared axis invites reading one line off the other. The mixture the desk asked for is worth that cost, but only if the panel says out loud what the axis cannot |

### Class 8 continued — the seventh pass: the panel says which days the exit question was asked on, and what each line actually is (2026-07-29)

*One desk item, three questions, asked of the sixth-pass chart: **"how does this
chart even work? how can there both be get in and get out at the same time?
doesnt that not make sense? / is the red line the gradient of the black line? it
is very unclear."** Two of the three had **correct answers the panel was hiding
rather than wrong answers**, and finding that out required reading the firing
path and measuring the shipped store rather than trusting the caption. The red
line is **not** a derivative of the navy one — they share **four of five
ingredients** (`level = 100 × mean(e1, e2, e3, e5)` at `analytics/euphoria.py:330`
vs `TOP_FEATURES = [e1, e2, e3, e5, fade]`), so red is a **sibling**; the
slope-like line is the **teal** one, four of whose five inputs are
short-window-versus-long-window ratios. And the two rules **cannot both fire**,
because candidacy already separates them: GET OUT is scored only where
`end_stage_mask` holds, GET IN's frame is explicitly `~end_stage_mask`. Two
design choices were put back to the desk rather than guessed, and both answers
are binding here: **judged vs not** = "shade the end-stage span", so both lines
stay solid and continuous and a faint band marks the region; **red vs black** =
"say it in the caption", so the legend entries were **not** renamed. **No score,
threshold, feature or alert date moved — the alert dates are bit-identical across
all seven passes.** Evidence: `analytics/euphoria_phases.py:984/1000/1010/1018`
read line by line; offline measurement of `euphoria_desk.parquet` (63,345 judged
name-days); Playwright DOM verification of the emitted shapes; the rendered
screenshot read twice.*

| Change | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| **THE END-STAGE BAND** | one `add_vrect` per contiguous run of `dk_i["end_stage"]`, padded `±12h`, `BEAR` at opacity **0.06**, `line_width=0`, `layer="below"`, row 2; labelled **once** on the widest run as `"shaded: exit question only"` | **DESK DECISION 2026-07-29 (display)** — a **partial reversal** of the fourth-pass "signal lines, and nothing else" removal, argued row by row rather than silently overwritten | the panel had no way to say that a teal crossing inside the end stage **cannot fire**, so a correct reading looked like a contradiction. Of that removal's three reasons, two do not apply and the third is answered: shading is wrong for a **dated verdict** but right for a **region statement**, and "entry was off the table across these days" is a region statement carrying no number; "three translucent bands mix into a fourth colour" cannot happen with **one** band; a 6% tint survives greyscale and print, and the label prints as text. `end_stage` is read straight off the store — nothing measured is added or removed |
| **THE TWO RULES CANNOT BOTH FIRE — measured, not asserted** | of **63,345** judged name-days: **0** with both readiness lines ≥ 100, **0** firing both alerts (95 GET OUT / 156 GET IN historically, never the same name-day), **0** end-stage days carrying a GET IN score; **189** carry two non-zero scores but none reaches both triggers | **GROUND TRUTH** | the desk's question deserved a number, not a promise about the code. The disjointness is structural (`desk_candidacy`: `end_f = frame_px[hype_ok & boom_state]` vs `onset_f = frame_px[(hype_raw >= 1) & ~end_stage_mask]`), and `desk_end_fit` additionally zeroes the exit score off end-stage days. The 189 two-score days are the trailing `ROLL = 7` mean dragging a score across a state change — **visible on the chart, harmless at the trigger**. `episode_coherent_alerts` then suppresses a GET IN landing within `EUPHORIA_COOLDOWN_DAYS = 21` after a GET OUT |
| **THE TRACKING FILL IS RETAINED; "let the teal line break" REJECTED** | `_s.combine_first(_tr)` stays exactly as the fifth pass shipped it | **DESK DECISION 2026-07-29** — the rejected option is recorded, not left as code | the defect was **ours**: judged and hypothetical readings were drawn in identical colour and weight. Two fixes were offered; the desk chose the band. Dropping the fill would have cut GET IN to **~10 visible points** on the charted name (theme `memory` / SMH: judged on 10 of ~82 days from 2026-05-01, fired **zero** times, 79 days end-stage), which trades one misreading for a chart that looks broken |
| **THE CAPTION STATES THE LEVEL ↔ GET OUT RELATIONSHIP; renaming the legend REJECTED** | added: GET OUT is "the same blend of ingredients plus one more (mood rolling over while the crowd is still large), rescaled so 100 is its trigger, which is why the two move together"; GET IN "is the slope-like one"; shaded days give "a what-it-would-have-said reading" that "cannot fire, however high it goes" | **DESK DECISION 2026-07-29 (wording)** | the panel's grammar is now stated in words: **navy = how high, red = high and rolling over, teal = climbing fast**. Renaming the legend entries was offered and not chosen — the legend names the *rule*, and overloading it with the rule's *construction* would make every entry a sentence |
| **THE LABEL IS SHORT BECAUSE A LONGER ONE MEASURED WIDER THAN ITS BAND** | `"shaded: exit question only"`, on the widest run only, `y=0.02` in `yref="y domain"` | **CONVENTION (display), caught in the rendered screenshot** | the first draft, `"shaded: exit question only - entry not asked here"`, rendered **wider than the 06 May–10 Jun run it sat on**, so it read as if the shading spanned 03 May–14 Jun. On the chart the label only has to say *which region is meant*; the full sentence lives in the caption, where it has room. Labelling every band would have re-created the collision defect the signal marks already solved |
| **PLOTLY TRAP: `add_vrect` silently drops shapes on trace-less subplots** | the band block is placed **after** the readiness trace loop, not before it | **LOAD-BEARING PLACEMENT (recorded so it is not "tidied" back)** | `add_vrect` defaults to `exclude_empty_subplots=True`, so a band added to row 2 before row 2 has traces is **dropped with no error** — and the annotation beside it still renders, so the panel looks merely unshaded. The first cut failed exactly this way; an instrumented DOM read printed `runs=3 shp=0`, which is what identified it. Z-order does **not** depend on insertion order (`layer="below"` does that), so moving the block down costs nothing |
| Verification of this pass | **108** tests pass; flake8 **41** findings, identical to the delivered sixth-pass baseline; DOM read: **3** rect shapes on `x2` / `y2 domain` at opacity 0.06 `layer="below"`, 5 traces, **no page errors**; and **all four** GET OUT alert dates (2026-05-10, 05-31, 06-22, 07-13) fall **inside** shaded runs | **GROUND TRUTH** | the last check is the internal-consistency one: GET OUT can only fire on an end-stage day, so a vertical mark outside a band would prove the store and the display disagree. It also confirms the band is being read from the same rows the alerts came from |

## Class 9 — PIPELINE CADENCE & THE RESEARCH CONTRACT (added 2026-07-28)

*Desk instruction: "why does the update_data have to check what is the best
model everytime? shouldnt it just used the already pre established best model?
no need to show me the stats everytime. if i need to know the stats i will use
th enotebooks. update_data should simply be to just update the data and run the
model on these new data downloaded". This class exists because the answer is a
**contract about when a number is allowed to change**, and until now that
contract lived only in source docstrings — including two that pointed at
sections of this register and of DECISIONS.xlsx that did not hold it. Evidence:
`update_data.py` module docstring, `analytics/euphoria.py::needs_research` and
`::record_lags_data`, DECISIONS.xlsx sheet `3b. Pipeline & Cadence`.*

| Rule | What it is | Class | Why / what the evidence says |
|---|---|---|---|
| A live run NEVER re-selects | `update_data.py` does not choose a model, re-select a threshold, or re-run the walk-forward / ablation / ML challenger | DESK DECISION 2026-07-28 | research decides ONCE, in the notebooks, and the answer is frozen into a stored record. A live run refreshes data and scores it with the already-frozen winner. This is not a shortcut: **re-fitting on every run makes the number on screen untraceable** — nothing on disk would describe how today's threshold differs from yesterday's, and a threshold nobody can reconstruct cannot be defended |
| The ONE exception: the bootstrap | a machine with no frozen record at all derives one, once | DERIVED | `needs_research(stored)` returns True only for `not stored or not stored.get("thresholds")`. You cannot score against a record that does not exist |
| Two explicit ways to re-open research | `python -m analytics.run_analytics --what phases --research`, or `python update_data.py --full` | CONVENTION | both are typed on purpose, never reached by drift. `--full` counts as research **because a backfill rewrites the history the thresholds were chosen on** — scoring new history against thresholds fitted on the old history would be a silent lookahead |
| Staleness is reported, not repaired | `record_lags_data` returns the newest year the frozen record covers when the data has outgrown it; the run prints ONE notice line and keeps scoring | DERIVED | a live run in this state is perfectly legitimate — it is out-of-sample use, which is **exactly what the walk-forward licenses**. Refitting in January because a new year began does not make the threshold more correct; it makes it a moving target no stored record describes. So the pipeline tells the desk to run the research pass instead of quietly doing it |
| No stats printed on a live run | performance output belongs to the notebooks | DESK DECISION 2026-07-28 | same principle as the dashboard's no-performance-metrics decision (Class 8): the research record is the notebooks, the operational surfaces state conclusions |
| Cadence | the pipeline is run roughly **2× a week**; a full refresh must stay under ~10 minutes | DESK DECISION | Alex's answer when asked directly. `PIPELINE_BUDGET_S = 600` is that instruction expressed as a number the code can enforce |

**Pointer correction, recorded because it was wrong in shipped source.**
`analytics/euphoria.py` promised this contract lived in DECISIONS.xlsx
`"2. Pipeline & Cadence"` and PARAMETER_REGISTER `Class 6`. Neither held it —
sheet 2 is *Literature* and Class 6 is the *influence tracker*. Rather than
renumber existing sheets and classes (every other cross-reference in the repo
would break), the missing sections were CREATED — this class, and sheet
`3b. Pipeline & Cadence` — and the two docstrings were repointed at them.

*Full derivations: `analytics/euphoria.py` and `analytics/euphoria_phases.py`
docstrings, `analytics/influence_graph.py` and `analytics/influence_ml.py`
docstrings, `src/pipeline_budget.py` docstrings, `src/config.py` inline
comments, `docs/DECISIONS.xlsx`, notebooks 01–06 and notebook 05 for the
influence tracker.*
