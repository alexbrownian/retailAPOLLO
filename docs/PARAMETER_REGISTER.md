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
| Desk trigger smoothing | 7 d (ROLL, trailing) | the house one-week window, reused | measured raw vs smoothed (NB06): GET OUT AP 0.435→0.449, FA 41→39, −2 captures; GET IN adjacency 8→2 on the phase-aware frame; one-day blips structurally removed. Full sweep w∈{1,3,5,7,10,14} (NB07): no window strictly dominates w=7; run-rule triggers k>1 also dominated by the smoothed single crossing |
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
| Comments decoupled; dynamic panel (cap 1/month) | 2026-07-24 | slow fetch out of the daily path; panel expands where the crowd points, one denominator step per month. **The decoupling half was REVERSED on 2026-07-27** — see the next row and Class 7; the dynamic-panel half stands |
| Comments BUDGETED back INTO the live pipeline | 2026-07-27 | an influence board that rescores month-old comments is not a live board, so the fetch returns to every run — but under a measured page allowance rather than uncapped, because a 7-day gap costs ~16.5 min against a 10-min ceiling. Hitting the cap defers (the watermark does not advance), so the board is never silently partial, only ever less fresh. Every number behind the allowance is in Class 7 |
| Phase-aware onset tested & REJECTED | 2026-07-24 | halves start/end adjacency and LATE alerts but costs 5 captures with no utility gain (NB03); adjacency readability solved by episode-span shading instead (display, zero capture cost) — **superseded the same day by the desk configuration (below)** |
| DESK CONFIGURATION adopted (GET IN / GET OUT) | 2026-07-24 | the desk stated three times that a START landing on an END destroys PM trust → adjacency overrules the raw-capture utility rule. GET OUT = boom-gated + 7d-smoothed END (cap 24/122, FA 39, AP 0.449); GET IN = phase-aware + 7d-smoothed onset (adjacency 20→2, LATE 21→10, FA 169→124, capture cost 29→20 RECORDED). Selection rule pre-stated; NB06 drift guard pins notebook == production |

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
| Per-author hit rate on the dashboard | never displayed | DESK DECISION 2026-07-27 | still computed, still stored in `author_scores.parquet`, still an input to the composite — simply not on this screen. (i) A raw hit rate and a shrunk composite answer different questions and disagree **by design**: shrinkage exists precisely to stop 3-from-3 outranking 28-from-40, so a reader reconciling the two columns is fighting the method. (ii) An unqualified per-person accuracy invites position sizing off a sample of five, which the cohort split says nothing here supports. Reported in notebook 05 **with sample size and CI attached**, the only defensible form. The stored column name is unchanged — a schema test asserts it |

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

*Full derivations: `analytics/euphoria.py` and `analytics/euphoria_phases.py`
docstrings, `analytics/influence_graph.py` and `analytics/influence_ml.py`
docstrings, `src/pipeline_budget.py` docstrings, `src/config.py` inline
comments, `docs/DECISIONS.xlsx`, notebooks 01–06 and notebook 05 for the
influence tracker.*
