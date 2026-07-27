# Detecting the Start and End of Retail Euphoria from Crowd Data Alone: A Walk-Forward Study on Nine Years of Social-Media Attention

**Alex Brown — GIP 2026 Project — MAARS Global Macro**
*RetailRadar research report — LIVING DOCUMENT (see Changelog, Appendix C). Last updated 2026-07-27.*

---

## Abstract

Retail attention episodes — GameStop (2021), gold (2026), the meme-stock
complex — follow a recognisable arc: a crowd arrives, sentiment goes
one-way, attention grows super-exponentially, price peaks, and a bust of
15–90% follows within weeks. This report asks whether the START and the
END of such episodes can be detected from **crowd data alone** (mention
counts and scored sentiment across 17 finance subreddits, X and
StockTwits), with price never entering any predictive input — price
defines ground truth and scores the detectors, nothing more. We build a
price-defined catalog of 333 boom-bust episodes (2017–2026) across 59
instruments (34 theme-anchor ETFs, 25 data-chosen single names), evaluate
two five-feature crowd-only banks with an importance battery adapted from
Chan (2026) — per-feature AUROC/AP with instrument-cluster bootstrap
confidence intervals, leave-one-out ablation, noise perturbation — and
run a walk-forward model tournament (random baseline, rule bank, logistic
regression, gradient boosting, MLP) under a criterion pre-stated before
any result was computed. The rule-based detectors win both tasks under a
parsimony rule; every learner either ties inside bootstrap noise or
(the MLP) falls below the random baseline. Out of sample, the onset
detector captures 23.2% of coverage-detectable episode starts with a
median entry 17 days after the trough and a median 66 days of rally
still ahead, at 0.35 false alarms per instrument-year; the top detector
(unchanged from the validated incumbent) captures ~23% of detectable
peaks with a median 6-day lead. A trading-translation study (onset→buy,
top→sell) was tested under its own pre-stated criterion and **rejected**
— the detectors' demonstrated value is risk timing and monitoring, not
standalone alpha, and this report says so plainly. A companion model
ports the thesis's graph-learning question to the project's live
influence store: can high-predictive authors be identified from
behaviour and reply-graph position before their track record is known?

---

## 1. Introduction

### 1.1 Motivation

Retail crowds move prices at the extremes. The literature established
the mechanism years before GameStop made it front-page: attention-driven
buying pushes prices beyond fundamentals, and attention extremes predict
*reversal*, not continuation (Barber & Odean 2008; Barber et al. 2022).
What a trading desk needs is not the correlation but the **clock**: when
has a crowd episode *started* (the rally has fuel), and when is it
*ending* (the top is near)? Those are different questions with different
observable signatures, and this project answers both with one
architecture and one hard rule:

> **The crowd-only rule.** Every predictive input is built from the
> social-media aggregates. Price never enters any feature, level, gate
> or alert. Price exists only to define ground truth and score the
> detectors. The claim being defended is *"the crowd alone called it"*,
> and a unit test enforces it (no detector function accepts a price
> argument).

### 1.2 Contributions

1. **An episode ground truth** (§4): 333 price-defined boom-bust arcs
   over 2017–2026, built by extending the project's validated peak
   definition backward to its trough and forward to its bust — no new
   fitted quantities.
2. **A systematic feature study** (§5.2): five onset candidates, every
   window derived from an already-validated constant, evaluated with the
   test battery of Chan (2026) — including one feature (`source_breadth`)
   with the best raw AUROC that we **reject** because an integrity check
   shows its skill is a coverage-regime artifact.
3. **A pre-registered model tournament** (§5.3–§6.2): rules vs learners
   under identical inputs, gates and walk-forward discipline, with the
   selection criterion written before results; the rules win both tasks.
4. **An honest operating record** (§6.3): capture, lead-times, false
   alarms against a derived budget, label-sensitivity, calibration — all
   regenerated from current data on every rebuild, with both
   denominators (all episodes vs coverage-detectable) always reported.
5. **A trading-translation verdict** (§6.5): pre-stated criterion,
   negative result, recorded — the same treatment the project's retired
   BUY/SELL engine received.
6. **A standing influence-graph experiment** (§5.5): the thesis's
   semi-supervised node-classification task ported to the desk's own
   live, text-free influence store.

### 1.3 Why this is useful for trading (the short version)

Full discussion in §7. In one paragraph: the END detector is a
**risk-timing overlay** — busts in the catalog average double-digit
drawdowns within 90 days, and a flag that arrives a median 6 days before
the peak on ~1 in 4 detectable manias, at ~1 false alarm per instrument
every 4 years, is actionable for trimming longs into strength, gating
new entries, and hedging theme exposure. The START detector is a
**radar** — flagged names have, historically, a median 66 days of rally
still ahead, which is watchlist-building lead time, not a buy signal
(the buy claim was tested and rejected). The influence tracker is a
**signal-quality filter** over the noisiest data source a desk touches.
And the sparse-by-design alert policy means the whole system costs
minutes of attention per week, not hours.

---

## 2. Related Work

**Attention and reversal.** Barber & Odean (2008) establish
attention-induced buying by retail investors; Barber, Huang, Odean &
Schwarz (2022, JFE) show attention-herding episodes on zero-commission
platforms are followed by negative abnormal returns. These motivate
features E1 (attention extremity) and E3 (crowd influx), and the
contrarian reading of attention extremes.

**Bubble signatures.** Sornette's log-periodic power-law (LPPLS)
framework identifies super-exponential growth as the mathematical
signature of an unsustainable, self-reinforcing process; applications to
meme stocks exist (arXiv:2110.06190). Under the crowd-only rule we apply
the signature to **attention** rather than price (feature E5/O5): crowd
contagion whose growth rate is itself growing must saturate, and
attention saturation is where tops form.

**Social-media finance.** Studies of r/WallStreetBets (e.g. Jame et
al.) document both information content and coordinated-behaviour
distortion in aggregated retail sentiment — motivating per-name
percentile normalisation (a name's "extreme" is measured against its own
history, never absolute counts that coverage shifts could fake).

**Methods source.** Chan (2026), *Informed Trading Decisions via Social
Network Analysis* (Oxford M.Eng thesis), supplies (i) the influence
tracker's author-scoring method (§4.5–4.6 of the thesis: volatility-
scaled correctness bar, confidence-weighted scores, Bayesian shrinkage,
composite tiers), already implemented verbatim in `analytics/influence.py`;
(ii) the evaluation discipline this study adopts throughout — AP and
AUROC as threshold-independent primary metrics under class imbalance,
class-weighted training, multi-seed reporting, threshold tuning as
constrained optimisation on validation data only, single-feature
ablation with the correlated-features caveat, graph perturbation tests,
and labelling-criteria sensitivity; and (iii) the graph-learning
question ported in §5.5. Their benchmark: GraphSAGE, AP 0.140 ± 0.002,
AUROC 0.632 ± 0.016 on 2,617 nodes at 13:1 imbalance.

---

## 3. Data

**Crowd data.** Daily, text-free aggregates (the committed
`ABSTRACTED_DATA/` contract: no titles, bodies, authors or IDs ever
leave the ingestion boundary; a write-time check enforces it) covering
17 finance subreddits plus X and StockTwits: per-name mention counts,
post counts, average sentiment, net-bullish share, 2017-01-01 →
present. Sentiment is VADER-scored at ingestion.

**Prices.** Bloomberg daily closes (`PX_LAST`), 2017-01-03 → present,
~287 symbols, incremental append-only store. Prices are used only for
ground truth and scoring (the crowd-only rule).

**Universe.** 59 instruments: 34 theme-anchor ETFs (rates/bonds and
real-estate themes excluded by desk decision) and the top-25
most-mentioned single names with ≥3,000 scored posts — data-chosen, not
hand-picked, because the point is catching the *next* GME.

**Dynamic panel (added 2026-07-24).** The subreddit list self-expands
where the crowd points: a monthly review mines panel text for `r/<name>`
referrals, qualifies candidates at ≥100 unique referring authors per 28
days (the A0 coverage floor, reused), screens them for finance content
with the panel's own ticker-mention rate as the ruler, and auto-adds at
most one community per month into an EXPLORATION tier (the founding 17
are the frozen CORE). Every addition is logged in a committed manifest
because a panel change steps the mention-share denominator — the
`source_breadth` lesson institutionalised: coverage-regime changes must
be *recorded and measurable* (a local by-subreddit aggregate exists for
exactly that re-cut), never silent. Comments ingestion was simultaneously
decoupled from the daily pipeline into a dedicated runner
(`update_comments.py`) — an operational change only; no analytic input
changed. *That decoupling was reversed on 2026-07-27*: comments are now
fetched on every live run under a measured page allowance (§3.1b), because
an influence board that rescores month-old comments is not a live board.
`update_comments.py` survives as the unbudgeted runner for backfills and
long-gap catch-up. Still an operational change only — no analytic input
changed, and the euphoria detector never reads comments.

**Coverage honesty.** Archive coverage is deep in 2020–2022 and 2026 and
thin in 2023–2025 (2024–25 have zero detectable episodes). Every rate in
this report is therefore stated against two denominators: *all* episodes
(price truth) and *detectable* episodes (the coverage gate A0 — ≥100
scored posts in 28 days — held during the relevant window). The detector
is blind where coverage is absent, not wrong; conflating the two is the
classic way this kind of study lies to itself.

---

## 4. Problem Formulation

### 4.1 Episodes (ground truth; price-only, testing-only)

A confirmed **peak** is the project's validated definition, unchanged:
G1 local 43-day maximum; G2 boom — close ≥ 25% (ETF/theme) or 50%
(single name) above the minimum close of the preceding 120 days; G3 bust
— drawdown ≥ 15% / 30% within the following 90 days. An **episode**
extends each peak into an arc using only quantities G2/G3 already
measure: the **trough** is the argmin of the same 120-day prior window,
and the **bust date** is the first day G3's condition is met. The
catalog holds 333 episodes; medians: boom +42%, bust −24%, run length
86 days (right-truncated at 120 by construction — recorded, §8).

### 4.2 Hit windows and the LATE bucket

- **Top alert hit:** inside `[peak−30d, peak+1d]` (the incumbent's aim).
- **Onset alert hit:** inside `[trough, min(trough+45d, peak)]`. The
  45-day length mirrors the existing false-alarm horizon in
  `score_alerts` (no new constant); the cap at the peak prevents an
  alert fired *after* a fast top from counting as "caught the start".
- **LATE ≠ FALSE:** an onset alert inside `(window end, peak]` fired
  during a genuine rally but after its start. Counting it a hit would
  inflate the onset claim; counting it false would punish an alert
  inside a real episode. It gets its own reported bucket.
- **Judgeable window:** alerts are judged only where price exists at the
  alert and for 45 days after; recent alerts are **pending**, not false.

### 4.3 Tasks

Binary, rare-positive detection per (instrument, day): is this day
inside an onset window (prevalence ≈ 6%), or inside a top window
(≈ 6%)? Alerts must be sparse (cooldown 21d — one alert per episode) and
are evaluated operationally (episodes captured, lead times, false-alarm
rate) as well as by threshold-independent score quality (AP, AUROC).

---

## 5. Methodology

### 5.1 Design rules

All features are trailing (day *t* uses only data ≤ *t*), all are
percentile ranks against the same instrument's own trailing 365 days
(fat-tailed inputs → ranks; coverage shifts → shares), and every window
is inherited from an already-validated constant. No number in the
feature layer is fitted.

### 5.2 Feature banks and the importance battery

**Top bank (validated, unchanged):** E1 attention extremity, E2
sustained bullishness (persistence-gated), E3 crowd influx, E5
super-exponential attention growth, E4 fade flag.

**Onset bank (new, aimed at the left side of an episode):**

| Feature | Definition | Provenance |
|---|---|---|
| `attention_accel` | rank of (7d share − 28d share) | 7, 28 = existing ROLL / E-windows |
| `hype_ratio` | rank of 7d share ÷ own 120d median | the A1 hype gate's exact ratio, continuous |
| `bull_inflection` | rank of 14d change of 14d net-bullish share | 14 = the fade rule's window, reversed |
| `influx_speed` | rank of 14d change in share | E3 at half the horizon |
| `attention_convexity` | ≡ E5 | contagion acceleration is inherently early-phase |

The battery (adapted from thesis §6.2.3/§7.2): per-feature AUROC/AP
against the episode labels with **instrument-cluster bootstrap** 90% CIs
(daily rows within a name are serially dependent; the instrument is the
plausibly-independent unit), drop-one ablation of the un-weighted bank
mean, rank-space noise perturbation, and a Spearman matrix. A sixth
candidate, `source_breadth` (how many platforms mention the name),
posted the best raw AUROC (0.65) and was **rejected**: the by-source
archive shows X/StockTwits data essentially exists only from 2026, so
the feature encodes *which coverage regime it is*, not crowd behaviour.
The check, not the AUROC, is the finding (notebook 02).

### 5.3 The tournament (criterion pre-stated)

Contestants per task — random (seeds 42/100/2026), the rule bank
(un-weighted mean; for the top task with the incumbent's A2/A2b gates
baked in), logistic regression, small gradient-boosted trees, one-layer
MLP — all seeing exactly the bank features, all behind identical
non-fitted prerequisites (coverage gate; for onset, attention ≥ its own
120d median — the A1 construction with multiplier 1, parameter-free;
for top, the incumbent A1 at 2×), all walk-forward (train strictly on
earlier years).

**Criterion, recorded before results:** (Layer 1) rank by AP, ties by
AUROC; a model must beat random on both AP and AUROC (score quality is
where chance is beaten — a random alerter's raw capture count is
inflated by indiscriminate firing and is deliberately not the bar); a
learner dethrones the rules only if its AP clears the rules' outside a
paired instrument-bootstrap 90% CI (**parsimony rule** — the standard
that rejected the project's first ML challenger). (Layer 2) the
operating threshold per test year is chosen on train years only:
maximise captured episodes subject to a false-alarm budget **derived**
from the incumbent's accepted 0.23 FAs/instrument-year. (Layer 3)
out-of-sample FA is reported against the same budget, whatever it is.

### 5.4 Sensitivity and robustness

The winner is re-run under softened (20/40) and hardened (30/60) boom
labels (thesis §7.1.2 port) and its per-year thresholds inspected for
stability; the bank scorer's perturbation curves confirm graceful
degradation with no knife-edge input.

### 5.5 The influential-users model (companion study)

The thesis's core question on the desk's own store: semi-supervised node
classification — labels = HIGH tier (composite ≥ 0.66, computed by the
production influence tracker), features = behavioural (call counts,
comment/post style, mean confidence) + structural (degree, weighted
degree, PageRank, audience), with a **leakage guard** excluding every
labelling-derived column. Models: random, feature-only MLP,
structure-only label propagation, and `sage_lite` — one GraphSAGE
mean-aggregation layer, the honest small-data version of the thesis's
winner (their own GNNs managed ~12% precision on 133 positives; model
appetite must not exceed label supply). Discipline identical to §5.3's
Layer-1 conventions plus the thesis's threshold rule (max precision
s.t. recall ≥ 0.05 on validation). Robustness: category ablation and
random/DICE graph perturbation. The store is live-only and seeds on the
first live pipeline run; notebook 05 is a **standing experiment** that
re-renders from the real store, with a pre-stated maturity criterion:
model outputs are decision-grade only at ≥130 labelled positives.

---

## 6. Results

### 6.1 Features

No single feature is a detector: AUROCs sit at 0.51–0.60 with cluster-CIs
frequently touching 0.5 — consistent with the project's standing finding
that attention features SELECT candidates while gates and combination do
the work. Two structural results: **E2 is anti-predictive at onset**
(AUROC 0.48 — sustained bullishness has not built yet when a rally
starts; direct evidence the onset task needs its own bank), and the two
speed features (`attention_accel`, `influx_speed`) are mutually
redundant in combination (drop-one deltas positive) — left for the
learners to arbitrate, which is what the tournament is for.

### 6.2 Tournament

| Task | Model | AP | AUROC | Captured | FA/instr-yr |
|---|---|---|---|---|---|
| Onset | **rules** | **0.098** | 0.565 | 29/125 | 0.35 |
| Onset | gbm | 0.098 | 0.609 | 26/125 | 0.37 |
| Onset | logreg | 0.089 | 0.595 | 28/125 | 0.36 |
| Onset | random | 0.072 (=prevalence) | 0.504 | 43/125 | **1.04** |
| Onset | mlp | 0.063 | 0.462 | 31/125 | 0.49 |
| Top | **rules** | **0.286** | 0.557 | 16/122 | 0.13 |
| Top | logreg | 0.265 | 0.592 | 16/122 | 0.15 |
| Top | mlp | 0.228 | 0.482 | 10/122 | 0.13 |
| Top | gbm | 0.213 | 0.525 | 12/122 | 0.16 |
| Top | random | 0.204 | 0.486 | 10/122 | 0.15 |

**Verdicts (mechanical application of the pre-stated criterion):**
ONSET — GBM ties rules on AP but the paired bootstrap CI of the
difference straddles zero ([−0.017, +0.019]) → **rules win by
parsimony**. TOP — **rules win outright**. The random row is the
cautionary display: 43 "captures" bought with 508 false alarms at
zero score quality — capture without precision is spraying. The MLP
below random reproduces the thesis's small-label warning. (The top-task
tournament evaluates the score construction; the production top
detector remains the validated incumbent implementation, which this
result confirms rather than replaces.)

### 6.3 Operating record (walk-forward, out of sample)

**Onset detector (ships):** 29/125 detectable starts captured (23.2%;
8.7% of all 333 episodes — both denominators, always), 21 late, 169
false alarms = 0.348/instrument-year vs the derived 0.23 budget — **50%
above budget, stated on the dashboard pane itself** as the current cost
of onset detection at this coverage. Median entry **17 days after the
trough**, median **66 days of rally still ahead** at the flag. Yearly
thresholds are stable (~0.89 throughout — the detector is not re-fitted
into a new personality each year).

**Top detector (unchanged):** ~23% of detectable peaks captured inside
[peak−30d, peak+1d], median lead 6 days, 0.23 FAs/instrument-year
(incumbent report). An earlier price-assisted variant captured 46% — the
documented price of the crowd-only claim.

**Calibration:** non-monotone in the low deciles (U-shaped), monotone and
steep in the top deciles where the gates live — annotated on the figure,
not smoothed away.

### 6.4 Sensitivity

Capture rate under softened / standard / hardened boom labels: 22.4% /
23.2% / 22.1% — flat; no cliff, no sign flip. The conclusion is not a
labelling artifact.

### 6.5 Trading translation — tested and rejected

Pre-stated criterion (recorded before any return was computed): n ≥ 30
judged alerts; mean 20-day forward return (the project's frozen
HOLD_DAYS) beats the same-instrument unconditional candidate-day
baseline in the expected direction; 90% instrument-cluster CI of the
difference excludes zero.

| Claim | n | Alert 20d fwd | Baseline | Diff | 90% CI | Verdict |
|---|---|---|---|---|---|---|
| ONSET→BUY | 215 | +0.34% | +0.98% | −0.63% | [−1.51%, +0.26%] | **REJECTED** |
| TOP→SELL | 67 | −0.21% | +1.34% | −1.55% | [−5.63%, +2.63%] | **REJECTED** |

The top→sell direction is right but the CI spans zero at n=67; the onset
→buy edge does not exist at the 20-day horizon. Recorded with the same
finality as the retired BUY/SELL engine. §7 explains why the system is
valuable *anyway* — and why claiming otherwise would be the fastest way
to lose a defense.

### 6.6 Influential-users model — and the units the tracker is read in

*Moved here from after §6.9 (it had drifted out of numerical order) and
rewritten 2026-07-27: the "results pend the store's first live seeding"
placeholder was stale, the store has been seeded and the study has
concluded.*

The Chan (2026) replication ran on the real store (5,071 authors with a
judged call; 107k accounts and 259k reply edges in the raw graph, of which
417,208 edges survive cleaning). Headline: `logit` on the 17-feature bank
with softened labels, **AP 0.0977 ± 0.0242 against a 0.0467 random floor,
AUROC 0.6681, permutation p = 0.005** over 200 shuffles. **Every graph rung
was rejected** under the paired 10-seed CI rule — `mixhop_lite` +0.0011,
CI [−0.0063, +0.0085]; `h2gcn` −0.0078, CI [−0.0130, −0.0026], i.e.
significantly *worse* — and the diagnosis is measured rather than asserted:
positive-class node homophily is **0.0948** against negative-class 0.9628
(sharper than the thesis's own 0.08 / 0.93), and DICE perturbation *raises*
AP from 0.1031 to 0.2063 at 50% corruption. For orientation, the thesis
benchmark was GraphSAGE AP 0.140 / AUROC 0.632, ≈ +61% AP over its random
floor; ours is +109% over ours, on a smaller positive class.

The decisive limitation is the one that determines what ships: on a
tenure/cohort split the model sits **at the random floor for unseen
authors** (lift −0.046). A model that can only rank authors it has already
seen is a research exhibit, not a ranker — so the dashboard's influence tab
is **information only** and ranks by the *measured* record.

**What the desk actually could not read, and what fixed it.** The tab's
first version printed bare sums, and the desk's verdict was literal: *"I
still don't get it."* Two quantities were at fault. A min-max-normalised
composite printed as "usefulness 0.987" reads as an accuracy and is not
one; and a Σ(influence × conviction) printed as "3.42" has no unit, so it
cannot be compared between two windows, two names, or two readers. Both
were replaced by **rescalings of themselves**, which is why no conclusion
moved:

- **`influence_index = 100 · composite / max(composite)`** — a positive
  rescaling (the real store's max is 0.9855), so every ranking is
  bit-identical to the composite's.
- **`backing_share = 100 · weighted / Σ weighted`** — share of the room's
  conviction, bounded 0–100, additive across names, and immune to the long
  tail. Measured on the 30-day cross-section (51 names): MSFT **26.10%**,
  ADBE 7.05, FICO 4.82, INTU 4.04, MELI 3.58, down to STRC 0.17.

The share is read against a **derived** reference line, not a chosen one:
`even_share(n) = 100 / n`, what each name would print if attention were
split evenly — 2.0% across those 51 names, so MSFT is running thirteen
times an even share.

**A rejected intermediate, recorded because it was rejected on data.** The
first attempt normalised by the **median name** in the window
(`weighted / median(weighted)`), on the argument that this mirrors the
euphoria detector's A1 convention, "2× its own 120d median". *That argument
is withdrawn.* A1 divides a name by **its own history**, which is a stable
reference; the median *name* in a cross-section is a ticker mentioned once
by one person. In the week to 2026-06-28, 163 names were mentioned and the
median one carried 0.24 of backing, so MSFT printed **141×**; weekly maxima
ran 141×, 41×, 2.7×, 14× and 26× on the 30-day view and **171×** on the
90-day view, and with an empty author list the median is exactly 0 so every
ratio came back **NaN**. A unit whose scale swings 50-fold between adjacent
weeks and undefines itself on an empty filter is not a unit. Full row in
Class 6b of the parameter register.

### 6.7 The desk-signal study: price + crowd, and the danger state

With the desk's 2026-07-24 decision to permit price in a SECOND signal
family, three pre-registered tests ran (notebooks 03/06). (i) The
**price-assisted END gate** (G2's boom thresholds as a live prerequisite)
passed every condition: capture 16→26 detectable tops on matched years,
AP 0.286→0.435, FAs 44→41, capture-gain CI [+3.5pp, +13pp]. (ii) The
**combined price+crowd alert bank** (adding chart-side Sornette
convexity, boom-magnitude and momentum ranks; raw and 7d-smoothed
triggers; LR/GBM challengers) did NOT clear its pre-stated criterion:
its 65% cliff-30 hit rate is indistinguishable from its own candidate
days' 62% baseline — because (iii) **the candidacy state itself is the
signal**: on days when the crowd is ≥2× its own normal AND price is in a
G2 boom (the *danger state*), a ≥10%-in-7-days drop begins within 30
days **62% of the time, versus 19% on ordinary days** (uplift 90%
cluster CI [+28pp, +50pp]). The product conclusion: the danger state is
the standing PM warning (an amber band on the terminal, derived entirely
from existing constants), and the END alerts time the peak within it
(median 8-day lead at the boom-gated operating point). The smoothed
trigger cut false alarms 31→24 at equal capture and lengthened trigger
runs (median 6→8 days), addressing the one-day-blip complaint at the
signal level.

### 6.8 The adopted desk configuration (GET IN / GET OUT)

The 2026-07-24 decision cycle closed with a production configuration,
chosen by a rule written before the deciding table was computed (NB06,
"the adopted desk configuration"; every variant runs through the
production code path in `analytics/euphoria_phases.py §6`, so notebook
and live system cannot drift). **GET OUT** (euphoria ending) is the
boom-gated end detector of §6.7 with its trigger on the 7d-smoothed
score: walk-forward capture 24/122 detectable tops, 39 false alarms
(0.195/instr-yr), AP 0.449 versus the raw gate's 0.435, median warning
8 days before the peak — the smoothing adopted because it lowers FAs
and raises AP while structurally removing one-day-blip alerts, at a
recorded cost of two captures. **GET IN** (euphoria starting) is the
tournament-winning onset rules with *phase-aware candidacy* — a day
already satisfying every END gate (A1 ∧ A2 ∧ A3-persistence, existing
constants only) is end-stage and may not host a "start" — plus the same
smoothing: adjacency (a START within one 21d cooldown before an END)
falls 20 → 2, LATE starts 21 → 10, false alarms 169 → 124, at a
recorded capture cost 29 → 20 of 125. This *overrules* the earlier
utility-rule rejection of the phase-aware variant (§Changelog row i) as
an explicit desk decision: the desk stated in three separate briefs
that a START landing on an END is the failure mode that destroys PM
trust, making adjacency the binding constraint, and the capture cost is
carried openly in NB06 and the parameter register rather than hidden.
Frozen live thresholds (budget rule on full prior years): GET IN 0.848,
GET OUT 0.630. The terminal now renders these two signals as explicit
GET IN / GET OUT banners with a window-adaptive scorecard (hit rate,
median lead, FAs, signal count recomputed for whatever window is
selected, judged by the same functions as the research record); the
crowd-only detectors remain unchanged as the thesis headline and the
research baseline.

### 6.9 The performance battery and the improvement campaign (NB07)

At the desk's request the shipped configuration was put through a
literature-grounded metric battery and a fresh round of pre-registered
challengers (notebook 07; every measure names its source). **The
battery**: precision–recall analysis with F1 and Matthews correlation
at the true operating point (Davis & Goadrich 2006; Chicco & Jurman
2020), a detection-delay-versus-false-alarm frontier in the
quickest-detection framing (Page 1954) with the shipped threshold
marked against the accepted FA budget, drift-adjusted cumulative
abnormal returns after each signal (MacKinlay 1997; one vote per
instrument, cluster CIs), a reliability diagram (Murphy & Winkler
1977), information coefficients (Grinold & Kahn 2000), and descriptive
overlay risk ratios (Sharpe; Sortino). One deliberately-kept "wrong"
number matters at a defense: the GET OUT score's pooled daily IC is
*positive* (~+0.11) — high scores sit mid-boom where momentum still
pays — which is precisely the evidence that this detector is an
episode-timing alarm, not a daily cross-sectional alpha, and that its
skill must be read at the event level (episode capture, post-alert CAR,
the NB06 10-day edge), never as a ranking signal.

**The improvement campaign** (adoption rule pre-stated: more captures,
FAs not higher, paired cluster-bootstrap gain CI excluding zero):
per-kind thresholds, a smoothing-window sweep w∈{1,3,5,7,10,14}, a
logistic-weighted score on the desk candidacy, and Western-Electric
run-rule triggers (k consecutive days). **Nothing displaced the shipped
configuration.** No smoothing window strictly dominates w=7 and no
run-rule k>1 dominates the single crossing (the smoothed score already
encodes persistence). Two watch items are recorded with an automatic
re-test at the next research pass: the logistic-weighted GET OUT (30 vs
24 captures, FA 37 vs 39, median warning 14.5d vs 8d — gain CI [−1,+11]
still touches zero, so parsimony holds) and per-kind thresholds (false
alarms halved at equal capture rate, but on unmatched walk-forward
years).

**The "buy-buy-buy / sell-sell-sell" hypothesis** was tested in both
forms. Per-instrument intensity (trigger days inside one trailing
cooldown): STRONG (≥2 days) alerts show a 52.9% cliff-30 rate vs 55.9%
for single-day alerts — uplift CI [−21pp, +22pp] straddles zero, so
repetition adds NO conviction; the cooldown-plus-smoothed-trigger
already absorbs it, and the exit criterion remains the GET OUT alert
itself (with the danger-state band as the standing warning, §6.7).
Cross-market breadth (share of names in GET OUT within a week) also
fails to separate the tradeable-theme basket's forward returns at 90%
(block-bootstrapped). The practical conclusion for the desk: a second
GET OUT inside the same episode is not "more sell" — it is the same
sell, and the first one already carried the information.

*(§6.6 used to sit here, after §6.9, still saying its results were
pending. It has been rewritten and moved into numerical order between
§6.5 and §6.7 — 2026-07-27.)*

---

## 7. Why This Is Useful for Trading

The trading-translation verdict (§6.5) rules out one specific,
mechanical claim: *"buy every onset alert / short every top alert and
earn excess return over 20 days."* That is the weakest possible use of
these detectors, and the honest number says it isn't there. What the
validated record does support is different and, for a desk, more
valuable:

**1. Exit and de-risk timing (the top detector).** Catalog busts average
−24% within 90 days (−30%+ for single names). A flag that fires a median
6 days *before* the peak, on roughly a quarter of detectable manias,
with ~1 false alarm per instrument every 4 years, is precisely the
moment to trim longs into strength, tighten stops, roll into options, or
stop adding. Being early on 1-in-4 of the worst drawdown events in the
retail universe — at near-zero monitoring cost — is risk management no
volatility model provides, because the input (crowd saturation) is
orthogonal to price-derived risk measures. Crucially, the desk does not
need the alert to be a profitable *short* (it is not, §6.5) for it to be
a profitable *exit*: avoiding the left tail of an existing long is a
different, easier trade than timing a short entry.

**2. Entry discipline (the top detector, inverted).** A name currently
carrying a top flag is a name where the crowd is at its own historical
saturation point. The validated contrarian literature (§2) says expected
forward returns after attention extremes are poor. "Do not initiate new
longs while the flag is up" costs nothing and dodges the entry pattern
retail flow punishes hardest — buying the crowd's peak.

**3. Radar and lead time (the onset detector).** A flagged onset has,
historically, a median 66 days of rally remaining. That is not a buy
signal (§6.5) — it is **watchlist lead time**: time to do fundamental
work on an unfamiliar name, size a theme, check borrow and options
liquidity before the crowd peaks, and pre-position the *monitoring*
(the top detector watches the names the onset detector surfaced). For a
macro desk, the onset list is also a positioning sensor: clusters of
onset flags across themes (as in 2021) are themselves a regime datum —
retail risk appetite turning on — usable as an input to broader
sentiment and flow views.

**4. Signal-quality filtering (the influence tracker).** Crowd chatter
is the noisiest input a desk touches. The tracker converts it into a
forward, volatility-judged track record per author, with the thesis's
"loud but wrong" finding operationalised: reply-graph hubs had 40%
accuracy vs 79% for quiet high-composite users — so the board never
ranks by audience size. When a proven-HIGH author turns bearish inside a
flagged episode, that is a qualitatively better datum than aggregate
sentiment, and the boom/bust column records exactly who called past
tops.

**5. The danger state — the PM warning in one glance.** The amber band
(crowd ≥2× its normal AND price in a confirmed boom) is the single most
actionable statistic this project produced: sharp drops (≥10% in a week)
begin within a month on 62% of danger-state days versus 19% of ordinary
days. A PM needs no model literacy to use it: amber on = de-risking
season; red line inside amber = the timing call.

**6. The economics of sparsity.** The whole system alerts a few names at
a time (live read at delivery: 3 starting, 3 ending), one alert per
episode per name, with its measured error rates printed on the pane. It
consumes minutes of attention per week, its claims are sized to its
evidence, and every number on the dashboard regenerates from current
data — nothing is hand-typed, nothing silently goes stale.

**Why the project is usable (not just publishable):** one command
(`python update_data.py`) runs ingestion → aggregation → all analytics →
prices; the dashboard runs pipelines from the sidebar with progress and
cancel; the committed data is text-free by construction (compliance);
research and production import the same modules with a drift-guard
assert (the deck's numbers *are* the dashboard's numbers); and the full
test suite (**84** invariants: crowd-only rule, no look-ahead, window
caps, cooldowns, text-free stores, share-unit algebra, chart-label
layout) plus a dashboard smoke harness gate every change.

---

## 8. Limitations (pre-answered)

1. **Coverage deserts.** 2023–25 archive coverage is thin to zero; the
   detectors are blind there, not validated there. Both denominators are
   always reported; the top capture-recovery lever (folding comments
   into the aggregates) is named future work.
2. **Trough truncation.** The trough is the measured 120-day low, so
   `run_days ≤ 120` by construction and slow multi-quarter rallies read
   as "starting" later than their narrative start.
3. **V-recoveries qualify.** A boom off a 120-day low admits post-crash
   rebounds (e.g. energy 2020) as episodes. Inherited knowingly from the
   validated boom rule; the crowd features must earn the mania-vs-
   recovery distinction, it is not assumed.
4. **Onset FA above budget.** 0.348 vs 0.23 per instrument-year — the
   stated cost of onset detection at current coverage, printed on the
   pane; not hidden, not excused.
5. **No standalone alpha claim.** §6.5. The detectors time risk; they do
   not, at the tested horizon, generate a mechanical trading profit.
6. **Calibration is not monotone** at low score deciles (annotated on
   the figure); the gates, not the raw level, carry the operating point.
7. **The influence-ML model is unproven on real data** until the store
   matures (pre-stated criterion: ≥130 labelled positives).
8. **Ground truth requires a confirmed bust** — euphoria that deflates
   slowly never enters the catalog, so capture rates speak only to the
   arcs a desk most needs flagged.

---

## 9. Future Work

Fold comments into the ABSTRACTED aggregates (the single largest
capture-recovery lever for the coverage deserts); LLM-based call
extraction to replace the VADER stance heuristic; graph features from
the influence store as euphoria inputs once the store matures;
`source_breadth` re-admission once the multi-source archive spans a full
episode cycle; re-run of notebook 05 on the seeded store and promotion
of this report's §6.6 from "pending" to results.

---

## 10. Conclusion

Two rule-based, crowd-only detectors — one for the start of retail
euphoria, one for its end — survive a pre-registered tournament against
learned challengers, post honest walk-forward records (23% of detectable
starts, median 17 days after the trough; ~23% of detectable tops, median
6 days early), fail an honestly-specified alpha test, and earn their
place as risk-timing and monitoring infrastructure rather than as a
black-box signal. Every fitted number in the system is either derived
from a validated constant or walk-forward-learned from the past only;
every negative result is documented with the same care as the positives.
That — more than any single capture rate — is the claim this project
defends.

---

## References

1. Barber, B. & Odean, T. (2008). *All That Glitters: The Effect of
   Attention and News on the Buying Behavior of Individual and
   Institutional Investors.* RFS.
2. Barber, B., Huang, X., Odean, T. & Schwarz, C. (2022). *Attention-
   Induced Trading and Returns: Evidence from Robinhood Users.* JFE.
3. Sornette, D. (2003). *Why Stock Markets Crash: Critical Events in
   Complex Financial Systems.* Princeton University Press.
4. *LPPLS applications to meme-stock bubbles*, arXiv:2110.06190.
5. Jame, R. et al. — r/WallStreetBets crowd-sentiment studies (signal vs
   noise amplification under coordinated attention).
6. Chan, J. J. J. (2026). *Informed Trading Decisions via Social Network
   Analysis: A Graph Learning Approach.* M.Eng thesis, University of
   Oxford. — methods source for the influence tracker (§4.5–4.6), the
   evaluation discipline (§6.2, §7.1–7.2), and the §5.5 companion study.
7. Hutto, C. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based
   Model for Sentiment Analysis of Social Media Text.* ICWSM.
8. MacKinlay, A.C. (1997). *Event Studies in Economics and Finance.*
   Journal of Economic Literature 35(1). — CAR methodology (§6.9).
9. Grinold, R. & Kahn, R. (2000). *Active Portfolio Management* (2nd
   ed.), McGraw-Hill. — information coefficient (§6.9).
10. Page, E.S. (1954). *Continuous Inspection Schemes.* Biometrika
    41(1/2); Poor, H.V. & Hadjiliadis, O. (2009). *Quickest Detection.*
    Cambridge UP. — the delay-vs-false-alarm frontier (§6.9).
11. Davis, J. & Goadrich, M. (2006). *The Relationship Between
    Precision-Recall and ROC Curves.* ICML; Saito, T. & Rehmsmeier, M.
    (2015). PLOS ONE 10(3). — PR analysis under imbalance (§6.9).
12. Chicco, D. & Jurman, G. (2020). *The advantages of the Matthews
    correlation coefficient (MCC) over F1 score and accuracy.* BMC
    Genomics 21:6. — operating-point MCC (§6.9).
13. Murphy, A.H. & Winkler, R.L. (1977). *Reliability of Subjective
    Probability Forecasts.* JRSS Series C 26(1). — reliability
    diagrams (§6.9).
14. Western Electric Co. (1956). *Statistical Quality Control
    Handbook.* — run-rule triggers, tested and not adopted (§6.9).

*(The full defense reference list, keyed to slides, lives in
`docs/DECISIONS.xlsx`, sheet 12.)*

---

## Appendix A — The fitted-numbers register

The complete inventory of every constant, its derivation and its status
is `docs/DECISIONS.xlsx` (15 sheets, defense-ordered). Summary of THIS
study's quantities: **zero new fitted numbers.** Onset window 45d =
existing FA horizon; onset gate multiplier 1 = parameter-free
above-median; FA budget 0.23 = incumbent's accepted rate; feature
windows 7/14/28/120/365 = existing ROLL, fade, E-window, hype-baseline
and rank-window constants; the only learned quantity remains each
detector's alert threshold, walk-forward from past years only.

## Appendix B — Reproducibility

```text
python -m pytest tests/ -v                         # 84 invariants
python -m analytics.run_analytics --what phases    # rebuild onset outputs
cd notebooks && jupyter nbconvert --to notebook \
    --execute --inplace 0*.ipynb                   # re-render the study
```

Every figure and number in notebooks 01–05 regenerates from current
data; the notebooks import the same `analytics/` modules the pipeline
runs (drift-guard assert in notebook 02).

## Appendix C — Changelog (the living-document record)

| Date | Update |
|---|---|
| 2026-07-27 (p) | **The influence tab re-cut into units that can be spoken aloud, and two defects found only by LOOKING at the rendered charts** (§6.6, moved into numerical order and rewritten; Class 6b of the parameter register). Desk verdict on the row-(n) tab was literal — *"I still don't get it"* — and the cause was measurable: every headline number was a bare sum with no unit. A min-max composite printed as "usefulness 0.987" reads as an accuracy and is not one; a Σ(influence × conviction) printed as "3.42" cannot be compared between two windows. Both are now **rescalings of themselves**, so no ranking and no conclusion changed: `influence_index = 100·composite/max` (real-store max 0.9855, a positive rescaling) and `backing_share = 100·weighted/Σweighted` — **share of the room's conviction**, bounded 0–100, additive, immune to the long tail (30-day cross-section, 51 names: MSFT **26.10%**, ADBE 7.05, FICO 4.82, INTU 4.04, MELI 3.58, tail STRC 0.17). Read against a **DERIVED** line, `even_share(n) = 100/n` — 2.0% over 51 names, so MSFT runs 13× an even share — replacing the old 1.0× median hairline. **One intermediate REJECTED on data, and the argument for it formally withdrawn**: `backing_ratio = weighted/median(weighted)` was justified as mirroring the euphoria detector's A1 convention ("2× its own 120d median"); A1 divides a name by *its own history*, a stable reference, whereas the median NAME in a cross-section is a ticker mentioned once by one person. In the week to 2026-06-28, 163 names were mentioned and the median carried 0.24 of backing, so MSFT printed **141×**; weekly maxima ran 141×/41×/2.7×/14×/26× (30d) and **171×** (90d), and with `authors=None` the median is exactly 0 so every ratio came back **NaN**. A unit that swings 50-fold between adjacent weeks and undefines itself on an empty filter is not a unit. **Time panels re-based on all 340 recorded voices, not the top-N panel**: the top-25 weeks held 124, 25, **1**, 23, 9 calls (the one-call week is one name at 100% by definition), against 733, 404, 23, 98, 155 calls across 14–171 names for the full pool — a time series whose population changes with a slider is not a time series. Thin weeks are **ENCODED, never gated**: tilt-marker area ∝ `n_calls`, so a thin week LOOKS thin and no week is dropped by a threshold. **Two defects visible only in rendered element screenshots** (Streamlit's `full_page=True` silently returns the viewport, which is why four charts had never actually been looked at): (i) the bubble chart printed MSFT at **29.5%** while KPI 4 one row above printed **26.10%** for the same name in the same window, because the figure received `dig.head(30)` and denominated over 30 names instead of the window's 51 — one quantity cannot have two values on one screen; fixed by passing the FULL digest plus `top_n` and fixing the denominator (and `even_share`) **before** truncation, in both `fig_influence_bubbles` and `crowding_history`; (ii) labels printed through each other (INTU 4.04% vs MELI 3.58% sit 0.46pp ≈ 9px apart at 520px; a 10pt label needs 13px), fixed by `_thin_labels`, a greedy geometric de-collider, plus nudged annotations on the weekly chart where MSTR and ADBE both ended near 0.3% as one smear. `_thin_labels` is declared **the only pixel-level rule on this project** and is fenced as such: its gaps are derived from plot geometry (13px = one 10pt line box at 1.3 leading; 30px = a four-character ticker's width), it suppresses only when boxes overlap in BOTH directions, it never decides which names *matter* (every point is still drawn, still hovers, still appears in the exact-numbers table), and five unit tests pin it. Removals recorded rather than deleted: **`fig_consensus`** (it plotted the bubble chart's x-axis with the y-axis folded into bar opacity — its own docstring said so; its encoding decision is kept as a REJECTED register row), the **HIGH-tier cut display**, and the **per-author hit rate**, which the desk asked to drop — the stored `hit_rate` column is unchanged and a schema test asserts it. KPI renamed to "calls vs last week". Ingestion cadence answered from the ledger, not typed: `--dry-run` prints `comment budget: 449 pages (7.5 min) = ceiling 10.0 min − other stages 2.5 min`. **9 new tests (75 → 84 passed)**; AppTest clean, all five sidebar buttons 0 exceptions. One self-caught bug worth recording: my own de-collider's first `sort(reverse=True)` on `(y, x, i)` made the highest row index win ties, handing the label to the *least*-backed name — fixed to `key=lambda t: (-t[0], t[2])` so ties resolve toward the better-backed name. |
| 2026-07-27 (o) | **Comments are BUDGETED, not optional — row (b)'s decoupling reversed** (§3.1b, §3.1b-i). An influence board that rescores month-old comments is not a live board, so `update_data.py` now fetches comments on every run; `--with-comments` is accepted and ignored, `--skip-comments` is the new opt-out. The reason it could not simply be switched back on is measured, not assumed: intersecting comment-call `rec_id`s with `reply_edges` gives 12,010 / 417,208 = **2.879% call rate among comments**, which against two independent months (403 and 414 comment-calls/day) implies **~14,000 comments/day ≈ 140 pages/day** across the 17-sub panel — a 7-day gap costs **16.3–16.8 min** at the API's committed 1 req/s, over the desk's 10-minute ceiling. (Caveat recorded: the call rate is measured on the edge-covered subpopulation, 48% of comment-calls.) What ships is a **page allowance** computed from two self-measuring EWMA ledgers — `pipeline_stage_times.json` (what the non-fetch stages actually cost this machine) and `reddit_comments_cost.json` (pages/day per subreddit) — allocated proportionally to each subreddit's owed days with a one-page anti-starvation floor. EWMA α is DERIVED, not typed: `N = round(28 / 3.2) = 9` runs → `α = 2/(N+1) = 0.2`. Hitting a cap sets `completed = False`, so the watermark does **not** advance and the next run resumes exactly there — deferral, never data loss: the board is never silently partial, only ever less fresh. Four alternatives REJECTED with reasons: a wall-clock stopwatch (makes data collected a function of network luck, so no two runs are comparable), parallel workers (W workers × 1s pauses = W req/s, breaking the politeness contract the project accepted when it chose a free public API), uniform window narrowing (penalises quiet subreddits to subsidise loud ones), per-subreddit yield ranking (the store has no `subreddit` column). The one legitimate speedup was removing dead time: a `Pacer` sleeping the *remainder* of the second rather than a flat second after each round-trip — **28% dead time removed at an unchanged request rate** (261ms vs 360ms over 6 requests). `PIPELINE_BUDGET_S = 600` is the single DESK-CHOSEN number here. Measured on real data: `analytics.run_analytics` 73.3s (reproduced twice), fold 0.44s, coverage 0.31s, hydrate 0.012s → residual allowance **465 pages ≈ 7.8 min ≈ 3.3 days** of panel volume, so the derived cadence is **3.32 days** — the desk chose **~2×/week**, which the ledger independently confirms. Both ledgers are gitignored (they measure ONE machine's speed). `update_comments.py` repositioned as the UNBUDGETED runner for backfills and long-gap catch-up, and its hand-written time ranges ("roughly 10–25 minutes") replaced by a ledger-computed estimate — those were typed-in numbers, which this project does not keep. Operational change only: no analytic input changed, and the euphoria detector still never reads comments. 75 tests pass; verified end-to-end with a real pipeline run (ledger written, allowance self-corrected 449 → 465) and two no-network stub tests (cap/watermark/ledger semantics; all three `fetch_all` hand-off paths). |
| 2026-07-27 (n) | **Influence tracker: Chan (2026) replicated end-to-end, and its negative result shipped honestly (NB05).** The store is now real (5,071 authors with a judged call; 107k accounts, 259k reply edges in the raw graph), so the harness built in row (a) was run for the first time. Ported from the thesis: §4.6 composite scoring with Bayesian shrinkage, §5 network and label analysis, §6.1 eight architectures (GAT deliberately NOT ported — attention must *learn* edge weights and ~250 positives cannot support it; `mixhop_lite` is the named small-data stand-in), §7.1.2 labelling sensitivity, §7.1.3 misclassification, §7.2 ablation + DICE/random perturbation, §8 limitations. All graph layers are pure numpy/scipy — hand-rolled multi-level Louvain, Brandes sampled betweenness, k-core, Fruchterman-Reingold; **no networkx anywhere**. Headline: `logit` on the 17-feature bank, softened labels, **AP 0.0977 ± 0.0242 vs a 0.0467 random floor, AUROC 0.6681, permutation p = 0.005** (200 shuffles). **Every graph rung rejected** under the paired 10-seed CI rule (mixhop +0.0011 CI [−0.0063, +0.0085]; h2gcn −0.0078 CI [−0.0130, −0.0026], i.e. significantly worse), and the diagnosis is measured, not asserted: positive-class node homophily **0.0948** vs negative 0.9628 (sharper than Chan's 0.08/0.93), and DICE perturbation *raises* AP 0.1031 → 0.2063 at 50% corruption. Two disciplines applied beyond the thesis: `mean_conf`/`stance_sd` refused as arithmetic factors of their own target with the **price of that honesty recorded** (+0.0983 AP, CI [+0.0834, +0.1132], 10/10 seeds), and Bonferroni within the round (6 candidates → conf 0.99167) which turned the one nominally-significant bank change into **adopted: null**. Decisive limitation: on a tenure/cohort split the model sits **at the random floor for unseen authors** (lift −0.046) — so the dashboard's new INFORMATION-ONLY influence tab ranks by the **measured** record and files the model as a research exhibit. Tab ships a leaderboard, an influence-weighted "what they are suggesting" view (fade encodes weight of evidence, because bar length saturates at three agreeing voices), a k-core backbone / ego influence map over the scored pool (100% colour coverage, 0.9s vs 39.5s unrestricted), and a "why there is no model on this tab" panel quoting the four measured numbers. Nothing on the tab touches the euphoria level or the GET IN / GET OUT alerts. 17 new tests (**72 total**), AppTest clean across all six new branches, notebook re-executed 0 errors / 17 figures, `docs/research/nb05_influence.json` written. One latent bug fixed in passing: `stratified_split` shuffled a read-only view of the caller's index. |
| 2026-07-24 (m) | **Performance battery + improvement campaign (NB07)** — §6.9. Literature-grounded metrics added (PR/F1/MCC at the operating point, delay-vs-FA frontier, MacKinlay CAR, reliability diagram, Grinold-Kahn IC, overlay Sharpe/Sortino); the positive daily IC recorded as proof the signal is an episode alarm, not daily alpha. Four pre-registered challengers (per-kind thresholds, smoothing sweep, logistic weighting, run-rule triggers): **nothing displaced the shipped configuration**; watch items B3 (logreg: 30 vs 24 captures, gain CI touches zero) and B1 (per-kind: FA halved, unmatched years) recorded with auto re-test at next research pass. Clustering-as-conviction tested both ways (per-instrument intensity, cross-market breadth): **no separation** — the exit criterion remains the GET OUT alert itself; repetition adds no information. References section extended. |
| 2026-07-24 (l) | **DESK CONFIGURATION adopted and productionised (GET IN / GET OUT)** — §6.8. GET OUT = boom-gated END with 7d-smoothed trigger (cap 24/122, FA 39, AP 0.449, median warning 8d); GET IN = phase-aware onset + smoothing (adjacency 20→2, LATE 21→10, FA 169→124, capture cost 29→20 RECORDED — a desk decision overruling row (i)'s utility-rule rejection, per the thrice-stated adjacency priority). Selection rule pre-stated in NB06; drift guard asserts notebook == production record. New store `euphoria_desk.parquet` + frozen thresholds (GET IN 0.848 / GET OUT 0.630) in `euphoria_desk_report.json`; research/live split honoured (desk thresholds refreeze only on `--research`/year rollover). Terminal: explicit GET IN (green) / GET OUT (red) banners, GET IN/GET OUT chart labels, and a window-adaptive scorecard (hit rate, median lead, FAs, signals - recomputed for the selected window with the research judges; PENDING convention respected). 5 new tests (55 total). |
| 2026-07-24 (k) | **Desk-signal study concluded (NB06)**: combined price+crowd ALERT bank rejected under its pre-stated cliff criterion (65% vs its own 62% candidate-day baseline, CI includes 0) - but the candidacy STATE validated decisively as the drop-warning: cliff-30 62% on danger-state days vs 19% ordinary, CI [+28pp,+50pp]. DANGER STATE (A1 2x + G2 boom, existing constants only) shipped as an amber band on the terminal price panels; smoothed trigger evidence recorded (FA 31->24 at equal capture, median run 6->8d). Event-study reading guide: after an END alert, flat/down = success; the +50d mean spike is an outlier artifact (median path flat). |
| 2026-07-24 (j) | **Price-assisted END gate: commissioned, tested, PASSED.** Gate = G2's own boom thresholds as a live prerequisite (trailing prices only). On matched test years: capture 16->26 detectable tops (+62%), AP 0.286->0.435, FAs 44->41, utility -28->-15; dose-response confirmed via a half-strength gate; capture-gain 90% cluster CI [+0.035,+0.130] excludes zero. All pre-stated adoption conditions met -> OFFERED as a clearly-labelled SECOND signal (claim: crowd + chart-confirmed boom); the crowd-only detector remains the headline claim. Not yet wired to the dashboard - desk decision pending. |
| 2026-07-24 (i) | **Phase-aware onset variant tested and REJECTED** under a pre-stated rule (halves start/end adjacency 19->8 and LATE 21->12, but -5 captures, no utility gain) - recorded in NB03; the start/end adjacency READABILITY is solved on the terminal by **episode-span shading** (START->next END shaded on the price panel; a tight pair reads as a short violent episode, zero capture cost). **NB06 validity section** added: forest plot of edges with 90% cluster CIs, event-study with bootstrap bands, hit rates vs drift-adjusted baseline with CIs, and END-10d edge by year (regime stability). |
| 2026-07-24 (h) | **Parameter register** (`docs/PARAMETER_REGISTER.md`, surfaced on both EUPHORIA tabs): every number classified LEARNED / DERIVED / CONVENTION (ablation-accountable) / GROUND-TRUTH / DESK DECISION with its reason. **NB03 threshold exhibit**: the training curves behind each threshold, both selection rules drawn - and an honest finding recorded: the legacy utility rule saturates toward the conservative grid edge in FA-rich regimes (the incumbent's '85' is partly its grid cap), while the budget rule selects an interior point; the budget rule is the selection used for the onset detector. **Dashboard fix**: add_vline epoch-ms workaround for the plotly Timestamp crash. |
| 2026-07-24 (g) | **NB06 signal-efficacy report** added (descriptive; NB04 stays confirmatory): forward moves at desk-requested 3/10/21/84d horizons with same-instrument baselines and cluster CIs, event-study paths, overlay PnL view, full per-alert and per-name tables. Headline: the END signal's 10d edge is the one CI excluding zero (-2.6% vs +0.6% baseline; down-hit 57% vs 45% drift-adjusted baseline) - the risk-timing claim quantified; START confirms lead time, no buy edge. **Dashboard**: START/END labels on alert lines; per-alert plain-English 'why did this fire' expander (component values now persisted in both stores). |
| 2026-07-24 (f) | **Episode coherence (asymmetric, by measurement)**: on the terminal, a START within one 21d cooldown after an END is suppressed as contradictory; an END after a START is never suppressed - the symmetric rule was tested and REJECTED (cost half the top captures, 17->9, for 8 fewer FAs; onset direction: -2 captures, -7 FAs - adopted). **Chart clarity**: level shown 7d-smoothed (ROLL) with the frozen threshold line and alert-eligible (A1) stretches highlighted; per-name lookup added. Levels store now carries hype_ok. 4 new tests (50 total). |
| 2026-07-24 (e) | **Terminal display policy**: EUPHORIA tabs list every instrument alerted in the selected window, newest first; single names show STARTING only past the full A1 2x hype bar (existing constant - display policy, detector unchanged). **Notebooks**: plain-English glossary added (NB02/NB04); per-theme and per-ticker hit-rate tables vs the official ETF list added to NB04 (with the small-denominator caveat stated). |
| 2026-07-24 (d) | **Dashboard restructure**: EUPHORIA split into Themes / Singles tabs; state (STARTING blue / ENDING red) drawn on the charts with a current-state strip; validation tables (walk-forward, ablation, ML) removed from the terminal - conclusions caption only, evidence lives in notebooks 01-04 + DECISIONS.xlsx. |
| 2026-07-24 (c) | **Research/live split**: live pulls now score at frozen thresholds (euphoria stage minutes→~20s); full validation (walk-forward, ablation, ML, threshold selection) runs on explicit `--research`, auto-triggering on year rollover / missing report, forced by `--full`. Rationale: walk-forward thresholds train on strictly earlier years, so intra-year recompute is provably a no-op. Phases' live threshold now selects on full prior years only (intra-year stable by construction; 0.891→0.895 under the new convention, scorecard unchanged). 2 new tests (46 total). |
| 2026-07-24 (b) | **Dynamic subreddit panel** shipped (crowd-referral discovery, monthly, watermarked; qualification = the A0 floor reused; same-ruler finance screen; cap 1 add/review; committed audit manifest + local by-subreddit aggregate for panel-step re-cuts) — §3 amended. **Comments decoupled** from the daily pipeline into `update_comments.py` (watermark-aware time estimate, resumable) + dashboard button; operational change only. 4 new tests (44 total), AppTest clean. |
| 2026-07-24 | Initial report: episode ground truth (NB01, 333 episodes); feature battery incl. `source_breadth` rejection (NB02); tournament — rules win onset by parsimony over tied GBM, top outright, MLP below random (NB03); operating record 23.2% onset capture / 17d after trough / 66d ahead / 0.348 FA-iy; label sensitivity flat; trading translation REJECTED both directions (NB04); onset detector productionised (`phases` stage + Start/End radar panes); influence-ML harness built and fixture-validated, real run pending store seeding (NB05); 7 new test invariants (40 total), AppTest clean. |

*Update rule: every session that changes a rule, adds evidence, or
closes an open question appends one dated row here and amends the
affected section in place. The report is regenerated-truth, like the
dashboard: if a number can drift, its source is named.*
