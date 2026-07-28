# Detecting the Start and End of Retail Euphoria from Crowd Data Alone: A Walk-Forward Study on Nine Years of Social-Media Attention

**Alex Brown — GIP 2026 Project — MAARS Global Macro**
*RetailRadar research report — LIVING DOCUMENT (see Changelog, Appendix C). Last updated 2026-07-28.*

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

**Why the bank is hand-specified and not searched.** An obvious
alternative was to let a search procedure assemble the feature bank —
forward/backward stepwise selection, an L1 path, or a genetic search over
transformations of the daily aggregates. That was **considered and
refused**, and the reason is the sample, not a preference for handwork:
the label is 59 instruments, and the plausibly-independent unit of
evidence is the *instrument*, not the day (§6.2). A search that scores
candidates on 183k daily rows is choosing among thousands of options
using an effective sample of 59, which is precisely the regime in which
selection noise dominates and the winner does not survive out of sample.
Every feature above is instead *inherited*: its window is a constant
already validated elsewhere in the project (ROLL=7, the A1 120d
baseline, the fade's 14d, E5's LPPLS window), so no feature adds a
degree of freedom that a test year has not already seen. The cost of
this discipline is stated plainly in §6.1 — the hand-specified bank's
per-feature AUROCs are only 0.51–0.60, and a search would very likely
have posted better *in-sample* numbers. That is the trade being made
deliberately, and it is the same trade the thesis makes in §6.2.3.

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

### 6.10 The euphoria gauge — one dial per instrument, and where its red zone comes from

The desk's request was for "a very clear speedometer thing for each
graph ... for each theme / ticker", showing the current euphoria
percentage, a red zone, and the change. The methodological problem is
that a dial is the single most compressive exhibit on this project: it
reduces a whole instrument to one number and a colour, so a reader will
act on it. Three constructions were therefore forced.

**The needle is not a new quantity.** It is `lvl_raw.rolling(ROLL).mean()`
read at its last day — the identical object the chart's lower panel
already draws. The dial is a readout of the curve beneath it. If the two
could ever disagree the dashboard would be telling two stories about one
name, which is the fastest way to lose a PM; a unit test asserts the
equality. The delta reference is `ROLL` = 7 days back, the same window the
curve is smoothed over, so "the change" is a change in the plotted
quantity rather than in daily noise.

**The red edge is not a new number either.** It is read at import time out
of `euphoria_report.json`'s `thresholds` dict — the level the walk-forward
already selected for the END alert, 85 in every test year 2018–2026. A
gauge that chose its own red edge would be a second, softer threshold
competing with the detector. The notebook asserts the walk-forward agreed
across years, on the grounds that if it had not, a single red edge would
be a fiction and the honest exhibit would be the year's own level.

**One new number, and it had to be earned: the amber edge, 76.** The
outcome graded is the desk's own horizon, unchanged from §6.7: does a fall
of 10% or more over 7 days *start* within the next 30 days. Over 183,394
name-days on 59 instruments (2017-06-29 → 2026-06-15) the unconditional
rate is **23.3%**.

*The obvious test was the wrong test, and this is worth recording.* Comparing
a 95% CI for `P(drop | level>=L)` against a 95% CI for the base rate found
nothing significant at any cut from 40 to 95. That is the
overlapping-confidence-interval fallacy rather than a null result: with 59
instruments both intervals are wide, and two overlapping intervals do not
imply the difference is zero. Bootstrapping the **difference** — state
minus not-state, on the *same* resampled instruments — cancels the shared
instrument-level noise, and recovers a significant effect at every cut
from 76 upward. The resampling unit is the **instrument**, not the day,
because adjacent days on one name are the same episode; a day-level
bootstrap would treat 4,000 days of one mania as 4,000 independent facts
and report an interval several times too tight.

The selection rule was stated before the grid was run: take the lowest cut
on a 2-point grid from 68 whose 95% paired lower bound excludes zero under
**all five** seeds. Cut 74 flips sign across seeds (−0.0004, +0.0009,
−0.0012, −0.0007, +0.0004); cut 76 does not (+0.0045, +0.0060, +0.0049,
+0.0034, +0.0046). A cut that changes sign with the random seed is not a
parameter, so 74 was refused and **76** adopted.

What the bands are worth, measured, and printed on the dial itself:

| State | days | P(>=10% fall starts within 30d) | vs not-in-state | 95% paired CI |
|---|---|---|---|---|
| any day (base rate) | 183,394 | **23.3%** | — | — |
| level >= 76 (amber) | 11,235 | **26.9%** | +3.9pp | [+0.6, +7.4] |
| level >= 85 (red) | 4,203 | **30.4%** | +7.3pp | [+2.1, +12.8] |
| danger state alone (A1 2x AND G2 boom) | 3,819 | **53.8%** | +31.2pp | [+22.2, +40.1] |
| level >= 85 **AND** danger state | 454 | **71.4%** | +48.2pp | [+34.3, +59.5] |

**The dial's own limitation is on its face.** The level alone at the red
edge is 30% against a 23% base rate — about **1.3x**. The level *together
with* the danger state is 71% against 23% — about **3.1x**. So the caption
under every dial quotes the band the needle is actually in rather than a
generic legend, and a unit test pins `P(red AND danger) > P(red)` so the
exhibit can never be re-worded into implying the needle is sufficient on
its own.

**And the dial is a STATE, never an instruction.** GET IN and GET OUT come
from the walk-forward detector and can fire with the needle anywhere; a
unit test asserts the words "get in" and "get out" can never appear in a
band label. Nine tests in total fence the gauge: that neither edge is a
literal in `dashboard.py` (the numbers live in
`docs/research/gauge_zones.json`, which notebook 06 writes), that the red
edge equals the frozen walk-forward level, that the needle equals the
plotted curve's endpoint, that rising euphoria is never painted green
(plotly's default would invert the meaning of the arrow), that the bands
tile [0,100] with no dead gap, and that with no measured evidence the dial
declines to exist rather than guessing its own bands.

SO WHAT: the PM gets the compressed read they asked for, and every element
of it — needle, red edge, band colour, the percentage in the sentence
underneath — is either a quantity the project already defended or a single
number selected by a rule stated in advance and stable across seeds.

### 6.11 Three legibility defects, and why only one of them was a display problem

Three desk reports arrived together on 2026-07-28. They are recorded here
rather than as housekeeping because the first turned out to be a
**self-contradiction inside the chart**, and because the fix for the second
is a case study in the project's standing refusal to move a threshold to
make an exhibit look better.

**(i) The chart was quoting a threshold that had not fired anything.** The
desk wrote: *"its quite unclear to see WHEN is the actual change / or get
out flag ... its like flat and then suddently a get out flag."* The old
lower panel drew one dotted line labelled "signal level" at the
level-detector's walk-forward threshold, 85, and plotted the euphoria
**level** against it. But when the desk store is present — the normal case
since §6.8 — the flags on screen are not produced by the level at all. They
are produced by the desk score crossing its own frozen threshold. The two
disagree constantly: **measured over all 95 GET OUT alerts in the store, the
plotted level sat BELOW the drawn 85 line on 79 of them (83%), with a median
plotted level at a GET OUT of 74.8.** The chart was showing a curve
comfortably under the line it said mattered, and then a flag appearing
anyway. The desk was reading the chart correctly; the chart was wrong.

The correction is a rule rather than a tweak: *draw the threshold that
gated the flags being drawn, and plot the series that actually crossed it.*
The GET OUT / GET IN scores are overlaid at their frozen desk thresholds,
rescaled ×100 onto the panel's existing 0–100 axis so the panel keeps one
axis. Only alert kinds that actually fired inside the window are drawn, so
every element on the panel explains a flag the reader can see. The score is
drawn as `lines+markers` with gaps left open, because it is **sparse by
construction** — `out_score` exists on 2.5% of name-days, in runs as short
as one day — and the gaps are information: no score means "not judgeable
here". The 85 line survives only on the no-desk-store fallback path, where
the level genuinely is the decider. **No threshold moved.**

The desk also asked for "a clear peak or something", and got one: the
window maximum of the display curve is marked and dated. It is a label on a
value already plotted — the test for whether a display addition is safe is
that removing it changes nothing about the signal, and removing this
changes nothing.

**(ii) The gauge "always shows calm", and it was right to.** The dial was
not stuck. **Measured over the default window (2026-01-01 → latest), 50 of
59 instruments read calm at the last day while 17 of those same names
touched the red zone somewhere inside the window.** Both facts hold at once
because the page is ordered by *most recent signal*, so a name earns its
place with an episode that may have peaked months ago, while the needle —
correctly — reports today. A dial that answers "how hot is it now?" on a
name selected for "it was hot recently" reads calm almost always and looks
broken while being right.

The tempting fix is to soften the amber edge until the dials look busier.
That would be exactly the arbitrary-threshold move §6.10 spent a page
earning the right not to make, and it would degrade a number that survived
five seeds. The adopted fix instead makes the dial answer *both* questions:
the big needle stays today, and the window's high-water mark is drawn
behind it as one dated line, so a calm reading carries its own explanation
— "calm now, peaked 99 in red on 27 Jun". **No edge moved and no number was
invented.** The same report asked that the dial stop explaining itself in
full on every chart; the standing caveat and the measured band percentages
moved into an info hover, which is a change of *placement, not of
evidence* — a page of six names had been carrying the same ninety words six
times, burying the reading that mattered in its own footnotes.

**(iii) Handle masking, and why it is a convention rather than a result.**
The desk asked to "censor the innapropriate stuff with `**`". There is no
ground truth for "offensive", so no bootstrap and no walk-forward can make
a word list evidence-backed, and the register says so: this is a Class 3
**convention**. What *is* evidence, and what is registered, is the measured
behaviour of the list over the real corpus of 12,528 handles. A naive
one-list substring scan flags 345 and is dominated by false positives —
`AfraidAnalyst`, `Valuable-Analyst-464`, `MeridianAllocation`, seven
`Grapefruit` handles, `SatoshiTrails`. Three measured corrections followed:
a match must lie inside a single token (which alone resolves `SatoshiTrails`
and `MeridianAllocation` at the cost of no word list at all); six stems were
demoted to whole-token matching on counted innocent-versus-genuine hits; and
six mild words were dropped entirely. Two candidate promotions were adopted
on 10/10 and 3/3 genuine hits and a third was **rejected** at a 50% error
rate. The final state is **97 of 12,528 masked (0.774%)** with one residual
false positive and, critically, **12,528 unique handles mapping to 12,528
unique censored strings** — zero collisions, which is what makes span-level
masking safe on a plotly category axis where duplicate labels merge into
one bar.

Two invariants make this defensible rather than cosmetic. Masking happens
**only at the display layer**: `author` is the join key shared by
`author_scores`, `calls` and `reply_edges`, so rewriting it would collide
distinct authors and destroy the ability to re-judge anyone against new
prices — a test asserts no mask ever reaches the parquet. And the direction
of error is deliberately **under-mask**: a missed handle is one embarrassing
name on a board everyone knows is scraped from Reddit, whereas an
over-masked handle corrupts identity for every reader. The known misses are
named in the register, as is the standing limitation that the list is
**English only**.

SO WHAT: two of the three reports were legibility, and one was a real
defect that had been on screen since the desk configuration shipped. The
pattern worth keeping is that in all three cases the fix was to make the
exhibit say what the method already does — never to adjust the method so the
exhibit reads better.

### 6.12 Can influence convergence predict? A pre-registered null, and the exhibit that shipped instead

The desk asked: *"lets try and use the follower monitoring (influence) for
some sort of bullish / euphoria indicator? anyway to make that clearer?
perhaps if we have lots of influential accounts converging on a theme?"*

That is two questions wearing one sentence, and treating them as one is how
a project ends up shipping a chart that quietly implies a forecast it never
earned. **"Use it as an indicator" is a predictive claim** and had to clear
the same bar as everything else here. **"Make it clearer" is a legibility
request** and did not. The first was tested and rejected; the second
shipped. What follows is the separation, kept explicit because the two
answers live on the same screen.

**The design was fixed before any number was seen.** Three candidates, all
built from quantities the desk has already accepted, so nothing new had to
be justified: **C1 breadth**, the influence-weighted count of distinct
voices on a name in the trailing seven days; **C2 convergence**, breadth ×
agreement; **C3 backing share**. Agreement is
`|Σ w·s| / Σ w·|s| ∈ [0,1]` — the same arithmetic as the accepted
`consensus`, and it encodes the distinction the desk's own question turns
on: one voice, or many voices saying the same thing, scores 1, while a room
split down the middle scores 0. **Many influential accounts that disagree
is not convergence.**

The author weight is where a threshold would normally sneak in, and it was
refused. The obvious cut — take the HIGH tier — is not available on this
store as a population: it holds **25 HIGH authors against 12,503 low**, and
only **234 of 27,881** live calls come from a HIGH author. So authors enter
continuously at `influence_index / 100 ∈ [0,1]`: a nobody contributes ~0,
the strongest measured record contributes 1, and **no boundary is invented
anywhere**. The outcome is the accepted gauge outcome copied verbatim (a
**>10% fall inside a week**, occurring any time in the next **30 days**),
which is what makes the comparison below like-for-like. Significance is a
paired bootstrap resampling the **instrument** rather than the day (name-days
inside one ticker are not independent), 5 seeds × 300 reps, Bonferroni over
the three declared tests → confidence **0.98333**, and adoption requires the
**worst** seed's lower bound to clear zero.

One further data constraint shaped the window rather than being worked
around: the call history is **two disjoint blocks**. 2021-06 holds 5,521
calls spread over only **two distinct days** — an archive snapshot, not
history — then a five-year hole, then the live block from 2026-04. A
trailing-window feature cannot be computed across the hole, so the test runs
on the live block only.

**Result: adopted, none.** On 15,615 name-days across 233 instruments
(2026-04-08 → 2026-06-15, base rate 0.539), the top-decile-versus-rest
differences are C1 **−0.0291** (worst-seed lower bound −0.1701), C2
**−0.0153** (−0.1513), C3 **−0.0103** (−0.1706).

**A null is only worth reporting if the harness could have found something,
so the same harness was given a positive control.** The accepted euphoria
level, on the identical days through the identical code, separates cleanly:
top decile (cut 78.91) **0.846 vs 0.414**, difference **+0.4321**, lower
bound +0.0623; at the frozen RED edge of 85, **0.925 vs 0.428**, difference
**+0.4970**, lower bound **+0.2515**. The window is not too short. The
method is not broken. The feature is the thing that failed.

**The decisive exhibit is the like-for-like test.** Restricted to the
control's own 1,552 name-days across 23 instruments, with an identical
156-day state size and a base rate of 0.457, all three candidates are
**wrong-signed**: C1 0.237 vs 0.482 (**−0.2449**), C2 0.250 vs 0.481
(**−0.2307**), C3 0.282 vs 0.477 (**−0.1950**). Same days, same names, same
state size — the only thing that differs is the feature. These are not
underpowered measurements of a weak effect; they point the other way.

**And the wrong sign has a mechanism, which is what turns a rejection into
a finding.** The top-decile-breadth names by state-days are MSFT 64, NVDA
59, TSLA 56, AMZN 46, RDDT 45, INTC 44, AAPL 44, SNDK 38, GOOGL 37, MSTR 37
— the influence board converges on the **most-discussed liquid mega-caps**,
and those cliff less often than the small-cap tail. The between-name split
settles it: "ever top-decile" **0.534** (n = 6,576) against "never"
**0.542** (n = 9,039), nearly flat. The effect is **cross-sectional — which
names — and not temporal — when**, and a quantity that tells you which
names are big is not a timing indicator. Per the standing rule, no code
from this test survives anywhere in the repo; the register (Class 6c) is
the trace.

**What shipped is the other half of the question.** The tab already had the
right exhibit — the bubble chart placing each name by how one-sided the
panel is (x) against its share of the room's conviction (y), with the
CROWDED LONG / CROWDED SHORT / GENUINE DISAGREEMENT quadrants — but it was
**ticker-only**, and the desk asked about **themes**. A names/themes toggle
now regroups the same chart, and the roll-up reuses the accepted consensus
and backing arithmetic through **one shared `_digest_frame(c, key)`**: a
second copy written for themes would be a second place for those formulas
to be wrong, and the two views sit side by side on one control where any
disagreement would be visible to a PM and impossible to explain. Themes use
the same membership as the euphoria Themes tab, so a theme means one thing
across the whole application.

Two consequences of the grain are stated on the control rather than hidden.
A ticker in several themes **counts in every one** — NVDA is semiconductors
and ai and ai_megacap, and a PM asking "is the panel crowded into AI" must
see the NVDA call — which means theme shares are shares of the
**theme-mapped room**, a different denominator from the ticker room, so the
two views are not expected to agree name-for-name. And calls on tickers in
no theme are **dropped rather than bucketed as "other"**: "other" is not
something a desk can position in, and at **58.5% of live calls** it would
be the largest bar on the chart purely by being a residue.

The view earns its place by making a distinction the ticker view could not.
Over 90 days, **AI megacap sits at 26.6% of the room's conviction with
consensus +0.807** — a genuine convergence — while **semiconductors sits at
10.2% with consensus +0.222**, which is visibly an *argument* rather than a
crowd. That is exactly the difference the desk asked to be made clear, and
it is invisible one ticker at a time.

SO WHAT: the honest answer to "can we use influence convergence as a
bullish indicator" is no, and the evidence for that no is stronger than the
evidence usually offered for a yes — a pre-registered design, a positive
control that fires, a like-for-like test that inverts, and a diagnosed
mechanism. What the desk gets instead is a description of where the
informed room is positioned, labelled as a description. The chart carries
the rejection in its own caption, with the numbers, so nobody downstream
can rediscover the indicator by looking at the picture.

### 6.13 One line, and 100 is the trigger — the readiness panel, and the contract on when a number may change

Two desk reports arrived after §6.11 shipped, and although both read as
interface complaints, each turned out to be about something the project had
never actually written down.

**(i) "I don't get what is activating a signal, is it a crossing?
inflection?"** The answer is that it is a crossing, and only a crossing:
`alerts_from_scores` fires on `score >= threshold` and then suppresses
anything inside a 21-day cooldown. There is **no inflection test anywhere in
the firing path** — the convexity language in earlier discussion describes
features *inside* the score, not the trigger. The panel could not show this
because it drew five elements at equal visual weight (a faint raw level, a
bold 7d-smoothed level, an eligibility ribbon, a dated peak marker, and the
deciding score) and, decisively, because the two frozen thresholds sit at
**different heights** — GET IN `0.848141`, GET OUT `0.630231`. With two dotted
lines at two heights, neither one means *the* line.

The fix is a normalisation, not a new measurement. Each rule is drawn as

```
readiness = deciding score / that rule's own frozen threshold × 100
```

which places the trigger at **100 for every name, every rule, every window**,
and is precisely what allows both rules to share a single dotted line. **No
new number enters the model**: the same stored score is divided by the same
frozen threshold, the alert dates are bit-identical, and the vertical signal
lines still come from `coherent`.

Two construction choices carry a cost that is visible on screen, so both were
measured rather than asserted. First, a rule is drawn whenever its score
*exists* in the window, not only when it fired. With the level curve gone, the
old draw-on-fire rule would leave the instrument-lookup box — the one place a
PM checks a name that never alerted — showing an empty panel; and a line that
climbs to 80 and rolls over is exactly the answer to *why did nothing fire
here?*. Second, `connectgaps=False`, because the deciding score is sparse by
construction: `desk_candidacy` scores a name only on days the gates permit a
judgement, which over the live store (**63,345 name-days**) is **2.5% of days
for GET OUT and 48.8% for GET IN**. The GET OUT arcs are drawable rather than
dust — **1,600 scored days form 160 runs of median length 7 days** (mean 10,
max 91, only **28 single days**) — but those 28 must still render, and a gap
must never be bridged into a trend that was never scored. A blank day means
nothing *could* have fired there, whatever the crowd was doing.

The 0-100 euphoria level is not lost; it is what the dial above the panel
reads. The two questions are separated rather than merged: the dial answers
*how hot is this name*, the panel answers *how close is it to firing*.
`level`, `hype_ok` and both raw scores are untouched in the stores.

Alongside this, **performance reporting was withdrawn from the terminal**
entirely — no hit rate, lead time, false-alarm count or confidence interval
appears on the dashboard, and all of it lives in notebook 07. This is the
standing research/conclusions split applied consistently rather than a new
policy. The scope is stated here so it can be challenged: the justification
numbers inside the decision log and the band-meaning percentages on the gauge
were **kept**, because removing them would leave every remaining choice on the
page looking arbitrary, which is the failure mode this entire register exists
to prevent.

**(ii) "Why does update_data have to check what is the best model every
time?"** It does not, and the reason is defensibility rather than runtime.
Re-fitting on every live run makes the number on screen **untraceable**:
nothing on disk would describe how today's threshold differs from yesterday's,
and a threshold nobody can reconstruct cannot be defended in a review. The
contract is therefore explicit. A live run refreshes data and scores it with
the already-frozen winner. The single exception is the bootstrap —
`needs_research(stored)` is true only when no record exists at all, because
you cannot score against a record that is not there. Research re-opens only by
being typed: `run_analytics --what phases --research`, or `update_data.py
--full`, which counts as research **because a backfill rewrites the history
the thresholds were chosen on**, and scoring rebuilt history at thresholds
fitted on the old history would be a silent lookahead.

Staleness is *reported*, not repaired. When the frozen record stops at an
earlier year than the data, `record_lags_data` returns that year and the run
prints one notice line while continuing to score. That state is legitimate: it
is out-of-sample use, which is exactly what a walk-forward licenses. Refitting
every January would not make the threshold more correct — it would make it a
moving target that no stored record describes.

Writing this down exposed an error in shipped source. The docstring that
argued the case pointed at DECISIONS.xlsx `"2. Pipeline & Cadence"` and
parameter register `Class 6`; sheet 2 is *Literature* and Class 6 is the
*influence tracker*, so the contract's own citation led nowhere. Renumbering
the existing sheets and classes was rejected — every other cross-reference in
the repo would have broken silently — and the missing sections were created
instead (`3b. Pipeline & Cadence`, Class 9), with the docstring repointed.

### 6.14 The firing path written out end to end, and the third legibility pass

§6.13 asserted that a signal fires on a crossing. The desk then asked the
question that assertion should already have answered — *"how does it actually
work? the signal for get in / get out?"* — and the honest reading of that is
that the mechanism was distributed across §6.7, §6.8, a parameter-register
class and four source files, and had never been written out as one path. It is
written out here, read off shipped source rather than reconstructed from
memory, because a detector nobody can trace end to end cannot be defended in a
room.

Both rules run the same four stages, and the only differences between them are
which days are eligible and which features are averaged.

**Stage 1 — candidacy: is this day judgeable at all?** `desk_candidacy` splits
the day frame in two. A GET OUT candidate needs `hype_ok ∧ boom_state`: the
crowd is at or above its own swollen-attention gate *and* the price is already
at least its ground-truth boom threshold above its own trailing 120-day low
(25% for ETFs, 50% for single names, `boom_state_frame`, trailing closes only).
A GET IN candidate needs `hype_raw ≥ 1 ∧ ¬end_stage_mask`: the crowd is at or
above its own norm, and the day does not already satisfy every END gate — you
cannot start a euphoria that is already late. Days that fail candidacy get **no
score at all**, which is where the sparsity comes from: over 63,345 name-days
the GET OUT score exists on 2.5% of days and the GET IN score on 48.8%. The
price test inside the GET OUT gate is an **eligibility gate only** (§6.7); it
is never a scored feature, and the crowd-only detectors remain the headline
result.

**Stage 2 — the score.** GET OUT: `desk_end_fit` takes the plain, unweighted
mean of `["e1","e2","e3","e5","fade"]`, then zeroes it on any day failing
`e1 ≥ EUPHORIA_ATT_GATE ∧ e2 > 0` — the A2/A3 gates re-expressed in score
space, so one threshold governs the outcome instead of two. GET IN:
`desk_onset_fit` takes the mean of
`["attention_accel","hype_ratio","bull_inflection","influx_speed","attention_convexity"]`
(note `source_breadth` is in `ONSET_FEATURES` but deliberately **excluded** from
`ONSET_BANK`: its apparent skill was a coverage-regime artefact, since X and
StockTwits exist in the archive only from 2026). Neither score is fitted —
the family is rules, so fitting is a no-op, and that is why a live run has
nothing to re-select.

**Stage 3 — smoothing.** `_smooth_by_name` takes a **trailing** 7-day mean of
each instrument's own candidate-day sequence (`ROLL = 7`, the house one-week
window). Trailing, so there is no look-ahead. This is the stage that
structurally removes one-day blip alerts, at the recorded cost of two captures
(§6.8).

**Stage 4 — the crossing, then two suppressions.** `alerts_from_scores` is
`np.flatnonzero(s >= threshold)` walked in date order, keeping an alert only if
at least `EUPHORIA_COOLDOWN_DAYS = 21` days have passed since the last one.
**There is no inflection test, no peak test and no convexity test anywhere in
the firing path** — the convexity that appears in the GET IN bank is a
*feature* being averaged, not the trigger. The frozen thresholds are GET OUT
`0.630231` and GET IN `0.848141`, chosen once by the budget rule on full prior
years and never re-selected on a live run (§Class 9). Finally
`episode_coherent_alerts` applies the asymmetric state machine: a START within
21 days *after* an END is suppressed as a contradictory flip, an END is never
suppressed, and on a same-day tie the END wins — asymmetric by measurement,
since the symmetric rule cost the top detector 17 → 9 captures for only 8 fewer
false alarms.

That is the whole path, and it is what the panel's readiness line plots:
`stage-3 score ÷ that rule's frozen threshold × 100`, so the crossing in stage
4 is the moment the line touches 100.

**The same pass fixed two rendering defects that could only be found in the
browser.** A bold **undefined** was printing over every euphoria chart, and the
word appears in no Python file in this repo — four source-level hypotheses were
formed and discarded before the DOM was dumped, and the decisive step was
reading `data-unformatted` on `text.gtitle`, which returned
`"<b><b>undefined</b></b>"` and so named the mechanism: Streamlit's plotly
theming rewrites the title as `"<b>" + spec.layout.title.text + "</b>"`, and
with the title strip deliberately removed (it cost 55px per name) that inner
value is the JavaScript `undefined`. An explicit empty string is a real string,
so the same rewrite renders nothing. Second, the `GET OUT <date>` labels were
being clipped: they sit at y-domain 1.0 with `yanchor="bottom"` and alternating
`yshift` 4/18 to avoid colliding, and 9.5px text is ~13px tall, so the tallest
reaches ~31px above the panel against an 8px top margin — 38px clears it and is
still under the 55px the title used to cost. Both are recorded because the
general lesson is the one from §6.11: charts have to be *looked at*, and when
the source does not contain the symptom, the rendered object does.

**The ghost line, and a reversal made honestly.** Asked for *"a continuos line
… but maybe make it like not as prominant"*, the panel now draws each readiness
series a second time underneath — time-interpolated across gaps, 1px, dotted,
30% opacity, out of the legend. This reverses §6.13's own rejection of a dense
line, which was rejected for inventing a reading on ungated days, so the
reversal is only defensible with the reason the old version was unsafe
identified and removed. `hoverinfo="skip"` is that reason: the ghost never
reports a number, hover still comes only from the measured trace, and the
earlier objection was to a dense line that could be *queried* rather than one
that could be *seen*. `limit_area="inside"` confines interpolation to the span
between two real scored days, so the ghost cannot imply a reading in a stretch
the detector never judged. Nothing here touches the model and the alert dates
are bit-identical.

**Four explanation expanders left the page** on instruction — the
seven-decision summary, the long-form evidence log, the full method record and
the printed parameter register — leaving one plain-English opener whose label is
now frozen. The cost is stated rather than buried: a PM who challenges a
threshold live can no longer answer it from the page, and must go to the
register or a notebook. The three functions and constants behind those
expanders are left defined but unreferenced so restoring them is a two-line
change, and they are explicitly exempt from the dead-code sweep.

### 6.15 The watch track: one line that changes state, and a reversal replaced within the day

The ghost line above survived less than a day, and the reason it was replaced
is the more interesting result. The desk's next instruction was *"i dont want
the red / blue dots to appead just suddenly, perhaps plot it when it is not
eligable (not boom_state) but still hype_ok, e.g. tracks it until boom_state
and then becoems colored and becomes an alert"* — which is not a request for a
prettier construction line. It is the observation that the panel's sparsity
(§6.14: the GET OUT score exists on 2.5% of name-days because candidacy
requires `hype_ok ∧ boom_state`) makes an alert look like it arrives from
nowhere, and the fix for that is not to *draw* the missing stretch but to
*compute* it.

**Two ambiguities were put back to the desk rather than resolved by
assumption.** *"the 7day smoothed"* could mean the deciding score or the 0-100
euphoria level; both readings were offered with their costs, and the level
version was explicitly the one that would have resurrected the 27-Jul defect
(a curve sitting under a line while a flag flies, §6.13). The deciding score
was chosen. The second was whether the pre-eligible stretch should be
hoverable; *faint and un-hoverable* was chosen, which preserves the ghost's
one genuinely load-bearing guard.

**The wide domain.** The track calls the *production* scorers — `desk_end_fit`
and `desk_onset_fit` — with `train=None`. That is legitimate rather than a
shortcut: the desk family is a rules family, both functions ignore `train`
because fitting is a no-op, and using them rather than a reimplementation
means the track cannot drift from the thing it is drawn beside. Each is run
over the candidate set the instruction names, with the *eligibility* gate
dropped and every scoring gate intact:

| rule | live candidacy | wide (watch) domain | coverage |
|---|---|---|---|
| GET OUT | `hype_ok ∧ boom_state` | `hype_ok` only — price/boom gate dropped | 2.5% → **24.1%** of all name-days (15,244) |
| GET IN | `hype_raw ≥ 1 ∧ ¬end_stage` | every onset row (`hype_raw ≥ 1` holds on all of them) | 48.8% → **52.6%** (33,306) |

**The measurement that dictated the design.** The obvious implementation —
draw the wide series across the whole span and let the coloured trace sit on
top of it — was built and measured first, and it is wrong. `_smooth_by_name`
rolls `ROLL` days over each name's **candidate-day sequence, not the
calendar**, so widening the candidate set changes which days fall inside each
window: the wide recomputation is close to the stored score but not equal to
it. Over the 1,600 overlapping name-days the median absolute difference is
**0.0000** — but p95 is **0.2234 (35 threshold-points)**, the max is **0.7505
(119 points)**, and the two **disagree about whether the threshold was crossed
on 66 days (4.1%)**. On one day in twenty-five, therefore, a naive
implementation would have drawn a line above 100 under a day the detector had
judged to be below it, or vice versa — the §6.13 self-contradiction, restored
by accident. The shipped version masks the grey off **every** day a stored
score exists (`connectgaps=False`, so the mask is a real hole rather than a
bridge), which makes the two series structurally incapable of disagreeing: the
coloured trace owns every judged day, the grey owns only days with no verdict
at all. Verified in the rendered DOM as zero overlapping non-null points.

**What the grey then shows, which is more than continuity.** On ineligible
days the track is at or above the trigger on **1,891 GET OUT name-days (13.9%
of the ineligible wide domain)** and **425 GET IN name-days (17.6%)**. Those
are days the crowd score alone would have fired and the price gate held it
back — the eligibility argument of §6.8 made visible for the first time, and
the direct answer to a question a PM asks constantly and the previous panel
could not answer: *why did nothing fire here?* A second apparent anomaly is
equally informative: the grey sits flat on zero for long stretches on quiet
names, because `desk_end_fit` zeroes the score wherever the attention gate
fails. That is the production number, not a rendering gap, and the caption
says so. Neither case is clipped or suppressed; hiding them would remove the
only reason the grey earns its space.

**A note on how these three passes should be read together.** §6.13 removed a
dense line, §6.14 added a faint interpolated one back, and §6.15 replaced that
with a computed one. It would be tidier to present only the last, and the
standing instruction for this project is that a rejected attempt need leave no
trace in the code. It must, however, leave a trace *here* — because the
sequence is the argument. The first version was rejected for asserting a
reading on days nothing was judged; the second neutralised the assertion by
making the line unqueryable; the third removed the need to assert at all by
finding a real number for those days. Each step is a strictly weaker claim
than the one before, and the alert dates are bit-identical across all three.
Nothing in this section touches the detector.

**Two smaller items in the same pass.** The GET OUT explainer stopped printing
`E1`/`E2`/`E3`/`E5` and now labels every component from the single glossary in
`analytics/plain_english.py`; the stored column names are untouched, because
the parquet schema is an interface and translation belongs at the display
layer. And the chart cap stopped being silent: asked *"why are there so little
charts displayed? like only 8 charts"*, the answer is that three filters stack
and the binding one is not the cap but the requirement that a name have
**alerted inside the window** — 8 of 34 themes and 3 of the 8 present singles
in the default window, against 30/31 themes and 22/23 singles over all
history. The requirement stays (it is the standing "no filler names"
instruction), but the slider ceiling now exceeds the largest set the filters
can produce and a line above the charts states how many qualified against how
many are drawn, so a truncation can never again be mistaken for a quiet
universe.

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
15. Hanley, J.A. & McNeil, B.J. (1982). *The Meaning and Use of the Area
    under a Receiver Operating Characteristic (ROC) Curve.* Radiology
    143(1). — primary source for AUROC (§5.2, §6.1).
16. van Rijsbergen, C.J. (1979). *Information Retrieval* (2nd ed.),
    Butterworths. — primary source for the F-measure (§6.9).
17. Matthews, B.W. (1975). *Comparison of the predicted and observed
    secondary structure of T4 phage lysozyme.* Biochimica et Biophysica
    Acta 405(2). — primary source for MCC, read alongside [12] (§6.9).
18. Efron, B. & Tibshirani, R. (1993). *An Introduction to the
    Bootstrap.* Chapman & Hall. — the cluster (block) bootstrap used for
    every confidence interval in §6 (§5.2, §6.2).

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
| 2026-07-28 (x) | **The WATCH TRACK replaces the ghost line: one readiness line that changes state instead of appearing from nowhere; component names moved to plain English on screen; the chart cap stops being silent** (new §6.15; parameter register Class 8 fourth pass; DECISIONS `4. Detector Design` x4; ARCHITECTURE §8; RUNBOOK lower-panel + chart-selection blocks). **(i)** Asked to stop the coloured dots *"appead[ing] just suddenly"* and to track the score while a name is watched but not yet eligible, the ghost line of (w) was replaced within the day: an interpolated line is a guess about a number nobody computed, and a real number was available. The **production** scorers `desk_end_fit` / `desk_onset_fit` are called with `train=None` (legitimate - the desk family is a rules family, fitting is a no-op, so the track cannot drift from the live score) over the wide candidate set with only the ELIGIBILITY gate dropped: GET OUT `hype_ok` alone, **2.5% → 24.1%** of name-days (15,244); GET IN every onset row, **48.8% → 52.6%** (33,306). The line now runs grey/dashed while watched, turns rule-coloured on the day eligibility opens, and a crossing of 100 in the coloured stretch is the alert. **(ii)** The naive version - grey drawn everywhere, colour on top - was **built and measured first and is wrong**: `_smooth_by_name` rolls over each name's candidate-day SEQUENCE, not the calendar, so widening the candidate set moves every window. Over 1,600 overlapping name-days the median absolute difference is 0.0000 but p95 is **0.2234 (35 threshold-points)**, max **0.7505 (119 points)**, and the two **disagree about the crossing on 66 days (4.1%)** - one day in twenty-five would have recreated the §6.13 defect by accident. Shipped version masks the grey off every stored-score day with `connectgaps=False`, making the two series **incapable** of disagreeing; verified in the DOM as zero overlapping non-null points. **(iii)** The grey exceeding 100 is the exhibit, not a bug: **1,891 GET OUT name-days (13.9%)** and 425 GET IN name-days sit at or above the trigger while ineligible - days the crowd score alone would have fired and the price gate held it back, which is the §6.8 eligibility argument made visible and the answer to *"why did nothing fire here?"*. Grey flat on zero is likewise the production number (the attention gate zeroes the score), not missing data; both are stated in the caption rather than clipped. **(iv)** `E1`/`E2`/`E3`/`E5` gone from the GET OUT explainer, every label now from the single `PLAIN` glossary - **stored column names untouched**, because the parquet schema is an interface. **(v)** "Why are there so little charts?" answered by tracing three stacked filters: the binding one is **"must have alerted inside the window"** (8/34 themes, 3/8 present singles in the default window, against 30/31 and 22/23 over all history), not the cap - but the cap WAS silently truncating past 15, so the ceiling is raised beyond the largest possible qualifying set and the qualified-versus-drawn count is printed above the charts. Alert dates bit-identical; 108 tests pass. |
| 2026-07-28 (w) | **The firing path written out end to end; two rendering defects root-caused in the live DOM; a ghost line added under the readiness trace; four explanation expanders removed** (new §6.14; parameter register Class 8 third pass; DECISIONS `4. Detector Design` x4; ARCHITECTURE §8; RUNBOOK lower-panel block). **(i)** Asked *"how does it actually work? the signal for get in / get out?"*, the answer existed in shipped source but had never been written out as ONE path — it was distributed across §6.7, §6.8, a register class and four files. §6.14 now states all four stages read off source: candidacy (`hype_ok ∧ boom_state` for GET OUT, `hype_raw ≥ 1 ∧ ¬end_stage` for GET IN, which is where the 2.5% / 48.8% sparsity comes from), the unweighted bank mean with the A2/A3 gates re-expressed in score space, the trailing 7-day per-name mean, then `np.flatnonzero(s >= threshold)` with a 21-day cooldown and the asymmetric coherence machine. **It is a crossing; there is no inflection, peak or convexity test anywhere in the firing path** — the convexity in the GET IN bank is a feature being averaged, not a trigger. **(ii)** A bold **undefined** was printing over every euphoria chart and appears in NO Python file in the repo; four source hypotheses were discarded before the DOM was dumped, and `data-unformatted` on `text.gtitle` gave `"<b><b>undefined</b></b>"` — Streamlit's plotly theming bolds `layout.title.text`, which is the JavaScript `undefined` when the title strip is deliberately absent. Fixed with an explicit empty string. The `GET OUT <date>` labels were separately being clipped: derived from their own geometry (yshift up to 18 plus ~13px of text = ~31px above the panel) the top margin went 8 → 38, still under the 55px the deleted title cost. **(iii)** The **ghost line**: each readiness series drawn a second time underneath, interpolated across gaps at 1px / dotted / 30% / no legend. This knowingly reverses §6.13's rejection of a dense line, and is defensible only because the reason that version was unsafe is removed — `hoverinfo="skip"`, so the ghost NEVER reports a number (the objection was to a line that could be *queried*, not seen), plus `limit_area="inside"` so interpolation never extends past a real scored day. Alert dates bit-identical. **(iv)** Four model-evidence expanders removed from the page on instruction, one plain-English opener kept with its label frozen; **the cost is recorded, not buried** — a threshold challenged live now has to be answered from the register or a notebook. `decisions_simple()`, `DECISIONS_DOC` and `EUPHORIA_DEF_FULL` are left DEFINED BUT UNREFERENCED and are exempt from the dead-code sweep. 108 tests pass. |
| 2026-07-28 (v) | **One line, and 100 is the trigger: the euphoria panel rebuilt; performance reporting withdrawn from the terminal; the live run's research contract written down** (parameter register Class 8 continued and new Class 9; DECISIONS `4. Detector Design` x6 and new sheet `3b. Pipeline & Cadence`). **(i) The panel could not show what fires a signal.** The desk read the corrected chart from (s) and still could not answer its own question - *"i dont get what is activating a signal, is it a crossing? inflection? i just want one line"*. It **is** a crossing: `alerts_from_scores` fires on `score >= threshold` then applies a 21-day cooldown, and there is **no inflection test anywhere in the firing path**. The panel could not show that because it carried five elements at equal weight (raw level, 7d-smoothed level, eligibility ribbon, peak marker, deciding score) and the two frozen thresholds sat at DIFFERENT heights - GET IN `0.848141`, GET OUT `0.630231` - so neither dotted line meant *the* line. Replaced by one series per firing rule, **readiness = deciding score / that rule's own frozen threshold x 100**, which puts the trigger at **100 for every name, every rule, every window** and is what lets both rules share ONE line. **No new number enters the model**: same stored score, same frozen threshold, divided - alert dates are bit-identical and the vertical signal lines still come from `coherent`. Two deliberate construction choices, both with a cost that is visible on screen and therefore measured. Rules are drawn whenever their score EXISTS, not only when they fired, because with the level curve gone the draw-on-fire rule would leave the instrument-lookup box - the one place a PM checks a name that never alerted - showing an empty panel, and a line that climbs to 80 and turns over IS the answer to *why did nothing fire here*. And `connectgaps=False`, because the deciding score is sparse by construction: over the live store (**63,345 name-days**) it exists on **2.5% of days for GET OUT and 48.8% for GET IN**. The GET OUT arcs are drawable rather than dust - **1,600 scored days form 160 runs of median length 7 days** (mean 10, max 91, only **28 single days**) - but those 28 must still render and a gap must never be bridged into a trend that was never scored, since a blank day means nothing *could* fire there whatever the crowd is doing. **The 0-100 level is not lost**: it is what the dial above now reads, so the dial answers *how hot is this name* and the panel answers *how close is it to firing*; `level`, `hype_ok` and both raw scores are untouched in the stores. **(ii) Performance metrics withdrawn from the dashboard** on the desk's instruction (*"i will just keep this for the notebooks only"*), consistent with the standing split - the notebooks are the research record, the terminal states conclusions. Scope stated so it can be challenged: headline reporting is gone everywhere, but the justification numbers inside `decisions_simple()` / `DECISIONS_DOC` and the band-meaning percentages in `gauge_caption()` were KEPT, because stripping those would leave every remaining choice on the page looking arbitrary - the exact failure mode the parameter register exists to prevent. **(iii) Panel re-laid out** to *"less white space"*: header line, then dial sharing one row with five facts, then the figure at `title=None` and an 8px top margin. The old stack paid for a half-used dial row PLUS a second title strip inside the figure's own 55px margin, per name - most of a screen on a six-name page. The five facts (state, today's reading, 7d change, window peak, last signal) all answer **where this name is**; none answers how well the detector has done, which is (ii) honoured in the one place the new layout had made room to break it. The plain-English *"what is euphoria - start here"* opener was explicitly KEPT on instruction, and that is recorded so a later cleanup does not read it as leftover explanation. **(iv) The research contract, written down at last.** Asked *"why does the update_data have to check what is the best model everytime?"*, the answer is that it must not, and the reason is defensibility rather than runtime: **re-fitting on every run makes the number on screen untraceable**, since nothing on disk would describe how today's threshold differs from yesterday's, and a threshold nobody can reconstruct cannot be defended. One exception only - the bootstrap, `needs_research` true iff no record exists. Research re-opens by typing it: `run_analytics --what phases --research`, or `--full`, which counts **because a backfill rewrites the history the thresholds were chosen on** and scoring new history at old thresholds would be a silent lookahead. Staleness is REPORTED, not repaired - `record_lags_data` returns the lagging year and the run keeps scoring, which is out-of-sample use and exactly what a walk-forward licenses. Cadence and budget registered with it: ~2 runs a week, `PIPELINE_BUDGET_S = 600`. **A shipped-source pointer error was found and fixed in the same pass**: `analytics/euphoria.py::needs_research` promised this contract lived in DECISIONS `"2. Pipeline & Cadence"` and register `Class 6`, but sheet 2 is *Literature* and Class 6 is the *influence tracker*. Renumbering was rejected (it would silently break every other cross-reference in the repo); the missing sections were created and the docstring repointed. |
| 2026-07-28 (u) | **Notebook 07 closes with a verdict and a fork table; the short deck rewritten as an adoption case; three notebooks found to be silently un-runnable.** **(i) NB07 PART D - the final scorecard.** The battery ended without saying what shipped. It now prints a 24-field side-by-side record for both signals, and every rate carries an **instrument-cluster bootstrap 90% CI** (name-days inside a ticker are not independent). The bootstrap is **self-proving**: `record_ci` decomposes capture / detectable / false alarms per instrument and **asserts all three sums equal the shipped record** before resampling, so a CI can never be reported against a decomposition that silently disagrees with the record - the assert chain passes for both signals. Headline, unchanged from the shipped walk-forward and now stated with uncertainty: GET OUT **19.7% [14.3, 25.2]** capture on 122 detectable episodes over 2020-2026, **0.195 [0.120, 0.280]** FA per instrument-year against an inherited 0.230 budget, AP 0.449 vs a 0.374 random floor; GET IN **16.0% [8.7, 23.3]** on 125 over 2018-2026, **0.255 [0.204, 0.313]** FA-iy, AP 0.084 vs 0.062. **A reporting defect in that very table was caught by reading its output rather than trusting the green run**: the single row *median warning (days before peak)* printed **69 days for GET IN**, arithmetically correct and editorially wrong, and in flat contradiction with both the shipped record (16) and the deck (17 after the trough / 66 ahead). GET IN alerts carry **two different clocks** - `after_trough` is the entry lag, `before_peak` is the rally still ahead - and collapsing them into one label meant two things at once. Split into two named rows (`n/a for this signal` where a clock does not apply, never a blank), with a printed reading note; the saved `final_verdict` now records `median_lead_d` on the after-trough clock, the shipped record's convention, with `median_rally_ahead_d` carrying the other. **16.5 here vs 16 in the record is the same statistic** - `euphoria_phases.py` truncates with `int(np.median(...))` - and the notebook now says so, because an unexplained half-day gap between two documents reads as a disagreement. **(ii) NB07 PART E - the fork table.** Every methodological choice the project made is tabulated with what was tried and why it was not adopted, read from the notebooks' own saved JSON rather than retyped: **66 forks on the record - 52 rejected, 6 shipped, 8 recorded without a change**. It puts the price question on the record in the form the project actually settled it: **price in the feature bank LOST, and the boom-state eligibility GATE shipped instead**, a labelled second claim, so the crowd-only headline stands intact. **(iii) The short deck, v1.1.** Rewritten end to end as an adoption case - each slide leads with the desk benefit and closes with the evidence - with the three negative-result slides KEPT and reframed as the warrant for the positive ones rather than as a hedge, and an explicit closing ask (run it live on the book for a quarter). **Not one figure was changed to achieve the tone**, and that is asserted in the file's own header block so it survives a challenge. Verified against a **pre-edit baseline built in the same directory** (0 overfull), because attributing overflow without a baseline is guesswork: the rewrite introduced three overfull boxes, all three cleared by prose trims, final build 15 pages / 0 overfull / 0 errors, every measured-layout comment preserved. The long deck was left untouched by decision. **(iv) A silent kernel-killer in three notebooks.** jupytext un-escapes `# %matplotlib inline` in the paired `.py` into a LIVE magic in the `.ipynb`, so a trailing same-line comment becomes invalid magic arguments and **kills the kernel at cell 1**. Notebooks 03, 04 and 06 had been un-runnable this way; fixed in both halves of each pair, which unblocks the re-execution sweep. |
| 2026-07-28 (t) | **Influence convergence tested as a bullish / euphoria indicator and REJECTED; the theme-level crowding exhibit shipped in its place** (new §6.12; parameter register Class 6c; DECISIONS "8. Influence Tracker" ×5). The desk's sentence *"lets try and use the follower monitoring (influence) for some sort of bullish / euphoria indicator? anyway to make that clearer? perhaps if we have lots of influential accounts convergint on a theme?"* contains a PREDICTIVE claim and a LEGIBILITY request, and they were answered separately. **The predictive half was pre-registered before any number was seen:** three candidates built only from accepted quantities — C1 breadth (influence-weighted count of distinct voices, trailing `ROLL = 7`), C2 convergence (breadth × agreement, agreement = `\|Σ w·s\| / Σ w·\|s\| ∈ [0,1]`, the same arithmetic as the accepted `consensus`, so *many voices that DISAGREE is not convergence*), C3 backing share — against the accepted gauge outcome copied verbatim (>10% fall inside a week, any time in the next 30 days). **No threshold was invented**: the obvious HIGH-tier population cut is unusable on this store (**25 HIGH vs 12,503 low**; only **234 of 27,881** live calls come from a HIGH author), so authors enter continuously at `influence_index/100 ∈ [0,1]` — a nobody contributes ~0, the strongest record 1. The live window starts 2026-04-01 because the call history is **two disjoint blocks** (2021-06 holds 5,521 calls over only **2 distinct days**, an archive snapshot, then a five-year hole) and a trailing window cannot cross the hole. Paired bootstrap on **instrument** (name-days inside a ticker are not independent), 5 seeds × 300 reps, Bonferroni n=3 → conf **0.98333**, worst seed must clear zero. **ADOPTED: NONE** — on 15,615 name-days / 233 instruments (2026-04-08 → 2026-06-15, base rate 0.539): C1 **−0.0291** (lo −0.1701), C2 **−0.0153** (−0.1513), C3 **−0.0103** (−0.1706). **The null is informative because the harness has a positive control**: the accepted euphoria level, identical days, identical code, DETECTS (top decile cut 78.91 → 0.846 vs 0.414, **+0.4321**, lo +0.0623; level ≥ 85 → 0.925 vs 0.428, **+0.4970**, lo **+0.2515**). **The decisive exhibit is like-for-like**: on the control's OWN 1,552 name-days / 23 instruments, identical 156-day state size, base rate 0.457, all three are **WRONG-SIGNED** — C1 0.237 vs 0.482 (**−0.2449**), C2 0.250 vs 0.481 (**−0.2307**), C3 0.282 vs 0.477 (**−0.1950**). Not underpowered; pointing the other way. **The sign is diagnosed, not just reported**: top-decile-breadth names are MSFT 64, NVDA 59, TSLA 56, AMZN 46, RDDT 45, INTC 44, AAPL 44, SNDK 38, GOOGL 37, MSTR 37 — the board converges on the most-discussed liquid mega-caps, which cliff less often than the small-cap tail — and the between-name split is nearly flat (**0.534**, n=6,576 "ever top-decile" vs **0.542**, n=9,039 "never"), so the effect is **cross-sectional (which names), not temporal (when)**, which is exactly what disqualifies it as a timing indicator. Per the standing no-dead-traces rule, **no code from this test remains in the repo**; §6.12 and Class 6c are the trace. **The legibility half shipped**: the influence tab's bubble chart gained a names/themes toggle. `suggestion_digest` and the new `theme_digest` both delegate to ONE `_digest_frame(c, key)`, so the accepted consensus and backing formulas exist in a single place and the two views cannot drift — they sit on one control where any disagreement would be visible and unexplainable. Themes reuse `src/themes.py`, so a theme means one thing app-wide. A ticker in several themes **counts in every one** (NVDA is semiconductors AND ai AND ai_megacap; a PM asking "are we crowded into AI" must see that call), which makes theme shares shares of the **theme-mapped room** — a different denominator, stated on the control, so the views are not expected to agree name-for-name. Calls on tickers in no theme are **dropped, not bucketed as "other"**: "other" is not something a desk can position in and at **58.5% of live calls** it would be the chart's largest bar purely by being a residue. Live 90-day reading, which is why the view earns its place: **AI megacap 26.6% of the room's conviction at consensus +0.807** (a real convergence) against **semiconductors 10.2% at +0.222** (visibly an argument) — a distinction invisible one ticker at a time. The chart carries the rejection above in its own caption **with the numbers**, because "we checked" is not defensible and "−0.245 on the same days the accepted signal reads +0.497" is. Two supporting fixes: `_thin_labels` gained an optional `label_w_px` because its 30px gap silently hard-coded a four-character ticker while a theme label runs to nineteen — centred labels collide when the centre gap is under the **mean of their two widths**, and with all widths at 30px that expression IS the old scalar rule, so **the ticker view is unchanged by construction** (verified: old and new keep the identical 11 and 17 labels on the live 30d / 90d cross-sections), with 7.5px per character read off the accepted 30px rather than introduced as a new constant; and a tab-local `_unit` variable was found to have rebound the module-level `_unit()` scaler — the dashboard body executes at module scope — killing the influence map three hundred lines later with `'str' object is not callable`, now fenced by a hygiene test that asserts the module's helpers are still callable after the script runs. **9 new tests (99 → 107 passed)**, including a price-free invariant on the theme path that parses the AST and drops docstrings rather than grepping source, since the prose legitimately says "not a forecast about the price". |
| 2026-07-28 (s) | **Three legibility reports, one of which was a real chart defect** (new §6.11; parameter register Classes 3b and 8). **(i) The euphoria panel was quoting a threshold that had fired nothing.** The desk read *"its quite unclear to see WHEN is the actual change / or get out flag ... its like flat and then suddently a get out flag"* off a panel that drew one dotted line at the level-detector's walk-forward **85** and plotted the euphoria LEVEL against it — while the flags on screen come from the **desk score** crossing its own frozen threshold. MEASURED over all **95 GET OUT alerts** in the store: the plotted level sat BELOW the drawn line on **79 of them (83%)**, median plotted level at a GET OUT **74.8**. The desk was reading the chart correctly; the chart was wrong. Corrected by a rule rather than a tweak — *draw the threshold that gated the flags being drawn, and plot the series that crossed it* — with the GET OUT / GET IN scores overlaid at their frozen desk thresholds rescaled ×100 onto the panel's existing 0–100 axis (one axis, house rule), only for alert kinds that actually fired in the window, as `lines+markers` with gaps left open because the score is **sparse by construction** (`out_score` exists on **2.5% of name-days**, in runs as short as one day) and the gaps mean *not judgeable here*. The 85 line survives only on the no-desk-store fallback path. **No threshold moved.** The requested *"clear peak"* is the window maximum of the display curve, marked and dated — a label on a value already plotted, so removing it changes nothing about the signal (passed as `pd.DatetimeIndex([d])`, because a bare `[Timestamp]` survives the live app but is not JSON-serialisable by kaleido and silently breaks the PNG export the decks use). **(ii) The gauge "always shows calm" — and was right to.** MEASURED over the default window (2026-01-01 → latest): **50 of 59 instruments read calm at the last day while 17 of those same names touched the RED ZONE inside the window**, because the page is ordered by *most recent signal* so a name earns its place with an episode that may have peaked months ago while the needle correctly reports today. Softening the amber edge to make the dials look busier would have been exactly the arbitrary-threshold move §6.10 spent a page earning the right not to make; instead the dial now answers **both** questions — big needle = today, plus the window's high-water mark as one dated line behind it ('calm now, peaked 99 in red on 27 Jun'). **No edge moved, no number invented.** Drawn as TEXT rather than a second needle because plotly's Indicator has a single `threshold` slot already carrying the red edge as a hard line (colour alone does not survive greyscale or a projector), and two needles on a 268px dial reads worse than one sentence; frame 268→300px, bottom margin 36→74px to keep it in canvas. Per the desk's *"dont need to explain it fully all the time, maybe an info icon hover"*, the standing caveat and the measured band percentages moved into the `help=` tooltip — a change of **placement, not of evidence**: a page of six names had been carrying the same ~90 words six times. Gauge number and delta now `valueformat=".0f"`, because a tenth of a point on a percentile-rank index is below the resolution of the input. **(iii) Offensive handles masked on screen, and honestly labelled a CONVENTION.** There is no ground truth for "offensive", so no bootstrap can make a word list evidence-backed — what is registered instead is the **measured behaviour over all 12,528 real handles**. A naive one-list substring scan flags **345** and is dominated by false positives (`AfraidAnalyst`, `Valuable-Analyst-464`, `MeridianAllocation`, seven `Grapefruit` handles, `SatoshiTrails`). Three measured corrections: a match must lie **inside one token** (free, no word list — it alone resolves `SatoshiTrails`, *shit* spanning `oshi|Trails`, and `MeridianAllocation`, *anal* spanning `Meridian|Allocation`); six stems **demoted** to whole-token matching on counted innocent-vs-genuine hits (anal 3v2, rape 7v0, cock 3 innocent, boob 2 innocent, piss 1v0, wank 1v0); six mild words dropped entirely. Two promotions **adopted** on measurement (*retard* 10/10 genuine, *boobs* 3/3) and one **rejected** (*tits*: 2 hits, one genuine `Murrrtits` and one not, `Iplayminecraftitsfun` — a 50% error rate is not worth one handle). Fixed-point iteration is required and the proof is a real store case, not defensiveness: `Buttslut69696969` tokenises as `[Buttslut, 69696969]`, so pass 1 removes only *slut* and yields `Butt**69696969` where `Butt` IS now a whole token — that case FAILED the store-wide test before the fix. Final measured state: **97 / 12,528 masked (0.774%)**, **12,528 unique handles → 12,528 unique censored strings (zero collisions**, which is what makes span-level masking safe on a plotly category axis where duplicate labels merge into one bar), **one** residual false positive (`sashitadesol`), 4 of the top 120 by composite and 1 of the 25 HIGH-tier authors affected. Two invariants make it defensible: masking is **display-layer only** (`author` is the join key across `author_scores` / `calls` / `reply_edges`; a test asserts no mask reaches the parquet, and the map's `centre=`, the leaderboard's `_push` merge key and the ego selectbox's return value all keep the true handle), and the direction of error is deliberately **under-mask** (a missed handle is one embarrassing name on a board everyone knows is scraped from Reddit; an over-masked handle corrupts identity for every reader). Known misses and the **English-only** limitation (`fickdichdock` sits unmasked) are named in the register rather than hidden. **6 new tests (92 → 98 passed)**; AppTest 0 exceptions across 47 figures and 9 dataframes, with masking confirmed live on the rendered leaderboard, the influence-map hover and labels, and the ticker-backers axis. |
| 2026-07-27 (q) | **The euphoria GAUGE: one dial per theme and per ticker** (new §6.10; Class 1b of the parameter register). Desk request verbatim: "a very clear speedometer thing for each graph (and showing the change) for each theme / ticker". Built so that it introduces exactly **one** new number. The needle is `lvl_raw.rolling(ROLL).mean()` read at its last day - the identical object the lower panel already plots, so the dial and the curve beneath it cannot disagree (unit-tested); the delta reference is ROLL = 7d back, the same window the curve is smoothed over. The **red edge is 85 read out of `euphoria_report.json`**, i.e. the level the walk-forward already froze for the END alert in every test year 2018-2026 - a gauge that picked its own red edge would be a second, softer threshold competing with the detector (unit-tested against the report; the notebook asserts the walk-forward agreed across years, because otherwise a single red edge would be a fiction). The **one new number is the amber edge, 76**, and the route to it is the methodological content of this row. **The obvious test was the wrong test**: comparing a 95% CI for P(drop | level>=L) against a 95% CI for the base rate found NOTHING significant at any cut from 40 to 95 - the overlapping-CI fallacy, not a null result, because with 59 instruments both intervals are wide and overlap everywhere. Bootstrapping the **DIFFERENCE** on the SAME resampled instruments cancels shared instrument-level noise and recovers a significant effect at every cut from 76 up. Resampling unit = the **instrument, never the day** (adjacent days on one name are one episode; a day-level bootstrap would call 4,000 days of a single mania 4,000 independent facts). Rule pre-stated: lowest cut on a 2-point grid from 68 whose 95% paired lower bound excludes zero under **all 5 seeds** - cut 74 flips sign across seeds (-0.0004, +0.0009, -0.0012, -0.0007, +0.0004), cut 76 does not (+0.0045, +0.0060, +0.0049, +0.0034, +0.0046), and a cut that changes sign with the seed is not a parameter. Outcome graded is the desk's own "<1 month" horizon unchanged from S6.7 (a >=10% fall over 7d STARTS within 30d, as a reversed rolling max; the last 37 days are NaN because scoring an incomplete look-ahead as "no drop" would bias the base rate down exactly at the live edge). Measured on 183,394 name-days / 59 instruments / 2017-06-29 to 2026-06-15, base rate **23.3%**: level>=76 **26.9%** (+3.9pp, CI [+0.6,+7.4]), level>=85 **30.4%** (+7.3pp [+2.1,+12.8]), danger state alone **53.8%** (+31.2pp [+22.2,+40.1]), level>=85 AND danger **71.4%** (+48.2pp [+34.3,+59.5]). **The dial admits its own weakness on its face**: the level alone at the red edge is only ~1.3x base rate, the level plus an already-run-up price is ~3.1x, so the caption quotes the band the needle is actually in and a test pins P(red AND danger) > P(red) so it can never be re-worded into implying the needle is sufficient. The dial is a **STATE, never an instruction** - GET IN / GET OUT come from the detector and can fire with the needle anywhere; a test asserts those words can never appear in a band label. Every percentage the caption prints is read from `docs/research/gauge_zones.json`, which notebook 06's own code writes; a test greps the function body to prove neither edge is a literal in `dashboard.py`, and with the JSON absent the dial declines to exist rather than inventing bands. **Layout defect caught only by rendering the PNG and looking at it**: plotly draws an Indicator `title` inside the same domain as the arc, so the two-line header was struck through by the navy value bar - the header is now paper-space annotations in the top margin, and the value bar was thinned 0.28 -> 0.15 because at 0.28 it covered the very band colours it is meant to be read against. **9 new tests (84 -> 93 passed)**; smoke-tested on the real store at 2026-07-21 (semiconductors 70.9 calm, TSLA 82.3 warming, GME 87.6 RED ZONE) with all four caption branches rendering their measured percentages. |
| 2026-07-27 (r) | **Backlog audit against the file system, three items migrated out of a doomed document, and the repo swept.** The desk asked for the outstanding list to be *proved* rather than recalled, so every claim below is a grep or a stat, not a memory. Three findings changed the documentation. **(i) The feature bank's hand-specification was never recorded as a REJECTION** — §5.2 explained how the five onset features were derived but never that automated search (stepwise, L1 path, genetic) had been considered and refused, nor why. Now recorded with the argument that matters: the independent unit of evidence is the **instrument (59)**, not the day, so a search scoring candidates against 183,394 daily rows chooses among thousands of options on an effective n of 59 — the regime where selection noise dominates and the in-sample winner dies out of sample. Because every window is inherited from an already-validated constant, no feature adds a degree of freedom a test year has not seen; the **cost is stated openly** (per-feature AUROCs of only 0.51–0.60, and a search would very likely post better in-sample numbers). **(ii) The bootstrap replication count was in five files of code and in no register** — now Class 3, with the point that the replication count is not a lever (300 vs 1,000 moves an interval below the third decimal) while the **59-instrument cluster count** is what actually bounds every width; 200 is used only where a band is drawn at every event-window offset. **(iii) `notebooks/PROJECT_SUMMARY.md` was deleted, and it had to be**: a 28KB unreferenced reading-guide, frozen before the 2026-07-24 desk pivot, that had become a **contradiction source against this report inside the same repository** — episodes 211 vs 333, period 2021-2026 vs 2017-2026, onset capture 66% vs 23.2%, top capture 68% vs 13.1%, false alarms "0.23 budget maintained" vs the measured 0.348 that §8 lists as Limitation 4, lead time "17 days *before* starts" vs 17 days *after the trough* (sign-flipped), onset AP 0.0062 vs 0.098, test years "2025-2026" vs 2018-2026, precision "~75%" vs 0.21/0.34, START edge "+2 to +5%" vs the measured −0.81% with a CI containing zero. It also **invented a ground-truth rule that exists nowhere in the code** ("G3 duration ≥ 10 days") and preserved a **false** record of why phase-aware gating was rejected. Nineteen wrong numbers on a defence-day desk is a worse outcome than a missing document, and its three unique contributions were migrated first (items (i) and (ii) above, plus primary citations Hanley & McNeil 1982 / van Rijsbergen 1979 / Matthews 1975, now References 15–17 alongside Efron & Tibshirani 1993 for the bootstrap). Two of its unique items were deliberately **not** migrated and the refusal recorded: its asset-class scope caveat (contradicted by the commodities-in-universe decision) and its crowd-lag rationale for the 7d windows (an unmeasured assertion). **Also resolved without deleting anything: the "which euphoria report is headline, remove the loser" question has no loser** — `euphoria_desk.parquet` is the headline GET IN / GET OUT, `euphoria_onset.parquet` is the documented fallback that makes a fresh clone degrade instead of crash, and `euphoria_report.json` carries the frozen walk-forward thresholds *and is the source of the gauge's red edge 85*, so removing it would convert a derived number back into a hard-coded one. Emoji sweep completed and its boundary stated: the deleted summary was the **only** file carrying true emoji (15 codepoints plus a U+FFFD corruption); README's four U+25BA / three U+25BC arrowheads were replaced by same-width ASCII so no glyph can render as a coloured emoji, while box-drawing characters and mathematical notation (≥, −, →, ≈) are kept as **notation, not decoration**. `docs/HANDOFF_PROMPT.md` corrected: it still carried the "junior-programmer level, no clever tricks" instruction the desk scrapped on 2026-07-24, so any session bootstrapped from it re-imported a dead rule. 93 tests pass throughout. |
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
