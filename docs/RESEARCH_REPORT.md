# Detecting the Start and End of Retail Euphoria from Crowd Data Alone: A Walk-Forward Study on Nine Years of Social-Media Attention

**Alex Brown — GIP 2026 Project — MAARS Global Macro**
*RetailRadar research report — LIVING DOCUMENT (see Changelog, Appendix C). Last updated 2026-07-24.*

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
changed.

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

### 6.6 Influential-users model

Harness validated end-to-end (synthetic schema-faithful fixture, deleted
after use); real results pend the store's first live seeding. Thesis
benchmark for orientation: GraphSAGE AP 0.140 / AUROC 0.632 ≈ +61% AP
over random. This section will be updated from notebook 05's first real
run (see Changelog).

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

**5. The economics of sparsity.** The whole system alerts a few names at
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
test suite (40 invariants: crowd-only rule, no look-ahead, window caps,
cooldowns, text-free stores) plus a dashboard smoke harness gate every
change.

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

*(The full 12-entry defense reference list, keyed to slides, lives in
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
python -m pytest tests/ -v                         # 40 invariants
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
