# RetailRadar — pitch deck brief

**What this file is.** A complete prompt for a second Claude to build the
pitch deck. Everything it needs — audience, narrative, every number, the
house palette, the exact figure list — is here. Hand it this file and the
repository; it should not need to ask a question to start.

**Audience:** GIC traders and portfolio managers. Technical enough to
know what AUROC is and to be suspicious of a backtest, but they are not
data scientists and they are not sitting an exam. They have seen a lot of
alt-data pitches that did not survive contact with a live book.

**The tone that wins this room:** confident about the engineering,
scrupulously honest about the statistics. This deck's single biggest risk
is over-claiming. A PM who catches one inflated number stops believing
the other twenty, and this project's whole credibility rests on having
refused to overfit. Sell the *rigour* — the walk-forward, the frozen
thresholds, the no-price-leakage rule, the things that were tested and
rejected. That is the differentiator, not the hit rate.

---

## 0. HARD REQUIREMENTS

**Format:** self-contained HTML deck (inline CSS/JS, no external assets
except the figure PNGs). One slide per full viewport, arrow-key and
scroll navigation.

**Progress bar — explicitly requested.** A thin bar pinned to the top of
every slide showing position through the deck, in `NAVY` on a `HAIRLINE`
track, 3px tall, animating on slide change. Add a small slide counter
(`7 / 22`) bottom-right in `INK_MUTED`. The audience should always know
how much is left.

**Palette — take these verbatim from the dashboard so the deck and the
product look like one thing:**

| token | hex | use |
|---|---|---|
| `NAVY` | `#0A1E2E` | headers, primary series, the progress bar |
| `NAVY_MID` | `#1B3A52` | secondary series |
| `WHITE` | `#FFFFFF` | slide background |
| `PANEL` | `#F6F7F8` | recessed blocks, table headers |
| `INK` | `#111111` | body text |
| `INK_MUTED` | `#666666` | captions, axis labels, slide counter |
| `HAIRLINE` | `#ECECEC` | gridlines and rules — never thicker than 1px |
| `SLATE` | `#7A8794` | price lines, context series |
| `BULL` | `#1F6F5C` | long / bullish / GET IN |
| `BEAR` | `#A6413B` | short / bearish / GET OUT |
| `TEAL` | `#2E6E7E` | the one cool accent |
| `OCHRE` | `#8A6D1F` | warnings, the danger state |

**Font:** `Inter, 'Neue Haas Grotesk', 'Helvetica Now', -apple-system,
'Segoe UI', Roboto, sans-serif`. Numbers in tabular figures
(`font-variant-numeric: tabular-nums`) so columns align.

**Chart house style** (matches `_theme()` in `dashboard.py`): white
background, no chart frame, gridlines at `HAIRLINE` only, no axis lines,
no legend border, whitespace does the separating. Never use default
matplotlib or Plotly colours. **Regenerate every figure in this palette**
— do not screenshot old figures in other colours.

**Design language:** navy on white, high whitespace, hairline rules,
restrained accents. "Annual report, not startup." No gradients, no drop
shadows, no emoji, no icon soup. One idea per slide.

---

## 1. THE NUMBERS — all measured, all in this repo

Do not invent figures. Everything below is real and sourced. If a number
is not here, either find it in `docs/research/*.json` or leave it out.

### Scale of the data
- **786,669** ticker mentions extracted, **7,995** distinct symbols
- **305,905** daily ticker rows, **2017-01-01 → 2026-07-29** (9.5 years)
- **39** themes tracked, **34** with a firm-approved tradeable instrument
- **17** subreddits, panel maintained by an automatic monthly discovery
  pass — plus StockTwits and X
- **417** instruments priced from Bloomberg, 616,748 price rows
- Full refresh runs in **under 10 minutes**, ~2×/week

### The detector's walk-forward record (`docs/research/nb04_evaluation.json`)
**GET OUT** (euphoria ending, test years 2021/2022/2023/2026):
captured **21 of 98** detectable tops = **21.4%**, false alarms **9**
= **0.075 per instrument-year**, **zero late**, median warning
**8 days** (90% CI 1–24), AP 0.614.

**GET IN** (euphoria starting, test years 2018–2026):
captured **17 of 125** = **13.6%**, FA **0.195/instrument-year**,
median entry lag **22 days**, AP 0.089 vs 0.067 baseline.

Frozen thresholds: **0.6298** (GET OUT) / **0.8777** (GET IN).

### Ground truth (`docs/research/nb01_episode_stats.json`)
**333** episodes, median run **86 days**, median boom **+63.5%**, median
bust **−36.3%**.

### Feature battery (`docs/research/nb02_feature_stats.json`)
Per-feature AUROC **0.51–0.56**, AP lift **1.06–1.35×** baseline.

### Influence tracker (live store)
**15,122** authors scored, **25** in the HIGH tier, **42,743** directional
calls, **10,207** judged, base rate **0.391**.

### Engineering
**155** tests, dangling-reference check at 0 findings, euphoria stage
**4.9s** (was 19.0s before a vectorisation pass).

### Market context — VERIFY BEFORE USING
Goldman Sachs put retail at roughly **30% of US equity trading volume**
with about **$12 trillion** in self-directed capital. **These are the
only external stats in this brief and they must be re-checked against a
current primary source before the deck is shown.** Add one or two more —
retail share of single-stock options volume, and the 2021 GME peak — and
cite each on the slide in `INK_MUTED` 10px.

### HOW TO TALK ABOUT THE HIT RATE — read this before writing slide 12

21.4% capture is **not** "we are right 21% of the time". It is: *of every
top that met a strict pre-registered definition and was measurable, the
crowd signal flagged one in five, a median of 8 days early, at 0.075
false alarms per instrument-year.* The denominator is a definition, not a
universe of trades.

Say it that way. A PM who has been mis-sold a backtest will trust the
person who volunteers the denominator far more than the person quoting a
bigger number. **The 0.075 FA/instrument-year is the impressive figure**
— roughly one false alarm per instrument every 13 years — and the "zero
late" is arguably the most important: the signal never arrived after the
top had already broken.

---

## 2. SLIDE PLAN

24 slides. Slides marked **[SUGGESTED]** are additions to the requested
14 — each earns its place and the reason is given. Drop any of them for
time; the spine (1–4, 6, 11, 14, 17, 20) must stay.

### ACT I — Introduction and the product (slides 1–6)

**1. Title**
> **RetailRadar — the crowd, measured**
> Turning 786,669 retail posts into a tradeable signal
Subtitle: your name, GIC, the date. No figure. Full-bleed `NAVY` panel,
white type.

**2. The flow has changed**
Title that sells: **"A third of the tape is now retail. We measure the
institutions."**
Content: retail as % of US equity volume, the self-directed capital
number, GME/GLD/SMH as episodes where retail *set* the marginal price.
Land the gap in one sentence: GIC is strong on institutional flow and
has no systematic read on the other third.
→ **FIGURE 1**

**3. And it is discussed in public, in advance** **[SUGGESTED]**
Why it belongs: slide 2 says retail matters; a PM's immediate objection
is "and how would you possibly know what they are doing?" Answer it
before it is asked.
Content: retail coordinates in the open on Reddit, StockTwits and X. Show
mention volume for one theme building *ahead of* its price move.
→ **FIGURE 2**

**4. The product**
Title: **"A daily read on what retail is crowding into — and when that
crowd is about to break."**
Three sources → one pipeline → two signals (GET IN / GET OUT) on 34
tradeable themes and 25 single names. Live, refreshed twice a week in
under 10 minutes.
→ **FIGURE 3**

**5. The dashboard**
Screenshots, minimal text. Let the product carry it.
→ **FIGURES 4, 5**

**6. What a signal looks like on a real name**
Title: **"Eight days before the top."** One worked example, price with
the flag marked, crowd attention underneath.
→ **FIGURE 6**

### ACT II — What is behind the dashboard (slides 7–19)

**7. Section divider** — "What is behind it". `NAVY` panel, one line:
*"Every number on that screen is reproducible from raw text."*

**8. The funnel**
Raw text → cleaning → counts, derivatives, time series **(credit: Hao
Quan)** → sentiment **(credit: Yi Peng)** → features → signal.
Put the volume at each stage on the funnel so the attrition is visible.
→ **FIGURE 7**

**9. Why a lexicon beats an LLM here**
Title: **"We chose the auditable engine — and we can prove why."**
FinVADER on a social-media finance corpus, versus an LLM: no training-set
bias toward names the model has read about, every score traceable to the
tokens that produced it, and it is free to run on 786k posts. Note the
LLM was benchmarked (notebook 10), not assumed away.
→ **FIGURE 8**

**10. From posts to factors**
The decomposition: attention level, attention acceleration, convexity,
hype ratio versus the name's own normal, bullish persistence, crowd
influx speed. Every feature is *relative to that instrument's own
history* — "extreme" always means extreme for this name.
→ **FIGURE 9**

**11. Do the factors separate at all? (AUROC)**
Title: **"Small edges, honestly measured."** Per-feature AUROC with
confidence intervals against the 0.5 line. **Do not hide that these are
0.51–0.56.** Frame it: single weak features, combined under a rule that
was frozen before it was tested — that is what a real edge looks like at
this scale, and anything claiming 0.8 on crowd text is fitting noise.
→ **FIGURE 10**

**12. Precision where it matters (AP)**
AP versus base rate per feature, with the lift. Tops are rare; AP is the
metric that respects that and AUROC does not.
→ **FIGURE 11**

**13. Feature correlation** **[SUGGESTED]**
Why: the first question a quant asks is "are these five features one
feature wearing five hats?" Answer with the Spearman matrix
(`docs/research/research_stats.json` → `feature_correlation_spearman`).
→ **FIGURE 12**

**14. Threshold engineering: no overfitting, no price leakage**
**The most important slide in the deck.** Title: **"The rules were frozen
before they were graded."**
- Walk-forward: for each test year the threshold is picked on *earlier
  years only*, applied unchanged, never revisited.
- The crowd side of the signal sees **no price at all**. Price enters in
  exactly two places, both stated out loud: the boom gate, and the
  grading of whether a top was real.
- A live run **never re-selects**. Research decides once; the answer is
  frozen into a stored record.
- False-alarm budget fixed at 0.23/instrument-year *in advance*.
→ **FIGURE 13**

**15. What we tried and threw away** **[SUGGESTED — strongly recommend]**
Why: nothing buys credibility with this audience faster than a list of
your own rejected ideas. An ML challenger (logistic regression) lost to
the rules on the pre-stated utility rule and was dropped. Graph/network
layers on the influence model were all rejected. An index-level signal
for the S&P was built, measured, and killed because a diversified index
does not stage the run-up-then-bust arc. Ground-truth thresholds were
swept and left unchanged because the only setting that improved the
ratio deleted GameStop from the sample.
→ **FIGURE 14**

**16. The frontier**
The capture/false-alarm frontier: every candidate configuration, the
budget line, the adopted point marked. This is the slide that shows a
choice was *made*, not stumbled into.
→ **FIGURE 15**

**17. The noise test**
Title: **"We fed it noise. It found nothing — which is the point."**
Gaussian-noise control: the same pipeline on synthetic series produces no
signal, so the measured edge is not an artefact of the machinery.
→ **FIGURE 16**

**18. Robustness** **[SUGGESTED]**
Why: "does it survive a small change to your assumptions?" gets asked
every time. Show the parameter sweeps (`config_sweep.json`) — the result
is flat across neighbouring values, which is what a non-overfit system
looks like.
→ **FIGURE 17**

**19. Where it works, and where it does not**
Title: **"It works on retail-heavy flow. We can show you exactly where
the line is."**
Performance split by retail intensity. High-retail names carry the
signal; institutional-flow names do not. Stating the boundary is what
makes the working half believable.
→ **FIGURE 18**

### ACT III — Beyond the core signal (slides 20–24)

**20. The performance report**
The headline table: GET OUT and GET IN, captured/detectable, FA per
instrument-year, median lead, with confidence intervals. Phrase the
denominator per §1.
→ **FIGURE 19**

**21. Additional capability** — one slide, bullets only:
influence tracking, AI market pulse, agentic-trading watch, emerging-term
detection, conviction tracking.

**22. Influence: not every poster is worth the same**
Louvain community detection over the reply graph; 15,122 authors scored,
25 in the HIGH tier, ranked by their **measured** record (10,207 judged
calls, 39.1% base rate) rather than by follower count.
→ **FIGURES 20, 21**

**23. AI Pulse**
An LLM writes the words; the numbers come from the stores and are saved
beside the prose so any sentence can be audited against its inputs. The
model is never asked to produce a statistic.
→ **FIGURE 22**

**24. Handover — this runs without me**
Title: **"One CSV to maintain."**
- Themes and keyword maps are refreshed by an automatic weekly AI audit
  that proposes into a CSV and **cannot write to the config** — a human
  types YES.
- The only genuinely manual input is the **NPA-approved ETF list**.
- Two commands, 155 tests, a dangling-reference check, full documentation
  (RUNBOOK, ARCHITECTURE, PARAMETER_REGISTER, handover).
→ **FIGURE 23**

**Closing slide:** the one-line ask. What do you want from this room —
a pilot on N names, a desk sponsor, a data budget? Make it explicit.

---

## 3. FIGURE LIST

**Existing assets** — real dashboard screenshots, use directly:
`docs/figures/dashboard/00_euphoria-themes.png`, `01_euphoria-singles.png`,
`02_influence-tracker.png`, `03_overlays-themes.png`, `04_top-trends.png`,
`05_emerging-trends.png`, `06_conviction.png`, `07_ai-pulse-sample.png`,
`08_historical-checker.png`.

**Everything else must be generated** into `docs/presentation/figures/`
in the house palette. Titles below are the *chart* titles — write them to
sell, in sentence case, stating the finding rather than the variable
names. "Attention builds three weeks before the price does" beats
"mention_share vs px_last".

| # | Slide | Figure | Source |
|---|---|---|---|
| 1 | 2 | Retail share of US equity volume, trend | external, verify first |
| 2 | 3 | Mention volume vs price for one theme, attention leading | `daily_theme_counts.parquet` + `prices.parquet` |
| 3 | 4 | System schematic: 3 sources → pipeline → 2 signals | draw |
| 4 | 5 | Dashboard, themes tab | `docs/figures/dashboard/00_*` |
| 5 | 5 | Dashboard, single names | `01_*` |
| 6 | 6 | Worked example: price, GET OUT flag, attention below | `euphoria_desk.parquet` + prices |
| 7 | 8 | Funnel with volume at each stage | counts from §1 |
| 8 | 9 | FinVADER vs LLM comparison | `nb10_ai_sentiment.json` |
| 9 | 10 | The six factors on one instrument, stacked panels | `euphoria_levels.parquet` |
| 10 | 11 | AUROC per feature + CI, 0.5 reference line | `nb02_feature_stats.json` |
| 11 | 12 | AP vs base rate per feature | `nb02_feature_stats.json` |
| 12 | 13 | Feature correlation heatmap | `research_stats.json` |
| 13 | 14 | Walk-forward schematic: train ≤ Y−1, test Y, threshold frozen | draw |
| 14 | 15 | Rejected-ideas table as a graphic | `nb03_tournament.json`, `nb07_*` |
| 15 | 16 | Capture vs FA frontier, budget line, adopted point marked | `nb06_strictness.json` |
| 16 | 17 | Gaussian-noise control vs real signal | generate |
| 17 | 18 | Parameter sweeps, flat response | `config_sweep.json` |
| 18 | 19 | Performance split by retail intensity | `nb06_*` / `nb07_*` |
| 19 | 20 | Headline performance table as a graphic | `nb04_evaluation.json` |
| 20 | 22 | Louvain community graph of the reply network | influence store |
| 21 | 22 | Influencer hit rate vs call volume | influence store |
| 22 | 23 | AI Pulse panel | `07_ai-pulse-sample.png` |
| 23 | 24 | Maintenance model: what is automatic vs manual | draw |

**Chart rules:** one idea per chart; the finding in the title; units and
n on every axis; confidence intervals wherever a rate is quoted; source
file in 9px `INK_MUTED` bottom-left of every generated figure, so any
number can be traced back during Q&A.

---

## 4. THINGS THAT WILL LOSE THE ROOM — avoid

- Any performance number without its denominator or its interval.
- The word "AI" doing work that "a frozen rule set" actually does. The
  core detector is rules, not a model. Say so — it is a strength.
- Claiming the S&P or index-level version works. It was built and killed;
  slide 15 turns that into credibility.
- Hiding that AUROC is ~0.55. Volunteer it, then explain why that is the
  honest shape of a crowd-text edge.
- Screenshots in a different colour scheme from the deck.
- More than ~40 words of body text on any slide.

## 5. BEFORE HANDING IT OVER

Re-read every number against `docs/research/*.json`. Check every claim
about what ships against `docs/ARCHITECTURE.md` §6. If a slide asserts
something that the repo does not support, cut the slide — that rule is
why the underlying work is defensible, and the deck should be held to it
too.
