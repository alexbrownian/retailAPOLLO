# RetailRadar — pitch deck brief

**Read this first.** You are building this deck on a machine that has
**only this file and the PNGs in `figures/`**. No repository, no data, no
notebooks. Everything you need is therefore written out below — every
number, every slide, every figure and what it shows. Do not go looking
for a source file; if a number is not in this document, leave it out.

---

## OUTPUT FORMAT — READ THIS TWICE

**The deliverable is ONE PowerPoint file: `RetailRadar_pitch.pptx`.**

Not HTML. Not a markdown outline. Not a web deck, a Google Slides link,
a PDF or a set of images. A `.pptx` file that opens in PowerPoint on a
locked-down corporate machine and can be edited there. If you find
yourself writing a `<div>`, you have taken a wrong turn.

**Build it on the GIC corporate template.** Open the firm's `.potx` /
`.pptx` template and add slides using **its** layouts — title, section
divider, title-and-content, full-bleed image. Do not invent a theme, do
not override the master's fonts or logo placement, and do not rebuild
GIC's branding by hand. The template's look wins every time it conflicts
with anything in this brief; the only things here that must survive are
the *figures* (already rendered, do not restyle) and the *words*.

**How to build it.** Use `python-pptx`, opening the GIC template as the
base presentation so its masters and layouts carry through:

```python
from pptx import Presentation
prs = Presentation("GIC_template.potx")     # NOT Presentation()
layout = prs.slide_layouts[1]               # use the template's own layouts
slide = prs.slides.add_slide(layout)
```

Starting from `Presentation()` with no argument gives you the stock
Office theme and loses the branding entirely — that is the most common
way this goes wrong.

**If a `pptx` skill is available to you, read it before you start.** It
covers template handling, layout selection and image placement, and will
save you from the usual traps.

**Slide size:** whatever the GIC template uses — almost certainly 16:9.
Do not change it.

**Placing the figures:** insert each PNG at its **native aspect ratio**;
never stretch to fill a placeholder. The charts are 200 dpi and sized for
full-width placement, so set the width to the content area and let the
height follow. Leave the figure's own title in place — it states the
finding and is part of the argument — and do not add a duplicate title
above it in the slide's text box.

**Speaker notes:** put the supporting detail in the notes pane, not on
the slide. The forty-word limit is for what the audience reads; anything
you want said out loud belongs in the notes. This matters most on slides
11, 14, 15 and 20, where the honest framing of the numbers is the whole
point and will not fit on the face of the slide.

**Progress indicator:** the GIC template probably has none. Add a thin
navy rectangle across the top of each content slide showing position
through the deck (a filled bar over a light-grey track, ~4pt tall,
width proportional to slide number), plus a `7 / 24` counter bottom-right
in grey. If the template's footer already carries slide numbers, keep the
bar and drop the counter. Skip both on the title slide and the section
dividers.

The palette below is for matching accents to the figures — the figures
themselves are final.

---

## THE AUDIENCE, AND THE ONE WAY THIS PITCH FAILS

GIC traders and portfolio managers. Technical enough to know what AUROC
is and to be suspicious of a backtest. They are not data scientists and
they have all been shown alt-data that did not survive contact with a
live book.

**The single biggest risk in this deck is over-claiming.** One inflated
number and they stop believing the other twenty. The differentiator here
is not the hit rate — it is that the work refused to overfit, and can
prove it. Sell the rigour: the walk-forward, the frozen thresholds, the
no-price-leakage rule, and the long list of things that were tested and
thrown away. Confident about the engineering, scrupulous about the
statistics.

### How to talk about the hit rate — read before writing slide 20

21.4% capture is **not** "we are right 21% of the time." It is: *of every
top that met a strict, pre-registered definition and was measurable, the
crowd signal flagged one in five, a median of 8 days early, at 0.075
false alarms per instrument-year.* The denominator is a definition, not a
universe of trades.

Say it exactly that way. A PM who has been mis-sold a backtest will trust
the person who volunteers the denominator far more than the person
quoting a bigger number. **The headline is not 21% — it is 0.075 false
alarms per instrument-year** (roughly one per instrument every 13 years)
**and zero late.** The signal never once arrived after the top had
already broken. That is the number that matters to someone carrying risk.

---

## HOUSE STYLE

The figures are already rendered in this palette. Match the deck's
accents to it so the slides and the charts read as one system.

| role | hex |
|---|---|
| primary / headers / progress bar | `#0A1E2E` navy |
| secondary series | `#1B3A52` |
| background | `#FFFFFF` |
| recessed panels, table headers | `#F6F7F8` |
| body text | `#111111` |
| captions, axis labels, counter | `#666666` |
| rules and gridlines (never thicker than 1px) | `#ECECEC` |
| price / context series | `#7A8794` |
| bullish / GET IN | `#1F6F5C` |
| bearish / GET OUT | `#A6413B` |
| accent | `#2E6E7E` |
| warning | `#8A6D1F` |

Font: Inter, or the GIC template's own sans. Numbers in tabular figures
so columns align. Design language: navy on white, high whitespace,
hairline rules, restrained accents — annual report, not startup. No
gradients, no drop shadows, no emoji, no icon soup. **Maximum ~40 words
of body text per slide.**

---

## EVERY NUMBER YOU MAY USE

All measured on the live system. Nothing here is rounded up.

**Scale.** 786,669 ticker mentions extracted from 7,995 distinct symbols.
305,905 daily rows spanning 2017-01-01 to 2026-07-29 — 9.5 years. 39
themes tracked, 34 with a firm-approved tradeable instrument. 17
subreddits plus StockTwits and X. 417 instruments priced from Bloomberg
across 616,748 price rows. A full refresh runs in under 10 minutes and
is run about twice a week.

**GET OUT (euphoria ending)** — walk-forward test years 2021, 2022, 2023,
2026: captured **21 of 98** detectable tops = **21.4%** (90% CI
15.0–28.6%); false alarms **9** = **0.075 per instrument-year** (90% CI
0.033–0.117); **0 late**; median warning **8 days** (90% CI 1–24);
AP 0.614.

**GET IN (euphoria starting)** — test years 2018–2026: captured **17 of
125** = **13.6%**; false alarms **0.195 per instrument-year**; 4 late;
median entry lag **22 days** (90% CI 14–30); median rally still ahead
**61 days**; AP 0.089 against a 0.067 base rate; AUROC 0.552.

Frozen thresholds: **0.6298** GET OUT, **0.8777** GET IN.

**Ground truth.** 333 episodes. Median run 86 days, median boom **+63.5%**,
median bust **−36.3%**.

**Factors.** Per-feature AUROC **0.51–0.56**, AP lift **1.06–1.35×**
baseline. Only two of the five factors correlate strongly (E1 with E3 at
ρ 0.80); the rest are near-independent.

**Influence tracker.** 15,122 authors scored, 25 in the HIGH tier, 42,743
directional calls of which 10,207 are judged, base rate 39.1%.

**Engineering.** 155 automated tests, a dangling-reference check at zero
findings, and the euphoria stage runs in 4.9s (down from 19.0s after a
vectorisation pass).

**Market context — the only external numbers here, and they MUST be
re-verified against a current primary source before this is shown.**
Goldman Sachs has put retail at roughly **30% of US equity trading
volume**, with about **$12 trillion** in self-directed capital. Add the
2021 GameStop peak and retail's share of single-stock options volume if
you can source them. Cite each on-slide in 10px grey.

---

## THE WORKED EXAMPLE — it runs through the whole technical section

**This was specifically requested and it is the spine of Act II.** Rather
than explaining the method abstractly and then showing one example, the
same instrument is followed through four consecutive slides, from raw
attention to the price outcome. The audience never has to hold two
examples in their head.

The instrument is the **memory** theme (DRAM/HBM — Micron, SK Hynix,
Samsung, SanDisk), traded through SMH. Figures `W1` → `W4` are the four
steps and each carries a "WORKED EXAMPLE · memory" ribbon so the thread
is visible.

- **W1 — the crowd, counted.** Attention as a share of all theme chatter,
  against that theme's *own* 120-day normal. The point: "extreme" always
  means extreme for this name, so a permanently loud name is judged
  against loud-for-itself.
- **W2 — decomposed into factors.** The same period, four factors
  stacked: attention level, sustained bullishness, crowd influx,
  super-exponential attention.
- **W3 — the factors become one level, and the flag fires.** Three GET
  OUT flags, the last one landing on the top.
- **W4 — the payoff.** The same flags against the SMH price. The price
  fell **19.7%** within 60 days of the final flag. Say clearly that the
  crowd side of the signal never saw this price series.

**Do not oversell W4.** Deliver it as "this is what one episode looks
like", then move immediately to slide 20's aggregate record. One good
chart is an anecdote; the walk-forward is the evidence.

---

## SLIDE PLAN — 24 slides

### ACT I — Introduction and the product

**1. Title.** *RetailRadar — the crowd, measured.* Subtitle: turning
786,669 retail posts into a tradeable signal. Your name, GIC, date.
No figure.

**2. A third of the tape is now retail. We measure the institutions.**
Retail as a share of US equity volume, the self-directed capital figure,
and GME / GLD / SMH as episodes where retail set the marginal price. Land
the gap in one line: GIC is strong on institutional flow and has no
systematic read on the other third. *No figure supplied — build a simple
chart from the verified stat, or run the slide on type alone.*

**3. And they discuss it in public, before they act.** Pre-empts the
obvious objection. Retail coordinates in the open on Reddit, StockTwits
and X. → **`W1_walkthrough_attention.png`** (introduce the memory thread
here — attention building well before the flag).

**4. The product.** *A daily read on what retail is crowding into — and
when that crowd is about to break.* Three sources, one pipeline, two
signals across 34 themes and 25 single names, refreshed twice a week in
under 10 minutes. *Draw a simple schematic.*

**5. The dashboard.** Screenshots, minimal text. **Screenshots are not
supplied in `figures/` — take them from the live dashboard on the day.**
Themes tab and single-names tab.

**6. Eight days before the top.** The signal on a real name.
→ **`W4_walkthrough_outcome.png`**

### ACT II — What is behind it

**7. Section divider.** Navy panel: *"Every number on that screen is
reproducible from raw text."*

**8. The funnel.** Raw text → cleaning → counts, derivatives and time
series **(credit: Hao Quan)** → sentiment **(credit: Yi Peng)** →
features → signal. Put the volume at each stage on it: 786,669 mentions,
7,995 symbols, 39 themes, 59 scored instruments. *Draw this.*

**9. We chose the auditable engine, and we can prove why.** FinVADER on a
social-media finance corpus rather than an LLM: no training-set bias
toward names the model has read about, every score traceable to the
tokens that produced it, and free to run across 786k posts. Note the LLM
was benchmarked, not assumed away. *No figure — a two-column comparison
in the template's table style.*

**10. From posts to factors.** → **`W2_walkthrough_factors.png`**
Attention level, acceleration, convexity, hype ratio versus own normal,
bullish persistence, crowd influx speed — every one relative to that
instrument's own history.

**11. Small edges, honestly measured.** → **`F10_feature_auroc.png`**
AUROC per factor with confidence intervals against the coin-flip line.
**Do not hide that these are 0.51–0.56.** Frame it: weak individual
features combined under a rule frozen before it was tested is what a real
edge looks like at this scale, and anything claiming 0.8 on crowd text is
fitting noise.

**12. Tops are rare, so precision is the metric that counts.**
→ **`F11_feature_ap.png`** Average precision against base rate, with the
lift on each bar.

**13. Five factors, not one factor five times.**
→ **`F12_feature_correlation.png`** The Spearman matrix. Answers the
first question a quant will ask.

**14. The rules were frozen before they were graded.** *The most
important slide in the deck.* → **`W3_walkthrough_level_and_flag.png`**
- Walk-forward: for each test year the threshold is chosen on *earlier
  years only*, applied unchanged, never revisited.
- The crowd side sees **no price at all**. Price enters in exactly two
  places, both stated out loud: the boom gate, and grading whether a top
  was real.
- A live run **never re-selects**. Research decides once and the answer
  is frozen into a stored record.
- The false-alarm budget was fixed at 0.23 per instrument-year **in
  advance**.

**15. What we tried and threw away.** *Requested, and it is the
credibility slide.* Bullets, no figure needed:
- An **ML challenger** (logistic regression on the same features) lost to
  the rules on the pre-stated utility rule — 17 captures against 6 — and
  was dropped.
- Every **graph and network layer** on the influence model was rejected;
  the model does not generalise to unseen authors, so the dashboard ranks
  by measured record instead.
- An **index-level signal** for the S&P 500 was built and killed: at the
  frozen bars the S&P produces **zero** gradeable episodes, and so does
  XLI. Volatility-scaling the bars does not rescue it. Diversification is
  precisely what stops an index staging the run-up-then-bust arc.
- The **ground-truth thresholds were swept** across eight settings and
  left unchanged, because the only setting that improved the ratio also
  deleted GameStop and gold from the sample.
- A **rally/mobilisation detector** was built, validated on the 2021
  regime, and cut on cost-benefit.

**16. The operating point was chosen on a stated rule.**
→ **`F15_frontier.png`** Capture against false alarms, the pre-declared
budget line, the adopted point circled.

**17. We fed it noise. It found nothing — which is the point.**
→ **`F16_noise_control.png`**

**18. Move any knob one step and the answer barely changes.**
→ **`F17_parameter_sweeps.png`** Flat response is what a system that was
not overfit looks like.

**19. It works on retail-heavy flow. Here is exactly where the line is.**
Honest boundary slide. High-retail names carry the signal; institutional
names do not. **Use the semiconductors counter-example here**: that
theme's most recent GET OUT was followed by a further **+23%** over 60
days — the flag was early and the rally continued. Volunteering that buys
more credibility than any chart on the previous three slides.

### ACT III — The record, and what happens next

**20. The performance report.** → **`F19_performance_table.png`**
Phrase the denominator as set out at the top of this brief.

**21. Additional capability.** Bullets only: influence tracking, AI
market pulse, agentic-trading watch, emerging-term detection, conviction
tracking.

**22. Not every poster is worth the same.** Louvain community detection
over the reply graph. 15,122 authors scored, 25 in the HIGH tier, ranked
by their **measured** record — 10,207 judged calls against a 39.1% base
rate — not by follower count. *No figure supplied; screenshot the
influence tab from the live dashboard.*

**23. AI Pulse.** An LLM writes the words; the numbers come from the
stores and are saved beside the prose so any sentence can be audited
against its inputs. The model is never asked to produce a statistic.
*Screenshot the AI Pulse tab.*

**24. This runs without me.** *One CSV to maintain.*
- Theme and keyword maps are refreshed by an automatic weekly AI audit
  that proposes into a CSV and **cannot write to the config** — a human
  types YES.
- The only genuinely manual input is the **NPA-approved ETF list**.
- Two commands, 155 tests, a dangling-reference check, and full
  documentation: RUNBOOK, ARCHITECTURE, PARAMETER_REGISTER, handover.

**Closing.** The ask, stated explicitly — a pilot on N names, a desk
sponsor, a data budget. Do not end on "thank you".

---

## FIGURE INDEX — 15 files, one folder, all of them used

`figures/` holds **exactly** what the deck needs and nothing else. Every
file below appears on a slide; there is no "extras" pile to sift, and
nothing else in the repository needs to be hunted down.

Place them full-width at native aspect ratio. **Do not recolour, crop or
re-title them** — each already carries its finding as a title and its
data source in small grey type at the bottom left, which is what lets the
number be defended in Q&A.

**The worked example — four consecutive slides, one instrument**

| file | slide | shows |
|---|---|---|
| `W1_walkthrough_attention.png` | 3 | memory attention against its own 120-day normal |
| `W2_walkthrough_factors.png` | 10 | the same period decomposed into four scored factors |
| `W3_walkthrough_level_and_flag.png` | 14 | factors → one level → three GET OUT flags |
| `W4_walkthrough_outcome.png` | 6 | those flags against SMH; −19.7% within 60 days |

**The evidence**

| file | slide | shows |
|---|---|---|
| `F10_feature_auroc.png` | 11 | every factor's CI clears the coin-flip line |
| `F11_feature_ap.png` | 12 | average precision beats base rate on all five |
| `F12_feature_correlation.png` | 13 | the factors are near-independent |
| `F15_frontier.png` | 16 | capture vs false alarms, budget line, adopted point |
| `F16_noise_control.png` | 17 | real crowd vs Gaussian control |
| `F17_parameter_sweeps.png` | 18 | six knobs, flat response |
| `F19_performance_table.png` | 20 | the headline walk-forward record |

**The product**

| file | slide | shows |
|---|---|---|
| `S05a_dashboard_themes.png` | 5 | the themes tab |
| `S05b_dashboard_singles.png` | 5 | the single-names tab |
| `S22_influence_tracker.png` | 22 | the influence board |
| `S23_ai_pulse.png` | 23 | the AI Pulse panel |

**Only three slides need artwork you must draw:** 2 (retail share of
volume — verify the stat first), 4 (system schematic), 8 (the funnel).
Keep all three in the template's own diagram style; simple boxes and
arrows in navy beat anything decorative.

---

## TONE — CONFIDENCE AND CANDOUR ARE NOT OPPOSITES

This deck volunteers three things that are not flattering: AUROC is
~0.55, GET IN lags by 22 days, and the most recent semiconductors flag
was early. **That is deliberate, and it must be delivered as strength,
not apology.** The distinction is placement and phrasing:

- **On the slide:** state what the evidence shows. "Every factor clears
  the coin-flip line." "Zero late." "One false alarm per instrument every
  13 years." Never put a hedge in a chart title or a headline — a slide
  that argues with itself reads as a lack of confidence.
- **In the speaker notes and out loud:** give the caveat, unprompted, the
  moment the number lands. "These are 0.51 to 0.56, and anything claiming
  0.8 on crowd text is fitting noise." That sentence, said before anyone
  asks, is worth more than the number itself.
- **On one slide only (15) and one boundary slide (19):** concentrate the
  negatives. Four rejected approaches and one honest miss, gathered in
  one place, read as a research process. The same material sprinkled
  across twelve slides reads as a project full of holes.

The evidence base is genuinely strong — a nine-year walk-forward, frozen
thresholds, no price leakage, a noise control, flat parameter response
and 155 tests. Present it that way. The candour is what makes the strong
parts believable; it is not a disclaimer on them.

---

## WHAT WILL LOSE THE ROOM

- Any performance number without its denominator or its interval.
- Letting "AI" do work that "a frozen rule set" actually does. The core
  detector is rules, not a model. Say so — it is a strength.
- Claiming the index or S&P version works. It was built and killed;
  slide 15 turns that into credibility.
- Hiding that AUROC is ~0.55, or that the semis flag was early.
- Screenshots or new charts in a different colour scheme.
- Walls of text. Forty words a slide.
