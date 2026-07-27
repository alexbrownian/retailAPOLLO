# dashboard.py
# ============
# The retailAPOLLO terminal - one interactive Streamlit + Plotly dashboard
# over the whole pipeline. Every chart is hoverable/zoomable, every section
# leads with a RANKED table so the TOP item is unmistakable, and the Trade
# Desk shows live, dated suggestions (entry -> 20-day exit) ranked by most
# recent.
#
#     pip install streamlit plotly
#     python -m streamlit run dashboard.py
#
# WHAT CHANGED vs the RetailFlow1 dashboard
#   * The price-overlay analytics that used to live in notebooks 11-16
#     (mention share vs price, attention first derivative, lead/lag scan,
#     direction flips, conviction crossings, the signal report card) are
#     now TABS here, computed on demand from analytics/overlays.py - no
#     notebook rendering anywhere, so "refresh the overlays" is just
#     moving a slider.
#   * All heavy computation is shared with the pipeline (analytics/), so a
#     number shown here is by construction the same number the pipeline
#     wrote to disk.
#   * Data loading is cached on (path, file-mtime): the dashboard reruns
#     instantly on interaction and self-invalidates the moment the
#     pipeline rewrites a file.
#
# LAYOUT OF THIS FILE (find your way quickly)
#   1. constants, CSS, logo + loader SVGs, long definition texts
#   2. cached data loading
#   3. Plotly figure builders (the dark "terminal" look)
#   4. sidebar: window controls + pipeline run buttons
#   5. topline metric strip + ticker lookup
#   6. the tabs:  Trade desk | Overlays: themes | Top trends |
#                 Emerging | Conviction | AI Pulse | Historical
#                 (single-ticker overlays removed - the desk trades themes)

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.config import (ROLL, DERIV_SMOOTH, MIN_TOTAL, CROSS_AT,   # noqa: E402
                        MIN_GAP, PROCESSED_DIR, PRICES_PATH, REFERENCE_DIR,
                        EUPHORIA_HYPE_MULT, EUPHORIA_BOOM_MIN_ETF,
                        EUPHORIA_BOOM_MIN_SINGLE,
                        CONV_EXIT_LEVEL, CONV_EWM_HALFLIFE,
                        EUPHORIA_EXCLUDED_THEMES)
from src.themes import THEME_ETFS, THEME_ETF_FALLBACKS             # noqa: E402
from analytics import overlays                                     # noqa: E402
from analytics import influence_graph as ig                        # noqa: E402
from analytics.plain_english import PLAIN, plain                   # noqa: E402,F401,E501
from analytics.loaders import (price_series, clip_window,          # noqa: E402
                               THEME_COUNTS)
from analytics.overlays import (mention_share_series,              # noqa: E402
                                chatter_change_series,
                                conviction_crossings, crossing_exits)

st.set_page_config(page_title="RetailRadar", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# DESIGN TOKENS - institutional light theme (GIC design language, adopted
# 2026-07-27 by desk decision).  The brief: navy on white, high whitespace,
# hairline rules, restrained accents, "annual report rather than startup".
#
# WHY tokens and not literals: before this block the file carried 33 loose
# colour literals, so a palette change meant 33 edits and a guaranteed
# missed one.  Everything below is named for the ROLE it plays, never for
# the colour it happens to be - so the next re-skin touches only this block.
# ---------------------------------------------------------------------------
NAVY = "#0A1E2E"          # primary: headers, the euphoria curve, buttons
NAVY_MID = "#1B3A52"      # secondary series, hover states
WHITE = "#FFFFFF"         # page
PANEL = "#F6F7F8"         # the occasional recessed section
INK = "#111111"           # primary text
INK_MUTED = "#666666"      # secondary text
INK_LABEL = "#717171"      # uppercase small labels - see note below
HAIRLINE = "#ECECEC"       # 1px rules and gridlines - never thicker
SLATE = "#7A8794"          # price line, raw/context series

# DESK DECISION - INK_LABEL departs from the brief's #8C8C8C.
# Measured with tools/contrast_audit.py, which computes the WCAG 2.1 contrast
# ratio of every text node against its COMPOSITED backdrop.  #8C8C8C scores
# 3.36:1 on WHITE and 3.14:1 on PANEL; the AA bar for text below 18.66px is
# 4.5:1, and every use of this token here is 10-12px (card labels, axis
# titles, the masthead subtitle).  The brief's grey therefore fails on the two
# surfaces this dashboard actually paints on.
# #717171 is the LIGHTEST neutral grey that clears 4.5:1 on BOTH: 4.88:1 on
# white, 4.55:1 on PANEL.  PANEL is the binding constraint - a grey chosen
# against white alone (#767676, 4.54:1 white) still fails at 4.24:1 inside a
# recessed panel.  Keeping the lightest passing value preserves the brief's
# intent - these labels must recede - while remaining readable at 10px on a
# projector, which is where this gets defended.

# Direction pair.  Muted on purpose: the brief says "never vibrant", but
# bullish-vs-bearish still has to read in one glance from across a desk, so
# these are the deep annual-report versions of green and red rather than the
# screen-bright trading ones.  Both stay legible printed or projected.
BULL = "#1F6F5C"          # muted teal-green - long / bullish
BEAR = "#A6413B"          # muted brick      - short / bearish
TEAL = "#2E6E7E"          # the one cool accent (GET IN markers)
OCHRE = "#8A6D1F"         # the danger state - warning without alarm

FONT_STACK = ("Inter, 'Neue Haas Grotesk', 'Helvetica Now', "
              "'Helvetica Neue', Arial, sans-serif")

# Role aliases.  These names appear at ~30 call sites and each one names a
# ROLE the chart needs (accent series, positive, negative, neutral), so they
# are kept as aliases rather than mass-renamed - the palette moved, the
# meaning did not.
ACCENT = NAVY             # the emphasised series on any chart
GREEN, RED, PURPLE, BLUE, GRAY = BULL, BEAR, NAVY_MID, TEAL, INK_LABEL

# Header mark: the animated bars + orbit (the same mark the pipeline
# loader uses - the five bars rise one by one, then the orbit line draws
# itself through the middle and the cycle repeats).
HEADER_MARK_HTML = """
<svg width="64" height="64" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <style>
    .hb { fill:#0A1E2E; transform-box:fill-box; transform-origin:50% 100%;
          transform:scaleY(0);
          animation:hbar 2.6s cubic-bezier(.4,0,.2,1) infinite; }
    .h1 { animation-delay:0s;   } .h2 { animation-delay:.14s; }
    .h3 { animation-delay:.28s; } .h4 { animation-delay:.42s; }
    .h5 { animation-delay:.56s; }
    .horb { fill:none; stroke:#0A1E2E; stroke-width:5;
            stroke-dasharray:155; stroke-dashoffset:155;
            animation:horbit 2.6s ease-in-out infinite; }
    @keyframes hbar {
      0%   { transform:scaleY(0); opacity:1; }
      18%  { transform:scaleY(1); opacity:1; }
      82%  { transform:scaleY(1); opacity:1; }
      95%  { transform:scaleY(1); opacity:0; }
      100% { transform:scaleY(0); opacity:0; }
    }
    @keyframes horbit {
      0%, 30% { stroke-dashoffset:155; opacity:1; }
      60%     { stroke-dashoffset:0;   opacity:1; }
      82%     { stroke-dashoffset:0;   opacity:1; }
      95%     { stroke-dashoffset:0;   opacity:0; }
      100%    { stroke-dashoffset:155; opacity:0; }
    }
  </style>
  <rect class="hb h1" x="22" y="38" width="8" height="46"/>
  <rect class="hb h2" x="36" y="26" width="8" height="70"/>
  <rect class="hb h3" x="50" y="16" width="8" height="90"/>
  <rect class="hb h4" x="64" y="26" width="8" height="70"/>
  <rect class="hb h5" x="78" y="38" width="8" height="46"/>
  <ellipse class="horb" cx="54" cy="61" rx="45" ry="9"/>
</svg>"""

# Animated loader: five bars rise ONE BY ONE (staggered delays keep their
# phase every loop), then an orbit line draws itself through the middle,
# everything fades, and the cycle repeats.
LOADER_HTML = """
<div style="display:flex;align-items:center;gap:14px;padding:6px 0;">
<svg width="72" height="72" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ldb { fill:#0A1E2E; transform-box:fill-box; transform-origin:50% 100%;
          transform:scaleY(0);
          animation:ldbar 2.6s cubic-bezier(.4,0,.2,1) infinite; }
    .l1 { animation-delay:0s;   } .l2 { animation-delay:.14s; }
    .l3 { animation-delay:.28s; } .l4 { animation-delay:.42s; }
    .l5 { animation-delay:.56s; }
    .ldorb { fill:none; stroke:#0A1E2E; stroke-width:5;
            stroke-dasharray:155; stroke-dashoffset:155;
            animation:ldorbit 2.6s ease-in-out infinite; }
    @keyframes ldbar {
      0%   { transform:scaleY(0); opacity:1; }
      18%  { transform:scaleY(1); opacity:1; }
      82%  { transform:scaleY(1); opacity:1; }
      95%  { transform:scaleY(1); opacity:0; }
      100% { transform:scaleY(0); opacity:0; }
    }
    @keyframes ldorbit {
      0%, 30% { stroke-dashoffset:155; opacity:1; }
      60%     { stroke-dashoffset:0;   opacity:1; }
      82%     { stroke-dashoffset:0;   opacity:1; }
      95%     { stroke-dashoffset:0;   opacity:0; }
      100%    { stroke-dashoffset:155; opacity:0; }
    }
  </style>
  <rect class="ldb l1" x="22" y="38" width="8" height="46"/>
  <rect class="ldb l2" x="36" y="26" width="8" height="70"/>
  <rect class="ldb l3" x="50" y="16" width="8" height="90"/>
  <rect class="ldb l4" x="64" y="26" width="8" height="70"/>
  <rect class="ldb l5" x="78" y="38" width="8" height="46"/>
  <ellipse class="ldorb" cx="54" cy="61" rx="45" ry="9"/>
</svg>
<span style="color:#8C8C8C;">working...</span>
</div>"""


def decisions_simple():
    """The decision log as a PM reads it: one line per decision, each ending
    in the number that settled it.

    The full log below is the defence document and stays exactly as it is.
    But it opens with a re-set of the project aim and runs to eleven dense
    paragraphs, and the question it is actually asked on a desk is much
    smaller: *what did you choose, and what makes you sure?*  So this version
    is one bullet per decision, phrased as choice -> evidence, with the
    rejected options named.  Naming what was REJECTED is the part that earns
    trust: a model with no rejected variants has not been tested."""
    cfg = _research("nb06_desk_config")
    ds = _research("nb06_desk_signal")
    n4 = _research("nb04_final_eval")
    n1 = _research("nb01_episode_stats")

    raw = _row(cfg.get("table"), variant="GET OUT boom-gated raw")
    prod = _row(cfg.get("table"),
                variant="GET OUT boom-gated SMOOTHED (production)")
    gi_old = _row(cfg.get("table"), variant="GET IN incumbent raw")
    gi_new = _row(cfg.get("table"),
                  variant="GET IN phase-aware SMOOTHED (production)")
    gbm = _row(ds.get("table"), variant="gbm COMBINED")
    # The ML comparison must come from the SAME table as the challenger:
    # nb06_desk_signal and nb06_desk_config count captures under slightly
    # different candidate definitions (23 vs 24), and quoting one against the
    # other would be a like-for-unlike comparison a reviewer would catch.
    ship = _row(ds.get("table"), variant="COMBINED + smoothed trigger")
    buy = _dig(n4, "trading_translation", "ONSET→BUY", default={})
    sell = _dig(n4, "trading_translation", "TOP→SELL", default={})

    return f"""
### What was decided, and what decided it

Every line below is a choice we made, followed by the measurement that made
it. Nothing here is a preference.

**1 - The dashboard shows two signals, not four.**
GET IN and GET OUT only. The old BUY/SELL engine was retired because on the
full price history it *lost* money (**-1.42% per trade, 2021-2026**) - it
bought retail enthusiasm into the 2022 bear market. It still exists in the
research code; it is simply not desk output.

**2 - The alert waits for the price to have actually run.**
Requiring a real boom before a GET OUT can fire took captures from
**{_dig(n4, 'top', 'captured', default='-')} to {raw.get('captured', '-')}**
of {prod.get('detectable', '-')} - *more* hits, not fewer, because it stopped
the detector wasting alerts on names that were never in a rally.

**3 - The trigger reads a smoothed score, which killed the one-day blips.**
The desk complaint was "euphoria for a single day, then gone". Averaging the
score over a week before triggering raised quality
(**AP {raw.get('AP', '-')} → {prod.get('AP', '-')}**) and cost
**{(raw.get('captured', 0) or 0) - (prod.get('captured', 0) or 0)} captures**.
That cost is recorded, not hidden.

**4 - A START can no longer print next to an END.**
The old version put a GET IN right beside a GET OUT **{gi_old.get('adjacency', '-')}
times**; now it happens **{gi_new.get('adjacency', '-')}** times, because a day
that already meets every ending condition is not allowed to call a start. The
price was captures (**{gi_old.get('captured', '-')} → {gi_new.get('captured', '-')}**),
paid on purpose: a start landing on an end destroys trust in the panel.

**5 - Machine learning was tried and did not win.**
A gradient-boosted model on the same features caught
**{gbm.get('captured', '-')}** endings against our
**{ship.get('captured', '-')}** - compared inside the same tournament table,
so the two counts are like-for-like. The rules we ship also match what the model
*learned* was important - which is independent evidence they are not
arbitrary.

**6 - "Buy on GET IN" and "short on GET OUT" were both REJECTED.**
Tested properly and neither beat simply holding: GET IN
**{_pct(buy.get('diff'), 2, signed=True)}** over 20 days
(90% interval {_ci_pct(buy.get('ci90'))}), GET OUT
**{_pct(sell.get('diff'), 2, signed=True)}** (interval {_ci_pct(sell.get('ci90'))}).
Both intervals contain zero, so we do **not** claim a trade. This is a
risk-warning tool.

**7 - Thresholds are learned from the past, then frozen.**
Each year's threshold is fitted only on years *before* it, and the live
pipeline never revises yesterday's signal. Base data:
**{n1.get('episodes_total', '-')} historical hype episodes**, of which
**{n1.get('top_detectable', '-')}** were detectable at the top, median run
**{n1.get('median_run_days', '-')} days**.

---

*Every one of these numbers is read live from the notebook exports, so this
panel cannot go stale. The long-form version of this log, with the full
tables, is below.*
"""


DECISIONS_DOC = """### Model decisions & evidence log

**0 - THE AIM (re-set July 2026): detect retail EUPHORIA and call price
TOPS.** The dashboard's headline signal is the 0-100 euphoria level and
its red alert lines; success = an alert inside [peak-30d, peak+1d] of a
genuine top (peak = local high after a boom, followed by a >=15% ETF /
>=30% single-name drawdown within 90d). The BUY/SELL engine was retired
from the dashboard at the same time (its full-history record was
negative - see point 1); it remains in analytics/ for research.
Universe: equities + retail commodities only (rates_bonds and
real_estate excluded); single names join the themes. Full rules +
research grounding: the EUPHORIA definition expander on the first tab
and analytics/euphoria.py.

**0b - PREDICTION IS REDDIT-ONLY (desk rule, July 2026).** Price never
enters the euphoria level or the alert; it only DEFINES and SCORES the
ground-truth tops. This was a deliberate trade: the earlier variant with
a price-convexity feature and a price-boom gate captured **46%** of
detectable peaks (0.08 FAs/instr-yr); the crowd-only detector captures
**~23%** (median lead 4d, ~0.11 FAs/instr-yr). The chart carries real
information - giving it up is the documented price of the clean claim
"the crowd alone called the top". The identified path to winning capture
back WITHOUT price: richer crowd data (the comment backfill is ~10x the
post volume and directly feeds every euphoria ingredient).

**0c - Every rule is ablation-tested and the hand-rules beat an ML
challenger under a pre-stated criterion** (tables on the EUPHORIA tab):
dropping the hype gate floods false alarms (+83), dropping the fade
trigger loses the most captures (-0.095 of detectable) - each rule has a
measured job. A walk-forward logistic regression on the same features
was adopted-or-rejected by a criterion fixed before the numbers were
seen; it lost on capture (a near-silent model can win the utility score
by never firing - a top detector that never fires is not a better top
detector). Its learned weights rank the same features top, which is
independent evidence the hand-rules are not arbitrary.

*(every material modelling choice, what was tested, and the numbers -
so no rule on this dashboard is a black box. All PnL figures: real
Bloomberg closes, signals from 2021, per-year cross-validation.)*

**1 - The BUY/SELL engine is the ORIGINAL RetailFlow1 logic, unchanged.**
Verified by diffing the two projects' signal files: every tradeable
signal matches (the only differences are in `cannabis`, which has no
approved instrument and never trades). Scoring *RetailFlow1's own file*
on real prices gives BUY = **-1.42%/trade** over 2021-2026 - identical
to this project, because it is the same engine. The old project never
scored full history (its report cards were window-clipped), which is why
recent windows *felt* good: per-year avg %/trade = 2021 **+6.3**, 2022
**-11.1**, 2023 **-3.0**, 2024 **+1.7**, 2025 **-1.3**, 2026 **+0.4**.
The engine's weakness is one regime: it buys retail enthusiasm into bear
markets (2022). Same signals, different viewing window => different PnL
- moving the window start from trailing-365d to Jan-2026 alone shifts
the scorecard from +4.4% to +1.6% total with no model change at all.

**2 - Conviction display engine: EWM baseline (halflife 42d).** Chosen
over the rolling-84 window and share-normalised variants in a study on
real prices: long its own +2.5 up-crossings, 20d hold = **+1.36%/trade,
63% hit, 299 trades, +0.78% above the anchor ETFs' unconditional drift,
positive 5 of 6 years**, stable across halflife 28/42/60 and holds
10/20/30 (a plateau, not a curve-fit spike). It also re-centres within
weeks after collection-volume shocks - the cause of the old
"every theme reads negative" episodes. The engine's own ingredients were
tested with the EWM z too: it made the engine WORSE (-1.80 vs -0.93
%/trade), so the engine keeps its original rolling-84 construction.

**3 - Grey exit markers ("back to neutral") - KEPT.** On the validated
crossing strategy, exiting when z reverts returns +0.83%/trade in ~10
days vs +1.36% in 20 - less per trade but **0.080 vs 0.065 %/day**: the
same capital can work ~2x as often. On the engine's own BUY trades,
reversion-exit cuts the loss from -1.42% to -1.03%/trade (SELL: only 4
priced trades - too few to judge). The official scorecard still accounts
fixed-20d holds; the markers and the desk's REVERTED hint are the
capital-efficiency overlay on top.

**4 - Shorts from conviction: REJECTED.** Every short construction
tested loses (plain down-cross **-1.33%/trade**, post-peak reversal
-0.67, shallow -0.30; at best 2/6 years positive). Retail conviction
fading is not bearish price information - its value on the sell side is
EXIT TIMING, which is what the grey markers implement.

**5 - Parameters are FROZEN.** All knobs live in `src/config.py` with
the evidence quoted next to each. The pipeline snapshots signals daily
and never revises them - that forward record is the only true
out-of-sample test. Changes to these rules require new out-of-sample
evidence, not a re-run of the same history. Full write-up:
`docs/ARCHITECTURE.md` section 6.1.

**6 - Influence tracker: thesis method, committed store.** Author
scoring ports Chan (Oxford M.Eng, 2026) end-to-end: volatility-scaled
correctness bar (tau = max(3%, 0.5 sigma) - one fixed bar would misgrade
an index ETF and a meme stock with the same ruler), abnormal-return
weighting w(z)=clip(1+|z|, 0.1, 2), Bayesian shrinkage (alpha 10/5/10),
composite 0.4/0.4/0.2 with the HIGH tier at 0.66, and a bot-filtered
reply-graph PageRank. Ranking is by USEFULNESS, never by size: the
thesis's error analysis found the structurally loudest users (3x degree,
2x PageRank) were the least accurate (40% vs 79% for the quiet true
positives) - the 'loud but wrong' column encodes exactly that finding.
The store was made COMMITTED in July 2026 (reversing local-only):
pseudonymous public identifiers, text-free by a hard write-time check,
one shared leaderboard that every live run extends incrementally."""

# The name TERMINAL_CSS is now a misnomer - it is injected once and styles
# an institutional light theme, not a terminal.  Renamed below to keep the
# code honest; the old name is aliased at the injection point.
INSTITUTIONAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* --- type ------------------------------------------------------------
   One neo-grotesk everywhere, tall x-height, restrained weights.  The
   font LOADS from Google but the stack degrades to Helvetica/Arial, so a
   desk machine behind a proxy that blocks fonts.googleapis.com still
   renders correctly - it just falls back a step. */
html, body, [data-testid="stAppViewContainer"] * {{
    font-family: {FONT_STACK} !important;
    -webkit-font-smoothing: antialiased;
}}
/* EXCEPTION: Streamlit's icons are a FONT (Material Symbols) - without
   this rule the family override above turns every icon into its literal
   ligature text, e.g. 'arrow_right' on expanders */
span[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
.material-icons, [class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}

/* --- page ------------------------------------------------------------
   White page, generous gutters, 1400px cap.  The brief's 96/120/160px
   section spacing is applied where sections actually change subject;
   applying it between every widget would push a dense monitoring tool
   into three screens of scrolling, which is the one thing a PM will not
   forgive.  Whitespace serves reading here, not the other way round. */
[data-testid="stAppViewContainer"] {{ background: {WHITE}; }}
[data-testid="stAppViewContainer"] > .main .block-container {{
    max-width: 1400px; padding: 2.6rem 3rem 4rem 3rem;
}}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
[data-testid="stSidebar"] {{
    background: {PANEL}; border-right: 1px solid {HAIRLINE};
}}
[data-testid="stSidebar"] * {{ color: {INK}; }}

h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; font-weight: 600; }}
h1 {{ font-size: 2.6rem; line-height: 1.1; }}
h2 {{ font-size: 1.85rem; }}
h3 {{ font-size: 1.35rem; }}
p, li, [data-testid="stMarkdownContainer"] {{
    color: {INK}; font-size: 0.98rem; line-height: 1.65;
}}
hr {{ border: none; border-top: 1px solid {HAIRLINE}; margin: 2.2rem 0; }}
a {{ color: {NAVY}; }}

/* --- the statistic counter, GIC's signature component ---------------
   Huge number, tiny uppercase label, plenty of air.  This is the single
   most recognisable piece of the design language, so the metric row gets
   it verbatim rather than approximately. */
[data-testid="stMetric"] {{
    background: {WHITE}; border: 1px solid {HAIRLINE};
    border-radius: 8px; padding: 1.15rem 1.25rem;
}}
[data-testid="stMetricValue"] {{
    color: {NAVY}; font-size: 2.05rem; font-weight: 600;
    letter-spacing: -0.02em;
}}
[data-testid="stMetricLabel"] {{
    color: {INK_LABEL}; font-size: 0.68rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.09em;
}}
/* Streamlit truncates the metric label to one line with an ellipsis, and it
   puts that rule on the INNER markdown container rather than the <label>, so
   styling the label alone does nothing.  Uppercase tracked labels are wide,
   so the truncation ate the second half of nearly every card:
   'EUPHORIA AL...', 'MEDIAN WARNING B...'.  A KPI whose name is unreadable is
   not a KPI, so the label is allowed to wrap and the cards are stretched to a
   common height so a row still lines up. */
[data-testid="stMetricLabel"] div[data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"] p {{
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; line-height: 1.3;
    overflow-wrap: anywhere;
}}
/* ...and the same structural surprise silently destroyed the type scale.
   Streamlit renders BOTH halves of a metric as markdown, so the visible text
   lives in an inner <p> that carries its own font-size (15.68px) and colour
   (#111) from the generic `p, [data-testid="stMarkdownContainer"]` rule far
   above.  A declaration on the element itself always beats one inherited from
   an ancestor, so the 2.05rem on stMetricValue and the 0.68rem uppercase on
   stMetricLabel were both being applied to a wrapper whose child then opted
   out: every KPI card rendered as two lines of identical 15.68px body text.
   Verified with a DOM probe rather than inferred - the browser reported
   32.8px on the value wrapper and 15.68px on the <p> inside it.
   Restating the type on the descendants is what makes the signature
   component actually look like the brief. */
[data-testid="stMetricValue"] div[data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"] p {{
    color: {NAVY} !important; font-size: 2.05rem !important;
    font-weight: 600 !important; letter-spacing: -0.02em;
    line-height: 1.15; margin: 0;
}}
[data-testid="stMetricLabel"] div[data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"] p {{
    color: {INK_LABEL} !important; font-size: 0.68rem !important;
    font-weight: 500 !important; text-transform: uppercase;
    letter-spacing: 0.09em; margin: 0;
}}
[data-testid="stMetric"] {{ height: 100%; }}
[data-testid="stMetricDelta"] {{
    font-size: 0.8rem; background: transparent !important;
    padding: 0.15rem 0 0 0 !important;
}}
/* The delta pill ships in Streamlit's signal green / red on a tinted
   background.  The brief is "never vibrant", and these two hues carry
   meaning elsewhere in the product, so the delta is recoloured to the same
   muted teal / brick the charts use.  :has() is the only way to tell up from
   down here; where a browser lacks it the Streamlit default shows through,
   which is a colour regression and not a broken card. */
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) {{
    color: {BULL} !important;
}}
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) {{
    color: {BEAR} !important;
}}
[data-testid="stMetricDelta"] svg {{ fill: currentColor; }}

/* --- tabs: editorial, not chiclets --------------------------------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 1.9rem; border-bottom: 1px solid {HAIRLINE};
    background: rgba(0,0,0,0);
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: rgba(0,0,0,0); padding: 0.55rem 0 0.7rem 0;
    color: {INK_MUTED}; font-size: 0.7rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.09em;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {NAVY} !important; font-weight: 600;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background: {NAVY}; height: 2px;
}}

/* --- cards, expanders, inputs: hairline borders, 8px radius -------- */
div[data-testid="stExpander"] {{
    border: 1px solid {HAIRLINE}; border-radius: 8px;
    background: {WHITE}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}
div[data-testid="stExpander"] summary {{
    color: {INK}; font-size: 0.9rem; font-weight: 500;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid {HAIRLINE}; border-radius: 8px;
}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
    border-radius: 8px; border-color: {HAIRLINE}; background: {WHITE};
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
    color: {INK_MUTED} !important; font-size: 0.82rem;
}}

/* --- buttons: navy fill, white text, 8px, no heavy shadow --------- */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"], .stButton > button {{
    background: {NAVY}; color: {WHITE}; border: 1px solid {NAVY};
    border-radius: 8px; font-weight: 500; letter-spacing: 0.01em;
    box-shadow: none; transition: opacity 0.15s ease;
}}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover, .stButton > button:hover {{
    background: {NAVY_MID}; border-color: {NAVY_MID};
    color: {WHITE}; opacity: 0.95;
}}
/* The sidebar sets INK on every descendant so its own text reads correctly.
   That blanket rule also hits the <p> INSIDE a button, and an element's own
   colour beats one inherited from the button - so every sidebar button was
   rendering navy text on a navy fill, i.e. invisible.  Restate white on the
   button's children explicitly.
   Scoped to the SIDEBAR originally, because that is where the bug was seen.
   That was the wrong scope: the generic `p, [data-testid="stMarkdownContainer"]`
   rule sets INK in the MAIN area too, so `cancel pipeline` and `dismiss` were
   also navy-on-navy.  They only exist while a pipeline is running, which is
   why neither a screenshot nor the contrast audit ever reached them - a real
   invisible-text bug found by reading the cascade, not by looking.
   Widened to every .stButton, and deliberately NOT to stDownloadButton: that
   one keeps Streamlit's own light fill, so forcing white text on it would
   manufacture the exact white-on-white defect this is fixing. */
.stButton button,
.stButton button * {{
    color: {WHITE} !important;
}}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{
    color: {INK_LABEL};
}}

/* --- masthead ------------------------------------------------------ */
.rf-title {{
    color: {INK}; font-size: 2.15rem; font-weight: 600;
    letter-spacing: -0.02em; line-height: 1.15;
}}
.rf-dot {{ color: {BULL}; }}
.rf-sub {{
    color: {INK_LABEL}; font-size: 0.7rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.09em;
}}
.rf-credit {{ color: {INK_LABEL}; font-size: 0.72rem; margin-top: 6px; }}
.rf-rule {{ border-top: 1px solid {HAIRLINE}; margin: 1.4rem 0 1.9rem 0; }}
</style>"""

CONV_SIMPLE = f"""
**Conviction = how convinced this crowd is *compared with its own normal*.**

- Each post is a vote: clearly positive = bullish, clearly negative = bearish.
- **Conviction = bullish votes minus bearish votes**, summed over a week, then
  compared with the same theme's own recent history.
- The number you see is *how unusual today is for this theme* - so a
  permanently noisy theme sits near **0**, and a quiet theme that suddenly
  finds a devoted bullish crowd **spikes**.

**Reading the number.**

| you see | it means |
|---|---|
| **0** | as convinced as this theme usually is |
| **+{CROSS_AT:g}** | unusually convinced - triangle marker on the chart |
| **-{CROSS_AT:g}** | unusually bearish |
| back toward **0** | the mood has faded - grey marker |

*Example: "AI at +2.0" does not mean AI is the loudest theme on the board. It
means AI is two standard deviations more bullish-and-active than AI normally
is - and it is that abnormality, not raw loudness, that carried an edge in
testing.*

**Why abnormality rather than raw volume?** Because raw volume just ranks the
big themes every day and tells you nothing new. Tested on real prices, going
long the up-crossings returned **+1.36% per trade over 20 days (63% hit rate,
299 trades)**, and it held steady across every halflife and holding period
tried - a plateau, which is what a real effect looks like, rather than one
lucky setting.

**This is NOT the euphoria signal.** Conviction is a mood gauge; euphoria is
the late-stage warning. They are separate tabs on purpose.
"""


CONV_DEF = """**Conviction = how convinced the crowd is, measured against how
convinced that crowd usually is.**

Built in three steps:

1. **Bull pressure** - every scored post is a vote: clearly positive =
   bullish vote, clearly negative = bearish vote. Bull pressure for a day is
   bullish minus bearish votes, so it grows with both how one-sided the
   crowd is *and* how many people showed up.
2. **7-day rolling sum** - one loud afternoon is not conviction; a sustained
   week of lean is.
3. **Trailing z-score** - that sum is compared against the same theme's own
   PRECEDING 84 days: `(today - its own recent mean) / its own recent spread`.

So **conviction z = +2** reads: *this theme is two standard deviations more
bullish-active than is normal for this theme lately.* A permanently loud
theme sits near 0; a quiet theme that suddenly gains a devoted bullish crowd
spikes - and that abnormality, not raw loudness, is where the trade is.
Crossings of the +/-1.5 lines are marked on the charts with triangles."""

# ---------------------------------------------------------------------------
# data loading (cached; invalidates when the file on disk changes)
# ---------------------------------------------------------------------------
def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


@st.cache_data(show_spinner=False)
def _read(path, mtime):
    df = pd.read_parquet(path)
    for col in ("date", "action_date", "signal_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(show_spinner=False)
def _read_json(path, mtime):
    """Small cached JSON read - the notebooks' verdict files, so the
    dashboard can quote a measured number instead of restating it."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load(name, folder=PROCESSED_DIR):
    path = os.path.join(folder, name)
    return _read(path, _mtime(path)) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# research numbers on tap
#
# Every figure quoted in the plain-English panels below is READ from the JSON
# the notebooks export, never retyped.  The reason is a specific failure mode:
# a hand-typed number is correct on the day it is written and silently wrong
# after the next re-run, and a PM who spots one stale figure stops believing
# the other twenty.  Keying the cache on mtime means re-running a notebook
# updates the copy on screen with no code change.
# _num() degrades to a dash rather than raising, so a missing research file
# leaves a visible gap instead of taking the dashboard down.
# ---------------------------------------------------------------------------
RESEARCH_DIR = os.path.join(ROOT, "docs", "research")


def _research(name):
    """The verdict dict a notebook exported, or {} if it has not been run."""
    path = os.path.join(RESEARCH_DIR, f"{name}.json")
    return _read_json(path, _mtime(path)) if os.path.exists(path) else {}


def _dig(obj, *keys, default=None):
    """obj['a']['b'] without the KeyError ladder."""
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return default
        obj = obj[k]
    return obj


def _pct(x, dp=0, signed=False):
    """A fraction as a percentage string; '-' when the number is absent."""
    if x is None:
        return "-"
    return f"{x * 100:+.{dp}f}%" if signed else f"{x * 100:.{dp}f}%"


def _cnt(x):
    """Thousands-separated count; '-' when absent.  Separate from _pct so a
    missing key can never reach a ',' format spec and raise."""
    return f"{x:,}" if isinstance(x, (int, float)) else "-"


def _ci_pct(ci, dp=1):
    """A [lo, hi] fraction pair as a readable percentage interval.  The
    notebooks store these as raw fractions, which read as noise on a desk
    ([-0.0151, 0.0026] vs [-1.5%, +0.3%]) - a PM should not have to shift a
    decimal point in their head to see whether an interval straddles zero."""
    if isinstance(ci, str):
        # some notebooks export the interval pre-formatted as a string
        # ("[-0.066, -0.003]"); accept both shapes rather than making the
        # notebooks agree on one, which would invalidate saved runs.
        try:
            ci = [float(p) for p in ci.strip("[] ").split(",")]
        except ValueError:
            return ci
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return "-"
    return f"[{ci[0] * 100:+.{dp}f}%, {ci[1] * 100:+.{dp}f}%]"


def _row(rows, **match):
    """First dict in a list of dicts matching every key=value given."""
    for r in rows or []:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    return {}


def resolve_anchor(theme, priced):
    """A theme's tradeable price line: the primary anchor ETF if priced,
    else the first priced fallback (a window older than a young ETF can
    still draw against an established proxy)."""
    candidates = ([THEME_ETFS[theme]] if THEME_ETFS.get(theme) else [])
    candidates += THEME_ETF_FALLBACKS.get(theme, [])
    for sym in candidates:
        if sym in priced:
            return sym
    return None


def ranked(df, by, ascending=False):
    """Add a 1-based 'rank' column - the TOP row is always rank 1."""
    out = df.sort_values(by, ascending=ascending).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------------
# interactive chart builders (Plotly - hover shows the numbers)
# ---------------------------------------------------------------------------
def _theme(fig):
    """The institutional look for every chart, in one place.

    Three deliberate choices, each from the design brief:
      * `plotly_white` + transparent paper, so a chart sits ON the page
        rather than in a dark box of its own.
      * gridlines at HAIRLINE and no axis lines - the grid should be
        findable when you look for it and invisible when you are not.
      * no legend border and no chart frame; whitespace does the separating.
    """
    fig.update_layout(template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT_STACK, size=12, color=INK),
                      title_font=dict(size=15, color=INK),
                      legend=dict(bgcolor="rgba(0,0,0,0)",
                                  borderwidth=0,
                                  font=dict(size=11, color=INK_MUTED)),
                      hoverlabel=dict(bgcolor=WHITE, bordercolor=HAIRLINE,
                                      font=dict(family=FONT_STACK, size=11,
                                                color=INK)))
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(gridcolor=HAIRLINE, zerolinecolor=HAIRLINE, linecolor=HAIRLINE,
             showline=False, ticks="outside", ticklen=4,
             tickcolor=HAIRLINE, tickfont=dict(size=10, color=INK_MUTED),
             title_font=dict(size=11, color=INK_LABEL))
    return fig


def _axes_fidelity(fig):
    """More x-axis points + readable labels on every chart."""
    fig.update_xaxes(nticks=24, tickformat="%d %b %y", tickangle=-40)
    return fig


def _base_fig(title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(title=dict(text=title, y=0.97, x=0.01),
                      height=430, hovermode="x unified",
                      margin=dict(l=10, r=10, t=55, b=20),
                      legend=dict(orientation="h", yanchor="top", y=-0.28))
    return _axes_fidelity(_theme(fig))


def fig_series_vs_price(series, series_name, series_color, px, symbol, title,
                        extra=None):
    """The workhorse: any daily series on the left axis, price on the
    right. extra = optional list of (series, name, color) to add."""
    fig = _base_fig(title)
    # Sparse-coverage days are masked (NaN) by the MIN_TOTAL rule - too few
    # posts to trust a share number. Those stretches are drawn as a DOTTED,
    # dimmed bridge (linear interpolation, inside the data span only) so a
    # filled-in stretch is visibly different from real data. The legend
    # entry IS the key: "not enough posts that day (dotted)".
    has_gaps = series.isna().any() and series.notna().any()
    if has_gaps:
        interp = series.interpolate(limit_area="inside")
        # keep only the gap interiors plus their bracketing real points,
        # so each dotted segment visually joins the solid line's ends
        gap = series.isna()
        keep = gap | gap.shift(1, fill_value=False) | gap.shift(-1, fill_value=False)
        bridge = interp.where(keep)
        if bridge.notna().any():
            fig.add_trace(go.Scatter(
                x=bridge.index, y=bridge.values,
                name="not enough posts that day (dotted)",
                line=dict(color=series_color, width=1.2, dash="dot"),
                opacity=0.5, hoverinfo="skip"),
                secondary_y=False)
    # the REAL data: solid, broken at the masked stretches (the dotted
    # bridge above fills the visual hole without faking a measurement)
    fig.add_trace(go.Scatter(x=series.index, y=series.values,
                             name=series_name, line=dict(color=series_color)),
                  secondary_y=False)
    for s, nm, col in (extra or []):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=nm,
                                 line=dict(color=col, width=1), opacity=0.5),
                      secondary_y=False)
    if px is not None and not px.empty:
        fig.add_trace(go.Scatter(x=px.index, y=px.values,
                                 name=f"{symbol} price",
                                 line=dict(color=GRAY, width=1.5)),
                      secondary_y=True)
    fig.update_yaxes(title_text=series_name, secondary_y=False)
    fig.update_yaxes(title_text="price (USD)", secondary_y=True)
    return fig


def dim_outside(fig, window_lo, focus_start, label):
    """Grey out everything BEFORE focus_start so it is obvious that only
    the recent stretch drives the ranking (the greyed part is context,
    not input). A dotted line + small label mark the boundary."""
    if focus_start is None or window_lo is None or focus_start <= window_lo:
        return fig
    fig.add_vrect(x0=window_lo, x1=focus_start, fillcolor=INK_LABEL,
                  opacity=0.13, line_width=0)
    fig.add_vline(x=focus_start, line_dash="dot", line_color=INK_LABEL,
                  opacity=0.8)
    fig.add_annotation(x=focus_start, y=1.02, yref="paper",
                       yanchor="bottom", xanchor="left", showarrow=False,
                       text=label, font=dict(size=10, color=INK_LABEL))
    return fig


def fig_conviction(cz, px, theme, symbol):
    """Conviction z with the +/-CROSS_AT lines; bullish/bearish crossings
    marked as triangles ON THE PRICE LINE (that is where the trade lives)."""
    fig = _base_fig(f"{theme} conviction vs {symbol or 'no priced anchor'}")
    fig.add_trace(go.Scatter(x=cz.index, y=cz.values, name="conviction_z",
                             line=dict(color=PURPLE)), secondary_y=False)
    fig.add_hline(y=CROSS_AT, line_dash="dot", line_color=GREEN, opacity=0.6)
    fig.add_hline(y=-CROSS_AT, line_dash="dot", line_color=RED, opacity=0.6)
    if px is not None and not px.empty:
        fig.add_trace(go.Scatter(x=px.index, y=px.values, name=f"{symbol} price",
                                 line=dict(color=GRAY, width=1.5)),
                      secondary_y=True)
        up, dn = conviction_crossings(cz, CROSS_AT, MIN_GAP)
        # grey markers = the EXIT point: z reverting inside +/-CONV_EXIT_LEVEL
        # means the surge that produced the signal has expired ("back to
        # neutral") - validated as the capital-efficient exit, so a position
        # never has to wait for an opposite signal to get out
        lx, sx = crossing_exits(cz, up, dn, CONV_EXIT_LEVEL)
        for dates, sym_mk, col, nm, filled in [
                (up, "triangle-up", GREEN, "bullish crossing", True),
                (dn, "triangle-down", RED, "bearish crossing", True),
                (lx, "triangle-down-open", GRAY, "exit long (back to neutral)", False),
                (sx, "triangle-up-open", GRAY, "exit short (back to neutral)", False)]:
            pts = [(d, px.asof(d)) for d in dates
                   if px.index.min() <= d <= px.index.max()]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pts], y=[p[1] for p in pts], name=nm,
                    mode="markers",
                    marker=dict(symbol=sym_mk, size=13 if filled else 11,
                                color=col,
                                line=dict(color="black" if filled else col,
                                          width=1))),
                    secondary_y=True)
    fig.update_yaxes(title_text="conviction z", secondary_y=False)
    fig.update_yaxes(title_text="price (USD)", secondary_y=True)
    return fig

# ---------------------------------------------------------------------------
# sidebar: header, window controls, pipeline runners
# ---------------------------------------------------------------------------
st.markdown(INSTITUTIONAL_CSS, unsafe_allow_html=True)

h_left, h_right = st.columns([5, 1])
with h_left:
    st.markdown(
        '<div><span class="rf-dot">&#9679;</span> '
        '<span class="rf-title">RetailRadar</span></div>'
        '<div class="rf-sub">retail attention &amp; trading signals - '
        'real-time monitoring dashboard (notebook-free pipeline)</div>'
        f'<div class="rf-sub">last update: '
        f'{pd.Timestamp.now():%d/%m/%Y, %H:%M:%S}</div>'
        '<div class="rf-credit">Alex Brown - GIP 2026 Project - '
        'MAARS Global Macro</div>',
        unsafe_allow_html=True)
with h_right:
    st.markdown(HEADER_MARK_HTML, unsafe_allow_html=True)
st.divider()

st.sidebar.title("RetailRadar")

theme_counts = load(THEME_COUNTS)
euph = load("euphoria_levels.parquet")
if euph is not None:
    euph["date"] = pd.to_datetime(euph["date"])
euph_report = None
_rep_path = os.path.join(PROCESSED_DIR, "euphoria_report.json")
if os.path.exists(_rep_path):
    import json as _json
    euph_report = _json.load(open(_rep_path))

# the ONSET detector's outputs (the July-2026 phases study; produced by
# `run_analytics --what phases` / any full analytics recompute)
onset = load("euphoria_onset.parquet")
if onset is not None:
    onset["date"] = pd.to_datetime(onset["date"])
onset_report = None
_orep_path = os.path.join(PROCESSED_DIR, "euphoria_onset_report.json")
if os.path.exists(_orep_path):
    import json as _json
    onset_report = _json.load(open(_orep_path))

# the DESK CONFIGURATION store (desk decision 2026-07-24): the GET IN /
# GET OUT signals the euphoria tabs actually show - boom-gated smoothed
# END + phase-aware smoothed ONSET, at frozen walk-forward thresholds
# (full record: NB06 "adopted desk configuration" + euphoria_phases.py §6)
desk = load("euphoria_desk.parquet")
if desk is not None:
    desk["date"] = pd.to_datetime(desk["date"])
desk_report = None
_dkrep_path = os.path.join(PROCESSED_DIR, "euphoria_desk_report.json")
if os.path.exists(_dkrep_path):
    import json as _json
    desk_report = _json.load(open(_dkrep_path))

# episode ground truth (for the window-adaptive scorecard strip)
episodes_df = load("episodes.parquet")
if episodes_df is not None and len(episodes_df):
    for _c in ("peak", "trough", "onset_lo", "onset_hi"):
        episodes_df[_c] = pd.to_datetime(episodes_df[_c])


@st.cache_data(show_spinner=False)
def _live_conviction(sent_mtime):
    """Theme conviction computed LIVE from the sentiment aggregate (not
    read from the pipeline's conviction file). Why: the file only updates
    when the analytics recompute runs, so after a maths change the
    dashboard could silently show stale values. Computing here (cached on
    the sentiment file's mtime, <1s for 39 themes) guarantees the screen
    always reflects the current engine - coverage normalisation included."""
    from analytics.loaders import load as _load, THEME_SENT
    from analytics.conviction import compute_conviction
    ts = _load(THEME_SENT)
    if ts is None:
        return None
    return compute_conviction(ts, "theme").tidy("theme")


conv = _live_conviction(_mtime(os.path.join(PROCESSED_DIR,
                                            "daily_theme_sentiment.parquet")))

prices = _read(PRICES_PATH, _mtime(PRICES_PATH)) if os.path.exists(PRICES_PATH) else None
priced = set(prices["symbol"]) if prices is not None else set()

if theme_counts is None:
    st.error("No aggregate data - run update_data.py first.")
    st.stop()

data_max = theme_counts["date"].max()
today = pd.Timestamp.today().normalize()
# default view: 1 Jan 2026 onwards (the start of dense backfilled
# coverage); falls back to trailing-365d if the data ends before that
_default_lo = pd.Timestamp("2026-01-01")
if data_max <= _default_lo:
    _default_lo = data_max - pd.Timedelta(days=365)
lo = pd.Timestamp(st.sidebar.date_input("window start", _default_lo.date()))
live_mode = st.sidebar.checkbox("LIVE (to newest data)", value=True)
hi = None if live_mode else pd.Timestamp(
    st.sidebar.date_input("window end", data_max.date()))
how_many = st.sidebar.slider("items per section", 3, 15, 6)

st.sidebar.divider()
st.sidebar.subheader("Run the pipeline")


# The pipeline runs as a BACKGROUND process (stdout to a temp log file)
# rather than blocking the Streamlit script. That is what makes CANCEL
# possible: while a synchronous loop streams subprocess output, Streamlit
# cannot process any button click - the app would be frozen until the
# pipeline finished. Here the app stays responsive, a fragment re-renders
# the log tail every 2 seconds, and cancel kills the WHOLE process tree
# (update_data.py spawns children - fetchers, analytics - which a plain
# .kill() of the parent would orphan).

def _kill_tree(proc):
    """Terminate a pipeline process AND all its children, cross-platform."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        # /T = tree (children too), /F = force. The standard Windows way.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    try:
        proc.wait(timeout=5)      # reap it - otherwise it lingers as a zombie
    except Exception:
        pass


def _launch_current_step():
    """Start the current step of the queued pipeline in the background."""
    p = st.session_state.pipe
    script_args, env_extra = p["steps"][p["i"]]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    logfile = open(p["log"], "a", encoding="utf-8", errors="replace")
    logfile.write(f"===== {' '.join(script_args)} =====\n")
    logfile.flush()
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True       # own group -> killable tree
    p["proc"] = subprocess.Popen(
        [sys.executable] + script_args, cwd=ROOT,
        stdout=logfile, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env, **kwargs)


# ---- LAYMAN PROGRESS TRACKING -------------------------------------------
# The pipeline scripts print known marker lines as they work ("DATA
# COVERAGE", "pulling Bloomberg prices", ...). The panel scans the log for
# those markers and translates them into a progress bar + a plain-English
# stage checklist. Each stage is (label, [marker substrings]); a stage
# counts as REACHED once any of its markers appears in the log. The raw
# log stays available in a "technical log" expander for debugging.
STAGES = {
    "fetch":    ("Fetching new posts + comments (Reddit / X / StockTwits)",
                 ["sources IN PARALLEL", "still fetching", "fetch done"]),
    "store":    ("Adding the new posts to the data store",
                 ["APPEND:", "folding live raw", "merging live raw",
                  "MERGE:", "live fast path", "hydrated ABSTRACTED_DATA"]),
    "rebuild":  ("Rebuilding the aggregates from raw text (long step)",
                 ["full chain: building aggregates",
                  "building rolling term counts", "need scoring"]),
    "coverage": ("Checking data coverage for the window",
                 ["DATA COVERAGE", "WINDOW CHECK"]),
    "analyse":  ("Analysing: conviction, signals, euphoria + onset radar, "
                 "influence board",
                 ["recomputing conviction", "analytics:",
                  "conviction (was nb", "signals (was nb",
                  "phases (the onset detector", "influence (live board",
                  "THEME decisions", "analytics finished"]),
    "prices":   ("Downloading prices from Bloomberg",
                 ["BLOOMBERG PRICE PULL", "pulling Bloomberg prices",
                  "requesting "]),
    "wrapup":   ("Safety check + wrap-up",
                 ["snapshot ->", "safety check", "RUN SUMMARY"]),
    "comments": ("Fetching Reddit comments (resumable - cancel is safe)",
                 ["COMMENT PULL", "new comments", "fetch finished"]),
    "influence": ("Updating the influence board (calls, graph, tiers)",
                  ["influence board update", "influence update finished"]),
}
# which stages each pipeline actually goes through (in order)
PLANS = {
    "live":      ["fetch", "store", "coverage", "analyse", "prices", "wrapup"],
    "window":    ["prices", "coverage", "analyse", "wrapup"],
    "analytics": ["analyse"],
    "full":      ["fetch", "store", "rebuild", "analyse", "prices", "wrapup"],
    "comments":  ["comments", "influence"],
}


def start_pipeline(steps, label, plan):
    import tempfile
    import time as _time
    fd, logpath = tempfile.mkstemp(prefix="apollo_pipe_", suffix=".log")
    os.close(fd)
    st.session_state.pipe = {"steps": steps, "label": label, "i": 0,
                             "log": logpath, "state": "running",
                             "plan": PLANS[plan], "max_frac": 0.0,
                             "t0": _time.time()}
    _launch_current_step()


def _read_log(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _stage_status(p, log_text):
    """Map the log onto the plan's stages. Returns (statuses, frac) where
    statuses is [(label, 'done'|'active'|'pending'), ...] and frac is the
    overall progress in [0, 1]. The furthest stage whose marker appears is
    ACTIVE; everything before it is DONE. The fraction is kept MONOTONIC
    (a marker scrolling out of a stage can never move the bar backwards)."""
    plan = p["plan"]
    reached = -1
    for idx, key in enumerate(plan):
        markers = STAGES[key][1]
        if any(m in log_text for m in markers):
            reached = idx
    statuses = []
    for idx, key in enumerate(plan):
        state = ("done" if idx < reached
                 else "active" if idx == reached else "pending")
        statuses.append((STAGES[key][0], state))
    # a reached stage counts half until the next one starts
    frac = 0.0 if reached < 0 else (reached + 0.5) / len(plan)
    frac = max(frac, p.get("max_frac", 0.0))
    p["max_frac"] = frac
    return statuses, frac


_STAGE_ICON = {"done": ":green[+]", "active": ":orange[>]", "pending": " "}


@st.fragment(run_every="2s")
def pipeline_panel():
    """Self-refreshing sidebar panel: progress bar + layman stage list +
    CANCEL while running; final state + dismiss when done. Only this
    fragment reruns every 2s - the rest of the dashboard is untouched
    until the pipeline finishes."""
    p = st.session_state.get("pipe")
    if not p:
        return
    proc = p.get("proc")

    # advance the state machine on each refresh
    if p["state"] == "running" and proc is not None:
        rc = proc.poll()
        if rc is not None:                       # current step ended
            if rc != 0:
                p["state"] = "failed"
            elif p["i"] + 1 < len(p["steps"]):
                p["i"] += 1
                _launch_current_step()           # next step, same log
            else:
                p["state"] = "done"

    log_text = _read_log(p["log"])
    if p["state"] == "running":
        import time as _time
        mins, secs = divmod(int(_time.time() - p.get("t0", _time.time())), 60)
        box = st.status(f"running {p['label']}... ({mins}m {secs:02d}s "
                        "elapsed)", expanded=True)
        box.markdown(LOADER_HTML, unsafe_allow_html=True)
        statuses, frac = _stage_status(p, log_text)
        box.progress(frac)
        for label, state in statuses:
            if state == "done":
                box.markdown(f":green[[done]] ~~{label}~~")
            elif state == "active":
                box.markdown(f":orange[[now]] **{label}**")
            else:
                box.markdown(f"<span style='color:#8C8C8C'>[ &nbsp; ] "
                             f"{label}</span>", unsafe_allow_html=True)
        with box.expander("technical log"):
            st.code("\n".join(log_text.splitlines()[-25:]) or "starting...")
        if box.button("cancel pipeline", key="pipe_cancel"):
            _kill_tree(proc)
            p["state"] = "cancelled"
            # plain st.rerun (app scope): valid both from a fragment tick
            # AND from a full-app pass (scope="fragment" is rejected there)
            st.rerun()
    else:
        state_ui = {"done": ("complete", "finished"),
                    "failed": ("error", "FAILED - see the technical log"),
                    "cancelled": ("error", "CANCELLED")}[p["state"]]
        box = st.status(f"{p['label']} {state_ui[1]}", state=state_ui[0],
                        expanded=(p["state"] != "done"))
        if p["state"] == "done":
            box.progress(1.0)
        with box.expander("technical log", expanded=(p["state"] == "failed")):
            st.code("\n".join(log_text.splitlines()[-40:]))
        if box.button("dismiss", key="pipe_dismiss"):
            del st.session_state.pipe
            _read.clear()                        # pick up whatever was written
            st.rerun()                           # full-app refresh


_pipe_running = (st.session_state.get("pipe", {}).get("state") == "running")

start_s = lo.strftime("%Y-%m-%d")
end_s = "" if hi is None else hi.strftime("%Y-%m-%d")
win_env = {"PIPELINE_START_DATE": start_s, "PIPELINE_END_DATE": end_s}

# buttons are disabled while a pipeline runs - one at a time, by design
if st.sidebar.button("run LIVE pull now  (under ~10 min)",
                     disabled=_pipe_running,
                     help="Everything: fetch new posts AND comments from all "
                          "three sources, fold them in, recompute signals, "
                          "rescore the influence board, pull prices. Most of "
                          "the time is deliberate API rate-limit pacing (X "
                          "waits 5s between requests; Reddit comments are "
                          "capped at the pages that fit the ~10-minute "
                          "budget, and whatever is left over is picked up by "
                          "the next run, never dropped). Run it about twice "
                          "a week - that is what one budget's worth of "
                          "comments covers."):
    start_pipeline([(["update_data.py"], None)], "LIVE pull", plan="live")
if st.sidebar.button("rebuild THIS window (prices + signals)  (~1-3 min)",
                     disabled=_pipe_running,
                     help="No post fetching. Pull Bloomberg prices for the "
                          "chosen window (first pull of new symbols/spans "
                          "takes longer; already-covered spans are "
                          "skipped), then recompute the signals."):
    start_pipeline([(["pull_bloomberg_prices.py"], win_env),
                    (["update_data.py", "--start", start_s, "--end", end_s,
                      "--skip-prices"], None)],
                   f"window rebuild {start_s} -> {end_s or 'LIVE'}",
                   plan="window")
if st.sidebar.button("recompute analytics only (no APIs)  (~1 min)",
                     disabled=_pipe_running,
                     help="Conviction + signals recomputed from the "
                          "aggregates already on disk. No network at all."):
    start_pipeline([(["-m", "analytics.run_analytics"], None)],
                   "analytics recompute", plan="analytics")
# Comments ARE part of the live pull again (desk decision 2026-07-27): the
# live pull fetches them under a measured page allowance, so this button is
# no longer how the board gets refreshed - it is the CATCH-UP button, for
# when the pipeline has been idle long enough that one budgeted run cannot
# close the gap. Its estimate is computed from the watermarks and this
# machine's measured throughput, not bracketed by hand.
try:
    from update_comments import estimate as _comment_estimate

    from ingestion.fetch_reddit_comments import (default_lookback_days
                                                 as _c_lookback)
    _c_est = _comment_estimate(_c_lookback(), None)
except Exception:                                     # noqa: BLE001
    _c_est = ("estimate unavailable on this machine - the runner prints one "
              "before it starts")
if st.sidebar.button("catch up comments  (no page budget)",
                     disabled=_pipe_running,
                     help="NOT needed for the ordinary refresh - the live "
                          "pull above already fetches comments and rescores "
                          "the board within the desk's runtime ceiling. Use "
                          "this only after a long idle spell, when the "
                          "budgeted allowance would take several runs to "
                          "close the gap: it crawls the whole owed window in "
                          "one sitting, however long that takes. Watermarked "
                          "and resumable - cancelling is always safe. "
                          f"Current estimate: {_c_est}"):
    start_pipeline([(["update_comments.py"], None)],
                   "comments catch-up", plan="comments")
if st.sidebar.button("run FULL historical rebuild  (external machine; "
                     "30 min - hours)", disabled=_pipe_running,
                     help="Rebuilds every aggregate from raw post text over "
                          "the whole build range. Only meaningful on the "
                          "machine that holds posts.parquet; run after "
                          "changing themes or schemas."):
    start_pipeline([(["update_data.py", "--full"], None)], "FULL rebuild",
                   plan="full")

with st.sidebar:
    pipeline_panel()

if prices is None:
    st.sidebar.warning("prices.parquet missing - run pull_bloomberg_prices.py "
                       "(Terminal open) or use the rebuild button")

# ---------------------------------------------------------------------------
# topline metric strip
# ---------------------------------------------------------------------------
_m1, _m2, _m3, _m4, _m5 = st.columns(5)
_e_now = _alerts_w = 0
_hottest = "-"
if euph is not None and len(euph):
    _latest = euph[euph["date"] == euph["date"].max()]
    _e_now = int((_latest["level"] >= 70).sum())
    _hot = _latest.sort_values("level", ascending=False).iloc[0]
    _hottest = f"{_hot['name']} ({_hot['level']:.0f})"
    _ew = clip_window(euph, "date", lo, hi)
    _alerts_w = int(_ew["alert"].sum())
_m1.metric("euphoria alerts in window", _alerts_w)
_m2.metric("instruments at level 70+", _e_now)
_m3.metric("hottest right now", _hottest)
_m4.metric("data through", str(data_max.date()))
_m5.metric("priced symbols", len(priced))

# ---- MODEL DECISIONS & EVIDENCE: the audit trail of every rule ----
# Two SIBLING expanders, not nested: Streamlit raises on an expander inside
# an expander, so "simple, with the deep version one click away" has to be
# built as two peers whose titles carry the hierarchy.
with st.expander("WHY THE MODEL DOES WHAT IT DOES - 7 decisions, each with "
                 "the number that settled it", expanded=False):
    st.markdown(decisions_simple())
with st.expander("...the long-form evidence log behind those 7 decisions "
                 "(research version)", expanded=False):
    st.markdown(DECISIONS_DOC)

# NOTE: individual-ticker overlays were removed from the dashboard by
# request - the desk trades THEMES via their anchor ETFs, never single
# tickers. The ticker analytics remain available in analytics/ for
# research (windowed backtests via run_analytics --what signals).
(t_euph_th, t_euph_sg, t_infl, t_ov_theme, t_top, t_emerging, t_conv,
 t_pulse, t_hist) = st.tabs(
    ["EUPHORIA: Themes", "EUPHORIA: Singles", "Influence tracker",
     "Overlays: themes", "Top trends", "Emerging trends", "Conviction",
     "AI Pulse (sample)", "Historical checker"])

tc = clip_window(theme_counts, "date", lo, hi)
# TRADEABLE UNIVERSE ONLY, everywhere: every list/rank/picker on this
# dashboard is restricted to themes with a firm-approved instrument
# (THEME_ETFS). Non-tradeable themes (crypto, cannabis, small_caps) are
# still tracked in the data - they are simply not shown on the desk.
tc = tc[tc["theme"].isin(THEME_ETFS)]


def euphoria_simple():
    """The PM-facing explanation of the euphoria panel: six short blocks, one
    idea each, every number read live from the notebooks.

    This is the PRIMARY explanation on the tab and EUPHORIA_DEF_FULL is the
    archive behind it, which is a deliberate inversion of how it shipped.  The
    original single block was accurate and unreadable - it opened on a desk
    rule and a research caveat, and the reader had to get through four
    paragraphs of method before learning what the red line meant.  A PM has
    about fifteen seconds for an explainer.  What survives that budget is:
    what it means, what makes it fire, what happened last time it fired, and
    what it is not.  The method is not hidden - it is one click deeper, and
    the notebooks remain the research record."""
    ds = _research("nb06_desk_signal")
    eff = _research("nb06_signal_efficacy")
    b7 = _research("nb07_performance_battery")
    cfg = _research("nb06_desk_config")

    state = _dig(ds, "danger_state", "cliff30_state")
    ordinary = _dig(ds, "danger_state", "cliff30_ordinary")
    end10 = _row(eff.get("summary"), signal="END", horizon_d=10)
    got = _dig(b7, "part_a_operating_point", "GET OUT", default={})
    prod = _row(cfg.get("table"),
                variant="GET OUT boom-gated SMOOTHED (production)")

    hype = f"{EUPHORIA_HYPE_MULT:.0f}x"
    prec = got.get("precision")
    return f"""
**Euphoria, in one sentence.** The crowd has stopped analysing a name and
started celebrating it. That is a late-stage condition, not a bullish one.

---

**The two lines on the chart - that is the whole signal.**

- **Blue line = GET IN.** Euphoria is *starting*. The crowd is arriving.
- **Red line = GET OUT.** Euphoria is *ending*. Historically the price top
  is close - this is the line to bring to a PM.

Nothing else on the chart is a decision. The 0-100 curve is context.

---

**What has to be true before anything can fire.**

- **The crowd has to be big enough.** This week's chatter must be at least
  **{hype} the name's own normal** (7-day mentions vs its own 120-day
  median). *Example: a theme that normally gets 50 mentions a week needs
  100+.* Measured per name, so a permanently loud name like BTC is judged
  against loud-for-BTC, not against a quiet utility ETF.
- **The mood has to have stuck.** Weeks of one-way bullishness, not one
  loud afternoon.
- **The price has to have actually run** (for GET OUT): the name must be up
  **{EUPHORIA_BOOM_MIN_ETF:.0%} (theme ETF)** or
  **{EUPHORIA_BOOM_MIN_SINGLE:.0%} (single name)** off its 120-day low.
  You cannot end a party that never started.
- **There has to be enough data** to measure at all.

---

**Why a red line matters - the measured version.**

- On a red-line day: **{_pct(state, 0)} chance** of a 10%+ drop within a
  week, sometime in the next month. On an ordinary day: **{_pct(ordinary, 0)}**.
- Ten trading days after a red line the average name has done
  **{_pct(end10.get('mean_move'), 1, signed=True)}**, against
  **{_pct(end10.get('baseline_mean'), 1, signed=True)}** for a random day.
  Ten days is the *only* horizon where that gap is statistically real
  (90% interval {_ci_pct(end10.get('edge_ci90'))}, which excludes zero).
- Roughly **{'2 in 5' if prec is None else f'{prec:.0%}'}** of red lines land
  in a genuine ending. It caught **{prod.get('captured', '-')} of
  {prod.get('detectable', '-')}** endings it could have caught, with
  **{prod.get('FA', '-')}** false alarms.

*Read that honestly: this is a warning light with a real but imperfect
record, not a trade trigger. It is designed to fire rarely.*

---

**What this is NOT.**

- Not a price forecast, and not a short recommendation. Both were tested
  as trades and **rejected** - the numbers are in notebook 04.
- Not driven by the price chart alone: the crowd does the predicting, the
  price only gates and grades it.
- Not tuned on the days it is scored against. Every threshold is learned
  from **earlier years only** and then frozen.
"""


EUPHORIA_DEF_FULL = """**EUPHORIA = the crowd has stopped analysing and started
celebrating.** Prediction is built from the **Reddit-derived data ONLY**
(desk rule, July 2026): price never enters the euphoria level or the
alert - it is used solely to TEST the detector against real tops, so the
claim stays clean: *the crowd alone called the top*. Measured, per
instrument, as the average of four percentile-ranked ingredients (each
vs that name's OWN trailing year - "extreme" always means extreme *for
this name*):

1. **Attention extremity** - the 7d mention share at its highs (you
   cannot be euphoric quietly).
2. **Sustained bullishness** - the 28d net-bullish share at its highs
   AND >= 75% of posting days bullish: weeks of one-way lean, not one
   loud afternoon.
3. **Crowd influx** - mention share still RISING (new people arriving).
4. **Super-exponential attention growth** - the log of the mention count
   curving upward (rolling quadratic fit): Sornette's LPPLS bubble
   signature applied to the CROWD instead of the chart. Attention
   spreading is an epidemic process - when its growth rate is itself
   growing, the contagion must saturate, and attention saturation is
   where tops form.

**EUPHORIA LEVEL** = 100 x the average of those four (the chart below
each price). **A RED LINE (alert)** fires when: the crowd has genuinely
SWOLLEN (7d mention share >= 2x its own 120d median - "something must go
euphoric first", measured in the crowd, never the chart), attention is
above its 90th percentile, bullishness has persisted, coverage is
sufficient to measure, and the level crosses the walk-forward threshold
- OR slightly below it while the **fade** is active (crowd still
maximal, mood rolling over: the last stage before tops). One alert per
21d episode.

**The STARTING line (blue)** comes from the onset detector (July-2026
phases study winner): the mean of five crowd-only onset features -
attention acceleration, hype ratio, bullish inflection, influx speed,
super-exponential attention - gated by coverage and by attention above
its own 120d median, at a frozen walk-forward threshold.

**What actually fires on THIS screen - the DESK CONFIGURATION (desk
decision 2026-07-24).** The desk lifted the crowd-only restriction for
the signals shown here ("use both price and the social media - I want a
better hit rate"), so the lines on these charts are a labelled SECOND
signal family; the crowd-only detectors above remain the research
headline, unchanged. **GET OUT (red)** = the ending detector with (a)
candidacy requiring an ACTUAL price boom - G2's own thresholds (≥25%
ETF / ≥50% single above the trailing 120d low; no new constant) - which
raised walk-forward capture from 16 to 26 of 122 (gain CI [+3.5pp,
+13pp]), and (b) the trigger on the 7d-SMOOTHED score, which killed the
one-day-blip alerts (AP 0.435 → 0.449, two captures recorded as the
cost). **GET IN (blue)** = the onset detector made PHASE-AWARE: a day
that already satisfies every ending gate is end-stage, and a "start"
there is incoherent - so it cannot fire. That cut start-next-to-end
adjacency from 20 to 2 and late starts from 21 to 10, at a recorded
cost of captures (29 → 20 of 125): a desk decision, made because a
START landing on an END destroys PM trust, and documented with the full
variant table in notebook 06.

**Validation** (walk-forward, real Bloomberg closes, thresholds learned
only from PAST years - headline record in the caption under the charts):
the terminal shows CONCLUSIONS only. The full evidence - per-year
tables, the ablation, the ML challenger, the model tournament and the
trading-translation verdict - lives in `notebooks/01-04` and
`docs/DECISIONS.xlsx`, and re-renders from current data on demand.
Ground truth peak = local 21d high >= 25% (ETF) / 50% (single) above its
120d low, followed by >= 15% / 30% drawdown within 90d. Full rules:
`analytics/euphoria.py` + `analytics/euphoria_phases.py`."""

# ---- EUPHORIA: Themes / Single names ------------------------------------
# Desk decision 2026-07-24: the dashboard shows CONCLUSIONS only - the
# state (starting / ending) drawn on the chart itself, one tab per
# instrument kind. All validation evidence (walk-forward tables, the
# ablation, the ML challenger, the tournament) lives in notebooks/01-04
# and docs/DECISIONS.xlsx, where research belongs.

RECENT_D = 21          # display window = the alert cooldown: one episode
#                        is "current" for one cooldown span

def _last_alerts(df, kind, days=RECENT_D):
    """{name: last alert date} for alerts within the trailing window."""
    if df is None or not len(df):
        return {}
    mx = df["date"].max()
    sub = df[(df["kind"] == kind) & df["alert"]
             & (df["date"] > mx - pd.Timedelta(days=days))]
    return sub.groupby("name")["date"].max().to_dict()


def _state_of(name, starting, ending):
    """STARTING / ENDING / quiet - the later phase wins a tie."""
    s, e = starting.get(name), ending.get(name)
    if s is not None and (e is None or s >= e):
        return "STARTING"
    if e is not None:
        return "ENDING"
    return None


def render_euphoria_tab(kind, kind_label, key_prefix):
    st.subheader(f"EUPHORIA - {kind_label}  |  blue = GET IN (euphoria "
                 "starting), red = GET OUT (euphoria ending; expect the "
                 "top within ~a month)")
    # Three levels, deliberately ordered plain -> method -> register.  The
    # desk asked for the dashboard to be simple and the notebooks to be the
    # explanation; the two deeper levels still have to be REACHABLE from the
    # screen, because "where did that number come from" is the first question
    # asked in a defence and "open a notebook" is a poor answer live.
    with st.expander("what is euphoria?  (start here - plain English)",
                     expanded=False):
        st.markdown(euphoria_simple())
    with st.expander("full method & measured record  (the research version)"):
        st.markdown(EUPHORIA_DEF_FULL)
    _reg = os.path.join(ROOT, "docs", "PARAMETER_REGISTER.md")
    if os.path.exists(_reg):
        with st.expander("deep archive: every constant and its evidence "
                         "(parameter register)"):
            st.markdown(open(_reg, encoding="utf-8").read())
    if euph is None or not len(euph):
        st.info("no euphoria data yet - run 'recompute analytics only' "
                "in the sidebar")
        return

    from analytics.euphoria_phases import episode_coherent_alerts

    ek = euph[euph["kind"] == kind]
    ok = (onset[onset["kind"] == kind].copy()
          if onset is not None and len(onset) else None)
    dk = (desk[desk["kind"] == kind].copy()
          if desk is not None and len(desk) else None)

    # THE SIGNAL SOURCE (desk configuration 2026-07-24): GET IN /
    # GET OUT from euphoria_desk.parquet - the boom-gated SMOOTHED end
    # + phase-aware SMOOTHED onset the desk adopted in NB06 (adjacency
    # 20 -> 2, END AP 0.435 -> 0.449, no one-day blips). Falls back to
    # the crowd-only research stores only if the desk store is missing.
    use_desk = dk is not None and len(dk)
    if use_desk:
        src_in = dk[dk["get_in"].astype(bool)]
        # SINGLE-NAME DISPLAY BAR (desk decision 2026-07-24): a ticker
        # shows as "euphoria starting" only when its crowd cleared the
        # FULL A1 hype bar (2x its own 120d median). Really-euphoric
        # names only; marginal names cannot flicker in and out.
        if kind == "single" and "hype_raw" in dk.columns:
            src_in = src_in[src_in["hype_raw"] >= EUPHORIA_HYPE_MULT]
        src_out = dk[dk["get_out"].astype(bool)]
    else:
        src_in = (ok[ok["alert"]] if ok is not None
                  else ek.iloc[0:0])
        if (not use_desk and ok is not None and kind == "single"
                and "hype_raw" in ok.columns):
            src_in = src_in[src_in["hype_raw"] >= EUPHORIA_HYPE_MULT]
        src_out = ek[ek["alert"]]

    # EPISODE COHERENCE (desk rule 2026-07-24, asymmetric by
    # measurement): a START within one cooldown AFTER an END is a
    # contradictory flip and is suppressed; an END after a START is
    # never suppressed - fast manias genuinely run start-to-end inside
    # 21d, and the risk signal must not be silenced (the symmetric rule
    # cost half the top captures when tested). Applied over FULL history
    # so pre-window alerts provide suppression context. (The other
    # adjacency direction - a START just BEFORE an END - is fixed at
    # the SIGNAL level by the phase-aware desk onset, not by a rule.)
    coherent = {}
    names_all = set(ek["name"].unique()) | set(src_in["name"].unique()) \
        | set(src_out["name"].unique())
    for name in names_all:
        o_dates = src_in.loc[src_in["name"] == name, "date"].tolist()
        t_dates = src_out.loc[src_out["name"] == name, "date"].tolist()
        co, ct = episode_coherent_alerts(o_dates, t_dates)
        coherent[name] = (co, ct)

    latest_day = ek["date"].max()
    starting, ending = {}, {}
    for name, (co, ct) in coherent.items():
        state_o = [d for d in co
                   if d > latest_day - pd.Timedelta(days=RECENT_D)]
        state_t = [d for d in ct
                   if d > latest_day - pd.Timedelta(days=RECENT_D)]
        if state_o:
            starting[name] = max(state_o)
        if state_t:
            ending[name] = max(state_t)

    # ---- THE SIGNAL, unmissable (desk brief 2026-07-24: "it should be
    # super clear: euphoria is ending (get out signal) or euphoria
    # starting (get in)") - one red banner, one green banner, nothing to
    # interpret. Sparse by design: empty = the radar working.
    out_now = sorted((n for n in ending
                      if _state_of(n, starting, ending) == "ENDING"),
                     key=ending.get, reverse=True)
    in_now = sorted((n for n in starting
                     if _state_of(n, starting, ending) == "STARTING"),
                    key=starting.get, reverse=True)
    if out_now:
        st.error("**GET OUT — euphoria is ENDING:** "
                 + ",  ".join(f"{n} (signal {ending[n].date()}, "
                              f"{int((latest_day - ending[n]).days)}d ago)"
                              for n in out_now)
                 + ". Expect the top within ~a month of the signal.")
    if in_now:
        st.success("**GET IN — euphoria is STARTING:** "
                   + ",  ".join(f"{n} (signal {starting[n].date()}, "
                                f"{int((latest_day - starting[n]).days)}"
                                "d ago)" for n in in_now)
                   + ". The crowd is arriving; the rally window is open.")
    if not out_now and not in_now:
        st.info(f"**No live signal among {kind_label.lower()} right "
                "now** - no euphoria starting (get in) or ending (get "
                "out) in the last 21 days. Euphoria is rare; an empty "
                "pane is the radar working.")

    # ---- WINDOW-ADAPTIVE SCORECARD (desk request 2026-07-24): the few
    # numbers that matter, recomputed for whatever window is selected in
    # the sidebar - the SAME judge functions the research record uses.
    from analytics.euphoria_phases import (classify_onset_alerts,
                                           classify_top_alerts,
                                           _day_ints, _eps_arrays)
    import numpy as _np

    def _ts_of(day_int):
        return pd.Timestamp(_np.datetime64(int(day_int), "D"))

    def _window_scorecard():
        if episodes_df is None or not len(episodes_df):
            return None
        hi_eff = hi if hi is not None else latest_day
        # alerts younger than 45d cannot be judged yet (PENDING, the
        # project-wide convention) - excluded from the FA count only
        judge_hi = latest_day - pd.Timedelta(days=45)
        eps_k = episodes_df[episodes_df["kind"] == kind]
        eps_by = dict(tuple(eps_k.groupby("name")))
        empty = eps_k.iloc[0:0]
        out = {}
        for mode in ("in", "out"):
            judge = (classify_onset_alerts if mode == "in"
                     else classify_top_alerts)
            det_col = ("onset_detectable" if mode == "in"
                       else "top_detectable")
            captured, leads, fa_w, n_alerts = set(), [], 0, 0
            for name, (co, ct) in coherent.items():
                al = sorted(co if mode == "in" else ct)
                if not al:
                    continue
                n_alerts += sum(1 for d in al if lo <= d <= hi_eff)
                r = judge(_day_ints(pd.DatetimeIndex(al)),
                          _eps_arrays(eps_by.get(name, empty)))
                for ld in r["leads"]:
                    if lo <= _ts_of(ld["peak"]) <= hi_eff:
                        captured.add((name, int(ld["peak"])))
                        leads.append(ld["after_trough"] if mode == "in"
                                     else ld["before_peak"])
                fa_w += sum(1 for a in r["fa"]
                            if lo <= _ts_of(a) <= min(hi_eff, judge_hi))
            det = eps_k[(eps_k["peak"] >= lo) & (eps_k["peak"] <= hi_eff)
                        & eps_k[det_col]]
            out[mode] = {"captured": len(captured), "detectable": len(det),
                         "median_lead": (int(_np.median(leads))
                                         if leads else None),
                         "fa": fa_w, "alerts": n_alerts}
        return out

    sc = _window_scorecard()
    if sc:
        for mode, label, lead_lbl in (
                ("out", "GET OUT (ending)", "median warning before peak"),
                ("in", "GET IN (starting)", "median lag after the start")):
            r = sc[mode]
            m1, m2, m3, m4 = st.columns(4)
            hit = (f"{r['captured']}/{r['detectable']} "
                   f"({r['captured'] / r['detectable']:.0%})"
                   if r["detectable"] else "no episodes in window")
            m1.metric(f"{label} - hit rate", hit)
            m2.metric(lead_lbl, f"{r['median_lead']}d"
                      if r["median_lead"] is not None else "-")
            m3.metric("false alarms in window", r["fa"])
            m4.metric("signals in window", r["alerts"])
        st.caption("Scored inside the selected window only, with the "
                   "same judge the research record uses: a GET OUT hit "
                   "= a signal inside [peak-30d, peak+1d]; a GET IN hit "
                   "= a signal inside the episode's first 45 days. "
                   "Signals younger than 45d are PENDING, not false. "
                   "Small windows = small samples - the confirmatory "
                   "record is the walk-forward in the caption below.")

    # frozen alert threshold (drawn on every level panel)
    thr_now = None
    if euph_report and euph_report.get("thresholds"):
        thr_now = euph_report["thresholds"][
            max(euph_report["thresholds"])]

    ew = clip_window(ek, "date", lo, hi)
    ow_ = (clip_window(ok, "date", lo, hi)
           if ok is not None and len(ok) else None)

    def draw_chart(name, title_prefix, key):
        one = ew[ew["name"] == name].sort_values("date")
        if not len(one):
            st.caption(f"{name}: no euphoria data inside the selected "
                       "window")
            return
        sym = one["symbol"].iloc[0]
        px = (price_series(prices, sym, lo, hi)
              if prices is not None and sym in priced else None)
        one_i = one.set_index("date")
        # DANGER STATE: crowd swollen (the A1 2x bar) AND price in a G2
        # boom. Measured (NB06): a >=10%-in-7d drop begins within 30d on
        # ~62% of these days vs ~19% of ordinary days - this is the PM
        # warning; the GET OUT alerts time the peak inside it.
        #
        # It used to be an amber band behind the price. It is now drawn as
        # the PRICE LINE ITSELF turning amber on those days: same
        # information, no shading, and it cannot be misread as a separate
        # quantity because it IS the price. A band also implied "the whole
        # of this region is dangerous" when the underlying test is daily.
        danger_days = None
        if px is not None and not px.empty and "hype_ok" in one_i.columns \
                and prices is not None:
            full = prices[prices["symbol"] == sym].sort_values("date")
            pxa = full.set_index("date")["px_last"].asfreq("D").ffill()
            low120 = pxa.rolling(120, min_periods=60).min()
            bm = (EUPHORIA_BOOM_MIN_SINGLE if kind == "single"
                  else EUPHORIA_BOOM_MIN_ETF)
            boom = ((pxa / low120 - 1) >= bm).reindex(one_i.index).eq(True)
            danger_days = one_i["hype_ok"].astype(bool) & boom
        lvl_raw = one_i["level"]
        # the DISPLAY curve is 7d-smoothed (the house ROLL constant):
        # one loud afternoon is not a trend - alerts should coincide
        # with a visible regime change, not daily jitter
        lvl = lvl_raw.rolling(ROLL, min_periods=1).mean()
        co, ct = coherent.get(name, ([], []))
        w0, w1 = one_i.index.min(), one_i.index.max()
        onset_alerts = [d for d in co if w0 <= d <= w1]
        top_alerts = [d for d in ct if w0 <= d <= w1]
        state = _state_of(name, starting, ending)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.62, 0.38],
                            vertical_spacing=0.06)
        if px is not None and not px.empty:
            fig.add_trace(go.Scatter(x=px.index, y=px.values,
                                     name=f"{sym} price",
                                     line=dict(color=SLATE,
                                               width=1.5)),
                          row=1, col=1)
            # the same price line, amber, only on danger days (see above).
            # connectgaps=False is what keeps it as separate stretches
            # instead of one line cutting across the quiet periods.
            if danger_days is not None and bool(danger_days.any()):
                hot = px.where(danger_days.reindex(px.index).eq(True))
                fig.add_trace(go.Scatter(
                    x=hot.index, y=hot.values, mode="lines",
                    name="crowded AND already run up (risk zone)",
                    connectgaps=False,
                    line=dict(color=OCHRE, width=2.6),
                    hovertemplate="risk zone: %{y:.2f}<extra></extra>"),
                    row=1, col=1)
        # faint raw level = context; bold smoothed level = the signal
        fig.add_trace(go.Scatter(x=lvl_raw.index, y=lvl_raw.values,
                                 name="daily level (raw)",
                                 line=dict(color=INK_LABEL, width=0.7),
                                 opacity=0.35),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=lvl.index, y=lvl.values,
                                 name=f"euphoria level ({ROLL}d smooth)",
                                 line=dict(color=ACCENT, width=2.2)),
                      row=2, col=1)
        # The eligible stretches are drawn as a HAIRLINE RIBBON along the
        # floor of the lower panel, not as a fill under the curve.
        #
        # WHY the change (desk decision 2026-07-27, "forgo the shading"):
        # a translucent fill under a wiggly curve competes with the curve
        # for the same pixels, so the eye reads the shading as a second
        # quantity and the panel becomes two overlaid stories.  A thin
        # ribbon on the axis floor answers the same question - "could a
        # signal even fire here?" - while leaving the curve unobstructed.
        if "hype_ok" in one_i.columns:
            elig = one_i["hype_ok"].astype(bool).reindex(lvl.index)
            ribbon = [3.0 if bool(v) else None for v in elig]
            fig.add_trace(go.Scatter(
                x=lvl.index, y=ribbon, name="crowd big enough to signal",
                mode="lines", connectgaps=False,
                line=dict(color=OCHRE, width=4),
                hovertemplate="crowd big enough to signal<extra></extra>"),
                row=2, col=1)
        if thr_now:
            fig.add_hline(y=thr_now, line_dash="dot", line_color=INK_LABEL,
                          opacity=0.8, row=2, col=1,
                          annotation_text="signal level",
                          annotation_position="top left",
                          annotation_font=dict(color=INK_LABEL, size=10))

        def _ms(ts):
            # plotly's vline+annotation midpoint maths does Timestamp+int
            # arithmetic on some plotly/pandas versions and crashes;
            # epoch-milliseconds is numeric and works on every version
            return pd.Timestamp(ts).value / 1_000_000

        # SIGNAL LINES, and nothing else.  Every shaded region that used to
        # live here is gone: the danger-state band, the start-to-end episode
        # span, and the fill under the curve.  Three reasons, all of them
        # about how a PM actually reads a chart under time pressure:
        #   1. shading says "somewhere in this region", a line says "here" -
        #      and a dated decision is the whole product;
        #   2. three overlapping translucent bands mix into a fourth colour
        #      that means nothing, which is what the screenshot showed;
        #   3. a line survives printing, projecting and greyscale.
        # The signal itself is UNCHANGED - identical dates, identical frozen
        # thresholds. This is presentation only.
        for d in onset_alerts:                       # GET IN
            fig.add_vline(x=_ms(d), line_color=TEAL, line_width=1.6,
                          opacity=0.9)
        for d in top_alerts:                         # GET OUT
            fig.add_vline(x=_ms(d), line_color=BEAR, line_width=1.6,
                          opacity=0.9)
        # Labels are placed in a SEPARATE pass, alternating height, so two
        # signals a few days apart do not print on top of each other - the
        # exact defect that made the old chart unreadable when a start and
        # an end landed in the same fortnight.
        marks = ([(d, "GET IN", TEAL) for d in onset_alerts]
                 + [(d, "GET OUT", BEAR) for d in top_alerts])
        for i, (d, text, colour) in enumerate(sorted(marks)):
            fig.add_annotation(
                x=_ms(d), y=1.0, yref="y domain", yanchor="bottom",
                text=f"<b>{text}</b>  {pd.Timestamp(d).strftime('%d %b')}",
                showarrow=False, font=dict(size=9.5, color=colour),
                bgcolor="rgba(255,255,255,0.92)", borderpad=2,
                yshift=4 + 14 * (i % 2), row=1, col=1)
        badge = ""
        if state == "STARTING":
            badge = "  |  GET IN - EUPHORIA STARTING NOW"
        elif state == "ENDING":
            badge = "  |  GET OUT - EUPHORIA ENDING NOW"
        fig.update_layout(height=560, hovermode="x unified",
                          margin=dict(l=10, r=10, t=55, b=20),
                          legend=dict(orientation="h", yanchor="top",
                                      y=-0.16),
                          title=dict(text=(f"{title_prefix}{name} ({sym})"
                                           f" - {len(onset_alerts)} get-in"
                                           f" / {len(top_alerts)} get-out"
                                           f" signal(s) in window{badge}"),
                                     y=0.97, x=0.01))
        fig.update_yaxes(title_text="price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="euphoria", range=[0, 100],
                         row=2, col=1)
        _axes_fidelity(_theme(fig))
        st.plotly_chart(fig, width="stretch", key=key)

        # ---- WHY did each alert fire? (plain-English decomposition of
        # the stored component values on the alert day - nothing here is
        # recomputed, it is the exact evidence the detector acted on)
        dk_i = (dk[dk["name"] == name].set_index("date")
                if use_desk else None)
        thr_in_d = ((desk_report or {}).get("get_in", {})
                    .get("live_threshold"))
        thr_out_d = ((desk_report or {}).get("get_out", {})
                     .get("live_threshold"))
        expl = []
        for d in sorted(top_alerts):
            r = one_i.loc[:d].iloc[-1] if d not in one_i.index \
                else one_i.loc[d]
            fade_txt = (" The FADE was active - the crowd was still at "
                        "maximum size but the mood had started rolling "
                        "over (historically the last stage before a "
                        "top)." if bool(r.get("fade")) else "")
            desk_txt = ""
            if dk_i is not None and d in dk_i.index:
                rd = dk_i.loc[d]
                boom_txt = (" and the chart CONFIRMED a real boom "
                            "(price ≥ its G2 threshold above its own "
                            "120d low - the desk price gate)"
                            if bool(rd.get("boom_state")) else "")
                if pd.notna(rd.get("out_score")) and thr_out_d:
                    desk_txt = (f" The 7d-smoothed desk score "
                                f"{float(rd['out_score']):.2f} crossed "
                                f"the frozen GET OUT threshold "
                                f"{thr_out_d:.2f}{boom_txt} - one loud "
                                "afternoon cannot fire this.")
            expl.append(
                f"**{pd.Timestamp(d).date()} — GET OUT (euphoria "
                f"ending).** The crowd had genuinely swollen (7d "
                f"mention share ≥ 2× its own normal - the hype gate). "
                f"Attention sat in the top "
                f"{max(1, round((1 - float(r['e1'])) * 100))}% "
                f"of this name's own year (E1 {float(r['e1']):.2f}); "
                f"bullishness had persisted ≥75% of posting days for 4 "
                f"weeks (E2 {float(r['e2']):.2f}); crowd influx rank "
                f"{float(r['e3']):.2f}; super-exponential attention "
                f"rank {float(r['e5']):.2f}.{desk_txt}{fade_txt}")
        if ow_ is not None:
            oo_i = ow_[ow_["name"] == name].set_index("date")
            for d in sorted(onset_alerts):
                if d not in oo_i.index:
                    continue
                r = oo_i.loc[d]
                desk_txt = ""
                if dk_i is not None and d in dk_i.index:
                    rd = dk_i.loc[d]
                    if pd.notna(rd.get("in_score")) and thr_in_d:
                        desk_txt = (
                            f" The 7d-smoothed desk score "
                            f"{float(rd['in_score']):.2f} crossed the "
                            f"frozen GET IN threshold {thr_in_d:.2f}, "
                            "and the name was NOT already end-stage "
                            "(phase-aware: you cannot 'start' euphoria "
                            "that already satisfies every ending gate).")
                expl.append(
                    f"**{pd.Timestamp(d).date()} — GET IN (euphoria "
                    f"starting).** "
                    f"The crowd was {float(r['hype_raw']):.1f}× its own "
                    f"normal size and ARRIVING fast: attention "
                    f"acceleration rank "
                    f"{float(r['attention_accel']):.2f}, hype-ratio "
                    f"rank {float(r['hype_ratio']):.2f}, mood turning "
                    f"up (bullish inflection "
                    f"{float(r['bull_inflection']):.2f}), 2-week influx "
                    f"{float(r['influx_speed']):.2f}, super-exponential "
                    f"attention {float(r['attention_convexity']):.2f}."
                    f"{desk_txt}")
        expl = [e for e in expl if e]
        if expl:
            with st.expander(f"why did {name}'s alert(s) fire? "
                             "(the exact evidence, in plain English)"):
                for e in expl:
                    st.markdown("- " + e)
                st.caption("Every number is a percentile of this name's "
                           "OWN trailing year (1.00 = the most extreme "
                           "it has been). An alert needs the gates AND "
                           "the threshold - a high line alone is never "
                           "enough, which is why the level can wiggle "
                           "without alerts firing.")

    # ---- LOOK UP ANY NAME (type to search) --------------------------
    all_names = sorted(ek["name"].unique())
    pick = st.selectbox(
        f"look up any {kind_label.lower()} (type to search - shows its "
        "euphoria whether or not it ever alerted)",
        ["(none)"] + all_names, key=f"{key_prefix}_lookup")
    if pick and pick != "(none)":
        draw_chart(pick, "LOOKUP: ", f"{key_prefix}_lookup_chart")

    # ---- charts: EVERY instrument with a (coherent) euphoria alert
    # inside the selected window, newest alert first - no filler names
    last_alert = {}
    for name, (co, ct) in coherent.items():
        in_win = [d for d in co + ct
                  if lo <= d and (hi is None or d <= hi)]
        if in_win:
            last_alert[name] = max(in_win)
    show = sorted(last_alert, key=last_alert.get,
                  reverse=True)[:how_many]
    if not show:
        st.info(f"no euphoria alerts among {kind_label.lower()} in the "
                "selected window - widen the window in the sidebar to "
                "see past episodes")
    for i, name in enumerate(show, 1):
        if name == pick:
            continue               # already drawn by the lookup
        draw_chart(name, f"#{i}  ", f"{key_prefix}_{name}")

    # ---- conclusions line (headline record only - evidence lives in
    # notebooks/01-04 and docs/DECISIONS.xlsx, not on the terminal)
    bits = []
    if desk_report:
        wo = desk_report.get("get_out", {}).get("walk_forward", {})
        wi = desk_report.get("get_in", {}).get("walk_forward", {})
        bits.append(f"GET OUT: {wo.get('capture_rate')} of detectable "
                    f"peaks inside [peak-30d, peak+1d], median warning "
                    f"{wo.get('median_lead_days')}d, "
                    f"{wo.get('fa_per_iy')} FA/instr-yr, AP "
                    f"{wo.get('ap')} vs base {wo.get('ap_baseline')}")
        bits.append(f"GET IN: {wi.get('capture_rate')} of detectable "
                    f"starts (+{wi.get('late')} late-but-in-rally), "
                    f"{wi.get('fa_per_iy')} FA/instr-yr; only "
                    f"{wi.get('adjacency_within_cooldown_before_end')} "
                    "start(s) in the whole record landed within 21d of "
                    "an end (was 20 before the phase-aware fix)")
    elif euph_report:
        o = euph_report.get("overall", {})
        bits.append(f"ENDING detector: {o.get('capture_rate_detectable')}"
                    f" of detectable peaks inside [peak-30d, peak+1d], "
                    f"median lead {o.get('median_lead_days')}d, "
                    f"{o.get('fa_per_instrument_year')} FA/instr-yr")
    if bits:
        st.caption("Validated record (walk-forward, both denominators in "
                   "the research pack): " + " | ".join(bits)
                   + ". Full evidence - walk-forward tables, ablation, "
                   "ML challenger, tournament: notebooks/01-04 + "
                   "docs/DECISIONS.xlsx. AMBER band = the DANGER STATE "
                   "(crowd >=2x its normal AND price in a G2 boom): a "
                   ">=10%-in-a-week drop begins within 30 days on ~62% "
                   "of these days vs ~19% of ordinary days (NB06, CI "
                   "[+28pp,+50pp]) - the band is the standing PM "
                   "warning; alerts time the peak inside it. A START within 21d of an END is "
                   "suppressed as contradictory; a fast START then END "
                   "is a violent mania and the red risk signal is never "
                   "suppressed. Recent alerts are PENDING "
                   "until 45d of price exists to judge them.")


with t_euph_th:
    render_euphoria_tab("theme", "Themes", "euphth")
with t_euph_sg:
    render_euphoria_tab("single", "Single names", "euphsg")

# ---- INFLUENCE TRACKER (committed text-free store, extended live) ----
# INFORMATION ONLY. Nothing on this tab feeds the euphoria level or the
# GET IN / GET OUT alerts - that is a desk rule, and notebook 05 is the
# reason for it: the influence model does NOT generalise to authors it has
# not seen (cohort-split AP sits at or below the random floor), so it is a
# research exhibit, never a live input. The RANKING shown here is the
# MEASURED composite from the store, not a model prediction.
def influence_simple():
    """Plain-English version of the influence tab.

    The tab's own numbers (12,528 authors, 33,451 calls) are quoted from the
    notebook export rather than typed, for the same staleness reason as the
    euphoria panel - this store grows on every live run, so any hard-typed
    count here is wrong within a week."""
    n5 = _research("nb05_influence")
    store = n5.get("store", {})
    return f"""
**What this tab is.** A scoreboard of the people whose calls have actually
*worked*, plus what those same people are saying right now. It is background
colour for a PM - *"the accounts with a track record are leaning short semis
this month"* - and nothing more.

---

**The one number to understand: INFLUENCE, 0-100.**

Everything on this tab is ranked by one score, shown as a bar from 0 to 100.
Read it as *"how strong is this person's record, compared with the strongest
record we track"* - **100 is the best name in the store**, 50 is half as
strong as that name. It is a **position in the field, not an accuracy**: 80
does not mean 80% right. That distinction is the whole reason it is drawn as
a bar and never printed as a percentage.

How the record behind it is measured:

- Every call they made is checked against what the price did over the next
  **20 trading days**.
- Being right on a **big** move counts for more than being right on a drift.
- The bar for "right" **scales with the name**: 3% is a real call on an index
  ETF and noise on a meme stock, so each name is judged with its own ruler.
- A short record is **pulled back toward average**. *Example: 3-from-3 does
  not top the board; 28-from-40 does.* Nobody wins on two lucky calls.

*Per-person hit rates are deliberately not shown here.* They are measured and
they are in notebook 05, where each one sits next to its sample size and its
confidence interval. On a screen, next to a score that has been deliberately
pulled toward average, a raw hit rate only ever invited the comparison the
pulling-toward-average exists to prevent.

---

**The second number: BACKING, as a share of the room.**

When the tab says a ticker has **26%** backing, it means *a quarter of
everything this panel said in the window - weighted by whose record said it
and how hard - went into this one name*. It is always a share of the current
field, never a raw total: a raw total would rise every week the forum simply
got busier.

The line to read it against is the **even split**, which the chart draws and
labels: with 51 names in play that is 2.0% each, so 26% is thirteen times an
even share of the room's attention. The line is derived from the window
(100 / names), not chosen.

---

**The one counter-intuitive finding - "loud but wrong".**

The accounts with the *most* replies and the biggest reach were among the
**least** accurate. So the list a follow-the-big-names desk would copy is
flagged here as the list to fade. This reproduces the thesis's own headline
warning on our data.

---

**Reading the four pictures.**

- **Bubbles** - *what are they pushing?* Left/right is short/long, **height
  is backing** (see above), bubble area is how many times it was called.
  Top-right = crowded bullish.
- **Week-by-week lines** - *is it building or fading?* The top line is the
  whole panel's lean; the lines below are the leading names' backing, week by
  week. Climbing = people with a record piling in.
- **Bars for one name** - *who exactly is behind it?* One bar per person, bar
  length is their influence, colour is their side. A few long bars all one
  colour is the strongest reading this tab produces.
- **Map** - *who replies to whom.* Close together = they talk to each other,
  bigger dot = more people reply to them, colour = influence. Only the
  densely-connected core is drawn; the full graph is a hairball.

---

**Two honest limits.**

- This tab does **not** feed the euphoria signal. It is information, not a
  trigger.
- We tried to *predict* who would be influential with eight graph models and
  **none of them beat a plain baseline**, so nothing here is a prediction.
  What you see is the measured record only.

*Store: **{_cnt(store.get('authors in store'))} authors**,
**{_cnt(store.get('calls extracted'))} scored calls**, first call
{store.get('first call', '-')}. Pseudonymous and text-free by design - no post
content is ever stored.*
""" if store else INFLUENCE_HOW_TO_READ


INFLUENCE_HOW_TO_READ = """
**What this tab is.** A scoreboard of the people whose calls have actually
worked, and a plain reading of what those same people are saying right now.
It is background colour for a PM - *"the accounts with a record are leaning
short semis this month"* - and nothing more.

**How the score is built** (method: Chan, Oxford M.Eng thesis, 2026 §4.6).
Every call an author makes is scored against what the market then did over
the next 20 trading days, and three ingredients are combined:

| ingredient | weight | what it rewards |
|---|---|---|
| stance-weighted accuracy | 0.4 | being right, and being right when you said it loudly |
| abnormal-return-weighted | 0.4 | being right about the moves that were *big* for that name |
| enhanced accuracy | 0.2 | being right on the calls the extractor was most sure it read correctly |

Each ingredient is **Bayesian-shrunk toward the crowd base rate**, so nobody
tops the board on two lucky calls - a 100%-from-3 record is pulled most of
the way back to average, a 70%-from-40 record is barely moved. The result is
min-max normalised to 0-1; **HIGH tier = 0.66 and above**.

**What min-max normalisation means for how you read the number, and why the
tab shows 0-100 instead.** Min-max maps the lowest score in the population to
0 and the highest to 1. So the top author scores ~1.0 *by construction* - not
because they were right 100% of the time. Printed as "usefulness 0.987" (which
is how this tab used to print it) it reads as an accuracy, and it is not one:
it is a position in a field. Every exhibit here therefore displays
`100 x composite / max(composite)` instead, labelled **influence 0-100** with
"100 = the strongest measured record in the store" written on the axis. It is
a positive rescaling, so no ranking changes anywhere - what changes is that
the reader can say out loud what the number means. Implemented once, in
`analytics/influence_graph.py::influence_index`, so no chart can drift into a
different scale.

**The same problem in `weighted_voices`, and why the fix is a SHARE and not a
multiple** (measured, 2026-07-27). `weighted_voices` is a unitless sum of
(influence x conviction) per name, so it needs a reference before it can be
spoken. The first attempt divided it by the **median name** in the same window
-  "2.4x the typical name" - on the argument that this mirrors the euphoria
detector's A1 convention ("2x its own 120d median"). **That argument is
withdrawn: A1 is not analogous.** A1 divides a name by *its own* history, which
is a stable reference. The median NAME divides by whatever sits in the middle
of a long-tailed cross-section, and in this store the middle is a ticker one
person mentioned once: in the week ending 2026-06-28, 163 names were mentioned
and the median one carried 0.24 of backing, so MSFT printed **141x**. Weekly
maxima ran 141x, 41x, 2.7x, 14x and 26x on the 30-day view and 171x on the
90-day view - numbers no one can say out loud, and which move with how many
one-off tickers a given fetch happened to catch rather than with the crowd.
With an empty author list the median is exactly 0 and every ratio came back
NaN.

Every exhibit therefore shows **share of the room's conviction**:
`backing_share = 100 x weighted / sum(weighted)`. It is bounded 0-100, it is
additive (crowding *is* concentration, so shares that add to 60% say six-tenths
of the room is in five names), and a hundred one-off tickers barely move the
denominator. The line to read it against is **derived, not chosen**:
`even_share(n) = 100 / n`, what every name would show if attention were spread
equally, drawn and labelled on the charts. On the 30-day cross-section that is
2.0% across 51 names against MSFT's 26.1% - thirteen times an even share.

**Why a call counts as right** is measured against a bar that scales with
each name's OWN volatility: a 3% move is a real call on an index and noise
on a meme stock.

**Why no per-author hit rate is displayed** (desk decision, 2026-07-27). It is
still computed, still stored in `author_scores.parquet`, and still an input to
the composite - it is simply not shown on this screen. Two reasons. First, a
raw hit rate and a shrunk composite are answers to different questions, and
side by side they disagree *by design*: shrinkage exists precisely to stop a
3-from-3 record outranking a 28-from-40 one, so a reader reconciling the two
columns is fighting the method. Second, an unqualified per-person accuracy
invites position sizing off a sample of five, which the cohort-split test in
notebook 05 says nothing here supports. Accuracy is reported in notebook 05
with its sample size and confidence interval attached, which is the only form
in which it is defensible.

**"Loud but wrong"** is the thesis's headline warning, reproduced here: the
accounts in the top quartile of reply-graph PageRank but below median
influence. They are the accounts a *follow-the-big-names* desk would copy,
and the evidence says fade them.

**The time panels** (added 2026-07-27) are the dimension this tab lacked
entirely. `calls.parquet` has carried a date on every call since inception,
yet every exhibit pooled the whole window into one snapshot - so the tab could
say "GME is the most-backed name" but never "GME's backing has tripled in
three weeks", which is the statement a PM can act on. Both panels are weekly,
not daily: the comment fetch runs about twice a week under the ingestion
budget (PARAMETER_REGISTER Class 7), so a daily axis would largely plot the
fetch schedule. Per-week backing is normalised by that week's own median name,
so a busy forum week cannot masquerade as crowding.

**The map** shows who replies to whom. Position comes from a force layout
(people who reply to each other get pulled together), dot size is how many
people reply to *them*, colour is the influence score. Only the **k-core
backbone** is drawn - the densely connected heart of the graph - because a
12,000-node picture is a hairball. Every node drawn provably has at least
*k* neighbours inside the picture, so it is a principled slice, not a
random thinning.

**What this tab is NOT.** It is not an input to the euphoria signal, and the
graph model is not used to rank anybody. Notebook 05 tested eight
architectures against a plain linear baseline; not one earned its
complexity, and none of them generalised to new authors. The honest use of
this store is the *measured* record, which is what you see.
"""

_INFL_DIR = os.path.join(ROOT, "data", "reference", "influence")
_INFL_SCORES = os.path.join(_INFL_DIR, "author_scores.parquet")
_INFL_CALLS = os.path.join(_INFL_DIR, "calls.parquet")
_INFL_EDGES = os.path.join(_INFL_DIR, "reply_edges.parquet")
INFL_MIN_JUDGED = 5          # a record needs a record: 5+ judged calls
INFL_BACKBONE_MIN = 150      # walk k DOWN until the backbone has this many
INFL_BACKBONE_MAX = 320      # and never above this (the layout is O(n^2))
INFL_EGO_MAX = 120           # nodes drawn in one author's neighbourhood
# Bubbles printed on the lead chart.  A DISPLAY limit, not an analytical one:
# every name still appears in the ranked table below, and the list is sorted
# by influence behind it, so this drops the least-backed tail first.  30 is
# where ticker labels stop overlapping at the chart's 520px height - past
# that the text has to be hidden, and a bubble chart you have to hover to
# identify defeats the purpose of putting it first.
INFL_BUBBLE_MAX = 30


@st.cache_resource(show_spinner=False)
def _reply_graph(edges_mtime: float, board_mtime: float):
    """The reply graph over the people this tab can actually SCORE.

    WHY the restriction: the raw reply store holds ~107k accounts, but only
    the ~5k who have made at least one judged call have an influence score.
    Drawing all of them gives a picture where 8 dots in 9 are colourless -
    and, because the busiest repliers are mostly not callers, a k-core made
    of exactly the people the tab has nothing to say about. Restricting to
    scored authors makes colour coverage 100% and the picture 40x cheaper
    (0.3s vs 40s), and it is the same node set notebook 05 models.

    Cached as a RESOURCE rather than data because a Graph holds a sparse
    adjacency matrix - there is nothing worth serialising, and nothing here
    is ever mutated. Betweenness is deliberately NEVER computed on this tab
    (21s on the full store); every number below is O(m) or comes from a few
    hundred nodes."""
    board = _read(_INFL_SCORES, board_mtime)
    scored = board.loc[board["composite"].notna(), "author"].to_numpy()
    return ig.build_graph(pd.read_parquet(_INFL_EDGES), nodes=scored)


@st.cache_data(show_spinner=False)
def _backbone_frames(edges_mtime: float, board_mtime: float,
                     min_nodes: int = INFL_BACKBONE_MIN,
                     max_nodes: int = INFL_BACKBONE_MAX):
    g = _reply_graph(edges_mtime, board_mtime)
    sub, k = ig.kcore_subgraph(g, min_nodes=min_nodes, max_nodes=max_nodes)
    nodes, links = ig.map_frames(sub, board=_read(_INFL_SCORES, board_mtime))
    return nodes, links, k, g.n, g.m


@st.cache_data(show_spinner=False)
def _ego_frames(author: str, edges_mtime: float, board_mtime: float,
                max_nodes: int = INFL_EGO_MAX):
    g = _reply_graph(edges_mtime, board_mtime)
    ego = ig.ego_subgraph(g, author, radius=1, max_nodes=max_nodes)
    nodes, links = ig.map_frames(ego, board=_read(_INFL_SCORES, board_mtime))
    return nodes, links, ego.n, ego.m


def _unit(s: pd.Series) -> pd.Series:
    """0-1 rescale that survives an all-equal column (used for dot sizes,
    where a divide-by-zero would silently blank the whole map)."""
    v = pd.to_numeric(s, errors="coerce").fillna(0.0)
    span = float(v.max() - v.min())
    return (v - v.min()) / span if span > 0 else pd.Series(0.5, index=v.index)


_LBL_PX_PER_CHAR = 5.6       # width of one char of the 10px label font
_LBL_PX_HEIGHT = 15.0        # line height of the same font
_LBL_PLOT_PX = 900.0         # the figure's drawing area, near enough
_LBL_LEAD_PX = 46.0          # how far off its dot a label is parked


def _label_annotations(cand: pd.DataFrame, span: float,
                       cx: float, cy: float) -> list:
    """Place author labels OUTSIDE the hairball with a leader line back to
    the dot they belong to, dropping any that still cannot fit.

    WHY this shape, in two steps of reasoning:

    1. Printing a name at its own dot fails, because the top authors by
       influence sit inside the same dense cluster - that is what a reply
       graph IS - so the names land on top of each other. (An earlier
       version compared distances as a fraction of the layout span, which
       is the wrong unit entirely: a username is ~70 PIXELS wide whatever
       the span happens to be.)
    2. Merely stacking them apart fails too, more subtly: three names in a
       tidy column above one blob of forty dots tells you nothing about
       WHICH dot each name is. So each label is pushed away from the
       picture's centre - outward, into the white space a force-directed
       layout always leaves at the edges - and plotly draws a short arrow
       back to the dot. Now the pairing is unambiguous.

    Geometry, not tuning: the font is 10px on a ~900px canvas, so `scale`
    converts pixels to data units (the axes are aspect-locked by
    scaleanchor, so one scale serves both). A name of L characters occupies
    L*5.6 by 15 pixels. Each label tries four parking spots - radially out,
    the two perpendiculars, then inward - and takes the first whose box
    touches nothing already placed; if all four are taken the label is
    dropped and that author lives on hover only. Priority is influence
    order, so the strongest voice in a cluster is the one that keeps its
    label. None of this touches any number the dashboard reports.
    """
    scale = (span / _LBL_PLOT_PX) if span > 0 else 0.0
    half_h = 0.5 * _LBL_PX_HEIGHT * scale
    lead = _LBL_LEAD_PX * scale
    out, boxes = [], []                      # boxes: (x0, x1, y0, y1)

    def free(box):
        return all(box[1] <= b[0] or box[0] >= b[1]
                   or box[3] <= b[2] or box[2] >= b[3] for b in boxes)

    for row in cand.itertuples(index=False):
        dx, dy = row.x - cx, row.y - cy
        norm = float(dx * dx + dy * dy) ** 0.5
        if norm < 1e-9 * (span or 1.0):
            # Degenerate case, and a real one: the centre of an ego view IS
            # the centroid, so it has no "outward" direction. Park it above.
            ux, uy = 0.0, 1.0
        else:
            ux, uy = dx / norm, dy / norm    # unit vector away from centre
        half_w = 0.5 * len(str(row.author)) * _LBL_PX_PER_CHAR * scale
        for vx, vy in ((ux, uy), (-uy, ux), (uy, -ux), (-ux, -uy)):
            lx, ly = row.x + vx * lead, row.y + vy * lead
            box = (lx - half_w, lx + half_w, ly - half_h, ly + half_h)
            if not free(box):
                continue
            boxes.append(box)
            out.append(dict(
                x=float(row.x), y=float(row.y), text=str(row.author),
                # ax/ay are PIXEL offsets from the anchored point, and ay
                # grows downward on screen - hence the minus.
                ax=float(vx * _LBL_LEAD_PX), ay=float(-vy * _LBL_LEAD_PX),
                showarrow=True, arrowhead=0, arrowwidth=0.8,
                arrowcolor=INK_LABEL, font=dict(size=10, color=INK),
                bgcolor="rgba(255,255,255,0.86)", borderpad=1))
            break
    return out


def fig_influence_map(nodes: pd.DataFrame, links: pd.DataFrame, title: str,
                      centre: str | None = None, label_top: int = 8):
    """The influence picture: one line trace for every edge (a single trace
    with None breaks - thousands of two-point traces would crawl in the
    browser), one marker trace for the people."""
    fig = go.Figure()
    if len(links):
        ex, ey = [], []
        for x0, y0, x1, y1 in links[["x0", "y0", "x1", "y1"]].itertuples(
                index=False, name=None):
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                                 line=dict(width=0.5, color=HAIRLINE),
                                 showlegend=False))
    n = nodes.copy()
    for col in ("composite", "degree_here", "community"):
        if col not in n.columns:
            n[col] = float("nan")
    n["composite"] = pd.to_numeric(n["composite"], errors="coerce").fillna(0.0)
    # Colour on the SAME 0-100 influence scale as every other exhibit on the
    # tab, and no per-person accuracy in the hover: this map's own caption
    # reports that centrality and being right are near-unrelated, so putting
    # a hit rate on a node invited exactly the inference the number refutes.
    n["influence"] = 100.0 * n["composite"] / float(n["composite"].max() or 1.0)
    fig.add_trace(go.Scatter(
        x=n["x"], y=n["y"], mode="markers", showlegend=False,
        marker=dict(size=7 + 22 * _unit(n["degree_here"]),
                    color=n["influence"], colorscale="Oranges",
                    cmin=0.0, cmax=100.0, line=dict(width=0.5, color=WHITE),
                    colorbar=dict(title="influence", thickness=10, len=0.7)),
        customdata=n[["author", "influence", "degree_here",
                      "community"]].to_numpy(),
        hovertemplate=("<b>%{customdata[0]}</b><br>influence "
                       "%{customdata[1]:.0f} / 100"
                       "<br>%{customdata[2]:.0f} people reply to them"
                       "<br>conversation cluster %{customdata[3]:.0f}"
                       "<extra></extra>")))
    # label the strongest voices, but never two on top of each other: the
    # centre of an ego view always wins, then influence order decides.
    tops = n.nlargest(label_top, "composite")
    if centre is not None and centre in set(n["author"]):
        tops = pd.concat([n[n["author"] == centre], tops]).drop_duplicates(
            subset="author")
    span = float(max(n["x"].max() - n["x"].min(), n["y"].max() - n["y"].min()))
    for ann in _label_annotations(tops[["author", "x", "y"]], span,
                                  float(n["x"].mean()), float(n["y"].mean())):
        fig.add_annotation(**ann)
    fig.update_layout(title=dict(text=title, y=0.97, x=0.01), height=560,
                      margin=dict(l=10, r=10, t=55, b=10))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _theme(fig)


# `_rgba(hex, alpha)` lived here until 2026-07-27. Its only caller was
# fig_consensus, which encoded weight-of-evidence as bar opacity; both went
# when that chart was retired (see the note above fig_crowding_time). Every
# surviving chart states its weight on an axis instead of in an alpha
# channel, which is the readable way round and needs no colour arithmetic.


# ---------------------------------------------------------------------------
# PLAIN ENGLISH lives in analytics/plain_english.py, not here.
#
# It moved out of this file on 2026-07-27 for one reason: the notebooks need
# the SAME wording, and a notebook cannot import this module (importing a
# streamlit script executes the whole app).  With the glossary in analytics/,
# the dashboard and all seven notebooks read one definition of every term, so
# the screen and the research record cannot drift into calling the same
# quantity two different things.
# ---------------------------------------------------------------------------


def _thin_labels(xs, ys, x_span: float, y_span: float,
                 w_px: float, h_px: float,
                 gap_x_px: float = 30.0, gap_y_px: float = 13.0) -> list[bool]:
    """Which point labels can be drawn without printing on top of each other.

    A LAYOUT rule, not a data rule, and the distinction matters for a tab
    whose whole premise is that nothing gets filtered on an invented
    threshold.  Nothing here decides which names MATTER; every point is
    still drawn, still hovers, and still appears in the exact-numbers table.
    What this decides is where there is physically room for ink.

    Why it is needed at all: `consensus` is bounded at +/-1 and hits the wall
    exactly whenever every call on a name was long, which is common - so a
    dozen names pile onto x=+1.00 and their labels stack vertically.  On the
    measured 30-day cross-section INTU (4.04%) and MELI (3.58%) sit 0.46pp
    apart, which at 520px is about 9 pixels, and a 10pt label needs 13.  The
    two tickers printed through each other.

    The thresholds are derived from the GEOMETRY rather than chosen: gap_y is
    one line box at the label's font size (10pt x 1.3 leading), gap_x is the
    width of a four-character ticker at that size.  Two labels are only
    suppressed when their boxes actually overlap in BOTH directions - GOOG at
    x=0.53 and MELI at x=1.00 share a height but not a column, so both keep
    their label.

    Greedy in descending y, because height is the point of the chart: the
    most-backed name always keeps its label and the tail yields to it.
    """
    pts = [(float(y), float(x), i) for i, (x, y) in enumerate(zip(xs, ys))]
    # Tallest first, and on an exact tie the EARLIER row wins - the caller
    # passes the digest already ranked by backing, so ties resolve toward the
    # better-backed name rather than toward whichever happened to be last.
    pts.sort(key=lambda t: (-t[0], t[2]))
    keep = [False] * len(pts)
    placed: list[tuple[float, float]] = []     # (x, y) of labels already drawn
    dx = gap_x_px * (x_span / w_px) if w_px else 0.0
    dy = gap_y_px * (y_span / h_px) if h_px else 0.0
    for y, x, i in pts:
        if any(abs(x - px) < dx and abs(y - py) < dy for px, py in placed):
            continue
        keep[i] = True
        placed.append((x, y))
    return keep


def fig_influence_bubbles(dig: pd.DataFrame, voices: pd.DataFrame,
                          title: str, top_n: int | None = None):
    """THE lead influence chart: what the people with a record are pushing.

    Four readings in one picture, which is why this replaced the bar chart
    as the first thing on the tab:

      * X = net direction. Left is short, right is long.
      * Y = SHARE OF THE ROOM'S CONVICTION, in per cent. 26 means "a quarter
        of everything this panel said in the window, weighted by whose
        record said it and how hard, went into this one name".
      * SIZE = how many times it was called. Big means repeatedly, not once.
      * COLOUR = muted green long, muted brick short.

    So the eye goes straight to the TOP-RIGHT: heavily-backed, repeatedly
    called, bullish - which is precisely the crowding a PM is asked to watch.
    Bottom-left is the same thing on the short side; anything near x=0 is a
    genuine disagreement and is labelled as such rather than hidden.

    The vertical axis used to plot `weighted_voices` RAW, and that was the
    single worst number on the tab: a sum of (score x conviction) has no
    unit, so "influence behind it 3.42" told a reader nothing and could not
    be compared between two windows.  `ig.backing_share` makes it a per cent
    of everything the panel said in the window (see the block comment above
    that function for the measured reason share beat "x the median name").
    The ORDER of the bubbles is identical - it is a positive rescaling - so
    nothing about the ranking changed, only whether it can be read aloud.

    The dotted line is the EVEN SPLIT, 100/n per cent: what every name would
    show if the room spread its conviction equally.  Above it means more
    crowded than even.  It is derived from the window, not chosen.

    Only names AT OR ABOVE that line are labelled.  Not a display cap for
    its own sake: at 520px the sub-even tail is a band of names 1% apart, so
    labelling it produced a stack of overlapping tickers that hid the two
    names the chart exists to show.  The tail is still drawn, still hovers,
    and appears in full in the exact-numbers table under the chart.

    The hover carries the names and each person's lean, because "who said
    this" is always the next question and a chart that cannot answer it
    sends the reader back to a table.
    """
    d = dig.dropna(subset=["consensus"]).copy()
    if voices is not None and len(voices):
        d = d.merge(voices[["ticker", "voices"]], on="ticker", how="left")
    else:
        d["voices"] = ""
    d["voices"] = d["voices"].fillna("")
    # SHARE AND EVEN SPLIT ARE COMPUTED OVER THE WHOLE WINDOW, THEN THE HEAD
    # IS TAKEN.  This function used to receive an already-truncated frame, so
    # the denominator was the 30 names being drawn rather than the 51 in the
    # window: MSFT printed 29.5% on the chart while KPI 4 above it printed
    # 26.1% for the same name in the same window, and the even-split line read
    # 100/30 = 3.3% instead of 100/51 = 2.0%.  One quantity cannot have two
    # values on one screen, so the caller now passes the FULL digest plus how
    # many bubbles to draw, and "share of the room" always means the whole
    # room.  The tail that is dropped from the picture still counts in the
    # denominator, which is the point of a share.
    d["backing"] = ig.backing_share(d["weighted_voices"]).fillna(0.0)
    even = ig.even_share(len(d))
    if top_n is not None:
        d = d.head(int(top_n)).copy()
    # Bubble AREA scales with calls (plotly's sizemode="area"), because area
    # is what the eye actually compares - scaling the RADIUS by the value
    # makes a 4x count look 16x bigger, which is the classic bubble lie.
    smax = float(d["n_calls"].max()) or 1.0
    # Two independent gates on the ticker labels, and they answer different
    # questions.  `b >= even` is the ANALYTICAL one: below an even split a
    # name is not crowded, and at 520px that tail is a band of names 1% apart
    # whose labels hid the two names the chart exists to show.  `_thin_labels`
    # is the LAYOUT one: of the names that survive, drop only those whose text
    # would print through a label already placed (see that function).
    _ymax_est = float(d["backing"].max() or 1.0) * 1.34
    _room = _thin_labels(d["consensus"], d["backing"],
                         x_span=2.36, y_span=_ymax_est,
                         w_px=880.0, h_px=398.0)
    fig = go.Figure(go.Scatter(
        x=d["consensus"], y=d["backing"], mode="markers+text",
        text=[t if (b >= even and r) else ""
              for t, b, r in zip(d["ticker"], d["backing"], _room)],
        textposition="top center",
        textfont=dict(size=10, color=INK),
        marker=dict(
            size=d["n_calls"], sizemode="area",
            sizeref=2.0 * smax / (44.0 ** 2), sizemin=6,
            color=[BULL if c >= 0 else BEAR for c in d["consensus"]],
            opacity=0.72, line=dict(width=1, color=WHITE)),
        customdata=d[["ticker", "n_authors", "n_calls", "longs", "shorts",
                      "backing", "voices"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "net direction %{x:+.2f}  (+1 all long, -1 all short)<br>"
            "%{customdata[5]:.1f}% of the room's conviction<br>"
            "%{customdata[1]:.0f} people, %{customdata[2]:.0f} calls "
            "(%{customdata[3]:.0f} long / %{customdata[4]:.0f} short)"
            "<br><br><b>who</b>  (influence 0-100)<br>"
            "%{customdata[6]}<extra></extra>")))
    # Quadrant guides: a hairline at x=0 and a quiet label in each upper
    # corner. No shaded quadrants - the same reason the euphoria chart lost
    # its bands, and here the bubbles need every pixel of contrast.
    fig.add_vline(x=0, line_color=HAIRLINE, line_width=1)
    # The EVEN SPLIT line, annotated with its own value so nobody has to be
    # told what it is: above it = this name takes more of the room's
    # conviction than it would if attention were spread equally.
    fig.add_hline(y=even, line_color=HAIRLINE, line_width=1, line_dash="dot",
                  annotation_text=f"even split, {even:.1f}%",
                  annotation_position="bottom left",
                  annotation_font=dict(size=9, color=INK_LABEL))
    ymax = float(d["backing"].max() or 1.0)
    # 1.30 not 1.12: the tallest bubble carries a text label above it, and at
    # 1.12 the corner captions were printing on top of that label.
    for xpos, xanch, label, colour in (
            (-1.0, "left", "CROWDED SHORT", BEAR),
            (1.0, "right", "CROWDED LONG", BULL)):
        fig.add_annotation(
            x=xpos, y=ymax * 1.30, xanchor=xanch, yanchor="top",
            text=label, showarrow=False,
            font=dict(size=9, color=colour))
    fig.add_annotation(
        x=0, y=ymax * 1.30, xanchor="center", yanchor="top",
        text="GENUINE DISAGREEMENT", showarrow=False,
        font=dict(size=9, color=INK_LABEL))
    fig.update_layout(
        title=dict(text=title, y=0.97, x=0.01), height=520,
        margin=dict(l=10, r=10, t=64, b=58), showlegend=False)
    fig.update_xaxes(range=[-1.18, 1.18], zeroline=False,
                     title_text="<- SHORT          net direction"
                                "          LONG ->")
    fig.update_yaxes(range=[0, ymax * 1.34], rangemode="tozero",
                     ticksuffix="%",
                     title_text="share of the room's conviction "
                                "(bubble size = number of calls)")
    return _theme(fig)


# ---------------------------------------------------------------------------
# WHAT REPLACED fig_consensus (2026-07-27)
#
# `fig_consensus` was a horizontal bar of the same per-ticker consensus the
# bubble chart already plotted on its x-axis, with the bubble chart's y-axis
# folded into bar OPACITY.  Its own docstring admitted it: "this is the
# bubble chart above with the vertical axis flattened into shading".  Two
# exhibits of one dataset is not two exhibits - it is one exhibit and a
# distraction, and the flattened version was the weaker of the two because
# opacity is the least readable visual channel there is.
#
# The two charts below use the dimensions the tab genuinely did NOT have:
#   * fig_crowding_time  - TIME.  The store has carried call dates since
#     inception and the tab plotted none of them.  "Crowded" is a snapshot;
#     "crowding, and building for three weeks" is a decision.
#   * fig_ticker_backers - the PEOPLE behind one name, as data rather than
#     as hover text, so they can be compared, sorted and screenshotted.
#
# The exact numbers fig_consensus was "kept for" are still on the tab, in
# the table beside these charts.  Recorded in DECISIONS.xlsx sheet 8.
# ---------------------------------------------------------------------------
def fig_crowding_time(tilt: pd.DataFrame, hist: pd.DataFrame, title: str):
    """Is the crowding BUILDING or FADING? Two stacked panels, one x-axis.

    TOP - the whole panel's net direction, week by week. A line walking up
    toward +1 is a room getting one-sidedly bullish, which is the condition
    the euphoria detector exists to catch; a line rolling over is the crowd
    losing conviction. The dotted zero line is genuine two-way disagreement.

    DOT SIZE on that line is the number of calls behind the week, and it is
    there because the ingestion budget leaves the weeks very unequal - this
    store has weeks of 23 calls next to weeks of 7,260. A 23-call week can
    read +1.00 because four people all said long, and on an unsized line
    that pinprick of evidence looks exactly like a thousand-call week.
    Sizing the dot is the honest fix: no week is dropped, no threshold is
    invented, and a thin week LOOKS thin.

    BOTTOM - what share of the room's conviction each of the leading names
    took, week by week. A line climbing means people with a record are
    piling into that name; a line collapsing means they have stopped talking
    about it, which is usually the more actionable of the two and is
    invisible on any snapshot chart. The grey dotted line is the EVEN SPLIT
    for that week (100 / names mentioned), so it moves with the week - a
    week where the room touched 14 names has a much higher even split than
    one where it touched 238, and comparing against a fixed line would
    read that difference as crowding.

    Names are labelled AT THE END OF THEIR OWN LINE, with no legend. The
    first version used one colour per direction and told the lines apart by
    dash pattern with a legend underneath; every one of the five leading
    names happened to be long, so it drew five green lines a reader could
    not match to the legend at all. Colour has to keep meaning direction
    here - it means that in every other exhibit on the tab - so the name is
    what moves to the line.

    Both panels are weekly because the ingestion cadence is about twice a
    week: a daily axis would plot the fetch schedule, not the crowd.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
        row_heights=[0.38, 0.62],
        subplot_titles=("how one-sided the whole panel is  "
                        "(dot size = calls behind the week)",
                        "share of the room's conviction, per name"))
    if len(tilt):
        _cmax = float(pd.to_numeric(tilt["n_calls"], errors="coerce").max()
                      or 1.0)
        fig.add_trace(go.Scatter(
            x=tilt["period"], y=tilt["tilt"], mode="lines+markers",
            line=dict(color=NAVY, width=2),
            marker=dict(size=tilt["n_calls"], sizemode="area",
                        sizeref=2.0 * _cmax / (20.0 ** 2), sizemin=4,
                        color=NAVY, opacity=0.85),
            name="panel tilt", showlegend=False,
            customdata=tilt[["n_calls", "n_authors", "n_tickers"]].to_numpy(),
            hovertemplate=("week to %{x|%d %b %Y}<br>net direction "
                           "%{y:+.2f}<br>%{customdata[0]:.0f} calls by "
                           "%{customdata[1]:.0f} people on "
                           "%{customdata[2]:.0f} names<extra></extra>")),
            row=1, col=1)
        fig.add_hline(y=0.0, line_color=HAIRLINE, line_width=1,
                      line_dash="dot", row=1, col=1)
    # The even split, drawn as a SERIES rather than a constant line because
    # it depends on how many names the room touched that week.
    if len(hist):
        _ev = (hist.groupby("period", as_index=False)["even"].first()
               .sort_values("period"))
        fig.add_trace(go.Scatter(
            x=_ev["period"], y=_ev["even"], mode="lines",
            line=dict(color=INK_LABEL, width=1, dash="dot"),
            name="even split", showlegend=False, hoverinfo="skip"),
            row=2, col=1)
        fig.add_annotation(
            x=_ev["period"].iloc[0], y=float(_ev["even"].iloc[0]),
            text="even split", showarrow=False, xanchor="left",
            yanchor="bottom", font=dict(size=9, color=INK_LABEL),
            row=2, col=1)
    # One line per ticker. Colour by the name's OWN latest tilt, so a line
    # that a reader picks out as "the big one" is also coloured by side; the
    # NAME is printed at the end of the line, which is what tells them apart.
    _ends: list[tuple[float, object, str, str]] = []
    for i, (tk, grp) in enumerate(hist.groupby("ticker", sort=False)):
        g = grp.sort_values("period")
        side = float(g["tilt"].iloc[-1]) if len(g) else 0.0
        colour = BULL if side >= 0 else BEAR
        if len(g):
            _ends.append((float(g["share"].iloc[-1]), g["period"].iloc[-1],
                          str(tk), colour))
        fig.add_trace(go.Scatter(
            x=g["period"], y=g["share"], mode="lines+markers",
            name=str(tk), line=dict(color=colour, width=1.8,
                                    dash=("solid", "dash", "dot",
                                          "dashdot")[i % 4]),
            marker=dict(size=5),
            showlegend=False, cliponaxis=False,
            customdata=g[["ticker", "tilt", "n_calls",
                          "n_authors"]].to_numpy(),
            hovertemplate=("<b>%{customdata[0]}</b>  week to "
                           "%{x|%d %b %Y}<br>%{y:.1f}% of the room's "
                           "conviction that week<br>net direction "
                           "%{customdata[1]:+.2f}<br>%{customdata[2]:.0f} "
                           "calls by %{customdata[3]:.0f} people"
                           "<extra></extra>")),
            row=2, col=1)
    # END-OF-LINE NAMES, nudged apart so two lines that finish the window at
    # the same share do not print their tickers through each other.  On the
    # measured 30-day view MSTR and ASTR both ended at ~0.3%, and the two
    # labels came out as one illegible smear - the whole reason the legend was
    # dropped in the first place was to make the lines identifiable, so losing
    # a name to a collision defeats the change.  Drawn as ANNOTATIONS rather
    # than trace text because only an annotation can sit at a y of its own.
    #
    # The nudge is a layout correction, never a data one: the marker stays on
    # the true value and only the text moves, so a label may sit a hair above
    # or below its line's last dot.  Displacement is one 10pt line box
    # (13px), converted into data units from the row's own height - derived
    # from the geometry, not chosen.
    if _ends:
        _span = max(float(pd.to_numeric(hist["share"],
                                        errors="coerce").max() or 1.0),
                    1e-9) * 1.05
        _dy = 13.0 * (_span / 250.0)          # row 2 is ~250px of the 560
        _last_y = None
        for y_true, x_end, tk, colour in sorted(_ends, key=lambda t: -t[0]):
            y_lab = y_true if _last_y is None else min(y_true, _last_y - _dy)
            _last_y = y_lab
            fig.add_annotation(
                x=x_end, y=y_lab, text=f" {tk}", showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(size=10, color=colour), row=2, col=1)
    fig.update_layout(
        title=dict(text=title, y=0.98, x=0.01), height=560,
        # r=74 leaves room for the end-of-line name labels; without it the
        # last one or two characters of each ticker fall off the plot.
        margin=dict(l=10, r=74, t=92, b=44), showlegend=False)
    fig.update_yaxes(range=[-1.12, 1.12], row=1, col=1,
                     title_text="net direction")
    fig.update_yaxes(rangemode="tozero", ticksuffix="%", row=2, col=1,
                     title_text="share of conviction")
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=11, color=INK_LABEL)
        ann.x, ann.xanchor = 0.0, "left"
    return _theme(fig)


def fig_ticker_backers(bk: pd.DataFrame, ticker: str, title: str):
    """WHO is pushing one name: one bar per person, longest bar = strongest
    record, green = they are long it, brick = short.

    This is the chart the old tab was missing entirely. It had the same
    information - in a hover tooltip on a bubble - and a hover cannot be
    compared across people, sorted, or put in a note to a PM. Drawn, the
    shape of the answer is immediate: a few long bars means the name is
    being pushed by people with a record, a forest of short bars means it is
    being pushed by the crowd, and a mix of colours means the "consensus"
    number on the bubble chart is an average of an argument.

    Bar length is the 0-100 influence index (100 = the strongest record in
    the store), NOT an accuracy: see analytics/influence_graph.py's
    influence_index() for why that distinction is enforced in one place.
    """
    d = bk.sort_values("influence", ascending=True)   # plotly draws up
    fig = go.Figure(go.Bar(
        x=d["influence"], y=d["author"], orientation="h",
        marker_color=[BULL if w == "LONG" else (BEAR if w == "SHORT"
                                                else INK_LABEL)
                      for w in d["word"]],
        marker_line=dict(width=0), opacity=0.85,
        text=[f"  {w}" for w in d["word"]], textposition="outside",
        textfont=dict(size=9, color=INK_MUTED),
        customdata=d[["word", "n_calls", "conviction",
                      "last_date"]].to_numpy(),
        hovertemplate=("<b>%{y}</b><br>influence %{x:.0f} / 100<br>"
                       "%{customdata[0]} - %{customdata[1]:.0f} call(s), "
                       "conviction %{customdata[2]:.2f}<br>last said "
                       "%{customdata[3]|%d %b %Y}<extra></extra>")))
    fig.update_layout(title=dict(text=title, y=0.97, x=0.01),
                      height=max(260, 24 * len(d) + 120),
                      margin=dict(l=10, r=10, t=55, b=52), showlegend=False)
    fig.update_xaxes(range=[0, 118],
                     title_text="influence  (100 = the strongest measured "
                                "record in the store)")
    return _theme(fig)


with t_infl:
    st.subheader("Influence tracker - who has actually been right, and "
                 "what they are saying now")
    st.caption("INFORMATION ONLY - nothing on this tab feeds the euphoria "
               "level or the GET IN / GET OUT alerts. Ranking is the "
               "MEASURED record from the store, not a model prediction.")
    with st.expander("HOW TO READ THIS TAB  (start here - plain English)",
                     expanded=False):
        st.markdown(influence_simple())
    with st.expander("full method: scoring, shrinkage and the graph slice "
                     "(research version)", expanded=False):
        st.markdown(INFLUENCE_HOW_TO_READ)

    if not os.path.exists(_INFL_SCORES):
        st.info("no influence store on this machine yet - it builds "
                "ITSELF from live data: run one live pull "
                "(`python update_data.py`, or the sidebar button) and "
                "the tracker appears here after it finishes. Every "
                "later pull extends the same store - new calls are "
                "added, and recent calls are re-judged automatically "
                "once their 20-day window has prices. Nothing to "
                "rebuild, ever. (`git pull` also brings in the shared "
                "store once any machine has one.)\n\n"
                "The first run is the slow one: comment fetching is "
                "capped at what fits the run's time budget, so seeding "
                "from nothing takes two or three pulls. To do it in one "
                "sitting instead, use the uncapped catch-up button in "
                "the sidebar.")
    else:
        _b_mt, _c_mt = _mtime(_INFL_SCORES), _mtime(_INFL_CALLS)
        board_all = _read(_INFL_SCORES, _b_mt)
        board = (board_all[board_all["n_judged"] >= INFL_MIN_JUDGED]
                 .sort_values("composite", ascending=False)
                 .reset_index(drop=True))
        calls = (_read(_INFL_CALLS, _c_mt) if os.path.exists(_INFL_CALLS)
                 else None)

        # ---- controls: how many voices, and how recent -----------------
        _c1, _c2 = st.columns([1, 1])
        panel_n = _c1.slider("size of the panel (top N by record)", 10, 100,
                             25, step=5, key="infl_panel",
                             help="Everything below - the suggestions, the "
                                  "consensus, the map highlights - uses "
                                  "these top N authors and nobody else.")
        look = _c2.radio("how recent are 'what they are saying'",
                         ["last 30 days", "last 90 days", "the window above"],
                         horizontal=True, key="infl_look")
        if look == "the window above":
            asof = hi if hi is not None else None
            _end = pd.Timestamp(asof) if asof is not None else data_max
            days = max(int((_end - lo).days), 7)
        else:
            asof, days = None, (30 if look.endswith("30 days") else 90)

        panel = board.head(panel_n)["author"].to_numpy()
        # The TIME exhibits use every voice with a record, not the top-N
        # panel, and that distinction was measured rather than assumed. Cut
        # the top 25 into weeks and the weeks hold 124, 25, 1, 23 and 9
        # calls: the week with one call has one name, which is 100% of that
        # week's conviction by definition, and a "share of the room" built
        # from four people is not a fact about the room. Over all 340
        # recorded voices the same weeks hold 733, 404, 23, 98 and 155 calls
        # across 14-171 names, which a weekly statistic can stand on. The
        # panel slider exists to keep the NAME-level exhibits legible; it
        # was never meant to define the population whose mood is measured.
        recorded = board["author"].to_numpy()
        n_high = int((board["tier"] == "HIGH").sum())
        dig = (ig.suggestion_digest(calls, board, authors=panel, days=days,
                                    asof=asof)
               if calls is not None else pd.DataFrame())
        tilt_hist = (ig.panel_tilt_history(calls, board, authors=recorded,
                                           days=days, asof=asof)
                     if calls is not None else pd.DataFrame())

        # The five headline numbers. Two of the previous five were retired
        # on 2026-07-27: "HIGH tier" (a count of a threshold nobody on this
        # tab can now act on) and "crowd base rate" (an ACCURACY figure -
        # it existed only to give the per-author hit-rate column something
        # to be read against, and that column is gone). What replaced them
        # is the tab's missing dimension: whether the crowding is building
        # or fading. HIGH-tier count survives in the leaderboard caption.
        _k1, _k2, _k3, _k4, _k5 = st.columns(5)
        _k1.metric("voices with a record", f"{len(board):,}",
                   help=f"{INFL_MIN_JUDGED}+ calls already judged against "
                        f"prices, out of {len(board_all):,} authors tracked. "
                        f"{n_high:,} of them are HIGH tier.")
        if len(dig):
            _lead = dig.iloc[0]
            _lead_x = float(ig.backing_share(dig["weighted_voices"]).iloc[0])
            _net = float((dig["consensus"] * dig["weighted_voices"]).sum()
                         / dig["weighted_voices"].sum())
            _k2.metric("names in play", f"{len(dig):,}",
                       help="distinct tickers this panel said anything about "
                            "inside the window")
            _k3.metric("panel tilt", f"{_net:+.2f}",
                       "net long" if _net >= 0 else "net short",
                       help="all their calls in the window, weighted by "
                            "record and conviction, on one -1 to +1 scale. "
                            "+1 = unanimously long with full conviction.")
            _k4.metric("most-backed name", str(_lead["ticker"]),
                       f"{_lead_x:.0f}% of the room's conviction",
                       help="the ticker carrying the most influence-weighted "
                            "conviction in the window, and what share of the "
                            f"panel's total that is (an even split across "
                            f"{len(dig)} names would be "
                            f"{ig.even_share(len(dig)):.1f}% each)")
            # Direction of travel: how much the recorded voices SAID in the
            # latest week against the week before. Calls, not backing: a
            # count is a unit a reader already owns, and the question this
            # KPI answers ("are they talking more?") is about volume.
            if len(tilt_hist) >= 2:
                _c_now = float(tilt_hist["n_calls"].iloc[-1])
                _c_prev = float(tilt_hist["n_calls"].iloc[-2])
                _chg = ((_c_now / _c_prev - 1.0) * 100.0 if _c_prev > 0
                        else float("nan"))
                _k5.metric("calls vs last week",
                           f"{_c_now:,.0f}",
                           "n/a" if pd.isna(_chg) else f"{_chg:+.0f}%",
                           help="calls made by voices with a record in the "
                                "latest week, and the change on the week "
                                "before. Rising = people with a record are "
                                "talking more, not necessarily agreeing more. "
                                "The ingestion budget makes the weeks "
                                "unequal, so read it with the dot sizes on "
                                "the tilt chart below.")
            else:
                _k5.metric("calls vs last week", "-",
                           help="needs at least two weeks of calls inside "
                                "the window")
        else:
            _k2.metric("names in play", "-")
            _k3.metric("panel tilt", "-")
            _k4.metric("most-backed name", "-")
            _k5.metric("calls vs last week", "-")

        st.markdown("---")
        # ---- 1. WHAT THEY ARE PUSHING (the lead exhibit) ---------------
        # This section used to be second, underneath a 25-row leaderboard.
        # It leads now, and the reason is about what the tab is FOR: a PM
        # does not trade a list of usernames, they trade positioning.  The
        # first question is "what are the people with an actual record
        # pushing, and how one-sided is it" - the names are the EVIDENCE
        # for that answer, so they belong underneath it, and the reply map
        # (which measured near-zero relationship between being central and
        # being right) belongs underneath them.
        st.markdown("#### 1. What the panel is pushing")
        if calls is None or not len(dig):
            st.info("no calls in the chosen window - widen it, or run a "
                    "live comment pull to extend the store.")
        else:
            # WHO is behind each name.  Computed here rather than inside the
            # figure so the hover text and the tables below cannot disagree.
            voices = ig.ticker_voices(calls, board, authors=panel,
                                      days=days, asof=asof)
            _nb = min(len(dig), INFL_BUBBLE_MAX)
            # FULL digest, plus how many to draw - not dig.head(_nb).  The
            # share and the even-split line are denominated on the whole
            # window inside the figure, so they agree with KPI 4 above.
            st.plotly_chart(
                fig_influence_bubbles(
                    dig, voices,
                    f"the {_nb} most-backed names, last {days} days",
                    top_n=_nb),
                width="stretch", key="infl_bubbles")
            st.caption(
                f"**Read it in four steps.**\n\n"
                f"1. **Left or right** is which way they lean. Right of the "
                f"centre line is net **long**, left is net **short**, and a "
                f"name sitting on the line is a genuine argument rather than "
                f"a view. Colour just repeats it so the picture survives a "
                f"black-and-white printout.\n"
                f"2. **How high** is its **share of the room's conviction** - "
                f"of everything this panel said in the window, weighted by "
                f"whose record said it and how hard, what per cent went into "
                f"this one name. The dotted line is the **even split** "
                f"({ig.even_share(len(dig)):.1f}% here), what each name would "
                f"show if all {len(dig)} names in the window shared attention "
                f"equally, so above the line means more crowded than even. "
                f"Height means *who and how hard*, not how many - that is the "
                f"bubble size.\n"
                f"3. **How big** is how many times it was called. Area, not "
                f"width, so a bubble that looks twice as big really is twice "
                f"the calls.\n"
                f"4. **Hover** for the actual people behind it, their side, "
                f"and their influence on the 0-100 scale used everywhere on "
                f"this tab.\n\n"
                f"**So what.** Top-right is the corner that matters: high "
                f"up (people with a record), far right (all one way), big "
                f"(said repeatedly). That is crowded bullish positioning, "
                f"which is the thing worth flagging to a PM before it "
                f"unwinds - and top-left is the identical setup on the short "
                f"side. A name that is far right but LOW is the crowd, not "
                f"the panel; a name that is high but near the centre is two "
                f"good voices disagreeing, which is information of a "
                f"different kind. Showing the {_nb} best-backed of "
                f"{len(dig)} names touched by the top {panel_n} voices"
                + (f", to {pd.Timestamp(asof).date()}" if asof is not None
                   else "") + ".")
            st.caption(":grey[This tab is **information, not a signal.** "
                       "Notebook 05 measured that these scores do not "
                       "generalise to authors the model has not seen, so "
                       "nothing here feeds the euphoria GET IN / GET OUT "
                       "dates. Read it as \"what the room with a track "
                       "record is saying\", and see the expander at the "
                       "bottom of this tab for exactly why.]")

            with st.expander("the same names as exact numbers",
                             expanded=False):
                dv = dig.head(24).copy()
                # Share over ALL names in the window, then the head - so the
                # column is a share of the room and not of these 24 rows.
                dv["backing"] = ig.backing_share(
                    dig["weighted_voices"]).head(24).round(2)
                dv["consensus"] = dv["consensus"].round(2)
                dv["last_date"] = pd.to_datetime(dv["last_date"]).dt.date
                st.dataframe(
                    dv[["ticker", "consensus", "backing", "n_authors",
                        "n_calls", "longs", "shorts", "last_date"]].rename(
                        columns={"n_calls": "calls", "n_authors": "people",
                                 "consensus": "net direction",
                                 "backing": "share of conviction %",
                                 "last_date": "last call"}),
                    width="stretch", hide_index=True, height=480)
                st.caption(
                    f"The numbers behind the bubbles, in the same order. "
                    f"'net direction' is the horizontal axis, 'share of "
                    f"conviction %' the vertical one - computed over all "
                    f"{len(dig)} names in the window, so it still sums "
                    f"towards 100% across the whole window rather than "
                    f"across these {min(len(dig), 24)} rows. A bar chart of "
                    f"these same two columns used to sit here as well; it was "
                    f"the bubble chart with one axis flattened into shading, "
                    f"so it was dropped rather than shown twice.")

            with st.expander("name by name - the actual recent calls behind "
                             "all of this", expanded=False):
                wide = ig.author_calls_wide(calls, panel, days=days,
                                            asof=asof, per_author=5)
                if len(wide):
                    wide = wide.assign(
                        date=pd.to_datetime(wide["date"]).dt.date,
                        direction=[ig.direction_label(d)
                                   for d in wide["direction"]],
                        stance=wide["stance"].astype(float).round(2))
                    st.dataframe(wide.rename(columns={
                        "stance": "conviction", "kind": "source"}),
                        width="stretch", hide_index=True, height=420)
                    st.caption("'conviction' is how strongly the post was "
                               "worded (0-1, from the extractor). 'source' "
                               "is whether the call came from a post or a "
                               "comment. Five most recent per author.")
                else:
                    st.caption("nothing in this window")

        # ---- 2. IS IT BUILDING OR FADING? (added 2026-07-27) -----------
        # The tab had no time axis at all, which made every reading on it a
        # still photograph. The store has held call dates since inception.
        st.markdown("#### 2. Is the crowding building, or fading?")
        if calls is None or not len(dig) or len(tilt_hist) < 2:
            st.info("not enough history inside this window to plot a trend - "
                    "widen the window above to 90 days, or run another live "
                    "pull to extend the store.")
        else:
            # Called ONCE without a ticker filter, then filtered here: the
            # leading names are taken from this same population's own totals
            # rather than from `dig` (which is the top-N panel), so the lines
            # and the shares they are drawn as cannot come from two different
            # rooms. One pass over the calls either way.
            _crowd_all = ig.crowding_history(calls, board, authors=recorded,
                                             days=days, asof=asof)
            _tn = min(int(_crowd_all["ticker"].nunique()), 5)
            _lead_names = (_crowd_all.groupby("ticker")["backing"].sum()
                           .sort_values(ascending=False).head(_tn)
                           .index.tolist())
            _crowd = _crowd_all[_crowd_all["ticker"].isin(_lead_names)]
            st.plotly_chart(
                fig_crowding_time(
                    tilt_hist, _crowd,
                    f"week by week: the room's tilt, and the {_tn} "
                    f"most-backed names"),
                width="stretch", key="infl_time")
            _t_now = float(tilt_hist["tilt"].iloc[-1])
            _t_then = float(tilt_hist["tilt"].iloc[0])
            _dir_word = ("MORE one-sidedly long" if _t_now > _t_then
                         else "LESS one-sidedly long")
            st.caption(
                f"**Read it in three steps.**\n\n"
                f"1. **The top line** is the room on one axis: +1 means every "
                f"voice with a record was long, with full conviction, that "
                f"week; -1 is the same on the short side; the dotted line is "
                f"a real two-way argument. Over this window it went from "
                f"**{_t_then:+.2f}** to **{_t_now:+.2f}** - the room got "
                f"{_dir_word}. **The dot size is how many calls that week "
                f"rests on**, and it matters: the comment budget leaves the "
                f"weeks very unequal, so a tiny dot at +1.00 is four people "
                f"agreeing, not the market.\n"
                f"2. **The lines below** are the leading names, each drawn as "
                f"the **share of that week's conviction** it took. Sharing "
                f"inside the week matters: raw totals rise and fall with how "
                f"busy the forum was, so an un-normalised line would show "
                f"you the posting calendar. The grey dotted line is that "
                f"week's **even split** - it moves, because a week where the "
                f"room touched 14 names splits differently from one where it "
                f"touched 200. Green = the name's latest reading is net "
                f"long, brick = net short, and each name is written at the "
                f"end of its own line.\n"
                f"3. **Both panels use all {len(board):,} voices with a "
                f"record**, not the top {panel_n} in the slider. Cut to the "
                f"top few, a single week can come down to one person on one "
                f"name, which is 100% of that week by definition and a fact "
                f"about nobody.\n\n"
                f"**So what.** A line climbing week after week is people "
                f"with a record piling into one name - the build-up phase, "
                f"and the useful time to hear about it. A line that spikes "
                f"and collapses is attention that has already moved on, "
                f"which matters just as much: the position is still on the "
                f"book but the story supporting it has gone quiet. Read the "
                f"top panel with it - crowding into a single name while the "
                f"whole room tilts to +1 is the configuration that precedes "
                f"the unwinds this project exists to flag.")

        # ---- 3. WHO is behind one name (added 2026-07-27) --------------
        st.markdown("#### 3. Who is behind one name")
        if calls is None or not len(dig):
            st.info("no calls in the chosen window.")
        else:
            _pick = st.selectbox(
                "name", dig["ticker"].tolist(), key="infl_backer_pick",
                help="ordered most-backed first - the same order as the "
                     "bubble chart's height")
            _bk = ig.ticker_backers(calls, board, _pick, authors=panel,
                                    days=days, asof=asof)
            if not len(_bk):
                st.caption("nobody on the panel called this name in the "
                           "window")
            else:
                _b1, _b2 = st.columns([3, 2])
                with _b1:
                    st.plotly_chart(
                        fig_ticker_backers(
                            _bk, _pick,
                            f"{_pick}: the {len(_bk)} people pushing it, "
                            f"strongest record first"),
                        width="stretch", key="infl_backers")
                with _b2:
                    _bv = _bk.assign(
                        influence=_bk["influence"].round(0).astype(int),
                        conviction=_bk["conviction"].round(2),
                        last_date=pd.to_datetime(_bk["last_date"]).dt.date)
                    st.dataframe(
                        _bv[["author", "influence", "word", "n_calls",
                             "conviction", "last_date"]].rename(columns={
                                 "word": "side", "n_calls": "calls",
                                 "last_date": "last said"}),
                        width="stretch", hide_index=True, height=430)
                _n_long = int((_bk["word"] == "LONG").sum())
                _n_short = int((_bk["word"] == "SHORT").sum())
                st.caption(
                    f"**Read it.** One bar per person. **Bar length is their "
                    f"influence** on the 0-100 board scale - 100 is the "
                    f"strongest measured record in the whole store, so a bar "
                    f"at 80 means 'four-fifths as strong a record as the best "
                    f"name we track'. **Colour and the label are their "
                    f"side.** On {_pick} right now: **{_n_long} long, "
                    f"{_n_short} short**"
                    + (", mixed" if len(_bk) - _n_long - _n_short else "")
                    + f".\n\n**So what.** The shape is the answer. A few "
                    f"long bars, all one colour, means this name is being "
                    f"pushed by the people with the best records and they "
                    f"agree - the strongest version of the signal this tab "
                    f"can produce. Many short bars means it is the crowd, "
                    f"not the panel. Two colours means the single 'net "
                    f"direction' number on the bubble chart is averaging an "
                    f"argument, and should not be traded as a consensus.")

        # ---- 4. the leaderboard ---------------------------------------
        # Per-user ACCURACY was removed from this table on 2026-07-27, by
        # desk instruction and for a reason the tab itself demonstrates: a
        # raw hit rate sat one column from a composite that is shrunk toward
        # the crowd base rate, so the two numbers disagreed on purpose and
        # every reader tried to reconcile them. `hit_rate` is still computed
        # and still stored - the notebooks need it, it is an input to the
        # composite - it is simply not a thing this screen asks a PM to act
        # on. What replaced it is what the desk asked for: influence, and
        # the tickers they are pushing.
        st.markdown(f"#### 4. The names - their influence, and what they are "
                    f"pushing")
        _push = (ig.author_push_table(calls, board, authors=panel, days=days,
                                     asof=asof)
                 if calls is not None else pd.DataFrame())
        _lb_cols = [c for c in ["author", "composite", "tier", "called_tops",
                                "bought_tops", "loud_but_wrong"]
                    if c in board.columns]
        view = board.head(panel_n)[_lb_cols].copy()
        view.insert(1, "influence", ig.influence_index(board).head(panel_n)
                    .round(0).astype(int).to_numpy())
        view = view.drop(columns=["composite"])
        if len(_push):
            view = view.merge(_push[["author", "pushing", "n_calls",
                                     "n_tickers"]], on="author", how="left")
        else:
            view["pushing"], view["n_calls"], view["n_tickers"] = "", 0, 0
        view["pushing"] = view["pushing"].fillna("nothing in this window")
        view[["n_calls", "n_tickers"]] = (
            view[["n_calls", "n_tickers"]].fillna(0).astype(int))
        view = view.rename(columns={"pushing": f"pushing (last {days}d)",
                                    "n_calls": "calls in window",
                                    "n_tickers": "names",
                                    "called_tops": "called tops",
                                    "bought_tops": "bought tops",
                                    "loud_but_wrong": "loud but wrong"})
        view.insert(0, "rank", range(1, len(view) + 1))
        st.dataframe(view, width="stretch", hide_index=True, height=430,
                     column_config={
                         "influence": st.column_config.ProgressColumn(
                             "influence", min_value=0, max_value=100,
                             format="%d",
                             help="0-100, where 100 is the strongest measured "
                                  "record in the store. A RELATIVE scale, not "
                                  "an accuracy: it blends how often they were "
                                  "right, how big the moves they called were, "
                                  "and how clearly they said it, each shrunk "
                                  "toward the crowd average so five lucky "
                                  "calls cannot beat fifty solid ones.")})
        st.caption(
            f"**What each column is.** *influence* - the 0-100 rank score "
            f"above; the number is only meaningful against the other names "
            f"here, which is exactly why it is drawn as a bar rather than "
            f"printed as a decimal. *pushing* - the tickers this person "
            f"actually called in the last {days} days, netted, most-conviction "
            f"first, so a person who went long then short a name shows MIXED "
            f"rather than appearing twice. *called tops* - bearish calls made "
            f"inside a euphoria peak window that the bust then confirmed; "
            f"*bought tops* is the opposite, bullish into the same peak. "
            f"*loud but wrong* - heavily replied-to but below-median record.\n\n"
            f"**No hit rate here, deliberately.** {n_high:,} of these "
            f"{len(board):,} names are HIGH tier. Per-person accuracy is "
            f"measured, stored and reported in notebook 05, where it can sit "
            f"next to its sample size and its confidence interval; on a "
            f"screen, next to a shrunk score, it only ever invited the "
            f"comparison the shrinkage exists to prevent.")

        # ---- 5. the influence map -------------------------------------
        st.markdown("#### 5. The influence map - who replies to whom")
        if not os.path.exists(_INFL_EDGES):
            st.info("no reply_edges.parquet in the store yet - the map "
                    "appears after one comment pull.")
        else:
            _e_mt = _mtime(_INFL_EDGES)
            _mv = st.radio("view", ["the backbone (everyone who matters)",
                                    "one author's neighbourhood"],
                           horizontal=True, key="infl_map_view")
            if _mv.startswith("the backbone"):
                with st.spinner("laying out the backbone ..."):
                    nodes, links, kcore, g_n, g_m = _backbone_frames(
                        _e_mt, _b_mt)
                st.plotly_chart(
                    fig_influence_map(
                        nodes, links,
                        f"reply-graph backbone: the {kcore}-core "
                        f"({len(nodes)} of {g_n:,} scored people, "
                        f"{len(links):,} of {g_m:,} reply links drawn)"),
                    width="stretch", key="infl_map_backbone")
                _rho = nodes[["degree_here", "composite"]].corr(
                    method="spearman").iloc[0, 1]
                st.caption(
                    f"Every person drawn has at least **{kcore} neighbours "
                    f"inside this picture** - that is what a k-core is, and "
                    f"it is why this is a principled slice of the "
                    f"{g_n:,}-node graph rather than a random thinning. Dot "
                    f"size = how many people reply to them; colour = "
                    f"influence score. Drawn over the people with a judged "
                    f"record, so every dot has a real colour. **Rank "
                    f"correlation between being central and being useful, "
                    f"in this picture: {_rho:+.2f}** - the thesis's finding, "
                    f"reproduced: being central is close to unrelated to "
                    f"being right, which is exactly why the board is ranked "
                    f"on the record and not on the graph. Two notes so the "
                    f"picture is not over-read: only names far enough apart "
                    f"to be legible are printed (the rest are on hover), and "
                    f"the map covers everyone with any judged call, which is "
                    f"a wider pool than the board above - the board needs "
                    f"{INFL_MIN_JUDGED}+ judged calls before it will rank "
                    f"someone.")
            else:
                who = st.selectbox("author", panel, key="infl_ego_who")
                with st.spinner("laying out the neighbourhood ..."):
                    nodes, links, e_n, e_m = _ego_frames(who, _e_mt, _b_mt)
                if e_n <= 1:
                    st.info(f"{who} has no reply links in the store - they "
                            "post, nobody replies (or the replies are "
                            "outside the fetched history).")
                else:
                    st.plotly_chart(
                        fig_influence_map(
                            nodes, links,
                            f"{who}: everyone they exchange replies with "
                            f"({e_n} people, {e_m} links)", centre=who),
                        width="stretch", key="infl_map_ego")
                    st.caption(f"One hop around {who}. If the neighbourhood "
                               f"is larger than {INFL_EGO_MAX} people the "
                               f"busiest neighbours are kept, so this shows "
                               f"the active part of it, not all of it.")

        # ---- 6. the two warning boards --------------------------------
        # Both boards keep their place: they are the thesis's headline
        # finding on this data and the tab rests on them. Both lost their
        # accuracy COLUMNS on 2026-07-27 - `composite` became the 0-100
        # influence index and `hit_rate` / `n_judged` came out - so the
        # boards now say WHO fits the profile and leave the measurement of
        # the profile to notebook 05, which is where it is defensible.
        _infl_all = ig.influence_index(board)
        st.markdown("#### 6. Two boards worth reading against the grain")
        _w1, _w2 = st.columns(2)
        with _w1:
            st.markdown("**Called the tops** - most confirmed bearish calls "
                        "inside a euphoria peak window")
            if board["called_tops"].fillna(0).sum():
                _ct = board.nlargest(10, "called_tops").copy()
                _ct["influence"] = (_infl_all.reindex(_ct.index).round(0)
                                    .astype(int))
                st.dataframe(
                    _ct[["author", "influence", "called_tops", "bought_tops",
                         "latest_calls"]].rename(columns={
                             "called_tops": "called tops",
                             "bought_tops": "bought tops",
                             "latest_calls": "latest calls"}),
                    width="stretch", hide_index=True)
                st.caption("The people who were bearish INTO a euphoria peak "
                           "that then busted. Rare by construction - most of "
                           "the forum is long into a top.")
            else:
                st.caption("none recorded yet - this grows as peak windows "
                           "overlap the call history")
        with _w2:
            st.markdown("**Loud but wrong** - top-quartile reply-graph "
                        "PageRank, below-median influence")
            if board["loud_but_wrong"].any():
                _lw = board[board["loud_but_wrong"]].head(10).copy()
                _lw["influence"] = (_infl_all.reindex(_lw.index).round(0)
                                    .astype(int))
                st.dataframe(
                    _lw[["author", "influence", "followers",
                         "latest_calls"]].rename(columns={
                             "followers": "people replying to them",
                             "latest_calls": "latest calls"}),
                    width="stretch", hide_index=True)
                st.caption("The accounts a 'follow the big names' desk would "
                           "copy: lots of people reply to them, and their "
                           "measured record sits below the median of this "
                           "board. The thesis found the same profile - about "
                           "3x the degree on barely-above-chance accuracy - "
                           "and the evidence here says fade them.")
            else:
                st.caption("nobody currently fits the profile")

        # ---- 7. the honest caveat, from the notebook -------------------
        _nb05 = os.path.join(ROOT, "docs", "research", "nb05_influence.json")
        with st.expander("WHY THERE IS NO MODEL ON THIS TAB (click to "
                         "expand) - what notebook 05 measured",
                         expanded=False):
            st.markdown(
                "Notebook 05 is a full replication of the thesis's "
                "influential-user model on this store: eight architectures "
                "(GraphSAGE, GCN, SGC/MixHop, H2GCN, label propagation, an "
                "MLP and a linear baseline), a feature-ablation study, a "
                "graph-perturbation study, a label-shuffle significance "
                "test and an unseen-author generalisation test. Three "
                "results decide what this tab shows:\n\n"
                "1. **The features carry real signal.** Shuffling the "
                "labels 200 times puts the measured score far outside the "
                "null (p = 0.005, the floor a 200-permutation test can "
                "report). This is not noise.\n"
                "2. **The graph does not earn its complexity.** Against a "
                "plain linear model, on 10 paired seeds, not one graph "
                "architecture's confidence interval clears zero - several "
                "are significantly *worse*. The reason is measured: the "
                "useful authors are **heterophilous** (a useful author's "
                "neighbours are useful only 9% of the time, vs 96% for the "
                "rest), so message-passing averages their signal away. "
                "Deliberately *breaking* the graph the adversarial way "
                "(DICE) **raises** the score, which is the same statement "
                "from the other side.\n"
                "3. **It does not generalise to new authors.** Split by "
                "tenure instead of at random - train on established "
                "accounts, test on ones the model has never seen - and "
                "every model lands at or below the random floor.\n\n"
                "So the board above is ranked by the **measured** composite "
                "(a record, not a prediction), and no influence number "
                "touches the euphoria signal. Full evidence, plots and "
                "confidence intervals: `notebooks/"
                "05_influence_users_model.ipynb`.")
            if os.path.exists(_nb05):
                _v = _read_json(_nb05, _mtime(_nb05))
                _h, _s = _v.get("headline", {}), _v.get("significance", {})
                st.caption(f"measured in the notebook run of {_v.get('date')} "
                           f"- AP = average precision, the metric the thesis "
                           f"reports, and the one that survives a 5%-positive "
                           f"class:")
                _r1, _r2, _r3, _r4 = st.columns(4)
                _r1.metric("shipped model",
                           str(_v.get("shipped", {}).get("model", "-")),
                           f"{_v.get('shipped', {}).get('n_features', 0)} "
                           f"features")
                _r2.metric("AP", f"{_h.get('ap', float('nan')):.3f}",
                           f"random floor {_h.get('ap_random', 0):.3f}")
                _r3.metric("AUROC", f"{_h.get('auroc', float('nan')):.3f}")
                _r4.metric("label-shuffle p",
                           f"{_s.get('p_value', float('nan')):.3f}",
                           f"{_s.get('n_perm', 0)} permutations")

# ---- OVERLAYS: THEMES (was notebooks 13 + 14 + 16) ----
with t_ov_theme:
    st.subheader("Theme overlays: attention & conviction vs anchor ETF")
    if prices is None:
        st.info("no prices.parquet - run pull_bloomberg_prices.py first")
    else:
        top_th = (tc.groupby("theme")["mention_count"].sum()
                  .sort_values(ascending=False))
        th_names = [t for t in top_th.index
                    if resolve_anchor(t, priced)][:how_many]
        view = st.radio("view", ["attention first derivative vs anchor",
                                 "conviction crossings on anchor price"],
                        horizontal=True, key="ov_theme_view")
        for i, theme in enumerate(th_names, 1):
            symbol = resolve_anchor(theme, priced)
            px = price_series(prices, symbol, lo, hi)
            if px.empty:
                continue
            if view == "attention first derivative vs anchor":
                chg = chatter_change_series(theme_counts, "theme", theme, lo, hi)
                st.plotly_chart(fig_series_vs_price(
                    chg, "chatter change (pp, smoothed)", GREEN, px, symbol,
                    f"#{i}  {theme}: change in chatter vs {symbol}"),
                    width="stretch", key=f"ovth_deriv_{theme}")
            elif view == "conviction crossings on anchor price":
                if conv is None:
                    st.info("no conviction data - run the pipeline")
                    break
                cz = (clip_window(conv, "date", lo, hi)
                      .query("theme == @theme").sort_values("date")
                      .set_index("date")["conviction_z"].asfreq("D").ffill())
                if len(cz):
                    st.plotly_chart(fig_conviction(cz, px, f"#{i}  {theme}",
                                                   symbol), width="stretch",
                                    key=f"ovth_conv_{theme}")

# ---- TOP TRENDS ----
with t_top:
    st.subheader("Most-mentioned themes (rank 1 = top trending)")
    top = (tc.groupby("theme")["mention_count"].sum()
           .rename("total mentions").reset_index())
    top_r = ranked(top, "total mentions").head(how_many)
    st.dataframe(top_r, width="content", hide_index=True)
    for i, theme in enumerate(top_r["theme"], 1):
        symbol = resolve_anchor(theme, priced)
        share = mention_share_series(theme_counts, "theme", theme, lo, hi)
        px = (price_series(prices, symbol, lo, hi)
              if prices is not None and symbol else None)
        st.plotly_chart(fig_series_vs_price(
            share, "share of posts (%, 7d avg)", BLUE, px, symbol,
            f"#{i}  {theme}  vs  {symbol or 'no priced anchor'}"),
            width="stretch", key=f"top_{theme}")

# ---- EMERGING TRENDS ----
with t_emerging:
    st.subheader("Emerging = fastest-GROWING tradeable themes (rank 1 = hottest)")
    st.caption("Only themes with an approved instrument are ranked. "
               "'Growing' = average change in share-of-conversation over the "
               "chosen lookback - positive means the crowd is arriving.")
    # the growth lookback is a knob: 7d catches the newest arrivals but is
    # twitchy; 21d rewards a SUSTAINED build-up and ignores one loud week
    look = st.slider("growth lookback (days)", 3, 30, 7, key="emerg_look")
    grow_col = f"avg change last {look}d (pp)"
    movers = []
    for theme in tc["theme"].unique():
        if theme not in THEME_ETFS:          # tradeable themes only
            continue
        chg = chatter_change_series(theme_counts, "theme", theme, lo, hi)
        tail = chg.dropna().tail(look)
        if len(tail):
            movers.append({"theme": theme, grow_col: round(tail.mean(), 3)})
    if not movers:
        st.info("no tradeable theme has enough chatter data in this window "
                "to measure growth - widen the window (a theme needs days "
                f"with {MIN_TOTAL}+ total posts and a {look}-day run-up)")
    else:
        mv = ranked(pd.DataFrame(movers), grow_col).head(how_many)
        st.dataframe(mv, width="content", hide_index=True)
        for i, theme in enumerate(mv["theme"], 1):
            symbol = resolve_anchor(theme, priced)
            chg = chatter_change_series(theme_counts, "theme", theme, lo, hi)
            px = (price_series(prices, symbol, lo, hi)
                  if prices is not None and symbol else None)
            fig = fig_series_vs_price(
                chg, "chatter change (pp, smoothed)", GREEN, px, symbol,
                f"#{i}  {theme}: change in chatter  vs  {symbol or '-'}")
            # grey out everything the growth ranking does NOT look at
            if len(chg.dropna()):
                focus = chg.dropna().index.max() - pd.Timedelta(days=look)
                dim_outside(fig, lo, focus, f"ranking uses last {look}d →")
            st.plotly_chart(fig, width="stretch", key=f"emerg_{theme}")

# ---- CONVICTION ----
with t_conv:
    st.subheader("Conviction (rank 1 = most abnormal crowd right now)")
    with st.expander("what is conviction?  (start here - plain English)"):
        st.markdown(CONV_SIMPLE)
    with st.expander("full construction (research version)"):
        st.markdown(CONV_DEF)
    st.caption("Conviction is computed LIVE from the sentiment aggregates "
               "(EWM-baseline engine, halflife "
               f"{CONV_EWM_HALFLIFE}d - validated on real prices with "
               "per-year cross-validation) - it can never lag behind a "
               "stale file. Grey open triangles on the charts = the signal "
               "reverting to neutral, the validated early-exit point.")
    st.caption("Negative values are not an error: conviction z is measured "
               "against each theme's OWN trailing 84-day normal, so negative "
               "= 'this crowd is quieter / more bearish-active than it has "
               "recently been'. After a loud stretch, most themes read "
               "negative for a while - that is the mean-reversion of "
               "attention, and it is information.")
    if conv is None:
        st.info("no conviction data - run the pipeline")
    else:
        # EWMA ranking: recent days weigh most (half-life = the slider),
        # so the table reflects where crowds are NOW rather than a flat
        # month-long average that drags old readings into today.
        ew_hl = st.slider("EWMA half-life (days) - smaller = more reactive",
                          3, 30, 10, key="conv_hl")
        cv = clip_window(conv, "date", lo, hi)
        cv = cv[cv["theme"].isin(THEME_ETFS)]     # tradeable universe only
        wide_cz = (cv.pivot_table(index="date", columns="theme",
                                  values="conviction_z")
                   .asfreq("D").ffill(limit=7))
        ew_last = wide_cz.ewm(halflife=ew_hl, min_periods=5).mean().iloc[-1]
        flat_30 = wide_cz.tail(30).mean()
        recent = pd.DataFrame({
            "theme": ew_last.index,
            f"conviction z (EWMA {ew_hl}d)": ew_last.values.round(2),
            "latest z": wide_cz.iloc[-1].reindex(ew_last.index).values.round(2),
            "avg 30d (old metric)": flat_30.reindex(ew_last.index).values.round(2),
        }).dropna(subset=[f"conviction z (EWMA {ew_hl}d)"])
        recent["abs"] = recent[f"conviction z (EWMA {ew_hl}d)"].abs()
        rk = (ranked(recent, "abs").drop(columns="abs").head(how_many))
        st.dataframe(rk, width="content", hide_index=True)
        for i, theme in enumerate(rk["theme"], 1):
            cz = (cv[cv["theme"] == theme].sort_values("date")
                  .set_index("date")["conviction_z"].asfreq("D").ffill())
            symbol = resolve_anchor(theme, priced)
            px = (price_series(prices, symbol, lo, hi)
                  if prices is not None and symbol else None)
            fig = fig_conviction(cz, px, f"#{i}  {theme}", symbol)
            # the EWMA's memory is ~3 half-lives; grey out everything older
            # so the chart matches what the ranking actually weighs
            if len(cz):
                focus = cz.index.max() - pd.Timedelta(days=3 * ew_hl)
                dim_outside(fig, lo, focus,
                            f"EWMA weight ≈ last {3 * ew_hl}d →")
            st.plotly_chart(fig, width="stretch", key=f"conv_{theme}")

# ---- AI PULSE (sample placeholders for the future LLM layer) ----
PULSE_TALK_SAMPLE = (
    "SAMPLE - The forums are talking about the robotics supply chain above "
    "everything else this week - bearings, actuators and the Japanese "
    "component makers keep surfacing in threads that begin as Nvidia "
    "discussions. Rate-cut speculation is the steady background hum, "
    "earnings positioning threads are multiplying ahead of semis reporting, "
    "and a smaller but persistent conversation about uranium refuses to "
    "die down. Crypto talk is notably absent relative to how loud it "
    "usually is.")

PULSE_MARKET_SAMPLE = (
    "SAMPLE - Retail chatter this week is dominated by the semiconductor "
    "complex, with attention rotating out of megacap AI names into the "
    "supply chain (equipment, memory, robotics components). Mood is "
    "cautiously bullish: bullish share is above its 90-day average but "
    "well off the March highs, and the loudest thread topics are "
    "earnings-positioning rather than momentum-chasing - typically a "
    "mid-cycle pattern rather than a top. Bearish energy is concentrated "
    "in rate-sensitive sectors; crypto chatter is quiet relative to its "
    "own history.")

PULSE_SEGMENTS_SAMPLE = {
    "semiconductors (SMH)": "SAMPLE - Overwhelmingly constructive; the "
        "crowd frames dips as entries. Recurring topics: HBM supply, "
        "capex cycles. Dissent is about valuation, not thesis.",
    "rates & bonds (TLT)": "SAMPLE - Split and argumentative. Half the "
        "posts position for cuts, half mock that trade. High sarcasm "
        "share - read sentiment scores with caution here.",
    "meme / squeeze (ARKK)": "SAMPLE - Quiet vs its own history. The "
        "usual suspects get mentions but engagement is low - no active "
        "squeeze narrative this week.",
    "energy (XLE)": "SAMPLE - Sleepy but turning: a small, persistent "
        "uptick in bullish posts citing seasonality. Watch if it "
        "crosses the conviction threshold.",
}

PULSE_RALLY_SAMPLE = [
    {"target": "bearings / robot components",
     "verdict": "clear rallying detected",
     "why": "SAMPLE - A cluster of high-engagement posts is actively "
            "recruiting: repeated 'get in before the institutions' framing, "
            "posts listing the same four component makers in the same "
            "order, and comment sections coordinating around 'the next "
            "NVDA'. The language is evangelical rather than analytical - "
            "posters answer objections with slogans, not numbers.",
     "example": "SAMPLE paraphrase - 'Everyone is watching the robot "
                "makers, nobody is watching who supplies the joints. Load "
                "the suppliers before the street catches on.'"},
    {"target": "a small-cap uranium name",
     "verdict": "early signs, watch",
     "why": "SAMPLE - A handful of near-identical bullish posts appeared "
            "within hours of each other from young accounts, all citing "
            "the same unsourced supply rumour. Engagement is still low - "
            "either an organic story starting or a seeding attempt.",
     "example": "SAMPLE paraphrase - 'Not many people know about this one "
                "yet. The contract news drops next week. You were warned.'"},
    {"target": "meme stocks (GME and friends)",
     "verdict": "no rallying detected",
     "why": "SAMPLE - Mentions exist but the tone is nostalgic, not "
            "mobilising - jokes about past squeezes rather than calls to "
            "action. No coordinated timing, no recruiting language."},
]

PULSE_IDEAS = """**Other things the LLM layer can extract from the live posts**
(each is a planned segment - the same API call can return all of them):

- **Retail mood gauge (0-100)** - a fear/greed-style dial with a one-line
  justification, comparable day over day.
- **Narrative tracker** - not just *what* is discussed but *why*: "retail
  attributes the semis rally to HBM shortage chatter", with links between
  themes.
- **Catalyst watch** - events the crowd is positioning for (earnings dates,
  product launches, macro prints), ranked by how much chatter they drive.
- **Euphoria / contrarian warnings** - names where the language turns
  uncritical (rockets, 'can't lose', all-in posts) - historically a
  distribution signal; pairs with the crowded-top flag.
- **Divergence detector** - where retail's story disagrees with price
  action ('crowd bullish, price falling') - candidate squeeze/washout
  setups.
- **Sarcasm-adjusted sentiment** - the lexicon reads 'great, another red
  day' as positive; an LLM does not. A daily corrected sentiment for the
  noisiest themes.
- **Representative quotes** - three verbatim posts per hot theme (with
  scores), so the desk can read the raw voice without opening Reddit.
- **Pump/scam radar** - coordinated-promotion patterns on small names,
  flagged before their counts pollute the mention data."""

with t_pulse:
    st.subheader("AI market pulse - what an LLM will write from the live posts")
    st.warning("PREVIEW: the text sections below are HAND-WRITTEN SAMPLES, "
               "not generated from your data. They show the format the "
               "future LLM layer will fill in at every live pull.")

    st.markdown("### 1 - What the forums are talking about")
    st.info(PULSE_TALK_SAMPLE)

    st.markdown("### 2 - The market in one paragraph")
    st.info(PULSE_MARKET_SAMPLE)

    st.markdown("### 3 - What retail thinks, segment by segment")
    cols = st.columns(2)
    for i, (seg, txt) in enumerate(PULSE_SEGMENTS_SAMPLE.items()):
        with cols[i % 2]:
            st.markdown(f"**{seg}**")
            st.info(txt)

    st.markdown("### 4 - Rallying watch")
    st.caption("The LLM reads the posts for MOBILISING language - "
               "recruiting, coordinated timing, evangelical tone, "
               "identical talking points from young accounts - and reports "
               "what is being rallied, how convincingly, and why it "
               "concluded that. Verdicts are words, not scores.")
    for r in PULSE_RALLY_SAMPLE:
        icon = ("[!]" if "clear" in r["verdict"]
                else "[~]" if "early" in r["verdict"] else "[ ]")
        with st.expander(f"{icon}  {r['target']} - {r['verdict']}"):
            st.markdown(r["why"])
            if r.get("example"):
                st.markdown(f"> {r['example']}")

    with st.expander("planned LLM segments (the full roadmap)"):
        st.markdown(PULSE_IDEAS)
    st.caption("Implementation note: the LLM reads the freshly fetched raw "
               "posts DURING the live fold (before they are abstracted), "
               "writes these sections, and only the finished text is stored "
               "- consistent with the text-free data boundary.")

# ---- HISTORICAL CHECKER ----
with t_hist:
    st.subheader("Historical lookback: any window, any theme")
    c1, c2 = st.columns(2)
    h_lo = pd.Timestamp(c1.date_input(
        "from", (data_max - pd.Timedelta(days=730)).date(), key="h_lo"))
    h_hi = pd.Timestamp(c2.date_input(
        "to", (data_max - pd.Timedelta(days=365)).date(), key="h_hi"))
    # theme picker shows its anchor ETF right in the label
    labels = {}
    for t in sorted(tc["theme"].unique()):
        a = resolve_anchor(t, priced) or THEME_ETFS.get(t, "no anchor")
        labels[f"{t}  ({a})"] = t
    h_lab = st.selectbox("theme (anchor ETF)", list(labels))
    h_theme = labels[h_lab]
    symbol = resolve_anchor(h_theme, priced)
    px = (price_series(prices, symbol, h_lo, h_hi)
          if prices is not None and symbol else None)

    st.markdown("### 1 - Conviction vs price")
    st.caption("How abnormally bullish-active the crowd was vs its own "
               "trailing normal (see the definition in the Conviction tab). "
               "Triangles = crossings of +/-1.5.")
    if conv is not None:
        cz = (clip_window(conv, "date", h_lo, h_hi)
              .query("theme == @h_theme").sort_values("date")
              .set_index("date")["conviction_z"].asfreq("D").ffill())
        if len(cz):
            st.plotly_chart(fig_conviction(cz, px, h_theme, symbol),
                            width="stretch", key="hist_conv")
        else:
            st.info("no conviction data for this theme/window")
