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
INK_LABEL = "#8C8C8C"      # uppercase small labels
HAIRLINE = "#ECECEC"       # 1px rules and gridlines - never thicker
SLATE = "#7A8794"          # price line, raw/context series

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
   button's children explicitly. */
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stButton button * {{
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
    "fetch":    ("Fetching new posts (Reddit / X / StockTwits)",
                 ["sources IN PARALLEL", "still fetching", "fetch done"]),
    "store":    ("Adding the new posts to the data store",
                 ["APPEND:", "folding live raw", "merging live raw",
                  "MERGE:", "live fast path", "hydrated ABSTRACTED_DATA"]),
    "rebuild":  ("Rebuilding the aggregates from raw text (long step)",
                 ["full chain: building aggregates",
                  "building rolling term counts", "need scoring"]),
    "coverage": ("Checking data coverage for the window",
                 ["DATA COVERAGE", "WINDOW CHECK"]),
    "analyse":  ("Analysing: conviction, signals, euphoria + onset radar",
                 ["recomputing conviction", "analytics:",
                  "conviction (was nb", "signals (was nb",
                  "phases (the onset detector",
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
if st.sidebar.button("run LIVE pull now  (~3-10 min)", disabled=_pipe_running,
                     help="Fetch new posts from all three sources, fold "
                          "them in, recompute signals, pull prices. Most "
                          "of the time is deliberate API rate-limit pacing "
                          "(X waits 5s between requests); Reddit is "
                          "incremental after the first run of the day."):
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
# comments are the slow fetch (10-50x post volume at the API's polite
# 1s/page) so they left the daily pull (desk decision 2026-07-24) and
# live behind their own button; the estimate is watermark-aware
_cwm = os.path.join(REFERENCE_DIR, "reddit_comments_watermark.json")
_c_est = "~1-4 min" if os.path.exists(_cwm) else "first run ~10-25 min"
if st.sidebar.button(f"pull comments + influence board  ({_c_est})",
                     disabled=_pipe_running,
                     help="Fetch new Reddit comments (watermarked and "
                          "resumable - cancelling is always safe) and "
                          "update the influence board: new calls judged, "
                          "reply graph extended, tiers rescored. The "
                          "daily LIVE pull no longer includes comments; "
                          "run this when you want the board refreshed."):
    start_pipeline([(["update_comments.py"], None)],
                   "comments + influence", plan="comments")
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
with st.expander("MODEL DECISIONS & EVIDENCE (click to expand) - why every "
                 "rule is the way it is, with the tested numbers",
                 expanded=False):
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


EUPHORIA_DEF = """**EUPHORIA = the crowd has stopped analysing and started
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
    with st.expander("what is euphoria? (definitions & headline record)"):
        st.markdown(EUPHORIA_DEF)
    _reg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "docs", "PARAMETER_REGISTER.md")
    if os.path.exists(_reg):
        with st.expander("every number & its reason (the parameter "
                         "register - nothing in this system is a magic "
                         "constant)"):
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

**Why a call counts as right** is measured against a bar that scales with
each name's OWN volatility: a 3% move is a real call on an index and noise
on a meme stock.

**"Loud but wrong"** is the thesis's headline warning, reproduced here: the
accounts in the top quartile of reply-graph PageRank but below median
usefulness. They are the accounts a *follow-the-big-names* desk would copy,
and the evidence says fade them.

**The map** shows who replies to whom. Position comes from a force layout
(people who reply to each other get pulled together), dot size is how many
people reply to *them*, colour is the usefulness score. Only the **k-core
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
    the ~5k who have made at least one judged call have a usefulness score.
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
       usefulness sit inside the same dense cluster - that is what a reply
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
    dropped and that author lives on hover only. Priority is usefulness
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
    for col in ("composite", "hit_rate", "degree_here", "n_judged",
                "community"):
        if col not in n.columns:
            n[col] = float("nan")
    n["composite"] = pd.to_numeric(n["composite"], errors="coerce").fillna(0.0)
    fig.add_trace(go.Scatter(
        x=n["x"], y=n["y"], mode="markers", showlegend=False,
        marker=dict(size=7 + 22 * _unit(n["degree_here"]),
                    color=n["composite"], colorscale="Oranges",
                    cmin=0.0, cmax=1.0, line=dict(width=0.5, color=WHITE),
                    colorbar=dict(title="usefulness", thickness=10, len=0.7)),
        customdata=n[["author", "composite", "hit_rate", "n_judged",
                      "degree_here", "community"]].to_numpy(),
        hovertemplate=("<b>%{customdata[0]}</b><br>usefulness "
                       "%{customdata[1]:.3f}<br>hit rate "
                       "%{customdata[2]:.0%} of %{customdata[3]:.0f} judged"
                       "<br>%{customdata[4]:.0f} people reply to them"
                       "<br>conversation cluster %{customdata[5]:.0f}"
                       "<extra></extra>")))
    # label the strongest voices, but never two on top of each other: the
    # centre of an ego view always wins, then usefulness order decides.
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


def _rgba(hex_colour: str, alpha: float) -> str:
    """#rrggbb -> rgba(r,g,b,a); used to fade a bar by weight of evidence."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


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


def fig_influence_bubbles(dig: pd.DataFrame, voices: pd.DataFrame,
                          title: str):
    """THE lead influence chart: what the people with a record are pushing.

    Four readings in one picture, which is why this replaced the bar chart
    as the first thing on the tab:

      * X = net direction. Left is short, right is long.
      * Y = influence behind it - the summed usefulness of everyone calling
        it. High means people with an actual record, not first-timers.
      * SIZE = how many times it was called. Big means repeatedly, not once.
      * COLOUR = muted green long, muted brick short.

    So the eye goes straight to the TOP-RIGHT: heavily-backed, repeatedly
    called, bullish - which is precisely the crowding a PM is asked to watch.
    Bottom-left is the same thing on the short side; anything near x=0 is a
    genuine disagreement and is labelled as such rather than hidden.

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
    # Bubble AREA scales with calls (plotly's sizemode="area"), because area
    # is what the eye actually compares - scaling the RADIUS by the value
    # makes a 4x count look 16x bigger, which is the classic bubble lie.
    smax = float(d["n_calls"].max()) or 1.0
    fig = go.Figure(go.Scatter(
        x=d["consensus"], y=d["weighted_voices"], mode="markers+text",
        text=d["ticker"], textposition="top center",
        textfont=dict(size=10, color=INK),
        marker=dict(
            size=d["n_calls"], sizemode="area",
            sizeref=2.0 * smax / (44.0 ** 2), sizemin=6,
            color=[BULL if c >= 0 else BEAR for c in d["consensus"]],
            opacity=0.72, line=dict(width=1, color=WHITE)),
        customdata=d[["ticker", "n_authors", "n_calls", "longs", "shorts",
                      "weighted_voices", "voices"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "net direction %{x:+.2f}  (+1 all long, -1 all short)<br>"
            "influence behind it %{customdata[5]:.2f}<br>"
            "%{customdata[1]:.0f} people, %{customdata[2]:.0f} calls "
            "(%{customdata[3]:.0f} long / %{customdata[4]:.0f} short)"
            "<br><br><b>who</b><br>%{customdata[6]}<extra></extra>")))
    # Quadrant guides: a hairline at x=0 and a quiet label in each upper
    # corner. No shaded quadrants - the same reason the euphoria chart lost
    # its bands, and here the bubbles need every pixel of contrast.
    fig.add_vline(x=0, line_color=HAIRLINE, line_width=1)
    ymax = float(d["weighted_voices"].max() or 1.0)
    for xpos, xanch, label, colour in (
            (-1.0, "left", "CROWDED SHORT", BEAR),
            (1.0, "right", "CROWDED LONG", BULL)):
        fig.add_annotation(
            x=xpos, y=ymax * 1.12, xanchor=xanch, yanchor="top",
            text=label, showarrow=False,
            font=dict(size=9, color=colour))
    fig.add_annotation(
        x=0, y=ymax * 1.12, xanchor="center", yanchor="top",
        text="GENUINE DISAGREEMENT", showarrow=False,
        font=dict(size=9, color=INK_LABEL))
    fig.update_layout(
        title=dict(text=title, y=0.97, x=0.01), height=520,
        margin=dict(l=10, r=10, t=64, b=58), showlegend=False)
    fig.update_xaxes(range=[-1.18, 1.18], zeroline=False,
                     title_text="<- SHORT          net direction"
                                "          LONG ->")
    fig.update_yaxes(range=[0, ymax * 1.22], rangemode="tozero",
                     title_text="influence behind it "
                                "(bubble size = number of calls)")
    return _theme(fig)


def fig_consensus(dig: pd.DataFrame, title: str):
    """What the panel is suggesting, per ticker: an influence-weighted net
    direction in [-1, +1]. +1 = every voice with a record is long this name
    with full conviction, -1 = every one is short.

    TWO things are encoded, and keeping them apart is the whole point:

      * bar LENGTH  = direction and agreement (the consensus number). This
        saturates easily - three people all long a name and it reads +1.00,
        exactly as loudly as thirteen people all long a name.
      * bar OPACITY = how much backing that reading actually has
        (`weighted_voices`, the sum of the callers' usefulness scores). A
        faint +1.00 is a thin +1.00. Without this the chart tells a
        10-call/3-author name and a 49-call/13-author name apart nowhere.

    Rows are ordered by `weighted_voices` (most-backed at the TOP), which is
    the digest's own order and what the title claims - the earlier version
    sorted by consensus while saying "most-backed first".

    Opacity floor 0.30 is presentation only: a fully transparent bar would
    read as missing data rather than as weakly-backed.
    """
    d = dig.dropna(subset=["consensus"]).copy()
    # plotly draws the FIRST category at the bottom of a horizontal bar
    # chart, so ascending here puts the most-backed name at the top.
    d = d.sort_values(["weighted_voices", "n_calls"], ascending=True)
    alpha = 0.30 + 0.70 * _unit(d["weighted_voices"]).pow(0.5)
    labels = [f"{t}  ({int(a)} voice{'s' if a != 1 else ''})"
              for t, a in zip(d["ticker"], d["n_authors"])]
    fig = go.Figure(go.Bar(
        x=d["consensus"], y=labels, orientation="h",
        marker_color=[_rgba(GREEN if c >= 0 else RED, a)
                      for c, a in zip(d["consensus"], alpha)],
        customdata=d[["n_authors", "n_calls", "longs", "shorts",
                      "weighted_voices", "ticker"]].to_numpy(),
        hovertemplate=("<b>%{customdata[5]}</b><br>net direction %{x:+.2f}"
                       "<br>%{customdata[0]:.0f} authors, "
                       "%{customdata[1]:.0f} calls"
                       "<br>%{customdata[2]:.0f} long / "
                       "%{customdata[3]:.0f} short"
                       "<br>backing (sum of their usefulness) "
                       "%{customdata[4]:.2f}<extra></extra>")))
    fig.update_layout(title=dict(text=title, y=0.97, x=0.01),
                      height=max(280, 26 * len(d) + 110),
                      margin=dict(l=10, r=10, t=55, b=52),
                      xaxis_title="<- SHORT      influence-weighted net "
                                  "direction      LONG ->      "
                                  "(faded bar = few voices behind it)")
    fig.update_xaxes(range=[-1.05, 1.05], zeroline=True,
                     zerolinecolor=HAIRLINE)
    return _theme(fig)


with t_infl:
    st.subheader("Influence tracker - who has actually been right, and "
                 "what they are saying now")
    st.caption("INFORMATION ONLY - nothing on this tab feeds the euphoria "
               "level or the GET IN / GET OUT alerts. Ranking is the "
               "MEASURED record from the store, not a model prediction.")
    with st.expander("HOW TO READ THIS TAB (click to expand) - what the "
                     "score means and how it was built", expanded=False):
        st.markdown(INFLUENCE_HOW_TO_READ)

    if not os.path.exists(_INFL_SCORES):
        st.info("no influence store on this machine yet - it builds "
                "ITSELF from live data: run one live pull "
                "(`python update_comments.py`, or the sidebar button) and "
                "the tracker appears here after it finishes. Every "
                "later pull extends the same store - new calls are "
                "added, and recent calls are re-judged automatically "
                "once their 20-day window has prices. Nothing to "
                "rebuild, ever. (`git pull` also brings in the shared "
                "store once any machine has one.)")
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
        n_high = int((board["tier"] == "HIGH").sum())
        base = float(board["base_rate"].iloc[0]) if len(board) else float("nan")
        dig = (ig.suggestion_digest(calls, board, authors=panel, days=days,
                                    asof=asof)
               if calls is not None else pd.DataFrame())

        _k1, _k2, _k3, _k4, _k5 = st.columns(5)
        _k1.metric("voices with a record", f"{len(board):,}",
                   help=f"{INFL_MIN_JUDGED}+ calls already judged against "
                        f"prices, out of {len(board_all):,} authors tracked")
        _k2.metric("HIGH tier", n_high, help="composite score >= 0.66")
        _k3.metric("crowd base rate",
                   "n/a" if pd.isna(base) else f"{base:.0%}",
                   help="how often an average call is right - the bar every "
                        "score is shrunk toward and measured against")
        if len(dig):
            _lead = dig.iloc[0]
            _net = float((dig["consensus"] * dig["weighted_voices"]).sum()
                         / dig["weighted_voices"].sum())
            _k4.metric("most-backed name", str(_lead["ticker"]),
                       f"{_lead['consensus']:+.2f} net",
                       help="the ticker carrying the most influence-weighted "
                            "conviction in the chosen window")
            _k5.metric("panel tilt", f"{_net:+.2f}",
                       "net long" if _net >= 0 else "net short",
                       help="all their calls in the window, weighted by "
                            "record and conviction, on one -1 to +1 scale")
        else:
            _k4.metric("most-backed name", "-")
            _k5.metric("panel tilt", "-")

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
            st.plotly_chart(
                fig_influence_bubbles(
                    dig.head(_nb), voices,
                    f"the {_nb} most-backed names, last {days} days"),
                width="stretch", key="infl_bubbles")
            st.caption(
                f"**How to read it.** Right of the centre line is net "
                f"**long**, left is net **short** - that is the horizontal "
                f"axis. **Height** is how much measured record is behind the "
                f"call (everyone's usefulness score, summed), so a name high "
                f"up is being pushed by people who have been right before, "
                f"not by first-timers. **Bubble size** is how many times it "
                f"was called, by area, so a bubble twice as wide really is "
                f"four times the calls. **Colour** just repeats the "
                f"direction so the picture survives a black-and-white "
                f"printout.\n\n"
                f"**So what.** The top-right corner is the trade-relevant "
                f"one: heavily-backed, repeatedly called, bullish - crowded "
                f"positioning, which is the thing worth flagging before it "
                f"unwinds. Top-left is the same crowding on the short side. "
                f"Anything sitting near the centre line is a real "
                f"disagreement, not a signal. **Hover any bubble** for the "
                f"individual people behind it, which way each of them leans "
                f"and how useful their record has been. Showing the "
                f"{_nb} best-backed of {len(dig)} names touched by the top "
                f"{panel_n} voices"
                + (f", to {pd.Timestamp(asof).date()}" if asof is not None
                   else "") + ".")
            st.caption(":grey[This tab is **information, not a signal.** "
                       "Notebook 05 measured that these scores do not "
                       "generalise to authors the model has not seen, so "
                       "nothing here feeds the euphoria GET IN / GET OUT "
                       "dates. Read it as \"what the room with a track "
                       "record is saying\", and see the expander at the "
                       "bottom of this tab for exactly why.]")

            with st.expander("the same names ranked, with the numbers",
                             expanded=False):
                _g1, _g2 = st.columns([3, 2])
                with _g1:
                    st.plotly_chart(
                        fig_consensus(dig.head(18),
                                      "panel consensus by name (most-backed "
                                      "first)"),
                        width="stretch", key="infl_consensus")
                with _g2:
                    dv = dig.head(18).copy()
                    dv["consensus"] = dv["consensus"].round(2)
                    dv["weighted_voices"] = dv["weighted_voices"].round(2)
                    dv["last_date"] = pd.to_datetime(dv["last_date"]).dt.date
                    st.dataframe(
                        dv.rename(columns={
                            "n_calls": "calls", "n_authors": "people",
                            "consensus": "net direction",
                            "weighted_voices": "influence behind it",
                            "last_date": "last call"}),
                        width="stretch", hide_index=True, height=560)
                st.caption("Bars right of the line are net LONG, left of it "
                           "net SHORT; length is how one-sided the panel is, "
                           "not how big a position should be. A **faded** bar "
                           "means few voices are behind that reading - three "
                           "people agreeing scores the same +1.00 as thirteen "
                           "agreeing, so read length and fade together. This "
                           "is the bubble chart above with the vertical axis "
                           "flattened into shading, kept for the exact "
                           "numbers.")

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

        # ---- 2. the leaderboard ---------------------------------------
        st.markdown(f"#### 2. The names behind it - top {panel_n} by "
                    f"measured record")
        _lb_cols = [c for c in ["author", "composite", "tier", "hit_rate",
                                "n_judged", "n_calls", "called_tops",
                                "bought_tops", "loud_but_wrong",
                                "latest_calls"] if c in board.columns]
        view = board.head(panel_n)[_lb_cols].copy()
        for c in ("composite", "hit_rate"):
            view[c] = view[c].astype(float).round(3)
        view = view.rename(columns={"composite": "usefulness",
                                    "hit_rate": "hit rate",
                                    "n_judged": "calls judged",
                                    "n_calls": "calls made",
                                    "called_tops": "called tops",
                                    "bought_tops": "bought tops",
                                    "loud_but_wrong": "loud but wrong",
                                    "latest_calls": "latest calls"})
        view.insert(0, "rank", range(1, len(view) + 1))
        st.dataframe(view, width="stretch", hide_index=True, height=430,
                     column_config={
                         "usefulness": st.column_config.ProgressColumn(
                             "usefulness", min_value=0.0, max_value=1.0,
                             format="%.3f",
                             help="0-1 composite: accuracy + size of the "
                                  "moves called + extractor confidence, "
                                  "each shrunk toward the crowd base rate"),
                         "hit rate": st.column_config.NumberColumn(
                             "hit rate", format="%.0f%%",
                             help="raw share of judged calls that worked - "
                                  "UNshrunk, so read it next to 'calls "
                                  "judged'")})
        st.caption("'called tops' = bearish calls made inside a euphoria "
                   "peak window that the bust then confirmed. 'bought tops' "
                   "= the opposite, bullish into the same peak. 'hit rate' "
                   "is raw and unshrunk; 'usefulness' is the number the "
                   "board is ranked on.")

        # ---- 3. the influence map -------------------------------------
        st.markdown("#### 3. The influence map - who replies to whom")
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
                    f"usefulness score. Drawn over the people with a judged "
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

        # ---- 4. the two warning boards --------------------------------
        st.markdown("#### 4. Two boards worth reading against the grain")
        _w1, _w2 = st.columns(2)
        with _w1:
            st.markdown("**Called the tops** - most confirmed bearish calls "
                        "inside a euphoria peak window")
            if board["called_tops"].fillna(0).sum():
                st.dataframe(
                    board.nlargest(10, "called_tops")[
                        ["author", "called_tops", "bought_tops", "composite",
                         "latest_calls"]].round(3),
                    width="stretch", hide_index=True)
            else:
                st.caption("none recorded yet - this grows as peak windows "
                           "overlap the call history")
        with _w2:
            st.markdown("**Loud but wrong** - top-quartile reply-graph "
                        "PageRank, below-median usefulness")
            if board["loud_but_wrong"].any():
                st.dataframe(
                    board[board["loud_but_wrong"]][
                        ["author", "composite", "hit_rate", "n_judged",
                         "followers", "latest_calls"]].head(10).round(3),
                    width="stretch", hide_index=True)
                st.caption("The accounts a 'follow the big names' desk would "
                           "copy. The thesis found this profile - 3x the "
                           "degree, barely-above-chance accuracy - and the "
                           "evidence here says fade them.")
            else:
                st.caption("nobody currently fits the profile")

        # ---- 5. the honest caveat, from the notebook -------------------
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
    with st.expander("what is conviction? (definition)"):
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
