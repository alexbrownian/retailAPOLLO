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
from analytics.loaders import (price_series, clip_window,          # noqa: E402
                               THEME_COUNTS)
from analytics.overlays import (mention_share_series,              # noqa: E402
                                chatter_change_series,
                                conviction_crossings, crossing_exits)

st.set_page_config(page_title="RetailRadar", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#e8845c"                       # coral - the dashboard accent
GREEN, RED, PURPLE, BLUE, GRAY = ("#3fb950", "#f85149", "#b58bd8",
                                  ACCENT, "#9aa0a6")

# Header mark: the animated bars + orbit (the same mark the pipeline
# loader uses - the five bars rise one by one, then the orbit line draws
# itself through the middle and the cycle repeats).
HEADER_MARK_HTML = """
<svg width="64" height="64" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <style>
    .hb { fill:#e8845c; transform-box:fill-box; transform-origin:50% 100%;
          transform:scaleY(0);
          animation:hbar 2.6s cubic-bezier(.4,0,.2,1) infinite; }
    .h1 { animation-delay:0s;   } .h2 { animation-delay:.14s; }
    .h3 { animation-delay:.28s; } .h4 { animation-delay:.42s; }
    .h5 { animation-delay:.56s; }
    .horb { fill:none; stroke:#e8845c; stroke-width:5;
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
    .ldb { fill:#e8845c; transform-box:fill-box; transform-origin:50% 100%;
          transform:scaleY(0);
          animation:ldbar 2.6s cubic-bezier(.4,0,.2,1) infinite; }
    .l1 { animation-delay:0s;   } .l2 { animation-delay:.14s; }
    .l3 { animation-delay:.28s; } .l4 { animation-delay:.42s; }
    .l5 { animation-delay:.56s; }
    .ldorb { fill:none; stroke:#e8845c; stroke-width:5;
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
<span style="color:#9aa0a6;">working...</span>
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

TERMINAL_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'SFMono-Regular', Consolas, 'Cascadia Mono',
                 'Courier New', monospace !important;
}
/* EXCEPTION: Streamlit's icons are a FONT (Material Symbols) - without
   this rule the monospace override above turns every icon into its
   literal ligature text, e.g. 'arrow_right' on expanders */
span[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
.material-icons, [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
[data-testid="stMetricValue"] { color: #e8845c; }
[data-testid="stMetricLabel"] { color: #9aa0a6; }
h1, h2, h3 { letter-spacing: 0.02em; }
div[data-testid="stExpander"] { border: 1px solid #24262b; border-radius: 8px; }
.rf-title { color: #e8845c; font-size: 1.7rem; font-weight: 700; }
.rf-dot { color: #3fb950; }
.rf-sub { color: #9aa0a6; font-size: 0.85rem; }
.rf-credit { color: #5b6067; font-size: 0.72rem; margin-top: 2px; }
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
def _dark(fig):
    """Terminal look for every chart: dark template, transparent card."""
    fig.update_layout(template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Consolas, monospace", size=12))
    fig.update_xaxes(gridcolor="#24262b")
    fig.update_yaxes(gridcolor="#24262b")
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
    return _axes_fidelity(_dark(fig))


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
    fig.add_vrect(x0=window_lo, x1=focus_start, fillcolor="#9aa0a6",
                  opacity=0.13, line_width=0)
    fig.add_vline(x=focus_start, line_dash="dot", line_color="#9aa0a6",
                  opacity=0.8)
    fig.add_annotation(x=focus_start, y=1.02, yref="paper",
                       yanchor="bottom", xanchor="left", showarrow=False,
                       text=label, font=dict(size=10, color="#9aa0a6"))
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
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

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
                box.markdown(f"<span style='color:#5b6067'>[ &nbsp; ] "
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
        # DANGER STATE (amber band): crowd swollen (the A1 2x bar) AND
        # price in a G2 boom. Measured (NB06): a >=10%-in-7d drop begins
        # within 30d on ~62% of these days vs ~19% of ordinary days -
        # the band IS the PM warning; alerts time the peak inside it.
        danger_runs = []
        if px is not None and not px.empty and "hype_ok" in one_i.columns \
                and prices is not None:
            full = prices[prices["symbol"] == sym].sort_values("date")
            pxa = full.set_index("date")["px_last"].asfreq("D").ffill()
            low120 = pxa.rolling(120, min_periods=60).min()
            bm = (EUPHORIA_BOOM_MIN_SINGLE if kind == "single"
                  else EUPHORIA_BOOM_MIN_ETF)
            boom = ((pxa / low120 - 1) >= bm).reindex(one_i.index).eq(True)
            danger = one_i["hype_ok"].astype(bool) & boom
            d_idx = danger[danger].index
            if len(d_idx):
                run_start = prev = d_idx[0]
                for d in list(d_idx[1:]) + [None]:
                    if d is not None and (d - prev).days <= 2:
                        prev = d
                        continue
                    danger_runs.append((run_start, prev))
                    if d is not None:
                        run_start = prev = d
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
                                     line=dict(color="#d6d8dc",
                                               width=1.5)),
                          row=1, col=1)
        # faint raw level = context; bold smoothed level = the signal
        fig.add_trace(go.Scatter(x=lvl_raw.index, y=lvl_raw.values,
                                 name="daily level (raw)",
                                 line=dict(color="#8a8f98", width=0.7),
                                 opacity=0.35),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=lvl.index, y=lvl.values,
                                 name=f"euphoria level ({ROLL}d smooth)",
                                 line=dict(color=ACCENT, width=2.2)),
                      row=2, col=1)
        # alert-ELIGIBLE stretches (crowd genuinely swollen - the A1
        # gate): the only periods where an alert is even possible.
        # This is what reconciles a wiggly level with rare alerts.
        if "hype_ok" in one_i.columns:
            # NaN gaps make plotly bridge the fill polygon diagonally
            # (wedge artifact) - fill to ZERO outside eligible stretches
            elig_y = lvl.where(one_i["hype_ok"].astype(bool),
                               0.0).values
            fig.add_trace(go.Scatter(x=lvl.index, y=elig_y,
                                     name="alert-eligible (crowd swollen)",
                                     mode="lines",
                                     line=dict(color=ACCENT, width=0),
                                     fill="tozeroy",
                                     fillcolor="rgba(232,132,92,0.20)"),
                          row=2, col=1)
        if thr_now:
            fig.add_hline(y=thr_now, line_dash="dot", line_color=RED,
                          opacity=0.7, row=2, col=1)
        def _ms(ts):
            # plotly's vline+annotation midpoint maths does Timestamp+int
            # arithmetic on some plotly/pandas versions and crashes;
            # epoch-milliseconds is numeric and works on every version
            return pd.Timestamp(ts).value / 1_000_000
        for a, b in danger_runs:      # AMBER = the danger state
            fig.add_vrect(x0=_ms(a), x1=_ms(b + pd.Timedelta(days=1)),
                          fillcolor="rgba(237,161,0,0.12)", line_width=0,
                          row=1, col=1)
        # EPISODE SPANS: shade from each START to the next END so a
        # tight start/end pair reads as what it is - a short, violent
        # euphoria episode - instead of contradictory clutter. A START
        # with no END yet shades to the newest data day (ongoing).
        # Spans are CLIPPED to the visible window - a span from alerts
        # outside it must never stretch the axis or paint empty space.
        ends_sorted = sorted(ct)
        for d in sorted(co):
            nxt = next((t for t in ends_sorted if t >= d), None)
            x1 = nxt if nxt is not None else w1
            x0c, x1c = max(d, w0), min(x1, w1)
            if x0c >= x1c:
                continue                    # span entirely off-window
            fig.add_vrect(x0=_ms(x0c), x1=_ms(x1c),
                          fillcolor="rgba(232,132,92,0.10)", line_width=0,
                          row=1, col=1)
        for d in onset_alerts:     # BLUE = GET IN (euphoria starting)
            kw = {}
            if d == max(onset_alerts):   # label only the newest - stacked
                kw = dict(annotation_text="GET IN",    # labels collide
                          annotation_position="top left",
                          annotation_font=dict(color="#2a78d6", size=11))
            fig.add_vline(x=_ms(d), line_color="#2a78d6", line_width=2,
                          opacity=0.85, **kw)
        for d in top_alerts:       # RED = GET OUT (euphoria ending)
            kw = {}
            if d == max(top_alerts):
                kw = dict(annotation_text="GET OUT",
                          annotation_position="top right",
                          annotation_font=dict(color=RED, size=11))
            fig.add_vline(x=_ms(d), line_color=RED, line_width=2,
                          opacity=0.85, **kw)
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
        _axes_fidelity(_dark(fig))
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
with t_infl:
    st.subheader("Influence tracker - who has actually been right, and "
                 "what they say now")
    st.caption("Method from Chan (Oxford M.Eng, 2026). Rank = COMPOSITE "
               "usefulness (0.4 stance-weighted accuracy + 0.4 abnormal-"
               "return-weighted + 0.2 enhanced accuracy, each Bayesian-"
               "shrunk toward the base rate so nobody looks sharp on 2 "
               "lucky calls; HIGH tier >= 0.66). Calls are judged against "
               "a bar that scales with each name's OWN volatility - a 3% "
               "move is a call on an index, noise on a meme stock. "
               "PageRank comes from the reply graph (bot-filtered); the "
               "thesis found the LOUD hubs were the least accurate group "
               "(3x the degree, barely-above-chance accuracy) - the "
               "'loud but wrong' flag marks exactly that profile here. "
               "'called tops' = bearish calls inside a euphoria peak "
               "window that the bust then confirmed. The store is "
               "committed to git (text-free, pseudonymous) and every "
               "live pipeline run extends it automatically.")
    _sc_path = os.path.join(ROOT, "data", "reference", "influence",
                            "author_scores.parquet")
    if not os.path.exists(_sc_path):
        st.info("no influence store on this machine yet - it builds "
                "ITSELF from live data: run one live pull "
                "(`python update_data.py`, or the sidebar button) and "
                "the tracker appears here after it finishes. Every "
                "later pull extends the same store - new calls are "
                "added, and recent calls are re-judged automatically "
                "once their 20-day window has prices. Nothing to "
                "rebuild, ever. (`git pull` also brings in the shared "
                "store once any machine has one; an optional "
                "`--backfill` on the comment fetcher deepens history "
                "if ever wanted.)")
    else:
        board = pd.read_parquet(_sc_path)
        board = board[board["n_judged"] >= 5]
        n_high = int((board.get("tier") == "HIGH").sum()) \
            if "tier" in board.columns else 0
        st.markdown(f"**{len(board)} authors with 5+ judged calls** "
                    f"({n_high} HIGH tier; base rate "
                    f"{board['base_rate'].iloc[0] if len(board) else '?'})")
        show_cols = [c for c in ["author", "composite", "tier", "hit_rate",
                                 "n_judged", "n_calls", "pagerank",
                                 "followers", "comment_post_ratio",
                                 "loud_but_wrong", "called_tops",
                                 "bought_tops", "latest_calls"]
                     if c in board.columns]
        view = board[show_cols].copy()
        for c in ("composite", "hit_rate", "comment_post_ratio"):
            if c in view.columns:
                view[c] = view[c].astype(float).round(3)
        if "pagerank" in view.columns:
            view["pagerank"] = (view["pagerank"].astype(float) * 1e4).round(2)
            view = view.rename(columns={"pagerank": "pagerank (x1e4)"})
        view.insert(0, "rank", range(1, len(view) + 1))
        st.dataframe(view.head(30), width="stretch", hide_index=True,
                     height=520)
        st.markdown("**Top-callers** - most confirmed euphoria-top calls")
        if "called_tops" in board.columns and board["called_tops"].sum():
            tc_cols = [c for c in ["author", "called_tops", "bought_tops",
                                   "composite", "latest_calls"]
                       if c in board.columns]
            tc_ = (board.sort_values("called_tops", ascending=False)
                   .head(10)[tc_cols])
            st.dataframe(tc_, width="stretch", hide_index=True)
        else:
            st.caption("none recorded yet - grows as peak windows overlap "
                       "the call history")
        if "loud_but_wrong" in board.columns and board["loud_but_wrong"].any():
            st.markdown("**Loud but wrong** - top-quartile PageRank, "
                        "below-median usefulness (the thesis's false-"
                        "positive profile: the accounts a 'follow the "
                        "big names' desk would copy, and the evidence "
                        "says to fade)")
            lw_cols = [c for c in ["author", "composite", "hit_rate",
                                   "n_judged", "followers", "latest_calls"]
                       if c in board.columns]
            st.dataframe(board[board["loud_but_wrong"]][lw_cols].head(10),
                         width="stretch", hide_index=True)

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
