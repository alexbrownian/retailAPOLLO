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
                        MIN_GAP, HOLD_DAYS, PROCESSED_DIR, PRICES_PATH,
                        CONV_EXIT_LEVEL, CONV_EWM_HALFLIFE, DESK_EXIT_Z)
from src.themes import THEME_ETFS, THEME_ETF_FALLBACKS             # noqa: E402
from analytics import overlays                                     # noqa: E402
from analytics.loaders import (price_series, clip_window,          # noqa: E402
                               THEME_COUNTS, THEME_SIGNALS)
from analytics.overlays import (mention_share_series,              # noqa: E402
                                chatter_change_series,
                                conviction_crossings, crossing_exits,
                                signal_scorecard, trade_desk,
                                certainty_table)

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
`docs/ARCHITECTURE.md` section 6.1."""

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

SIGNAL_DEF = """### The philosophy: fewer trades, more conviction

A signal only fires when **momentum and sentiment agree** - neither alone
is enough. Everything is measured against each theme's OWN trailing 84-day
baseline (never whole-window statistics), so a signal on day *t* uses only
information available on day *t* - what you see in a backtest is exactly
what the live run would have produced.

### What makes a BUY

All three must hold on the same day:

1. **A momentum trigger CROSSES up.** Attention z or conviction z crosses
   above **+K (2.5)** - crossing means *yesterday <= K, today > K*, so one
   surge produces exactly one trade, not a signal every day the surge
   lasts. K=2.5 means only the top ~1% most abnormal days for that theme
   even qualify.
2. **Sentiment agrees.** The 5-day change of the net-bullish share is
   POSITIVE - the mood is improving vs its own recent past, not just loud.
3. **Score >= 4 of 5** (see the checklist below).

### What makes a SELL

The mirror image, with a deliberately harder bar (retail skews bullish, so
bearish evidence must be stronger): a bearish trigger (conviction z crosses
below **-K**, or the *crowded-top* divergence activates - attention above K
while the mood deteriorates, the classic distribution/top pattern), plus a
NEGATIVE 5-day sentiment change, plus a sell score >= **4 of 5**.

### The score: one point per independent check

| # | BUY check | SELL check |
|---|---|---|
| 1 | attention z > K (crowd unusually large) | same |
| 2 | 5d sentiment change > 0 (mood improving) | 5d change < 0 (deteriorating) |
| 3 | conviction z > K (crowd large AND bullish) | conviction z < -K |
| 4 | crowded-top flag NOT active | crowded-top flag ACTIVE |
| 5 | Reddit AND X mentions both rising (where X has coverage) | same |

**Score 5/5** = every independent line of evidence agreed;
**4/5** = the minimum that trades. The `reason` column of every signal
spells out exactly which checks fired with their actual numbers - no
signal is a black box.

### Glossary of the columns

- **attention z (att_z)** - how unusually LARGE the crowd is: 7-day rolling
  mentions vs the theme's own trailing 84-day normal. Says nothing about
  direction, only size.
- **conviction z (conv_z)** - how unusually BULLISH-ACTIVE the crowd is:
  bullish-minus-bearish post votes, 7-day rolling, same trailing baseline.
  Size x direction in one number.
- **sentiment 5d change (sent_5d_chg)** - is the *mood itself* improving or
  deteriorating: the 5-day change in the share of bullish posts. The
  earliest, twitchiest ingredient.
- **crowded top** - attention > K while sentiment deteriorates: everyone is
  watching but enthusiasm is fading, i.e. whoever wanted to buy already
  has. Counts FOR a sell, AGAINST a buy.
- **signal date vs action date** - the signal is computed on day *t* from
  data through day *t*; the order is stamped for the NEXT day (no
  look-ahead).
- **exit by** - every suggestion is a **20-day hold** (chosen from the
  horizon analysis: the edge peaks and plateaus around 3-4 weeks).
- **cooldown (21d)** - once a theme signals, the SAME side is suppressed
  for 21 days: one episode, one trade.
- **certainty** - the desk's ranking metric: score (breadth of evidence)
  + |conviction z| capped at 3 (strength) + a recency bonus fading over
  90 days (a live edge beats an old one)."""


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


def fig_signals(px_line, sig_rows, name, symbol, cz=None):
    """BUY/SELL triangles placed on the price line, hover shows the score
    and conviction z of each trade. When the theme's conviction series
    `cz` is supplied, each trade also gets a GREY open triangle at its
    exit point: the first day conviction reverts to neutral after entry
    (BUY: z back below +DESK_EXIT_Z; SELL: z back above -DESK_EXIT_Z),
    or the HOLD_DAYS cap if it never reverts - the same rule as the trade
    desk's REVERTED hint, so chart and ledger always agree."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px_line.index, y=px_line.values,
                             name=f"{symbol} price",
                             line=dict(color=GRAY, width=1.5)))
    exit_pts, exit_texts = [], []
    for side, mk, col in [("BUY", "triangle-up", GREEN),
                          ("SELL", "triangle-down", RED)]:
        rows = sig_rows[sig_rows["action"] == side]
        pts, texts = [], []
        for _, r in rows.iterrows():
            d = r["action_date"]
            if px_line.empty:
                continue
            i = px_line.index.get_indexer([d], method="nearest")[0]
            pts.append((px_line.index[i], px_line.iloc[i]))
            texts.append(f"{side} {d.date()}<br>score {r.get('score', '?')}/5"
                         f"<br>conv z {r.get('conv_z', float('nan')):+.2f}")
            # the exit companion for this trade
            if cz is not None and len(cz):
                path = cz[d:d + pd.Timedelta(days=HOLD_DAYS)]
                hit = (path[path < DESK_EXIT_Z] if side == "BUY"
                       else path[path > -DESK_EXIT_Z])
                if len(hit):
                    x_d, why = hit.index[0], "conviction reverted to neutral"
                else:
                    x_d, why = d + pd.Timedelta(days=HOLD_DAYS), f"{HOLD_DAYS}d cap"
                if x_d <= px_line.index.max():
                    j = px_line.index.get_indexer([x_d], method="nearest")[0]
                    exit_pts.append((px_line.index[j], px_line.iloc[j]))
                    exit_texts.append(f"EXIT {side} {x_d.date()}<br>{why}")
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], name=side,
                mode="markers", hovertext=texts, hoverinfo="text",
                marker=dict(symbol=mk, size=14, color=col,
                            line=dict(color="black", width=1))))
    if exit_pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in exit_pts], y=[p[1] for p in exit_pts],
            name="exit (reverted to neutral / cap)", mode="markers",
            hovertext=exit_texts, hoverinfo="text",
            marker=dict(symbol="circle-open", size=11, color=GRAY,
                        line=dict(color=GRAY, width=2))))
    fig.update_layout(title=dict(text=f"{name} ({symbol}): BUY/SELL signals",
                                 y=0.97, x=0.01),
                      height=430, hovermode="closest",
                      margin=dict(l=10, r=10, t=55, b=20),
                      legend=dict(orientation="h", yanchor="top", y=-0.28))
    return _axes_fidelity(_dark(fig))

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
sig_file = load(THEME_SIGNALS)


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


def theme_cz(theme):
    """One theme's conviction z as a daily forward-filled series (for the
    grey exit markers on signal charts). None when conviction is absent."""
    if conv is None:
        return None
    one = conv[conv["theme"] == theme]
    if not len(one):
        return None
    return (one.sort_values("date").set_index("date")["conviction_z"]
            .asfreq("D").ffill())
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
    "analyse":  ("Analysing: conviction scores + trade signals",
                 ["recomputing conviction", "analytics:",
                  "conviction (was nb", "signals (was nb",
                  "THEME decisions", "analytics finished"]),
    "prices":   ("Downloading prices from Bloomberg",
                 ["BLOOMBERG PRICE PULL", "pulling Bloomberg prices",
                  "requesting "]),
    "wrapup":   ("Safety check + wrap-up",
                 ["snapshot ->", "safety check", "RUN SUMMARY"]),
}
# which stages each pipeline actually goes through (in order)
PLANS = {
    "live":      ["fetch", "store", "coverage", "analyse", "prices", "wrapup"],
    "window":    ["prices", "coverage", "analyse", "wrapup"],
    "analytics": ["analyse"],
    "full":      ["fetch", "store", "rebuild", "analyse", "prices", "wrapup"],
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
_sig_n = len(clip_window(sig_file, "action_date", lo, hi)) if sig_file is not None else 0
_open_n = 0
if sig_file is not None and len(sig_file):
    _open_n = int((sig_file["action_date"]
                   > today - pd.Timedelta(days=HOLD_DAYS)).sum())
_m1.metric("signals in window", _sig_n)
_m2.metric("open positions", _open_n)
_m3.metric("tradeable themes", int(theme_counts[theme_counts["theme"].isin(THEME_ETFS)]["theme"].nunique()))
_m4.metric("data through", str(data_max.date()))
_m5.metric("priced symbols", len(priced))

# ---- TICKER LOOKUP: one instrument, its suggestions + the reasons ----
with st.expander("INSTRUMENT LOOKUP (click to expand) - all suggestions & "
                 "reasons for one tradeable instrument", expanded=False):
    if sig_file is None or not len(sig_file):
        st.info("no signals on file yet")
    else:
        lk_opts = sorted(sig_file["etf"].dropna().unique())
        lk = st.selectbox("instrument", lk_opts, key="lookup_etf")
        lk_rows = (sig_file[sig_file["etf"] == lk]
                   .sort_values("action_date", ascending=False))
        if not len(lk_rows):
            st.info(f"no signals ever recorded for {lk}")
        else:
            latest = lk_rows.iloc[0]
            st.markdown(
                f"**latest: {latest['action']} {lk} on "
                f"{latest['action_date'].date()}** - theme "
                f"{latest.get('theme', '?')}, score {latest.get('score', '?')}/5, "
                f"conv z {latest.get('conv_z', float('nan')):+.2f}, exit by "
                f"{(latest['action_date'] + pd.Timedelta(days=HOLD_DAYS)).date()}")
            if latest.get("reason"):
                st.markdown(f"why: _{latest['reason']}_")
            lk_show = [c for c in ["action_date", "action", "theme", "score",
                                   "att_z", "conv_z", "sent_5d_chg", "reason"]
                       if c in lk_rows.columns]
            # round only the numeric columns (rounding a datetime warns)
            lk_view = lk_rows[lk_show].copy()
            for c in ("att_z", "conv_z", "sent_5d_chg"):
                if c in lk_view.columns:
                    lk_view[c] = lk_view[c].round(2)
            st.dataframe(lk_view, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------
# ---- MODEL DECISIONS & EVIDENCE: the audit trail of every rule ----
with st.expander("MODEL DECISIONS & EVIDENCE (click to expand) - why every "
                 "rule is the way it is, with the tested numbers",
                 expanded=False):
    st.markdown(DECISIONS_DOC)

# NOTE: individual-ticker overlays were removed from the dashboard by
# request - the desk trades THEMES via their anchor ETFs, never single
# tickers. The ticker analytics remain available in analytics/ for
# research (windowed backtests via run_analytics --what signals).
(t_desk, t_ov_theme, t_top, t_emerging, t_conv,
 t_pulse, t_hist) = st.tabs(
    ["Trade desk (live)", "Overlays: themes",
     "Top trends", "Emerging trends", "Conviction", "AI Pulse (sample)",
     "Historical checker"])

tc = clip_window(theme_counts, "date", lo, hi)
# TRADEABLE UNIVERSE ONLY, everywhere: every list/rank/picker on this
# dashboard is restricted to themes with a firm-approved instrument
# (THEME_ETFS). Non-tradeable themes (crypto, cannabis, small_caps) are
# still tracked in the data - they are simply not shown on the desk.
tc = tc[tc["theme"].isin(THEME_ETFS)]


# ---- TRADE DESK: dated live suggestions, most recent first ----
with t_desk:
    st.subheader("Model suggestions - most recent first, 20-day holds")
    with st.expander("how a BUY/SELL is decided - full definition & glossary"):
        st.markdown(SIGNAL_DEF)
    if sig_file is None or not len(sig_file):
        st.info("no signals on file - run the pipeline")
    else:
        # follow ONE instrument: filters the table, scorecard, ranking and
        # charts below to just that ETF
        etf_opts = (["ALL (every instrument)"]
                    + sorted(x for x in sig_file["etf"].dropna().unique()))
        pick_etf = st.selectbox("follow one ETF (filters everything below)",
                                etf_opts, key="desk_etf")
        sig_w = clip_window(sig_file, "action_date", lo, hi)
        if pick_etf != "ALL (every instrument)":
            sig_w = sig_w[sig_w["etf"] == pick_etf]
        if not len(sig_w):
            st.info("no signals in this window"
                    + ("" if pick_etf.startswith("ALL")
                       else f" for {pick_etf}"))
        else:
            desk = trade_desk(sig_w.head(200) if len(sig_w) > 200 else sig_w,
                              prices, priced, today)
            # EXIT HINT: an OPEN trade whose theme conviction has reverted
            # to neutral (|z| < DESK_EXIT_Z) has lost the surge that
            # produced it - the study showed exiting there frees capital
            # ~2x faster at a better %/day than waiting out the 20d cap
            if conv is not None and len(desk):
                latest_z = (conv.sort_values("date").groupby("theme")
                            ["conviction_z"].last())
                def _hint(row):
                    if row["status"] != "OPEN":
                        return ""
                    z_now = latest_z.get(row["theme"])
                    if z_now is None or pd.isna(z_now):
                        return "no z"
                    reverted = (abs(z_now) < DESK_EXIT_Z
                                if row["action"] == "BUY"
                                else z_now > -DESK_EXIT_Z)
                    return (f"z {z_now:+.1f} REVERTED - consider exit"
                            if reverted else f"z {z_now:+.1f} still on")
                desk["exit hint"] = desk.apply(_hint, axis=1)
            desk.insert(0, "rank", range(1, len(desk) + 1))
            open_n = int((desk["status"] == "OPEN").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("open positions", open_n)
            c2.metric("signals in window", len(sig_w))
            newest = sig_w["action_date"].max()
            c3.metric("latest signal", str(newest.date()))
            st.markdown("**Live trade ledger** - every suggestion, newest "
                        "first, with entry, dated 20-day exit, status and "
                        "P&L so far (signed - always 'money made')")
            st.dataframe(desk, width="stretch", hide_index=True, height=420)
            st.markdown(f"**Strategy scorecard ({HOLD_DAYS}d hold, signed P&L)**")
            if prices is not None:
                st.dataframe(signal_scorecard(sig_w, prices, priced, lo),
                             width="content", hide_index=True)
            st.markdown("**Certainty ranking (score + |conv z| + recency)**")
            cert = certainty_table(sig_w)
            show = ["action_date", "action", "theme", "etf", "score",
                    "conv_z", "certainty"]
            st.dataframe(ranked(cert[[c for c in show if c in cert.columns]]
                                .head(15), "certainty"),
                         width="stretch", hide_index=True)
            st.markdown("#### Signal charts - one per theme, ranked by "
                        "certainty (best trade first)")
            st.caption("Each chart shows a theme's anchor ETF price with "
                       "that theme's BUY/SELL triangles in the window. The "
                       "order follows the certainty ranking above; below "
                       "each chart, every trade is explained in words (the "
                       "signal engine's own `reason`).")
            for theme in cert["theme"].drop_duplicates().head(how_many):
                symbol = resolve_anchor(theme, priced)
                if prices is None or symbol is None:
                    continue
                px_line = price_series(prices, symbol, lo, hi)
                if px_line.empty:
                    continue
                th_rows = (sig_w[sig_w["theme"] == theme]
                           .sort_values("action_date", ascending=False))
                st.plotly_chart(fig_signals(px_line, th_rows, theme, symbol,
                                            cz=theme_cz(theme)),
                                width="stretch", key=f"desk_sig_{theme}")
                for _, r in th_rows.head(6).iterrows():
                    st.caption(
                        f"- {r['action_date'].date()} **{r['action']}** "
                        f"(score {r.get('score', '?')}/5, conv z "
                        f"{r.get('conv_z', float('nan')):+.2f}) - "
                        f"{r.get('reason', 'no reason recorded')}")

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
                                 "conviction crossings on anchor price",
                                 "BUY/SELL signals on anchor price"],
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
            else:
                if sig_file is None:
                    st.info("no signals on file - run the pipeline")
                    break
                s_th = clip_window(sig_file, "action_date", lo, hi)
                s_th = s_th[s_th["theme"] == theme]
                if len(s_th):
                    st.plotly_chart(fig_signals(px, s_th, f"#{i}  {theme}",
                                                symbol, cz=theme_cz(theme)),
                                    width="stretch",
                                    key=f"ovth_sig_{theme}")
                else:
                    st.caption(f"#{i} {theme}: no signals in this window")
        if view == "BUY/SELL signals on anchor price" and sig_file is not None:
            s_w = clip_window(sig_file, "action_date", lo, hi)
            if len(s_w) and prices is not None:
                st.markdown(f"**Report card, whole window ({len(s_w)} signals, "
                            "all themes)**")
                st.dataframe(signal_scorecard(s_w, prices, priced, lo),
                             width="content", hide_index=True)

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

    st.markdown("### 2 - Trading signals on price")
    st.caption("The model's actual BUY/SELL calls (all 5 checks, K "
               "threshold, cooldown) placed on the price line.")
    if sig_file is not None:
        s_h = clip_window(sig_file, "action_date", h_lo, h_hi)
        s_ht = s_h[s_h["theme"] == h_theme]
        if len(s_ht) and px is not None and not px.empty:
            st.plotly_chart(fig_signals(px, s_ht, h_theme, symbol,
                                        cz=theme_cz(h_theme)),
                            width="stretch", key="hist_sig")
        else:
            st.info(f"no signals for {h_theme} in this window")
        if prices is not None and len(s_h):
            st.markdown(f"**Scorecard, whole window ({len(s_h)} signals, "
                        "all themes)**")
            st.dataframe(signal_scorecard(s_h, prices, priced, h_lo),
                         width="content", hide_index=True)
