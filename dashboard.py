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
# WHAT CHANGED vs the predecessor dashboard
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
import re
import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.config import (ROLL, MIN_TOTAL,                        # noqa: E402
                        PROCESSED_DIR, PRICES_PATH,
                        EUPHORIA_HYPE_MULT, EUPHORIA_BOOM_MIN_ETF,
                        EUPHORIA_BOOM_MIN_SINGLE, EUPHORIA_BOOM_WINDOW_D,
                        EUPHORIA_BOOM_WINDOW_MIN_D, EUPHORIA_ONSET_HYPE_MIN,
                        EUPHORIA_ATT_GATE, EUPHORIA_MIN_COVERAGE,
                        EUPHORIA_MIN_HISTORY, EUPHORIA_PCT_WINDOW,
                        EUPHORIA_CRASH_MIN_ETF, EUPHORIA_CRASH_MIN_SINGLE,
                        EUPHORIA_COOLDOWN_DAYS, EUPHORIA_FA_BUDGET_PER_IY,
                        EUPHORIA_FA_PENALTY)
import src.themes as _themes                                       # noqa: E402


def _theme_etf_maps():
    """(THEME_ETFS, THEME_ETF_FALLBACKS), RE-READ when the CSV changes.

    `src/themes.py` builds these once at import, which is right for the
    pipeline (one process, one run) and wrong for a dashboard that stays
    up for days.  Streamlit's "Rerun" re-executes this script but does
    NOT re-import an already-imported module, so an edit to
    `config/theme_etfs.csv` was invisible until somebody killed and
    restarted the server - and nothing on screen said so.  That cost a
    real correction: the china_geopolitics anchor was moved KWEB -> FXI
    in the CSV and the running app kept showing KWEB, which reads as the
    fix having failed rather than as a stale process.

    So the anchor map is now re-derived from the file whenever its mtime
    moves.  `_load_theme_etfs` is reused rather than reimplemented, so
    the validation (every symbol must be on the approved list) and the
    tracked-only rule still apply exactly once, in one place."""
    return _cached_theme_etf_maps(
        os.path.getmtime(os.path.join(ROOT, "config", "theme_etfs.csv")))


@st.cache_data(show_spinner=False)
def _cached_theme_etf_maps(mtime):
    return _themes._load_theme_etfs()


THEME_ETFS, THEME_ETF_FALLBACKS = _theme_etf_maps()


@st.cache_data(show_spinner=False)
def _security_names():
    """symbol -> company name, from the Nasdaq symbol directories this
    project already caches for the ticker universe. Used to label the
    single-name dropdown the way the theme dropdown is labelled with its
    anchor ETF (requirement: "it should be like the theme
    dashboard") - a bare list of tickers makes you remember what NBIS or
    SNDK are; a labelled one does not."""
    out = {}
    for fname, col in (("nasdaqlisted.txt", "Symbol"),
                       ("otherlisted.txt", "ACT Symbol")):
        path = os.path.join(ROOT, "data", "reference", fname)
        if not os.path.exists(path):
            continue
        try:
            lines = open(path, encoding="utf-8").read().splitlines()
            head = lines[0].split("|")
            si, ni = head.index(col), head.index("Security Name")
        except (ValueError, OSError):
            continue
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) <= max(si, ni):
                continue
            sym = parts[si].strip().upper()
            if sym and sym not in out:
                out[sym] = _clean_security_name(parts[ni])
    return out


# The directories spell the instrument, not the company: "NVIDIA
# Corporation - Common Stock", "AMC Entertainment Holdings, Inc. Class A
# Common Stock".  The share class is noise in a label whose whole job is
# recognition, and truncating a raw string at 38 characters produced
# "AMC Entertainment Holdings, Inc. Class" - a cut mid-phrase reads as a
# bug in the data rather than as a shortened name.  So the boilerplate
# comes off FIRST and the truncation, if it is still needed, is marked.
_NAME_TAIL = re.compile(
    r"\s*[-–]?\s*(Class\s+[A-Z]\s+)?"
    r"(Common Stock|Ordinary Shares?|Common Shares?|American Depositary"
    r"\s+Shares?|Depositary Shares?|Class\s+[A-Z])"
    r".*$", re.IGNORECASE)


def _clean_security_name(raw: str, width: int = 38) -> str:
    name = str(raw).split(" - ")[0]
    name = _NAME_TAIL.sub("", name).strip().rstrip(",")
    return name if len(name) <= width else name[:width - 1].rstrip(" ,") + "…"


@st.cache_data(show_spinner=False)
def _approved_symbols():
    """Every symbol on config/approved_instruments.csv. Read here rather
    than through src.themes because that module only exposes the ones a
    theme happens to point at - and the whole point of the caller is the
    ones no theme points at."""
    import csv as _csv
    p = os.path.join(ROOT, "config", "approved_instruments.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        return [str(r.get("symbol", "")).strip()
                for r in _csv.DictReader(f)
                if str(r.get("symbol", "")).strip()]
from analytics import overlays                                     # noqa: E402
from analytics import influence_graph as ig                        # noqa: E402
from analytics.plain_english import (PLAIN, censor,                # noqa: E402,F401,E501
                                     censor_series, plain,         # noqa: E402,F401,E501
                                     half_mask, half_mask_series,  # noqa: E402,F401,E501
                                     theme_label)                  # noqa: E402,F401,E501
from analytics.euphoria import resolve_anchor                      # noqa: E402
from analytics.loaders import (price_series, clip_window,          # noqa: E402
                               THEME_COUNTS, TICKER_COUNTS)
from analytics.overlays import (mention_share_series,              # noqa: E402
                                chatter_change_series, sentiment_series,
                                relative_sentiment_series)

st.set_page_config(page_title="RetailRadar", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Hosted-deployment bootstrap.
#
# On a workstation the pipeline has already written data/processed and the
# two calls below are no-ops. On a hosted clone (Streamlit Community Cloud)
# data/processed and data/prices are gitignored and therefore absent, so the
# committed bundles are copied into place once per container: ABSTRACTED_DATA
# (the text-free aggregates) and DASHBOARD_DATA (the finished display frames,
# staged by tools/publish_dashboard.py).
#
# Copy, never recompute: analytics.run_analytics needs prices, forks a process
# pool, and auto-opens a full walk-forward research pass when the frozen
# record is missing - which is precisely the state of a fresh clone. The host
# draws published frames; it does not derive them.
# ---------------------------------------------------------------------------
BUNDLE_DIR = os.path.join(ROOT, "DASHBOARD_DATA")


def _bootstrap_from_bundles():
    """Populates data/processed from the committed bundles when absent.

    A file is copied only when the destination is missing or older than
    the published copy, so a workstation whose pipeline output is newer
    is never overwritten.
    """
    import shutil

    from src import abstracted_data
    from src.config import PRICES_DIR

    def _place(src, dst):
        if not os.path.exists(src):
            return
        if (os.path.exists(dst)
                and os.path.getmtime(dst) >= os.path.getmtime(src)):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # The five text-free aggregates travel in ABSTRACTED_DATA already.
    abstracted_data.hydrate(verbose=False)

    if not os.path.isdir(BUNDLE_DIR):
        return
    for name in sorted(os.listdir(BUNDLE_DIR)):
        src = os.path.join(BUNDLE_DIR, name)
        if name == "prices.parquet":
            _place(src, os.path.join(PRICES_DIR, name))
        elif name in ("nasdaqlisted.txt", "otherlisted.txt"):
            _place(src, os.path.join(ROOT, "data", "reference", name))
        elif name == "nb05_influence.json":
            _place(src, os.path.join(ROOT, "docs", "research", name))
        elif name == "signal_snapshots" and os.path.isdir(src):
            for snap in os.listdir(src):
                _place(os.path.join(src, snap),
                       os.path.join(PROCESSED_DIR, "signal_snapshots", snap))
        elif name == "influence" and os.path.isdir(src):
            for infl in os.listdir(src):
                _place(os.path.join(src, infl),
                       os.path.join(ROOT, "data", "reference", "influence",
                                    infl))
        elif os.path.isfile(src):
            _place(src, os.path.join(PROCESSED_DIR, name))


def _bundle_signature() -> float:
    """Newest mtime across the committed bundles.

    The cache key for the bootstrap below. A hosted redeploy does not
    necessarily restart the Python process - Community Cloud
    "automatically copy[ies] any file changes you commit" into the
    running container - so keying the bootstrap on the process alone
    would copy the bundle once and never notice a later publish. Keyed
    on this instead, new files landing in the checkout invalidate the
    cache and are placed on the next rerun. Walking ~40 files costs
    under a millisecond.
    """
    newest = 0.0
    for root in (BUNDLE_DIR, os.path.join(ROOT, "ABSTRACTED_DATA")):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                try:
                    newest = max(newest,
                                 os.path.getmtime(os.path.join(dirpath, name)))
                except OSError:            # vanished mid-walk; ignore
                    pass
    return newest


@st.cache_resource(show_spinner="Preparing the published data...")
def _bootstrap_once(signature: float):
    """Places the bundle once per distinct bundle state.

    cache_resource (not cache_data): the work is a filesystem side
    effect, not a value, and every session in the container shares it.
    """
    from src.config import ensure_dirs
    ensure_dirs()
    _bootstrap_from_bundles()
    return signature


_bootstrap_once(_bundle_signature())

# Pipeline controls (fetch, price pull, comment catch-up) are workstation-only.
# The marker file is gitignored, so it exists on a machine that has run
# tools/publish_dashboard.py and never on a hosted clone - a public viewer is
# not offered buttons that spend API credit or call a Bloomberg Terminal.
LOCAL_CONTROLS = (os.path.exists(os.path.join(ROOT, ".local_controls"))
                  or os.environ.get("RETAILAPOLLO_CONTROLS") == "1")


def watch_gate(row_or_frame):
    """The gate used to decide WHICH SIDE IS DISPLAYED.

    Prefers boomed120_stable - the hysteresis+debounce twin the phases
    step writes (see boomed120_frame) - and falls back to the raw
    boomed120 on bundles published before it existed. FIRES are never
    routed through here: they are read from the stored get_in/get_out
    columns, which the model computed against the raw gate.
    """
    if row_or_frame is None:
        return False
    if hasattr(row_or_frame, "columns"):            # frame -> Series
        for c in ("boomed120_stable", "boomed120"):
            if c in row_or_frame.columns:
                return row_or_frame[c].fillna(False).astype(bool)
        import pandas as _pd
        return _pd.Series(False, index=row_or_frame.index)
    for c in ("boomed120_stable", "boomed120"):     # row -> bool
        v = row_or_frame.get(c)
        if v is not None and pd.notna(v):
            return bool(v)
    return False


def ai_text(s):
    """Model-written free text -> markdown-safe. Streamlit renders
    $...$ as LaTeX math, so one price mention and a later one swallowed
    whole sentences into italic mush ("$12B ... $100" -> math; defect
    2026-08-31). Escaping the dollar sign is enough - the model's own
    markdown (bold, bullets) still renders."""
    return str(s).replace("$", "\\$") if s is not None else ""


def flag_label(name, kind):
    """The ONE way this app spells an instrument on screen.

    Requirement 2026-08-04: "copy the thematic dashboard side on the
    format i like".  The theme side already read `China geopolitics
    (FXI)` - a name a human recognises, followed by the thing you would
    actually trade.  The single-name side read `IREN`, which is neither:
    a bare four-letter ticker in a INCREASE EXPOSURE banner makes the reader stop
    and remember what IREN, NBIS or SNDK are, and the whole point of the
    banner is that it needs no interpretation.

    So both sides now use this function, and so does the lookup
    dropdown.  Themes get their anchor ETF from `THEME_ETFS` (fixing a
    mapping in config/theme_etfs.csv therefore fixes every label on the
    page at once); single names get their company name from the Nasdaq
    directories this project already caches.  Anything unresolvable
    falls back to the bare symbol rather than to a blank paren.

    A theme is labelled with the instrument ACTUALLY being drawn, not the
    one the config nominates.  The two differ more often than is
    comfortable: an anchor with no price history in the store silently
    resolves to its first priced fallback, so on 2026-08-04
    `europe_defense` was labelled EUAD and drawn on ITA - a US aerospace
    line standing in for a European one, which is precisely the
    substitution that theme's own config note warns about.  Where they
    differ the label says so, because quoting an instrument the chart is
    not using is worse than quoting none."""
    if kind != "theme":
        who = _security_names().get(name)
        return f"{name}  ({who})" if who else str(name)
    cfg = THEME_ETFS.get(name)
    if not cfg:
        return theme_label(name)
    live = _live_anchor(name)
    if live and live != cfg:
        return f"{theme_label(name)}  ({live} - {cfg} unpriced)"
    return f"{theme_label(name)}  ({cfg})"


def _live_anchor(theme):
    """The first instrument in the theme's chain that HAS prices - the one
    `resolve_anchor` picks when something is actually drawn."""
    have = _priced_symbols()
    if not have:
        return THEME_ETFS.get(theme)
    chain = ([THEME_ETFS[theme]] if THEME_ETFS.get(theme) else [])
    chain += THEME_ETF_FALLBACKS.get(theme, [])
    for sym in chain:
        if sym in have:
            return sym
    return None


def _priced_symbols():
    """Every symbol with price history, cached on the store's mtime."""
    return _cached_priced_symbols(_mtime(PRICES_PATH))


@st.cache_data(show_spinner=False)
def _cached_priced_symbols(mtime):
    if not os.path.exists(PRICES_PATH):
        return frozenset()
    return frozenset(pd.read_parquet(
        PRICES_PATH, columns=["symbol"])["symbol"].unique())

# ---- TECHNICAL CONFIRMATION (the five gates) -----------------------------
# A five-condition momentum screen, in two windows, computed from close
# prices alone. Provenance: Jonathan's momentum framework; the study of
# whether it agrees with this project's crowd signal is notebook 09.
#
# READ THIS BEFORE TRUSTING THE FLAG. Nothing here moves a trigger,
# gates a signal, or changes a score. It is a DISPLAY annotation: the
# crowd detector decides when a name is worth acting on, and this says
# whether price action agreed on that same day. Notebook 09 measured the
# agreement as promising and NOT established - the direction is
# consistent across two signal sets and three horizons, but nothing
# survives multiple-comparison correction and the matched gate rests on
# 22 INCREASE events. It is shown so a reader can weigh it, never to
# suppress a signal it disagrees with.
#
# Every window is TRAILING and every percentile is against the name's
# own trailing 252 observations, so no forward data enters any column.
# cache_RESOURCE, not cache_data. cache_data returns a DEEP COPY of its
# value on every call to protect against mutation, and this value is a
# dict of ~420 DataFrames. The row builder asks for it three times per
# instrument, so the copying alone cost 4.4 SECONDS per rerun - the
# single largest item on the landing page once the charts were made
# lazy. The dict is read-only, so handing back the same object is both
# correct and free.
@st.cache_resource(show_spinner=False)
def _momentum_gates(mtime):
    """{symbol: DataFrame(date, EM, ES, px_last)} from close prices.

    TREAT THE RETURN VALUE AS READ-ONLY - it is shared, not copied.
    """
    if not os.path.exists(PRICES_PATH):
        return {}
    px = pd.read_parquet(PRICES_PATH, columns=["date", "symbol", "px_last"])
    px["date"] = pd.to_datetime(px["date"])
    out = {}
    for sym, g in px.groupby("symbol"):
        g = g.sort_values("date")
        if len(g) < 300:                 # 252-day windows need the history
            continue
        c = g["px_last"]
        r1m = c.pct_change(21) * 100
        a1 = r1m - r1m.shift(21)
        em = ((r1m.rolling(252, min_periods=252).rank(pct=True) * 100 >= 85)
              & (((a1 - a1.rolling(252, min_periods=252).mean())
                  / a1.rolling(252, min_periods=252).std()) >= 1.0)
              & (c > c.rolling(50, min_periods=50).mean())
              & (((c.rolling(252, min_periods=252).max() - c)
                  / c.rolling(252, min_periods=252).max() * 100) <= 10)
              & (r1m > 0))
        r2w = c.pct_change(10) * 100
        a2 = r2w - r2w.shift(10)
        es = ((r2w.rolling(252, min_periods=252).rank(pct=True) * 100 >= 85)
              & (((a2 - a2.rolling(252, min_periods=252).mean())
                  / a2.rolling(252, min_periods=252).std()) >= 1.0)
              & (c > c.rolling(20, min_periods=20).mean())
              & (((c.rolling(126, min_periods=126).max() - c)
                  / c.rolling(126, min_periods=126).max() * 100) <= 10)
              & (r2w > 0))
        out[sym] = pd.DataFrame({"date": g["date"].values,
                                 "EM": em.values, "ES": es.values,
                                 "px_last": c.values})
    return out


def _tech_confirms(symbol, when, side):
    """Does the MATCHED window agree with this side on that day?

    EM (medium) for INCREASE - an entry wants an established trend.
    ES (short) for CUT - an exit wants a stretched one. The pairing is
    notebook 09's, chosen on mechanism before the numbers were seen.
    """
    if not symbol or when is None:
        return False
    g = _momentum_gates(_mtime(PRICES_PATH)).get(str(symbol))
    if g is None or not len(g):
        return False
    i = g["date"].searchsorted(pd.Timestamp(when))
    if i >= len(g):
        i = len(g) - 1
    col = "EM" if str(side).upper().startswith("INCREASE") else "ES"
    return bool(g[col].iloc[i])


def _price_change(symbol, when, days):
    """% change in close over the `days` before `when`. None if unknown."""
    if not symbol or when is None:
        return None
    g = _momentum_gates(_mtime(PRICES_PATH)).get(str(symbol))
    if g is None or not len(g):
        return None
    i = g["date"].searchsorted(pd.Timestamp(when))
    i = min(i, len(g) - 1)
    j = g["date"].searchsorted(pd.Timestamp(when) - pd.Timedelta(days=days))
    if j >= len(g) or j >= i:
        return None
    p0, p1 = g["px_last"].iloc[j], g["px_last"].iloc[i]
    return None if not p0 else (p1 / p0 - 1) * 100


HIGH_CONV_HELP = (
    "**HIGH CONVICTION** means the crowd signal and the price action "
    "agree on the same day.\n\n"
    "The crowd detector decides WHEN a name is worth acting on. This "
    "flag adds whether the price action agreed on that same day.\n\n"
    "**The five factors**, from Jonathan's momentum framework. All "
    "five must hold, and each is measured against the name's OWN "
    "trailing year rather than against the market:\n\n"
    "1. **Return percentile** - the trailing return sits in the top "
    "15% of the name's own history.\n"
    "2. **Acceleration** - the return is not just high but "
    "increasing, at least one standard deviation above its own "
    "normal rate of change.\n"
    "3. **Above its moving average** - price is above the trend "
    "line, so the move is a trend and not a bounce.\n"
    "4. **Near its own high** - within 10% of the highest price it "
    "has reached in the window.\n"
    "5. **Positive return** - the move is upward at all.\n\n"
    "Two windows apply the same five: a MEDIUM window (one-month "
    "return, 50-day average, 52-week high) and a SHORT one "
    "(two-week return, 20-day average, 26-week high). INCREASE is "
    "confirmed by the medium window - an entry wants an established "
    "trend. CUT is confirmed by the short one - an exit wants a "
    "stretched one.\n\n"
    "**How much to trust it.** Measured lift is real but not "
    "established: the direction is consistent across two signal sets "
    "and three horizons, and INCREASE accuracy at 20 days rose from "
    "61% to 82% - but on 22 events, and no single result survives "
    "correction for multiple testing. It never suppresses a signal; it "
    "only annotates one."
)

# ---- LEVEL-CONDITIONED OUTCOMES ------------------------------------------
# "Given where the crowd heat sits TODAY, what did price do next the
# other times it sat here?" - the question a PM actually asks, replacing
# the after-a-signal record (n=1 on most names, which is a story, not a
# statistic). One row per (name, day): the level and the forward price
# change at 5/20/84 trading days. Built once, cached on the two source
# files' mtimes; per-name slices and the all-theme pool both cut from
# the same frame so they can never disagree.
@st.cache_resource(show_spinner=False)
def _level_outcome_frame(lv_mtime, px_mtime):
    """DataFrame(name, level, f5, f20, f84) for every theme day."""
    lv_path = os.path.join(PROCESSED_DIR, "euphoria_levels.parquet")
    if not (os.path.exists(lv_path) and os.path.exists(PRICES_PATH)):
        return None
    lv = pd.read_parquet(lv_path,
                         columns=["date", "name", "symbol", "kind", "level"])
    lv = lv[(lv["kind"] == "theme") & lv["level"].notna()]
    lv["date"] = pd.to_datetime(lv["date"])
    px = pd.read_parquet(PRICES_PATH)
    px["date"] = pd.to_datetime(px["date"])
    out = []
    for sym, g in lv.groupby("symbol"):
        p = px[px["symbol"] == sym].sort_values("date")
        if len(p) < 100:
            continue
        pdates, pvals = p["date"].values, p["px_last"].values
        idx = pd.Series(pdates).searchsorted(g["date"].values)
        for (_, r), i in zip(g.iterrows(), idx):
            i = min(int(i), len(pvals) - 1)
            row = {"name": r["name"], "date": r["date"],
                   "level": float(r["level"])}
            for h in (5, 20, 84):
                j = i + h
                row[f"f{h}"] = (float(pvals[j] / pvals[i] - 1)
                                if j < len(pvals) and pvals[i] else None)
            out.append(row)
    return pd.DataFrame(out) if out else None


def level_conditioned_stats(name, level_now, band=10.0, min_n=10):
    """Median forward px change on days this name's level sat near
    today's, WITH the baseline it must be read against and the
    downside the median hides.

    Returns (rows, pooled_flag) where each row is
    (horizon, median, n, baseline_median, p_loss, p10).

    WHY THE EXTRA THREE. Shown alone, the median read as an argument
    AGAINST the CUT call sitting beside it ("if px is up at this level
    why reduce?" - 2026-09-02). Two things were missing and both flip
    the reading:

      * NO BASELINE. Equities drift up, so almost any forward median is
        positive. Cloud SaaS at level ~94 shows +4.8% over 84td, which
        looks bullish until you see its own unconditional median is
        +7.8% - the level-conditioned number is 3 POINTS WORSE than an
        ordinary day. Across all themes at level 85+, the 84d median is
        +3.9% against a +4.4% baseline. The raw figure was measuring
        market drift, not the level.
      * NO TAIL. A top call is a claim about the LEFT TAIL thickening,
        not about the typical day. At level 85+ the median is +3.9%
        while 39% of outcomes are losses and the worst tenth is -18%.
        A median can rise while the tail gets much worse.

    The baseline is the same population WITHOUT the level condition
    (this name's own history, or the pool when pooled), so the two
    numbers differ in exactly one thing: the level.
    """
    lf = _level_outcome_frame(
        _mtime(os.path.join(PROCESSED_DIR, "euphoria_levels.parquet")),
        _mtime(PRICES_PATH))
    if lf is None or level_now is None:
        return None, False
    m = (lf["level"] - float(level_now)).abs() <= band
    own = lf[m & (lf["name"] == name)]
    pooled = len(own.dropna(subset=["f5"])) < min_n
    d = lf[m] if pooled else own
    # BASELINE = days NOT in the band. Using the full history made the
    # baseline a SUPERSET of the band (33% of Cloud SaaS's history sits
    # inside +-10 of a mid-range level), so the two medians were largely
    # the same days and every difference collapsed toward 0.0pp. The
    # complement is the honest contrast: days like today vs days unlike.
    base = (lf[~m] if pooled
            else lf[(lf["name"] == name) & ~m])
    rows = []
    for h in (5, 20, 84):
        v = d[f"f{h}"].dropna()
        if not len(v):
            continue
        b = base[f"f{h}"].dropna()
        rows.append((h, float(v.median()), int(len(v)),
                     float(b.median()) if len(b) else None,
                     float((v < 0).mean()), float(v.quantile(0.10))))
    return (rows or None), pooled


# ---- DISPLAY-ONLY INSTRUMENT HIDING --------------------------------------
# Names the dashboard does not draw. They are still fetched, still scored
# and still written to the stores - this is a screen filter and nothing
# more, so hiding one costs no data and needs no research pass.
#
# NOT src.config.EUPHORIA_EXCLUDED_THEMES: that set removes a theme from
# the DETECTOR's universe, which changes what is computed and would make
# the frozen thresholds stale. This set is the display's own.
#
# meme_stocks: anchored to ARKK, the same ETF as short_squeeze, and both
# now label as "ARK Innovation". Shown together they render two
# identical rows in every instrument list, which cannot be told apart
# when picking one. The squeeze theme is kept as the ARKK line.
#
# japan: anchored to 1622 JT, which is TOPIX-17 Autos & Transport
# Equipment rather than broad Japan - a narrow line the desk does not
# want on the page (request 2026-09-02). Hidden for DISPLAY only: the
# name stays in the stores and in every fit, so no threshold moves and
# the history stays intact if it is ever unhidden.
HIDDEN_THEMES = {"meme_stocks", "japan"}

# Landing-page sizing. A name is worth showing when its score has
# reached this share of its own trigger; below that the page still
# shows the nearest few so it is never blank on a quiet day.
ACTION_READY_PCT = 90.0
ACTION_MIN_ROWS = 5
# How many recently-fired signals the landing page lists.
ACTION_FIRED_ROWS = 8
# A reading older than this is dropped from the lists. 21 days - one
# signal cooldown - is the shelf life of "act on this now": beyond it
# the crowd state that produced the score has turned over. This page
# once showed a 499-day-old reading two rows above one from yesterday,
# visually identical; a short honest list beats a long misleading one.
ACTION_STALE_DAYS = 21


def _hide(df):
    """Drops hidden instruments from anything about to be displayed."""
    if df is None or not len(df) or "name" not in df.columns:
        return df
    return df[~df["name"].isin(HIDDEN_THEMES)]


# The signal names as STORED. readiness_alerts.json is written by the
# pipeline and carries the engine's own vocabulary, so a file produced
# before the display rename - or by a machine that has not taken it -
# still says GET IN / GET OUT. Mapping at read time means no stored file
# has to be rewritten and no pipeline rerun is needed; anything not in
# the map (INFLECTION..., future names) passes through untouched.
_SIDE_DISPLAY = {"GET IN": "INCREASE EXPOSURE", "GET OUT": "CUT EXPOSURE"}


def eligible_scored_now(g):
    """The last row whose OWN eligible-side score exists: gate and score
    from the SAME day.

    The desk store's newest rows are unscored placeholders whose
    boomed120 reads False, so "gate from the last row, score from the
    last scored row" mixed two different days - and on a day when every
    placeholder read not-boomed, every theme landed on the INCREASE side
    and the CUT list rendered empty while four themes were genuinely
    CUT-eligible (biotech 83%, cloud_saas 86%, uranium 91%). One rule,
    used by the landing rows, the dial and the radar, so the three can
    never disagree about which side a name is on."""
    if g is None or not len(g):
        return None
    b = watch_gate(g)
    ok = pd.Series(False, index=g.index)
    if OUT_SCORE in g.columns:
        ok |= (b & g[OUT_SCORE].notna())
    if IN_SCORE in g.columns:
        ok |= (~b & g[IN_SCORE].notna())
    if not ok.any():
        return None
    r = g[ok].iloc[-1]
    # the FILLED gate, not the raw value: the mask above treats a NaN
    # gate as False, and returning bool(nan)=True here would report a
    # row the mask selected as IN-eligible as CUT - whose score on that
    # row is NaN, which min/max arithmetic then turns into a phantom
    # "100% of the way" entry
    return r, bool(b.loc[r.name])


def _side_label(side) -> str:
    return _SIDE_DISPLAY.get(str(side).strip().upper(), str(side))


# ---------------------------------------------------------------------------
# DESIGN TOKENS - institutional light theme (GIC design language, adopted
# by design).  The brief: navy on white, high whitespace,
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
# The ineligible stretch of the euphoria signal line.  It has to read as
# "present but not the point" against WHITE and PANEL while a 3px coloured
# stroke sits on the same series - so it is deliberately BELOW the text
# contrast floor (2.0:1 on white).  That is allowed here and only here: it is
# never used for text and never carries a value on its own, and the number it
# draws is always also legible from the axis and the hover.
SLATE_LIGHT = "#B9BFC7"    # same series, on a day that could not fire

# Deliberate departure: INK_LABEL departs from the brief's #8C8C8C.
# Measured with a WCAG 2.1 contrast audit of every text node against
# its composited backdrop.  #8C8C8C scores
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
TEAL = "#2E6E7E"          # cool accent (masthead eyebrow, attention series)
GETIN = "#1E7A4F"         # the INCREASE-EXPOSURE green (was TEAL; "green, not teal")
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

    raw = _row(cfg.get("table"), variant="CUT EXPOSURE boom-gated raw")
    prod = _row(cfg.get("table"),
                variant="CUT EXPOSURE boom-gated SMOOTHED (production)")
    gi_old = _row(cfg.get("table"), variant="INCREASE EXPOSURE incumbent raw")
    gi_new = _row(cfg.get("table"),
                  variant="INCREASE EXPOSURE phase-aware SMOOTHED (production)")
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
INCREASE EXPOSURE and CUT EXPOSURE only. The old BUY/SELL engine was retired because on the
full price history it *lost* money (**-1.42% per trade, 2021-2026**) - it
bought retail enthusiasm into the 2022 bear market. It still exists in the
research code; it is simply not desk output.

**2 - The alert waits for the price to have actually run.**
Requiring a real boom before a CUT EXPOSURE can fire took captures from
**{_dig(n4, 'top', 'captured', default='-')} to {raw.get('captured', '-')}**
of {prod.get('detectable', '-')} - *more* hits, not fewer, because it stopped
the detector wasting alerts on names that were never in a rally.

**3 - The trigger reads a smoothed score, which killed the one-day blips.**
The desk complaint was "crowd heat for a single day, then gone". Averaging the
score over a week before triggering raised quality
(**AP {raw.get('AP', '-')} → {prod.get('AP', '-')}**) and cost
**{(raw.get('captured', 0) or 0) - (prod.get('captured', 0) or 0)} captures**.
That cost is recorded, not hidden.

**4 - A START can no longer print next to an END.**
The old version put a INCREASE EXPOSURE right beside a CUT EXPOSURE **{gi_old.get('adjacency', '-')}
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

**6 - "Buy on INCREASE EXPOSURE" and "short on CUT EXPOSURE" were both REJECTED.**
Tested properly and neither beat simply holding: INCREASE EXPOSURE
**{_pct(buy.get('diff'), 2, signed=True)}** over 20 days
(90% interval {_ci_pct(buy.get('ci90'))}), CUT EXPOSURE
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

**0 - THE AIM (re-set July 2026): detect retail CROWD HEAT and call price
TOPS.** The dashboard's headline signal is the 0-100 crowd heat level and
its red alert lines; success = an alert inside [peak-30d, peak+1d] of a
genuine top (peak = local high after a boom, followed by a >=15% ETF /
>=30% single-name drawdown within 90d). The BUY/SELL engine was retired
from the dashboard at the same time (its full-history record was
negative - see point 1); it remains in analytics/ for research.
Universe: equities + retail commodities only (rates_bonds and
real_estate excluded); single names join the themes. Full rules +
research grounding: the CROWD HEAT definition expander on the first tab
and analytics/crowd heat.py.

**0b - PREDICTION IS REDDIT-ONLY (selection rule, July 2026).** Price never
enters the crowd heat level or the alert; it only DEFINES and SCORES the
ground-truth tops. This was a deliberate trade: the earlier variant with
a price-convexity feature and a price-boom gate captured **46%** of
detectable peaks (0.08 FAs/instr-yr); the crowd-only detector captures
**~23%** (median lead 4d, ~0.11 FAs/instr-yr). The chart carries real
information - giving it up is the documented price of the clean claim
"the crowd alone called the top". The identified path to winning capture
back WITHOUT price: richer crowd data (the comment backfill is ~10x the
post volume and directly feeds every crowd heat ingredient).

**0c - Every rule is ablation-tested and the hand-rules beat an ML
challenger under a pre-stated criterion** (tables on the CROWD HEAT tab):
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

**1 - The BUY/SELL engine is the original legacy logic, unchanged.**
Verified by diffing the two projects' signal files: every tradeable
signal matches (the only differences are in `cannabis`, which has no
approved instrument and never trades). Scoring *the legacy engine's own file*
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

**6 - Influence tracker: measured record, committed store.** Author
scoring runs end-to-end on the desk's own store: volatility-scaled
correctness bar (tau = max(3%, 0.5 sigma) - one fixed bar would misgrade
an index ETF and a meme stock with the same ruler), abnormal-return
weighting w(z)=clip(1+|z|, 0.1, 2), Bayesian shrinkage (alpha 10/5/10),
composite 0.4/0.4/0.2 with the HIGH tier at 0.66, and a bot-filtered
reply-graph PageRank. Ranking is by USEFULNESS, never by size: our own
error analysis found the structurally loudest users were the least
accurate - the 'loud but wrong' column encodes exactly that finding.
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
   this rule the family override above inflections every icon into its literal
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

/* --- the statistic counter, the house signature component -----------
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
   'CROWD HEAT AL...', 'MEDIAN CUT EXPOSURE B...'.  A KPI whose name is unreadable is
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
/* THE FRESHNESS LINE.  Same slot as .rf-sub, but when the data is not
   current it stops whispering: full sentence case instead of small
   uppercase, a coloured rule down its left edge, and the colour set
   inline by severity (ochre = late, brick = stale).  A masthead line
   is the one place a reader looks before trusting a number, which is
   why the notice lives here and not in the sidebar. */
.rf-stale {{
    font-size: 0.86rem; line-height: 1.45; font-weight: 400;
    border-left: 3px solid; padding: 3px 0 3px 10px; margin: 3px 0 2px 0;
}}
.rf-rule {{ border-top: 1px solid {HAIRLINE}; margin: 1.4rem 0 1.9rem 0; }}

/* --- tab bar: the selected view reads as SELECTED --------------------
   The segmented control's default selected state is a faint grey fill,
   which on this deliberately low-contrast page is nearly invisible -
   you cannot tell at a glance which view you are on.  The active
   segment is therefore filled with the primary navy and reversed out
   to white.  Navy rather than a bright blue: it is the colour already
   carrying "primary" everywhere else on the page (headers, the
   crowd heat curve, links, buttons), so the tab bar joins the existing
   system instead of introducing a second accent.

   Styled by data-testid, which is Streamlit's stable hook.  Every rule
   is cosmetic: if a future Streamlit renames the test id, the control
   reverts to its default appearance and keeps working. */
[data-testid="stSegmentedControl"] button {{
    border-radius: 2px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
}}
/* The first three tabs are the product; the rest are supporting. */
[data-testid="stSegmentedControl"] button:nth-of-type(-n+3),
[data-testid="stSegmentedControl"] button:nth-of-type(-n+3) *,
[data-testid="stSegmentedControl"] > div > div:nth-child(-n+3) button,
[data-testid="stSegmentedControl"] > div > div:nth-child(-n+3) button * {{
    font-weight: 700 !important;
}}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {{
    background: {NAVY} !important;
    border-color: {NAVY} !important;
}}
[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] * {{
    color: {WHITE} !important;
    font-weight: 600 !important;
}}
[data-testid="stSegmentedControl"] button:hover:not([aria-checked="true"]) {{
    border-color: {NAVY_MID} !important;
}}
[data-testid="stSegmentedControl"] button:hover:not([aria-checked="true"]) * {{
    color: {NAVY} !important;
}}

/* =====================================================================
   EDITORIAL LAYER  (experiment)
   ---------------------------------------------------------------------
   Appended LAST so it overrides the blocks above by cascade order, and
   so removing this one block returns the page exactly to the treatment
   above it. Nothing here is referenced from Python - deleting from this
   comment to the end of the style tag is the whole undo.

   What it does: takes the design language already established above -
   statistic counters, editorial tabs, hairline cards - and pushes it to
   a publication's manners. Headings get bigger and LIGHTER, the numbers
   get bigger and lighter still, and air is added BETWEEN subjects.

   The density rule it obeys: whitespace goes between sections, never
   between a label and the number it labels. No chart, table, control or
   tab changes size, position or behaviour.
   ================================================================== */

/* HEADINGS. Authority from scale and space rather than from weight. */
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.026em; }}
h1 {{ font-size: 3.05rem !important; font-weight: 300 !important;
      line-height: 1.05; }}
h2 {{ font-size: 2.05rem !important; font-weight: 300 !important;
      line-height: 1.16; }}
h3 {{ font-size: 1.42rem !important; font-weight: 400 !important;
      letter-spacing: -0.018em; }}
h4 {{ font-size: 1.08rem !important; font-weight: 500 !important; }}

/* BODY. Editorial measure: slightly larger, noticeably looser. This is
   the single biggest change in how the page reads. */
p, li, [data-testid="stMarkdownContainer"] p {{
    font-size: 1.02rem; line-height: 1.74;
}}

/* MASTHEAD, promoted to a front page. The elements and their order are
   unchanged - only the type scale moved. */
.rf-eyebrow {{
    color: {TEAL}; font-size: 0.66rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.18em;
    margin-bottom: 12px;
}}
.rf-title {{
    font-size: 3.7rem !important; font-weight: 300 !important;
    letter-spacing: -0.038em !important; line-height: 1.0 !important;
}}
.rf-standfirst {{
    color: {INK_MUTED}; font-size: 1.18rem; font-weight: 300;
    line-height: 1.5; max-width: 48ch; margin: 16px 0 20px 0;
}}
/* The one heavy rule on the page, closing the masthead the way a
   publication closes its front-page furniture. */
/* Landing-page list headings: the two verbs are the loudest words on
   the page, so they are set as section titles rather than as labels. */
/* Readiness bar: how far a name is from its own trigger. The tick is
   the trigger line, so a bar that reaches it has fired. Drawn rather
   than st.progress because the trigger mark is the point. */
.rf-readbar {{
    position: relative; height: 8px; border-radius: 2px;
    background: {HAIRLINE}; margin: 2px 0 8px 0; overflow: hidden;
}}
.rf-readfill {{ height: 100%; border-radius: 2px; }}
.rf-readtick {{
    position: absolute; top: -2px; right: 0;
    width: 2px; height: 12px; background: {INK};
}}
/* HIGH CONVICTION chip: outlined rather than filled, because it
   annotates a signal and must never outrank it. */
/* List rows are BUTTONS (so a row can open lazily) but must read as
   rows: full width, left-aligned, quiet border, no button chrome.
   
   SCOPED by PREFIX to the class Streamlit puts on a keyed
   container - st.container(key='rfrow_x') emits
   'st-key-rfrow_x', and a plain .st-key-rfrow selector matches
   NOTHING because CSS class selectors match whole names, not
   prefixes. Not scoped to every button in a block - the sidebar's
   pipeline buttons are meant to stay navy.
   
   AND colour is restated with !important on the button AND its
   children. The rule above sets `.stButton button * {{ color: WHITE
   !important }}` to stop navy-on-navy, so a row that only overrode the
   BACKGROUND rendered white-on-white: visible boxes with invisible
   text. Exactly the defect the comment above warns about, in the other
   direction. */
[class*="st-key-rfrow"] .stButton > button {{
    text-align: left; justify-content: flex-start;
    font-weight: 500; border-radius: 3px;
    border: 1px solid {HAIRLINE} !important;
    background: {WHITE} !important;
    color: {INK} !important;
    padding: 0.55rem 0.8rem;
}}
[class*="st-key-rfrow"] .stButton > button * {{
    color: {INK} !important; text-align: left;
}}
[class*="st-key-rfrow"] .stButton > button:hover {{
    border-color: {NAVY_MID} !important; background: {WHITE} !important;
}}
[class*="st-key-rfrow"] .stButton > button:hover * {{ color: {NAVY_MID} !important; }}
.rf-hiconv {{
    display: inline-block; border: 1px solid; border-radius: 2px;
    padding: 2px 8px; font-size: 0.66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 2px 0 6px 0;
}}
/* The page's two-line key: large enough to be furniture, quiet
   enough not to shout over the lists it explains. */
.rf-keyline {{
    font-size: 1.35rem; font-weight: 300; line-height: 1.55;
    color: {INK}; margin: 0.4rem 0 1.2rem 0;
    letter-spacing: -0.01em;
}}
.rf-actionhead {{
    font-size: 1.42rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding-bottom: 7px; margin-bottom: 2px;
    border-bottom: 2px solid currentColor;
}}
/* the line under each action head. A st.caption here rendered at the
   same tiny size as every footnote on the page, which buried the one
   sentence that says what the list IS (request: "make the font of
   this ... a bit bigger too"). */
.rf-actionsub {{
    font-size: 1.02rem; line-height: 1.45;
    color: {INK_LABEL}; margin: 6px 0 10px 0;
}}
.rf-rule-heavy {{
    border-top: 2px solid {INK}; margin: 2.4rem 0 2.2rem 0;
}}

/* STATISTIC COUNTERS, taken further: the number is the loudest thing on
   the page, and it gets there by size, not by weight. The card becomes a
   figure under a rule rather than a box - the label/number pairing and
   the wrap fix established above are preserved exactly. */
[data-testid="stMetric"] {{
    background: rgba(0,0,0,0) !important;
    border: none !important;
    border-top: 1px solid {INK} !important;
    border-radius: 0 !important;
    padding: 15px 16px 18px 0 !important;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] p {{
    font-size: 2.4rem !important; font-weight: 300 !important;
    letter-spacing: -0.035em !important; line-height: 1.14 !important;
    color: {NAVY} !important;
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] p {{
    font-size: 0.63rem !important; font-weight: 600 !important;
    letter-spacing: 0.14em !important; color: {INK_LABEL} !important;
}}

/* SECTION RHYTHM. Air between subjects only. */
[data-testid="stExpander"] {{
    border: none !important; border-top: 1px solid {HAIRLINE} !important;
    border-radius: 0 !important; background: rgba(0,0,0,0) !important;
}}
[data-testid="stAppViewContainer"] > .main .block-container {{
    padding-top: 3.2rem !important;
}}

/* TABLES as ruled figures rather than framed boxes. Columns, sorting
   and interactions are untouched. */
[data-testid="stDataFrame"] thead tr th {{
    background: rgba(0,0,0,0) !important;
    color: {INK_LABEL} !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-size: 0.63rem !important; font-weight: 600 !important;
    border-bottom: 1px solid {INK} !important;
}}
</style>"""


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


def _facts(items):
    """A compact stack of tiny-label / big-number counters, as HTML.

    `items` is a sequence of (label, value_html, colour) triples; colour None
    falls back to INK.  `value_html` is inserted verbatim, so a caller can
    demote a unit ("/100", a date) to a smaller muted span inside the number.

    WHY NOT `st.metric`.  Two reasons, one aesthetic and one substantive.
    The house language asks for a tiny uppercase label above a large plain
    number with no box around it; st.metric fixes its own type scale and
    cannot be told otherwise.  More importantly st.metric draws a green/red
    delta chip, and on this dashboard a coloured delta reads as a
    recommendation - which is precisely the inference the crowd heat panel
    exists to prevent.  Rendering the counters directly keeps the choice of
    what to colour, and what to leave alone, with the caller.
    """
    rows = []
    for label, value, colour in items:
        rows.append(
            "<div style='margin:0 0 14px 0'>"
            "<div style='font-size:12px;letter-spacing:.09em;"
            f"text-transform:uppercase;color:{INK_LABEL};"
            f"margin-bottom:2px'>{label}</div>"
            "<div style='font-size:28px;font-weight:600;line-height:1.15;"
            f"color:{colour or INK}'>{value}</div></div>")
    return "<div style='padding-top:6px'>" + "".join(rows) + "</div>"


def _row(rows, **match):
    """First dict in a list of dicts matching every key=value given."""
    for r in rows or []:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    return {}


# resolve_anchor lives in analytics.euphoria and is imported below - the
# dashboard used to carry a character-for-character copy, which is one
# more place for the fallback rule to drift out of step with the engine
# that actually scores. Removed 2026-08-05.
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


# STACKED CHARTS SHARE A LEFT GUTTER (defect report: "make sure the
# graphs of price and the below align i think it glitched").  Plotly
# sizes each figure's left margin to ITS OWN y tick labels, so the price
# panel (labels like "1,480") started 68 px further left than the band
# panel below it (labels "at INCREASE EXPOSURE") - two charts on the
# same dates whose x axes did not line up.  Both panels now pin the same
# explicit gutter and switch y automargin OFF, so the plot areas start
# and end on the same pixel at any window width.  136 px is the widest
# band label measured at 10 px plus the tick and a little headroom for
# Inter; x automargin is left ON so the angled date labels still size
# their own bottom margin.
STACK_GUTTER_PX = 136


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


# ---------------------------------------------------------------------------
# THE EUPHORIA GAUGE  (recorded decision: "a current euphoria
# percentage with a red zone to get out ... a very clear speedometer thing
# for each graph, and showing the change")
# ---------------------------------------------------------------------------
def gauge_zones():
    """The gauge's two edges and the measured meaning of each band.

    Read from `docs/research/gauge_zones.json`, the frozen record of the
    strictness study (notebook 06, retired 2026-08-07 - the JSON stays
    live; resurrect the notebook from notebooks/_retired_2026-08-07_
    presentation_refactor/ to re-derive it).
    NOTHING here is a literal: an edge that lived in this file could drift
    away from the evidence that justifies it, and "why 76?" is the first
    question a gauge invites.  Empty dict = no frozen gauge record on disk,
    and the caller draws no gauge rather than a gauge with invented bands.
    """
    return _research("gauge_zones")


def gauge_state(level_now, in_danger, z):
    """Which band the needle sits in, as (key, label, colour).

    The band is a description of WHERE THE CROWD IS.  It is deliberately
    NOT the signal: INCREASE EXPOSURE / CUT EXPOSURE come from the walk-forward detector
    and can fire with the needle anywhere.  Keeping the two apart is the
    whole reason this returns a *state* and never an *instruction*.
    """
    red, amber = z.get("red_edge"), z.get("amber_edge")
    if red is None or amber is None or level_now is None:
        return "unknown", "no reading", INK_LABEL
    if level_now >= red:
        return (("red_danger", "RED ZONE + already run up", BEAR)
                if in_danger else ("red", "RED ZONE", BEAR))
    if level_now >= amber:
        return "amber", "warming", OCHRE
    return "calm", "calm", SLATE


# ---------------------------------------------------------------------------
# READINESS - "how close is this name to a signal firing?"
#
# Requirement 2026-08-13, and it fixes a real inconsistency rather than
# adding a feature. The card used to show a 0-100 EUPHORIA level beside a
# "0.21 to go" gap, which read as two views of one quantity and is not:
# the level is a crowd-heat composite, the gap is the ML score against
# its frozen cut, and their rank correlation across the store is only
# 0.34. travel_airlines showed the split at its worst - level 14, GET OUT
# score 0.76 - because the level is CROWD-ONLY while the desk score also
# sees price, so a quiet name that has run hard scores high on a cold
# dial. Putting readiness ON the dial makes the headline number and the
# gap the same measurement.
#
# readiness = 100 x score / that side's frozen cut. 100 IS THE TRIGGER,
# for every name and both sides, which is the property that lets one
# dial serve names whose cuts differ.
READY_AMBER = 85.0      # display band only - "close"
READY_RED = 100.0       # the trigger itself, never a tuned number


def readiness_now(dk_row, thr_in, thr_out):
    """(pct, side, score, cut) for the side this name may actually fire.

    The eligible side is the one the phase gate allows TODAY: CUT EXPOSURE
    once the name has cleared the 120d boom bar, INCREASE EXPOSURE before. Showing
    the other side's score would be a number that cannot fire.
    """
    if dk_row is None:
        return None
    # EXPERIMENTAL trigger (defined later in the sidebar; this function
    # is only ever CALLED after that): no phase gate, so both sides are
    # always live - the dial shows whichever is nearer to firing.
    if globals().get("XP_TRIGGER", False):
        cands = []
        for side, col, thr in (("CUT EXPOSURE", "out_score_xp", thr_out),
                               ("INCREASE EXPOSURE", "in_score_xp", thr_in)):
            sc = dk_row.get(col)
            if sc is not None and thr not in (None, 0) and pd.notna(sc):
                cands.append((100.0 * float(sc) / float(thr), side,
                              float(sc), float(thr)))
        return max(cands) if cands else None
    boomed = watch_gate(dk_row)
    side, sc, thr = (("CUT EXPOSURE", dk_row.get("out_score"), thr_out)
                     if boomed else
                     ("INCREASE EXPOSURE", dk_row.get("in_score"), thr_in))
    if sc is None or thr in (None, 0) or pd.isna(sc):
        return None
    return (100.0 * float(sc) / float(thr), side, float(sc), float(thr))


READY_HELP = (
    "**100 = the signal fires.** The needle is this name's live "
    "{side} score divided by that side's frozen trigger, as a "
    "percentage - so 80 means it is four fifths of the way there and "
    "100 means it is at the line.\n\n"
    "Only the side that CAN fire today is shown. A name that has "
    "already boomed is measured against CUT EXPOSURE; one that has not is "
    "measured against INCREASE EXPOSURE. The other side is not just unlikely, it "
    "is blocked by the phase gate.\n\n"
    "This is a different quantity from the CROWD HEAT level in the facts "
    "beside it. The level is crowd heat only; the score behind this "
    "dial also sees price, which is why a quiet name that has run hard "
    "can read cold on crowd heat and high here."
)

READY_HELP_XP = (
    "**100 = the signal fires.** The needle is this name's live "
    "{side} score divided by that side's frozen trigger, as a "
    "percentage - so 80 means it is four fifths of the way there and "
    "100 means it is at the line.\n\n"
    "EXPERIMENTAL trigger: there is no phase gate, so BOTH sides are "
    "always live. The dial shows whichever side is nearer to firing "
    "today.\n\n"
    "The score behind this dial reads posts ONLY - no price enters "
    "it anywhere. That is the point of the mode, and also why it is "
    "the weaker detector (walk-forward AUROC ~0.57 vs the shipped "
    "~0.73): treat it as research, not a desk signal."
)


def fig_euphoria_gauge(level_now, level_prev, in_danger, z, as_of,
                       peak_val=None, peak_day=None,
                       title_text=None, band_text=None, band_colour_o=None):
    """A speedometer for one name.  Needle = the CURRENT smoothed crowd heat
    level; delta = the same curve one smoothing window ago.

    Four things about the construction are load-bearing:

      1. the needle is the *display* curve's last value, not a fresh
         calculation, so the gauge figure and the chart below it can never
         disagree - the defect that would destroy trust fastest;
      2. the band edges arrive from `gauge_zones()`, i.e. from measurement,
         not from this file;
      3. the delta reference is `ROLL` days back - the same window the
         curve itself is smoothed over - so "the change" is a change in
         the plotted quantity and not a change in daily noise;
      4. `peak_val`/`peak_day` mark where the needle GOT TO inside the
         selected window, as a second thin needle plus a dated line of text.

    On (4), because it is a bug fix and not decoration.  The desk reported
    the dial "seems to always show calm".  It was not stuck - MEASURED over
    the default window (2026-01-01 -> latest), 50 of 59 instruments read
    calm at the last day while 17 of those same names touched the RED ZONE
    somewhere inside the window.  Both facts are true at once because the
    page is ordered by MOST RECENT SIGNAL, so a name earns its place with an
    episode that may have peaked months ago, while the needle - correctly -
    reports today.  A dial that answers "how hot is it now?" on a name
    selected for "it was hot recently" reads calm almost always, and looks
    broken while being right.
    The fix is to make the dial answer both questions instead of moving any
    threshold: the big needle stays TODAY (the requirement was for a *current*
    percentage), and the window's high-water mark is drawn behind it so a
    calm reading carries its own explanation - "calm now, peaked 99 in red
    on 27 Jun".  No edge moves, no number is invented.
    """
    red, amber = z["red_edge"], z["amber_edge"]
    _, band_label, band_colour = gauge_state(level_now, in_danger, z)
    # The readiness dial passes its own header and band word: the crowd
    # -heat vocabulary (WARMING, the CROWD HEAT title) on a needle that
    # is measuring distance-to-trigger was two different instruments
    # sharing one face - the exact confusion this dial was built to end.
    if band_text is not None:
        band_label = band_text
    if band_colour_o is not None:
        band_colour = band_colour_o
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(level_now),
        # the delta arrow is INVERTED on purpose: euphoria rising is the
        # risk direction, so "up" must not be painted green here
        delta=dict(reference=float(level_prev),
                   increasing=dict(color=BEAR),
                   decreasing=dict(color=BULL),
                   valueformat=".0f",
                   font=dict(size=13)),
        # WHOLE numbers: the level is a 0-100 index built from percentile
        # ranks, so a tenth of a point is below the resolution of the thing
        # being measured, and "28.4" invites a precision the input does not
        # have. The stored value is untouched - this is display only.
        number=dict(font=dict(size=34, color=band_colour), suffix="",
                    valueformat=".0f"),
        # NO Indicator `title`: plotly draws it inside the same domain as the
        # arc, so a two-line title is struck through by the value bar.  The
        # header is laid out as paper-space annotations in the top margin
        # instead, which is the only way it is guaranteed clear of the dial.
        domain=dict(x=[0, 1], y=[0, 1]),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=HAIRLINE,
                      tickvals=[0, amber, red, 100],
                      tickfont=dict(size=10, color=INK_MUTED)),
            # a THIN sweep, not a fat one: at 0.28 the navy bar covered the
            # band colours it is supposed to be read against, so the red zone
            # stopped being visible at exactly the moment it mattered
            bar=dict(color=NAVY, thickness=0.15),
            bgcolor=WHITE, borderwidth=0,
            steps=[dict(range=[0, amber], color=PANEL),
                   dict(range=[amber, red], color="#F0E6C8"),
                   dict(range=[red, 100], color="#EBD3D1")],
            # the red edge repeated as a hard line: a colour change alone
            # is not readable in greyscale or on a projector
            threshold=dict(line=dict(color=BEAR, width=3), thickness=0.85,
                           value=red))))
    # the as-of date is ON the dial, not in a caption: the sidebar can select
    # a historical window, and a dial labelled "now" that is in fact showing
    # March would be the worst kind of wrong
    fig.update_layout(
        height=268, margin=dict(l=28, r=28, t=64, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=INK),
        annotations=[
            dict(text=(title_text or ("CROWD HEAT AT "
                       f"{pd.Timestamp(as_of).strftime('%d %b %Y').upper()}")),
                 xref="paper", yref="paper", x=0.5, y=1.32,
                 xanchor="center", yanchor="bottom", showarrow=False,
                 font=dict(size=10, color=INK_LABEL)),
            dict(text=f"<b>{band_label.upper()}</b>",
                 xref="paper", yref="paper", x=0.5, y=1.13,
                 xanchor="center", yanchor="bottom", showarrow=False,
                 font=dict(size=13, color=band_colour)),
            dict(text=f"change vs {ROLL}d earlier",
                 xref="paper", yref="paper", x=0.5, y=-0.13,
                 xanchor="center", yanchor="top", showarrow=False,
                 font=dict(size=10, color=INK_MUTED))])
    # THE WINDOW'S HIGH-WATER MARK, as one dated line (see docstring (4)).
    #
    # Text, not a second needle: plotly's Indicator has a single `threshold`
    # slot and it is already carrying the red edge as a hard line (a colour
    # change alone does not survive greyscale or a projector).  Faking a
    # second needle would mean drawing shapes against the arc's internal
    # geometry in paper space - fragile across plotly versions, and two
    # needles on a 268px dial is a worse read than one sentence anyway.
    if peak_val is not None and peak_day is not None:
        _, _pk_band, _pk_colour = gauge_state(peak_val, False, z)
        _same = float(peak_val) - float(level_now) < 1.0
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=-0.27,
            xanchor="center", yanchor="top", showarrow=False,
            font=dict(size=10.5,
                      color=INK_MUTED if _same else _pk_colour),
            text=("this IS the window's high point" if _same else
                  (f"window high <b>{float(peak_val):.0f}</b> "
                   f"({_pk_band}) on "
                   f"{pd.Timestamp(peak_day).strftime('%d %b %y')}")))
        # the taller frame and deeper bottom margin are what keep this line
        # inside the canvas - at the base 268/36 it renders below the cut
        fig.update_layout(height=300, margin=dict(l=28, r=28, t=64, b=74))
    return fig


# The standing caveat about what the dial is. It is CONSTANT text, so it is
# a module constant rather than something rebuilt per chart, and it lives in
# the info tooltip rather than on the page (recorded decision).
# THE SEVEN SIGNAL STATES, in one place (recorded decision:
# "a ? for the signal state part so I know the different states like
# blow-off"). Two families: a FIRED FLAG owns the state for its 21-day
# episode window, and when no flag is live the phase clock reads the
# crowd's geometry. Kept here, beside GAUGE_HELP, so the tooltip and
# the resolver in _phase_of/_disp_state can never drift apart.
STATE_HELP = (
    "**The signal state answers one question: where is this name in the "
    "crowd cycle right now?** It is resolved in strict priority — a flag "
    "fired in the last 21 days owns the state for its whole episode "
    "window, so you never see two directions at once; otherwise the "
    "phase clock reads it.\n\n"
    "**After a flag fires (these win):**\n\n"
    "- **EXIT WINDOW** — a CUT EXPOSURE fired within 21 days. The desk read "
    "is *reduce*, never short: manias overshoot, and the model's own "
    "record shows the upper quartile keeps rallying.\n"
    "- **ENTRY WINDOW** — a INCREASE EXPOSURE fired within 21 days. The crowd is "
    "arriving and the move is still young.\n\n"
    "**Otherwise, the phase clock** — two smooth coordinates (how "
    "EXTREME the crowd is × how fast it is still ARRIVING), so it cannot "
    "read entry and exit at once and cannot flip overnight:\n\n"
    "- **QUIET** — no meaningful crowd state either way. The default, "
    "and most names most of the time.\n"
    "- **BUILDING** — the crowd is arriving but is not yet extreme. The "
    "entry side of the clock: interest is growing from a low base.\n"
    "- **BLOW-OFF** — extreme AND still arriving fast. Late-stage: "
    "historically the rally often runs on for a while yet, which is "
    "exactly why it is dangerous — the tail risk is building "
    "underneath. Not the peak, and deliberately NOT a CUT EXPOSURE on its "
    "own.\n"
    "- **TOPPING** — extreme, and the arrivals are dying. The exit side "
    "of the clock: the last buyer has bought.\n"
    "- **COOLING** — the crowd is fading and neither rule is close. The "
    "episode, if there was one, is over.\n\n"
    "**A state is DISPLAY ONLY.** It never fires anything — the frozen "
    "walk-forward rules fire every flag, and the clock just narrates "
    "where the name sits between them. A name can sit in BLOW-OFF for "
    "weeks without a CUT EXPOSURE, and that is the system working: the flag "
    "waits for its own evidence."
)


GAUGE_HELP = (
    "**What the dial is.** A STATE, not an instruction. INCREASE EXPOSURE and CUT EXPOSURE "
    "come from the walk-forward detector and can fire with the needle "
    "anywhere; the bands only say how crowded this name is and what days "
    "like it did next.\n\n"
    "**Why the edges sit where they do.** Red starts at {red} - the same "
    "level the walk-forward picked for the END alert, not a second "
    "threshold invented for the dial. Amber starts at {amber}, the lowest "
    "level whose edge over the base rate survives all five bootstrap "
    "seeds; below it the difference is inside the noise.\n\n"
    "**Needle vs window high.** The needle is TODAY (the last day of the "
    "selected window). The window high is the hottest the same curve got "
    "inside that window. They differ whenever a name earns its place on "
    "the page with an episode that has already cooled - which is most of "
    "the time, because the page is ordered by most recent signal.")


def gauge_headline(level_now, in_danger, z, peak_val=None, peak_day=None):
    """The ONE line that sits next to the dial.

    Everything quantitative moved into the tooltip; what stays on the page is
    the reading itself and, when they differ, the window's high point. The
    second clause is the answer to "why does this say calm?" - without it a
    correct-but-stale dial looks like a broken one.
    """
    key, label, _ = gauge_state(level_now, in_danger, z)
    if key == "unknown":
        return ("**No gauge reading** - no frozen gauge record on disk, so the "
                "band edges have no evidence behind them and are not "
                "invented here.")
    now_txt = {"red_danger": "**RED ZONE - and the price has already run up**",
               "red": "**RED ZONE**",
               "amber": "**Warming**",
               "calm": "**Calm**"}[key]
    out = f"{now_txt} today ({float(level_now):.0f}/100)."
    if peak_val is not None and peak_day is not None \
            and float(peak_val) - float(level_now) >= 1.0:
        _, pk_label, _ = gauge_state(peak_val, False, z)
        out += (f" Peaked at **{float(peak_val):.0f}** ({pk_label}) on "
                f"{pd.Timestamp(peak_day).strftime('%d %b')}"
                f" - this name is on the page because of that run, "
                f"not because of today.")
    else:
        out += " This is the hottest point of the selected window."
    return out


def gauge_caption(level_now, in_danger, z):
    """One sentence under the dial, quoting the MEASURED risk of the band
    the needle is actually in - not a generic legend.  Every percentage
    here is read out of gauge_zones.json."""
    key, _, _ = gauge_state(level_now, in_danger, z)
    base = z.get("base_rate")
    bands = z.get("bands", {})
    red, amber = z.get("red_edge"), z.get("amber_edge")
    b_red = bands.get(f"level >= {red}", {})
    b_amber = bands.get(f"level >= {amber}", {})
    b_both = bands.get(f"level >= {red} AND danger state", {})
    horizon = ("a fall of 10% or more over a week starting within the "
               "next month")
    common = (f"For scale: across "
              f"{z.get('panel', {}).get('names', '?')} instruments since "
              f"{z.get('panel', {}).get('first', '?')}, that happened on "
              f"**{_pct(base, 0)}** of *all* days.")
    if key == "red_danger":
        return (f"**Needle in the red zone AND the price has already run "
                f"up.** On days like this {horizon} followed "
                f"**{_pct(b_both.get('p'), 0)}** of the time — the "
                f"strongest get-out state the record contains. " + common)
    if key == "red":
        return (f"**Needle in the red zone** (level {red}+, the "
                f"walk-forward END level). "
                f"{horizon.capitalize()} followed "
                f"**{_pct(b_red.get('p'), 0)}** of such days. The reading "
                f"gets much sharper once the price has *also* run up "
                f"(**{_pct(b_both.get('p'), 0)}**). " + common)
    if key == "amber":
        return (f"**Warming** (level {amber}–{red}). This is the lowest "
                f"level at which the crowd measurably shifts the odds: "
                f"**{_pct(b_amber.get('p'), 0)}** vs "
                f"**{_pct(base, 0)}** elsewhere. Below {amber} the "
                f"difference is not distinguishable from zero. " + common)
    if key == "calm":
        return (f"**Calm** (below {amber}). At these levels the crowd says "
                f"nothing measurable about {horizon} — the difference from "
                f"the base rate is inside the noise. " + common)
    return ("No gauge reading: no frozen gauge record on disk, so the band "
            "edges have no evidence behind them and are not invented here.")


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


def fig_theme_pulse(share, sent, px, symbol, title,
                    att_label="attention (share of chatter, %)",
                    att_axis="share of chatter (%)",
                    zero_line=False,
                    mood_label="sentiment (net-bullish, 28d)",
                    mood_relative=False):
    """THE PM CHART (reworked 2026-08-09, desk: "separate these charts -
    it's hard to read; the attention is way too noisy; fill the gaps in
    visually"): three STACKED panels on one shared time axis - attention,
    price, sentiment - each on its own scale, so nothing overlaps and no
    dual-axis reading is needed.

      row 1  attention (coverage-robust share), displayed as a trailing
             7-day mean on top of the estimator - a DISPLAY smooth only,
             the stored series and every signal are unchanged. Weeks with
             too few posts are bridged visually (time-interpolation) and
             marked with a lighter dotted stretch - the line never breaks,
             and the bridge is never mistaken for measured data.
      row 2  the anchor instrument's price.
      row 3  net-bullish sentiment (-1..+1, 28d post-weighted).
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.40, 0.34, 0.26],
                        vertical_spacing=0.055)
    # --- row 1: attention, display-smoothed + visually gap-bridged ----
    sm = share.rolling(7, min_periods=1).mean()
    filled = sm.interpolate(method="time", limit_area="inside")
    gap = share.isna()
    if gap.any() and share.notna().any():
        keep = (gap | gap.shift(1, fill_value=False)
                | gap.shift(-1, fill_value=False))
        bridge = filled.where(keep)
        if bridge.notna().any():
            fig.add_trace(go.Scatter(
                x=bridge.index, y=bridge.values,
                name="bridged (not enough posts, dotted)",
                line=dict(color=BLUE, width=1.4, dash="dot"),
                opacity=0.45, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=filled.index, y=filled.values, name=att_label,
        line=dict(color=BLUE, width=2.2)), row=1, col=1)
    if zero_line:
        fig.add_hline(y=0, line_color=INK_LABEL, line_width=1,
                      opacity=0.7, row=1, col=1)
    fig.update_yaxes(title_text=att_axis, row=1, col=1)
    # --- row 2: price ------------------------------------------------
    if px is not None and not px.empty:
        fig.add_trace(go.Scatter(
            x=px.index, y=px.values, name=f"{symbol} price",
            line=dict(color=GRAY, width=1.6)), row=2, col=1)
    fig.update_yaxes(title_text="price (USD)", row=2, col=1)
    # --- row 3: sentiment --------------------------------------------
    if sent is not None and sent.notna().any():
        s_filled = sent.interpolate(method="time", limit_area="inside")
        fig.add_trace(go.Scatter(
            x=s_filled.index, y=s_filled.values, name=mood_label,
            line=dict(color=BULL, width=1.6)), row=3, col=1)
        smin = float(sent.min(skipna=True))
        smax = float(sent.max(skipna=True))
        if mood_relative:
            # centred on 0 = "as bullish as the market today", so the
            # line reads both ways instead of hugging the top
            _lim = max(0.12, abs(smin), abs(smax)) * 1.15
            _rng = [-_lim, _lim]
            _ttl = "vs market"
        else:
            _rng = [min(-1.0, smin), 1.0]
            _ttl = "net-bullish"
        fig.update_yaxes(range=_rng, zeroline=True, zerolinewidth=1.4,
                         zerolinecolor=INK_LABEL, title_text=_ttl,
                         row=3, col=1)
    fig.update_layout(title=dict(text=title, y=0.98, x=0.01),
                      height=560, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=20),
                      legend=dict(orientation="h", yanchor="top",
                                  y=-0.10))
    return _axes_fidelity(_theme(fig))


def _attention_mode_control(key):
    """The desk toggle (2026-08-07): the attention line as the LEVEL
    (coverage-robust share of chatter) or its FIRST DERIVATIVE (the
    smoothed day-on-day change - 'is the crowd arriving or leaving',
    regardless of how big it already is)."""
    return st.radio(
        "attention line", ["level (% of mentions)",
                           "change (first derivative, pp/day)"],
        horizontal=True, key=key,
        help="LEVEL = the coverage-robust share of everything retail "
             "posted. CHANGE = its smoothed day-on-day move: positive = "
             "the crowd is arriving, negative = leaving. Same estimator "
             "underneath either way.")


def _mood_mode_control(key):
    """The mood line: RELATIVE to the market's mood (default) or the raw
    net-bullish level (design review: "how come net
    bullishness is always positive?").

    Measured on this store: 46% of posts score bullish vs 24% bearish,
    so the raw line is positive on ~82% of theme-days - part real (retail
    is structurally long), part the lexicon reading ordinary market
    language as upbeat. Subtracting the market's own mood that day inflections
    an always-positive level into a readable "more or less excited than
    everywhere else"."""
    return st.radio(
        "mood line", ["vs the market's mood that day",
                      "raw net-bullish level"],
        horizontal=True, key=key,
        help="Retail chatter is structurally bullish, so the RAW line "
             "sits above zero almost always and only its wiggles carry "
             "information. VS THE MARKET subtracts the same-day mood "
             "across every tracked name: 0 = as bullish as everyone "
             "else, above = this crowd is unusually excited, below = "
             "unusually cool. Same correction the forward-return study "
             "applies to price.")


def _mood_series(mode, sent_df, entity_col, name, lo, hi):
    if sent_df is None:
        return None, "sentiment (net-bullish, 28d)", False
    if mode.startswith("raw"):
        return (sentiment_series(sent_df, entity_col, name, lo, hi),
                "mood (raw net-bullish, 28d)", False)
    return (relative_sentiment_series(sent_df, entity_col, name, lo, hi),
            "mood vs the market that day (28d)", True)


def _attention_series(mode, entity_counts, entity_col, name, lo, hi):
    if mode.startswith("level"):
        return (mention_share_series(entity_counts, entity_col, name,
                                     lo, hi),
                "attention (share of chatter, %)",
                "share of chatter (%)", False)
    return (chatter_change_series(entity_counts, entity_col, name, lo, hi),
            "attention change (pp/day, smoothed)",
            "change in share (pp/day)", True)


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



# ---------------------------------------------------------------------------
# sidebar: header, window controls, pipeline runners
# ---------------------------------------------------------------------------
st.markdown(INSTITUTIONAL_CSS, unsafe_allow_html=True)

# ---- IS THIS THE NEWEST DATA?  (2026-09-04)
#
# The masthead used to print `last update: <now>`, which is the moment
# THE PAGE RENDERED - it says "just updated" on a dashboard whose data
# stopped three weeks ago, and a reader has no way to tell.  What a
# reader actually needs to know is the date of the newest reading and
# whether the pipeline is behind on it.
#
# BUSINESS days, not calendar days.  The pipeline has nothing to add on
# a Saturday, so a Monday morning looking at Friday's data is CURRENT.
# Counting calendar days would paint an amber line across the masthead
# every single Monday, and a warning that cries wolf weekly is a
# warning nobody reads.
#
# The bands: 0-1 business days behind is normal (today's run may not
# have happened yet).  2-4 is LATE - probably a missed run, possibly
# nothing.  5+ is a whole working week with no new data: STALE, and by
# then something is wrong rather than merely late.
_FRESH_OK, _FRESH_LATE, _FRESH_STALE = "ok", "late", "stale"
_FRESH_LATE_BD = 2
_FRESH_STALE_BD = 5


def _data_freshness(data_max, today):
    """(level, business_days_behind) for the newest data date.

    Pure - no Streamlit, no globals - so the bands can be tested
    without standing the app up.  An absent or unreadable date counts
    as STALE: silence about freshness is the failure mode this exists
    to remove."""
    import numpy as _np
    if data_max is None or pd.isna(data_max):
        return _FRESH_STALE, None
    _b = max(0, int(_np.busday_count(pd.Timestamp(data_max).date(),
                                     pd.Timestamp(today).date())))
    if _b >= _FRESH_STALE_BD:
        return _FRESH_STALE, _b
    if _b >= _FRESH_LATE_BD:
        return _FRESH_LATE, _b
    return _FRESH_OK, _b


h_left, h_right = st.columns([5, 1])
with h_left:
    # EDITORIAL MASTHEAD (experiment). Same elements, same order, same
    # column split, same header mark - the description simply moves from
    # a small uppercase line to a standfirst, with an eyebrow above the
    # name. Reverting is this block plus the editorial CSS layer.
    st.markdown(
        '<div class="rf-eyebrow">Retail attention &amp; trading signals</div>'
        '<div><span class="rf-dot">&#9679;</span> '
        '<span class="rf-title">RetailRadar</span></div>'
        '<div class="rf-standfirst">A real-time read on where retail '
        'attention is building, and where it is ending.</div>',
        unsafe_allow_html=True)
    # THE FRESHNESS LINE goes here, but the newest data date is not
    # known until the aggregates load a hundred lines below - so the
    # masthead reserves the slot now and fills it then. Keeping it in
    # the masthead (rather than moving the load up) means the header
    # still paints immediately on a cold cache.
    _fresh_slot = st.empty()
    st.markdown(
        # ON THE HOSTED COPY, NO ORG BRANDING (request: "no mention of
        # GIC / its things in the streamlit"). LOCAL_CONTROLS rides on
        # the gitignored .local_controls file, so the desk machine
        # keeps the full credit and the hosted clone never sees it -
        # one codebase, no second branch to maintain.
        ('<div class="rf-credit">Alex Brown - GIP 2026 Project - '
         'MAARS Global Macro</div>' if LOCAL_CONTROLS else
         '<div class="rf-credit">Alex Brown - Intern 2026 Project'
         '</div>'),
        unsafe_allow_html=True)
with h_right:
    # the animated header mark reads as an org logo - hosted copy
    # shows nothing in that corner
    if LOCAL_CONTROLS:
        st.markdown(HEADER_MARK_HTML, unsafe_allow_html=True)
st.markdown('<div class="rf-rule-heavy"></div>', unsafe_allow_html=True)

st.sidebar.title("RetailRadar")

# ---- WHICH BUILD AM I LOOKING AT?  (2026-07-29)
#
# `.streamlit/config.toml` sets `fileWatcherType = "none"` on purpose - the
# watcher restarting mid-read while the pipeline rewrites parquet in place is
# a source of spurious errors.  The cost is that an edited dashboard.py is
# NOT picked up automatically, and there is no banner saying so.  Worse, a
# second `streamlit run` while one is already listening quietly takes the
# next port (8501 -> 8502), so a pinned browser tab can keep serving the OLD
# process indefinitely - which looks exactly like "the change did not land".
#
# This line makes that distinguishable in one glance: it is the modification
# time and size of THE FILE THIS PROCESS IS EXECUTING, read at import from
# `__file__`.  If it does not match the file on disk, the tab is stale.
# Cheap, out of the way, and it answers the question without a terminal.
try:
    _bs = os.stat(__file__)
    _bt = (pd.Timestamp(_bs.st_mtime, unit="s", tz="UTC")
           .tz_convert(None).strftime("%d %b %H:%M"))
    st.sidebar.caption(f"build: {_bt} UTC · {_bs.st_size // 1024} KB · "
                       f"port {st.get_option('server.port')}")
except OSError:
    _bs = None


# ---- ... AND IS IT THE BUILD ON DISK?  (2026-08-04)
#
# The caption above was not enough, and the incident that proved it is worth
# writing down.  A corrected anchor (china_geopolitics KWEB -> FXI) was
# edited into config/theme_etfs.csv and a corrected dashboard.py was saved
# beside it.  Neither appeared.  Three separate mechanisms were involved and
# each one alone is invisible:
#
#   1. the watcher is OFF (see .streamlit/config.toml), so an edited
#      dashboard.py is never picked up by a running server;
#   2. `src/themes.py` builds the anchor map at IMPORT, and a Streamlit
#      rerun does not re-import a module that is already in sys.modules
#      (fixed separately - see `_theme_etf_maps`);
#   3. a second `streamlit run` against a busy port quietly takes the next
#      one, so the pinned tab keeps serving the ORIGINAL process forever.
#
# The user-visible symptom of all three is identical and misleading: "the
# fix did not work".  Hours go into re-checking correct code.
#
# `st.cache_resource` is per-PROCESS and survives reruns, so the mtime it
# returns is the one this process saw when it started.  Comparing that to
# the file on disk right now detects every case above, including the pinned
# stale tab - the old process still answers, and now it says so.
@st.cache_resource(show_spinner=False)
def _mtime_at_process_start(path):
    return _mtime(path)


if _bs is not None:
    _started_with = _mtime_at_process_start(__file__)
    if _mtime(__file__) > _started_with + 1:      # 1s: mtime granularity
        _edited = (pd.Timestamp(_mtime(__file__), unit="s", tz="UTC")
                   .tz_convert(None).strftime("%d %b %H:%M"))
        st.sidebar.error(f"**Stale tab.** This server started on the "
                         f"{_bt} UTC build; dashboard.py on disk was "
                         f"edited at {_edited} UTC. The file watcher is "
                         f"off by design, so **restart the server** - "
                         f"stop every running `streamlit` first, or the "
                         f"new one takes the next port and this tab keeps "
                         f"serving the old build.")
        st.error(f"You are looking at the {_bt} UTC build of dashboard.py. "
                 f"A newer one ({_edited} UTC) is on disk and is NOT "
                 f"loaded. Restart the server - see the sidebar.")

theme_counts = load(THEME_COUNTS)
# ticker mentions, for the euphoria tabs' attention sort (singles side)
ticker_counts = load(TICKER_COUNTS)
# per-day model component readings (written by the phases step from
# 2026-08-31; absent on older bundles - the hover then falls back to
# the observable-ingredient bars)
desk_components = load("euphoria_desk_components.parquet")


@st.cache_resource(show_spinner=False)
def _desk_weights():
    """The frozen logit weights + plain labels, for the hover's
    stacked reading-x-weight bar. None when the insight file is
    absent or unreadable."""
    _p = os.path.join(PROCESSED_DIR, "desk_model_insight.json")
    if not os.path.exists(_p):
        return None
    try:
        _j = json.load(open(_p))
        return {"in": (_j.get("get_in") or {}).get("logit_weights")
                or {},
                "out": (_j.get("get_out") or {}).get("logit_weights")
                or {},
                "labels": _j.get("plain_labels") or {}}
    except Exception:                                    # noqa: BLE001
        return None
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

# the DESK CONFIGURATION store (recorded decision): the GET IN /
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

# episodes.parquet is deliberately NOT loaded here. It was the ground truth
# behind the window-adaptive scorecard; with that strip removed (recorded decision;
# 2026-07-28, see the euphoria tab) nothing on this dashboard scores itself, so
# loading it would be a read with no reader. The file is unchanged on disk and
# is still the ground truth every notebook judges against.


# per-theme bullishness, for the AI Pulse date slider's market read
theme_sentiment = load("daily_theme_sentiment.parquet")


@st.cache_data(show_spinner=False)
def _ai_pulse_load_cached(mtime):
    from analytics.ai_pulse import load as _pl
    return _pl()


def _ai_pulse_load():
    return _ai_pulse_load_cached(
        _mtime(os.path.join(PROCESSED_DIR, "ai_pulse.json")))


@st.cache_data(show_spinner=False)
def _agentic_load_cached(mtime):
    from src.agentic_watch import load_series
    return load_series()


def _agentic_load():
    return _agentic_load_cached(
        _mtime(os.path.join(PROCESSED_DIR,
                            "daily_agentic_counts.parquet")))


@st.cache_data(show_spinner=False)
def _ai_poll_load_cached(mtime):
    from analytics.ai_poll import load_series
    return load_series()


def _ai_poll_load():
    return _ai_poll_load_cached(
        _mtime(os.path.join(PROCESSED_DIR, "ai_poll.parquet")))


prices = _read(PRICES_PATH, _mtime(PRICES_PATH)) if os.path.exists(PRICES_PATH) else None
priced = set(prices["symbol"]) if prices is not None else set()

if theme_counts is None:
    st.error("No aggregate data - run update_data.py first.")
    st.stop()

data_max = theme_counts["date"].max()
today = pd.Timestamp.today().normalize()
# ---- FILL THE MASTHEAD FRESHNESS LINE (see _data_freshness above).
# This is the ONE place on the page that answers "am I looking at
# current numbers?", so it names the newest reading's date either way -
# and when the pipeline is behind it says so in a full sentence and
# tells the reader what to do about it. The instruction differs by
# copy: on the desk machine the reader IS the person who can fix it, so
# it names the command; on the hosted clone the reader usually is not,
# so it points them at the owner.
_fresh_lvl, _fresh_bd = _data_freshness(data_max, today)

# ---- ...AND IS THE PAGE DRAWING WHAT WAS PUBLISHED?
#
# The age check above answers "has the pipeline run lately". It cannot
# answer the OTHER question a stale-looking page raises: "I refreshed
# it - why is it still old?" Those two have the same symptom and
# opposite fixes, and a page that cannot tell them apart sends the
# reader to re-run a pipeline that already ran.
#
# tools/publish_dashboard.py writes publish_manifest.json INTO the
# committed bundle, so the checkout carries a record of what the last
# publish contained. If that record is ahead of the data this page
# actually loaded, the refresh reached the repository and stopped
# there - a bundle that never got placed into data/processed, or a
# container still serving the previous copy. That is a deploy problem,
# not a data problem, and it is the only one of the two a running page
# can diagnose about itself.
_pub_path = os.path.join(BUNDLE_DIR, "publish_manifest.json")
_pub = (_read_json(_pub_path, _mtime(_pub_path))
        if os.path.exists(_pub_path) else None)
_pub_through = _pub_at = None
if isinstance(_pub, dict):
    try:
        if _pub.get("data_through"):
            _pub_through = pd.Timestamp(_pub["data_through"])
        if _pub.get("published_at"):
            _pub_at = pd.Timestamp(_pub["published_at"]).tz_localize(None)
    except (ValueError, TypeError):       # a hand-edited manifest
        _pub_through = _pub_at = None
# Only ONE direction is a fault. Published AHEAD of what loaded = the
# deploy is half-applied. Published BEHIND = the desk machine has run
# the pipeline and not published yet, which is the normal state of a
# workstation mid-morning and must stay silent.
_deploy_behind = (_pub_through is not None and pd.notna(data_max)
                  and _pub_through > pd.Timestamp(data_max))

_fresh_fix = ("Restart the Streamlit server on this machine."
              if LOCAL_CONTROLS else
              "Contact the dashboard owner to refresh it or check "
              "for issues.")
if _deploy_behind:
    _fresh_slot.markdown(
        f'<div class="rf-stale" style="color:{BEAR};border-color:{BEAR}">'
        f'<b>This page is not drawing the newest published data.</b> '
        f'The published bundle runs to '
        f'{_pub_through:%d %b %Y}'
        + (f' (published {_pub_at:%d %b %H:%M} UTC)' if _pub_at is not None
           else '')
        + f', but this page has loaded data through '
        f'{pd.Timestamp(data_max):%d %b %Y} &mdash; the refresh reached '
        f'the repository and not this app. {_fresh_fix}</div>',
        unsafe_allow_html=True)
elif _fresh_lvl == _FRESH_OK:
    _fresh_slot.markdown(
        f'<div class="rf-sub">data through '
        f'{pd.Timestamp(data_max):%d %b %Y}'
        + (f' &middot; published {_pub_at:%d %b %H:%M} UTC'
           if _pub_at is not None else '')
        + f' &middot; page loaded {pd.Timestamp.now():%d/%m/%Y, %H:%M}'
        '</div>',
        unsafe_allow_html=True)
else:
    # The pipeline itself is behind. Same slot, escalating colour, and
    # the instruction differs by copy: on the desk machine the reader
    # IS the person who can fix it, so it names the command; on the
    # hosted clone the reader usually is not, so it points at the owner.
    _fresh_col = BEAR if _fresh_lvl == _FRESH_STALE else OCHRE
    _fresh_lead = ("This dashboard is NOT showing the newest data"
                   if _fresh_lvl == _FRESH_STALE
                   else "This dashboard may not be showing the newest "
                        "data")
    _fresh_age = (f"{_fresh_bd} business day"
                  f"{'' if _fresh_bd == 1 else 's'} behind"
                  if _fresh_bd is not None else "age unknown")
    _fresh_run = ("Run <code>update_data.py</code> on this machine to "
                  "refresh it, or check the pipeline log for errors."
                  if LOCAL_CONTROLS else
                  "Contact the dashboard owner to refresh it or check "
                  "for issues.")
    _fresh_slot.markdown(
        f'<div class="rf-stale" style="color:{_fresh_col};'
        f'border-color:{_fresh_col}">'
        f'<b>{_fresh_lead}.</b> The newest reading is '
        f'{pd.Timestamp(data_max):%d %b %Y} &mdash; {_fresh_age}. '
        f'{_fresh_run}</div>',
        unsafe_allow_html=True)
# default view: 1 Jan 2026 onwards (the start of dense backfilled
# coverage); falls back to trailing-365d if the data ends before that
_default_lo = pd.Timestamp("2026-01-01")
if data_max <= _default_lo:
    _default_lo = data_max - pd.Timedelta(days=365)
lo = pd.Timestamp(st.sidebar.date_input("window start", _default_lo.date()))
live_mode = st.sidebar.checkbox("LIVE (to newest data)", value=True)
hi = None if live_mode else pd.Timestamp(
    st.sidebar.date_input("window end", data_max.date()))
# HOW MANY CHARTS (design review: "why are there so little charts
# displayed? like only 8 charts - also how do you select what do to show?").
#
# THREE filters stack, and only the last one is a preference:
#   1. the window - a name is only considered if it has euphoria rows in it;
#   2. IT MUST HAVE ALERTED INSIDE THE WINDOW.  This is deliberate and stays:
#      an earlier requirement was "just show all the themes / tickers
#      that have had a euphoria detected in the time frame selected", i.e. no
#      filler names.  It is also what BINDS in the default window: 8 of 31
#      themes and 3 of the 8 present singles alerted between 1 Jan and 21 Jul
#      2026.  Over all history 30/31 themes and 22/23 singles have alerted, so
#      "only 8 charts" is a statement about the window, not about the cap;
#   3. this slider, newest-alert-first.
#
# The maximum was 15, which silently truncated the moment a wide window pushed
# the alerting set past it - and a silent cap reads as "that is all there is".
# The ceiling is now the largest set the filters can possibly produce (every
# theme plus every single that has ever alerted, with slack), so the slider can
# always be opened far enough to see everything, and the line printed above the
# charts states how many qualified versus how many are drawn.
how_many = st.sidebar.slider("items per section", 3, 60, 15)

# ---------------------------------------------------------------------------
# Signal configuration (final production values; provenance in
# docs/DECISIONS.md). None of these is user-selectable: every surface
# reads them through sig_col()/sig_head()/IN_SCORE/OUT_SCORE, so a
# single flag change here re-routes the whole page consistently.
#
# Column naming, stated once because it looks inverted and is not: the
# pipeline writes BOTH operating points on every run. `get_in_strict` /
# `get_out_strict` are the F0.5 (precision-weighted) cut; the bare
# `get_in` / `get_out` are the F1 cut. The parquet names are frozen -
# renaming them would break every stored record, notebook, and research
# JSON - so the mapping is documented here and fenced by a test.
#
# Both operating points are SHAPED identically: GET IN only pre-boom
# (re-arms at the train-median score), GET OUT only post-boom (the
# ground truth's own 120d bar, re-arms at the cut), one call per name
# per quarter per side, and no GET IN within 21d of a GET OUT in either
# direction. Evidence: docs/research/alert_shape_sweep.json.
# ---------------------------------------------------------------------------

# Trigger: SHIPPED (crowd + price). The experimental price-blind pair
# reads crowd features only and ranks days at walk-forward AUROC ~0.57
# against the shipped pair's ~0.73 (notebook 08 §8); it answers "what
# can the posts alone see" and is a research record, not a desk signal.
# Its *_xp columns are still written on every run; set XP_TRIGGER = True
# to inspect them - nothing else changes.
_XP_STORE_OK = desk is not None and "in_score_xp" in desk.columns
XP_TRIGGER = False
_XP_PART = "_xp" if XP_TRIGGER else ""

# Dial: % of signal. The needle is the live model score over that
# side's frozen cut, so 100 means the call fires, identically for every
# name and both sides. The unthresholded euphoria level survives only
# as an automatic fallback when a name has no score to be a percentage
# of, and as a descriptive row in the facts panel.
READINESS_DIAL = True

# Operating point: WIDER (the F1 columns). Rationale: a top visible on
# the chart should produce a call on the chart - the F0.5 cut declines
# calls that miss it by ~0.01 of probability (gold_metals' 2026-01-29
# peak: out_score 0.958 against a 0.970 cut, while the F1 call fired a
# day early and was simply not drawn). Measured trade, walk-forward
# (tools/sweep_operating_point.py), Wider vs Standard:
#   GET OUT  capture 34% -> 45%,  precision 37% -> 32%,  FA/iy 0.35 -> 0.58
#   GET IN   capture 39% -> 57%,  precision 58% -> 46%,  FA/iy 0.17 -> 0.40
# No threshold is refitted by this flag; it selects which stored column
# the page reads. NOTE: the research pack quotes the F0.5 (Standard)
# figures - a capture rate read off this screen is not the number in
# the pack. Set RELAXED_SIGNALS = False to align the two.
RELAXED_SIGNALS = True
_SIG_SUFFIX = "" if RELAXED_SIGNALS else "_strict"

# UNGATED GET IN (production; notebook 08 §10.4): the 120d phase
# gate was measured blocking ~3/4 of the correct GET IN calls, so the
# desk view routes GET IN through the *_nogate columns. GET OUT keeps
# its gate always - ungated, its false alarms double. A store written
# before that change lacks the *_nogate columns, so warn and fall back
# to the gated variant rather than silently showing the wrong thing.
_NOGATE_STORE_OK = (desk is not None
                    and f"get_in_nogate{_SIG_SUFFIX}" in desk.columns)
GATED_GET_IN = XP_TRIGGER
if not GATED_GET_IN and not _NOGATE_STORE_OK:
    st.sidebar.warning("The stored signals predate the ungated INCREASE EXPOSURE. "
                       "Run the pipeline once (python -m "
                       "analytics.run_analytics --what phases) to "
                       "compute it; falling back to the price-gated "
                       "INCREASE EXPOSURE until then.")
    GATED_GET_IN = True

# The live-score columns for the active trigger. Every surface that
# reads a score reads THESE, so the experimental mode can never show a
# shipped score against an experimental cut or vice versa.
IN_SCORE = f"in_score{_XP_PART}"
OUT_SCORE = f"out_score{_XP_PART}"


def sig_col(base, frame):
    """The desk-signal column for the active trigger + gate setting -
    falls back to the shipped standard column if the store predates the
    extra columns (the experimental case is warned about above, not
    silently substituted - this fallback only fires mid-render if a
    frame lacks the column). INCREASE EXPOSURE routes through the ungated variant
    (nb08 §10.4: the phase gate blocks ~3/4 of its correct calls);
    CUT EXPOSURE never does - its gate is load-bearing, ungated its false
    alarms double."""
    _gate_part = ("_nogate" if (base == "get_in" and not GATED_GET_IN
                                and not XP_TRIGGER) else "")
    _c = f"{base}{_XP_PART}{_gate_part}{_SIG_SUFFIX}"
    if frame is not None and _c in frame.columns:
        return _c
    _fallback = f"{base}{_XP_PART}{_SIG_SUFFIX}"
    return (_fallback if frame is not None and _fallback in frame.columns
            else base)


def sig_head(rep, head):
    """The desk-record block holding this head's frozen cuts under the
    active trigger ('get_in' / 'get_out')."""
    _r = rep or {}
    if XP_TRIGGER:
        return (_r.get("experimental_price_blind") or {}).get(head)
    return _r.get(head)


def sig_thr(head_rec):
    """The frozen cut for the active signal setting (display lines)."""
    _r = head_rec or {}
    if not RELAXED_SIGNALS and _r.get("strict_threshold") is not None:
        return _r.get("strict_threshold")
    return _r.get("live_threshold")


# Heading only where the buttons themselves render. On a hosted clone
# LOCAL_CONTROLS is False and every pipeline button is hidden, so an
# unconditional heading would leave a viewer staring at "Run the
# pipeline" with nothing beneath it.
if LOCAL_CONTROLS:
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
    "analyse":  ("Analysing: conviction, signals, crowd heat + onset radar, "
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
    "pulse":    ("AI: agentic scan, the retail-prompt poll and the LLM "
                 "market pulse (skips politely without the gateway)",
                 ["agentic scan", "AI POLL:", "AI PULSE:"]),
}
# which stages each pipeline actually goes through (in order)
# "analytics" and "full" plans removed with their buttons (2026-07-31):
# analytics-only is folded into the QUICK UPDATE plan ("window"), and the
# full historical rebuild is a shell-only operation on the machine that
# holds posts.parquet (python update_data.py --full).
PLANS = {
    "live":      ["fetch", "store", "coverage", "analyse", "prices",
                  "pulse", "wrapup"],
    "window":    ["prices", "coverage", "analyse", "pulse", "wrapup"],
    "comments":  ["comments", "influence"],
    "pulse":     ["pulse"],
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
        # THE LIVE LINE (recorded decision: "add the AI progress
        # to it too") - the newest log line, so slow in-process stages
        # (the 12-prompt AI poll, the two pulse calls) show their
        # step-by-step progress without opening the technical log.
        _tail = next((ln for ln in reversed(log_text.splitlines())
                      if ln.strip()), "")
        if _tail:
            box.caption("latest: " + _tail[-100:])
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

# buttons are disabled while a pipeline runs - one at a time, by design.
#
# SIMPLIFIED TO THREE (recorded decision: "simplify the refresh
# buttons ... remove the external machine one").  What happened to the two
# that left:
#   * "recompute analytics only" - folded into QUICK UPDATE, which already
#     ends with the same signal recompute; a no-API-only path saved ~60s
#     and cost a fifth button to explain.
#   * "run FULL historical rebuild" - REMOVED from the dashboard.  It only
#     ever worked on the machine that holds posts.parquet, so on this
#     terminal it was a button that could not do its job.  The capability
#     is unchanged from a shell:  python update_data.py --full
if LOCAL_CONTROLS and st.sidebar.button(
                     "QUICK UPDATE - Bloomberg prices + signals  (~1-3 min)",
                     disabled=_pipe_running,
                     help="No post fetching. Pull Bloomberg prices for the "
                          "chosen window (already-covered spans are "
                          "skipped), then recompute every signal from the "
                          "aggregates on disk. The everyday refresh."):
    start_pipeline([(["pull_bloomberg_prices.py"], win_env),
                    (["update_data.py", "--start", start_s, "--end", end_s,
                      "--skip-prices"], None)],
                   f"quick update {start_s} -> {end_s or 'LIVE'}",
                   plan="window")
if LOCAL_CONTROLS and st.sidebar.button(
                     "FULL UPDATE - live pull  (~10 min)",
                     disabled=_pipe_running,
                     help="Everything: fetch new posts AND comments from all "
                          "three sources, fold them in, recompute signals, "
                          "rescore the influence board, pull prices. Most of "
                          "the time is deliberate API rate-limit pacing; "
                          "whatever the comment budget cannot fetch is "
                          "picked up by the next run, never dropped. Run it "
                          "about twice a week."):
    start_pipeline([(["update_data.py"], None)], "LIVE pull", plan="live")
# Comments ARE part of the full update (recorded decision): this is
# the CATCH-UP button, for when the pipeline has been idle long enough that
# one budgeted run cannot close the gap. Its estimate is computed from the
# watermarks and this machine's measured throughput, not bracketed by hand.
_c_est = ""
if LOCAL_CONTROLS:
    try:
        from update_comments import estimate as _comment_estimate

        from ingestion.fetch_reddit_comments import (default_lookback_days
                                                     as _c_lookback)
        _c_est = _comment_estimate(_c_lookback(), None)
    except Exception:                                 # noqa: BLE001
        _c_est = ("estimate unavailable on this machine - the runner prints "
                  "one before it starts")
if LOCAL_CONTROLS and st.sidebar.button(
                     "EXTRA - catch up comments  (no page budget)",
                     disabled=_pipe_running,
                     help="NOT needed for the ordinary refresh - the full "
                          "update already fetches comments and rescores "
                          "the board within the desk's runtime ceiling. Use "
                          "this only after a long idle spell: it crawls the "
                          "whole owed window in one sitting, however long "
                          "that takes. Watermarked and resumable - "
                          "cancelling is always safe. "
                          f"Current estimate: {_c_est}"):
    start_pipeline([(["update_comments.py"], None)],
                   "comments catch-up", plan="comments")

if LOCAL_CONTROLS:
    with st.sidebar:
        pipeline_panel()

if prices is None and LOCAL_CONTROLS:
    st.sidebar.warning("prices.parquet missing - run pull_bloomberg_prices.py "
                       "(Terminal open) or use the rebuild button")
elif prices is None:
    st.sidebar.warning("Price history is not in the published bundle; "
                       "price-linked panels are unavailable.")

# ---- ACKNOWLEDGEMENTS ----------------------------------------------------
# Last element in the sidebar, so it sits at the bottom left of the page
# on every tab. The sidebar rather than the main column: main-page
# content varies in length per tab, which would leave the credit
# floating mid-page on short tabs and far below the fold on long ones.
st.sidebar.divider()
st.sidebar.caption(
    "Special thanks to Shawn, Wang Han, Jonathan, Henry, and the whole "
    "of FIMA and GIC that helped with this project."
    if LOCAL_CONTROLS else
    "Special thanks to Shawn, Wang Han, Jonathan, Henry, and everyone "
    "that helped with this project.")

# ---------------------------------------------------------------------------
# topline metric strip
# ---------------------------------------------------------------------------
_m3, _m4, _m5 = st.columns(3)
_hottest, _hottest_share = "-", None
if euph is not None and len(euph):
    # HOTTEST = the theme with the most RETAIL ATTENTION right now: the
    # largest share of the tradeable
    # universe's total mentions over the trailing 7 days - the same
    # arithmetic as the euphoria tabs' attention sort. It used to show
    # the top EUPHORIA LEVEL, which is a percentile of a name's own
    # history - a tiny theme at its own extreme could outrank the theme
    # the whole crowd is actually talking about.
    _hottest, _hottest_share = "-", None
    if theme_counts is not None and len(theme_counts):
        _tc_h = theme_counts[theme_counts["theme"].isin(THEME_ETFS)
                             & ~theme_counts["theme"].isin(HIDDEN_THEMES)]
        if len(_tc_h):
            _hi_h = _tc_h["date"].max()
            _w_h = _tc_h[_tc_h["date"] > _hi_h - pd.Timedelta(days=7)]
            _tot_h = _w_h["mention_count"].sum()
            if _tot_h > 0:
                _s_h = (_w_h.groupby("theme")["mention_count"].sum()
                        / _tot_h)
                # Name on the value line, share underneath: the share
                # was competing with the name for the eye inside one
                # string. delta_color="off" keeps it neutral grey - it
                # is a magnitude, not a rise or a fall.
                _hottest = theme_label(_s_h.idxmax())
                _hottest_share = f"{_s_h.max():.0%} of mentions"
# "crowd heat alerts in window" and "instruments at level 70+" removed
# on request - the landing lists already say what is live today, and
# the level counter belonged to the retired level detector.
_m3.metric("most retail attention (7d)", _hottest,
           delta=_hottest_share, delta_color="off")
_m4.metric("data through", str(data_max.date()))
_m5.metric("priced symbols", len(priced))

# ---- NO MODEL-EVIDENCE EXPANDERS ON THE PAGE (final
# requirement): a single explainer is kept; all other
# model-evidence expanders are removed from the page.
#
# Two sibling expanders used to sit here - "WHY THE MODEL DOES WHAT IT DOES"
# (`decisions_simple()`, 7 decisions each with the number that settled it)
# and "the long-form evidence log" (`DECISIONS_DOC`).  Both are REMOVED from
# the screen.  The reasoning they carried is not lost and was not weakened:
# it is the same content as docs/DECISIONS.xlsx, docs/RESEARCH_RECORD.md
# and the notebooks, which are the research record by standing instruction
# ("the dashboard shows conclusions only").
#
# WHAT THIS COSTS, recorded honestly because I argued the other way on
# 2026-07-28 and was overruled: a PM who challenges a threshold live can no
# longer answer it from this page.  The answer now requires the register or a
# notebook.  `decisions_simple()` (defined at the top of this file) and
# `DECISIONS_DOC` are deliberately left in place rather than deleted, so
# restoring this block is a two-line change if the desk wants it back.
# Recorded in DECISIONS.xlsx ("4. Detector Design") and docs/RESEARCH_RECORD.md
# Class 8.

# NOTE: individual-ticker overlays were removed from the dashboard by
# request - the desk trades THEMES via their anchor ETFs, never single
# tickers. The ticker analytics remain available in analytics/ for
# research (windowed backtests via run_analytics --what signals).
# PERSISTENT TAB BAR, NOT st.tabs (defect report: "sometimes
# when i flick between tabs it just gets stuck on a certain tab").
#
# THE CAUSE, so this is never reverted to st.tabs: st.tabs keeps its active
# tab CLIENT-SIDE only.  Every widget interaction reruns the script, the
# tab bar is rebuilt server-side with no memory of the selection, and the
# browser races to re-apply it - lose the race (slow rerun, another widget
# firing) and the page snaps to a tab you did not pick and appears stuck.
# st.tabs also renders ALL nine tabs on every rerun, so each flick paid for
# the whole dashboard - the lag is the same defect's other face.
#
# The fix is a radio bound to session_state: the selection is server-side
# state, so a rerun cannot lose it, and ONLY the active tab's code runs -
# a flick now costs one tab, not nine.
# TAB SET (final): "Overlays: themes",
# "Conviction" and "Historical checker" removed from the screen (the
# conviction ENGINE still runs in the pipeline and its store is still
# written - only the display went); "[dev] Data Stats" added - the
# snapshot of the data behind everything (freshness, volumes, sources,
# ingestion status).
# st.segmented_control - not st.radio, and still not st.tabs. The radio
# fixed the sticking defect but reads as a form control rather than as
# navigation. A segmented control is the same server-side-state widget
# wearing the right clothes, so both properties above survive: the
# selection lives in session_state (a rerun cannot lose it), and only
# the active tab's branch executes (a flick costs one tab, not all of
# them). required=True means the active tab cannot be deselected.
#
# Single names are no longer displayed. The detector still scores them
# and the store still carries them - only the screen changed, the same
# way Conviction and Historical checker were retired from the display.
MAIN_TAB = "Today's calls  ·  main"
# ORDER IS THE RANKING. The first three answer the questions this
# product exists for - what to act on, what the model reads in the
# posts, who is saying it - and they are emboldened in the bar so the
# hierarchy survives a glance. The rest are supporting views.
PRIMARY_TABS = 3
_TAB_NAMES = [MAIN_TAB, "AI Pulse", "Influence tracker",
              "Top trends", "Emerging trends", "[dev] Data Stats"]
active_tab = st.segmented_control(
    "view", _TAB_NAMES, key="active_tab", default=MAIN_TAB,
    required=True, label_visibility="collapsed")
# Belt and braces: required=True should make None impossible, but a
# session_state left by an older build could still yield one.
active_tab = active_tab or MAIN_TAB

# The ±90% readiness banner was removed. The landing page now leads
# with the names closest to firing, ranked by the same readiness
# number, so a strip repeating three of them above the fold was
# saying the page's own headline twice. readiness_alerts.json is
# still written by every run - only the banner went.

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
    the notebooks remain the research record.

    NO PERFORMANCE NUMBERS HERE (recorded decision).  A fourth block,
    "Why a red line matters - the measured version", used to sit between the
    gate list and "What this is NOT".  It quoted the 30-day cliff rates on a
    red-line day versus an ordinary day, the ten-day mean move against its
    baseline with the 90% interval, the precision of a red line, and the
    captured / detectable / false-alarm triple - all read live from the
    notebook JSON, all correct.  It is deleted on instruction: "remove the
    hit rate etc stuff, i will just keep this for the notebooks only".

    The reasoning is the standing division of labour, not a doubt about the
    figures.  A performance claim needs its interval, its window and its
    judge standing next to it to be read correctly, and none of that fits in
    a fifteen-second explainer; stripped of that scaffolding the numbers
    become slogans a PM repeats without the caveat.  They live in
    `docs/research/nb07_performance_battery.json` (the index-composite
    study's frozen record; its notebook was retired 2026-08-07), where
    the scaffolding is present.  What remains here is definitional: what euphoria means, what
    must be true before anything can fire, and what this is not - none of
    which moves when the record is re-measured.  Recorded in DECISIONS.xlsx
    ("4. Detector Design") and the parameter register (docs/RESEARCH_RECORD.md §7, Class 8)."""
    hype = f"{EUPHORIA_HYPE_MULT:.0f}x"
    return f"""
**Euphoria, in one sentence.** The crowd has stopped analysing a name and
started celebrating it. That is a late-stage condition, not a bullish one.

---

**The two lines on the chart - that is the whole signal.**

- **Green line = INCREASE EXPOSURE.** Euphoria is *starting*. The crowd is arriving.
- **Red line = CUT EXPOSURE.** Euphoria is *ending*. Historically the price top
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
- **The price has to have actually run** (for CUT EXPOSURE): the name must be up
  **{EUPHORIA_BOOM_MIN_ETF:.0%} (theme ETF)** or
  **{EUPHORIA_BOOM_MIN_SINGLE:.0%} (single name)** off its
  **{EUPHORIA_BOOM_WINDOW_D}-day** low.
  You cannot end a party that never started.
- **There has to be enough data** to measure at all.

---

**Every gate and every number, both directions.**

Requirement 2026-08-05: one table, so the whole rule set is readable in
one place instead of being reconstructed from the register. Every value
below is imported live from `src/config.py` - if a constant moves, this
table moves with it, and the two can never disagree.

| # | Gate | INCREASE EXPOSURE (euphoria starting) | CUT EXPOSURE (euphoria ending) | Why this number |
|---|---|---|---|---|
| A0 | Enough data to measure | {EUPHORIA_MIN_COVERAGE} tagged posts (posts naming it) in the trailing 28d, and {EUPHORIA_MIN_HISTORY}d of history | same | below this the percentile ranks are noise dressed as a signal |
| A1 | Crowd size vs its OWN normal | 7d mentions >= **{EUPHORIA_ONSET_HYPE_MIN:g}x** its 120d median | 7d mentions >= **{EUPHORIA_HYPE_MULT:g}x** its 120d median | INCREASE EXPOSURE has to fire EARLY, so its bar is lower - but not 1.0x, which breached the FA budget from the day it shipped (0.255 vs {EUPHORIA_FA_BUDGET_PER_IY}) |
| A2 | Attention extremity | mention share at or above its **{EUPHORIA_ATT_GATE:.0%}** percentile vs its own trailing {EUPHORIA_PCT_WINDOW}d | same | "extreme" always means extreme FOR THIS NAME - a permanently loud name is judged against loud-for-itself |
| A3 | The price actually ran | not required - the crowd arrives before the run | up **{EUPHORIA_BOOM_MIN_ETF:.0%}** (theme) / **{EUPHORIA_BOOM_MIN_SINGLE:.0%}** (single name) off its {EUPHORIA_BOOM_WINDOW_D}d low, sustained >= {EUPHORIA_BOOM_WINDOW_MIN_D}d | you cannot end a party that never started |
| A4 | Not already in the other phase | must not already be in the ending stage | must be in a confirmed boom | the two rules cannot both own the same day |

**The factors each rule scores** - the crowd does the predicting; price
only gates and grades.

| | INCREASE EXPOSURE bank | CUT EXPOSURE bank |
|---|---|---|
| Factors | attention acceleration, hype ratio, bull inflection, influx speed, attention convexity | E1 attention level, E2 sustained bull, E3 crowd influx, E5 super-exponential attention, E4 fade trigger |
| Threshold | frozen by walk-forward on strictly EARLIER years | same |
| Cooldown | {EUPHORIA_COOLDOWN_DAYS}d - one episode flags once | {EUPHORIA_COOLDOWN_DAYS}d |
| False-alarm budget | **{EUPHORIA_FA_BUDGET_PER_IY}** per instrument-year | **{EUPHORIA_FA_BUDGET_PER_IY}** |

**Ground truth - what counts as a real top when the detector is graded.**
All three must hold: a local price max (highest close within +/-21d); a
run-up of at least **{EUPHORIA_BOOM_MIN_ETF:.0%}** (theme) or
**{EUPHORIA_BOOM_MIN_SINGLE:.0%}** (single) off the trailing **120d** low;
and a fall of at least **{EUPHORIA_CRASH_MIN_ETF:.0%}** (theme) or
**{EUPHORIA_CRASH_MIN_SINGLE:.0%}** (single) within 90 days after it. Note
the 120d here is the GRADING window and is not the same as the
{EUPHORIA_BOOM_WINDOW_D}d boom gate in A3 above - one decides what counts
as a real top, the other decides when the detector is allowed to fire. A
false alarm costs a full captured top in threshold selection
(penalty {EUPHORIA_FA_PENALTY:g}).

---

**What this is NOT.**

- Not a price forecast, and not a short recommendation. Both were tested
  as trades and **rejected** - the numbers are in notebook 04.
- Not driven by the price chart alone: the crowd does the predicting, the
  price only gates and grades it.
- Not tuned on the days it is scored against. Every threshold is learned
  from **earlier years only** and then frozen.
"""


def euphoria_simple_ml(rep):
    """The PM explainer for the LEARNED desk model (adopted 2026-08-07).
    Same fifteen-second budget as euphoria_simple(), but the machinery it
    describes is the model tournament winner, not the gate stack - the
    whole point of the change was fewer hand-set constants to defend, so
    the explainer leads with the three numbers that remain."""
    rep = rep or {}
    model = rep.get("model", "ens")
    model_words = {
        "ens": "two models averaged - a logistic regression (one weight "
               "per measurement, readable as a formula) and a "
               "gradient-boosted tree constrained so more crowd-heat can "
               "only RAISE the score",
        "logit": "a logistic regression - one weight per measurement, "
                 "readable as a formula",
        "gbm": "a gradient-boosted tree constrained so more crowd-heat "
               "can only RAISE the score",
        "mlp": "a small neural network (16-8 hidden units)",
    }.get(model, model)
    thr_in = (sig_head(rep, "get_in") or {}).get("live_threshold")
    thr_out = (sig_head(rep, "get_out") or {}).get("live_threshold")
    return f"""
**Euphoria, in one sentence.** The crowd has stopped analysing a name and
started celebrating it. That is a late-stage condition, not a bullish one.

---

**The two lines on the chart - that is the whole signal.**

- **Green line = INCREASE EXPOSURE.** Euphoria is *starting*. The crowd is arriving.
- **Red line = CUT EXPOSURE.** Euphoria is *ending*. Historically the price top
  is close - this is the line to bring to a PM.

---

**How the signal decides (the 2026-08 model - it replaced a stack of
hand-set gates).** Eleven plain-English measurements go in - how loud
the name is against its own year, whether the crowd is still arriving
(over a fortnight and a month), this week against this month, this week
against the name's own normal, whether attention growth is itself
accelerating (the bubble signature), how bullish the mood is, how
one-sided it has been, whether the mood is turning, plus the price's
run-up off its 54-day low and its one-month return. A probability model
({model_words}) inflections them into "chance this is the start / the end of
a euphoria episode", trained ONLY on years before the one being scored.

**Only three numbers survive from the old rule stack, one sentence
each:**

- **Coverage floor** - {EUPHORIA_MIN_COVERAGE} tagged posts (posts
  naming the ticker/theme) in the last
  28 days: we do not diagnose a crowd we cannot see.
- **Cooldown** - one alert per name per {EUPHORIA_COOLDOWN_DAYS} days:
  one call per episode.
- **The trigger** - the probability cut ({'' if thr_in is None else
  f'currently {thr_in:.2f} in / {thr_out:.2f} out, '}chosen on PAST
  years only, by the standard precision-recall balance F1 - no hand-set
  budget, no penalty constant).

---

**Ground truth - what counts as a real episode when the model is graded**
(price-only, so the test is honest): a local price max (highest close
within ±21d), a run-up of at least
**{EUPHORIA_BOOM_MIN_ETF:.0%} (theme ETF) / {EUPHORIA_BOOM_MIN_SINGLE:.0%}
(single name)** off the trailing 120d low, and a fall of at least
**{EUPHORIA_CRASH_MIN_ETF:.0%} / {EUPHORIA_CRASH_MIN_SINGLE:.0%}** within
90 days after it. (Bars lowered 2026-08-07 after a sweep - the looser
definition keeps every named desk episode and the model's accuracy
IMPROVED under it; the sweep table is in the research record.)

---

**What this is NOT.**

- Not a price forecast, and not a short recommendation.
- Not tuned on the days it is scored against: every fit and every cut is
  learned from **earlier years only**, then frozen.
- Not a black box: the tournament table below shows every model that
  competed and the pre-stated rule that picked this one - and a START is
  still never shown on a day that already satisfies the ending
  conditions, nor within {EUPHORIA_COOLDOWN_DAYS} days after an END call.
"""


def tournament_table_md(rep):
    """The thesis-style comparison table, read live from the frozen desk
    record (never re-computed here): every model that competed, on both
    heads, with the score-quality and operational columns side by side."""
    rep = rep or {}
    tour = rep.get("tournament") or {}
    winner = rep.get("model", "?")
    label = {"rules": "Hand rules (incumbent)",
             "logit": "Logistic regression",
             "logit_crowd": "Logistic (crowd-only)",
             "gbm": "Monotone grad. boosting",
             "gbm_crowd": "Monotone GBM (crowd-only)",
             "mlp": "Neural network (MLP)",
             "mlp_crowd": "MLP (crowd-only)",
             "ens": "Ensemble: logit + GBM",
             "ens_crowd": "Ensemble (crowd-only)"}
    out = []
    for head, title in (("get_out", "CUT EXPOSURE: calling the top"),
                        ("get_in", "INCREASE EXPOSURE: calling the start")):
        rows = tour.get(head) or {}
        if not rows:
            continue
        out.append(f"**{title}** (walk-forward test years only)\n")
        out.append("| model | AP | AP lift | AUROC | episodes caught "
                   "| false alarms /instr-yr | precision |")
        out.append("|---|---|---|---|---|---|---|")
        order = ["rules", "logit_crowd", "gbm_crowd", "mlp_crowd",
                 "ens_crowd", "logit", "gbm", "mlp", "ens"]
        for m in [m for m in order if m in rows] + \
                 [m for m in rows if m not in order]:
            r = rows[m]
            ap, base = r.get("ap"), r.get("ap_baseline")
            lift = f"{ap / base:.2f}x" if ap and base else "-"
            mark = " **(adopted)**" if m == winner else ""
            cap = (f"{r.get('captured')}/{r.get('detectable')} "
                   f"({(r.get('capture_rate') or 0) * 100:.0f}%)")
            prec = (f"{r['precision']:.0%}"
                    if r.get("precision") is not None else "-")
            out.append(f"| {label.get(m, m)}{mark} | {ap} | {lift} | "
                       f"{r.get('auroc')} | {cap} | "
                       f"{r.get('fa_per_iy')} | {prec} |")
        out.append("")
    out.append(f"*Selection rule (stated before the numbers were "
               f"computed): {rep.get('selection_rule', '-')}. Raw AP is "
               "not comparable across candidacy frames - the incumbent's "
               "gates give it a frame where roughly half the candidate "
               "days are already positive - so the LIFT column (AP over "
               "its own frame's base rate) is the comparison that "
               "means something.*")
    return "\n".join(out)


EUPHORIA_DEF_FULL = """*(ARCHIVE NOTE, 2026-08-07: the text below describes the
RULES-era desk configuration. The desk signals are now scored by the
tournament-selected learned model — see "what is euphoria?" above and
notebook 03 §SS — while the CROWD-ONLY research detectors described
here run unchanged as the research baseline.)*

**EUPHORIA = the crowd has stopped analysing and started
celebrating.** Prediction is built from the **Reddit-derived data ONLY**
(selection rule, July 2026): price never enters the euphoria level or the
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

**The STARTING line (green)** comes from the onset detector (July-2026
phases study winner): the mean of five crowd-only onset features -
attention acceleration, hype ratio, bullish inflection, influx speed,
super-exponential attention - gated by coverage and by attention above
its own 120d median, at a frozen walk-forward threshold.

**What actually fires on THIS screen - the DESK CONFIGURATION (desk
decision 2026-07-24).** The desk lifted the crowd-only restriction for
the signals shown here ("use both price and the social media - I want a
better hit rate"), so the lines on these charts are a labelled SECOND
signal family; the crowd-only detectors above remain the research
headline, unchanged. **WARNING (red)** = the ending detector with (a)
candidacy requiring an ACTUAL price boom - G2's own size thresholds (≥25%
ETF / ≥50% single, over a trailing 54d low since 2026-07-29; 120d let
crash-rebounds through, and 54d is the max-capture point inside the
false-alarm budget on the NB07 frontier - see src/config.py) - which
raised walk-forward capture from 16 to 26 of 122 (gain CI [+3.5pp,
+13pp]; that gate ablation was measured with the 120d window and has
not been re-run), and (b) the trigger on the 7d-SMOOTHED score, which
killed the one-day-blip alerts (AP 0.435 → 0.449, same caveat, two
captures recorded as the cost). Since the 60d re-fit the shipped record
is capture 21/122, 15 FAs (0.100/instrument-year against a 0.23
budget), AP 0.540 against a 0.498 base rate, median warning 7 days. The
54d re-fit of 2026-07-29 supersedes those: 22/98, 10 FAs
(0.083/instrument-year), AP 0.615, 9-day warning; and CONSIDER came inside
its own budget for the first time at 0.200.
**CONSIDER (green)** = the onset detector made PHASE-AWARE: a day
that already satisfies every ending gate is end-stage, and a "start"
there is incoherent - so it cannot fire. That cut start-next-to-end
adjacency from 20 to 2 and late starts from 21 to 10, at a recorded
cost of captures (29 → 20 of 125): a recorded decision, made because a
START landing on an END destroys PM trust, and documented with the full
variant table in the strictness study's frozen record (nb06_strictness.json; notebook retired 2026-08-07).

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
# Recorded decision 2026-07-24: the dashboard shows CONCLUSIONS only - the
# state (starting / ending) drawn on the chart itself, one tab per
# instrument kind. All validation evidence (walk-forward tables, the
# ablation, the ML challenger, the tournament) lives in notebooks/01-04
# and docs/DECISIONS.xlsx, where research belongs.

RECENT_D = 21          # display window = the alert cooldown: one episode
#                        is "current" for one cooldown span



def _state_of(name, starting, ending):
    """STARTING / ENDING / quiet - the later phase wins a tie."""
    s, e = starting.get(name), ending.get(name)
    if s is not None and (e is None or s >= e):
        return "STARTING"
    if e is not None:
        return "ENDING"
    return None


def render_euphoria_tab(kind, kind_label, key_prefix, mode="full"):
    """mode="full" renders the complete tab, unchanged.
    mode="action" renders the landing page - two ranked lists and
    nothing else - reusing this function's own draw_chart so the
    detail can never drift from the full view."""
    # HEADING AND LEGEND ARE SEPARATE (defect report, with a
    # screenshot).  The legend used to be glued onto the subheader, so on a
    # normal-width screen the h2 wrapped and its second line read
    #
    #     GET OUT (euphoria ending; expect the top within ~a month)
    #
    # as a heading in its own right - sitting directly above the GREEN
    # GET IN banner.  A section header contradicting the box underneath it
    # is worse than no header at all.  The legend also said "blue" while
    # the banner it described has been green since the alert boxes went in.
    #
    # So: the heading says what the panel is, in one line that cannot wrap
    # into a false claim, and the legend is a caption under it naming the
    # colours that are actually on screen.
    st.subheader(f"CROWD HEAT - {kind_label}")
    st.caption("**Green = INCREASE EXPOSURE** — attention is building, the crowd is "
               "arriving, expect growth.  **Red = CUT EXPOSURE** — attention is "
               "topping, expect a fall within ~a month of the signal.")
    # ONE explainer, and its label is not to be touched (frozen
    # requirement): the explainer label is frozen.  The wording below is therefore
    # verbatim and deliberate - do not retitle it.
    #
    # Two deeper levels used to follow it: "full method & measured record"
    # (`EUPHORIA_DEF_FULL`) and "deep archive: every constant and its
    # evidence", which read docs/RESEARCH_RECORD.md off disk and printed
    # the whole file.  Both removed, same reasoning as the page-level block
    # above.  `EUPHORIA_DEF_FULL` stays defined at the top of this file (it is
    # the research text, still cited by the report) and the register is still
    # on disk - the dashboard simply stops being a second copy of them.
    # The "what is euphoria?" explainer was removed from the screen, and
    # the model-tournament expander nested inside it went with it. Same
    # reasoning as the page-level block above: the dashboard shows
    # conclusions, and the definition lives in docs/RESEARCH_RECORD.md.
    # `euphoria_simple()`, `euphoria_simple_ml()` and
    # `tournament_table_md()` are left defined and still tested, so
    # restoring the block is a paste, not a rewrite.

    # ---- ANCHOR TROUBLE, AND ONLY WHEN THERE IS ANY -------------------
    # The two reference expanders that used to sit here were removed on
    # recorded decision ("please remove these drop downs, only
    # keep the what is euphoria one"). The CHECK they performed is kept,
    # because it caught something real: `europe_defense` was being drawn
    # on ITA, a US aerospace line standing in for a European one, and
    # nothing said so.
    #
    # It is now silent by default. Nothing renders while every anchor is
    # priced and in use - so this costs no space on a normal day and
    # speaks only when a theme is quietly being drawn on something other
    # than the instrument it names.
    if kind == "theme":
        _sub, _dark = [], []
        for _t in sorted(THEME_ETFS):
            _live = _live_anchor(_t)
            if _live is None:
                _dark.append(_t)
            elif _live != THEME_ETFS[_t]:
                _sub.append(f"{_t} ({THEME_ETFS[_t]} → {_live})")
        if _dark:
            st.error("**No priced instrument at all: " + ", ".join(_dark)
                     + ".** These are counted in the crowd data and then "
                     "dropped before scoring, so the universe holds fewer "
                     "themes than the config defines. One Bloomberg pull "
                     "fixes it.")
        if _sub:
            st.warning("**Drawn on a fallback, not the named anchor: "
                       + ",  ".join(_sub) + ".** The anchor has no price "
                       "history, so the first priced line in the chain is "
                       "substituted - correct for an old backtest window, "
                       "wrong to leave unsaid on a live screen.")

    if euph is None or not len(euph):
        st.info("no crowd heat data yet - run QUICK UPDATE in the sidebar")
        return

    # THE WATCH TRACK needs the PRODUCTION scorers, not a reimplementation.
    # `desk_end_fit` / `desk_onset_fit` ignore their `train` argument (the desk
    # family is a rules family - fitting is a no-op), so calling them with
    # `train=None` on a wider frame reproduces the live arithmetic exactly and
    # cannot drift from it: if the scorer changes, this changes with it.
    from src.config import EUPHORIA_COOLDOWN_DAYS \
        as EUPHORIA_COOLDOWN_DAYS_DISP
    from analytics.euphoria_phases import (ONSET_BANK, TOP_FEATURES,
                                           desk_end_fit, desk_onset_fit,
                                           episode_coherent_alerts)

    ek = _hide(euph[euph["kind"] == kind])
    ok = (_hide(onset[onset["kind"] == kind].copy())
          if onset is not None and len(onset) else None)
    dk = (_hide(desk[desk["kind"] == kind].copy())
          if desk is not None and len(desk) else None)

    # THE SIGNAL SOURCE (recorded decision): GET IN /
    # GET OUT from euphoria_desk.parquet - the boom-gated SMOOTHED end
    # + phase-aware SMOOTHED onset the desk adopted in NB06 (adjacency
    # 20 -> 2, END AP 0.435 -> 0.449 in the 120d-gate era, no one-day
    # blips). Falls back to
    # the crowd-only research stores only if the desk store is missing.
    use_desk = dk is not None and len(dk)
    _mdl = (desk_report or {}).get("model", "rules")
    if use_desk and _mdl != "rules":
        _mdl_words = {"ens": "logit + monotone-GBM ensemble",
                      "logit": "logistic regression",
                      "gbm": "monotone gradient boosting",
                      "mlp": "neural network (MLP)"}.get(_mdl, _mdl)
        _ins_path = os.path.join(PROCESSED_DIR, "desk_model_insight.json")
        # Model weights are reference, not an action - the landing page
        # keeps only what changes a decision today.
        if os.path.exists(_ins_path) and mode == "full":
            with st.expander("what drives INCREASE EXPOSURE / CUT EXPOSURE — the "
                             "model's own weights", expanded=False):
                import json as _json_i
                _ins = _json_i.load(open(_ins_path))
                st.caption(
                    "Two independent reads of the LIVE fit. **Logit "
                    "weight**: the linear member's coefficient — "
                    "positive means more of this measurement raises the "
                    "call's probability (every weight is on the same "
                    "0-1 percentile scale, so sizes compare). **GBM "
                    "importance**: shuffle one measurement on held-out "
                    "days and record how much precision the tree member "
                    "loses — how much the model actually USES it. "
                    f"Fitted on years before "
                    f"{_ins.get('fitted_on_years_before', '?')}; "
                    "recomputed by every research/live pass.")
                st.markdown("**And how is an INFLECTION made?**")
                st.caption(
                    "A different head, on the same crowd measurements "
                    "plus four it does not share with the two above "
                    "(how UNSTABLE attention has been, how spread out "
                    "the mood is, attention x mood, and the change in "
                    "how many forums carry the name). It is trained on "
                    "a different question: not *is this the start* or "
                    "*is this the top*, but **is a REVERSAL about to "
                    "land**.\n\n"
                    "A day counts as a reversal when its close is the "
                    "highest (or lowest) of the 43 days centred on it "
                    "AND the move away from it over the next month is "
                    "at least 8% relative to the market - that second "
                    "test is what separates a real inflection from a flat "
                    "stretch that happens to contain a local high. The "
                    "head is asked whether such a day falls in the "
                    "NEXT TEN, so it is allowed to be early rather "
                    "than exact.\n\n"
                    "Two deliberate differences from INCREASE EXPOSURE / CUT EXPOSURE. "
                    "It has NO phase gate - a reversal is as "
                    "interesting at the bottom of a bust as at the top "
                    "of a boom - and it therefore fires in both "
                    "directions, which is why it never tells you "
                    "WHICH way. It is price-free like the other two, "
                    "walk-forward like the other two, and its cut is "
                    "frozen from past years like the other two. Full "
                    "record: the parameter register (docs/RESEARCH_RECORD.md §7, Class 3)d and "
                    "docs/research/inflection_trigger_sweep.json.")
                _cols = st.columns(2)
                for _c, (_hk, _ht) in zip(_cols, (("get_in", "INCREASE EXPOSURE — "
                                                   "what starts one"),
                                                  ("get_out", "CUT EXPOSURE — "
                                                   "what ends one"))):
                    _h = _ins.get(_hk) or {}
                    _rows_i = []
                    for _f in _ins.get("bank", []):
                        _rows_i.append({
                            "measurement": _ins["plain_labels"].get(_f, _f),
                            "logit weight":
                                (_h.get("logit_weights") or {}).get(_f),
                            "GBM importance":
                                (_h.get("gbm_permutation_importance")
                                 or {}).get(_f)})
                    _df_i = (pd.DataFrame(_rows_i)
                             .sort_values("GBM importance",
                                          ascending=False))
                    with _c:
                        st.markdown(f"**{_ht}**")
                        st.dataframe(_df_i, hide_index=True,
                                     width="content")
                _arch = os.path.join("docs", "figures", "deck",
                                     "F21_model_architecture.png")
                if os.path.exists(_arch):
                    st.image(_arch, caption="how a call is made: 9 crowd "
                             "measurements + 2 price measurements → two "
                             "models → rank ensemble → frozen cut → "
                             "INCREASE EXPOSURE / CUT EXPOSURE")
    if use_desk:
        src_in = dk[dk[sig_col("get_in", dk)].astype(bool)]
        # SINGLE-NAME DISPLAY BAR (recorded decision): a ticker
        # shows as "euphoria starting" only when its crowd cleared the
        # FULL A1 hype bar (2x its own 120d median). Really-euphoric
        # names only; marginal names cannot flicker in and out.
        if kind == "single" and "hype_raw" in dk.columns:
            src_in = src_in[src_in["hype_raw"] >= EUPHORIA_HYPE_MULT]
        src_out = dk[dk[sig_col("get_out", dk)].astype(bool)]
    else:
        src_in = (ok[ok["alert"]] if ok is not None
                  else ek.iloc[0:0])
        if (not use_desk and ok is not None and kind == "single"
                and "hype_raw" in ok.columns):
            src_in = src_in[src_in["hype_raw"] >= EUPHORIA_HYPE_MULT]
        src_out = ek[ek["alert"]]

    # EPISODE COHERENCE (selection rule 2026-07-24, asymmetric by
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

    # ---- SHARED EXPLANATION + OUTCOME MACHINERY --------------------------
    # One arithmetic, three surfaces (design brief: the hover, the
    # per-flag dropdowns and the per-chart "why did it fire" must never
    # disagree): every factor line a human reads is built by the SAME
    # helpers below, from the SAME stored rows the detector acted on.
    # Nothing here is recomputed into the model - display only.
    _oo_all = (onset[onset["kind"] == kind] if onset is not None
               and len(onset) else None)

    def _rows_for(name):
        """(levels, onset, desk) rows for one name, date-indexed, full
        history - the hover and the dropdowns read exact stored days."""
        li = (ek[ek["name"] == name].set_index("date").sort_index()
              if len(ek) else None)
        oi = (_oo_all[_oo_all["name"] == name].set_index("date").sort_index()
              if _oo_all is not None else None)
        di = (dk[dk["name"] == name].set_index("date").sort_index()
              if (dk is not None and len(dk)) else None)
        return li, oi, di

    _thr_in_d = sig_thr(sig_head(desk_report, "get_in"))
    _thr_out_d = sig_thr(sig_head(desk_report, "get_out"))

    def _row_at(idx_frame, d):
        """Exact stored row at d, else the latest row at-or-before d
        (weekends carry the last measured reading), else None."""
        if idx_frame is None or not len(idx_frame):
            return None
        if d in idx_frame.index:
            r = idx_frame.loc[d]
            return r.iloc[-1] if isinstance(r, pd.DataFrame) else r
        before = idx_frame.loc[:d]
        return before.iloc[-1] if len(before) else None

    def _fmt_rank(v):
        return f"{float(v):.2f}" if pd.notna(v) else "n/a"

    def _factor_lines(name, d, side):
        """Plain-English factor list for one name on one day, one side.
        Every value is the stored percentile of the name's OWN trailing
        year (1.00 = its most extreme); the gates carry their real
        thresholds. Returns (readiness_frac_or_None, [lines])."""
        li, oi, di = _rows_for(name)
        rd = _row_at(di, d)
        lines = []
        if side == "out":
            r = _row_at(li, d)
            if r is None:
                return None, ["no stored end-side reading near this day"]
            for c in ("e1", "e2", "e3", "e5"):
                lines.append(f"{PLAIN[c]}: **{_fmt_rank(r.get(c))}**")
            lines.append(f"{PLAIN['fade']}: "
                         f"**{'yes' if bool(r.get('fade')) else 'no'}**")
            hyp = rd.get("hype_raw") if rd is not None else None
            if pd.notna(hyp) if hyp is not None else False:
                lines.append(
                    f"gate - crowd size: **{float(hyp):.1f}x** its own "
                    f"normal (needs {EUPHORIA_HYPE_MULT:.1f}x)")
            else:
                lines.append("gate - crowd size: "
                             + ("open" if bool(r.get("hype_ok"))
                                else "shut (crowd not 2x its normal)"))
            if rd is not None and "boom_state" in rd.index:
                lines.append("gate - price in a confirmed boom: "
                             + ("open" if bool(rd.get("boom_state"))
                                else "shut"))
            sc = rd.get(OUT_SCORE) if rd is not None else None
            ready = (float(sc) / float(_thr_out_d)
                     if sc is not None and pd.notna(sc) and _thr_out_d
                     else None)
            if ready is not None:
                lines.append(f"deciding score (7d-smoothed mean of the "
                             f"factors): **{float(sc):.2f}** vs frozen "
                             f"trigger {_thr_out_d:.2f} -> "
                             f"**{ready:+.0%} of the way to CUT EXPOSURE**")
            return ready, lines
        # side == "in"
        r = _row_at(oi, d)
        if r is None:
            return None, ["no stored entry-side reading near this day "
                          "(the crowd was not building)"]
        lines.append(f"crowd size: **{float(r.get('hype_raw', float('nan'))):.1f}x** "
                     f"its own normal (entry floor "
                     f"{EUPHORIA_ONSET_HYPE_MIN:.2f}x)")
        for c in ("attention_accel", "hype_ratio", "bull_inflection",
                  "influx_speed", "attention_convexity"):
            lines.append(f"{PLAIN[c]}: **{_fmt_rank(r.get(c))}**")
        sc = rd.get(IN_SCORE) if rd is not None else None
        ready = (float(sc) / float(_thr_in_d)
                 if sc is not None and pd.notna(sc) and _thr_in_d else None)
        if ready is not None:
            lines.append(f"deciding score (7d-smoothed mean of the "
                         f"factors): **{float(sc):.2f}** vs frozen "
                         f"trigger {_thr_in_d:.2f} -> "
                         f"**{ready:+.0%} of the way to INCREASE EXPOSURE**")
        if rd is not None and bool(rd.get("end_stage", False)):
            lines.append("phase gate: name is END-STAGE - the entry "
                         "question is not asked here")
        return ready, lines

    def _explain_alert(name, d, side):
        """Markdown for one fired flag: the factor list plus the firing
        sentence. The dates come from the stored flags; the values are the
        stored evidence on that day."""
        ready, lines = _factor_lines(name, d, side)
        head = ("CUT EXPOSURE: expect a fall" if side == "out"
                else "INCREASE EXPOSURE: expect growth")
        md = [f"**{pd.Timestamp(d).date()} — {head}.**",
              "Every factor below is a percentile of this name's OWN "
              "trailing year (1.00 = the most extreme it has been); the "
              "signal is the 7d-smoothed mean of the factors crossing its "
              "frozen trigger while every gate is open."]
        md += [f"- {ln}" for ln in lines]
        return "\n".join(md)

    # ---- SIGNAL OUTCOME RECORD (recorded decision: "median time
    # after signal for price up / down by the X%" + "price change after
    # 5, 20, 84 days after each signal").  The X% move is the SAME test
    # NB06 uses for the danger state: a >=10%-in-7d move STARTING within
    # 30d of the signal (GAUGE_DROP / GAUGE_FWD / GAUGE_HORIZON there).
    # Down-moves are measured after GET OUT, up-moves after GET IN.
    # 5/20/84 are TRADING days (a week / a month / the project's baseline
    # window).  Alerts too new to judge are excluded, never counted
    # against the signal - same PENDING rule as the notebooks.
    def _outcome_stats(name, win_lo=None, win_hi=None):
        """Signal outcomes for one name, ALERTS CLIPPED TO THE SIDEBAR
        WINDOW (requirement: "make the performance metrics
        for each chart update according to the timeframe window").  The
        JUDGING always uses the full price history - an alert near the
        window edge is still judged on what actually followed it, the
        window only selects WHICH alerts are in the record.  Few alerts
        in a short window = a noisy median; n is always shown."""
        li, _oi, _di = _rows_for(name)
        if li is None or not len(li) or prices is None:
            return None
        sym_ = li["symbol"].iloc[0]
        pr = prices[prices["symbol"] == sym_].sort_values("date")
        if not len(pr):
            return None
        px_ = pr.set_index("date")["px_last"]
        px_ = px_[~px_.index.duplicated(keep="last")]
        # CALENDAR-DAILY series for the 10%-in-7d test - the SAME basis
        # notebook 04 judges on (pxd = asfreq("D").ffill() there).  The
        # 5/20/84 forward changes below stay on TRADING-day rows on
        # purpose (they are labelled "td"); the weekly-move test was
        # silently running on trading rows (~11 calendar days) and
        # overstating hits vs the record - aligned 2026-07-31.
        pxc = px_.asfreq("D").ffill()
        co_, ct_ = coherent.get(name, ([], []))
        if win_lo is not None:
            co_ = [d for d in co_
                   if d >= win_lo and (win_hi is None or d <= win_hi)]
            ct_ = [d for d in ct_
                   if d >= win_lo and (win_hi is None or d <= win_hi)]
        out = {}
        for side_, alerts_ in (("out", ct_), ("in", co_)):
            sgn = -1.0 if side_ == "out" else 1.0
            fwd_ext = (pxc.rolling(8).min() if side_ == "out"
                       else pxc.rolling(8).max()).shift(-7)
            week_move = (fwd_ext / pxc - 1) * sgn >= 0.10
            chg, waits, hits, judged = {5: [], 20: [], 84: []}, [], 0, 0
            for a in alerts_:
                pos = px_.index.searchsorted(pd.Timestamp(a))
                if pos >= len(px_):
                    continue
                p0 = float(px_.iloc[pos])
                for h in (5, 20, 84):
                    if pos + h < len(px_):
                        chg[h].append(float(px_.iloc[pos + h]) / p0 - 1)
                # the 10%-in-7d move: judgeable only with 30d of alert +
                # 7d of measurement window after it
                if pxc.index[-1] < (pd.Timestamp(a)
                                    + pd.Timedelta(days=37)):
                    continue
                judged += 1
                win = week_move.loc[pd.Timestamp(a):
                                    pd.Timestamp(a) + pd.Timedelta(days=30)]
                hit_days = win[win.fillna(False)]
                if len(hit_days):
                    hits += 1
                    waits.append((hit_days.index[0]
                                  - pd.Timestamp(a)).days)
            out[side_] = {
                "n": len(alerts_), "judged": judged, "hits": hits,
                "med_wait": (float(pd.Series(waits).median())
                             if waits else None),
                "chg": {h: (float(pd.Series(v).median()) if v else None)
                        for h, v in chg.items()},
            }
        return out

    # ---- THE SIGNAL, unmissable (design brief: "it should be
    # super clear: euphoria is ending (get out signal) or euphoria
    # starting (get in)") - one red banner, one green banner, nothing to
    # interpret. Sparse by design: empty = the radar working.
    out_now = sorted((n for n in ending
                      if _state_of(n, starting, ending) == "ENDING"),
                     key=ending.get, reverse=True)
    in_now = sorted((n for n in starting
                     if _state_of(n, starting, ending) == "STARTING"),
                    key=starting.get, reverse=True)
    # The live-signal banner is a FULL-TAB element. On the landing page
    # the ranked lists and the LAST FIRED section already lead with the
    # same names, so the banner repeated the page's own headline above
    # it.
    if out_now and mode == "full":
        st.error("**CUT EXPOSURE — crowd heat is ENDING:** "
                 + ";  ".join(f"{flag_label(n, kind)} — signal "
                              f"{ending[n].date()}, "
                              f"{int((latest_day - ending[n]).days)}d ago"
                              for n in out_now)
                 + ". Expect the top within ~a month of the signal.")
    if in_now and mode == "full":
        st.success("**INCREASE EXPOSURE — crowd heat is STARTING:** "
                   + ";  ".join(f"{flag_label(n, kind)} — signal "
                                f"{starting[n].date()}, "
                                f"{int((latest_day - starting[n]).days)}"
                                "d ago" for n in in_now)
                   + ". The crowd is arriving; the rally window is open.")
    if not out_now and not in_now and mode == "full":
        st.info(f"**No live signal among {kind_label.lower()} right "
                "now** - no crowd heat starting (consider) or ending (get "
                "out) in the last 21 days. Crowd heat is rare; an empty "
                "pane is the radar working.")

    # ---- WHY, PER FLAG (recorded decision: "make each get out /
    # get in flag have a drop down we can click and that shows the
    # reasons").  One expander per live flag, right under its banner, built
    # by the same helper as the hover - the two cannot disagree.
    if mode == "full":
        for n in out_now:
            with st.expander(f"why CUT EXPOSURE on "
                             f"{flag_label(n, kind)}?  "
                             f"(signal {ending[n].date()})"):
                st.markdown(_explain_alert(n, ending[n], "out"))
        for n in in_now:
            with st.expander(f"why INCREASE EXPOSURE on "
                             f"{flag_label(n, kind)}?  "
                             f"(signal {starting[n].date()})"):
                st.markdown(_explain_alert(n, starting[n], "in"))

    # ---- NO PERFORMANCE METRICS ON THIS PANEL (recorded decision).
    #
    # A window-adaptive scorecard used to sit here: hit rate, median lead,
    # false alarms and signal count, recomputed against the same judge the
    # research record uses, for whatever window the sidebar selected.  It
    # was accurate and it is deleted anyway, on the desk's instruction -
    # "remove the hit rate etc stuff, i will just keep this for the
    # notebooks only".
    #
    # The reasoning is the standing division of labour rather than a doubt
    # about the numbers.  This dashboard shows CONCLUSIONS; the notebooks
    # are the research record.  A scorecard that recomputes on a
    # user-chosen window is a research object wearing a dashboard's
    # clothes: a three-month window routinely leaves one or two scoreable
    # episodes, so the headline figure swings between 0% and 100% on a
    # sidebar drag, and the number a PM remembers is whichever window
    # happened to be open.  The walk-forward record does not move, and it
    # lives where it can be read with its confidence intervals attached -
    # the frozen nb07_performance_battery.json record (its notebook was
    # retired 2026-08-07).
    #
    # Nothing measured was lost: `classify_onset_alerts` / `classify_top_alerts`
    # are untouched and still drive the notebooks.  Recorded in DECISIONS.xlsx
    # ("4. Detector Design") and the parameter register (docs/RESEARCH_RECORD.md §7, Class 8).

    # FROZEN THRESHOLDS, and WHICH ONE THE CHART IS ALLOWED TO DRAW.
    #
    # This block is the fix for a real self-contradiction in the old chart
    # (defect report: "its quite unclear to see WHEN is the
    # actual change - its like flat and then suddenly a get out flag").
    #
    # The old panel drew ONE dotted line labelled "signal level" at the
    # level-detector's walk-forward threshold (85) and plotted the euphoria
    # LEVEL against it.  But when the desk store is present - which is the
    # normal case - the flags on screen are NOT produced by the level.  They
    # are produced by the desk score crossing its own frozen threshold.  The
    # two disagree constantly: MEASURED over all 95 GET OUT alerts in the
    # store, the plotted level was BELOW the drawn 85 line on 79 of them
    # (83%), median plotted level at a GET OUT 74.8.  So the chart showed a
    # curve sitting comfortably under the line it said mattered, and then a
    # flag appeared anyway.  That is not a legibility problem, it is the
    # chart quoting the wrong threshold.
    #
    # The rule now: draw the threshold that actually gated the flags being
    # drawn, and plot the series that actually crossed it.  `thr_now` (85)
    # survives ONLY for the fallback path, where the level really is the
    # decider.  No new number is introduced anywhere - both desk thresholds
    # are read straight out of euphoria_desk_report.json.
    #
    # 2026-07-28 UPDATE - the rescaling that used to live here is gone.  The
    # desk scores are 0-1 and the old panel multiplied both thresholds by 100
    # to share the euphoria level's 0-100 axis (`thr_out_100`, `thr_in_100`).
    # That kept ONE axis but still needed TWO dotted lines at two different
    # heights, because the two rules have different frozen thresholds.  The
    # panel now plots score/threshold*100 instead, which puts BOTH rules on a
    # single line at 100 and needs neither constant.  The thresholds are read
    # per-name inside draw_chart as `thr_out_d` / `thr_in_d`, still straight
    # out of euphoria_desk_report.json, still frozen, still unscaled.
    thr_now = None
    if euph_report and euph_report.get("thresholds"):
        thr_now = euph_report["thresholds"][
            max(euph_report["thresholds"])]

    ew = clip_window(ek, "date", lo, hi)

    # ---- ONE MASTER DIAL FOR THE WHOLE TAB -----------------------------
    # Requirement: all gauges share one master dial implementation.
    #
    # The scrubber shipped one pass earlier lived INSIDE draw_chart, so a
    # six-name page carried six independent sliders.  That is the wrong
    # object: the question a PM asks is "what did the book look like on
    # 3 March", not "what did NVDA look like on 3 March while uranium is
    # still showing today".  Six controls also means six chances to leave
    # one behind and read two different days side by side as if they were
    # the same moment - a comparison error the page itself would have
    # created.
    #
    # So there is now ONE control, owned by the tab and read by every
    # gauge, every state badge and every 7-day change on the page.  Each
    # chart still resolves the date against its OWN level curve (names
    # start and end on different days), but they all resolve the SAME
    # date, so the page is always a single point in time.
    # INFLECTION MARKERS NEED A REBUILT STORE. `inflection` / `inflection_score` arrived
    # 2026-08-12; a parquet written before that has neither, and the
    # panel's guard then draws nothing. Silent absence is
    # indistinguishable from "this name simply has no inflections", so say
    # which it is - once per page, not once per name.
    if (use_desk and dk is not None and len(dk)
            and "inflection" not in dk.columns):
        st.caption("Inflection markers are not in this data yet - the desk "
                   "store predates them. Run `python -m "
                   "analytics.run_analytics --what phases` (about 20 "
                   "seconds, no fetch needed) and reload.")

    master_day = None
    if len(ew):
        _mdi = pd.to_datetime(ew["date"]).dropna()
        if len(_mdi):
            _md0 = pd.Timestamp(_mdi.min()).to_pydatetime()
            _mdn = pd.Timestamp(_mdi.max()).to_pydatetime()
            if _md0 < _mdn:
                _msc, _ = st.columns([1.6, 2.4])
                with _msc:
                    master_day = st.slider(
                        "read every dial on", min_value=_md0,
                        max_value=_mdn, value=_mdn, format="DD MMM YY",
                        key=f"{key_prefix}_master_day",
                        help="One control for the whole page. Drag to move "
                             "EVERY dial, state badge and 7-day change on "
                             "this tab to the same day, so the names are "
                             "always compared at one moment. Each chart "
                             "marks the day with a vertical line and snaps "
                             "to its nearest reading at or before your "
                             "pick - never forward, so a weekend cannot "
                             "show you Monday's number. Leave it on the "
                             "right for today.")

    def draw_chart(name, title_prefix, key):
        # `name` stays the raw slug - it is the key every store is
        # filtered by. `_disp` is the only thing ever shown, so a theme
        # whose display name differs from its slug (see
        # plain_english._LABEL_OVERRIDES) is spelled the same here as in
        # the dropdown and the banners.
        _disp = theme_label(name) if kind == "theme" else str(name)
        one = ew[ew["name"] == name].sort_values("date")
        if not len(one):
            st.caption(f"{_disp}: no crowd heat data inside the selected "
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
        # Drawn as the PRICE LINE ITSELF turning amber on those days.
        # `boom_prog` (how far the price sits above its rolling low,
        # relative to the boom bar) is kept for the hover: it is the
        # boom gate's own "percentage of the way to its threshold".
        danger_days = None
        boom_prog = None
        if px is not None and not px.empty and "hype_ok" in one_i.columns \
                and prices is not None:
            full = prices[prices["symbol"] == sym].sort_values("date")
            pxa = full.set_index("date")["px_last"].asfreq("D").ffill()
            # THE SAME WINDOW THE LIVE GATE USES, read from the constant.
            low_w = pxa.rolling(EUPHORIA_BOOM_WINDOW_D,
                                 min_periods=EUPHORIA_BOOM_WINDOW_MIN_D
                                 ).min()
            bm = (EUPHORIA_BOOM_MIN_SINGLE if kind == "single"
                  else EUPHORIA_BOOM_MIN_ETF)
            run_up = (pxa / low_w - 1)
            boom_prog = (run_up / bm).reindex(one_i.index)
            boom = (run_up >= bm).reindex(one_i.index).eq(True)
            danger_days = one_i["hype_ok"].astype(bool) & boom
        lvl_raw = one_i["level"]
        # the DISPLAY curve is 7d-smoothed on the UNCLIPPED history, then
        # clipped (2026-07-29) - see the readiness construction below for
        # why the window edge must not move the reading.
        _lvl_hist = ek[ek["name"] == name].sort_values("date")
        if len(_lvl_hist):
            _ls = _lvl_hist.set_index("date")["level"]
            _ls = _ls[~_ls.index.duplicated(keep="last")]
            lvl = (_ls.rolling(ROLL, min_periods=1).mean()
                   .reindex(one_i.index))
        else:
            lvl = lvl_raw.rolling(ROLL, min_periods=1).mean()

        co, ct = coherent.get(name, ([], []))
        w0, w1 = one_i.index.min(), one_i.index.max()
        onset_alerts = [d for d in co if w0 <= d <= w1]
        top_alerts = [d for d in ct if w0 <= d <= w1]
        # INFLECTION MARKERS - CONTEXT, NOT A CALL. Drawn as
        # a small tick on the axis rather than a full-height rule, so it
        # can never be mistaken for GET IN / GET OUT at a glance. It is
        # not in the watchlist, it does not set the state, and nothing
        # downstream reads it. It fires on tops AND bottoms with no
        # direction, and its measured hit rate is only modestly above
        # the base rate - which is exactly why it is furniture and not a
        # signal. The numbers deliberately live in the RUNBOOK and
        # the parameter register (docs/RESEARCH_RECORD.md §7, Class 3)d rather than here: a hard-coded hit
        # rate in the UI goes stale the first time the head is re-fitted
        # and nothing would catch it.
        inflection_alerts = []
        if use_desk and dk is not None and "inflection" in dk.columns:
            _tg = dk[(dk["name"] == name) & dk["inflection"].eq(True)]
            inflection_alerts = [d for d in _tg["date"] if w0 <= d <= w1]
        state = _state_of(name, starting, ending)
        dk_i = (dk[dk["name"] == name].set_index("date").sort_index()
                if (use_desk and dk is not None) else None)
        thr_in_d = _thr_in_d
        thr_out_d = _thr_out_d

        # ---- HEADER, then DIAL + FACTS + RECORD on ONE ROW, then chart.
        _sig_all = sorted([(d, "INCREASE EXPOSURE", GETIN) for d in onset_alerts]
                          + [(d, "CUT EXPOSURE", BEAR) for d in top_alerts])
        _last_sig = (f"{_sig_all[-1][1]} · "
                     f"{pd.Timestamp(_sig_all[-1][0]).strftime('%d %b %y')}"
                     if _sig_all else "none in window")
        _last_col = _sig_all[-1][2] if _sig_all else INK_MUTED
        _badge = {"STARTING": (GETIN, "INCREASE EXPOSURE: attention building — expect growth"),
                  "ENDING": (BEAR, "CUT EXPOSURE: attention topping — expect a fall")}.get(
                      state, (INK_MUTED, "no live signal"))
        st.markdown(
            "<div style='display:flex;align-items:baseline;gap:12px;"
            "flex-wrap:wrap;margin:6px 0 2px 0'>"
            f"<span style='font-size:19px;font-weight:600;color:{INK}'>"
            f"{title_prefix}{_disp}</span>"
            f"<span style='font-size:13px;color:{INK_MUTED};"
            f"letter-spacing:.04em'>{sym}</span>"
            f"<span style='font-size:11px;font-weight:600;"
            f"letter-spacing:.06em;text-transform:uppercase;"
            f"color:{_badge[0]}'>{_badge[1]}</span></div>",
            unsafe_allow_html=True)
        # Technical agreement, on the panel itself, for whichever side
        # this name may actually fire - so the flag on a chart and the
        # flag on the list can never disagree.
        _hc_side = ("CUT EXPOSURE" if state == "ENDING"
                    else "INCREASE EXPOSURE" if state == "STARTING"
                    else ("CUT EXPOSURE"
                          if watch_gate(one.iloc[-1])
                          else "INCREASE EXPOSURE"))
        if _tech_confirms(sym, one["date"].max(), _hc_side):
            st.markdown(
                f"<span class='rf-hiconv' style='border-color:{TEAL};"
                f"color:{TEAL}'>◆ HIGH CONVICTION</span>",
                unsafe_allow_html=True)
            st.caption("Crowd signal and technical screen agree.",
                       help=HIGH_CONV_HELP)

        _z = gauge_zones()
        _lvl_ok = lvl.dropna()
        _have_dial = (_z.get("red_edge") is not None
                      and _z.get("amber_edge") is not None
                      and len(_lvl_ok) > 0)

        # resolve the tab's MASTER day against this name's own level curve
        # (nearest at or before, never forward - see the master dial above).
        _pos = len(_lvl_ok) - 1
        _as_of_click = None
        if _have_dial and len(_lvl_ok) > 1 and master_day is not None:
            _p = int(_lvl_ok.index.get_indexer(
                [pd.Timestamp(master_day).normalize()], method="ffill")[0])
            # a slider day BEFORE this name's first reading has no
            # at-or-before neighbour (-1); clamp to the FIRST reading and
            # say so via the as-of tag, never silently show the latest
            _pos = _p if _p >= 0 else 0
            if _pos != len(_lvl_ok) - 1:
                _as_of_click = _lvl_ok.index[_pos]


        # ---- THE PHASE CLOCK: one displayed state (NB08, 2026-07-31) -----
        # The design question: how could the panel read GET IN and GET OUT at
        # once, and why the shown side could flip overnight.  NB08 built
        # and judged two single-state designs; the PHASE CLOCK won: two
        # smooth coordinates - L (crowd extremity: the e-bank mean) and
        # M (arrival momentum: the onset-bank mean), both the house
        # 7d-smoothing - give an angle and an intensity, and every
        # episode traverses QUIET -> BUILDING -> BLOW-OFF -> TOPPING ->
        # COOLING around the circle.  One state per day by geometry
        # (co-firing 28 days -> 0), no fast flips (69 -> 0), nothing
        # suppressed - the blow-off is a NAMED region, not a
        # contradiction.  DISPLAY ONLY: NB08's pre-stated adoption rule
        # (beat the incumbent's flag utility both directions) was NOT
        # met - incumbent +12 vs clock -21 on GET OUT - so the frozen
        # rules keep firing every alert; this state is how the day is
        # DESCRIBED, never what fires.  Arc parameters come from the
        # notebook's frozen record, not from constants typed here.
        _clk = _research("nb08_single_state")
        _arc_in = _dig(_clk, "frozen_winner_params", "live", "get_in",
                       default=[80, 120, 0.25])
        _arc_out = _dig(_clk, "frozen_winner_params", "live", "get_out",
                        default=[-90, 20, 0.30])
        _oo_i = (_oo_all[_oo_all["name"] == name].set_index("date")
                 .sort_index() if _oo_all is not None else None)
        _on_mean = None
        if _oo_i is not None and len(_oo_i):
            _mc = [c for c in ("attention_accel", "hype_ratio",
                               "bull_inflection", "influx_speed",
                               "attention_convexity")
                   if c in _oo_i.columns]
            if _mc:
                _om = (_oo_i[_mc].apply(pd.to_numeric, errors="coerce")
                       .mean(axis=1))
                _om = _om[~_om.index.duplicated(keep="last")]
                # SMOOTHED OVER THE CALENDAR, NOT THE STORE SEQUENCE: a
                # day below the tracking floor is an arrival reading of
                # ~zero, so it enters the 7d mean as 0 and the momentum
                # DECAYS over a week when the crowd drops out - it cannot
                # snap to zero overnight, so the clock cannot jump arcs
                # on a tracking-floor blink; the state must not
                # oscillate rapidly.  No new window and
                # no look-ahead: this is the same house ROLL, applied on
                # the calendar the way NB08's dense frame applies it.
                _cal = pd.date_range(_om.index.min(), lvl.index.max(),
                                     freq="D")
                _on_mean = (_om.reindex(_cal).fillna(0.0)
                            .rolling(ROLL, min_periods=1).mean()
                            .reindex(lvl.index))
        _L_ser = lvl / 100.0

        def _phase_of(_li, _mi):
            """(label, colour, blurb) for one day's (L, M) reading.
            Unmeasured M = the crowd is below its own normal, which IS an
            arrival reading of ~zero - stated, not hidden."""
            import math as _math
            if pd.isna(_li):
                return None
            _mv = 0.0 if (_mi is None or pd.isna(_mi)) else float(_mi)
            _ang = _math.degrees(_math.atan2(_mv - 0.5, float(_li) - 0.5))
            _r = _math.hypot(_mv - 0.5, float(_li) - 0.5)
            _rmin = min(_arc_in[2], _arc_out[2])
            if _r < _rmin:
                return ("QUIET", INK_MUTED,
                        "no meaningful crowd state either way")
            if _arc_in[0] <= _ang <= _arc_in[1] and _r >= _arc_in[2]:
                return ("BUILDING", GETIN,
                        "crowd arriving, not yet extreme - the entry side "
                        "of the clock")
            if _arc_out[0] <= _ang <= _arc_out[1] and _r >= _arc_out[2]:
                return ("TOPPING", BEAR,
                        "crowd extreme, arrivals dying - the exit side of "
                        "the clock")
            if _arc_in[1] >= _ang > _arc_out[1]:
                return ("BLOW-OFF", OCHRE,
                        "crowd extreme AND still arriving fast - "
                        "late-stage; historically the rally often runs on "
                        "short-term while the tail risk builds")
            return ("COOLING", INK_MUTED,
                    "the crowd is fading; neither question is close")

        # ---- THE DISPLAYED STATE, full priority order (specified
        # 2026-07-31: "it has to be explainable when there is an alert" /
        # "only one clear direction at any time").
        #
        #   1. a GET OUT fired within the last 21 days  -> EXIT WINDOW
        #   2. else a GET IN fired within the last 21d  -> ENTRY WINDOW
        #   3. else the phase clock's geometric reading -> BUILDING /
        #      BLOW-OFF / TOPPING / COOLING / QUIET
        #
        # 21 days is the alert protocol's OWN cooldown/episode constant -
        # the same window the banner badge and the coherence engine
        # already use - so no new rule enters the system: the state simply
        # agrees with the flags it lives next to.  A fired flag OWNS the
        # narrative for its episode window; the clock resumes when the
        # window closes.  The flags themselves are untouched.
        _co_all, _ct_all = coherent.get(name, ([], []))

        def _disp_state(_day):
            _day = pd.Timestamp(_day)
            # STRICT <, ages 0-20 (design review #2: at exactly 21
            # days the banner said "no live signal" while this said "EXIT
            # WINDOW ... owns the state" - the badge, the coherence engine
            # and this resolver now share one boundary).
            _lo_ = [pd.Timestamp(x) for x in _ct_all
                    if 0 <= (_day - pd.Timestamp(x)).days
                    < EUPHORIA_COOLDOWN_DAYS_DISP]
            _li_ = [pd.Timestamp(x) for x in _co_all
                    if 0 <= (_day - pd.Timestamp(x)).days
                    < EUPHORIA_COOLDOWN_DAYS_DISP]
            _last_o = max(_lo_) if _lo_ else None
            _last_i = max(_li_) if _li_ else None
            if _last_o is not None and (_last_i is None
                                        or _last_o >= _last_i):
                _n_ = (_day - _last_o).days
                return ("EXIT WINDOW", BEAR,
                        f"CUT EXPOSURE fired {_n_}d ago - the top is expected "
                        "within ~a month of the signal; this flag owns "
                        "the state until the 21d episode window closes")
            if _last_i is not None:
                _n_ = (_day - _last_i).days
                return ("ENTRY WINDOW", GETIN,
                        f"INCREASE EXPOSURE fired {_n_}d ago - the rally window is "
                        "open; this flag owns the state until the 21d "
                        "episode window closes")
            return None

        # (_ph_now phase-state block removed - its panel is gone)

        # The after-signal outcome record was replaced by the
        # level-conditioned stats; its computation (two rolling scans
        # over the full price history, per chart, per rerun) ran on,
        # discarded - the single largest waste on the page.
        _log_scale = False
        if _have_dial:
            _now = float(_lvl_ok.iloc[_pos])
            _ref = float(_lvl_ok.iloc[_pos - ROLL]
                         if _pos >= ROLL else _lvl_ok.iloc[0])
            _dgr = bool(danger_days.reindex(_lvl_ok.index).iloc[_pos]) \
                if danger_days is not None else False
            _pk_v = float(_lvl_ok.max())
            _pk_d = _lvl_ok.idxmax()
            _zkey, _zlab, _zcol = gauge_state(_now, _dgr, _z)
            # READINESS MODE. Same dial, different
            # quantity: the live score as a percentage of its own frozen
            # trigger, so 100 is the firing line for every name and both
            # sides. The delta and the peak marker are recomputed in the
            # SAME units - a readiness needle over a euphoria peak would
            # be precisely the confusion this mode exists to remove.
            _ready = None
            if READINESS_DIAL and dk_i is not None and len(dk_i):
                # At the slider's right edge, use the FULL desk
                # history: the euphoria-levels store can end days before
                # the desk store, and capping by it made the dial read a
                # different day than the list above it (94 vs 91 on the
                # same screen). A dragged slider still caps as before.
                _upto = (dk_i if _as_of_click is None
                         else dk_i[dk_i.index <= _lvl_ok.index[_pos]])
                if len(_upto):
                    # gate and score from the SAME day - see
                    # eligible_scored_now. readiness_now then reads that
                    # row's own boom state and eligible-side score.
                    _es_u = eligible_scored_now(_upto)
                    if _es_u is not None:
                        _ready = readiness_now(_es_u[0].to_dict(),
                                               _thr_in_d, _thr_out_d)
            if _ready is not None:
                _rd_now, _rd_side, _rd_sc, _rd_cut = _ready
                _sc_col_r = (OUT_SCORE if _rd_side == "CUT EXPOSURE"
                             else IN_SCORE)
                _hist_r = pd.Series(dtype=float)
                if _sc_col_r in dk_i.columns:
                    _hist_r = (dk_i[_sc_col_r].dropna() / _rd_cut * 100.0)
                    _hist_r = _hist_r[_hist_r.index <= _lvl_ok.index[_pos]]
                _g_now = min(_rd_now, 130.0)
                _g_ref = (float(_hist_r.iloc[-1 - ROLL])
                          if len(_hist_r) > ROLL else
                          (float(_hist_r.iloc[0]) if len(_hist_r)
                           else _rd_now))
                _g_z = {"amber_edge": READY_AMBER, "red_edge": READY_RED}
                _g_pkv = min(float(_hist_r.max()), 130.0) if len(_hist_r) \
                    else _g_now
                _g_pkd = (_hist_r.idxmax() if len(_hist_r)
                          else _lvl_ok.index[_pos])
                # NAME THE SIDE. "91/100" with no side was how a dial
                # pointing at INCREASE sat over a chart whose recent
                # days were CUT-eligible and read as a contradiction.
                _facts_rows = [
                    (f"% of the way to {_rd_side.lower()}",
                     f"{_rd_now:.0f}<span style='font-size:17px;"
                     f"color:{INK_LABEL}'>/100</span>", None),
                ]
            else:
                _g_now, _g_ref, _g_z = _now, _ref, _z
                _g_pkv, _g_pkd = _pk_v, _pk_d
                _facts_rows = [
                    (f"change over {ROLL} days", f"{_now - _ref:+.0f}",
                     None),
                ]
            # TWO COLUMNS, not four. The gauge is tall; the facts
            # beside it are two lines each, so a four-column strip left
            # the whole right half empty and pushed the level record
            # into a narrow gutter below ("format it nicer ... in the
            # white space"). Now: gauge on the left, and everything
            # else stacked in ONE wide right-hand block that fills the
            # space the gauge's height creates.
            _gc, _rt = st.columns([1.0, 2.05])
            with _gc:
                _g_title = _g_band = _g_bcol = None
                if _ready is not None:
                    _g_title = ("% OF THE WAY TO A SIGNAL · "
                                + pd.Timestamp(_lvl_ok.index[_pos])
                                .strftime("%d %b %Y").upper())
                    _g_band = _rd_side
                    # TEAL, this palette's dark accent - TEAL_TEXT was a
                    # token from the reverted re-skin and does not exist
                    # here; referencing it crashed every INCREASE-side
                    # chart while the CUT branch sailed past the test.
                    _g_bcol = (BEAR if _rd_side == "CUT EXPOSURE"
                               else GETIN)
                st.plotly_chart(
                    fig_euphoria_gauge(_g_now, _g_ref,
                                       _dgr and _ready is None, _g_z,
                                       _lvl_ok.index[_pos],
                                       peak_val=_g_pkv, peak_day=_g_pkd,
                                       title_text=_g_title,
                                       band_text=_g_band,
                                       band_colour_o=_g_bcol),
                    width="stretch", key=f"{key}_gauge")
                # SAY WHEN. A dial on a past day looks exactly like a dial
                # on today, so the date is stated whenever the reading is
                # not the latest one.
                if _as_of_click is not None:
                    st.markdown(
                        f"<span style='font-size:11px;color:{ACCENT};"
                        "font-weight:600'>reading "
                        + pd.Timestamp(_as_of_click).strftime("%d %b %y")
                        + "</span>", unsafe_allow_html=True)
                # subtitle line removed on request ("remove this") -
                # the dial's own title already names the side and scale
            with _rt:
                _f1, _f2 = st.columns(2)
                with _f1:
                    st.markdown(_facts(_facts_rows),
                                unsafe_allow_html=True)
                with _f2:
                    # Trimmed to the one fact a reader acts on. The
                    # state words (calm/quiet), the window peak and the
                    # in/out counts were removed by request - they
                    # described the chart the reader is already
                    # looking at.
                    st.markdown(_facts([
                        ("last signal", _last_sig, _last_col),
                    ]), unsafe_allow_html=True)
                # CONDITIONED ON TODAY, not on past signals. The old
                # block showed the median move after this name's own
                # fired signals - n=1 on most names, a story rather than
                # a statistic. This one asks the question a PM actually
                # has: the other times crowd heat sat where it sits
                # TODAY, what did price do next? Same-name days first;
                # pooled across all themes when the name alone is too
                # thin, and labelled when it is.
                _cond, _pooled = level_conditioned_stats(name, _now)
                if _cond:
                    # A TABLE, not three wrapped sentences in a gutter.
                    # One row per horizon so the eye compares DOWN a
                    # column (median vs its baseline vs the downside)
                    # instead of parsing "+0.7% / +2.4% / +8.7%" and
                    # mentally aligning it with two more triples.
                    _HN = {5: "1 week", 20: "1 month", 84: "4 months"}
                    _cn = min(r[2] for r in _cond)
                    _hd = ("<tr>"
                           + "".join(
                               f"<th style='text-align:{a};padding:"
                               f"3px 10px 6px 0;font-size:10px;"
                               f"letter-spacing:.08em;text-transform:"
                               f"uppercase;color:{INK_LABEL};"
                               f"font-weight:600;width:{w}'>{t}</th>"
                               for t, a, w in (
                                   ("", "left", "20%"),
                                   ("median move", "right", "18%"),
                                   ("vs a day unlike today",
                                    "right", "26%"),
                                   ("ended lower", "right", "18%"),
                                   ("worst tenth", "right", "18%")))
                           + "</tr>")
                    _bd = ""
                    for _h, _v, _n, _b, _pl, _p10 in _cond:
                        _dif = (None if _b is None else (_v - _b) * 100)
                        _dc = (INK_LABEL if _dif is None or abs(_dif) < 0.5
                               else (BULL if _dif > 0 else BEAR))
                        _ds = ("-" if _dif is None
                               else ("about the same" if abs(_dif) < 0.5
                                     else f"{_dif:+.1f}pp"))
                        _bd += (
                            "<tr>"
                            f"<td style='padding:4px 14px 4px 0;"
                            f"color:{INK_LABEL};font-size:13px'>"
                            f"{_HN.get(_h, str(_h))}"
                            f"<span style='color:{INK_MUTED};"
                            f"font-size:11px'> · {_h}td</span></td>"
                            f"<td style='text-align:right;padding:"
                            f"4px 14px;font-size:17px;font-weight:600;"
                            f"color:{INK}'>{_v:+.1%}</td>"
                            f"<td style='text-align:right;padding:"
                            f"4px 14px;font-size:15px;font-weight:600;"
                            f"color:{_dc}'>{_ds}</td>"
                            f"<td style='text-align:right;padding:"
                            f"4px 14px;font-size:15px;color:{INK}'>"
                            f"{_pl:.0%}</td>"
                            f"<td style='text-align:right;padding:"
                            f"4px 0;font-size:15px;color:{BEAR}'>"
                            f"{_p10:+.1%}</td></tr>")
                    st.markdown(
                        f"<div style='font-size:10px;letter-spacing:"
                        f".09em;text-transform:uppercase;"
                        f"color:{INK_LABEL};margin-bottom:2px'>"
                        "what followed, the other times attention sat "
                        f"where it sits today</div>"
                        "<table style='border-collapse:collapse;"
                        "width:100%;table-layout:fixed;"
                        f"margin:0 0 4px 0'>{_hd}{_bd}</table>"
                        f"<div style='font-size:11px;color:{INK_MUTED};"
                        "margin-bottom:6px'>"
                        f"n≥{_cn} such days · "
                        + ("all themes pooled" if _pooled
                           else "this name only")
                        + " · attention level, not the signal score"
                        "</div>", unsafe_allow_html=True)
                else:
                    st.markdown(_facts([
                        ("what followed at this attention level",
                         "no comparable days on record", None)]),
                        unsafe_allow_html=True)
                st.markdown(
                    f"<span style='font-size:11px;color:{INK_MUTED}'>"
                    "what this record means</span>",
                    unsafe_allow_html=True,
                    help="Every past day this name's attention level sat "
                         f"within ±10 of today's {_now:.0f} was collected, "
                         "and the median price change 5, 20 and 84 "
                         "TRADING days after those days is shown. When "
                         "this name alone has fewer than 10 such days, "
                         "the same attention band is pooled across every "
                         "tracked theme and the row says so. It is a "
                         "conditional base rate, not a forecast: it says "
                         "what usually followed this crowd state, over "
                         "the history this project holds.\n\n"
                         "**Read the second column, not the first.** "
                         "Equities drift up, so almost any forward "
                         "median is positive - on its own the top line "
                         "mostly measures that drift. *vs its own "
                         "normal day* subtracts the median of this "
                         "name's days OUTSIDE the band - the days "
                         "unlike today - over the same horizons, "
                         "so the two figures differ in exactly one "
                         "thing: the attention level. Negative means "
                         "days like today did WORSE than an ordinary "
                         "day.\n\n"
                         "**And a median is not the risk.** A reduce-"
                         "exposure call is about the left tail getting "
                         "fatter, which a median hides: across themes "
                         "at high attention the median is positive "
                         "while roughly 2 days in 5 end lower and the "
                         "worst tenth is far into the red. The last "
                         "two columns state that directly.\n\n"
                         "Note this row conditions on the ATTENTION "
                         "LEVEL alone, while the signal beside it "
                         "conditions on eleven measurements plus the "
                         "run-up gate - so the two are answering "
                         "different questions and need not agree.")
                _log_scale = st.toggle(
                    "log price scale", key=f"{key}_log",
                    help="Plot the price on a logarithmic axis, so equal "
                         "PERCENTAGE moves take equal vertical space and "
                         "a long run-up does not flatten the early "
                         "history. Linear is the default; the crowd heat "
                         "signals are identical either way.")
        else:
            _log_scale = st.toggle(
                "log price scale", key=f"{key}_log",
                help="Plot the price on a logarithmic axis - equal "
                     "percentage moves take equal vertical space.")

        # ---- READINESS, for the HOVER (design brief: "if you
        # hover over the chart it shows all the factors that contribute to
        # a signal ... and the percentage of the way to their threshold ...
        # and a bar towards -100 to 100% of the signal firing").
        #
        # The -1..+1 panel that used to sit under the price is REMOVED on
        # the same instruction ("remove the euphoria chart below the price
        # chart").  Its arithmetic is NOT removed: the readiness series
        # below are the same `stored score / frozen threshold` construction
        # that panel drew, priority-merged with the production scorers'
        # recomputation so the stored value wins on every day the detector
        # actually judged (the full reasoning, and the two rejected
        # drawings, are recorded in DECISIONS.xlsx and docs/RESEARCH_RECORD.md
        # 6.13-6.16).  The numbers now live in the hover instead of on an
        # axis: same crossings, same eligibility, zero panel height.
        _ready = []
        if dk_i is not None:
            for _col, _thr, _lab in ((OUT_SCORE, thr_out_d, "CUT EXPOSURE"),
                                     (IN_SCORE, thr_in_d, "INCREASE EXPOSURE")):
                if _col not in dk_i.columns or not _thr:
                    continue
                _s = (pd.to_numeric(dk_i[_col], errors="coerce")
                      .reindex(lvl.index) / float(_thr) * 100.0)
                if _s.notna().any():
                    _ready.append((_lab, _s))
        # A name the desk store does not COVER must not inherit the
        # level fallback (design review #1: europe_defense rendered
        # "+107% of trigger · could fire today" from the retired level
        # threshold while its factor lines divided by the desk trigger -
        # two arithmetics in one panel).  The fallback exists for a
        # machine with NO desk store at all, nothing else.
        _no_desk = use_desk and (dk_i is None or not len(dk_i))
        if not _ready and thr_now and not use_desk:
            # FALLBACK: with no desk store the euphoria level really is
            # the decider, so the identical construction applies to it.
            _ready.append(("SIGNAL", lvl / float(thr_now) * 100.0))
        _merged = []
        # the rules-score EXTENSION below back-fills days the desk store
        # does not cover by recomputing the RULES arithmetic live. That
        # is only coherent when the rules ARE the shipped model - under
        # the learned model (2026-08-07) a rules score divided by the
        # model's probability cut would be two arithmetics in one panel
        # (the exact defect review 2026-08-02 #1 removed), so the
        # extension is disabled and uncovered days simply show nothing.
        _model_live = (desk_report or {}).get("model", "rules")
        for _lab, _s in _ready:
            _spec = {"CUT EXPOSURE": ("levels", desk_end_fit, TOP_FEATURES,
                                 thr_out_d),
                     "INCREASE EXPOSURE": ("onset", desk_onset_fit, ONSET_BANK,
                                thr_in_d)}.get(_lab)
            if _model_live != "rules":
                _spec = None
            _full = _s
            if _spec is not None:
                _src, _fit, _feats, _thr = _spec
                # the UNCLIPPED frame, deliberately: the trailing ROLL-day
                # mean must not shift when the sidebar window moves.
                _base = ek if _src == "levels" else ok
                if (_thr and _base is not None and len(_base)
                        and set(_feats).issubset(_base.columns)):
                    _w = _base[_base["name"] == name].sort_values("date")
                    if len(_w):
                        _tr = pd.Series(_fit(None, _w, _feats),
                                        index=_w["date"].values)
                        _tr = _tr[~_tr.index.duplicated(keep="last")]
                        _tr = _tr.reindex(lvl.index) / float(_thr) * 100.0
                        # stored wins wherever it exists; no interpolation.
                        _full = _s.combine_first(_tr)
            _merged.append((_lab, _s, _full))

        _idx = lvl.index
        _nan = pd.Series(float("nan"), index=_idx, dtype="float64")

        def _side(_want):
            for _lab, _s, _full in _merged:
                if _lab == _want:
                    return (pd.to_numeric(_s.reindex(_idx),
                                          errors="coerce"),
                            pd.to_numeric(_full.reindex(_idx),
                                          errors="coerce"))
            return (_nan.copy(), _nan.copy())

        _out_s, _out_f = _side("CUT EXPOSURE")
        if _out_f.isna().all():
            _out_s, _out_f = _side("SIGNAL")
        _in_s, _in_f = _side("INCREASE EXPOSURE")

        # which question is LIVE on each day: `end_stage` opens the exit
        # question and closes the entry one.  The hover shows BOTH rules
        # every day (GET OUT % is the default view) and marks the
        # live one - the two banks are different machines and seeing them
        # side by side is what makes a GET OUT distinguishable from a
        # GET IN.
        if dk_i is not None and "end_stage" in dk_i.columns:
            _es = (dk_i["end_stage"].astype(bool)
                   .reindex(_idx, fill_value=False))
        else:
            _es = pd.Series(False, index=_idx)

        # ---- THE HOVER TEXT, one block per day ---------------------------
        # Specification: the total bar at the top stays; every
        # factor below it gets its own box and its % of the trigger, so
        # the COMPOSITION is visible: each rule's score is the plain
        # average of its five factors, so one factor at value v
        # contributes v/5 to the raw score = v/(5 x trigger) of the way
        # to firing, and the five contributions SUM to the raw reading.
        # (The total bar is the 7d-SMOOTHED deciding score - the number
        # that actually fires - so it can differ a little from today's
        # raw sum; both are shown, labelled.)  Gates carry their reasons
        # in numbers, not just open/shut.  Exact stored rows only; a
        # weekend carries the last stored reading.
        def _rx(frame, col):
            if frame is None or col not in frame.columns:
                return _nan
            s = pd.to_numeric(frame[col], errors="coerce")
            s = s[~s.index.duplicated(keep="last")]
            return s.reindex(_idx)

        _h_e = {c: _rx(one_i, c) for c in ("e1", "e2", "e3", "e5")}
        _h_fade = _rx(one_i, "fade")
        _h_hok = _rx(one_i, "hype_ok")
        _h_on = {c: _rx(_oo_i, c) for c in
                 ("hype_raw", "attention_accel", "hype_ratio",
                  "bull_inflection", "influx_speed",
                  "attention_convexity")}
        _h_hyp_dk = _rx(dk_i, "hype_raw")
        _h_boom = (boom_prog if boom_prog is not None else _nan)
        _bm_lbl = (EUPHORIA_BOOM_MIN_SINGLE if kind == "single"
                   else EUPHORIA_BOOM_MIN_ETF)

        def _tbar(pct, colour):
            """the TOTAL box: 10 blocks of the way to that rule's
            trigger (>=100% = at/beyond it)."""
            _m = max(0.0, min(abs(pct), 100.0))
            _n_ = int(round(_m / 10))
            return (f"<span style='color:{colour}'>"
                    + "▰" * _n_ + "▱" * (10 - _n_)
                    + f" {pct:+.0f}% of trigger</span>")

        def _fbar(v):
            """a FACTOR's box: 5 blocks of its 0-1 percentile."""
            if pd.isna(v):
                return "▱▱▱▱▱"
            _n_ = max(0, min(5, int(round(float(v) * 5))))
            return "▮" * _n_ + "▱" * (5 - _n_)

        def _fline(v, label, thr):
            """one factor row: box, value, plain-English name, and its
            share of the trigger (value/5 of the raw score).  With no
            frozen trigger on disk the share clause is OMITTED, never
            printed as a false 0% (review 2026-08-02 #6)."""
            if pd.isna(v):
                return f"▱▱▱▱▱ n/a · {label}"
            if not thr:
                return f"{_fbar(v)} {float(v):.2f} · {label}"
            _c = float(v) / (5.0 * float(thr)) * 100.0
            return (f"{_fbar(v)} {float(v):.2f} · {label} → "
                    f"{_c:.0f}% of trigger")

        _out_bank = (("e1", _h_e["e1"]), ("e2", _h_e["e2"]),
                     ("e3", _h_e["e3"]), ("e5", _h_e["e5"]))
        _in_bank = (("attention_accel", _h_on["attention_accel"]),
                    ("hype_ratio", _h_on["hype_ratio"]),
                    ("bull_inflection", _h_on["bull_inflection"]),
                    ("influx_speed", _h_on["influx_speed"]),
                    ("attention_convexity",
                     _h_on["attention_convexity"]))

        # ONE SIDE PER DAY (recorded decision: "I dont want it
        # to be like both get in and get out ... how can it be both?").
        # The PHASE picks which rule's breakdown is shown - BUILDING is
        # the start of the bullishness, so it shows GET IN; BLOW-OFF and
        # TOPPING are the late stage / the peak, so they show GET OUT;
        # QUIET and COOLING show neither, just one muted summary line.
        # Because the phase is continuous and exclusive (NB08), the shown
        # side cannot contradict itself or flip overnight.  Both scores
        # are still COMPUTED every day - the flags come from the frozen
        # rules exactly as before; this chooses only what is DISPLAYED.
        _fired_out_days = {pd.Timestamp(d) for d in top_alerts}
        _fired_in_days = {pd.Timestamp(d) for d in onset_alerts}
        # ONE builder, two consumers (recorded decision: "hover and
        # then see the reasons outside the graph? right now its covering
        # the price chart"): compact (_full=False) feeds the HOVER - a
        # three-line glance that no longer buries the price - and full
        # (_full=True) feeds the WHY panel rendered UNDER the chart.  Same
        # arithmetic, same wording, so the two can never disagree.
        def _day_lines(_i2, _full=False):
            _d = _idx[_i2]
            _dl = pd.Timestamp(_d).strftime("%d %b %y")
            _es_i = bool(_es.iloc[_i2])
            parts = [f"<b>{_dl}</b>"]
            if _d in _fired_out_days:
                parts.append(f"<span style='color:{BEAR}'><b>★ CUT EXPOSURE "
                             "FIRED today</b></span>")
            elif _d in _fired_in_days:
                parts.append(f"<span style='color:{GETIN}'><b>★ INCREASE EXPOSURE "
                             "FIRED today</b></span>")
            if _no_desk and _on_mean is None:
                # outside the detector's universe: the clock would read
                # "TOPPING - arrivals dying" off a defaulted M=0 two lines
                # above real factor values (review 2026-08-02) - say the
                # truth instead.
                _ph = ("UNTRACKED", INK_MUTED,
                       "the desk detector does not cover this name (too "
                       "little crowd data) - no flag can fire here")
            else:
                _ph = _disp_state(_d) or _phase_of(
                    _L_ser.iloc[_i2],
                    _on_mean.iloc[_i2] if _on_mean is not None else None)
            if _ph is not None:
                parts.append(
                    f"<span style='color:{_ph[1]}'><b>state: "
                    f"{_ph[0]}</b></span> - {_ph[2]}")
            _lbl = _ph[0] if _ph is not None else "TOPPING"

            def _out_block(blowoff=False):
                _ev = [float(s.iloc[_i2]) if pd.notna(s.iloc[_i2])
                       else float("nan") for _, s in _out_bank]
                _fd = _h_fade.iloc[_i2]
                _fdv = 1.0 if bool(_fd) and pd.notna(_fd) else 0.0
                _tot_o = _out_f.iloc[_i2]
                if pd.isna(_tot_o) and all(pd.isna(v) for v in _ev):
                    return
                # BLOW-OFF is NOT the peak: the crowd is still arriving,
                # so its header is amber "watching", never the red exit
                # call - a building<->blow-off wobble must read as
                # ESCALATION, not as the opposite signal (measured: all
                # 44 adjacent-day side wobbles in the store were this
                # boundary; direct BUILDING->TOPPING swaps are zero).
                if blowoff:
                    _head = (f"<span style='color:{OCHRE}'><b>watching "
                             "the CUT EXPOSURE rule (late-stage - not yet "
                             "the peak)</b></span> ")
                    _head += _tbar(float(_tot_o) if pd.notna(_tot_o)
                                   else 0.0, OCHRE)
                else:
                    _head = (f"<span style='color:{BEAR}'><b>CUT EXPOSURE "
                             "(exit)</b></span> ")
                    _head += _tbar(float(_tot_o) if pd.notna(_tot_o)
                                   else 0.0, BEAR)
                if pd.notna(_out_s.iloc[_i2]):
                    _head += " · could fire today"
                parts.append(_head)
                if _full:
                    for (_c2, _), _v2 in zip(_out_bank, _ev):
                        parts.append(_fline(_v2, PLAIN[_c2], thr_out_d))
                    parts.append(_fline(_fdv, PLAIN["fade"], thr_out_d)
                                 .replace(f"{_fdv:.2f}",
                                          "yes " if _fdv else "no  "))
                    if thr_out_d:
                        _raw_o = ([v for v in _ev if pd.notna(v)]
                                  + [_fdv])
                        _raw_pct = (sum(_raw_o) / 5.0
                                    / float(thr_out_d) * 100.0)
                        parts.append(f"today's raw factor sum: "
                                     f"{_raw_pct:.0f}% of trigger (bar "
                                     "above = the 7d-smoothed score "
                                     "that actually fires)")
                _g = []
                _e1v = _ev[0]
                if pd.notna(_e1v):
                    _g.append(("attention " if _e1v >= EUPHORIA_ATT_GATE
                               else "attention LOW ")
                              + f"{_e1v:.2f} vs {EUPHORIA_ATT_GATE:.2f} "
                                "needed")
                _hyp = _h_hyp_dk.iloc[_i2]
                if pd.notna(_hyp):
                    _g.append(f"crowd {float(_hyp):.1f}x vs "
                              f"{EUPHORIA_HYPE_MULT:.1f}x needed "
                              f"({float(_hyp) / EUPHORIA_HYPE_MULT:.0%})")
                else:
                    _hk = _h_hok.iloc[_i2]
                    _g.append("crowd 2x bar: "
                              + ("MET" if bool(_hk) and pd.notna(_hk)
                                 else "not met"))
                _bp = _h_boom.iloc[_i2] if _h_boom is not None else None
                if _bp is not None and pd.notna(_bp):
                    _g.append(f"price +{float(_bp) * _bm_lbl:.0%} off "
                              f"its low vs +{_bm_lbl:.0%} needed "
                              f"({float(_bp):.0%})")
                _e2v = _ev[1]
                _all_open = (pd.notna(_out_s.iloc[_i2])
                             and pd.notna(_e1v)
                             and _e1v >= EUPHORIA_ATT_GATE
                             and pd.notna(_e2v) and _e2v > 0)
                parts.append(("gates OPEN - " if _all_open
                              else "gates SHUT - ") + " · ".join(_g))

            def _in_block():
                _on_row = any(pd.notna(s.iloc[_i2]) for _, s in _in_bank)
                if not _on_row:
                    parts.append(
                        f"<span style='color:{GETIN}'><b>INCREASE EXPOSURE (entry)"
                        "</b></span> ▱▱▱▱▱▱▱▱▱▱ 0% - crowd below its "
                        "own normal (&lt;1.0x); entry tracking starts at "
                        f"{EUPHORIA_ONSET_HYPE_MIN:.2f}x")
                    return
                _tot_i = _in_f.iloc[_i2]
                _head = (f"<span style='color:{GETIN}'><b>INCREASE EXPOSURE "
                         "(entry)</b></span> ")
                _head += _tbar(-(float(_tot_i) if pd.notna(_tot_i)
                                 else 0.0), GETIN)
                if pd.notna(_in_s.iloc[_i2]) and not _es_i:
                    _head += " · could fire today"
                parts.append(_head)
                if _full:
                    for _c2, _s2 in _in_bank:
                        parts.append(_fline(_s2.iloc[_i2], PLAIN[_c2],
                                            thr_in_d))
                _hyp = _h_on["hype_raw"].iloc[_i2]
                _g = []
                if pd.notna(_hyp):
                    _g.append(f"crowd {float(_hyp):.1f}x its normal vs "
                              f"{EUPHORIA_ONSET_HYPE_MIN:.2f}x floor "
                              f"({float(_hyp) / EUPHORIA_ONSET_HYPE_MIN:.0%})")
                _all_open = pd.notna(_in_s.iloc[_i2]) and not _es_i
                parts.append(("gates OPEN - " if _all_open
                              else "gates SHUT - ") + " · ".join(_g)
                             if _g else
                             ("gates OPEN" if _all_open else "gates SHUT"))

            if _lbl in ("BUILDING", "ENTRY WINDOW"):
                _in_block()
            elif _lbl == "BLOW-OFF":
                _out_block(blowoff=True)
            elif _lbl in ("TOPPING", "EXIT WINDOW"):
                _out_block()
            else:
                # QUIET / COOLING: neither breakdown - one muted line so
                # the reader still knows how far away both rules sit.
                _o_v = float(_out_f.iloc[_i2]) \
                    if pd.notna(_out_f.iloc[_i2]) else 0.0
                _i_v = float(_in_f.iloc[_i2]) \
                    if pd.notna(_in_f.iloc[_i2]) else 0.0
                # the wording must follow the numbers ("neither close"
                # beside a 71% reading was a contradiction), and a
                # gate-zeroed score must SAY it is gate-zeroed: the GET
                # OUT score is held at 0 while its gates are shut even
                # when the raw factors are elevated, and that is exactly
                # the day a reader asks "how did this fire from 0%?" -
                # the answer (the gates opened) belongs on screen.
                _ev0 = [float(s.iloc[_i2]) if pd.notna(s.iloc[_i2])
                        else float("nan") for _, s in _out_bank]
                _fd0 = _h_fade.iloc[_i2]
                _raw0 = ([v for v in _ev0 if pd.notna(v)]
                         + [1.0 if bool(_fd0) and pd.notna(_fd0)
                            else 0.0])
                _raw_p = (sum(_raw0) / 5.0 / float(thr_out_d) * 100.0
                          if thr_out_d and _raw0 else 0.0)
                _o_txt = f"CUT EXPOSURE {_o_v:+.0f}%"
                if _raw_p - _o_v > 10:
                    _o_txt += (f" (score held at 0 by shut gates; raw "
                               f"factors {_raw_p:.0f}%)")
                _mx = max(abs(_o_v), abs(_i_v))
                if _mx < 50:
                    _lead = "neither rule close"
                else:
                    _near = ("CUT EXPOSURE" if abs(_o_v) >= abs(_i_v)
                             else "INCREASE EXPOSURE")
                    _lead = f"no flag live · {_near} is nearest"
                parts.append(
                    f"<span style='color:{INK_MUTED}'>{_lead} - "
                    f"{_o_txt} · INCREASE EXPOSURE {-_i_v:+.0f}% of their "
                    "triggers</span>")
            return parts

        _hover_txt = ["<br>".join(_day_lines(_j))
                      for _j in range(len(_idx))]

        # ---- ONE PANEL: THE PRICE (design brief: "remove the
        # euphoria chart below the price chart").  Everything the lower
        # panel said is now in the hover above; the flags stay as vertical
        # rules + dots on the price line itself.
        fig = go.Figure()
        _px_ok = px is not None and not px.empty
        if _px_ok:
            _carrier = px
            fig.add_trace(go.Scatter(x=px.index, y=px.values,
                                     name=f"{sym} price",
                                     line=dict(color=SLATE, width=1.5),
                                     hovertemplate=(
                                         "%{y:.2f}<extra></extra>")))
            if danger_days is not None and bool(danger_days.any()):
                hot = px.where(danger_days.reindex(px.index).eq(True))
                fig.add_trace(go.Scatter(
                    x=hot.index, y=hot.values, mode="lines",
                    name="crowded AND already run up (risk zone)",
                    connectgaps=False,
                    line=dict(color=OCHRE, width=2.6),
                    hovertemplate="risk zone: %{y:.2f}<extra></extra>"))
        else:
            # no price data: the smoothed euphoria level carries the panel
            # so the hover still has a line to ride on.
            _carrier = lvl
            fig.add_trace(go.Scatter(x=lvl.index, y=lvl.values,
                                     name="crowd heat level (no price data)",
                                     line=dict(color=SLATE, width=1.5),
                                     hovertemplate=(
                                         "level %{y:.0f}<extra></extra>")))

        # the INVISIBLE hover carrier: rides the drawn line so the cursor
        # finds it anywhere along the chart, and speaks the factor block.
        # Values snap to the nearest euphoria day AT OR BEFORE the cursor
        # (ffill) - a weekend shows Friday's stored reading, never
        # Monday's.
        _hv = pd.Series(_hover_txt, index=_idx)
        _hv = _hv.reindex(_carrier.index, method="ffill")
        _hv_missing = _hv.isna()
        if bool(_hv_missing.any()):
            _hv = _hv.fillna("no crowd heat reading yet")
        fig.add_trace(go.Scatter(
            x=_carrier.index, y=_carrier.values, mode="lines",
            line=dict(width=0.5, color="rgba(0,0,0,0)"),
            name="signal detail", showlegend=False,
            text=_hv.values,
            hovertemplate="%{text}<extra></extra>"))

        def _ms(ts):
            # plotly's vline+annotation midpoint maths does Timestamp+int
            # arithmetic on some plotly/pandas versions and crashes;
            # epoch-milliseconds is numeric and works on every version
            return pd.Timestamp(ts).value / 1_000_000

        # THE MOMENT OF FIRING, ON THE LINE ITSELF: a dot at the price on
        # the alert day.  The dates come from the STORED flags, never from
        # the hover arithmetic, so a dot cannot drift off its alert.
        _fx, _fy, _fc, _ft = [], [], [], []
        for _d, _nm, _c in ([(d, "CUT EXPOSURE", BEAR) for d in top_alerts]
                            + [(d, "INCREASE EXPOSURE", GETIN) for d in onset_alerts]):
            _t = pd.Timestamp(_d)
            _cpos = _carrier.index.searchsorted(_t)
            if _cpos < len(_carrier) and pd.notna(_carrier.iloc[_cpos]):
                _fx.append(_carrier.index[_cpos])
                _fy.append(float(_carrier.iloc[_cpos]))
                _fc.append(_c)
                _ft.append(f"{_nm} FIRED "
                           f"{pd.Timestamp(_d).strftime('%d %b %y')}")
        if _fx:
            fig.add_trace(go.Scatter(
                x=_fx, y=_fy, mode="markers",
                marker=dict(size=10, symbol="circle", color=_fc,
                            line=dict(color=WHITE, width=1.6)),
                name="signal fired", text=_ft,
                hovertemplate="%{text}<extra></extra>"))

        # WHERE THE DIAL IS READING - drawn only when the master slider is
        # off the latest day.  A cursor, not a signal.
        if _have_dial and _as_of_click is not None:
            fig.add_vline(x=_ms(_as_of_click), line_color=INK_MUTED,
                          line_width=1.2, line_dash="dot", opacity=0.85)

        # INFLECTION (context). ON THE LINE, exactly like the fired dots.
        # It used to be parked at `level.min()`, which is a EUPHORIA
        # value (0-100) placed on an axis that is usually PRICE - so
        # every diamond landed near y=0, detached from the series,
        # looking like a broken glyph on the axis. `_carrier` is
        # whatever line is actually drawn (price when there is price,
        # the level curve when there is not), so this cannot go wrong
        # again when the axis changes underneath it.
        if inflection_alerts:
            _tx, _ty, _tt = [], [], []
            # NB: scratch index variable is NOT named _pos - that name
            # carries the master-day position into _lvl_ok, and
            # clobbering it here sent the WHY panel indexing past the
            # end of that shorter series (IndexError, defect report
            # 2026-08-17)
            for _d in inflection_alerts:
                _ipos = _carrier.index.searchsorted(pd.Timestamp(_d))
                if _ipos < len(_carrier) and pd.notna(_carrier.iloc[_ipos]):
                    _tx.append(_carrier.index[_ipos])
                    _ty.append(float(_carrier.iloc[_ipos]))
                    _tt.append(
                        f"possible INFLECTION {pd.Timestamp(_d):%d %b %y}<br>"
                        f"A reversal is more likely than usual in the "
                        f"next two weeks.<br>"
                        f"Direction NOT implied - this fires at tops "
                        f"AND bottoms alike.<br><br>"
                        f"CONTEXT means: do not act on this on its "
                        f"own.<br>It is not sized, it is not in the "
                        f"watchlist, and nothing<br>else on this page "
                        f"changes because of it. Use it to<br>WEIGH a "
                        f"call you already have - a CUT EXPOSURE with a<br>"
                        f"inflection beside it is a more interesting CUT EXPOSURE "
                        f"than<br>one without.")
            if _tx:
                fig.add_trace(go.Scatter(
                    x=_tx, y=_ty, mode="markers",
                    marker=dict(size=9, symbol="diamond-open",
                                color=INK_MUTED,
                                line=dict(color=INK_MUTED, width=2)),
                    name="possible inflection (context)", text=_tt,
                    hovertemplate="%{text}<extra></extra>"))
        for d in onset_alerts:                       # GET IN
            fig.add_vline(x=_ms(d), line_color=GETIN, line_width=1.6,
                          opacity=0.9)
        for d in top_alerts:                         # GET OUT
            fig.add_vline(x=_ms(d), line_color=BEAR, line_width=1.6,
                          opacity=0.9)
        # Labels in a SEPARATE pass, on STACKED ROWS (greedy first-fit -
        # see the 2026-07-29 note: alternating two heights collides at
        # i and i+2 on long windows).
        # Each mark carries whether the technical screen agreed on the
        # day it fired, so a HIGH CONVICTION signal is identifiable on
        # the chart itself and not only in the lists.
        marks = sorted(
            [(d, "INCREASE EXPOSURE", GETIN,
              _tech_confirms(sym, d, "INCREASE EXPOSURE"))
             for d in onset_alerts]
            + [(d, "CUT EXPOSURE", BEAR,
                _tech_confirms(sym, d, "CUT EXPOSURE"))
               for d in top_alerts])
        _span_days = max(1.0, (one_i.index.max()
                               - one_i.index.min()).total_seconds() / 86400)
        _rows_used, _lvl_max = [], 0
        for d, text, colour, _hc in marks:
            _dt = pd.Timestamp(d)
            # WIDTH ESTIMATE. The label rendered is "GET OUT 29 Jan 21"
            # - the date adds ~9 characters that `text` does not contain,
            # and the packer only ever counted `text`. At ~7.6 px per
            # character in this font (not 5.4) on a chart that is wider
            # than the 900 px this was tuned for, consecutive labels
            # overlapped whenever two alerts fell in the same quarter -
            # visible as "GET OUT 29 Jan 1GET OUT 09 Jan" on a 9-year
            # window. Measure the WHOLE rendered string, at the real
            # per-character width, plus a gap so two labels never touch.
            # +2 for the "◆ " marker when it is drawn: the packer
            # measures the rendered string, and a marker left out
            # of the estimate is exactly how labels overlapped
            # before.
            _wid_days = ((len(text) + 10 + 6 + (2 if _hc else 0))
                         * 7.6 / 900.0) * _span_days
            _lvl2 = 0
            while _lvl2 < len(_rows_used) and _rows_used[_lvl2] > _dt:
                _lvl2 += 1
            _end = _dt + pd.Timedelta(days=_wid_days)
            if _lvl2 < len(_rows_used):
                _rows_used[_lvl2] = _end
            else:
                _rows_used.append(_end)
            _lvl_max = max(_lvl_max, _lvl2)
            fig.add_annotation(
                x=_ms(d), y=1.0, yref="y domain", yanchor="bottom",
                text=(f"<b>{'◆ ' if _hc else ''}{text}</b>  "
                      f"{_dt.strftime('%d %b %y')}"),
                showarrow=False, font=dict(size=9.5, color=colour),
                bgcolor="rgba(255,255,255,0.92)", borderpad=2,
                yshift=4 + 14 * _lvl2)
        # NO FIGURE TITLE (the header line carries it), and title as an
        # EXPLICIT empty string - Streamlit's plotly theming rewrites an
        # unset title into the literal string "undefined" (traced in the
        # live DOM 2026-07-28).  Keep it.
        _lbl_rows = _lvl_max + 1 if marks else 1
        fig.update_layout(title=dict(text=""),
                          height=380 + 14 * max(0, _lbl_rows - 2),
                          hovermode="x unified",
                          margin=dict(l=STACK_GUTTER_PX, r=10,
                                      t=24 + 14 * _lbl_rows, b=66),
                          showlegend=True,
                          legend=dict(orientation="h", yref="container",
                                      yanchor="bottom", y=0.01,
                                      xref="container", xanchor="center",
                                      x=0.5))
        # LOG IS DISPLAY ONLY: same prices, same flags, same hover - the
        # axis transform changes nothing measured.
        fig.update_yaxes(title_text=("price (USD)" if _px_ok
                                     else "crowd heat level"),
                         type="log" if _log_scale else "linear",
                         automargin=False)
        # same window as the band below - the band was pinned to
        # [w0, w1] while this one auto-ranged with plotly's padding, so
        # the two panels disagreed about where the dates sat
        fig.update_xaxes(range=[w0, w1])
        _axes_fidelity(_theme(fig))
        st.plotly_chart(fig, width="stretch", key=key)
        # ---- THE COMBINED BAND (notebook 08 §11.4:
        # "combine the get out and get in chart with this +1 to -1 one
        # ... make the chart clearer"). ONE panel, three layers, one
        # [-1, +1] axis:
        #   FILL  - the signed readiness (§10.5): green above zero, the
        #           IN side is live and this close to its frozen cut;
        #           red below, the OUT side likewise; dashed lines at
        #           ±1 = a signal fires. The phase routing supplies the
        #           sign, so the fill can never point both ways.
        #   LINE  - the retail-flow dial (§9): the slow posts-only tide
        #           that leads price by 1-3 weeks. Drawn as a LINE, not
        #           a fill, so it stays legible near zero - the
        #           visibility complaint this layout answers.
        #   MARKS - the calls that actually fired, on the band edges.
        # §11.3 is why the dial stays a line and not a trigger: its
        # crossings, judged like everything else, fire 7-30x the false
        # alarms of the shipped calls (utility ~-250 vs -19/+5).
        if (dk_i is not None and len(dk_i) and IN_SCORE in dk_i.columns
                and OUT_SCORE in dk_i.columns
                and _thr_in_d and _thr_out_d):
            _srg = dk_i.loc[(dk_i.index >= w0) & (dk_i.index <= w1)]
            # PER-SIDE coverage (defect report: "why do we
            # have missing data?"): only the LIVE side's score is needed
            # for the reading, so a day scored on one side still draws -
            # requiring both scores was punching holes wherever the
            # other head had no candidate row that day.
            if len(_srg):
                # THE TWO SIDES HAVE DIFFERENT RULES (defect report:
                # "cut side live but an INCREASE gets fired - that makes
                # no sense"). The old signed one-side series claimed the
                # boom gate picks a single live side; that is only true
                # of CUT. The shipped INCREASE signal is deliberately
                # ungated (nb08 §10.4: the gate blocks ~3/4 of its
                # correct calls; 36 of the store's 40 fires land on
                # boomed days), so it stays armed even through a red
                # stretch. The bands draw the WATCH side, one at a
                # time; the markers draw the firing truth, each at its
                # own side's value - see the ONE BAND AT A TIME note
                # below for how the two coexist without contradiction.
                # ONE BAND AT A TIME - STRUCTURALLY (defect report,
                # twice: "when red is on there shouldnt be blue at the
                # same time"). Two masked series with NaN holes were
                # tried; plotly's fill polygons BRIDGE NaN gaps even
                # though the line breaks, so the two bands painted over
                # each other as long diagonal wedges. The only airtight
                # shape is a SINGLE signed series - one value per day,
                # green when positive (watching INCREASE), red when
                # negative (run up -> watching CUT) - from which both
                # traces are clipped. Both drawn at once is then
                # impossible by construction, not by masking. The
                # MARKERS still tell the firing truth: each fire plots
                # at its own side's value, so an ungated INCREASE that
                # fires inside a red stretch draws its green triangle AT
                # the INCREASE line, on top of the red band.
                _b_g = watch_gate(_srg)
                _rin = (_srg[IN_SCORE] / float(_thr_in_d)).clip(0, 1.15)
                _rin = _rin[~_rin.index.duplicated()]
                _rout = (_srg[OUT_SCORE] / float(_thr_out_d)).clip(0, 1.15)
                _rout = _rout[~_rout.index.duplicated()]
                _rout = _rout.reindex(_rin.index)
                _b_g = _b_g[~_b_g.index.duplicated()]
                # SMOOTHED SIDE ("can we smooth this a bit more?"): the
                # raw boom gate flickers at its 20% bar (URA crossed it
                # six times in five weeks), painting one-day colour
                # slivers. A centred 5-day rolling MAJORITY of the gate
                # keeps every regime shift (at most ~2d of lag) and
                # erases the one-two-day flips. Display only - the
                # detector still fires off the raw gate, and the
                # fire-day override below pins every fire to its true
                # side regardless of what the smoothed gate says.
                _b_s = (_b_g.astype(float).reindex(_rin.index)
                        .fillna(0.0)
                        .rolling(5, center=True, min_periods=1)
                        .mean() >= 0.5)
                _sr = _rin.where(~_b_s, -_rout)
                # rolling magnitude, sign untouched: a 3-day centred
                # mean of the HEIGHT only, so the band rolls instead of
                # sawing while the colour boundary stays exact
                _sr = (_sr.abs()
                       .rolling(3, center=True, min_periods=1).mean()
                       * _sr.map(lambda v: (1.0 if v >= 0 else -1.0)
                                 if pd.notna(v) else float("nan")))
                # FIRE-DAY OVERRIDE (defect report, third round: "why is
                # there that floating increase exposure even though the
                # cut side is at the cut exposure side?"). The gate
                # picks the band's side by default, but the ungated
                # INCREASE signal can fire on a red day; a fired call
                # OWNS its day - the band flips to the FIRING side at
                # its own value, so every ▲▼ sits on its own colour,
                # touching its line. No marker ever floats over the
                # opposite band again.
                # ...and the override is DILATED two scored days each
                # way ("the days before and after are also blue"), so a
                # fire owns a small neighbourhood of its own colour
                # instead of a one-day sliver. Neighbour days take
                # their OWN day's score on the firing side, the fire
                # day its exact value - the marker still touches.
                for _fcb, _fsg in ((sig_col("get_in", _srg), 1.0),
                                   (sig_col("get_out", _srg), -1.0)):
                    if _fcb in _srg.columns:
                        _fdd = _srg.index[_srg[_fcb].astype(bool)]
                        _fdd = _fdd[~_fdd.duplicated()]
                        _own = _rin if _fsg > 0 else _rout
                        for _fd in _fdd:
                            _pi = _rin.index.get_indexer([_fd])
                            if _pi[0] < 0:
                                continue
                            for _j in range(max(0, _pi[0] - 2),
                                            min(len(_rin.index),
                                                _pi[0] + 3)):
                                _fv = _own.iloc[_j]
                                if pd.notna(_fv):
                                    _sr.iloc[_j] = _fsg * float(_fv)
                _sr = _sr.dropna().sort_index()
            if len(_srg) and len(_sr):
                # daily calendar; SHORT holes (<=7d) carry forward, long
                # gaps stay blank - an unscored month must look unscored
                _sr = _sr.reindex(
                    pd.date_range(_sr.index.min(),
                                  _sr.index.max())).ffill(limit=7)
                _fb = go.Figure()
                # bands carry NO hover of their own - the overlay trace
                # below owns the hover, so each day shows ONE readout
                # instead of two half-readouts
                _fb.add_scatter(x=_sr.index, y=_sr.clip(lower=0),
                                name="INCREASE side · distance to "
                                     "its line",
                                mode="lines",
                                line=dict(width=0.8, color=GETIN),
                                fill="tozeroy",
                                fillcolor="rgba(30,122,79,0.35)",
                                hoverinfo="skip")
                _fb.add_scatter(x=_sr.index, y=_sr.clip(upper=0),
                                name="CUT side · shown once the "
                                     "name has run up",
                                mode="lines",
                                line=dict(width=0.8, color=BEAR),
                                fill="tozeroy",
                                fillcolor="rgba(166,61,44,0.35)",
                                hoverinfo="skip")
                # black raw-retail-flow line REMOVED on request ("it
                # doesnt make sense") - replaced by the decision hover
                # below, which shows the model's stored readings for
                # that day.
                _hy_day = (_srg["hype_raw"]
                           [~_srg["hype_raw"].index.duplicated()]
                           .reindex(_rin.index)
                           .reindex(_sr.index).ffill(limit=7)) \
                    if "hype_raw" in _srg.columns else None
                _fi_all = {s: set(v) for s, v in (
                    ("in", _srg.index[_srg[sig_col("get_in", _srg)]
                                      .astype(bool)]
                     if sig_col("get_in", _srg) in _srg.columns else []),
                    ("out", _srg.index[_srg[sig_col("get_out", _srg)]
                                       .astype(bool)]
                     if sig_col("get_out", _srg) in _srg.columns else []),
                )}
                def _hbar(frac, colour, n=14):
                    # a bar gauge that survives inside a plotly hover:
                    # filled blocks in the side's colour, the remainder
                    # in light grey, the fire level = the full bar. The
                    # "▕" cap marks the threshold visually.
                    _f = max(0, min(n, int(round(frac * n))))
                    return ("<span style='color:" + colour + "'>"
                            + "█" * _f + "</span>"
                            + "<span style='color:#c9cfd8'>"
                            + "░" * (n - _f) + "</span>▕")
                # FACTOR SERIES for the hover bars (approved): the same
                # observable ingredients the tiles show - posts vs the
                # prior week, attention vs the name's own normal, net
                # bullishness - computed per DAY from the published
                # aggregates. NOT the model's internal weighted
                # readings (those are not in the published data); a
                # name absent from the aggregates shows fewer bars.
                _po_d = _at_d = _nb_d = None
                try:
                    if theme_counts is not None and len(theme_counts):
                        _tcn = theme_counts[
                            theme_counts["theme"] == name]
                        if len(_tcn):
                            _s_f = (_tcn.set_index("date")
                                    ["mention_count"].sort_index()
                                    .resample("D").sum())
                            _wk_f = _s_f.rolling(
                                7, min_periods=3).mean()
                            _po_d = (_wk_f / _wk_f.shift(7)
                                     - 1).reindex(_sr.index)
                            _md_f = _wk_f.rolling(
                                120, min_periods=30).median()
                            _at_d = (_wk_f / _md_f).reindex(_sr.index)
                    if (theme_sentiment is not None
                            and len(theme_sentiment)):
                        _tsn = theme_sentiment[
                            theme_sentiment["theme"] == name]
                        if len(_tsn):
                            _nb_d = (_tsn.set_index("date")
                                     ["net_bullish"].sort_index()
                                     [lambda x: ~x.index.duplicated()]
                                     .resample("D").mean()
                                     .ffill(limit=7)
                                     .reindex(_sr.index))
                except Exception:                        # noqa: BLE001
                    _po_d = _at_d = _nb_d = None

                def _fin(x):
                    return (x is not None and pd.notna(x)
                            and abs(float(x)) != float("inf"))
                # STACKED CONTRIBUTIONS (approved: "factor + factor +
                # factor ends up being more than the threshold"). When
                # the components store exists, each day's bar is split
                # by what the model is actually adding up - the frozen
                # logit weights x that day's readings - grouped into
                # three legible buckets. The bar's TOTAL length stays
                # the true progress (score/trigger), so segment widths
                # are each bucket's share of the positive sum.
                _BUCKETS = (
                    ("posts & attention", "#3a6ea5",
                     ("attention_accel", "hype_ratio", "influx_speed",
                      "attention_convexity", "e1", "e3")),
                    ("sentiment", "#a07d2e",
                     ("bull_inflection", "bull_level", "bull_persist")),
                    # labelled "momentum", not "price", on request -
                    # truthful: the bucket is the run-up + the 1-month
                    # return, i.e. momentum readings
                    ("momentum", "#5b6673",
                     ("price_runup", "price_ret21")),
                )
                _cmpN = None
                _W = _desk_weights()
                if (desk_components is not None
                        and len(desk_components) and _W
                        and (_W["in"] or _W["out"])):
                    _cgc = desk_components[
                        desk_components["name"] == name]
                    if len(_cgc):
                        _cgc = (_cgc.set_index("date").sort_index()
                                [lambda x: ~x.index.duplicated()])
                        _cmpN = (_cgc.drop(columns=["name"],
                                           errors="ignore")
                                 .reindex(_sr.index).ffill(limit=7))

                def _stack_lines(_d, _v, _pct, _col_h):
                    """(bar_line, contribs_line, pulling_line|None) for
                    one day, or None when no components that day."""
                    if _cmpN is None or _d not in _cmpN.index:
                        return None
                    _row_c = _cmpN.loc[_d]
                    _wts = _W["out"] if _v < 0 else _W["in"]
                    _bk = []
                    _negs = []
                    _any = False
                    for _bl, _bc, _fs in _BUCKETS:
                        _s = 0.0
                        for _f in _fs:
                            _val = _row_c.get(_f)
                            _wf = _wts.get(_f)
                            if _wf is None or not _fin(_val):
                                continue
                            _c = float(_wf) * float(_val)
                            _any = True
                            _s += _c
                            if _c <= -0.10:
                                _negs.append(
                                    (_W["labels"].get(_f, _f), _c))
                        _bk.append((_bl, _bc, _s))
                    if not _any:
                        return None
                    _tp = sum(max(_s, 0.0) for _bl, _bc, _s in _bk)
                    # 24-wide bar (was 14) with the FIRE line as a bold
                    # cap and the distance stated in words - "quite
                    # unclear how many % of the way" fix
                    _NW = 24
                    _n_ch = max(1, int(round(min(_pct, 1.0) * _NW)))
                    _segs = ""
                    _used = 0
                    _live = [(_bl, _bc, _s) for _bl, _bc, _s in _bk
                             if _s > 0]
                    for _i2, (_bl, _bc, _s) in enumerate(_live):
                        _w_ch = (max(1, int(round(_s / _tp * _n_ch)))
                                 if _i2 < len(_live) - 1
                                 else max(1, _n_ch - _used)) \
                            if _tp > 0 else 0
                        _used += _w_ch
                        _segs += (f"<span style='color:{_bc}'>"
                                  + "█" * _w_ch + "</span>")
                    _rest = max(0, _NW - min(_used, _NW))
                    _l_bar = (_segs
                              + "<span style='color:#c9cfd8'>"
                              + "░" * _rest + "</span><b>▌FIRE</b>")
                    _l_pc = (f"<b>{min(_pct, 1.15):.0%} of the way</b>"
                             + (f" · {max(0.0, 1 - _pct):.0%} to go"
                                if _pct < 1.0 else
                                " · <b>AT THE LINE</b>"))
                    _l_bar = _l_pc + "<br>" + _l_bar
                    _l_ct = " · ".join(
                        f"<span style='color:{_bc}'>█</span> {_bl} "
                        f"<b>{_s:+.2f}</b>" for _bl, _bc, _s in _bk)
                    _l_ng = ("pulling back: " + " · ".join(
                        f"{_nl} {_nc:+.2f}" for _nl, _nc in _negs[:3])
                        if _negs else None)
                    return _l_bar, _l_ct, _l_ng
                _htxt = []
                for _d, _v in _sr.items():
                    if pd.isna(_v):
                        _htxt.append("")
                        continue
                    _red = _v < 0
                    _sd = "out" if _red else "in"
                    _col_h = BEAR if _red else GETIN
                    _pct = min(abs(_v), 1.15)
                    _fired_td = _d in _fi_all[_sd]
                    _l1h = (f"<b>{_d:%d %b %y}</b> · "
                            + ("red — reducing side"
                               if _red else "green — increasing side"))
                    # fire-checks line removed on request ("just make
                    # it the bar - keep it simple"); FIRED days keep
                    # their one bold line
                    _stk = _stack_lines(_d, _v, _pct, _col_h)
                    if _stk is not None:
                        _rows_h = [_stk[0], _stk[1]]
                        if _stk[2]:
                            _rows_h.append(_stk[2])
                        if _fired_td:
                            _rows_h.append("<b>FIRED today</b>")
                        _htxt.append("<br>".join([_l1h] + _rows_h))
                        continue
                    _f_ch = max(0, min(24,
                                       int(round(min(_pct, 1.0) * 24))))
                    _rows_h = [
                        f"<b>{_pct:.0%} of the way</b>"
                        + (f" · {max(0.0, 1 - _pct):.0%} to go"
                           if _pct < 1.0 else " · <b>AT THE LINE</b>")
                        + "<br>"
                        + f"<span style='color:{_col_h}'>"
                        + "█" * _f_ch + "</span>"
                        + "<span style='color:#c9cfd8'>"
                        + "░" * (24 - _f_ch) + "</span><b>▌FIRE</b>"]
                    _po = _po_d.get(_d) if _po_d is not None else None
                    if _fin(_po):
                        _po = float(_po)
                        _rows_h.append(
                            f"posts, wk vs prior  "
                            f"{_hbar(max(0.0, min(_po, 1.0)), _col_h)} "
                            f"<b>{min(_po, 9.99):+.0%}</b>")
                    _atv = _at_d.get(_d) if _at_d is not None else None
                    if _fin(_atv):
                        _atv = float(_atv)
                        _rows_h.append(
                            f"attention  "
                            f"{_hbar(min(_atv / 2.0, 1.0), _col_h)} "
                            f"<b>{_atv:.1f}×</b> its normal "
                            "(2× = elevated)")
                    _nbv = _nb_d.get(_d) if _nb_d is not None else None
                    if _fin(_nbv):
                        _nbv = float(_nbv)
                        _rows_h.append(
                            f"bullishness  "
                            f"{_hbar(max(0.0, min(_nbv, 1.0)), _col_h)} "
                            f"<b>{_nbv:+.2f}</b>  (full bar = +1)")
                    if len(_rows_h) == 1:
                        # aggregates hold nothing for this name (a
                        # single, or a data gap) - fall back to the
                        # stored crowd-heat ratio so the hover never
                        # goes bare
                        _hy = (float(_hy_day.get(_d))
                               if _hy_day is not None
                               and pd.notna(_hy_day.get(_d)) else None)
                        if _hy is not None:
                            _rows_h.append(
                                f"interest  "
                                f"{_hbar(min(_hy / 2.0, 1.0), _col_h)} "
                                f"<b>{_hy:.1f}×</b> its normal")
                    if _fired_td:
                        _rows_h.append("<b>FIRED today</b>")
                    _htxt.append("<br>".join([_l1h] + _rows_h))
                _fb.add_scatter(
                    x=_sr.index, y=_sr, mode="lines",
                    line=dict(width=0.5, color="rgba(0,0,0,0)"),
                    showlegend=False, text=_htxt,
                    hovertemplate="%{text}<extra></extra>")
                # fired calls sit ON the curve; fall back to the line
                # they fired at when the curve has no reading that day
                # Each marker sits at ITS OWN side's score that day,
                # never at the signed band's value: get_in_nogate can
                # fire on a boomed day, where the signed band shows the
                # CUT side - plotting there put a green "INCREASE fired"
                # triangle deep in the red fill at the WRONG quantity.
                # 36 of the store's 40 nogate fires land on boomed days,
                # so this was the rule, not the exception.
                _y_in = (_srg[IN_SCORE] / float(_thr_in_d)).clip(0, 1.15) \
                    if IN_SCORE in _srg.columns else pd.Series(dtype=float)
                _y_out = (-(_srg[OUT_SCORE] / float(_thr_out_d))
                          .clip(0, 1.15)) \
                    if OUT_SCORE in _srg.columns else pd.Series(dtype=float)
                _y_in = _y_in[~_y_in.index.duplicated()]
                _y_out = _y_out[~_y_out.index.duplicated()]
                for _col_b, _mk_b, _cc_b, _own_y, _line_y, _nm_b in (
                        (sig_col("get_in", _srg), "triangle-up", GETIN,
                         _y_in, 1.0, "INCREASE EXPOSURE fired"),
                        (sig_col("get_out", _srg), "triangle-down", BEAR,
                         _y_out, -1.0, "CUT EXPOSURE fired")):
                    if _col_b in _srg.columns:
                        _dd_b = _srg.index[_srg[_col_b].astype(bool)]
                        if len(_dd_b):
                            _yy = [float(_own_y.get(d, _line_y))
                                   if pd.notna(_own_y.get(d, float("nan")))
                                   else _line_y for d in _dd_b]
                            _fb.add_scatter(
                                x=_dd_b, y=_yy,
                                name=_nm_b, mode="markers",
                                marker=dict(symbol=_mk_b, size=11,
                                            color=_cc_b,
                                            line=dict(width=1.2,
                                                      color="white")),
                                hovertemplate="%{x|%d %b %y}"
                                              f"<extra>{_nm_b}</extra>")
                # held-diamonds REMOVED on request ("remove the thing
                # thats like held ... no need") - the ▲▼ fires and the
                # bar itself carry the story now
                for _yv, _cc in ((1.0, GETIN), (-1.0, BEAR)):
                    _fb.add_hline(y=_yv, line_dash="dash", line_width=1,
                                  line_color=_cc, opacity=0.7)
                _fb.add_hline(y=0, line_width=1, line_color=INK_MUTED)
                _fb.update_layout(
                    title=dict(text=""), height=185, showlegend=True,
                    legend=dict(orientation="h", yref="container",
                                yanchor="bottom", y=0.0,
                                xref="container", xanchor="center",
                                x=0.5, font=dict(size=10)),
                    margin=dict(l=STACK_GUTTER_PX, r=10, t=4, b=34),
                    hovermode="x unified",
                    yaxis=dict(range=[-1.3, 1.3], automargin=False,
                               tickvals=[-1, 0, 1],
                               ticktext=["at CUT EXPOSURE", "", "at INCREASE EXPOSURE"],
                               tickfont=dict(size=10)))
                _fb.update_xaxes(range=[w0, w1])
                _axes_fidelity(_theme(_fb))
                st.plotly_chart(_fb, width="stretch",
                                key=f"{key}_srband")
                # band caption removed on request ("no need that txt") -
                # the legend names the two sides and the hover carries
                # the detail, so the paragraph was restating the chart
        st.markdown(
            f"<span style='font-size:11px;color:{INK_MUTED}'>"
            "how to read this chart</span>",
            unsafe_allow_html=True,
            help=(
                "**Hover anywhere on the line** to see the full state of "
                "the signal that day. It leads with **the state**, "
                "resolved in strict priority: a flag fired within the "
                "last 21 days owns the state for its whole episode "
                "window (**EXIT WINDOW** after a CUT EXPOSURE, **ENTRY "
                "WINDOW** after a INCREASE EXPOSURE - one clear direction, the "
                "flag's direction); otherwise the crowd-phase clock "
                "reads QUIET → BUILDING → BLOW-OFF → TOPPING → COOLING "
                "(notebook 08: two smooth coordinates, crowd extremity "
                "x arrival momentum, so it can never read entry and "
                "exit at once and cannot flip overnight; display only - "
                "the frozen rules still fire every flag). Below the "
                "state, one rule's breakdown - the one the state "
                "selects: a bar showing how far "
                "the deciding score sits "
                "toward its frozen trigger (-100% = INCREASE EXPOSURE fires, +100% = "
                "CUT EXPOSURE fires), whether the gates would have let it fire, "
                "and every contributing factor in plain English - each "
                "one a percentile of this name's own trailing year.\n\n"
                "**The flag is the score reaching 100% of its trigger "
                "while the gates are open** - no inflection test, no "
                "second condition. A vertical line + a dot on the price "
                "is a day the signal actually fired (the stored flag, "
                "never recomputed).\n\n"
                "**Amber stretches of the price line** are the danger "
                "state: the crowd at least twice its own normal AND the "
                "price in a confirmed boom.\n\n"
                "How hot the crowd is - the crowd heat level - is the dial "
                "and the facts above; the signal record beside them is "
                "this name's own measured history of what prices did "
                "after each flag."))

    # ======================================================================
    # ACTION VIEW - the landing page
    # ----------------------------------------------------------------------
    # One question, answered above the fold: what should exposure change
    # on TODAY. Two ranked lists - names closest to firing INCREASE
    # EXPOSURE, and names closest to firing CUT EXPOSURE - and nothing
    # else until asked for.
    #
    # NO NEW MODELLING. Readiness is score / trigger on the same stored
    # scores and the same frozen cuts every other surface reads. The
    # phase gate picks the WATCH side - the side the desk leads with
    # for a name today (boomed -> CUT, else INCREASE) - exactly as the
    # watchlist below computes it. NOTE this is a display convention,
    # not a firing rule: the shipped INCREASE signal is ungated (nb08
    # §10.4) and can fire on a boomed day too; the band chart draws
    # that truth. Only CUT is truly blocked off its side.
    #
    # Size rule: every name at >= ACTION_READY_PCT, or the top
    # ACTION_MIN_ROWS if fewer clear the bar - so the page is never empty
    # on a quiet day and never floods on a loud one.
    #
    # The detail is the SAME draw_chart the full tab uses, called inside
    # each row's expander. There is no second chart implementation to
    # drift.
    if mode == "action":
        # LOOK UP ANY NAME, above the lists. The lists answer "what
        # should I act on today"; this answers "how is X doing" for a
        # name nowhere near firing, which is most of the universe on
        # most days. It sits at the top because it is a question people
        # arrive with, not one the page provokes - and with nothing
        # selected it costs a single row of height.
        _lk_names = sorted(ek["name"].unique()) if ek is not None else []
        _lk = st.selectbox(
            f"Look up any {kind_label.lower()} - type to search",
            ["(none)"] + list(_lk_names), key=f"{key_prefix}_act_lookup",
            # The sentinel is passed through UNformatted, exactly as the
            # full tab's lookup does. Rewriting it to prose broke the
            # round-trip between a value and its rendered label.
            format_func=lambda n: n if n == "(none)" else flag_label(n, kind),
            help="Opens the same chart and detail the lists below show, "
                 "for any tracked name - whether or not it has ever "
                 "signalled.")
        if _lk and _lk != "(none)":
            draw_chart(_lk, "", f"{key_prefix}_act_lookup_chart")
            st.markdown('<div class="rf-rule"></div>',
                        unsafe_allow_html=True)

        # THE KEY - now behind a dropdown on request ("can we make
        # this a dropdown?"), exact wording unchanged.
        with st.expander("Definitions: What do the terms on this "
                         "page mean?"):
            st.markdown(
                "<div class='rf-keyline'>"
                "<b>A signal</b> means at this retail level we have had "
                "MAJOR price movements in the past (quite rare).<br>"
                "<b>High conviction</b> means technical factors (price, "
                "moving averages) also correlate.<br>"
                "<b>On the way to a signal</b> means consider this theme "
                "and use AI Pulse tab to understand more."
                "</div>", unsafe_allow_html=True)

        # ONE SIDE PER NAME. readiness_now decides which - the WATCH
        # side, the one the desk leads with today (boomed -> CUT, else
        # INCREASE). An earlier version scored every name on BOTH sides
        # and listed it twice, putting the same theme under INCREASE
        # and CUT at once: not two readings, one contradiction. NB:
        # this is a ranking convention, not a firing rule - the ungated
        # INCREASE signal can still fire on a boomed name (the band
        # chart shows both armed sides); only CUT is truly gated.
        #
        # TWO THINGS THIS GETS RIGHT THAT THE OBVIOUS VERSION DOES NOT.
        #
        # 1. The score is INTERMITTENT. in_score exists only on days the
        #    name clears the candidacy floor, so the newest row for a
        #    name is very often NaN while a perfectly current score sits
        #    a few days behind it. Taking `.iloc[-1]` therefore dropped
        #    ~25 of 33 themes for having "no score" when they had one.
        #    The most recent SCORED row is used instead.
        #
        # 2. A name's rows STOP when it leaves the scored universe, so
        #    "the last row" can be arbitrarily old. Read literally, this
        #    page showed europe_defense at 79% of its CUT trigger from a
        #    reading 494 DAYS OLD, presented as what to act on today.
        #    Anything older than ACTION_STALE_DAYS is therefore dropped,
        #    and every surviving row states the date its score is from -
        #    the same 60-day convention readiness_alerts.json already
        #    applies at source.
        _a_as_of = (dk["date"].max() if hi is None
                    else min(hi, dk["date"].max())) if dk is not None else None
        def _near_since(scored, col, thr):
            """First DATE of the current unbroken run of scored days at
            or above the 'close' band (READY_AMBER) - the honest answer
            to 'how new is this name here'. None when today is below.
            Dates come from the frame's date column: scored keeps the
            desk store's integer index, so positional index values would
            be meaningless as dates."""
            rs = pd.Series(scored[col].astype(float).values / thr * 100.0,
                           index=pd.to_datetime(scored["date"].values))
            if not len(rs) or rs.iloc[-1] < READY_AMBER:
                return None
            below = rs < READY_AMBER
            if not below.any():
                return rs.index[0]
            last_below = below[below].index[-1]
            after = rs.index[rs.index > last_below]
            return after[0] if len(after) else None

        _a_rows = []
        if dk is not None and len(dk) and _a_as_of is not None:
            for _n, _g in dk[dk["date"] <= _a_as_of].groupby("name"):
                _g = _g.sort_values("date")
                if _g.empty:
                    continue
                # gate and score from the SAME day (see
                # eligible_scored_now for the placeholder-row defect
                # this replaces)
                _es = eligible_scored_now(_g)
                if _es is None:
                    continue
                _row, _boomed = _es
                _col = OUT_SCORE if _boomed else IN_SCORE
                _thr = _thr_out_d if _boomed else _thr_in_d
                _side = "CUT EXPOSURE" if _boomed else "INCREASE EXPOSURE"
                if _thr in (None, 0):
                    continue
                _scored = _g[_g[_col].notna()]
                if _scored.empty:
                    continue
                if (_a_as_of - _row["date"]).days > ACTION_STALE_DAYS:
                    continue                    # stale - not a call
                _sc = float(_row[_col])

                def _at(days_back):
                    """The score as it stood `days_back` days earlier -
                    the nearest scored row at or before that date."""
                    _e = _scored[_scored["date"]
                                 <= _row["date"] - pd.Timedelta(days=days_back)]
                    return float(_e.iloc[-1][_col]) if len(_e) else None

                _a_rows.append({
                    "name": _n, "symbol": _row.get("symbol") or "-",
                    "side": _side,
                    "ready": max(0.0, min(100.0, 100.0 * _sc / float(_thr))),
                    "fired": _sc >= float(_thr),
                    "score_date": _row["date"],
                    "age": int((_a_as_of - _row["date"]).days),
                    "d7": (100.0 * (_sc - _at(7)) / float(_thr)
                           if _at(7) is not None else None),
                    "d14": (100.0 * (_sc - _at(14)) / float(_thr)
                            if _at(14) is not None else None),
                    "near_since": _near_since(_scored, _col, float(_thr)),
                    "hiconv": _tech_confirms(_row.get("symbol"),
                                             _row["date"], _side),
                    "px7": _price_change(_row.get("symbol"),
                                         _row["date"], 7),
                    "px14": _price_change(_row.get("symbol"),
                                          _row["date"], 14)})
        _a_df = pd.DataFrame(_a_rows)

        def _a_pick(side):
            """Everything at or above the bar, or the nearest few when
            nothing clears it - so the page is never empty and never
            floods."""
            if _a_df.empty:
                return _a_df
            # CLOSEST TO FIRING, full stop. Age is deliberately not a
            # tiebreak and is not shown: the list answers "what is
            # nearest its trigger", and a name that has not been scored
            # recently still holds its last measured position in that
            # ranking.
            s = (_a_df[_a_df["side"] == side]
                 .sort_values("ready", ascending=False))
            hot = s[s["ready"] >= ACTION_READY_PCT]
            return hot if len(hot) >= ACTION_MIN_ROWS else s.head(
                ACTION_MIN_ROWS)

        # LAZY ROWS. st.expander renders its contents EAGERLY, whether
        # or not it is open, so a list of 14 rows each holding a full
        # price chart built 42 charts before the page could paint: 16s
        # cold and 11s on EVERY rerun, which reads as a page that never
        # loads. One row is open at a time, tracked in session_state, and
        # only that row builds a chart - the rest cost one button each.
        _OPEN = f"{key_prefix}_open_row"

        def _a_row(r, side, tone, key):
            rid = f"{side}:{r.name}"
            _head = (f"{theme_label(r.name)}  ({r.symbol})   "
                     f"{r.ready:.0f}% of the way to a signal"
                     + ("   ● FIRING NOW" if r.fired else "")
                     + ("   ◆ HIGH CONVICTION" if r.hiconv else ""))
            _was = st.session_state.get(_OPEN) == rid
            with st.container(key=f"rfrow_{key}"):
                _hit = st.button(("▼  " if _was else "▶  ") + _head,
                                 key=f"btn_{key}", width="stretch")
            if _hit:
                st.session_state[_OPEN] = None if _was else rid
                st.rerun()
            if st.session_state.get(_OPEN) != rid:
                return
            st.markdown(
                f"<div class='rf-readbar'><div class='rf-readfill' "
                f"style='width:{min(100.0, r.ready):.0f}%;"
                f"background:{tone}'></div>"
                f"<div class='rf-readtick'></div></div>",
                unsafe_allow_html=True)
            if r.hiconv:
                st.markdown(
                    f"<span class='rf-hiconv' style='border-color:{tone};"
                    f"color:{tone}'>◆ HIGH CONVICTION</span>",
                    unsafe_allow_html=True)
                # the blue banner treatment, matched to the "Technical
                # factors also correlate" one ("thats good")
                # the (?) rides ON the banner, right after Jonathan's
                # framework - st.markdown places its help icon at the
                # end of the text ("put the ? right next to Jonathan's")
                st.markdown(
                    "<div style='background:rgba(28,131,225,0.10);"
                    "border-radius:8px;padding:12px 16px'>"
                    "<b>◆ HIGH CONVICTION.</b> The crowd signal and the "
                    "technical screen (price and moving averages) agree "
                    "on this name today — <b>Jonathan's momentum "
                    "framework</b>, five factors behind the (?)</div>",
                    unsafe_allow_html=True, help=HIGH_CONV_HELP)
            # pd.notna, not `is not None`: DataFrame construction
            # coerces the near_since column to datetime64, so a row
            # without a streak holds NaT - which passes an
            # `is not None` check and then crashes strftime. That
            # crash killed everything below the row (the fired list
            # and the radar vanished with it).
            if pd.notna(getattr(r, "near_since", None)):
                _ns_days = (pd.Timestamp(_a_as_of)
                            - pd.Timestamp(r.near_since)).days
                st.caption(f"near its trigger since "
                           f"{pd.Timestamp(r.near_since):%d %b %Y} "
                           f"({_ns_days}d)")
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("% of the way to a signal", f"{r.ready:.0f}%")
            for _c, _v, _lbl in ((_c2, r.d7, "7d change"),
                                 (_c3, r.d14, "14d change")):
                if _v is None:
                    _c.metric(_lbl, "-")
                else:
                    _c.metric(_lbl, f"{_v:+.0f} pts",
                              delta=("approaching" if _v > 0
                                     else "receding" if _v < 0 else "flat"),
                              delta_color=("inverse" if _v < 0 else "normal"))
            # DO READINESS AND PRICE AGREE? Both changes are measured
            # over the SAME two windows, so this is like-for-like. Only
            # stated when both windows agree - one window agreeing is a
            # coin toss with extra steps.
            _agree = [w for w, _rv, _pv in
                      ((7, r.d7, r.px7), (14, r.d14, r.px14))
                      if _rv is not None and _pv is not None
                      and _rv != 0 and _pv != 0 and ((_rv > 0) == (_pv > 0))]
            if len(_agree) == 2:
                _dirw = "rising" if (r.d7 or 0) > 0 else "falling"
                st.info(
                    f"**Technical factors also correlate.** Readiness and "
                    f"price are both {_dirw} over 7 and 14 days "
                    f"(price {r.px7:+.1f}% / {r.px14:+.1f}%), so the crowd "
                    "read and the price action point the same way.")
            # "100% is the trigger..." caption removed on request; the
            # staleness warning stays - a reader must still know when a
            # reading is days old
            if r.age > 3:
                st.caption(f"Scored {r.score_date:%d %b %Y}, {r.age}d "
                           "ago - the name has not been scored since.")
            # THE INGREDIENTS, in the open. The readiness number is a
            # model output; these are the observable inputs behind it,
            # straight from the text-free aggregates - so a reader can
            # see WHAT is pushing the name toward its trigger instead of
            # taking a percentage on faith.
            # shown in the open, not behind a dropdown ("no need to
            # dropdown - just show this by default")
            st.markdown("**What makes it close to a trigger?**")
            with st.container():
                _ing = _ingredients(r.name)
                if _ing is None:
                    st.caption("no aggregate history for this name")
                else:
                    (_pd7, _pchg, _hyp, _nb, _nb5, _boomed_i,
                     _shr, _shr4) = _ing
                    # DRIVER flags FIRST, so the hot tiles themselves
                    # can be highlighted ("bold the tile that has it")
                    # - a driver tile gets a border. Thresholds are the
                    # tiles' own conventions (2x = elevated; +-0.5
                    # bullishness is very one-sided).
                    _bdy = None
                    if pd.notna(getattr(r, "near_since", None)):
                        _bdy = (pd.Timestamp(_a_as_of)
                                - pd.Timestamp(r.near_since)).days + 1
                    _dr_posts = _pchg is not None and _pchg >= 0.25
                    _dr_att = _hyp is not None and _hyp >= 1.5
                    _dr_nb = _nb is not None and _nb >= 0.5
                    _dr_nb5 = _nb5 is not None and _nb5 >= 0.15
                    _dr_days = _bdy is not None and _bdy >= 3
                    _l1, _l2, _l3 = st.columns(3)
                    _l1.metric("posts, 7d vs prior week",
                               f"{_pchg:+.0%}" if _pchg is not None
                               else "-",
                               delta=("driving the signal"
                                      if _dr_posts else None),
                               delta_color="off", border=_dr_posts)
                    _l2.metric("vs its own 120d norm",
                               f"{_hyp:.1f}×" if _hyp is not None else "-",
                               delta=("driving the signal" if _dr_att
                                      else ("elevated"
                                            if (_hyp or 0) >= 2
                                            else "normal range")),
                               delta_color="off", border=_dr_att)
                    _l3.metric("net bullishness",
                               f"{_nb:+.2f}" if _nb is not None else "-",
                               delta=(("driving the signal · "
                                       if (_dr_nb or _dr_nb5) else "")
                                      + (f"{_nb5:+.2f} over 5d"
                                         if _nb5 is not None else "")
                                      or None),
                               delta_color="off",
                               border=(_dr_nb or _dr_nb5))
                    # second row (approved additions): persistence,
                    # size in the room
                    _l4, _l5, _l6 = st.columns(3)
                    _l4.metric("days building",
                               f"{_bdy}d" if _bdy is not None else "new",
                               delta=(("driving the signal"
                                       if _dr_days else
                                       "consecutive days near the "
                                       "trigger") if _bdy is not None
                                      else "first day near the trigger"),
                               delta_color="off", border=_dr_days)
                    _l5.metric("share of theme chatter",
                               f"{_shr:.1%}" if _shr is not None else "-",
                               delta=(f"4wk avg {_shr4:.1%}"
                                      if _shr4 is not None else None),
                               delta_color="off")
                    # breadth tile removed on request ("just remove the
                    # tile that says breadth") - _l6 stays empty so the
                    # two rows keep the same grid
                    # THE DRIVERS, named. Price line removed on request
                    # - the run-up state already shows in the chart.
                    _drv = []
                    if _dr_posts:
                        _drv.append(f"posts +{_pchg:.0%} vs prior week")
                    if _dr_att:
                        _drv.append(f"attention {_hyp:.1f}× its normal")
                    if _dr_nb:
                        _drv.append("very one-sided bullishness "
                                    f"({_nb:+.2f})")
                    if _dr_nb5:
                        _drv.append(f"mood turning up ({_nb5:+.2f} "
                                    "in 5d)")
                    if _dr_days:
                        _drv.append(f"{_bdy} days building")
                    if _drv:
                        st.markdown(
                            f"<div style='font-size:13.5px;"
                            f"margin:4px 0 2px 0;color:{tone}'>"
                            "<b>What's pushing it toward the "
                            "trigger:</b> " + " · ".join(_drv)
                            + "</div>", unsafe_allow_html=True)
                    else:
                        st.caption("no single reading is extreme - "
                                   "the score comes from several "
                                   "mildly elevated readings together")
            draw_chart(r.name, "", key)

        def _ingredients(nm):
            """(posts_day7, wow_change, hype_x, net_bull, nb_5d_chg,
            boomed) from the aggregates - observable, no model."""
            if theme_counts is None or not len(theme_counts):
                return None
            tcn = theme_counts[theme_counts["theme"] == nm]
            if not len(tcn):
                return None
            s = (tcn.set_index("date")["mention_count"].sort_index()
                 .resample("D").sum())
            if _a_as_of is not None:
                s = s[s.index <= pd.Timestamp(_a_as_of)]
            if not len(s):
                return None
            wk = float(s.tail(7).mean())
            prev = float(s.tail(14).head(7).mean()) if len(s) >= 14 else None
            wow = ((wk / prev - 1) if prev else None)
            med120 = float(s.tail(120).rolling(7).mean().median()) \
                if len(s) >= 30 else None
            hyp = (wk / med120) if med120 else None
            nb = nb5 = None
            if theme_sentiment is not None and len(theme_sentiment):
                tsn = theme_sentiment[theme_sentiment["theme"] == nm]
                if len(tsn):
                    q = tsn.set_index("date").sort_index()
                    if _a_as_of is not None:
                        q = q[q.index <= pd.Timestamp(_a_as_of)]
                    if len(q):
                        nb = float(q["net_bullish"].iloc[-1])
                        if len(q) > 5:
                            nb5 = nb - float(q["net_bullish"].iloc[-6])
            _bmn = False
            if dk is not None and len(dk):
                _gg = dk[dk["name"] == nm]
                if _a_as_of is not None:
                    _gg = _gg[_gg["date"] <= pd.Timestamp(_a_as_of)]
                _esn = eligible_scored_now(_gg.sort_values("date"))
                if _esn is not None:
                    _bmn = _esn[1]
            # share of ALL theme chatter (with the 4-week average for
            # contrast): how big this theme is in the room, not just
            # against its own history
            shr = shr4 = None
            _tot = (theme_counts.groupby("date")["mention_count"].sum()
                    .sort_index().resample("D").sum())
            if _a_as_of is not None:
                _tot = _tot[_tot.index <= pd.Timestamp(_a_as_of)]
            _wt = float(_tot.tail(7).sum())
            if _wt:
                shr = float(s.tail(7).sum()) / _wt
            _t28 = float(_tot.tail(28).sum())
            if _t28:
                shr4 = float(s.tail(28).sum()) / _t28
            return wk, wow, hyp, nb, nb5, _bmn, shr, shr4

        def _a_list(side, tone, blurb, title=None, head_help=None):
            st.markdown(f"<div class='rf-actionhead' style='color:{tone}'>"
                        f"{title or side}</div>", unsafe_allow_html=True,
                        help=head_help)
            st.markdown(f"<div class='rf-actionsub'>{blurb}</div>",
                        unsafe_allow_html=True)
            rows = _a_pick(side)
            if rows is None or rows.empty:
                if side == "CUT EXPOSURE":
                    st.caption(
                        "Nothing on this side today - which is quite "
                        "rare, and worth reading as information: this "
                        "tracker exists mainly to catch bubbles, and an "
                        "empty reduce-exposure list means no tracked "
                        "theme is currently both run-up and crowded "
                        "enough to be near a top call.")
                else:
                    st.caption("nothing scored on this side today")
                return
            for _r in rows.itertuples():
                _a_row(_r, side, tone,
                       f"{key_prefix}_act_{side[:3]}_{_r.name}")

        # CUT first: a position already held is the more urgent decision.
        # Stacked full width rather than two columns - the row labels and
        # the charts inside them need the whole measure.
        _a_list("CUT EXPOSURE", BEAR,
                "Names that have already run and are topping out. "
                "Closest to firing first.",
                title="Extreme Bullishness — Consider Reducing Exposure",
                head_help=(
                    "**What this list means:**\n\n"
                    "- Every name here has **already had its big price "
                    "run** — up at least 20% from its recent low.\n"
                    "- The % measures how much today looks like past "
                    "peaks:\n"
                    "  - a lot more posts than before\n"
                    "  - almost everyone bullish\n"
                    "  - it has lasted a while\n"
                    "  - the price has run a long way"))
        st.markdown('<div class="rf-rule"></div>', unsafe_allow_html=True)
        _a_list("INCREASE EXPOSURE", GETIN,
                "Names the crowd is arriving at, before the run. "
                "Closest to firing first.",
                title="Start of Bullishness — Consider Increasing "
                      "Exposure",
                head_help=(
                    "**What this list means:**\n\n"
                    "- These names have **not had their price run yet** "
                    "— the interest is arriving first.\n"
                    "- The % measures how much interest is building "
                    "versus before:\n"
                    "  - more posts than usual\n"
                    "  - growing faster and faster\n"
                    "  - the mood turning positive\n"
                    "  - the price starting to move\n"
                    "- ◆ HIGH CONVICTION means price trends (moving "
                    "averages and momentum) also agree that same day."))

        # ---- LAST FIRED SIGNALS ------------------------------------------
        # The lists above are about what has NOT fired yet. This is the
        # record of what did: the most recent coherent alerts across the
        # panel, newest first, each opening the same chart.
        st.markdown('<div class="rf-rule"></div>', unsafe_allow_html=True)
        st.markdown("<div class='rf-actionhead' style='color:"
                    f"{INK}'>LAST FIRED</div>", unsafe_allow_html=True)
        st.caption("The most recent signals the detector actually fired, "
                   "newest first.")
        _fired = []
        for _n, (_co, _ct) in coherent.items():
            for _d, _sd in [(d, "INCREASE EXPOSURE") for d in _co] + \
                           [(d, "CUT EXPOSURE") for d in _ct]:
                if _d is None:
                    continue
                if hi is not None and _d > hi:
                    continue
                _fired.append({"name": _n, "date": _d, "side": _sd})
        _fired = sorted(_fired, key=lambda r: r["date"], reverse=True)
        _fired = [f for f in _fired if f["name"] not in HIDDEN_THEMES]
        if not _fired:
            st.caption("no signals fired in the stored history")
        else:
            for _f in _fired[:ACTION_FIRED_ROWS]:
                _ago = (pd.Timestamp(_a_as_of) - _f["date"]).days \
                    if _a_as_of is not None else None
                _tone = GETIN if _f["side"].startswith("INCREASE") else BEAR
                _sym_f = None
                if dk is not None and len(dk):
                    _mf = dk[dk["name"] == _f["name"]]
                    if len(_mf):
                        _sym_f = _mf.iloc[-1].get("symbol")
                _hc_f = _tech_confirms(_sym_f, _f["date"], _f["side"])
                _fid = f"F:{_f['name']}:{_f['date']:%Y%m%d}"
                # The tradeable instrument belongs on the row: the theme
                # is what was measured, the ticker is what you would act
                # on, and every other list already pairs them.
                _head = (f"{_f['side']}   {theme_label(_f['name'])}"
                         + (f"  ({_sym_f})" if _sym_f else "")
                         + f"   {_f['date']:%d %b %Y}"
                         + (f"   ({_ago}d ago)" if _ago is not None else "")
                         + ("   ◆ HIGH CONVICTION" if _hc_f else ""))
                _wasf = st.session_state.get(_OPEN) == _fid
                with st.container(key=f"rfrow_{key_prefix}_{_fid}"):
                    _hitf = st.button(("▼  " if _wasf else "▶  ") + _head,
                                      key=f"btn_{key_prefix}_{_fid}",
                                      width="stretch")
                if _hitf:
                    st.session_state[_OPEN] = None if _wasf else _fid
                    st.rerun()
                if st.session_state.get(_OPEN) != _fid:
                    continue
                if _hc_f:
                    st.markdown(
                        f"<span class='rf-hiconv' style='border-color:"
                        f"{_tone};color:{_tone}'>◆ HIGH CONVICTION</span>",
                        unsafe_allow_html=True)
                    st.markdown(
                        "<div style='background:rgba(28,131,225,0.10);"
                        "border-radius:8px;padding:12px 16px'>"
                        "<b>◆ HIGH CONVICTION.</b> The technical screen "
                        "(price and moving averages) agreed with this "
                        "signal on the day it fired — <b>Jonathan's "
                        "momentum framework</b>, five factors behind "
                        "the (?)</div>",
                        unsafe_allow_html=True, help=HIGH_CONV_HELP)
                draw_chart(_f["name"], "",
                           f"{key_prefix}_fired_{_f['name']}_"
                           f"{_f['date']:%Y%m%d}")

        st.markdown('<div class="rf-rule"></div>', unsafe_allow_html=True)
        st.toggle(
            "Show the full list - every tracked name and the side the "
            "desk is watching today", key="show_full_list", value=True,
            help="The complete radar: every instrument in the store, "
                 "ranked by how close it is to its trigger, whether or "
                 "not it is near one.")
        return

    # ---- LOOK UP ANY NAME (type to search) --------------------------
    # Themes are shown WITH their tradeable anchor ETF (requirement
    # 2026-07-31) - the instrument comes straight from config/theme_etfs.csv
    # via THEME_ETFS, so fixing a mapping there fixes every dropdown at once.
    all_names = sorted(ek["name"].unique())
    pick = st.selectbox(
        f"look up any {kind_label.lower()} (type to search - shows its "
        "crowd heat whether or not it ever alerted)",
        ["(none)"] + all_names, key=f"{key_prefix}_lookup",
        format_func=lambda n: n if n == "(none)" else flag_label(n, kind))
    if pick and pick != "(none)":
        draw_chart(pick, "LOOKUP: ", f"{key_prefix}_lookup_chart")

    # ---- WHICH CHARTS, IN WHICH ORDER (recorded decision: "sort
    # the charts ... by either share of total mentions (which tickers have
    # the most chatter or retail attention at the moment) or by recent
    # get out / get in flags (as it is now)").
    #
    # Two orderings, two different questions:
    #   * RECENT SIGNALS - "what is alerting?"  Only names with a coherent
    #     alert inside the window are charted, newest first (the original
    #     behaviour, unchanged).
    #   * RETAIL ATTENTION - "where is the crowd RIGHT NOW?"  Every name
    #     is ranked by its share of the universe's total mentions over the
    #     trailing 7 days (the house ROLL window, same as the A1 hype
    #     gate's numerator), alerted or not - a name can dominate chatter
    #     without a flag, and that absence is itself the answer.  The
    #     share is stated in each chart's header so the ordering is
    #     readable, not inferred.
    last_alert = {}
    for name, (co, ct) in coherent.items():
        in_win = [d for d in co + ct
                  if lo <= d and (hi is None or d <= hi)]
        if in_win:
            last_alert[name] = max(in_win)

    sort_mode = st.radio(
        "order charts by", ("recent INCREASE EXPOSURE / CUT EXPOSURE signals",
                            "closest to firing (the watchlist)",
                            "share of total mentions (retail attention)"),
        horizontal=True, key=f"{key_prefix}_sort",
        help="RECENT SIGNALS charts only the names that actually alerted "
             "in the window, newest signal first. CLOSEST TO FIRING "
             "ranks every name by how far its score sits below the "
             "trigger right now, and which way it is moving - the names "
             "to watch before they alert. SHARE OF MENTIONS charts the "
             "names the crowd is talking about most - each name's share "
             "of the universe's total mentions over the last 7 days - "
             "whether or not it ever alerted.")

    if sort_mode.startswith("closest"):
        # THE WATCHLIST (recorded decision: "which ones are
        # closest / eligible to fire, how close, and in what
        # direction - the ones I should be watching").
        #
        # No new modelling: this reads the SAME stored scores and the
        # SAME frozen cuts the alerts themselves use, as of the day the
        # slider is on. Three facts per name, per side:
        #   GAP        cut - score, in score points (0 = at the trigger)
        #   DIRECTION  the score's 21-day change (rising = approaching)
        #   ELIGIBLE   does the phase gate allow this side to fire at
        #              all today? (GET OUT needs the name to have
        #              boomed; GET IN needs it NOT to have boomed) - an
        #              ineligible side can NEVER fire however high its
        #              score, which is exactly what a watcher must know.
        _as_of = dk["date"].max() if hi is None else min(
            hi, dk["date"].max())
        _prev = _as_of - pd.Timedelta(days=EUPHORIA_COOLDOWN_DAYS_DISP)
        _rows_w = []
        for _n, _g in dk[dk["date"] <= _as_of].groupby("name"):
            _g = _g.sort_values("date")
            if _g.empty:
                continue
            _cur = _g.iloc[-1]
            _old = _g[_g["date"] <= _prev]
            _boomed = watch_gate(_cur)
            # INFLECTION joins the watchlist as a third SIDE (desk
            # requirement: rank by proximity to an
            # inflection). It is always eligible - the inflection head has no
            # phase gate, deliberately - and it stays labelled as
            # context wherever it is rendered, because a row in a
            # watchlist is the furthest this signal is allowed to go.
            _infl_thr = ((desk_report or {}).get("inflection") or {}).get(
                "threshold")
            # GET IN eligibility follows the gate checkbox (2026-08-17:
            # ungated by default, so a boomed name's GET IN CAN fire
            # unless the stricter price-gated variant is ticked)
            for _side, _sc_col, _thr, _elig in (
                    ("CUT EXPOSURE", OUT_SCORE, _thr_out_d,
                     _boomed or XP_TRIGGER),
                    ("INCREASE EXPOSURE", IN_SCORE, _thr_in_d,
                     (not _boomed) or XP_TRIGGER or not GATED_GET_IN),
                    ("INFLECTION (context)", "inflection_score", _infl_thr, True)):
                if _sc_col not in _g.columns:
                    continue
                _sc = _cur.get(_sc_col)
                if _thr is None or _sc is None or pd.isna(_sc):
                    continue
                _d21 = (float(_sc) - float(_old.iloc[-1][_sc_col])
                        if len(_old) and pd.notna(_old.iloc[-1][_sc_col])
                        else None)
                _rows_w.append({
                    "name": _n, "symbol": _cur.get("symbol") or "-",
                    "side": _side,
                    "score": float(_sc), "trigger": float(_thr),
                    "gap": float(_thr) - float(_sc),
                    "d21": _d21, "eligible": _elig})
        _watch = pd.DataFrame(_rows_w)
        if _watch.empty:
            st.info("no scored names as of this day")
            _watch = pd.DataFrame(columns=["name", "side", "gap"])
        else:
            # rank: eligible first, then smallest gap to its trigger.
            # A name already OVER its trigger (gap <= 0) is on the
            # verge / just fired - it sorts to the very top.
            # ONE TABLE, ONE ORDERING (final: the two
            # "closest to..." options produced nearly the same table
            # from the same rows and only differed in which side sorted
            # first - two ways to ask one question). INFLECTION is now a
            # COLUMN rather than a competing sort: the ranking stays on
            # the two real CALLS, and each row also says how close that
            # name is to an inflection. An inflection cannot out-rank a call, which
            # is right - it is context, and context should not decide
            # what you look at first.
            _watch["_is_infl"] = _watch["side"].str.startswith("INFLECTION")
            _infl_gap = dict(zip(_watch.loc[_watch["_is_infl"], "name"],
                                 _watch.loc[_watch["_is_infl"], "gap"]))
            _calls = _watch[~_watch["_is_infl"]]
            if _calls.empty:                     # inflection-only store
                _calls = _watch
            _watch = _calls.sort_values(["eligible", "gap"],
                                        ascending=[False, True])
            _best = _watch.drop_duplicates("name")

            def _arrow(v):
                if v is None or pd.isna(v):
                    return "·  flat"
                if v > 0.02:
                    return "▲ approaching"
                if v < -0.02:
                    return "▼ receding"
                return "·  flat"

            _disp = pd.DataFrame({
                "name": _best["name"].map(theme_label),
                "ticker": _best["symbol"],
                "watch for": _best["side"],
                "how close": [
                    ("AT / OVER the trigger" if g <= 0
                     else f"{g:.2f} below the trigger")
                    for g in _best["gap"]],
                "% of the way there": [
                    f"{min(s / t, 1.0):.0%}" if t else "-"
                    for s, t in zip(_best["score"], _best["trigger"])],
                "direction (21d)": [_arrow(v) for v in _best["d21"]],
                "can it fire today?": [
                    "yes" if e else "no - wrong phase"
                    for e in _best["eligible"]],
                "INFLECTION (context)": [
                    ("-" if _infl_gap.get(n) is None
                     else ("AT / OVER" if _infl_gap[n] <= 0
                           else f"{_infl_gap[n]:.2f} away"))
                    for n in _best["name"]],
            })
            st.markdown(
                f"**The watchlist — {kind_label.lower()} ranked by how "
                f"close they are to their next call**, as of "
                f"{pd.Timestamp(_as_of):%d %b %Y}. *Can it fire today* "
                "is the phase gate: a CUT EXPOSURE only exists once a name "
                "has boomed, a INCREASE EXPOSURE only before it has — so a name "
                "in the wrong phase cannot fire whatever its score. "
                "The last column is how close that name is to an **INFLECTION "
                "— context, not a call**: a reversal is more likely "
                "than usual, in neither direction in particular. It "
                "does not affect the ranking, because context should "
                "not decide what you look at first.")
            st.dataframe(_disp.head(max(how_many, 10)), hide_index=True,
                         width="stretch")
        show = [n for n in _best["name"].tolist()][:how_many] \
            if not _watch.empty else []
        _gap_by = dict(zip(_best["name"], _best["gap"])) \
            if not _watch.empty else {}
        _side_by = dict(zip(_best["name"], _best["side"])) \
            if not _watch.empty else {}
        _sym_by_w = dict(zip(_best["name"], _best["symbol"])) \
            if not _watch.empty else {}
        for i, name in enumerate(show, 1):
            if name == pick:
                continue
            _g = _gap_by.get(name, float("nan"))
            _lbl = ("AT trigger" if _g <= 0 else f"{_g:.2f} to go")
            _sym_w = _sym_by_w.get(name)
            _tag = f" ({_sym_w})" if _sym_w and _sym_w != "-" else ""
            draw_chart(name,
                       f"#{i}{_tag} · {_side_by.get(name, '')} {_lbl}  ",
                       f"{key_prefix}_{name}")
    elif sort_mode.startswith("share"):
        # trailing-7d mention share of the euphoria universe, from the
        # SAME aggregates the pipeline builds (theme or ticker counts).
        _cnts = theme_counts if kind == "theme" else ticker_counts
        _ecol = "theme" if kind == "theme" else "ticker"
        share_by = {}
        if _cnts is not None and len(_cnts):
            _cw = _cnts[_cnts[_ecol].isin(set(ek["name"].unique()))]
            _hi_d = _cw["date"].max() if hi is None else min(
                hi, _cw["date"].max())
            _cw = _cw[(_cw["date"] > _hi_d - pd.Timedelta(days=ROLL))
                      & (_cw["date"] <= _hi_d)]
            _tot_m = _cw["mention_count"].sum()
            if _tot_m > 0:
                share_by = (_cw.groupby(_ecol)["mention_count"].sum()
                            / _tot_m).to_dict()
        show = sorted(share_by, key=share_by.get,
                      reverse=True)[:how_many]
        if not show:
            st.info("no mention data in the selected window")
        else:
            st.markdown(
                f"**Top {len(show)} {kind_label.lower()} by share of "
                f"the crowd's chatter over the last {ROLL} days** "
                f"(of {len(share_by)} with any mentions). Charted "
                "loudest first, alerted or not - a name can dominate "
                "chatter without a flag, and that absence is itself "
                "information.")
        for i, name in enumerate(show, 1):
            if name == pick:
                continue           # already drawn by the lookup
            draw_chart(name,
                       f"#{i} · {share_by[name]:.1%} of chatter  ",
                       f"{key_prefix}_{name}")
    else:
        # ---- charts: EVERY instrument with a (coherent) euphoria alert
        # inside the selected window, newest alert first - no filler names
        show = sorted(last_alert, key=last_alert.get,
                      reverse=True)[:how_many]
        if not show:
            st.info(f"no crowd heat alerts among {kind_label.lower()} in "
                    "the selected window - widen the window in the "
                    "sidebar to see past episodes")
        else:
            # NO SILENT CAPS.  Coverage reported separately from alerting
            # (design review: "why does euphoria singles only
            # show 3 graphs?") - the denominator is names WATCHABLE in the
            # window, and names whose history ends before it are counted
            # out loud with the fix attached.
            _tot = len(ek["name"].unique())
            _present = len(ew["name"].unique()) if ew is not None else _tot
            _msg = (f"**{len(last_alert)} of {_present} "
                    f"{kind_label.lower()} with data in this window "
                    f"alerted** ({lo.date()} to "
                    f"{'newest' if hi is None else hi.date()}). "
                    "Only names that actually alerted are charted - "
                    "newest signal first, no filler.")
            if _tot > _present:
                _gone = sorted(set(ek["name"].unique())
                               - set(ew["name"].unique()))
                _last = (ek[ek["name"].isin(_gone)]
                         .groupby("name")["date"].max().sort_values())
                _msg += (f" A further **{_tot - _present} "
                         f"{kind_label.lower()} have history that ends "
                         f"before this window** and so cannot appear: "
                         f"{', '.join(theme_label(n) for n in _last.index[-3:])}"
                         f" and {max(0, len(_gone) - 3)} others, latest "
                         f"data {_last.max().date()}. Widen the date "
                         "window in the sidebar to see their episodes.")
            if len(show) < len(last_alert):
                _msg += (f" Showing the {len(show)} most recent; raise "
                         "\"items per section\" in the sidebar to see "
                         f"the other {len(last_alert) - len(show)}.")
            st.markdown(_msg)
        for i, name in enumerate(show, 1):
            if name == pick:
                continue           # already drawn by the lookup
            draw_chart(name, f"#{i}  ", f"{key_prefix}_{name}")

    # ---- how to read the charts (BEHAVIOUR ONLY - no performance record).
    #
    # A "Validated record" caption used to close this tab: capture rate,
    # median warning, FA/instrument-year and AP against baseline for both
    # directions, read live from `desk_report` / `euph_report`, plus the
    # danger-state cliff comparison with its confidence interval.  Removed on
    # the desk's instruction (2026-07-28), same call as the scorecard strip
    # and the "measured version" block in the explainer: performance belongs
    # to the notebooks, this terminal shows conclusions.
    #
    # What is kept below is the part a reader needs to interpret what is
    # ON the chart - what the amber band is, why a START can be missing next
    # to an END, and why a fresh alert has no verdict yet.  Those are
    # behavioural rules, not claims about accuracy: they describe what the
    # detector DOES, and they stay true whatever the next re-measurement
    # says.  `desk_report` / `euph_report` are still loaded and still feed
    # the alert lists; only their scoring fields go unread here.
    st.caption("How to read these charts. AMBER band = the DANGER STATE: "
               "the crowd is at least twice its own normal AND the price "
               "is in a confirmed boom. The band is the standing warning; "
               "the alert line times the peak inside it. A START inside "
               "21 days of an END is suppressed as contradictory - so a "
               "lone END with no START before it is expected, not missing "
               "data. A fast START then END is a violent mania and the red "
               "risk signal is never suppressed. Recent alerts read "
               "PENDING until 45 days of price exists to judge them. The "
               "measured record - walk-forward tables, ablation, ML "
               "challenger, tournament - is in notebooks 00-05, the "
               "presentation pack (07) and "
               "docs/DECISIONS.xlsx, deliberately not here.")


if active_tab == MAIN_TAB:
    render_euphoria_tab("theme", "Themes", "euphth", mode="action")
# The single-name tab was removed from the display. render_euphoria_tab
# still takes kind="single" and is exercised by the tests, so restoring
# it is a one-line change if single names are ever shown again.

# ---- ETF RADAR (recorded decision: "add to the dashboard a table
# (tab) of all the ETFs we have and their score currently (even better
# if its the continuous graph over time) of the retail attention. order
# it by whats the closest to a get out") ----
# One row per tracked name, EVERY name in the desk store - no alert
# filter, no window filter: the whole point of this table is the names
# that have NOT called yet. Reads the same stored scores and the same
# frozen cuts the alerts use (respecting the trigger switch in the
# sidebar); computes nothing new. The attention sparkline is hype_raw -
# the mentions-based heat the euphoria measurements are built from - so
# the "graph over time" column and the score column come from the same
# store and can never disagree about what the crowd was doing.
# The radar is no longer its own tab: it is the "show the full list"
# panel underneath the landing page, driven by the toggle the action
# view writes into session_state. Same code, same table, one less tab.
if active_tab == MAIN_TAB and st.session_state.get("show_full_list"):
    if desk is None or not len(desk):
        # Every other consumer guards this; without it the tab raised
        # TypeError on a clone that has fetched but not yet run the
        # analytics pass, blanking the page instead of explaining.
        st.info("No scored store yet - the ETF radar appears once "
                "the analytics pass has run.")
        st.stop()
    _as_of_r = desk["date"].max() if hi is None else min(
        hi, desk["date"].max())
    st.subheader("ETF radar — every tradeable theme ETF, ranked by how "
                 "close it is to a CUT EXPOSURE")
    _thr_out_r = sig_thr(sig_head(desk_report, "get_out"))
    _thr_in_r = sig_thr(sig_head(desk_report, "get_in"))
    st.caption(
        f"As of **{pd.Timestamp(_as_of_r):%d %b %Y}** · trigger: "
        f"**{'experimental (posts only)' if XP_TRIGGER else 'shipped (crowd + price)'}**. "
        "One SIGNED readiness per name - red negative means the name "
        "has boomed and CUT EXPOSURE is the live side (-100% = at the "
        "trigger), green positive means INCREASE EXPOSURE is the live side. Most "
        "bearish sorts first, so the top of the table is what to look "
        "at; names whose last scoring run is over 60 days old sort "
        "last and are marked *stale*. *Retail attention* is the crowd "
        "heat the scores are built from: the bar is today against the "
        "name's own last year; the sparkline is the last six months.")
    # THEME ETFs ONLY - the tab exists to rank the tradeable theme
    # instruments. This filter predates the removal of the single-name
    # display and is kept regardless: the radar is about what can be
    # traded through an anchor ETF.
    _dkr = _hide(desk[(desk["date"] <= _as_of_r)
                      & (desk["kind"] == "theme")])
    def _c28(nm):
        """This theme's tagged posts in the trailing 28 days - the number
        the measurability floor is compared against."""
        if theme_counts is None or not len(theme_counts):
            return 0
        t = theme_counts[theme_counts["theme"] == nm]
        if not len(t):
            return 0
        cut28 = pd.to_datetime(_as_of_r) - pd.Timedelta(days=28)
        return int(t[pd.to_datetime(t["date"]) > cut28]
                   ["mention_count"].sum())

    _rows_r = []
    for _n, _g in _dkr.groupby("name"):
        _g = _g.sort_values("date")
        _cur = _g.iloc[-1]
        # the latest SCORED row: live ingestion appends heat days ahead
        # of the scoring pass, so the newest row can hold NaN scores -
        # a blank cell would read as "no signal" when the truth is
        # "signal as of the last scoring run".
        _gs = _g.dropna(subset=[OUT_SCORE]) if OUT_SCORE in _g else _g.iloc[:0]
        _out_sc = float(_gs.iloc[-1][OUT_SCORE]) if len(_gs) else None
        _out_day = _gs.iloc[-1]["date"] if len(_gs) else None
        _gi = _g.dropna(subset=[IN_SCORE]) if IN_SCORE in _g else _g.iloc[:0]
        _in_sc = float(_gi.iloc[-1][IN_SCORE]) if len(_gi) else None
        _h = _g.set_index("date")["hype_raw"].dropna()
        if not len(_h):
            continue
        _h1y = _h.loc[_h.index >= _h.index.max() - pd.Timedelta(days=365)]
        _att_pct = float((_h1y <= _h1y.iloc[-1]).mean()) if len(_h1y) else None
        _h6m = _h.loc[_h.index >= _h.index.max() - pd.Timedelta(days=183)]
        _spark = (_h6m.resample("3D").mean().dropna().round(3).tolist()
                  if len(_h6m) > 3 else None)
        _es_r = eligible_scored_now(_g)
        _boomed = _es_r[1] if _es_r is not None \
            else watch_gate(_cur)
        if _es_r is not None:
            # magnitude and freshness from the SAME row the side came
            # from. The per-column "last scored row" values happen to
            # coincide today (every scored row carries both sides), but
            # the moment the pipeline writes one-sided rows they drift:
            # a wrong-day score under the right side label, and fresh
            # IN-side names marked quiet because freshness was keyed to
            # the OUT column's day.
            _er_row = _es_r[0]
            _out_day = _er_row["date"]
            if OUT_SCORE in _er_row.index and pd.notna(_er_row[OUT_SCORE]):
                _out_sc = float(_er_row[OUT_SCORE])
            if IN_SCORE in _er_row.index and pd.notna(_er_row[IN_SCORE]):
                _in_sc = float(_er_row[IN_SCORE])
        _elig = _boomed or XP_TRIGGER
        # FRESHNESS OUTRANKS SIZE. Some names' newest scored row is
        # years old (a name can drop out of the scored universe and
        # keep ingesting heat); ranking a 2021 score above today's
        # would put the stalest data at the top of a table whose whole
        # job is "what should I look at NOW". Stale = last scoring run
        # for this name is more than 60 days behind the as-of day.
        _fresh = (_out_day is not None
                  and (pd.Timestamp(_as_of_r)
                       - pd.Timestamp(_out_day)).days <= 60)
        # ONE SIGNED NUMBER PER NAME (notebook 08
        # §10.5, replacing the two % columns this table shipped with -
        # which could and did read 100%/100% on the same name). The
        # phase routing supplies the sign, so the reading can never
        # point both ways: negative red = the OUT side is live and this
        # is how close it stands to its cut, positive green = the IN
        # side likewise. -100 = at the GET OUT trigger.
        if _boomed:
            _sgn = (-100.0 * _out_sc / _thr_out_r
                    if (_out_sc is not None and _thr_out_r) else None)
        else:
            _sgn = (100.0 * _in_sc / _thr_in_r
                    if (_in_sc is not None and _thr_in_r) else None)
        _rows_r.append({
            "name": _n, "ticker": _cur.get("symbol") or "-",
            "kind": _cur.get("kind", "-"),
            "att_pct": None if _att_pct is None else round(100 * _att_pct),
            "spark": _spark,
            "signed": (None if _sgn is None
                       else round(max(-100.0, min(100.0, _sgn)))),
            "side": "CUT EXPOSURE" if _boomed else "INCREASE EXPOSURE",
            "fresh": _fresh,
            # "(stale)" said the data was old; the truth is the CROWD
            # went quiet - chatter fell under the measurability floor
            # and scoring paused. Say that, with the live 28-day count,
            # so the row reads as the tool working rather than broken.
            "scored": ("-" if _out_day is None
                       else (f"{pd.Timestamp(_out_day):%d %b %Y} · "
                             f"quiet since ({_c28(_n)}/"
                             f"{EUPHORIA_MIN_COVERAGE} posts, 28d)"
                             if not _fresh
                             else f"{pd.Timestamp(_out_day):%d %b}")),
        })
    _radar = pd.DataFrame(_rows_r)
    if _radar.empty:
        st.info("no scored names as of this day")
    else:
        _radar = _radar.sort_values(
            ["fresh", "signed", "att_pct"],
            ascending=[False, True, False], na_position="last")
        _disp_r = pd.DataFrame({
            "name": _radar["name"].map(theme_label), "ticker": _radar["ticker"],
            "signed readiness": _radar["signed"],
            "watch side today": _radar["side"],
            "retail attention (vs own year)": _radar["att_pct"],
            "attention, last 6 months": _radar["spark"],
            "scored as of": _radar["scored"],
        })
        st.dataframe(
            _disp_r, hide_index=True, width="stretch",
            height=min(38 * (len(_disp_r) + 1) + 4, 1200),
            column_config={
                "signed readiness": st.column_config.ProgressColumn(
                    "signed readiness", format="%d%%",
                    min_value=-100, max_value=100,
                    help="One number per name. NEGATIVE: the name has "
                         "boomed, so CUT EXPOSURE is the side that can "
                         "fire, and this is how close its score "
                         "stands to the frozen cut (-100 = at the "
                         "CUT EXPOSURE trigger). POSITIVE: the same "
                         "reading for INCREASE EXPOSURE. The sign comes from the "
                         "phase routing, so a name can never read "
                         "toward-GET-IN and toward-GET-OUT at once. "
                         "Most bearish sorts first."),
                "retail attention (vs own year)":
                    st.column_config.ProgressColumn(
                        "retail attention (vs own year)", format="%d%%",
                        min_value=0, max_value=100,
                        help="Today's mentions-based heat as a "
                             "percentile of this name's own trailing "
                             "year - 90% = louder than 90% of its own "
                             "last twelve months."),
                "attention, last 6 months":
                    st.column_config.LineChartColumn(
                        "attention, last 6 months",
                        help="hype_raw, 3-day means - the continuous "
                             "view of the same heat the bar "
                             "summarises."),
            })
        st.caption(
            "Most bearish first: the top of the table is the names "
            "closest to a CUT EXPOSURE. Scores update on the analytics "
            "pass (*scored as of*), attention updates with live "
            "ingestion - a hot sparkline with a stale score means the "
            "pipeline should be re-run, and the [dev] Data Stats tab "
            "will say so.")

# ---- INFLUENCE TRACKER (committed text-free store, extended live) ----
# INFORMATION ONLY. Nothing on this tab feeds the euphoria level or the
# GET IN / GET OUT alerts - that is a selection rule, and notebook 05 is the
# reason for it: the influence model does NOT generalise to authors it has
# not seen (cohort-split AP sits at or below the random floor), so it is a
# research exhibit, never a live input. The RANKING shown here is the
# MEASURED composite from the store, not a model prediction.
def influence_simple():
    """Plain-English version of the influence tab.

    The tab's own numbers (12,528 authors, 33,451 calls) are quoted from the
    notebook export rather than typed, for the same staleness reason as the
    crowd heat panel - this store grows on every live run, so any hard-typed
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
flagged here as the list to fade. It is the single most actionable thing
on this tab.

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

- This tab does **not** feed the crowd heat signal. It is information, not a
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

**How the score is built.**
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

**Why no per-author hit rate is displayed.** It is
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

**"Loud but wrong"** is the headline warning of this tab: the accounts in
the top quartile of reply-graph PageRank but below median influence. They are the accounts a *follow-the-big-names* desk would copy,
and the evidence says fade them.

**The time panels** (added 2026-07-27) are the dimension this tab lacked
entirely. `calls.parquet` has carried a date on every call since inception,
yet every exhibit pooled the whole window into one snapshot - so the tab could
say "GME is the most-backed name" but never "GME's backing has tripled in
three weeks", which is the statement a PM can act on. Both panels are weekly,
not daily: the comment fetch runs about twice a week under the ingestion
budget (the parameter register (docs/RESEARCH_RECORD.md §7, Class 7)), so a daily axis would largely plot the
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
        # HALF-MASKED here, and the width is measured on the masked string:
        # this function's whole job is to stop labels overlapping, and it
        # can only do that if the length it reserves is the length that
        # actually prints. Masking downstream of the collision maths would
        # reserve space for a name nobody sees.
        _lbl = half_mask(str(row.author))
        half_w = 0.5 * len(_lbl) * _LBL_PX_PER_CHAR * scale
        for vx, vy in ((ux, uy), (-uy, ux), (uy, -ux), (-ux, -uy)):
            lx, ly = row.x + vx * lead, row.y + vy * lead
            box = (lx - half_w, lx + half_w, ly - half_h, ly + half_h)
            if not free(box):
                continue
            boxes.append(box)
            out.append(dict(
                x=float(row.x), y=float(row.y), text=_lbl,
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
        # `shown` is a DISPLAY column added next to `author`, never a
        # replacement for it: `author` is still the key that positions,
        # `centre` lookups and the ego join all run on.
        customdata=n.assign(shown=half_mask_series(n["author"]))[
            ["shown", "influence", "degree_here", "community"]].to_numpy(),
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
                 gap_x_px: float = 30.0, gap_y_px: float = 13.0,
                 label_w_px=None) -> list[bool]:
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

    `label_w_px` is the same rule when the labels are NOT all one width.  The
    scalar `gap_x_px` is only correct because every ticker is about four
    characters wide; a theme slug is not ("Robotics automation" is nineteen),
    so passing that a single average width either lets long labels print
    through each other or suppresses short ones that had room.  Labels are
    drawn CENTRED on the point, so two of them collide when the gap between
    their centres is less than the mean of their widths - and with every
    width equal to `gap_x_px` that expression IS the scalar rule, so the
    ticker view is unchanged by construction rather than by re-tuning.

    Greedy in descending y, because height is the point of the chart: the
    most-backed name always keeps its label and the tail yields to it.
    """
    pts = [(float(y), float(x), i) for i, (x, y) in enumerate(zip(xs, ys))]
    widths = ([float(w) for w in label_w_px] if label_w_px is not None
              else [float(gap_x_px)] * len(pts))
    # Tallest first, and on an exact tie the EARLIER row wins - the caller
    # passes the digest already ranked by backing, so ties resolve toward the
    # better-backed name rather than toward whichever happened to be last.
    pts.sort(key=lambda t: (-t[0], t[2]))
    keep = [False] * len(pts)
    placed: list[tuple[float, float, float]] = []   # (x, y, width) drawn
    to_x = (x_span / w_px) if w_px else 0.0
    dy = gap_y_px * (y_span / h_px) if h_px else 0.0
    for y, x, i in pts:
        wi = widths[i]
        if any(abs(x - px) < (wi + pw) / 2.0 * to_x and abs(y - py) < dy
               for px, py, pw in placed):
            continue
        keep[i] = True
        placed.append((x, y, wi))
    return keep


def fig_influence_bubbles(dig: pd.DataFrame, voices: pd.DataFrame,
                          title: str, top_n: int | None = None,
                          key: str = "ticker"):
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
    # The chart is key-agnostic: TICKER and THEME are the same four readings
    # over a different grouping, so one function draws both and the two views
    # cannot drift apart.  Only the label spelling differs - a ticker is
    # already a display string, a theme slug is not (see theme_label).
    _label = theme_label if key == "theme" else (lambda s: s)
    d = dig.dropna(subset=["consensus"]).copy()
    if voices is not None and len(voices):
        d = d.merge(voices[[key, "voices"]], on=key, how="left")
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
    # The label widths are MEASURED off the labels actually being drawn, not
    # assumed.  _thin_labels' 30px default is the width of a four-character
    # ticker, which is every ticker; a theme label is up to nineteen
    # characters, so reusing that number would reproduce the exact overlap
    # the function exists to prevent.  Per character is read straight off the
    # accepted default (30px / 4 chars) rather than introduced as a new
    # constant, so the ticker view keeps the geometry it was tuned with.
    _texts = [str(t) for t in d[key].map(_label)]
    _px_per_char = 30.0 / 4.0
    _room = _thin_labels(d["consensus"], d["backing"],
                         x_span=2.36, y_span=_ymax_est,
                         w_px=880.0, h_px=398.0,
                         label_w_px=[_px_per_char * len(t) for t in _texts])
    fig = go.Figure(go.Scatter(
        x=d["consensus"], y=d["backing"], mode="markers+text",
        text=[t if (b >= even and r) else ""
              for t, b, r in zip(d[key].map(_label), d["backing"], _room)],
        textposition="top center",
        textfont=dict(size=10, color=INK),
        marker=dict(
            size=d["n_calls"], sizemode="area",
            sizeref=2.0 * smax / (44.0 ** 2), sizemin=6,
            color=[BULL if c >= 0 else BEAR for c in d["consensus"]],
            opacity=0.72, line=dict(width=1, color=WHITE)),
        customdata=d.assign(**{key: d[key].map(_label)})[
            [key, "n_authors", "n_calls", "longs", "shorts",
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
    # X-RANGE FITTED TO THE DATA, not fixed at the full [-1, 1].
    #
    # Consensus is bounded at -1..+1, so a fixed range was the obvious
    # choice - but on a day when every name is net long, every bubble
    # stacks against the right edge and two thirds of the plot is blank.
    # The chart then reads as broken rather than as unanimous.
    #
    # So the window follows the points, with three constraints that keep
    # it honest: ZERO IS ALWAYS IN VIEW, because "how far from
    # disagreement" is the whole question and a pane that cropped the
    # centre would flatter a one-sided day; a minimum half-width stops a
    # tight cluster being magnified into apparent spread; and the
    # padding leaves room for the labels above each bubble.
    _cx = pd.to_numeric(d["consensus"], errors="coerce").dropna()
    if len(_cx):
        _lo, _hi = min(0.0, float(_cx.min())), max(0.0, float(_cx.max()))
        _half = max((_hi - _lo) / 2.0, 0.35)          # never over-magnify
        _mid = (_hi + _lo) / 2.0
        _lo, _hi = _mid - _half, _mid + _half
        _pad = (_hi - _lo) * 0.16
        _xr = [max(-1.18, _lo - _pad), min(1.18, _hi + _pad)]
    else:
        _xr = [-1.18, 1.18]
    fig.update_xaxes(range=_xr, zeroline=False,
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
    the crowd heat detector exists to catch; a line rolling over is the crowd
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
        # the y axis is a CATEGORY axis of handles, so it is a display
        # surface: half-mask it. Two authors whose masked forms collided
        # would merge into one bar, so `half_mask_series` tags any label
        # that stands for more than one person.
        x=d["influence"], y=half_mask_series(d["author"]), orientation="h",
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


if active_tab == "Influence tracker":
    st.subheader("Influence tracker - who has actually been right, and "
                 "what they are saying now")
    st.caption("INFORMATION ONLY - nothing on this tab feeds the crowd heat "
               "level or the INCREASE EXPOSURE / CUT EXPOSURE alerts. Ranking is the "
               "MEASURED record from the store, not a model prediction.")
    with st.expander("full method: scoring, shrinkage and the graph slice "
                     "(research version)", expanded=False):
        st.markdown(INFLUENCE_HOW_TO_READ)

    if not os.path.exists(_INFL_SCORES) and not LOCAL_CONTROLS:
        # HOSTED COPY. The advice below ("run a live pull") is not
        # actionable here - the host fetches nothing and its disk is
        # rebuilt from the published bundle on every deploy. The board is
        # absent because tools/publish_dashboard.py leaves it out by
        # default: its frames are keyed by Reddit author handle, so
        # publishing it puts real usernames on a hosted page. Say that
        # plainly instead of sending a viewer to a command they cannot
        # run.
        st.info("The influence board is not part of the published data. "
                "It is keyed by individual Reddit usernames, so it is "
                "kept off the hosted copy by default and is available "
                "on the machines that run the pipeline.")
    elif not os.path.exists(_INFL_SCORES):
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

        # SUB-TABS, NOT ONE LONG SCROLL (recorded decision:
        # "influence tracker why is everything crowded long?").  The tab
        # carried six numbered sections stacked vertically - roughly six
        # screens - so the reader had to scroll past four exhibits to reach
        # the one they wanted, and the shared controls above scrolled out of
        # sight with them.  Nothing is removed: the same six sections are
        # grouped into four sub-tabs, each about one screen, and the panel
        # size / recency controls stay pinned above them so they still govern
        # every sub-tab.  Grouping is by QUESTION, not by section number:
        # what is being pushed / is it building / who the people are / how
        # they connect.
        # THE MAP LEADS. It is the exhibit that answers the tab's own
        # question at a glance - who is talking to whom - and it was
        # third of four, behind two charts that need reading. "The
        # names" was removed with it: the leaderboard and the
        # per-name breakdown were a directory, not a decision.
        # "What they are pushing" was removed - the bubble chart was the
        # exhibit and it did not carry its weight. Two views remain: who
        # talks to whom, and whether the crowding is building.
        _i4, _i2 = st.tabs(["The map", "Building or fading?"])
        with _i2:
            # ---- 2. IS IT BUILDING OR FADING? (added 2026-07-27) -----------
            # The tab had no time axis at all, which made every reading on it a
            # still photograph. The store has held call dates since inception.
            st.markdown("#### 2. Is the crowding building, or fading?")
            if calls is None or not len(dig) or len(tilt_hist) < 2:
                st.info("not enough history inside this window to plot a "
                        "trend - widen the window above to 90 days, or run "
                        "another live pull to extend the store.")
            else:
                # Called ONCE without a ticker filter, then filtered here: the
                # leading names are taken from this same population's own
                # totals rather than from `dig` (which is the top-N panel), so
                # the lines and the shares they are drawn as cannot come from
                # two different rooms. One pass over the calls either way.
                _crowd_all = ig.crowding_history(
                    calls, board, authors=recorded, days=days, asof=asof)
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
                    "**Read it in three steps.**\n\n1. **The top line** is "
                    "the room on one axis: +1 means every voice with a record "
                    "was long, with full conviction, that week; -1 is the "
                    "same on the short side; the dotted line is a real "
                    "two-way argument. Over this window it went from "
                    f"**{_t_then:+.2f}** to **{_t_now:+.2f}** - the room got "
                    f"{_dir_word}. **The dot size is how many calls that week "
                    "rests on**, and it matters: the comment budget leaves "
                    "the weeks very unequal, so a tiny dot at +1.00 is four "
                    "people agreeing, not the market.\n2. **The lines below** "
                    "are the leading names, each drawn as the **share of that "
                    "week's conviction** it took. Sharing inside the week "
                    "matters: raw totals rise and fall with how busy the "
                    "forum was, so an un-normalised line would show you the "
                    "posting calendar. The grey dotted line is that week's "
                    "**even split** - it moves, because a week where the room "
                    "touched 14 names splits differently from one where it "
                    "touched 200. Green = the name's latest reading is net "
                    "long, brick = net short, and each name is written at the "
                    "end of its own line.\n3. **Both panels use all "
                    f"{len(board):,} voices with a record**, not the top "
                    f"{panel_n} in the slider. Cut to the top few, a single "
                    "week can come down to one person on one name, which is "
                    "100% of that week by definition and a fact about "
                    "nobody.\n\n**So what.** A line climbing week after week "
                    "is people with a record piling into one name - the "
                    "build-up phase, and the useful time to hear about it. A "
                    "line that spikes and collapses is attention that has "
                    "already moved on, which matters just as much: the "
                    "position is still on the book but the story supporting "
                    "it has gone quiet. Read the top panel with it - crowding "
                    "into a single name while the whole room tilts to +1 is "
                    "the configuration that precedes the unwinds this project "
                    "exists to flag.")

        with _i4:
            # ---- 5. the influence map -------------------------------------
            st.markdown("#### 5. The influence map - who replies to whom",
                        help=(
                "**How to read the map:**\n\n"
                "- Every **dot** is one poster.\n"
                "- A **line** between two dots means one replies to the "
                "other; people who reply to each other a lot get pulled "
                "close together.\n"
                "- **Dot size** = how many people reply to them (the "
                "bigger, the more the room responds).\n"
                "- **Colour** = the influence score - how much engagement "
                "what they post actually gets.\n"
                "- Only the densely connected core of the network is "
                "drawn - the thousands of one-off posters around it are "
                "left out so the picture stays readable."))
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
                        f"Every person drawn has at least **{kcore} "
                        "neighbours inside this picture** - that is what a "
                        "k-core is, and it is why this is a principled slice "
                        f"of the {g_n:,}-node graph rather than a random "
                        "thinning. Dot size = how many people reply to them; "
                        "colour = influence score. Drawn over the people "
                        "with a judged record, so every dot has a real "
                        "colour. **Rank correlation between being central "
                        f"and being useful, in this picture: {_rho:+.2f}** - "
                        "being central is close to unrelated to being right, "
                        "which is exactly why the board is ranked on the "
                        "record and not on the graph. Two notes so the "
                        "picture is not over-read: only names far enough "
                        "apart to be legible are printed (the rest are on "
                        "hover), and the map covers everyone with any judged "
                        "call, which is a wider pool than the board above - "
                        f"the board needs {INFL_MIN_JUDGED}+ judged calls "
                        "before it will rank someone.")
                else:
                    # `format_func`, NOT a masked option list: the value this
                    # widget returns is the key `_ego_frames` looks the person
                    # up by, so the options must stay the true handles and only
                    # their rendering is masked. `_who` is the display form,
                    # used in every string a human reads below.
                    who = st.selectbox("author", panel, key="infl_ego_who",
                                       format_func=half_mask)
                    _who = half_mask(str(who))
                    with st.spinner("laying out the neighbourhood ..."):
                        nodes, links, e_n, e_m = _ego_frames(who, _e_mt, _b_mt)
                    if e_n <= 1:
                        st.info(f"{_who} has no reply links in the store - "
                                "they post, nobody replies (or the replies "
                                "are outside the fetched history).")
                    else:
                        st.plotly_chart(
                            fig_influence_map(
                                nodes, links,
                                f"{_who}: everyone they exchange replies with "
                                f"({e_n} people, {e_m} links)", centre=who),
                            width="stretch", key="infl_map_ego")
                        st.caption(f"One hop around {_who}. If the "
                                   "neighbourhood is larger than "
                                   f"{INFL_EGO_MAX} people the busiest "
                                   "neighbours are kept, so this shows the "
                                   "active part of it, not all of it.")

            # ---- 6. the two warning boards --------------------------------
            # Both boards keep their place: they are this tab's headline
            # finding and it rests on them. Both lost their
            # accuracy COLUMNS on 2026-07-27 - `composite` became the 0-100
            # influence index and `hit_rate` / `n_judged` came out - so the
            # boards now say WHO fits the profile and leave the measurement of
            # the profile to notebook 05, which is where it is defensible.
            _infl_all = ig.influence_index(board)
            st.markdown("#### 6. Two boards worth reading against the grain")
            _w1, _w2 = st.columns(2)
            with _w1:
                st.markdown("**Called the tops** - most confirmed bearish "
                            "calls inside a crowd heat peak window")
                if board["called_tops"].fillna(0).sum():
                    _ct = board.nlargest(10, "called_tops").copy()
                    _ct["influence"] = (_infl_all.reindex(_ct.index).round(0)
                                        .astype(int))
                    _ct["author"] = half_mask_series(_ct["author"])
                    st.dataframe(
                        _ct[["author", "influence", "called_tops",
                             "bought_tops", "latest_calls"]].rename(columns={
                                 "called_tops": "called tops",
                                 "bought_tops": "bought tops",
                                 "latest_calls": "latest calls"}),
                        width="stretch", hide_index=True)
                    st.caption("The people who were bearish INTO a crowd heat "
                               "peak that then busted. Rare by construction - "
                               "most of the forum is long into a top.")
                else:
                    st.caption("none recorded yet - this grows as peak "
                               "windows overlap the call history")
            with _w2:
                st.markdown("**Loud but wrong** - top-quartile reply-graph "
                            "PageRank, below-median influence")
                if board["loud_but_wrong"].any():
                    _lw = board[board["loud_but_wrong"]].head(10).copy()
                    _lw["influence"] = (_infl_all.reindex(_lw.index).round(0)
                                        .astype(int))
                    _lw["author"] = half_mask_series(_lw["author"])
                    st.dataframe(
                        _lw[["author", "influence", "followers",
                             "latest_calls"]].rename(columns={
                                 "followers": "people replying to them",
                                 "latest_calls": "latest calls"}),
                        width="stretch", hide_index=True)
                    st.caption("The accounts a 'follow the big names' desk "
                               "would copy: lots of people reply to them, and "
                               "their measured record sits below the median "
                               "of this board. That is the profile to fade: "
                               "high reach on barely-above-chance accuracy.")
                else:
                    st.caption("nobody currently fits the profile")

# ---- TOP TRENDS ----
if active_tab == "Top trends":
    st.subheader("Most-mentioned themes (rank 1 = top trending)",
                 help=(
        "**How this ranking works:**\n\n"
        "- Themes are ranked by **how much of the conversation they took "
        "up** in the chosen window - total mentions, biggest first.\n"
        "- This is about SIZE: the themes everyone is already talking "
        "about.\n"
        "- For what is growing fastest instead, use the Emerging trends "
        "tab."))
    st.caption("Each chart is the PM view: the anchor "
               "ETF's price, the crowd's attention and the crowd's mood "
               "on ONE graph - price up + attention up + sentiment up "
               "reads in a single glance. Attention uses the "
               "coverage-robust share estimator, sentiment the 28-day "
               "post-weighted lean, so neither line carries pull-day "
               "noise.")
    _att_mode = _attention_mode_control("top_att_mode")
    _mood_mode = _mood_mode_control("top_mood_mode")
    top = (tc[~tc["theme"].isin(HIDDEN_THEMES)]
           .groupby("theme")["mention_count"].sum()
           .rename("total mentions").reset_index())
    top_r = ranked(top, "total mentions").head(how_many)
    # Display copy only: the slug stays the key every later lookup uses.
    st.dataframe(top_r.assign(theme=top_r["theme"].map(theme_label)),
                 width="content", hide_index=True)
    for i, theme in enumerate(top_r["theme"], 1):
        symbol = resolve_anchor(theme, priced)
        share, _albl, _aax, _zl = _attention_series(
            _att_mode, theme_counts, "theme", theme, lo, hi)
        sent, _mlbl, _mrel = _mood_series(
            _mood_mode, theme_sentiment, "theme", theme, lo, hi)
        px = (price_series(prices, symbol, lo, hi)
              if prices is not None and symbol else None)
        st.plotly_chart(fig_theme_pulse(
            share, sent, px, symbol,
            f"#{i}  {theme_label(theme)}  vs  "
            f"{symbol or 'no priced anchor'}",
            att_label=_albl, att_axis=_aax, zero_line=_zl,
            mood_label=_mlbl, mood_relative=_mrel),
            width="stretch", key=f"top_{theme}")

# ---- EMERGING TRENDS ----
if active_tab == "Emerging trends":
    st.subheader("Emerging = fastest-GROWING tradeable themes (rank 1 = hottest)",
                 help=(
        "**How this ranking works:**\n\n"
        "- Themes are ranked by how fast their **share of the "
        "conversation is growing** over the lookback - not by how big "
        "they are.\n"
        "- Rank 1 = the crowd is arriving fastest right now, even if "
        "the theme is still small.\n"
        "- Only themes with a tradeable ETF are ranked.\n"
        "- The slider trades speed for steadiness: 7 days catches the "
        "newest arrivals, 21 days rewards a sustained build-up."))
    st.caption("Only themes with an approved instrument are ranked. "
               "'Growing' = average change in share-of-conversation over the "
               "chosen lookback - positive means the crowd is arriving.")
    # the growth lookback is a knob: 7d catches the newest arrivals but is
    # twitchy; 21d rewards a SUSTAINED build-up and ignores one loud week
    look = st.slider("growth lookback (days)", 3, 30, 7, key="emerg_look")
    grow_col = f"avg change last {look}d (pp)"
    movers = []
    for theme in tc["theme"].unique():
        if theme not in THEME_ETFS or theme in HIDDEN_THEMES:
            continue            # tradeable, and not hidden from display
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
        st.dataframe(mv.assign(theme=mv["theme"].map(theme_label)),
                     width="content", hide_index=True)
        st.caption("Charts show the PM view: price + "
                   "attention + sentiment on one graph. The RANKING "
                   "above always uses attention growth over the chosen "
                   "lookback; the toggle below picks what the chart's "
                   "attention line shows.")
        _att_mode_e = _attention_mode_control("emerg_att_mode")
        _mood_mode_e = _mood_mode_control("emerg_mood_mode")
        for i, theme in enumerate(mv["theme"], 1):
            symbol = resolve_anchor(theme, priced)
            share, _albl, _aax, _zl = _attention_series(
                _att_mode_e, theme_counts, "theme", theme, lo, hi)
            sent, _mlbl, _mrel = _mood_series(
                _mood_mode_e, theme_sentiment, "theme", theme, lo, hi)
            px = (price_series(prices, symbol, lo, hi)
                  if prices is not None and symbol else None)
            fig = fig_theme_pulse(
                share, sent, px, symbol,
                f"#{i}  {theme_label(theme)}: the crowd arriving  vs  "
                f"{symbol or '-'}",
                att_label=_albl, att_axis=_aax, zero_line=_zl,
                mood_label=_mlbl, mood_relative=_mrel)
            # grey out everything the growth ranking does NOT look at
            if len(share.dropna()):
                focus = share.dropna().index.max() - pd.Timedelta(days=look)
                dim_outside(fig, lo, focus, f"ranking uses last {look}d →")
            st.plotly_chart(fig, width="stretch", key=f"emerg_{theme}")

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

# PULSE_IDEAS (the roadmap text) deleted 2026-08-05 with the expander
# that displayed it - it described sections that do not exist.
# ---- AI PULSE (LLM-written, via the Apollo gateway) ----

@st.cache_data(show_spinner=False)
def _market_read(day_ts, tc_m, ts_m, tk_m):
    """The market's mood on ONE day, computed from the stores.

    Requirement: "the slider at the top of the page so i
    can see what the market is feeling / its sentiment / bullishness
    score ... and the ability to go to a specific point in time".
    
    WHY THIS IS NOT THE LLM. The prose on this page needs a gateway call
    that takes ~30s and only runs on the desk machine, so it cannot move
    with a slider. But the FEELING is arithmetic - bullishness, breadth,
    attention, acceleration are all in the committed stores, for every
    day back to 2017. So the slider drives the numbers, instantly and
    over the whole history, and the model's words layer on top when a
    pulse has been written for that day.
    
    That split is also the project's standing rule (ARCHITECTURE §4.1):
    numbers come from the stores, words come from the model. The slider
    is that rule turned into an interaction.
    
    Cached on the three store mtimes, so it recomputes only when the
    data changes - not on every drag.
    """
    import pandas as _pd
    day = _pd.Timestamp(day_ts)
    out = {"day": day}
    tc = theme_counts
    if tc is not None and len(tc):
        t = tc[tc["theme"].isin(THEME_ETFS)].copy()
        t["date"] = _pd.to_datetime(t["date"])
        w = t[(t["date"] <= day) & (t["date"] > day - _pd.Timedelta(days=7))]
        prev = t[(t["date"] <= day - _pd.Timedelta(days=7))
                 & (t["date"] > day - _pd.Timedelta(days=35))]
        cur = w.groupby("theme")["mention_count"].sum()
        base = prev.groupby("theme")["mention_count"].sum() / 4.0
        if cur.sum():
            share = (cur / cur.sum()).sort_values(ascending=False)
            out["top"] = [(k, float(v),
                           float(cur[k] / base[k]) if base.get(k) else None)
                          for k, v in share.head(6).items()]
            movers = {k: float(cur[k] / base[k]) for k in cur.index
                      if base.get(k, 0) > 0 and cur[k] >= 20}
            out["movers"] = sorted(movers.items(), key=lambda kv: -kv[1])[:3]
            out["faders"] = sorted(movers.items(), key=lambda kv: kv[1])[:3]
    ts = theme_sentiment
    if ts is not None and len(ts):
        d = ts.copy()
        d["date"] = _pd.to_datetime(d["date"])
        r = d[(d["date"] <= day) & (d["date"] > day - _pd.Timedelta(days=7))]
        if len(r) and r["n_posts"].sum():
            nb = float((r["net_bullish"] * r["n_posts"]).sum()
                       / r["n_posts"].sum())
            out["net_bullish"] = nb
            # NO BULLISHNESS SCORE (recorded decision: "remove
            # the bullish score number and all the associated code").
            # This function used to rank the weekly net-bullish figure
            # against its own trailing two years and print it as a
            # 0-100 score; the requirement was for the number to go. The raw
            # net_bullish stays available to the snapshot text, and
            # breadth (below) remains the participation read.
            per = r.groupby("theme").apply(
                lambda g: (g["net_bullish"] * g["n_posts"]).sum()
                / max(g["n_posts"].sum(), 1), include_groups=False)
            per = per[per.index.isin(THEME_ETFS)]
            if len(per):
                out["breadth"] = float((per > 0).mean())
                out["most_bull"] = per.sort_values(ascending=False).head(3)
                out["most_bear"] = per.sort_values().head(3)
            out["n_posts"] = int(r["n_posts"].sum())
    tk = ticker_counts
    if tk is not None and len(tk):
        k = tk.copy()
        k["date"] = _pd.to_datetime(k["date"])
        r = k[(k["date"] <= day) & (k["date"] > day - _pd.Timedelta(days=7))]
        if len(r):
            out["loud"] = (r.groupby("ticker")["mention_count"].sum()
                           .sort_values(ascending=False).head(10))
    # the words the crowd had just picked up that week - "everyone is
    # suddenly talking about tariffs" is often the actual subject, ahead
    # of any ticker
    try:
        tm = _read(os.path.join(PROCESSED_DIR, "daily_term_counts.parquet"),
                   _mtime(os.path.join(PROCESSED_DIR,
                                       "daily_term_counts.parquet")))
        tm["date"] = _pd.to_datetime(tm["date"])
        cur = (tm[(tm["date"] <= day)
                  & (tm["date"] > day - _pd.Timedelta(days=7))]
               .groupby("term")["mention_count"].sum())
        prv = (tm[(tm["date"] <= day - _pd.Timedelta(days=7))
                  & (tm["date"] > day - _pd.Timedelta(days=35))]
               .groupby("term")["mention_count"].sum() / 4.0)
        sp = {t: cur[t] / prv[t] for t in cur.index
              if prv.get(t, 0) >= 3 and cur[t] >= 12}
        out["terms"] = sorted(sp.items(), key=lambda kv: -kv[1])[:5]
    except Exception:                                    # noqa: BLE001
        pass
    # what the detector said that week
    try:
        dk = desk.copy()
        dk["date"] = _pd.to_datetime(dk["date"])
        r = dk[(dk["date"] <= day)
               & (dk["date"] > day - _pd.Timedelta(days=21))]
        out["flag_in"] = sorted(set(
            r[r[sig_col("get_in", r)].astype(bool)]["name"]))
        out["flag_out"] = sorted(set(
            r[r[sig_col("get_out", r)].astype(bool)]["name"]))
    except Exception:                                    # noqa: BLE001
        pass
    return out


def _breadth_clause(br):
    """Breadth is the ABSOLUTE share of themes that are net bullish.
    Retail is structurally positive, so breadth is high almost always -
    the sentence has to say wide-or-narrow, not loud-or-quiet. (The
    intensity percentile this used to pair with was removed with the
    bullishness score; see docs/DECISIONS.md.)"""
    if br is None:
        return None
    if br >= 0.7:
        return (f"The lean was broad - **{br:.0%}** of tradeable themes "
                "were net bullish.")
    return (f"Only **{br:.0%}** of themes were net bullish - the mood "
            "was carried by a few.")


def _snapshot_text(mr):
    """A written snapshot of one week, composed FROM THE NUMBERS.

    Requirement: dragging the date slider must update the text, giving
    a market snapshot as of the selected day.

    The model's prose cannot do this - it is one stored document behind
    a ~30s gateway call. But the narrative a desk actually wants from a
    scrub is not literary: what was the crowd on, what was accelerating,
    how did it feel against its own normal, what was flagged. All of
    that is in the stores for every day since 2017, so it can be
    WRITTEN rather than retrieved, instantly, with no model involved.
    
    This is deliberately not trying to sound like the LLM. It is
    labelled as computed, it cites its own numbers inline, and every
    clause is checkable against the panel above it. The model's read
    stays the richer one for any day somebody generates deliberately -
    this is the one that moves with the slider."""
    if not mr:
        return None
    bits = []
    _bc = _breadth_clause(mr.get("breadth"))
    if _bc:
        bits.append(_bc)
    top = mr.get("top") or []
    if top:
        lead = top[0]
        rest = ", ".join(theme_label(t) for t, _sh, _r in top[1:4])
        bits.append(
            f"Attention was led by **{theme_label(lead[0])}** at "
            f"{lead[1]:.0%} of theme chatter"
            + (f" ({lead[2]:.2f}x its own 4-week average)"
               if lead[2] else "")
            + (f", with {rest} behind it." if rest else "."))
    mv = [(k, v) for k, v in (mr.get("movers") or []) if v >= 1.3]
    fd = [(k, v) for k, v in (mr.get("faders") or []) if v <= 0.75]
    if mv or fd:
        parts = []
        if mv:
            parts.append("accelerating into "
                         + ", ".join(f"**{theme_label(k)}** ({v:.1f}x)"
                                     for k, v in mv[:3]))
        if fd:
            parts.append("and out of "
                         + ", ".join(f"**{theme_label(k)}** ({v:.1f}x)"
                                     for k, v in fd[:3]))
        bits.append("Rotation was " + " ".join(parts) + ".")
    tms = [t for t, _r in (mr.get("terms") or [])][:4]
    if tms:
        bits.append("The words spreading that week: "
                    + ", ".join(f"*{t}*" for t in tms) + ".")
    loud = mr.get("loud")
    if loud is not None and len(loud):
        bits.append("Loudest names: "
                    + ", ".join(f"**{t}**" for t in loud.head(6).index)
                    + ".")
    fi, fo = mr.get("flag_in") or [], mr.get("flag_out") or []
    if fo:
        bits.append("The detector was calling **CUT EXPOSURE** on "
                    + ", ".join(theme_label(n) for n in fo[:4]) + ".")
    elif fi:
        bits.append("The detector was calling **INCREASE EXPOSURE** on "
                    + ", ".join(theme_label(n) for n in fi[:4]) + ".")
    else:
        bits.append("No crowd heat flag was live that week.")
    return " ".join(bits)


if active_tab == "AI Pulse":
    st.subheader("AI - the pulse, the advice, and the chatter")

    # ---- THE MOOD SLIDER --------------------------------------------
    # Requirement: a slider at the top of the page for the market's
    # feeling and bullishness at any point in time.
    #
    # It drives the NUMBERS, not the prose. The model's words need a
    # ~30s gateway call that only runs on the desk machine, so they
    # cannot follow a drag; bullishness, breadth, attention and
    # acceleration are arithmetic over the committed stores and exist
    # for every day back to 2017. Dragging is therefore instant and
    # covers the whole history, which is the opposite trade-off from
    # regenerating text and the right one for an exploratory control.
    #
    # It is also the project's standing rule made interactive: numbers
    # from the stores, words from the model (ARCHITECTURE §4.1).
    _mr_lo = _mr_hi = None
    if theme_counts is not None and len(theme_counts):
        _d = pd.to_datetime(theme_counts["date"])
        _mr_lo, _mr_hi = _d.min().to_pydatetime(), _d.max().to_pydatetime()
    if _mr_lo and _mr_hi and _mr_lo < _mr_hi:
        _day = st.slider("the market's mood on", min_value=_mr_lo,
                         max_value=_mr_hi, value=_mr_hi, format="DD MMM YYYY",
                         key="pulse_mood_day",
                         help="Drag to any day since 2017. The themes, "
                              "breadth and loud names are recomputed "
                              "from the stores for the 7 days ending "
                              "there - instantly, no model call.")
        _mr = _market_read(
            pd.Timestamp(_day),
            _mtime(os.path.join(PROCESSED_DIR, "daily_theme_counts.parquet")),
            _mtime(os.path.join(PROCESSED_DIR,
                                "daily_theme_sentiment.parquet")),
            _mtime(os.path.join(PROCESSED_DIR,
                                "daily_ticker_counts.parquet")))
        # NO bullishness-score metric here (removed by design) - breadth,
        # volume and the written snapshot carry the mood read.
        _k2, _k3, _k4 = st.columns(3)
        if _mr.get("breadth") is not None:
            _k2.metric("themes net bullish", f"{_mr['breadth']:.0%}",
                       help="Breadth. A mood carried by one theme is a "
                            "different market from a mood carried by "
                            "all of them.")
        if _mr.get("n_posts"):
            _k3.metric("tagged posts (7d)", f"{_mr['n_posts']:,}",
                       help="Posts that name a ticker or theme we track "
                            "(and therefore carry a sentiment read).")
        _k4.metric("as of", f"{pd.Timestamp(_day):%d %b %Y}")

        # numbers-derived digest (snapshot quote + "what the crowd was
        # talking about" / "how it felt" columns) removed on request
        # ("remove this section") - the model's own pulse below carries
        # the read
        _pp = os.path.join(PROCESSED_DIR,
                           f"ai_pulse_{pd.Timestamp(_day):%Y-%m-%d}.json")
        if os.path.exists(_pp):
            with st.expander(f"the model's words for "
                             f"{pd.Timestamp(_day):%d %b %Y}"):
                st.json(_read_json(_pp, _mtime(_pp)))
        elif pd.Timestamp(_day).date() != pd.Timestamp(_mr_hi).date():
            # THE WORDS DO NOT MOVE WITH THE SLIDER, and the page has to
            # say so loudly: the AI pulse does not refresh when the
            # slider moves back in time, and pretending otherwise would
            # mislead.
            #
            # It never could: the written sections are one stored
            # document, produced by a ~30s gateway call that only runs on
            # the desk machine. But a page showing 12 March at the top
            # and today's prose underneath is indistinguishable from a
            # broken page, and "it is working as designed" is no defence
            # if the design cannot be seen. So when the slider leaves the
            # latest day, the mismatch is stated in a warning rather than
            # a caption, and the sections below are labelled as belonging
            # to a different date.
            _lp = os.path.join(PROCESSED_DIR, "ai_pulse.json")
            _live_d = None
            if os.path.exists(_lp):
                _live_d = (_read_json(_lp, _mtime(_lp)) or {}).get("as_of")
            st.warning(
                f"**The numbers above are {pd.Timestamp(_day):%d %b %Y}. "
                "The written sections below are NOT** — they are the "
                f"stored pulse"
                + (f" from {_live_d}" if _live_d else "")
                + ", and they do not move with the slider. To write the "
                  "words for this day, on the desk machine:  "
                  f"`python -m analytics.ai_pulse --as-of "
                  f"{pd.Timestamp(_day):%Y-%m-%d}`  — it saves a dated "
                  "file and this panel will then show it here instead.")
        st.divider()

    _stale_words = (_mr_hi is not None
                    and pd.Timestamp(st.session_state.get(
                        "pulse_mood_day", _mr_hi)).date()
                    != pd.Timestamp(_mr_hi).date())
    st.markdown("## A - AI market pulse: what the posts are saying"
                + ("  ⚠ *(not the selected date)*" if _stale_words else ""))
    st.caption("The LLM's qualitative summary of the live posts - the "
               "numbers come from the stores, the words from the model."
               + ("  **These words are the stored pulse, not the day on "
                  "the slider above.**" if _stale_words else ""))
    _pulse = _ai_pulse_load()
    _pulse_real = _pulse is not None and not _pulse.get("mock")
    # The refresh button SPAWNS THE PIPELINE, so it belongs with the
    # other pipeline controls: workstation only. A hosted viewer cannot
    # run it (no credentials, ephemeral disk) and should not be shown a
    # control that spends an API budget.
    _c1, _c2 = st.columns([3, 1] if LOCAL_CONTROLS else [1, 0.0001])
    with _c2:
        if LOCAL_CONTROLS and st.button(
                     "refresh this page now (poll + pulse)",
                     disabled=_pipe_running,
                     help="Re-runs BOTH sections: the retail-prompt "
                          "poll (section B, ~1 min - 12 gateway calls) "
                          "and the market pulse (section A, ~30s - 3 "
                          "calls). Needs the VPN. Both also run "
                          "automatically at the end of every update."):
            start_pipeline([(["-m", "analytics.ai_poll"], None),
                            (["-m", "analytics.ai_pulse"], None)],
                           "AI refresh (poll + pulse)", plan="pulse")
    with _c1:
        if _pulse_real:
            st.caption(f"generated {_pulse.get('generated_at')} · model "
                       f"{_pulse.get('model')} · posts through "
                       f"{_pulse.get('as_of')} - sections 1-4 are "
                       "written from the POSTS alone: the model is shown "
                       "no counts, shares or model output, so they are a "
                       "read of what people actually wrote. Section 5 is "
                       "the one exception - it is given the measured "
                       "numbers below, because asking whether the story "
                       "matches the record needs both halves")
        elif _pulse is not None:
            st.warning("This pulse is placeholder text, not a real "
                       "read - it was generated with no model reachable. "
                       "It refreshes on the next pipeline run that has "
                       "a provider available.")
        else:
            st.warning("No pulse generated yet - the sections below are "
                       "HAND-WRITTEN SAMPLES showing the format. Run an "
                       "update (or the button here) on the desk machine "
                       "to fill them from your data.")

    if _pulse_real:
        # the LLM's X/100 fear-greed metric was REMOVED here on desk
        # instruction 2026-08-07 ("remove the bullish score number and
        # all the associated code") - the vibe bullets and one-liner
        # carry the mood read; no numeric score is printed or requested
        # from the model (see analytics/ai_pulse.py, same date).
        _vibe = _pulse.get("market_vibe") or {}
        st.markdown("### 1 - The vibe: how the market feels right now")
        st.caption("The whole market's mood, not its top trends - "
                   "sentiment across every forum in the panel.")
        for _b in (_vibe.get("bullets") or []):
            st.markdown(f"- {ai_text(_b)}")
        if not (_vibe.get("bullets") or []):
            st.info(ai_text(_pulse.get("talk_of_the_town") or ""))
        _ol = str(_vibe.get("one_liner") or "").strip()
        if _ol:
            st.markdown(
                f"<div style='border-left:4px solid {ACCENT};"
                "padding:14px 18px;margin:6px 0 2px 0;"
                "background:rgba(127,127,127,.06);font-size:1.15rem;"
                f"font-style:italic'>&ldquo;{ai_text(_ol)}&rdquo;</div>",
                unsafe_allow_html=True)
            st.caption("The line that sums up the week's mood - a "
                       "PARAPHRASE the model composes to capture the "
                       "register, never a real post reproduced. "
                       + ai_text(_vibe.get("one_liner_why", "")))

        st.markdown("### 2 - What all the forums are saying")
        st.info(ai_text(_pulse.get("market_pulse") or "(empty)"))
        _tott = str(_pulse.get("talk_of_the_town") or "").strip()
        if _tott and (_vibe.get("bullets") or []):
            with st.expander("what the crowd keeps coming back to "
                             "(the recurring threads and arguments)"):
                st.markdown(ai_text(_tott))

        # ---- 3. per-theme read, on a DROPDOWN ----
        _tb = [b for b in (_pulse.get("theme_briefs") or [])
               if isinstance(b, dict) and str(b.get("brief", "")).strip()]
        if _tb:
            st.markdown("### 3 - What retail thinks about a theme")
            _share = ((_pulse.get("evidence") or {})
                      .get("theme_mention_share_7d") or {})
            _order = sorted(
                _tb, key=lambda b: -float(
                    (_share.get(b.get("theme"), {}) or {}).get("share", 0)))
            _labs = {}
            for b in _order:
                _th = b.get("theme", "?")
                _s = (_share.get(_th, {}) or {}).get("share")
                _labs[(f"{theme_label(_th) if _th in THEME_ETFS else _th}"
                       + (f"  -  {_s:.0%} of mentions" if _s else "")
                       + (f"  ({THEME_ETFS[_th]})"
                          if _th in THEME_ETFS and THEME_ETFS[_th] else ""))
                      ] = b
            _pick = st.selectbox("theme", list(_labs), key="pulse_theme",
                                 help="Every theme with a material share "
                                      "of this week's chatter. The model "
                                      "wrote each brief from that theme's "
                                      "own posts.")
            _b = _labs[_pick]
            _th = _b.get("theme", "?")
            # (the conviction-z chip was removed 2026-08-07 with the
            # rest of the conviction display surface)
            _m1, _m2 = st.columns(2)
            _sh = (_share.get(_th, {}) or {})
            _m1.metric("share of mentions (7d)",
                       f"{_sh.get('share', 0):.1%}" if _sh.get("share")
                       else "-")
            _m2.metric("vs its own 4-week pace",
                       f"{_sh.get('vs_4w_avg')}x"
                       if _sh.get("vs_4w_avg") else "-")
            st.info(ai_text(_b.get("brief", "")))

        _cw = _pulse.get("catalyst_watch") or []
        _dv = _pulse.get("divergences") or []
        if _cw or _dv:
            _k1, _k2 = st.columns(2)
            with _k1:
                if _cw:
                    st.markdown("### 4 - Catalyst watch")
                    for c in _cw:
                        st.markdown(f"- **{ai_text(c.get('event', '?'))}** "
                                    f"({', '.join(c.get('themes', []))}) - "
                                    f"{ai_text(c.get('chatter', ''))}")
            with _k2:
                if _dv:
                    st.markdown("### 5 - Story vs numbers "
                                "(divergences)")
                    for d in _dv:
                        st.markdown(f"- **{ai_text(d.get('name', '?'))}** - "
                                    f"{ai_text(d.get('story', ''))}")

        with st.expander("the measured numbers for the same week - "
                         "used ONLY by section 5 (sections 1-4 are "
                         "written from posts alone, so these are also "
                         "here for you to check them against by hand)"):
            st.json(_pulse.get("evidence") or {})
    else:
        # No pulse yet: the LLM-written sections fall back to labelled
        # SAMPLES, but section 4 is REAL - the rally detector needs no
        # gateway, so its numbers are the same ones the desk will see
        # after the first live run.
        st.markdown("### 1 - The vibe: how the market feels right now")
        st.info(PULSE_TALK_SAMPLE)
        st.markdown("### 2 - What all the forums are saying")
        st.info(PULSE_MARKET_SAMPLE)
        st.markdown("### 3 - What retail thinks about a theme")
        # THE FOUR-THEME DROPDOWN IS NOT A LIMIT. It is the hand-written
        # SAMPLE fallback, shown only because no pulse has been
        # generated on this machine. The old caption said so in grey
        # body text under a working-looking dropdown, which reads as
        # "the product covers four themes". Say it loudly, say how many
        # there really are, and say what to run.
        _n_tradeable = len(THEME_ETFS)
        st.warning(
            f"**No AI Pulse has been generated on this machine, so the "
            f"four themes below are HAND-WRITTEN PLACEHOLDERS** - they "
            f"are here to show the format, not the coverage. A live "
            f"pulse writes a brief for **every theme the crowd is "
            f"actually discussing** - all {_n_tradeable} tradeable "
            f"themes on the current data - each one written from that "
            f"theme's own posts.\n\n"
            f"Generate one on the desk machine (needs the VPN and "
            f"dimsum_lite): `python -m analytics.ai_pulse`, or just run "
            f"`python update_data.py`, which does it at the end of "
            f"every pass.")
        _seg = list(PULSE_SEGMENTS_SAMPLE.items())
        _lab = st.selectbox("theme", [s for s, _ in _seg],
                            key="pulse_theme_sample")
        st.info(dict(_seg)[_lab])

    st.divider()
    # ---- 1. THE POLL: what the AI recommends when asked like retail --
    st.markdown("## B - What the AI is recommending to retail")
    st.caption("The POLL: at every data refresh the pipeline itself asks the model the questions a retail trader asks (config/ai_poll_prompts.csv - editable) and records every name and theme it recommends - a direct reading of the advice flowing from AI into the crowd. The panel is 12 prompts, one per kind of asker: the plain questions retail types, the hedge-fund-PM and Warren-Buffett personas the popular open-source AI-investing repos ship as system prompts, the JSON-decision agent loop, a value screen and the meme-squeeze ask - balanced so gold, dividends and the boring-portfolio ask sit beside the single AI ask. No backfill is possible; the series starts the day you start polling, and its forward test against the flags is pre-registered in notebook 09 \u00a72b.")
    # the full prompt panel, on request ("add all the prompts in a
    # drop down") - read straight from the editable CSV so the list
    # can never drift from what the poll actually asks; rendered
    # before the data branch so it shows even with no runs on record
    _ppf = os.path.join("config", "ai_poll_prompts.csv")
    if os.path.exists(_ppf):
        with st.expander("the prompts the poll asks"):
            try:
                _ppd = pd.read_csv(_ppf)
                for _fam, _fg in _ppd.groupby("family", sort=False):
                    st.markdown(f"**{_fam}**")
                    for _pr in _fg.itertuples():
                        st.markdown(f"- `{_pr.prompt_id}` — {_pr.prompt}")
            except Exception:
                st.caption("could not read config/ai_poll_prompts.csv")
    _pl = _ai_poll_load()
    _pl = (_pl[~_pl["mock"].astype(bool)]
           if _pl is not None and "mock" in _pl.columns else _pl)
    if _pl is None or not len(_pl):
        st.info("No poll runs yet. The poll runs automatically at the "
                "end of every update on the desk machine (Apollo "
                "gateway), asking the retail prompt panel in "
                "config/ai_poll_prompts.csv - edit that file to change "
                "the questions. Run one now: "
                "`python -m analytics.ai_poll`")
    else:
        _last_run = _pl["run_date"].max()
        _today = _pl[_pl["run_date"] == _last_run]
        _prev_runs = sorted(_pl["run_date"].unique())
        _prev = (_pl[_pl["run_date"] == _prev_runs[-2]]
                 if len(_prev_runs) > 1 else None)
        st.caption(f"latest poll {_last_run.date()} · "
                   f"{_pl['run_date'].nunique()} run(s) on record · "
                   f"{_today['prompt_id'].nunique()} prompts answered")
        _tk = _today[_today["kind"] == "ticker"]
        _cnt = (_tk.groupby(["name", "direction"]).size()
                .reset_index(name="prompts"))
        _cnt = _cnt.sort_values("prompts", ascending=False).head(15)
        _pc1, _pc2 = st.columns([1.2, 1])
        with _pc1:
            st.markdown("**most-recommended names today** (how many of "
                        f"the {_today['prompt_id'].nunique()} retail "
                        "prompts surfaced each)")
            _flagged = set()
            if desk is not None and len(desk):
                _dw2 = desk[desk["date"] > desk["date"].max()
                            - pd.Timedelta(days=21)]
                _flagged = (set(
                    _dw2[_dw2[sig_col("get_out", _dw2)]
                         .astype(bool)]["name"])
                    | set(_dw2[_dw2[sig_col("get_in", _dw2)]
                               .astype(bool)]["name"]))
            for r in _cnt.itertuples():
                _mark = ""
                if r.name in _flagged:
                    _mark = ("  <span style='color:%s;font-weight:600'>"
                             "⚑ carries a live flag</span>" % BEAR)
                _new = (" <span style='color:%s'>· new</span>" % ACCENT
                        if _prev is not None
                        and r.name not in set(_prev["name"]) else "")
                st.markdown(
                    f"- **{r.name}** ({r.direction}) - {r.prompts} "
                    f"prompt(s){_new}{_mark}", unsafe_allow_html=True)
        with _pc2:
            _th2 = _today[_today["kind"] == "theme"]
            st.markdown("**themes the AI pushes**")
            for n, c in (_th2.groupby("name").size()
                         .sort_values(ascending=False).head(8).items()):
                st.markdown(f"- {n} ({c})")
            # DOES THE SCAFFOLD CHANGE THE ANSWER? p13-p30 copy the
            # personas and agent loops retail actually runs; if those
            # return different names from the plain questions, the
            # advice reaching a retail "AI agent" user is not the advice
            # reaching a chatbot user - which is worth knowing.
            if "family" in _today.columns:
                _sc = _tk[_tk["family"].isin(["persona", "agent"])]
                _pl2 = _tk[_tk["family"] == "plain"]
                if len(_sc) and len(_pl2):
                    _only = (set(_sc["name"]) - set(_pl2["name"]))
                    st.markdown(
                        "**the agent/persona scaffolds add:** "
                        + (", ".join(sorted(_only)[:8]) if _only
                           else "nothing the plain questions missed"))
            if _prev is not None:
                _dropped = (set(_prev[_prev['kind'] == 'ticker']['name'])
                            - set(_tk['name']))
                if _dropped:
                    st.markdown("**dropped since last poll:** "
                                + ", ".join(sorted(_dropped)[:8]))
        if _pl["run_date"].nunique() >= 5:
            _top5 = (_pl[_pl["kind"] == "ticker"].groupby("name").size()
                     .nlargest(5).index)
            _ts = (_pl[(_pl["kind"] == "ticker")
                       & _pl["name"].isin(_top5)]
                   .groupby(["run_date", "name"]).size()
                   .unstack(fill_value=0))
            fig0 = go.Figure()
            for c in _ts.columns:
                fig0.add_trace(go.Scatter(x=_ts.index, y=_ts[c],
                                          mode="lines+markers", name=c))
            fig0.update_layout(height=280, title=dict(text=""),
                               margin=dict(l=10, r=10, t=24, b=10),
                               yaxis_title="prompts recommending it",
                               legend=dict(orientation="h",
                                           yanchor="bottom", y=1.0, x=0))
            _axes_fidelity(_theme(fig0))
            st.plotly_chart(fig0, width="stretch", key="poll_ts")
            st.caption("Rotation in the AI's advice. When a name climbs "
                       "here while its crowd heat chart heats up, the "
                       "crowd and its AI are feeding each other - the "
                       "herding mechanism notebook 09 \u00a72b tests.")

    # The "planned LLM segments (the full roadmap)" expander was removed
    # on recorded decision. It described sections that do not
    # exist, which on a page whose whole claim is that every sentence is
    # auditable against stored evidence is the one thing that should not
    # be there.
    # The second time control that used to sit here is gone (desk
    # requirement). One control, at the top, owns the date - the mood
    # slider. Two ways to set the same thing in two places is how a page
    # ends up showing one date in the header and another in the body.
    # ---- WHAT WE ACTUALLY ASKED THE MODEL ----------------------------
    # Requirement: the exact prompt must be inspectable - a fair question
    # to ask of any page written by a model, and the answer should not
    # require opening a source file. The instruction text is read live
    # from analytics/ai_pulse.py, so it cannot drift from what was
    # actually sent.
    with st.expander("the exact prompt behind this page"):
        try:
            from analytics import ai_pulse as _apm
            st.markdown("**System instruction** — applies to every call:")
            st.code(_apm._PULSE_SYSTEM, language="text")
            st.markdown("**The per-theme brief** — what section 3 asks "
                        "for:")
            # _themes_prompt takes ONE argument (by_theme). Calling it
            # with two raised TypeError, which the broad except below
            # swallowed - so this expander was permanently broken on
            # every machine and read as an environment problem.
            st.code(_apm._themes_prompt({"<theme>": ["<recent posts>"]}),
                    language="text")
            st.caption("Read live from analytics/ai_pulse.py, so this is "
                       "the instruction that was actually sent - not a "
                       "copy that can drift. The EVIDENCE block is the "
                       "only source of numbers the model is permitted to "
                       "cite.")
        except Exception as _e:                      # noqa: BLE001
            st.caption(f"prompt unavailable here: {type(_e).__name__}")

    st.caption("The LLM reads the freshly fetched raw posts, writes "
               "these sections, and only the finished text is stored - "
               "paraphrases, no verbatim crowd text, no usernames: the "
               "same text-free boundary as the committed aggregates.")

# ---- [dev] DATA STATS - a snapshot of the data behind the page:
# freshness, volumes, sources, ingestion status ----
if active_tab == "[dev] Data Stats":
    st.subheader("[dev] Data stats - the data behind every chart")
    st.caption("Everything on this tab is read live from the stores and "
               "the ingestion ledgers - nothing is cached beyond the "
               "files' own mtimes.")

    # ---- freshness of every core store ------------------------------
    st.markdown("#### Store freshness")
    _stores = [
        ("ticker mentions", "daily_ticker_counts.parquet"),
        ("ticker mentions by source", "daily_ticker_counts_by_source.parquet"),
        ("ticker sentiment", "daily_ticker_sentiment.parquet"),
        ("theme mentions", "daily_theme_counts.parquet"),
        ("theme sentiment", "daily_theme_sentiment.parquet"),
        ("emerging terms", "daily_term_counts.parquet"),
        ("episodes (ground truth)", "episodes.parquet"),
        ("desk signals (INCREASE EXPOSURE/OUT)", "euphoria_desk.parquet"),
        ("crowd heat levels", "euphoria_levels.parquet"),
    ]
    _rows_ds = []
    for _lab, _fn in _stores:
        _pth = os.path.join(PROCESSED_DIR, _fn)
        if not os.path.exists(_pth):
            _rows_ds.append({"store": _lab, "file": _fn,
                             "status": "MISSING"})
            continue
        _df_ds = load(_fn)
        _dcol = "date" if "date" in _df_ds.columns else None
        _rows_ds.append({
            "store": _lab, "file": _fn,
            "rows": f"{len(_df_ds):,}",
            "newest data day": (f"{pd.to_datetime(_df_ds[_dcol]).max():%Y-%m-%d}"
                                if _dcol else "-"),
            "file written": pd.Timestamp(
                os.path.getmtime(_pth), unit="s",
                tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "size": f"{os.path.getsize(_pth) / 1e6:.1f} MB",
        })
    st.dataframe(pd.DataFrame(_rows_ds), hide_index=True,
                 width="stretch")

    # ---- what the panel holds, per source ---------------------------
    st.markdown("#### Volumes, per source")
    _bs = load("daily_ticker_counts_by_source.parquet")
    if _bs is not None:
        _bs["date"] = pd.to_datetime(_bs["date"])
        _sr = []
        for _s, _g in _bs.groupby("source"):
            _last30 = _g[_g["date"] > _g["date"].max()
                         - pd.Timedelta(days=30)]
            _sr.append({
                "source": _s,
                "first day": f"{_g['date'].min():%Y-%m-%d}",
                "newest day": f"{_g['date'].max():%Y-%m-%d}",
                "total mentions": f"{int(_g['mention_count'].sum()):,}",
                "mentions, last 30d":
                    f"{int(_last30['mention_count'].sum()):,}",
                "distinct tickers": int(_g["ticker"].nunique()),
            })
        st.dataframe(pd.DataFrame(_sr), hide_index=True,
                     width="stretch")
        _ts_all = load("daily_ticker_sentiment.parquet")
        _th_all = load("daily_theme_counts.parquet")
        # TOTAL posts = every post pulled, whether or not it names a
        # ticker/theme. The __TOTAL__ rows of the term store carry it
        # (one per day, kept since term-tracking began) - no raw text
        # is ever read here.
        _tm_all = load("daily_term_counts.parquet")
        _tot = (_tm_all[_tm_all["term"] == "__TOTAL__"]
                if _tm_all is not None else None)
        # WHAT IS ACTUALLY IN STORAGE, and why the two headline numbers
        # differ by an order of magnitude (both totals are shown
        # deliberately). Both are true and they count different
        # things:
        #   TOTAL POSTS  - one per post, and only since term-tracking
        #                  began (Aug 2025). The __TOTAL__ row did not
        #                  exist before that, so there is no honest way
        #                  to extend it backwards.
        #   MENTIONS     - one per post PER NAME, across all history
        #                  back to 2017. A post about semis that names
        #                  NVDA, AMD and the semiconductor theme is one
        #                  post and four mentions. This is the ~3m.
        # Showing only the first invites "why so few?"; showing only the
        # second invites "we have 3m posts", which we do not.
        _tk_all = load("daily_ticker_counts.parquet")
        _men_th = (int(_th_all["mention_count"].sum())
                   if _th_all is not None else 0)
        _men_tk = (int(_tk_all["mention_count"].sum())
                   if _tk_all is not None else 0)
        _k0, _k1, _k2, _k3, _k4, _k5 = st.columns(6)
        if _tot is not None and len(_tot):
            _t0 = pd.to_datetime(_tot["date"]).min()
            _k0.metric(f"TOTAL posts pulled (since {_t0:%b %Y})",
                       f"{int(_tot['mention_count'].sum()):,}",
                       help="Every post pulled from every source, "
                            "whether or not it names a ticker or theme. "
                            "ONE PER POST.\n\nIt starts in "
                            f"{_t0:%b %Y} because that is when the "
                            "pipeline began storing a daily total; "
                            "before then only per-name counts were "
                            "kept, and there is no honest way to "
                            "reconstruct a post count from those. For "
                            "the full-history figure see NAME MENTIONS "
                            "beside this - a much bigger number that "
                            "counts something different.")
        else:
            _k0.metric("TOTAL posts pulled", "-")
        _yr0 = (pd.to_datetime(_th_all["date"]).min().year
                if _th_all is not None and len(_th_all) else "?")
        _k5.metric(f"name mentions on file (since {_yr0})",
                   f"{_men_th + _men_tk:,}",
                   help=f"{_men_th:,} theme mentions + {_men_tk:,} "
                        "ticker mentions, all the way back.\n\nTHIS "
                        "IS NOT A POST COUNT. It is one row per post "
                        "PER NAME: a single post about semis that names "
                        "NVDA, AMD and the semiconductor theme "
                        "contributes one post and three mentions. It is "
                        "the right number for 'how much signal is in "
                        "the store' and the wrong one for 'how many "
                        "posts do we have'.")
        _k1.metric("tagged posts (all time)",
                   f"{int(_ts_all['n_posts'].sum()):,}"
                   if _ts_all is not None else "-",
                   help="Posts that name a ticker or theme we track - "
                        "the subset all mention/sentiment charts are "
                        "built from.")
        _k2.metric("themes tracked",
                   int(_th_all["theme"].nunique())
                   if _th_all is not None else 0)
        _k3.metric("tickers ever seen",
                   int(_bs["ticker"].nunique()))
        _wk_now = _bs[_bs["date"] > _bs["date"].max()
                      - pd.Timedelta(days=7)]
        _k4.metric("mentions, last 7d",
                   f"{int(_wk_now['mention_count'].sum()):,}")
        # weekly stacked volume, per source (the coverage regime)
        _wk = (_bs.groupby([pd.Grouper(key="date", freq="W"), "source"])
               ["mention_count"].sum().unstack(fill_value=0)
               .tail(52))
        _figv = _base_fig("posts per week, per source (last 12 months)")
        for _s in _wk.columns:
            _figv.add_trace(go.Bar(x=_wk.index, y=_wk[_s], name=_s))
        _figv.update_layout(barmode="stack", height=300)
        st.plotly_chart(_figv, width="stretch", key="devstats_vol")

    # ---- ingestion status -------------------------------------------
    st.markdown("#### Ingestion status")
    import json as _json_ds
    _ing_rows = []
    _ref = os.path.join(ROOT, "data", "reference")
    for _lab, _fn, _keys in [
            ("Reddit submissions (Arctic Shift)",
             "reddit_arctic_watermark.json", None),
            ("Reddit comments", "reddit_comments_watermark.json", None),
            ("comment budget / cost", "reddit_comments_cost.json", None),
            ("pipeline stage times", "pipeline_stage_times.json", None)]:
        _pth = os.path.join(_ref, _fn)
        if not os.path.exists(_pth):
            _ing_rows.append({"ledger": _lab, "file": _fn,
                              "content": "(missing)"})
            continue
        try:
            _d = _json_ds.load(open(_pth))
            _txt = _json_ds.dumps(_d)[:220]
        except (ValueError, OSError):
            _txt = "(unreadable)"
        _ing_rows.append({
            "ledger": _lab, "file": _fn,
            "updated": pd.Timestamp(os.path.getmtime(_pth), unit="s",
                                    tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "content": _txt})
    st.dataframe(pd.DataFrame(_ing_rows), hide_index=True,
                 width="stretch")
    _raw_root = os.path.join(ROOT, "data", "raw")
    if os.path.isdir(_raw_root):
        _rr = []
        for _dname in sorted(os.listdir(_raw_root)):
            _dpath = os.path.join(_raw_root, _dname)
            if not os.path.isdir(_dpath):
                continue
            _fs = [f for f in os.listdir(_dpath)
                   if f.endswith(".jsonl.zst") or f.endswith(".csv.zst")]
            if not _fs:
                _rr.append({"raw archive": _dname, "files": 0})
                continue
            _newest = max(_fs, key=lambda f: os.path.getmtime(
                os.path.join(_dpath, f)))
            _rr.append({
                "raw archive": _dname, "files": len(_fs),
                "total size": f"{sum(os.path.getsize(os.path.join(_dpath, f)) for f in _fs) / 1e6:.0f} MB",
                "newest file": _newest})
        st.dataframe(pd.DataFrame(_rr), hide_index=True,
                     width="stretch")
        st.caption("Raw archives live gitignored on the pull machine "
                   "only; the committed stores above are text-free. A "
                   "machine without raw archives is normal - it runs "
                   "entirely from the aggregates.")
    _mdl_ds = (desk_report or {}).get("model", "rules")
    st.markdown("#### Signal engine")
    _gi_ds = (desk_report or {}).get("get_in") or {}
    _go_ds = (desk_report or {}).get("get_out") or {}
    _xp_ds = (desk_report or {}).get("experimental_price_blind") or {}
    st.caption(f"Desk model: **{_mdl_ds}** | Standard cuts (F1): INCREASE EXPOSURE "
               f"{_gi_ds.get('live_threshold', float('nan')):.3f} / "
               f"CUT EXPOSURE {_go_ds.get('live_threshold', float('nan')):.3f}"
               f" | Strict cuts (F0.5): INCREASE EXPOSURE "
               f"{(_gi_ds.get('strict_threshold') or float('nan')):.3f} /"
               f" CUT EXPOSURE "
               f"{(_go_ds.get('strict_threshold') or float('nan')):.3f} | "
               "trigger: shaped crossings (phase-gated, re-armed, "
               "63d spacing, 21d IN/OUT separation) | records: "
               "euphoria_desk_report.json, desk_model_insight.json, "
               "docs/RESEARCH_RECORD.md, "
               "docs/research/alert_shape_sweep.json")
    if _xp_ds:
        _xgi, _xgo = (_xp_ds.get("get_in") or {}), (_xp_ds.get("get_out")
                                                    or {})
        st.caption(f"Experimental price-blind trigger: **{_xp_ds.get('model')}** "
                   f"on {len(_xp_ds.get('bank') or [])} crowd-only "
                   f"measurements, NO price features, NO phase gate | "
                   f"Standard cuts: INCREASE EXPOSURE "
                   f"{(_xgi.get('strict_threshold') or float('nan')):.3f} / "
                   f"CUT EXPOSURE {(_xgo.get('strict_threshold') or float('nan')):.3f}"
                   f" | F1 cuts: "
                   f"{(_xgi.get('live_threshold') or float('nan')):.3f} / "
                   f"{(_xgo.get('live_threshold') or float('nan')):.3f} | "
                   "record: docs/research/nb08_price_blind.json")
