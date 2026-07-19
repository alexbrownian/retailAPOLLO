#!/usr/bin/env python
"""
pull_bloomberg_prices.py - daily close prices from the Bloomberg Terminal API.

Runs on any machine with a Bloomberg Terminal + blpapi installed (both
machines have access), after update_data.py, so the tickers pulled match
what the dashboard shows:

    python pull_bloomberg_prices.py            # connect + pull + save
    python pull_bloomberg_prices.py --dry-run  # show what WOULD be pulled, no connect

WHAT IT PULLS
    Field   : PX_LAST (daily close)
    Window  : START_DATE -> END_DATE from update_data.py ('' end = up to today);
              the PIPELINE_* env vars set by update_data.py override for one run
    Symbols : the union of
                * the PRICE_TOP_N most-mentioned tickers over the window
                * the PRICE_TOP_N most-mentioned tickers of the last 60 days
                  (the overlays pick their tickers at render time, so a name
                  that got loud recently must be priced too)
                * every theme's anchor ETF and its fallbacks (src/themes.py)
                * anything that appears in the trade signals
              Each is sent to Bloomberg as "<SYMBOL> US Equity".

OUTPUT
    data/prices/prices.parquet  (long, tidy):  date, symbol, px_last
    'symbol' is the plain ticker/ETF (e.g. AAPL, XBI) so the dashboard overlays
    join straight onto the mentions / conviction / signals tables. The file
    stays local (gitignored - Bloomberg redistribution terms).

Uses only blpapi's HistoricalDataRequest - no wrapper packages. The Terminal
must be running and logged in; blpapi connects on localhost:8194.
"""

import argparse
import datetime
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# The window and PRICE_TOP_N come from src/config.py, which itself honours
# the PIPELINE_START_DATE / PIPELINE_END_DATE env vars update_data.py sets -
# so a one-off --start/--end override reaches this script automatically.
from src.config import START_DATE, END_DATE, PRICE_TOP_N
from src.themes import THEME_ETFS, THEME_ETF_FALLBACKS       # theme -> ETF map

PROCESSED = os.path.join(ROOT, "data", "processed")
PRICES_DIR = os.path.join(ROOT, "data", "prices")
OUT_PATH = os.path.join(PRICES_DIR, "prices.parquet")

FIELD = "PX_LAST"
CHUNK = 50            # securities per Bloomberg request (keeps each request small)


# ---------------------------------------------------------------------------
# 1. Decide WHICH symbols to pull - from the same aggregates the notebooks read
# ---------------------------------------------------------------------------
def _read(name):
    path = os.path.join(PROCESSED, name)
    return pd.read_parquet(path) if os.path.exists(path) else None


def window_dates():
    """(start 'YYYYMMDD', end 'YYYYMMDD'). Empty END_DATE means up to today."""
    start = START_DATE.replace("-", "")
    if END_DATE:
        # END_DATE is EXCLUSIVE in the pipeline; Bloomberg endDate is inclusive,
        # so step back one day to keep the same span.
        end_dt = datetime.date.fromisoformat(END_DATE) - datetime.timedelta(days=1)
    else:
        end_dt = datetime.date.today()
    return start, end_dt.strftime("%Y%m%d")


def build_symbol_universe():
    """Return a sorted list of plain symbols (tickers + ETFs) to price."""
    symbols = set()

    # top-N most-mentioned tickers over the window, PLUS the top-N of the
    # last 60 days - the overlays auto-pick their tickers at render time,
    # and a recently-loud name can out-rank the whole-window top.
    counts = _read("daily_ticker_counts.parquet")
    if counts is not None and len(counts):
        c = counts.copy()
        c["date"] = pd.to_datetime(c["date"])
        lo = pd.to_datetime(START_DATE)
        hi = pd.to_datetime(END_DATE) if END_DATE else c["date"].max()
        c = c[(c["date"] >= lo) & (c["date"] <= hi)]
        top = (c.groupby("ticker")["mention_count"].sum()
               .sort_values(ascending=False).head(PRICE_TOP_N).index.tolist())
        symbols.update(top)
        recent = c[c["date"] >= hi - pd.Timedelta(days=60)]
        top_recent = (recent.groupby("ticker")["mention_count"].sum()
                      .sort_values(ascending=False).head(PRICE_TOP_N).index.tolist())
        symbols.update(top_recent)

    # every theme's anchor ETF, plus every fallback anchor (a backtest window
    # older than a young ETF can still draw the theme against its fallback)
    symbols.update(THEME_ETFS.values())
    for fallbacks in THEME_ETF_FALLBACKS.values():
        symbols.update(fallbacks)

    # international names (Europe/Japan) priced through their US ADRs -
    # the keyword themes count them, these symbols let overlays price them
    from src.themes import INTERNATIONAL_ADRS
    symbols.update(INTERNATIONAL_ADRS.values())

    # anything named in the signals
    sig_theme = _read("trade_signals.parquet")
    if sig_theme is not None and "etf" in sig_theme.columns:
        symbols.update(sig_theme["etf"].dropna().astype(str))
    sig_tick = _read("trade_signals_tickers.parquet")
    if sig_tick is not None and "ticker" in sig_tick.columns:
        symbols.update(sig_tick["ticker"].dropna().astype(str))

    # clean up: drop blanks and the "?" placeholder (a theme with no
    # approved anchor stores "?" as its instrument), upper-case, sort
    symbols = {s.strip().upper() for s in symbols
               if s and str(s).strip() and str(s).strip() != "?"}
    return sorted(symbols)


# Non-US lines from the firm-approved list. Keys are how the symbol is
# stored in prices.parquet / THEME_ETFS; values are the exact Bloomberg
# security strings (exchange code included). Extend as more foreign lines
# get approved.
FOREIGN_SECURITIES = {
    "1622 JT": "1622 JT Equity",       # NF Topix-17 Auto & Transport Equip
}


def to_bloomberg(symbol):
    """Plain ticker/ETF -> Bloomberg security string. US-listed by default;
    approved non-US lines resolve through FOREIGN_SECURITIES."""
    if symbol in FOREIGN_SECURITIES:
        return FOREIGN_SECURITIES[symbol]
    return f"{symbol} US Equity"


# ---------------------------------------------------------------------------
# 2. Pull the prices (the only part that needs the Terminal)
# ---------------------------------------------------------------------------
def pull_prices(symbols, start_yyyymmdd, end_yyyymmdd):
    """Return a long DataFrame: date, symbol, px_last. Uses blpapi directly."""
    import blpapi

    session = blpapi.Session()          # default host localhost, port 8194
    if not session.start():
        raise RuntimeError("could not start blpapi Session - is the Terminal running?")
    try:
        if not session.openService("//blp/refdata"):
            raise RuntimeError("could not open //blp/refdata service")
        refdata = session.getService("//blp/refdata")

        rows = []
        # send the securities in small chunks so each request stays light
        for i in range(0, len(symbols), CHUNK):
            chunk = symbols[i:i + CHUNK]
            request = refdata.createRequest("HistoricalDataRequest")
            for sym in chunk:
                request.getElement("securities").appendValue(to_bloomberg(sym))
            request.getElement("fields").appendValue(FIELD)
            request.set("periodicitySelection", "DAILY")
            request.set("startDate", start_yyyymmdd)
            request.set("endDate", end_yyyymmdd)
            print(f"  requesting {len(chunk)} securities "
                  f"({i + 1}-{i + len(chunk)} of {len(symbols)}) ...", flush=True)
            session.sendRequest(request)

            # drain events until this request's RESPONSE arrives
            done = False
            while not done:
                event = session.nextEvent(500)
                for msg in event:
                    rows.extend(_parse_message(msg))
                if event.eventType() == blpapi.Event.RESPONSE:
                    done = True
        return pd.DataFrame(rows, columns=["date", "symbol", "px_last"])
    finally:
        session.stop()


def _parse_message(msg):
    """Pull (date, symbol, px_last) rows out of one HistoricalData message."""
    out = []
    if not msg.hasElement("securityData"):
        return out
    sec_data = msg.getElement("securityData")
    # Bloomberg returns 'IBM US Equity'; strip the suffix back to the plain
    # symbol. Foreign lines map back through FOREIGN_SECURITIES so the
    # stored symbol matches THEME_ETFS exactly (e.g. '1622 JT').
    security = sec_data.getElementAsString("security")
    _back = {v: k for k, v in FOREIGN_SECURITIES.items()}
    symbol = _back.get(security, security.replace(" US Equity", "").strip())

    if sec_data.hasElement("securityError"):
        print(f"    (no data for {security})")
        return out

    field_data = sec_data.getElement("fieldData")
    for i in range(field_data.numValues()):
        point = field_data.getValueAsElement(i)
        if not point.hasElement("date") or not point.hasElement(FIELD):
            continue
        d = point.getElementAsDatetime("date")
        px = point.getElementAsFloat(FIELD)
        out.append({"date": f"{d.year:04d}-{d.month:02d}-{d.day:02d}",
                    "symbol": symbol, "px_last": px})
    return out


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------
def plan_requests(symbols, start, end, existing):
    """INCREMENTAL planning: figure out, per symbol, which part of the
    requested window is NOT already in prices.parquet, and group symbols
    that need the same span so they batch into shared requests.

    Returns {(span_start, span_end): [symbols]}. Symbols whose window is
    already fully covered appear in no bucket - nothing is re-downloaded.
    A 3-day tolerance absorbs weekends/holidays at the span edges."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    tol = pd.Timedelta(days=3)

    coverage = {}
    if existing is not None and len(existing):
        agg = existing.groupby("symbol")["date"].agg(["min", "max"])
        coverage = {s: (r["min"], r["max"]) for s, r in agg.iterrows()}

    buckets = {}
    for sym in symbols:
        if sym not in coverage:
            spans = [(start_ts, end_ts)]                 # brand new symbol
        else:
            lo, hi = coverage[sym]
            spans = []
            if start_ts < lo - tol:
                spans.append((start_ts, lo - pd.Timedelta(days=1)))
            if end_ts > hi + tol:
                spans.append((hi + pd.Timedelta(days=1), end_ts))
        for a, b in spans:
            key = (a.strftime("%Y%m%d"), b.strftime("%Y%m%d"))
            buckets.setdefault(key, []).append(sym)
    return buckets


def main():
    p = argparse.ArgumentParser(description="Pull daily close prices from Bloomberg.")
    p.add_argument("--dry-run", action="store_true",
                   help="show the symbol universe + request window; do NOT connect")
    p.add_argument("--force", action="store_true",
                   help="re-download the whole window even if already covered")
    args = p.parse_args()

    symbols = build_symbol_universe()
    start, end = window_dates()
    print("=" * 64)
    print("BLOOMBERG PRICE PULL")
    print(f"  window : {start} -> {end}  (from src/config.py / --start/--end)")
    print(f"  field  : {FIELD} (daily close)")
    print(f"  symbols: {len(symbols)}  (top {PRICE_TOP_N} mentioned + theme ETFs + signals)")
    print(f"           {', '.join(symbols[:25])}{' ...' if len(symbols) > 25 else ''}")
    print("=" * 64)

    if args.dry_run:
        print("--dry-run: nothing pulled, nothing written.")
        return 0
    if not symbols:
        print("no symbols to pull - run update_data.py first so the aggregates exist.")
        return 1

    # INCREMENTAL: only fetch what prices.parquet does not already hold.
    # The store is append-only - a pull can only ADD rows, never lose any,
    # so switching windows back and forth costs nothing after the first pull.
    existing = None
    if os.path.exists(OUT_PATH) and not args.force:
        existing = pd.read_parquet(OUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        print(f"existing store: {len(existing):,} rows, "
              f"{existing['symbol'].nunique()} symbols "
              f"({existing['date'].min().date()} -> {existing['date'].max().date()})")

    buckets = plan_requests(symbols, start, end, existing)
    if not buckets:
        print("everything in this window is already in prices.parquet - "
              "nothing to fetch (use --force to re-download).")
        return 0
    n_req = sum(len(v) for v in buckets.values())
    print(f"incremental plan: {len(buckets)} span(s), {n_req} symbol-requests "
          f"(fully-covered symbols skipped)")

    parts = [] if existing is None else [existing]
    for (span_a, span_b), syms in sorted(buckets.items()):
        print(f"  span {span_a} -> {span_b}: {len(syms)} symbols")
        got = pull_prices(syms, span_a, span_b)
        if len(got):
            got["date"] = pd.to_datetime(got["date"])
            parts.append(got)

    prices = pd.concat(parts, ignore_index=True)
    if prices.empty:
        print("Bloomberg returned no rows - check the Terminal is logged in.")
        return 1
    # a symbol/date can arrive twice at span edges - keep the newest fetch
    prices = (prices.drop_duplicates(subset=["symbol", "date"], keep="last")
              .sort_values(["symbol", "date"]).reset_index(drop=True))

    os.makedirs(PRICES_DIR, exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    prices.to_parquet(tmp, index=False)      # atomic swap - never half-written
    os.replace(tmp, OUT_PATH)
    print(f"saved {len(prices):,} rows for {prices['symbol'].nunique()} symbols "
          f"-> {OUT_PATH}")

    # COVERAGE REPORT - name every requested symbol that came back empty, so
    # "no price rows" in the overlays is never a mystery.
    got = set(prices["symbol"].unique())
    missing = [s for s in symbols if s not in got]
    if missing:
        print(f"NO DATA for {len(missing)} of {len(symbols)} requested symbols "
              "(delisted, non-US listing, or younger than the window):")
        print("  " + ", ".join(missing))
    else:
        print("full coverage: every requested symbol returned prices.")

    print("next: python -m streamlit run dashboard.py  (the overlays render there)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
