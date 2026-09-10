#!/usr/bin/env python
"""Pull daily closes into ``data/prices/prices.parquet``.

    python pull_prices.py                       # provider from config/settings.csv
    python pull_prices.py --provider yfinance   # Yahoo Finance, no credentials
    python pull_prices.py --provider bloomberg  # Terminal; falls back to yfinance
    python pull_prices.py --dry-run             # show the plan, pull nothing

Providers (see ``src/prices.py``): ``bloomberg`` uses ``blpapi``
``HistoricalDataRequest`` for ``PX_LAST`` against a running Terminal;
``yfinance`` uses Yahoo Finance adjusted closes. ``auto`` picks Bloomberg
when it can connect and yfinance otherwise; ``bloomberg`` falls back to
yfinance if the pull fails unless ``--no-fallback`` is given.

Window
    ``START_DATE`` to ``END_DATE`` from ``src/config.py`` (an empty end
    means up to today); the ``PIPELINE_*`` environment variables set by
    ``update_data.py`` override them for one run.
Symbols
    The union of the ``PRICE_TOP_N`` most-mentioned tickers over the
    window, the ``PRICE_TOP_N`` most-mentioned tickers of the last 60
    days, every theme's anchor ETF and its fallbacks, every approved
    instrument, the international ADRs, and anything that appears in the
    trade signals.
Output
    ``data/prices/prices.parquet`` (long form): ``date``, ``symbol``,
    ``px_last``, ``source``. ``symbol`` is the plain ticker/ETF so the
    dashboard joins straight onto the mentions and signals tables. The
    file is not committed.

The pull is incremental (``plan_requests``): only the spans a symbol
does not already have are requested. A symbol whose stored rows came
from a different provider is re-pulled in full and its old rows
replaced, so adjusted (yfinance) and unadjusted (Bloomberg) closes are
never spliced into one series.
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
# 1. Decide WHICH symbols to pull - from the same aggregates the dashboard reads
# ---------------------------------------------------------------------------
def _read(name):
    """Read a ``data/processed`` parquet by name, or ``None`` if absent."""
    path = os.path.join(PROCESSED, name)
    return pd.read_parquet(path) if os.path.exists(path) else None


def window_dates():
    """Return the request window as ``(start, end)`` in ``YYYYMMDD`` form.

    An empty ``END_DATE`` means up to today. ``END_DATE`` is exclusive in
    the pipeline while Bloomberg's ``endDate`` is inclusive, so one day is
    stepped back to keep the same span.
    """
    start = START_DATE.replace("-", "")
    if END_DATE:
        # END_DATE is EXCLUSIVE in the pipeline; Bloomberg endDate is inclusive,
        # so step back one day to keep the same span.
        end_dt = datetime.date.fromisoformat(END_DATE) - datetime.timedelta(days=1)
    else:
        end_dt = datetime.date.today()
    return start, end_dt.strftime("%Y%m%d")


def build_symbol_universe():
    """Return the sorted list of plain symbols (tickers and ETFs) to price.

    See the module docstring for the sets that make up the union. Blanks
    and the ``"?"`` placeholder (a theme with no approved anchor) are
    dropped and everything is upper-cased.
    """
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

    # EVERY APPROVED INSTRUMENT, whether or not a theme points at it.
    # Some approved rows (factor and style lines, index lines) are
    # neither an anchor nor a fallback, so the loop above never asks
    # for them, and without a price series they are invisible on the
    # dashboard with nothing reporting it. The approved list is the
    # tradeable universe: if an instrument is approved the system must
    # be able to draw it. These lines carry no crowd signal and are not
    # becoming themes; they are benchmarks, and a benchmark with no
    # price series is not a benchmark.
    from src.themes import APPROVED_INSTRUMENTS
    symbols.update(APPROVED_INSTRUMENTS)

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


# Symbol mapping lives in src/prices.py (to_bloomberg / to_yfinance), read
# from config/approved_instruments.csv. Adding a foreign line = one CSV row.
from src.prices import to_bloomberg, provider_chain, fetch_with_fallback  # noqa: E402
from src.themes import APPROVED_INSTRUMENTS                                # noqa: E402

FOREIGN_SECURITIES = {
    sym: row["bloomberg"]
    for sym, row in APPROVED_INSTRUMENTS.items()
    if row.get("bloomberg") and row["bloomberg"] != f"{sym} US Equity"
}


# ---------------------------------------------------------------------------
# 2. Pull the prices (the only part that needs the Terminal)
# ---------------------------------------------------------------------------
def bloomberg_request(symbols, start_yyyymmdd, end_yyyymmdd, log=print):
    """Request daily closes for ``symbols`` over the window via blpapi.

    Securities are sent in chunks of ``CHUNK`` so each request stays
    light; events are drained until each request's RESPONSE arrives.

    Args:
        symbols: Plain symbols (mapped through ``to_bloomberg``).
        start_yyyymmdd: Inclusive start date.
        end_yyyymmdd: Inclusive end date.

    Returns:
        Long DataFrame with ``date``, ``symbol`` and ``px_last``.

    Raises:
        RuntimeError: When the blpapi session or the refdata service
            cannot be opened.
    """
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
            log(f"  requesting {len(chunk)} securities "
                f"({i + 1}-{i + len(chunk)} of {len(symbols)}) ...")
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
    """Extract ``(date, symbol, px_last)`` rows from one HistoricalData message."""
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
    """Plan the incremental pull.

    Works out, per symbol, which part of the requested window is not
    already in the store, and groups symbols that need the same span so
    they batch into shared requests. A 3-day tolerance absorbs weekends
    and holidays at the span edges.

    Only the TRAILING gap is planned for a symbol already in the store.
    A leading gap (the store starts after the window start) is the
    listing date, which does not move: re-asking for it every run cost
    one request per late-listed symbol per run and returned nothing.
    ``--force`` re-pulls whole windows when a leading gap is real.

    Args:
        symbols: Plain symbols to price.
        start: Window start, ``YYYYMMDD``.
        end: Window end, ``YYYYMMDD``.
        existing: The current store with a datetime ``date`` column, or
            ``None``.

    Returns:
        Dict ``{(span_start, span_end): [symbols]}`` with ``YYYYMMDD``
        keys. Symbols whose window is already fully covered appear in no
        bucket, so nothing is re-downloaded.
    """
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
            if end_ts > hi + tol:
                spans.append((hi + pd.Timedelta(days=1), end_ts))
        for a, b in spans:
            key = (a.strftime("%Y%m%d"), b.strftime("%Y%m%d"))
            buckets.setdefault(key, []).append(sym)
    return buckets


def _current_sources(existing):
    """``{symbol: source}`` from the store (``bloomberg`` when the column
    predates provider tracking)."""
    if existing is None or not len(existing):
        return {}
    if "source" not in existing.columns:
        return {s: "bloomberg" for s in existing["symbol"].unique()}
    return (existing.dropna(subset=["source"]).groupby("symbol")["source"]
            .agg(lambda s: s.iloc[-1]).to_dict())


def main():
    """Build the universe, choose the provider, plan the spans, pull, save.

    Returns:
        ``0`` on success or when nothing needs fetching; ``1`` when there
        are no symbols or no provider returned rows.
    """
    from src import settings
    p = argparse.ArgumentParser(description="Pull daily close prices.")
    p.add_argument("--provider", choices=("auto", "bloomberg", "yfinance"),
                   default=None,
                   help="price source (default: price_provider in "
                        "config/settings.csv)")
    p.add_argument("--no-fallback", action="store_true",
                   help="with --provider bloomberg or auto, do not fall "
                        "back to yfinance: neither when Bloomberg fails, "
                        "nor for symbols Bloomberg has no data for")
    p.add_argument("--dry-run", action="store_true",
                   help="show the symbol universe + request window; do NOT connect")
    p.add_argument("--force", action="store_true",
                   help="re-download the whole window even if already covered")
    args = p.parse_args()
    preference = args.provider or settings.get("price_provider")

    symbols = build_symbol_universe()
    start, end = window_dates()
    print("=" * 64)
    print("PRICE PULL")
    print(f"  provider: {preference}"
          + ("" if args.no_fallback or preference == "yfinance"
             else "  (falls back to yfinance on failure)"))
    print(f"  window  : {start} -> {end}  (from src/config.py / --start/--end)")
    print(f"  symbols : {len(symbols)}  (top {PRICE_TOP_N} mentioned + theme ETFs + signals)")
    print(f"            {', '.join(symbols[:25])}{' ...' if len(symbols) > 25 else ''}")
    print("=" * 64)

    if args.dry_run:
        print("--dry-run: nothing pulled, nothing written.")
        return 0
    if not symbols:
        print("no symbols to pull - run update_data.py first so the aggregates exist.")
        return 1

    chain = provider_chain(preference, fallback=not args.no_fallback)
    active = chain[0].name

    # INCREMENTAL: only fetch what prices.parquet does not already hold.
    existing = None
    if os.path.exists(OUT_PATH) and not args.force:
        existing = pd.read_parquet(OUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        print(f"existing store: {len(existing):,} rows, "
              f"{existing['symbol'].nunique()} symbols "
              f"({existing['date'].min().date()} -> {existing['date'].max().date()})")

    # ONE SYMBOL, ONE SOURCE: a symbol whose stored rows came from a
    # provider that is not in this run's chain is re-pulled in full and
    # its old rows dropped. A symbol stored from the FALLBACK provider is
    # kept when the fallback is still in the chain (it was filled from
    # there because the primary had nothing); `--provider bloomberg
    # --no-fallback` or `--force` moves such symbols back.
    sources = _current_sources(existing)
    chain_names = {c.name for c in chain}
    switching = sorted(s for s in symbols
                       if s in sources and sources[s] not in chain_names)
    if switching and existing is not None:
        print(f"provider changed for {len(switching)} symbol(s) "
              f"(stored as {sorted({sources[s] for s in switching})}, now "
              f"{active}) - re-pulling their full window: "
              f"{', '.join(switching[:10])}{' ...' if len(switching) > 10 else ''}")
        existing = existing[~existing["symbol"].isin(switching)]

    # Symbols already stored from the fallback provider keep being
    # extended from it (asking the primary would return nothing and
    # leave their series frozen).
    fb_syms = sorted(s for s in symbols
                     if sources.get(s) in chain_names and sources[s] != active)
    primary_syms = [s for s in symbols if s not in fb_syms]
    buckets = plan_requests(primary_syms, start, end, existing)
    buckets_fb = (plan_requests(fb_syms, start, end, existing)
                  if fb_syms else {})
    if not buckets and not buckets_fb:
        print("everything in this window is already in prices.parquet - "
              "nothing to fetch (use --force to re-download).")
        return 0
    n_req = sum(len(v) for v in buckets.values())
    print(f"incremental plan: {len(buckets)} span(s), {n_req} symbol-requests "
          f"(fully-covered symbols skipped)"
          + (f"; {len(fb_syms)} symbol(s) stay on {chain[-1].name}"
             if fb_syms else ""))

    parts = [] if existing is None else [existing]
    used = None
    for (span_a, span_b), syms in sorted(buckets_fb.items()):
        print(f"  span {span_a} -> {span_b}: {len(syms)} symbols via "
              f"{chain[-1].name} (fallback-stored)")
        try:
            got_fb = chain[-1].fetch(syms, span_a, span_b)
        except Exception as exc:                           # noqa: BLE001
            print(f"  {chain[-1].name} FAILED - {type(exc).__name__}: {exc}")
            continue
        if len(got_fb):
            got_fb["date"] = pd.to_datetime(got_fb["date"])
            parts.append(got_fb)
    for (span_a, span_b), syms in sorted(buckets.items()):
        print(f"  span {span_a} -> {span_b}: {len(syms)} symbols")
        got, used_now = fetch_with_fallback(chain, syms, span_a, span_b)
        if used is not None and used_now != used:
            # a mid-run fallback would splice sources; keep to one per run
            print(f"provider changed mid-run ({used} -> {used_now}); "
                  "re-run so every span comes from one provider")
            return 1
        used = used_now
        if len(got):
            got["date"] = pd.to_datetime(got["date"])
            parts.append(got)

    if used is None:                       # only fallback-stored spans ran
        used = chain[-1].name
    prices = pd.concat(parts, ignore_index=True)
    if prices.empty:
        print("no price rows returned - check the provider (Terminal logged in, "
              "or network access for yfinance).")
        return 1
    if "source" not in prices.columns:
        prices["source"] = "bloomberg"
    prices["source"] = prices["source"].fillna("bloomberg")
    # a symbol/date can arrive twice at span edges - keep the newest fetch
    prices = (prices.drop_duplicates(subset=["symbol", "date"], keep="last")
              .sort_values(["symbol", "date"]).reset_index(drop=True))

    os.makedirs(PRICES_DIR, exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    prices.to_parquet(tmp, index=False)      # atomic swap - never half-written
    os.replace(tmp, OUT_PATH)
    print(f"saved {len(prices):,} rows for {prices['symbol'].nunique()} symbols "
          f"via {used} -> {OUT_PATH}")

    # COVERAGE REPORT - name every requested symbol that came back empty, so
    # "no price rows" in the overlays is never a mystery.
    got = set(prices["symbol"].unique())
    missing = [s for s in symbols if s not in got]
    if missing:
        print(f"NO DATA for {len(missing)} of {len(symbols)} requested symbols "
              "(delisted, non-US listing, younger than the window, or no "
              "Yahoo mapping):")
        print("  " + ", ".join(missing))

    # FILL FROM THE FALLBACK - a symbol with NO rows at all from the
    # active provider (an OTC line the Terminal is not entitled to, a
    # delisted name it no longer serves) is pulled in full from the next
    # provider in the chain. Whole symbol, one source: the one-symbol-
    # one-source rule holds, nothing is spliced.
    if missing and len(chain) > 1:
        filler = chain[-1]
        print(f"filling {len(missing)} symbol(s) with no {used} data from "
              f"{filler.name} (full window)")
        try:
            extra = filler.fetch(missing, start, end)
        except Exception as exc:                           # noqa: BLE001
            extra = pd.DataFrame(columns=["date", "symbol", "px_last", "source"])
            print(f"  {filler.name} FAILED - {type(exc).__name__}: {exc}")
        if len(extra):
            extra["date"] = pd.to_datetime(extra["date"])
            prices = (pd.concat([prices, extra], ignore_index=True)
                      .drop_duplicates(subset=["symbol", "date"], keep="last")
                      .sort_values(["symbol", "date"]).reset_index(drop=True))
            prices.to_parquet(tmp, index=False)
            os.replace(tmp, OUT_PATH)
            filled = sorted(set(extra["symbol"]))
            print(f"  filled {len(filled)}: {', '.join(filled)} "
                  f"(stored as source={filler.name})")
            still = [s for s in missing if s not in filled]
            if still:
                print(f"  still no data: {', '.join(still)}")
        else:
            print(f"  {filler.name} had nothing for them either")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
