#!/usr/bin/env python
"""Pull daily closes into ``Data/prices/prices.parquet``.

    python Code/ingestion/pull_prices.py                       # provider from config/settings.csv
    python Code/ingestion/pull_prices.py --provider tiingo     # Tiingo API (TIINGO_API_KEY in .env)
    python Code/ingestion/pull_prices.py --provider bloomberg  # Terminal; falls back to Tiingo
    python Code/ingestion/pull_prices.py --daily               # theme anchor ETFs only (the light pull)
    python Code/ingestion/pull_prices.py --check               # report providers, store and plan; pull nothing
    python Code/ingestion/pull_prices.py --dry-run             # show the plan, pull nothing

Providers (see ``src/prices.py``): ``bloomberg`` uses ``blpapi``
``HistoricalDataRequest`` for ``PX_LAST`` against a running Terminal;
``tiingo`` uses Tiingo's daily closes, split-adjusted like ``PX_LAST``
(``TIINGO_API_KEY`` in ``.env``; a free key is issued at tiingo.com).
``auto`` picks Bloomberg when it can connect and Tiingo otherwise;
``bloomberg`` falls back to Tiingo if the pull fails unless
``--no-fallback`` is given.

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
Daily mode
    ``--daily`` prices the theme anchor ETFs alone - about 27 symbols,
    the smallest set that keeps every theme's line current - and extends
    a stored series with this run's provider instead of re-pulling it.
    A series may therefore carry one vendor's past and another's
    present; the ``source`` column records which rows came from where,
    and the run prints how many series that applies to.
Check mode
    ``--check`` writes nothing and connects only far enough to measure:
    each provider and why it is or is not usable, what the store holds
    and how stale it is, what a daily run and a full run would each cost
    against today's window, and one timed single-symbol round trip.
Output
    ``Data/prices/prices.parquet`` (long form): ``date``, ``symbol``,
    ``px_last``, ``source``. ``symbol`` is the plain ticker/ETF so the
    dashboard joins straight onto the mentions and signals tables. The
    file is not committed.

The pull is incremental (``plan_requests``): only the spans a symbol
does not already have are requested. Outside ``--daily``, a symbol whose
stored rows came from a different provider is re-pulled over the whole
window, and its stored rows inside that window are replaced by what came
back. Two limits keep the replacement from costing history: a symbol the
new provider returned nothing for keeps every row it had, and rows dated
before the window start are never in scope, so a series that reaches
further back than ``START_DATE`` keeps its older half. The seam that
leaves sits in the ``source`` column and is counted by
``tools/data_health.py``; the run prints how many series carry one.
"""

import argparse
import datetime
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The window and PRICE_TOP_N come from src/config.py, which itself honours
# the PIPELINE_START_DATE / PIPELINE_END_DATE env vars update_data.py sets -
# so a one-off --start/--end override reaches this script automatically.
from src.config import START_DATE, END_DATE, PRICE_TOP_N
from src.themes import THEME_ETFS, THEME_ETF_FALLBACKS       # theme -> ETF map
from src.config import DATA_DIR  # noqa: E402

PROCESSED = os.path.join(DATA_DIR, "processed")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
OUT_PATH = os.path.join(PRICES_DIR, "prices.parquet")

FIELD = "PX_LAST"
CHUNK = 50            # securities per Bloomberg request (keeps each request small)


# ---------------------------------------------------------------------------
# 1. Decide WHICH symbols to pull - from the same aggregates the dashboard reads
# ---------------------------------------------------------------------------
def _read(name):
    """Read a ``Data/processed`` parquet by name, or ``None`` if absent."""
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


def theme_anchor_symbols():
    """Return the sorted theme anchor ETFs - the ``--daily`` universe.

    Every theme is drawn against its anchor, so this is the smallest set
    that keeps the dashboard's theme lines current. The fallback
    anchors, the approved-instrument benchmarks, the top-mentioned
    ticker overlays and the single names are all left to a full pull:
    they change the picture over weeks, not over one close. Blanks and
    the ``"?"`` placeholder (a theme with no approved anchor) are
    dropped, as in the full universe.
    """
    return sorted({s.strip().upper() for s in THEME_ETFS.values()
                   if s and str(s).strip() and str(s).strip() != "?"})


# Symbol mapping lives in src/prices.py (to_bloomberg / to_tiingo), read
# from config/approved_instruments.csv. Adding a foreign line = one CSV row.
from src.prices import (to_bloomberg, to_tiingo, provider_chain,          # noqa: E402
                        fetch_with_fallback, ProviderUnavailable,
                        with_source, redact)
from src.themes import APPROVED_INSTRUMENTS                                # noqa: E402

FOREIGN_SECURITIES = {
    sym: row["bloomberg"]
    for sym, row in APPROVED_INSTRUMENTS.items()
    if row.get("bloomberg") and row["bloomberg"] != f"{sym} US Equity"
}

# The same mapping read the other way, for turning a returned security
# string back into the stored symbol. Built once: every message of every
# chunk of every span looks a security up in it.
SECURITY_TO_SYMBOL = {v: k for k, v in FOREIGN_SECURITIES.items()}


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
    symbol = SECURITY_TO_SYMBOL.get(
        security, security.replace(" US Equity", "").strip())

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


def _two_vendor_symbols(existing):
    """Sorted symbols whose stored rows carry more than one ``source``."""
    if (existing is None or not len(existing)
            or "source" not in existing.columns):
        return []
    n = (existing.dropna(subset=["source"]).groupby("symbol")["source"]
         .nunique())
    return sorted(n[n > 1].index)


def _plan_cost(symbols, start, end, existing):
    """``(spans, symbol-requests)`` the incremental planner would spend."""
    buckets = plan_requests(symbols, start, end, existing)
    return len(buckets), sum(len(v) for v in buckets.values())


def check_report(preference, daily=False, fallback=True, log=print):
    """Report the price step without writing anything.

    The price pull is the long step of a run, and the two things that
    make it long - whether a provider answers at all, and how many
    symbol-requests the plan holds - are only visible from inside it.
    This reports both, plus one timed single-symbol round trip, so the
    cost of a whole pull is a multiplication rather than a guess.

    Args:
        preference: ``auto``, ``bloomberg`` or ``tiingo``.
        daily: Whether the daily plan is the headline.
        fallback: Whether ``bloomberg`` carries Tiingo behind it.
        log: Line sink for the report.

    Returns:
        ``0`` when the report is complete, ``1`` when the store is on
        disk but cannot be read.
    """
    # The chain is assembled here rather than through provider_chain,
    # which keeps only the providers that answer and raises when none
    # do. A report needs the ones that do not, and the reason for each.
    from src.prices import BloombergProvider, TiingoProvider
    pref = (preference or "auto").strip().lower()
    bbg, tii = BloombergProvider(), TiingoProvider()
    if pref == "tiingo":
        chain = [tii]
    elif pref == "bloomberg":
        chain = [bbg] + ([tii] if fallback else [])
    else:
        chain = [bbg, tii]

    start, end = window_dates()
    log("=" * 64)
    log("PRICE CHECK  (connects only to measure; writes nothing)")
    log(f"  provider: {preference}")
    log(f"  window  : {start} -> {end}")
    log("-" * 64)

    # Every provider the preference would try, usable or not, with the
    # sentence available() gives for the ones that are not: that
    # sentence is the fix, so it is quoted rather than summarised.
    log("PROVIDERS")
    usable = []
    for prov in chain:
        ok, why = prov.available()
        if ok:
            usable.append(prov)
        log(f"  {prov.name:<10} {'usable' if ok else 'unusable':<9} {why}")

    log("STORE")
    rc = 0
    existing = None
    if not os.path.exists(OUT_PATH):
        log(f"  {OUT_PATH} is not on disk - every symbol is a full window")
    else:
        try:
            existing = pd.read_parquet(OUT_PATH)
            existing["date"] = pd.to_datetime(existing["date"])
        except Exception as exc:                           # noqa: BLE001
            log(f"  unreadable: {type(exc).__name__}: {redact(exc)}")
            existing, rc = None, 1
    if existing is not None and len(existing):
        newest = existing["date"].max()
        lag = (pd.Timestamp.today().normalize() - newest.normalize()).days
        two = _two_vendor_symbols(existing)
        log(f"  rows        : {len(existing):,}")
        log(f"  symbols     : {existing['symbol'].nunique()}")
        log(f"  span        : {existing['date'].min().date()} -> "
            f"{newest.date()}")
        log(f"  newest close: {lag} day(s) old")
        log(f"  two-vendor  : {len(two)} series carry more than one source"
            + (f" ({', '.join(two[:6])}{' ...' if len(two) > 6 else ''})"
               if two else ""))

    anchors = theme_anchor_symbols()
    full = build_symbol_universe()
    d_spans, d_req = _plan_cost(anchors, start, end, existing)
    f_spans, f_req = _plan_cost(full, start, end, existing)
    log(f"PLAN  (headline: {'daily' if daily else 'full'})")
    log(f"  daily : {len(anchors):>4} symbols | {d_spans} span(s) | "
        f"{d_req} symbol-request(s)")
    log(f"  full  : {len(full):>4} symbols | {f_spans} span(s) | "
        f"{f_req} symbol-request(s)")

    # One symbol over a short recent span against the live provider. It
    # is the unit the plan above is counted in, so seconds x
    # symbol-requests is what a pull costs on this machine and this
    # connection. No provider means no measurement, which is a line in
    # the report and not a failure of it.
    log("ROUND TRIP  (one symbol, live)")
    if not usable:
        log("  no usable provider - nothing to time")
    elif not anchors:
        log("  no theme anchor to ask for - nothing to time")
    else:
        prov = usable[0]
        # A line this provider has no symbol for is skipped rather than
        # requested, and timing a skip measures nothing, so the anchor
        # is one the provider can address.
        addressable = ([s for s in anchors if to_tiingo(s) is not None]
                       if prov.name == "tiingo" else anchors)
        sym = (addressable or anchors)[0]
        span_a = (pd.Timestamp(end) - pd.Timedelta(days=7)).strftime("%Y%m%d")
        t0 = time.time()
        try:
            got = prov.fetch([sym], span_a, end, log=lambda *_: None)
            log(f"  {sym} {span_a} -> {end} via {prov.name}: "
                f"{time.time() - t0:.2f}s, "
                f"{0 if got is None else len(got)} row(s)")
        except Exception as exc:                           # noqa: BLE001
            log(f"  {sym} via {prov.name} failed after "
                f"{time.time() - t0:.2f}s: {type(exc).__name__}: "
                f"{redact(exc)}")
    log("=" * 64)
    return rc


def main():
    """Build the universe, choose the provider, plan the spans, pull, save.

    Returns:
        ``0`` on success or when nothing needs fetching; ``1`` when there
        are no symbols or no provider returned rows.
    """
    from src import settings
    p = argparse.ArgumentParser(description="Pull daily close prices.")
    p.add_argument("--provider", choices=("auto", "bloomberg", "tiingo"),
                   default=None,
                   help="price source (default: price_provider in "
                        "config/settings.csv)")
    p.add_argument("--no-fallback", action="store_true",
                   help="with --provider bloomberg or auto, do not fall "
                        "back to Tiingo: neither when Bloomberg fails, "
                        "nor for symbols Bloomberg has no data for")
    p.add_argument("--dry-run", action="store_true",
                   help="show the symbol universe + request window; do NOT connect")
    p.add_argument("--force", action="store_true",
                   help="re-download the whole window even if already covered")
    p.add_argument("--daily", action="store_true",
                   help="price the theme anchor ETFs only (about 27 "
                        "symbols) and extend a stored series with this "
                        "run's provider instead of re-pulling it")
    p.add_argument("--check", action="store_true",
                   help="report the providers, the store, what a daily and "
                        "a full run would each cost, and one timed round "
                        "trip; connect only to measure and write nothing")
    args = p.parse_args()
    preference = args.provider or settings.get("price_provider")

    if args.check:
        return check_report(preference, daily=args.daily,
                            fallback=not args.no_fallback)

    symbols = theme_anchor_symbols() if args.daily else build_symbol_universe()
    start, end = window_dates()
    print("=" * 64)
    print("PRICE PULL" + ("  (daily: theme anchors only)" if args.daily else ""))
    print(f"  provider: {preference}"
          + ("" if args.no_fallback or preference == "tiingo"
             else "  (falls back to Tiingo on failure)"))
    print(f"  window  : {start} -> {end}  (from src/config.py / --start/--end)")
    if args.daily:
        print(f"  symbols : {len(symbols)}  (theme anchor ETFs)")
        print(f"            {', '.join(symbols)}")
    else:
        print(f"  symbols : {len(symbols)}  (top {PRICE_TOP_N} mentioned + theme ETFs + signals)")
        print(f"            {', '.join(symbols[:25])}{' ...' if len(symbols) > 25 else ''}")
    print("=" * 64)

    if args.dry_run:
        print("--dry-run: nothing pulled, nothing written.")
        return 0
    if not symbols:
        print("no symbols to pull - run update_data.py first so the aggregates exist.")
        return 1

    # No key, no Terminal, no requests: a plain sentence and a nonzero
    # exit, not a traceback. update_data.py treats this as non-fatal and
    # carries on with the closes already on disk.
    try:
        chain = provider_chain(preference, fallback=not args.no_fallback)
    except ProviderUnavailable as exc:
        print(redact(exc))
        return 1
    active = chain[0].name

    # INCREMENTAL: only fetch what prices.parquet does not already hold.
    existing = None
    if os.path.exists(OUT_PATH):
        existing = pd.read_parquet(OUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        # Every row that is kept is written back with a source on it,
        # rows whose columns predate provider tracking as bloomberg -
        # the convention _current_sources reads such a store under.
        existing = with_source(existing)
        print(f"existing store: {len(existing):,} rows, "
              f"{existing['symbol'].nunique()} symbols "
              f"({existing['date'].min().date()} -> {existing['date'].max().date()})")
        if args.force:
            # --force re-downloads the window for the symbols THIS RUN
            # asks for, so their coverage inside it goes and the planner
            # asks for the whole span again. Two things stay, for the
            # same reason the provider-change replacement leaves them:
            # rows for symbols outside this run's universe (a narrowed
            # universe - --daily is 27 of 447 - would otherwise write
            # the rest out of existence on its way past), and rows dated
            # before the window start, which are not what is being
            # re-downloaded and cannot be replaced by it.
            _in_win = existing["date"].between(pd.Timestamp(start),
                                               pd.Timestamp(end))
            kept = existing[~(existing["symbol"].isin(symbols) & _in_win)]
            _older = kept["symbol"].isin(symbols).sum()
            print(f"--force: re-downloading {len(symbols)} symbol(s) over "
                  f"{start} -> {end}; {len(kept):,} stored row(s) stay "
                  f"({_older:,} of them dated before the window)")
            existing = kept if len(kept) else None

    # ONE SYMBOL, ONE SOURCE: a symbol whose stored rows came from a
    # provider that is not in this run's chain is re-pulled over the
    # whole window, and the rows it had inside that window give way to
    # what comes back. A symbol stored from the FALLBACK provider is
    # kept when the fallback is still in the chain (it was filled from
    # there because the primary had nothing); `--provider bloomberg
    # --no-fallback` or `--force` moves such symbols back.
    sources = _current_sources(existing)
    chain_names = {c.name for c in chain}
    switching = sorted(s for s in symbols
                       if s in sources and sources[s] not in chain_names)
    # Symbols whose stored window is up for replacement once the fetch
    # has run, and what the PLANNER is shown. A symbol being re-pulled is
    # hidden from the planner so the whole window is requested rather
    # than the trailing gap, while `existing` itself stays whole: the
    # stored rows are what a symbol falls back on if the new provider
    # cannot answer, and dropping them here would spend them before
    # knowing whether anything replaces them.
    repull, plan_existing = [], existing
    if switching and existing is not None:
        if args.daily:
            # DAILY MODE EXTENDS RATHER THAN REPLACES. Re-pulling a full
            # window for every anchor is what makes a provider change
            # expensive, and the point of this mode is to spend a
            # handful of requests. The stored rows stay, the run's
            # provider fills the trailing gap, and the seam is written
            # into the source column of the rows it adds instead of
            # being smoothed over.
            print(f"{len(switching)} series carry two vendors after this "
                  f"run: stored as {sorted({sources[s] for s in switching})}"
                  f", extended with {active} from their newest close - "
                  f"{', '.join(switching[:10])}"
                  f"{' ...' if len(switching) > 10 else ''}")
        else:
            print(f"provider changed for {len(switching)} symbol(s) "
                  f"(stored as {sorted({sources[s] for s in switching})}, now "
                  f"{active}) - re-pulling their full window: "
                  f"{', '.join(switching[:10])}{' ...' if len(switching) > 10 else ''}")
            repull = switching
            plan_existing = existing[~existing["symbol"].isin(switching)]

    # Symbols already stored from the fallback provider keep being
    # extended from it (asking the primary would return nothing and
    # leave their series frozen).
    fb_syms = sorted(s for s in symbols
                     if sources.get(s) in chain_names and sources[s] != active)
    primary_syms = [s for s in symbols if s not in fb_syms]
    buckets = plan_requests(primary_syms, start, end, plan_existing)
    buckets_fb = (plan_requests(fb_syms, start, end, plan_existing)
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

    fetched = []
    used = None
    for (span_a, span_b), syms in sorted(buckets_fb.items()):
        print(f"  span {span_a} -> {span_b}: {len(syms)} symbols via "
              f"{chain[-1].name} (fallback-stored)")
        try:
            # One provider, but through the guarded path: a provider
            # already taken out of service is skipped on its
            # down_reason() rather than paying its connect timeout on
            # every one of these spans.
            got_fb, _ = fetch_with_fallback([chain[-1]], syms, span_a, span_b)
        except ProviderUnavailable as exc:
            print(f"  {redact(exc)}")
            continue
        if len(got_fb):
            got_fb["date"] = pd.to_datetime(got_fb["date"])
            fetched.append(got_fb)
    for (span_a, span_b), syms in sorted(buckets.items()):
        print(f"  span {span_a} -> {span_b}: {len(syms)} symbols")
        try:
            got, used_now = fetch_with_fallback(chain, syms, span_a, span_b)
        except ProviderUnavailable as exc:
            # Every provider raised on this span (no network, no session).
            # Stop here: nothing has been written, so the store on disk
            # keeps the closes it already had.
            print(f"  {redact(exc)}")
            return 1
        if used is not None and used_now != used:
            # a mid-run fallback would splice sources; keep to one per run
            print(f"provider changed mid-run ({used} -> {used_now}); "
                  "re-run so every span comes from one provider")
            return 1
        used = used_now
        if len(got):
            got["date"] = pd.to_datetime(got["date"])
            fetched.append(got)

    if used is None:                       # only fallback-stored spans ran
        used = chain[-1].name

    # REPLACE ONLY WHAT CAME BACK, ONLY WHERE IT CAME BACK. A re-pulled
    # symbol gives up its stored rows here, after the fetch rather than
    # before it, and on two conditions: the new provider answered for
    # that symbol, and the row sits inside the window that was asked
    # for. So a symbol nothing came back for keeps its whole series, and
    # a series reaching further back than the window start keeps
    # everything in front of it. Dropping first and fetching afterwards
    # bets a symbol's history on a request that has not been made yet,
    # and a provider that cannot address the line - a US ETF written
    # with an exchange code the mapping does not read, a name it does
    # not carry - collects that bet.
    answered = set()
    for frame in fetched:
        answered.update(frame["symbol"].unique())
    requested = {s for syms in list(buckets.values()) + list(buckets_fb.values())
                 for s in syms}
    kept_whole = []
    parts = list(fetched)
    if existing is not None and len(existing):
        keep = existing
        kept_whole = sorted((requested - answered)
                            & set(existing["symbol"].unique()))
        replaced = sorted(set(repull) & answered)
        if replaced:
            in_window = keep["date"].between(pd.Timestamp(start),
                                             pd.Timestamp(end))
            keep = keep[~(keep["symbol"].isin(replaced) & in_window)]
            seams = keep.loc[keep["symbol"].isin(replaced), "symbol"].nunique()
            print(f"re-pulled {len(replaced)} symbol(s) and replaced their "
                  f"{start} -> {end} rows; {seams} of them hold closes from "
                  f"before {start} and so carry two vendors from here on "
                  "(tools/data_health.py names them)")
        parts.insert(0, keep)
    if not parts:                          # nothing fetched, no store to keep
        print("no price rows returned - check the provider (Terminal logged in, "
              "or network access for Tiingo).")
        return 1
    prices = pd.concat(parts, ignore_index=True)
    if prices.empty:
        print("no price rows returned - check the provider (Terminal logged in, "
              "or network access for Tiingo).")
        return 1
    # Every row goes to disk with the provider that answered for it on
    # it; rows whose columns predate provider tracking are bloomberg.
    prices = with_source(prices)
    # a symbol/date can arrive twice at span edges - keep the newest fetch
    prices = (prices.drop_duplicates(subset=["symbol", "date"], keep="last")
              .sort_values(["symbol", "date"]).reset_index(drop=True))

    os.makedirs(PRICES_DIR, exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    prices.to_parquet(tmp, index=False)      # atomic swap - never half-written
    os.replace(tmp, OUT_PATH)
    print(f"saved {len(prices):,} rows for {prices['symbol'].nunique()} symbols "
          f"via {used} -> {OUT_PATH}")

    # COVERAGE REPORT - name every requested symbol that came back empty,
    # so "no price rows" in the overlays is never a mystery. The two
    # lists say different things and are printed apart: a symbol with a
    # stored series behind it is unchanged on disk, while a symbol in the
    # second list has no closes anywhere and is what the fallback below
    # goes after.
    got = set(prices["symbol"].unique())
    missing = [s for s in symbols if s not in got]
    if kept_whole:
        print(f"NO DATA for {len(kept_whole)} of {len(symbols)} requested "
              f"symbols that already have a series - {used} returned "
              "nothing for them and they keep every stored row:")
        print("  " + ", ".join(kept_whole))
    if missing:
        print(f"NO DATA for {len(missing)} of {len(symbols)} requested symbols "
              "with nothing stored either (delisted, non-US listing, younger "
              "than the window, or no Tiingo mapping):")
        print("  " + ", ".join(missing))

    # FILL FROM THE FALLBACK - a symbol with NO rows at all from the
    # active provider (an OTC line the Terminal is not entitled to, a
    # delisted name it does not serve) is pulled in full from the next
    # provider in the chain. Whole symbol, one source: the one-symbol-
    # one-source rule holds, nothing is spliced.
    if missing and len(chain) > 1:
        filler = chain[-1]
        print(f"filling {len(missing)} symbol(s) with no {used} data from "
              f"{filler.name} (full window)")
        try:
            # One provider, but through the guarded path: a provider
            # already taken out of service is skipped on its
            # down_reason() rather than paying its connect timeout.
            extra, _ = fetch_with_fallback([filler], missing, start, end)
        except ProviderUnavailable as exc:
            extra = pd.DataFrame(columns=["date", "symbol", "px_last", "source"])
            print(f"  {redact(exc)}")
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
