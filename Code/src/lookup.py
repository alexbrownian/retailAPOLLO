"""On-demand ETF lookup: assemble a theme for any ETF from the aggregates.

A configured theme is a set of tickers plus a set of words, defined in
``config/`` and counted at ingestion. This module builds the same two
sets for an ETF that is *not* configured, at query time, and reads its
attention and sentiment history straight from the text-free aggregates:

* tickers: the ETF's top holdings (from the Terminal, or typed), kept
  where the ticker extractor knows the symbol, summed over
  ``daily_ticker_counts`` / ``daily_ticker_sentiment``;
* words: the ETF's name and its holdings' company names, reduced to the
  tokens a person actually types (``brazil``, ``uranium``), summed over
  ``daily_term_counts`` (which keeps a rolling year);
* price: pulled through :mod:`src.prices` (Bloomberg or Tiingo)
  and cached under ``Data/prices/lookups/``.

The result is scored with the rule-based crowd measures the dashboard's
gauge already uses (:func:`analytics.euphoria.compute_euphoria` - the
0-100 euphoria level and its bands come from the frozen record). The
walk-forward model is *not* applied to a lookup: the ETF is not in the
validated universe, so a "% of the way to a signal" for it would be a
number with no record behind it. The panel says so.

Nothing here writes to ``config/``; promoting a lookup to a real theme
is a deliberate edit (``research.ipynb``).

Network calls (Terminal search, holdings, price) are isolated in small
functions so the assembly and scoring are testable offline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.config import PRICES_DIR, REFERENCE_DIR, ROLL

LOOKUP_PRICE_DIR = os.path.join(PRICES_DIR, "lookups")
PRICE_CACHE_HOURS = 20          # one pull per trading day is plenty

# Words that identify the issuer or the wrapper, never the theme.
_NOISE = {
    "etf", "etfs", "fund", "funds", "trust", "index", "shares", "share",
    "ishares", "spdr", "vanguard", "invesco", "proshares", "direxion",
    "global", "x", "wisdomtree", "vaneck", "ark", "first", "select",
    "sector", "portfolio", "ucits", "acc", "dist", "usd", "msci", "ftse",
    "s&p", "sp", "500", "russell", "nasdaq", "nyse", "dow", "jones",
    "inc", "corp", "corporation", "co", "ltd", "plc", "sa", "nv", "ag",
    "holdings", "holding", "group", "company", "companies", "class",
    "common", "stock", "ordinary", "adr", "the", "and", "of", "de",
    "capped", "equal", "weight", "weighted", "total", "market", "core",
    "leveraged", "bull", "bear", "2x", "3x", "daily", "ultra", "short",
    "long", "strategy", "active", "enhanced", "dividend", "income",
    "growth", "value", "small", "mid", "large", "cap", "smallcap",
    "midcap", "largecap", "all", "world", "international", "ex", "us",
    "u.s.", "united", "states", "america", "american", "north",
    "configured", "theme",
}
_WORD = re.compile(r"[a-z0-9&.-]+")


@dataclass
class Candidate:
    symbol: str
    name: str
    origin: str          # approved | catalogue | ticker | search (Terminal)


@dataclass
class Holding:
    ticker: str
    name: str
    weight: float
    known: bool          # does the ticker extractor know this symbol?


@dataclass
class LookupSpec:
    symbol: str
    name: str
    holdings: List[Holding] = field(default_factory=list)
    words: List[str] = field(default_factory=list)

    @property
    def known_tickers(self) -> List[str]:
        return [h.ticker for h in self.holdings if h.known]


# ---------------------------------------------------------------------------
# 1. resolve a free-text query to an ETF
# ---------------------------------------------------------------------------
def _approved() -> pd.DataFrame:
    from src.config import ROOT
    path = os.path.join(ROOT, "config", "approved_instruments.csv")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return df


_TICKER_RX = re.compile(r"^[A-Za-z][A-Za-z.]{0,5}$")


def _catalogue() -> pd.DataFrame:
    """``config/etf_catalogue.csv``: a convenience list of common ETFs
    (symbol, name, keywords) so a name search works with no network."""
    from src.config import ROOT
    path = os.path.join(ROOT, "config", "etf_catalogue.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["symbol", "name", "keywords"])
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")


def search(query: str, limit: int = 8, remote=None,
           status: Optional[dict] = None) -> List[Candidate]:
    """ETF candidates for ``query``.

    Order: configured themes and instruments, the local ETF catalogue,
    the Terminal's security lookup, and the query itself when it looks
    like a ticker and nothing else matched. ``remote`` is the search
    callable (injected in tests); it defaults to :func:`_remote_search`. When ``status`` is a dict, the
    remote outcome is recorded in it (``remote_error``: message or
    None), so a caller can say that the search service was down rather
    than that nothing matched.
    """
    q = (query or "").strip()
    if not q:
        return []
    out: List[Candidate] = []
    seen = set()
    ql = q.lower()
    # configured themes first: "gold" -> gold_metals -> GLD
    try:
        from src.config import ROOT
        themes = pd.read_csv(os.path.join(ROOT, "config", "theme_etfs.csv"),
                             encoding="utf-8-sig", dtype=str).fillna("")
    except Exception:                                      # noqa: BLE001
        themes = pd.DataFrame(columns=["theme", "etf"])
    approved_names = {r["symbol"]: r.get("name", "")
                      for _, r in _approved().iterrows()}
    for _, r in themes.iterrows():
        theme, etf = r.get("theme", ""), r.get("etf", "")
        label = theme.replace("_", " ")
        if etf and (ql in label.lower() or ql in etf.lower()) \
                and etf.upper() not in seen:
            nm = approved_names.get(etf) or f"{label} (configured theme)"
            out.append(Candidate(etf, nm, "approved"))
            seen.add(etf.upper())
    for sym, name in approved_names.items():
        if sym.upper() in seen:
            continue
        if ql in sym.lower() or (name and ql in name.lower()):
            out.append(Candidate(sym, name or sym, "approved"))
            seen.add(sym.upper())
    # the local catalogue: symbol, name or keyword match
    for _, r in _catalogue().iterrows():
        sym, name, kw = r["symbol"], r["name"], r["keywords"]
        if sym.upper() in seen:
            continue
        if ql == sym.lower() or ql in name.lower() or any(
                ql in k for k in kw.lower().split("|") if k):
            out.append(Candidate(sym, name, "catalogue"))
            seen.add(sym.upper())
    remote = remote or _remote_search
    try:
        for c in remote(q, limit):
            if c.symbol.upper() not in seen:
                out.append(c)
                seen.add(c.symbol.upper())
        if status is not None:
            status["remote_error"] = None
    except Exception as exc:                               # noqa: BLE001
        if status is not None:
            status["remote_error"] = f"{type(exc).__name__}: {exc}"[:200]
    # nothing matched anywhere and the text looks like a ticker: offer it
    # as one; the holdings and price steps will say whether it resolves
    if not out and _TICKER_RX.match(q):
        out.append(Candidate(q.upper(), q.upper(), "ticker"))
    # exact symbol matches float to the top
    out.sort(key=lambda c: (c.symbol.upper() != q.upper(),))
    return out[:limit]


# ---------------------------------------------------------------------------
# remote services: the Terminal when one answers, nothing otherwise
# ---------------------------------------------------------------------------
_BBG_AVAILABLE: Optional[bool] = None


def bloomberg_available(refresh: bool = False) -> bool:
    """Cached per process: does a Terminal answer? (a session start is
    a second or two, too slow to repeat on every rerun)."""
    global _BBG_AVAILABLE
    if _BBG_AVAILABLE is None or refresh:
        try:
            from src.prices import BloombergProvider
            _BBG_AVAILABLE = bool(BloombergProvider().available()[0])
        except Exception:                                  # noqa: BLE001
            _BBG_AVAILABLE = False
    return _BBG_AVAILABLE


def _bbg_session():
    import blpapi
    session = blpapi.Session()
    if not session.start():
        raise RuntimeError("could not start a blpapi session")
    return session


def _bbg_drain(session, request):
    """Send ``request`` and return every message until RESPONSE."""
    import blpapi
    session.sendRequest(request)
    msgs = []
    while True:
        ev = session.nextEvent(500)
        for msg in ev:
            msgs.append(msg)
        if ev.eventType() == blpapi.Event.RESPONSE:
            return msgs


def _bloomberg_search(query: str, limit: int) -> List[Candidate]:
    """Security lookup on ``//blp/instruments`` (the Terminal's own
    search), equities only; ETF names are returned as descriptions."""
    session = _bbg_session()
    try:
        if not session.openService("//blp/instruments"):
            raise RuntimeError("could not open //blp/instruments")
        svc = session.getService("//blp/instruments")
        req = svc.createRequest("instrumentListRequest")
        req.set("query", query)
        req.set("yellowKeyFilter", "YK_FILTER_EQTY")
        req.set("maxResults", max(limit * 3, 15))
        out = []
        for msg in _bbg_drain(session, req):
            if not msg.hasElement("results"):
                continue
            res = msg.getElement("results")
            for i in range(res.numValues()):
                r = res.getValueAsElement(i)
                sec = r.getElementAsString("security") if r.hasElement("security") else ""
                desc = r.getElementAsString("description") if r.hasElement("description") else ""
                # 'INDA US<equity>' -> INDA ; keep US lines as plain tickers,
                # others as 'TICKER XX' pipeline symbols
                sym = sec.split("<")[0].strip()
                parts = sym.split()
                if len(parts) >= 2 and parts[-1].upper() == "US":
                    sym = parts[0]
                if sym:
                    out.append(Candidate(sym, desc or sym, "search"))
        return out[:limit]
    finally:
        session.stop()


def _remote_search(query: str, limit: int) -> List[Candidate]:
    """The Terminal's security lookup when a session answers; otherwise
    nothing (the local catalogue and a typed ticker cover the rest).
    Raises when the Terminal is reachable but the request fails, so the
    caller can report it."""
    if not bloomberg_available():
        raise RuntimeError("no Bloomberg Terminal; search is local only")
    return _bloomberg_search(query, limit)


# ---------------------------------------------------------------------------
# 2. holdings and words
# ---------------------------------------------------------------------------
def _known_universe() -> set:
    """The extractor's symbol universe from the cached exchange lists;
    never downloads (a stale cache is fine for a known/unknown flag)."""
    from pathlib import Path
    try:
        from src.ticker_universe import load_us_ticker_universe
        return set(load_us_ticker_universe(Path(REFERENCE_DIR),
                                           max_cache_age_days=10_000))
    except Exception:                                      # noqa: BLE001
        return set()


def _clean_ticker(raw: str) -> str:
    """Vendor holdings symbols: ``BRK-B`` -> ``BRK.B``; ``PETR4.SA`` stays
    foreign (and unknown to the extractor)."""
    t = (raw or "").strip().upper()
    if "." in t and t.split(".")[-1].isalpha() and len(t.split(".")[-1]) == 2:
        return t                            # exchange-suffixed foreign line
    return t.replace("-", ".")


def holdings(symbol: str, remote=None, universe: Optional[set] = None,
             present: Optional[set] = None,
             status: Optional[dict] = None) -> List[Holding]:
    """Top holdings of ``symbol`` with a ``known`` flag per ticker: known
    means the extractor's universe carries it or it already has rows in
    the ticker aggregates (``present``)."""
    remote = remote or _remote_holdings
    uni = _known_universe() if universe is None else universe
    uni = set(uni) | set(present or ())
    try:
        rows = remote(symbol)
        if status is not None:
            status["remote_error"] = None
    except Exception as exc:                               # noqa: BLE001
        rows = []
        if status is not None:
            status["remote_error"] = f"{type(exc).__name__}: {exc}"[:200]
    out = []
    for r in rows:
        t = _clean_ticker(r.get("ticker", ""))
        if not t:
            continue
        out.append(Holding(ticker=t, name=str(r.get("name", "")),
                           weight=float(r.get("weight") or 0.0),
                           known=t in uni))
    return out


def manual_holdings(text: str, universe: Optional[set] = None,
                    present: Optional[set] = None) -> List[Holding]:
    """Holdings typed by hand (comma-separated tickers), equal weight."""
    uni = set(_known_universe() if universe is None else universe)
    uni |= set(present or ())
    ticks = [_clean_ticker(t) for t in re.split(r"[,\s]+", text or "") if t.strip()]
    ticks = [t for t in dict.fromkeys(ticks) if t]
    w = 1.0 / len(ticks) if ticks else 0.0
    return [Holding(t, t, w, t in uni) for t in ticks]


# Bloomberg bulk fields tried, in order, for an ETF's holdings. Field
# availability depends on the fund and the entitlement, so each is
# attempted and the first that returns member rows wins; the benchmark
# route (the fund's primary index, then its members) is the last resort.
BBG_HOLDINGS_FIELDS = ("FUND_HOLDINGS", "FUND_TOP_10_HOLDINGS")
BBG_BENCHMARK_FIELD = "FUND_BENCHMARK_PRIM"
BBG_INDEX_MEMBERS_FIELD = "INDX_MWEIGHT"


def _bbg_bulk(session, security: str, field: str) -> List[dict]:
    """One ReferenceDataRequest bulk field -> list of row dicts."""
    svc = session.getService("//blp/refdata")
    req = svc.createRequest("ReferenceDataRequest")
    req.getElement("securities").appendValue(security)
    req.getElement("fields").appendValue(field)
    rows = []
    for msg in _bbg_drain(session, req):
        if not msg.hasElement("securityData"):
            continue
        sd = msg.getElement("securityData")
        for i in range(sd.numValues()):
            sec = sd.getValueAsElement(i)
            if not sec.hasElement("fieldData"):
                continue
            fd = sec.getElement("fieldData")
            if not fd.hasElement(field):
                continue
            bulk = fd.getElement(field)
            for j in range(bulk.numValues()):
                el = bulk.getValueAsElement(j)
                row = {}
                for k in range(el.numElements()):
                    sub = el.getElement(k)
                    row[str(sub.name())] = sub.getValueAsString() if sub.numValues() else ""
                rows.append(row)
    return rows


def _bbg_rows_to_holdings(rows: List[dict]) -> List[dict]:
    """Pick the ticker/name/weight out of whatever a bulk field returned."""
    out = []
    for r in rows:
        keys = {k.lower(): k for k in r}
        tk = next((r[keys[k]] for k in keys
                   if any(w in k for w in ("ticker", "member", "security", "id"))), "")
        wt = next((r[keys[k]] for k in keys
                   if any(w in k for w in ("weight", "percent", "pct"))), "")
        nm = next((r[keys[k]] for k in keys
                   if any(w in k for w in ("name", "description"))), "")
        tk = str(tk).replace(" Equity", "").strip()
        parts = tk.split()
        if len(parts) >= 2 and parts[-1].upper() == "US":
            tk = parts[0]
        try:
            w = float(str(wt).replace("%", "").strip() or 0.0)
        except ValueError:
            w = 0.0
        if w > 1.5:                       # given in percent, not fraction
            w = w / 100.0
        if tk:
            out.append({"ticker": tk, "name": nm or tk, "weight": w})
    return out


def _bloomberg_holdings(symbol: str) -> List[dict]:
    from src.prices import to_bloomberg
    session = _bbg_session()
    try:
        if not session.openService("//blp/refdata"):
            raise RuntimeError("could not open //blp/refdata")
        sec = to_bloomberg(symbol)
        for field in BBG_HOLDINGS_FIELDS:
            got = _bbg_rows_to_holdings(_bbg_bulk(session, sec, field))
            if got:
                return got
        # benchmark route: the fund's primary index, then its members
        svc = session.getService("//blp/refdata")
        req = svc.createRequest("ReferenceDataRequest")
        req.getElement("securities").appendValue(sec)
        req.getElement("fields").appendValue(BBG_BENCHMARK_FIELD)
        bench = ""
        for msg in _bbg_drain(session, req):
            if msg.hasElement("securityData"):
                sd = msg.getElement("securityData")
                for i in range(sd.numValues()):
                    fd = sd.getValueAsElement(i).getElement("fieldData")
                    if fd.hasElement(BBG_BENCHMARK_FIELD):
                        bench = fd.getElementAsString(BBG_BENCHMARK_FIELD)
        if bench:
            if not bench.upper().endswith("INDEX"):
                bench = bench + " Index"
            return _bbg_rows_to_holdings(
                _bbg_bulk(session, bench, BBG_INDEX_MEMBERS_FIELD))
        return []
    finally:
        session.stop()


def _remote_holdings(symbol: str) -> List[dict]:
    """Holdings from the Terminal when a session answers; otherwise
    nothing (the panel offers a box to type them)."""
    if not bloomberg_available():
        raise RuntimeError("no Bloomberg Terminal; type the holdings")
    return _bloomberg_holdings(symbol)


def _tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall((text or "").lower())
            if len(t) > 2 and t not in _NOISE and not t.isdigit()]


def words(name: str, holds: Iterable[Holding], extra: str = "") -> List[str]:
    """The word list for a lookup: name tokens, each holding's first
    distinctive token, and anything in ``extra`` (a description). Order:
    name words first, then holdings by weight. Deduplicated."""
    out: List[str] = []
    for t in _tokens(name):
        if t not in out:
            out.append(t)
    for h in sorted(holds, key=lambda h: -h.weight):
        toks = _tokens(h.name)
        if toks and toks[0] not in out:
            out.append(toks[0])
    for t in _tokens(extra):
        if t not in out:
            out.append(t)
    return out


def expand_words_with_ai(name: str, base: List[str], n: int = 15) -> List[str]:
    """Ask the configured model for the words retail investors use for
    this theme. Words only - every number still comes from the stores.
    Returns ``base`` unchanged when no provider is configured."""
    try:
        from src import ai
        if not ai.available():
            return list(base)
        prompt = (f"ETF: {name}. Existing keywords: {', '.join(base)}. "
                  f"List up to {n} additional lower-case words or short "
                  "phrases retail investors on Reddit use when they talk "
                  "about this theme (company nicknames, slang, related "
                  "tickers without $). Reply as a JSON list of strings.")
        got = ai.chat(prompt, want_json=True, max_tokens=300)
        extra = [str(w).strip().lower() for w in (got or []) if str(w).strip()]
    except Exception:                                      # noqa: BLE001
        return list(base)
    out = list(base)
    for w in extra:
        if w not in out and 2 < len(w) <= 30:
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# 3. assemble the history from the aggregates
# ---------------------------------------------------------------------------
def assemble(spec: LookupSpec, tick_counts: pd.DataFrame,
             tick_sent: pd.DataFrame, term_counts: Optional[pd.DataFrame]
             ) -> pd.DataFrame:
    """Daily frame for the lookup: ``date, ticker_mentions, term_mentions,
    mention_count, n_posts, avg_sentiment, net_bullish``.

    ``mention_count`` is the sum of the two components, which is what a
    configured theme's count is (a post is tagged by a keyword OR a
    ticker). Sentiment comes from the holdings only: the term table
    carries no sentiment.
    """
    ticks = spec.known_tickers
    tc = tick_counts[tick_counts["ticker"].isin(ticks)]
    by_day = (tc.groupby("date")["mention_count"].sum()
              .rename("ticker_mentions"))
    ts = tick_sent[tick_sent["ticker"].isin(ticks)].copy()
    if len(ts):
        ts["_w"] = ts["avg_sentiment"] * ts["n_posts"]
        ts["_b"] = ts["net_bullish"] * ts["n_posts"]
        g = ts.groupby("date").agg(n_posts=("n_posts", "sum"),
                                   _w=("_w", "sum"), _b=("_b", "sum"))
        g["avg_sentiment"] = g["_w"] / g["n_posts"].replace(0, pd.NA)
        g["net_bullish"] = g["_b"] / g["n_posts"].replace(0, pd.NA)
        sent = g[["n_posts", "avg_sentiment", "net_bullish"]]
    else:
        sent = pd.DataFrame(columns=["n_posts", "avg_sentiment", "net_bullish"])
    if term_counts is not None and len(term_counts) and spec.words:
        tm = term_counts[term_counts["term"].isin([w.lower() for w in spec.words])]
        terms = (tm.groupby("date")["mention_count"].sum()
                 .rename("term_mentions"))
    else:
        terms = pd.Series(dtype=float, name="term_mentions")
    out = pd.concat([by_day, terms, sent], axis=1).sort_index()
    out.index.name = "date"
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"])
    for c in ("ticker_mentions", "term_mentions", "n_posts"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    out["mention_count"] = out["ticker_mentions"] + out["term_mentions"]
    out["avg_sentiment"] = pd.to_numeric(out["avg_sentiment"], errors="coerce")
    out["net_bullish"] = pd.to_numeric(out["net_bullish"], errors="coerce")
    return out[["date", "ticker_mentions", "term_mentions", "mention_count",
                "n_posts", "avg_sentiment", "net_bullish"]]


def euphoria_series(spec: LookupSpec, frame: pd.DataFrame,
                    theme_counts: pd.DataFrame, theme_sent: pd.DataFrame):
    """Run the rule-based crowd measures on the lookup as if it were one
    more theme (its share is measured against the configured themes'
    total, exactly as a configured theme's is). Returns an
    ``EuphoriaSeries`` or None when there is not enough history."""
    from src.analytics.euphoria import compute_euphoria
    name = f"lookup:{spec.symbol}"
    add_c = frame[["date", "mention_count"]].assign(theme=name)
    add_s = (frame[["date", "n_posts", "avg_sentiment", "net_bullish"]]
             .dropna(subset=["avg_sentiment"]).assign(theme=name))
    counts = pd.concat([theme_counts[["date", "theme", "mention_count"]],
                        add_c[["date", "theme", "mention_count"]]],
                       ignore_index=True)
    sents = pd.concat([theme_sent[["date", "theme", "n_posts",
                                   "avg_sentiment", "net_bullish"]],
                       add_s[["date", "theme", "n_posts", "avg_sentiment",
                              "net_bullish"]]], ignore_index=True)
    counts["date"] = pd.to_datetime(counts["date"])
    sents["date"] = pd.to_datetime(sents["date"])
    return compute_euphoria(name, spec.symbol, "theme", counts, sents, "theme")


def discussion_share(frame: pd.DataFrame, theme_counts: pd.DataFrame,
                     as_of=None) -> Optional[float]:
    """The lookup's share of theme discussion over the trailing ROLL
    days, on the same footing as the dashboard's share line."""
    if frame.empty:
        return None
    as_of = pd.Timestamp(as_of) if as_of is not None else frame["date"].max()
    lo = as_of - pd.Timedelta(days=ROLL - 1)
    mine = frame[(frame["date"] >= lo) & (frame["date"] <= as_of)]["mention_count"].sum()
    tc = theme_counts.copy()
    tc["date"] = pd.to_datetime(tc["date"])
    total = tc[(tc["date"] >= lo) & (tc["date"] <= as_of)]["mention_count"].sum()
    if total + mine <= 0:
        return None
    return 100.0 * mine / (total + mine)


# ---------------------------------------------------------------------------
# 4. price, cached
# ---------------------------------------------------------------------------
def price(symbol: str, start: str, end: str, provider: str = "auto",
          fetch=None, now=None) -> pd.DataFrame:
    """Daily closes for ``symbol``, cached per day under
    ``Data/prices/lookups/``. ``fetch(symbol, start, end)`` is injected in
    tests; the default goes through :mod:`src.prices`."""
    # a symbol the pipeline already prices needs no second pull
    try:
        from src.config import PRICES_PATH
        if os.path.exists(PRICES_PATH):
            store = pd.read_parquet(PRICES_PATH)
            mine = store[store["symbol"] == symbol]
            if len(mine):
                mine = mine.copy()
                mine["date"] = pd.to_datetime(mine["date"])
                if "source" not in mine.columns:
                    mine["source"] = "bloomberg"
                return mine[["date", "symbol", "px_last", "source"]]
    except Exception:                                      # noqa: BLE001
        pass
    os.makedirs(LOOKUP_PRICE_DIR, exist_ok=True)
    path = os.path.join(LOOKUP_PRICE_DIR, f"{symbol.replace('/', '_')}.parquet")
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    if os.path.exists(path):
        cached = pd.read_parquet(path)
        age_h = (now.timestamp() - os.path.getmtime(path)) / 3600.0
        # a cache is reused only while it is fresh AND already reaches the
        # last completed trading day; a series that stops earlier was
        # pulled before that close existed and must be refreshed
        if age_h < PRICE_CACHE_HOURS and len(cached) and \
                pd.to_datetime(cached["date"]).max() >= last_trading_day(now):
            return cached
    if fetch is None:
        def fetch(sym, a, b):
            from src.prices import fetch_with_fallback, provider_chain
            chain = provider_chain(provider, log=lambda *_: None)
            df, _ = fetch_with_fallback(chain, [sym], a, b, log=lambda *_: None)
            return df
    df = fetch(symbol, start, end)
    if df is None or not len(df):
        return pd.DataFrame(columns=["date", "symbol", "px_last", "source"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(path, index=False)
    return df


def last_trading_day(now=None) -> pd.Timestamp:
    """The most recent weekday whose close can exist: yesterday on a
    weekday morning, Friday over the weekend. Exchange holidays are not
    modelled - on the day after one the cache is refreshed once more
    than strictly needed, which is the cheap side to err on."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    d = now.normalize() - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def clear_price_cache(symbol: Optional[str] = None) -> None:
    if not os.path.isdir(LOOKUP_PRICE_DIR):
        return
    for f in os.listdir(LOOKUP_PRICE_DIR):
        if symbol is None or f == f"{symbol}.parquet":
            os.remove(os.path.join(LOOKUP_PRICE_DIR, f))


# ---------------------------------------------------------------------------
# 5. the production model, applied read-only
# ---------------------------------------------------------------------------
def _theme_frames(spec: LookupSpec, frame: pd.DataFrame,
                  theme_counts: pd.DataFrame, theme_sent: pd.DataFrame):
    """The configured themes' counts/sentiment with the lookup appended
    as one more theme (its share is measured against the same total a
    configured theme's is)."""
    name = f"lookup:{spec.symbol}"
    add_c = frame[["date", "mention_count"]].assign(theme=name)
    add_s = (frame[["date", "n_posts", "avg_sentiment", "net_bullish"]]
             .dropna(subset=["avg_sentiment"]).assign(theme=name))
    counts = pd.concat([theme_counts[["date", "theme", "mention_count"]],
                        add_c[["date", "theme", "mention_count"]]],
                       ignore_index=True)
    sents = pd.concat([theme_sent[["date", "theme", "n_posts",
                                   "avg_sentiment", "net_bullish"]],
                       add_s[["date", "theme", "n_posts", "avg_sentiment",
                              "net_bullish"]]], ignore_index=True)
    counts["date"] = pd.to_datetime(counts["date"])
    sents["date"] = pd.to_datetime(sents["date"])
    return name, counts, sents


def model_bundle_path() -> str:
    from src.config import PROCESSED_DIR
    from src.analytics.ml_detector import DESK_BUNDLE
    return os.path.join(PROCESSED_DIR, DESK_BUNDLE)


def model_score(spec: LookupSpec, es, frame: pd.DataFrame, px: pd.DataFrame,
                theme_counts: pd.DataFrame, theme_sent: pd.DataFrame,
                bundle: Optional[dict] = None) -> dict:
    """Score the lookup with the persisted production model.

    Nothing is fitted or re-thresholded: the fitted ensemble and the
    frozen cuts are read from ``euphoria_desk_model.joblib`` (written by
    the last analytics pass) and applied to the lookup's own feature
    days, built by the same ``build_day_frame`` the pipeline uses.

    Returns a dict with ``status`` (``ok``, ``no_model``, ``no_price``,
    ``not_eligible``), and when ok: ``days`` (DataFrame date, in_score,
    out_score, boomed120), ``now`` (the newest scored day's readiness:
    pct, side, score, cut, date) and ``thresholds``.
    """
    from src.analytics.ml_detector import (attach_price_features, candidate_frame,
                                       load_desk_bundle, score_with_bundle)
    from src.analytics.euphoria_phases import build_day_frame, boomed120_frame
    bundle = bundle or load_desk_bundle(model_bundle_path())
    if bundle is None:
        return {"status": "no_model",
                "why": "no fitted model on disk yet - run "
                       "`python -m src.analytics.run_analytics --what phases`"}
    if es is None or frame.empty:
        return {"status": "not_eligible", "why": "not enough crowd history"}
    if px is None or not len(px):
        return {"status": "no_price", "why": "no price series for the symbol"}
    pxs = (px.assign(date=pd.to_datetime(px["date"]))
             .sort_values("date").set_index("date")["px_last"]
             .asfreq("D").ffill())
    pxmap = {spec.symbol: pxs}
    name, counts, sents = _theme_frames(spec, frame, theme_counts, theme_sent)
    ep_path = os.path.join(os.path.dirname(model_bundle_path()), "episodes.parquet")
    episodes = (pd.read_parquet(ep_path) if os.path.exists(ep_path)
                else pd.DataFrame(columns=["name", "onset_lo", "onset_hi", "peak"]))
    day = build_day_frame([es], pxmap, episodes, {"theme": counts},
                          {"theme": sents}, clip_judgeable=False)
    if day.empty:
        return {"status": "not_eligible",
                "why": "coverage gate (100 tagged posts / 28 days) or the "
                       "180-day history floor is not met on any day"}
    day = candidate_frame(attach_price_features(day, [es], pxmap))
    if day.empty:
        return {"status": "not_eligible",
                "why": "no day with both crowd features and price features"}
    day = day.sort_values("date").reset_index(drop=True)
    out = day[["date"]].copy()
    out["in_score"] = score_with_bundle(bundle, "y_onset", day)
    out["out_score"] = score_with_bundle(bundle, "y_top", day)
    b120 = boomed120_frame([es], pxmap)
    out = out.merge(b120[["date", "boomed120"]], on="date", how="left")
    out["boomed120"] = out["boomed120"].eq(True)
    thr_in = float(bundle["meta"]["thr_in"])
    thr_out = float(bundle["meta"]["thr_out"])
    last = out.iloc[-1]
    if bool(last["boomed120"]):
        side, sc, thr = "CUT EXPOSURE", float(last["out_score"]), thr_out
    else:
        side, sc, thr = "INCREASE EXPOSURE", float(last["in_score"]), thr_in
    return {"status": "ok", "days": out,
            "now": {"pct": 100.0 * sc / thr, "side": side, "score": sc,
                    "cut": thr, "date": pd.Timestamp(last["date"]),
                    "boomed": bool(last["boomed120"])},
            "thresholds": {"in": thr_in, "out": thr_out},
            "model": bundle.get("model", "?"), "built": bundle.get("built"),
            "eligible_days": int(len(out))}


# ---------------------------------------------------------------------------
# command line: exercise the services for one query, write nothing
# ---------------------------------------------------------------------------
def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="ETF lookup self-test "
                                             "(search, holdings, words).")
    ap.add_argument("query", help="ETF name, theme word or ticker")
    ap.add_argument("--no-terminal", action="store_true",
                    help="skip Bloomberg even if a Terminal answers")
    a = ap.parse_args(argv)
    global _BBG_AVAILABLE
    if a.no_terminal:
        _BBG_AVAILABLE = False
    print(f"bloomberg: {'available' if bloomberg_available() else 'not available'}")
    st = {}
    cands = search(a.query, status=st)
    print(f"search {a.query!r}: {len(cands)} candidate(s)"
          + (f"  [remote error: {st.get('remote_error')}]" if st.get("remote_error") else ""))
    for c in cands:
        print(f"  {c.symbol:<10} {c.name[:50]:<50} ({c.origin})")
    if not cands:
        return 1
    sym = cands[0].symbol
    st = {}
    hs = holdings(sym, status=st)
    print(f"holdings {sym}: {len(hs)}"
          + (f"  [remote error: {st.get('remote_error')}]" if st.get("remote_error") else ""))
    for h in hs[:15]:
        print(f"  {h.ticker:<10} {h.name[:40]:<40} {100 * h.weight:5.1f}%  "
              f"{'counted' if h.known else 'not counted'}")
    print("words:", ", ".join(words(cands[0].name, hs)))
    print("model bundle:", "present" if os.path.exists(model_bundle_path())
          else "absent (run src.analytics.run_analytics --what phases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
