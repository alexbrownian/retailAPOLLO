"""Price providers: Bloomberg and Tiingo behind one interface.

The pipeline needs daily closes for every priced instrument in
``Data/prices/prices.parquet`` (long form: ``date``, ``symbol``,
``px_last``, ``source``). Two providers can fill it:

``bloomberg``
    ``blpapi`` ``HistoricalDataRequest`` for ``PX_LAST`` against a
    running, logged-in Terminal. Securities are addressed by the code in
    ``config/approved_instruments.csv`` (``bloomberg`` column) or
    ``"<SYMBOL> US Equity"`` by default.

``tiingo``
    Tiingo's daily-price API (api.tiingo.com), one request per symbol,
    authenticated with ``TIINGO_API_KEY`` from ``.env`` or the
    environment (a free key is issued at tiingo.com). The stored close is
    split-adjusted but not dividend-adjusted - the same convention as
    ``PX_LAST`` - built from Tiingo's raw close and per-day split
    factors. US lines map to the plain ticker (``BRK.B`` -> ``BRK-B``);
    non-US instruments need a symbol in the ``tiingo`` column of
    ``config/approved_instruments.csv`` or are skipped and reported.
    The provider paces itself and stops with a clear error when the
    account's request quota is hit (HTTP 429).

Why an API and not a free site: Yahoo Finance rate-limits by address
and Stooq fronts its CSV endpoint with a JavaScript bot challenge; both
refuse programs from a shared office network. A key is the permission.

Selection and fallback
----------------------
:func:`provider_chain` turns a preference into an ordered list:

* ``auto``       -> Bloomberg if it can connect, otherwise Tiingo;
* ``bloomberg``  -> Bloomberg, falling back to Tiingo if the pull
  fails (``fallback=False`` disables the fallback);
* ``tiingo``     -> Tiingo only.

The preference comes from ``--provider`` on ``pull_prices.py`` /
``update_data.py`` or ``price_provider`` in ``config/settings.csv``.

One symbol, one source
----------------------
Two vendors' closes can differ after a corporate action, so a symbol's
history is never spliced across providers: when the provider used for a
symbol changes, its whole window is re-pulled and the old rows are
replaced. The ``source`` column in ``prices.parquet`` is what makes that
check possible.
"""

from __future__ import annotations

import datetime as _dt
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROVIDERS = ("auto", "bloomberg", "tiingo")
COLUMNS = ["date", "symbol", "px_last", "source"]


class ProviderUnavailable(RuntimeError):
    """The provider cannot run on this machine (missing package, no
    Terminal, no network)."""


# ---------------------------------------------------------------------------
# symbol mapping
# ---------------------------------------------------------------------------
def _approved_rows() -> Dict[str, dict]:
    from src.themes import APPROVED_INSTRUMENTS
    return APPROVED_INSTRUMENTS


def to_bloomberg(symbol: str) -> str:
    """Bloomberg security string for a plain symbol.

    Uses the ``bloomberg`` column of ``approved_instruments.csv`` when
    present, else ``"<SYMBOL> US Equity"``.
    """
    row = _approved_rows().get(symbol)
    if row and row.get("bloomberg"):
        return row["bloomberg"]
    return f"{symbol} US Equity"


def to_tiingo(symbol: str) -> Optional[str]:
    """Tiingo ticker for a plain symbol, or ``None`` if unmappable.

    Plain US tickers map to themselves with ``.`` share classes written
    as ``-`` (``BRK.B`` -> ``BRK-B``). Instruments listed in
    ``approved_instruments.csv`` use the ``tiingo`` column if filled. A
    symbol whose Bloomberg code is not a US equity and that has no
    ``tiingo`` mapping returns ``None`` rather than guessing.
    """
    row = _approved_rows().get(symbol)
    if row:
        t_sym = (row.get("tiingo") or "").strip()
        if t_sym:
            return t_sym
        bbg = (row.get("bloomberg") or "").strip()
        if bbg and not bbg.endswith("US Equity") and not bbg.endswith(
                "UQ Equity") and not bbg.endswith("UW Equity"):
            return None                       # foreign line, no mapping
    if " " in symbol:
        return None                           # e.g. "1622 JT" with no map
    return symbol.replace(".", "-")


def tiingo_key() -> str:
    """``TIINGO_API_KEY`` from the environment or the project ``.env``."""
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if key:
        return key
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("TIINGO_API_KEY=") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
class BloombergProvider:
    """Daily closes from a running Bloomberg Terminal via ``blpapi``."""

    name = "bloomberg"

    def available(self) -> Tuple[bool, str]:
        """Whether ``blpapi`` imports and a session can start."""
        try:
            import blpapi                                  # noqa: F401
        except ImportError:
            return False, "blpapi is not installed"
        try:
            import blpapi
            session = blpapi.Session()
            if not session.start():
                return False, "could not start a blpapi session (is the Terminal running?)"
            session.stop()
        except Exception as exc:                           # noqa: BLE001
            return False, f"blpapi session failed: {type(exc).__name__}: {exc}"
        return True, "Terminal reachable"

    def fetch(self, symbols: Iterable[str], start: str, end: str,
              log=print) -> pd.DataFrame:
        """Pull ``PX_LAST`` for ``symbols`` between ``start`` and ``end``
        (``YYYYMMDD``, inclusive). Delegates to the request loop in
        ``pull_prices.py``."""
        from ingestion.pull_prices import bloomberg_request
        df = bloomberg_request(list(symbols), start, end, log=log)
        if df is None or df.empty:
            return pd.DataFrame(columns=COLUMNS)
        df = df.copy()
        df["source"] = self.name
        return df[COLUMNS]


class TiingoProvider:
    """Daily closes from Tiingo's price API, one request per symbol."""

    name = "tiingo"
    URL = "https://api.tiingo.com/tiingo/daily/{sym}/prices"
    PAUSE_S = 0.15                 # polite pacing between requests
    TIMEOUT_S = 30

    def available(self) -> Tuple[bool, str]:
        try:
            import requests                                # noqa: F401
        except ImportError:
            return False, "requests is not installed"
        if not tiingo_key():
            return False, ("TIINGO_API_KEY is not set (free key at "
                           "tiingo.com; put it in .env)")
        return True, "tiingo (key present)"

    def _get(self, sym: str, start: str, end: str) -> list:
        """The raw JSON rows for one ticker; ``[]`` when Tiingo has no
        such ticker (HTTP 404). A quota answer (HTTP 429) raises."""
        import requests
        r = requests.get(self.URL.format(sym=sym),
                         params={"startDate": f"{start[:4]}-{start[4:6]}-{start[6:]}",
                                 "endDate": f"{end[:4]}-{end[4:6]}-{end[6:]}",
                                 "format": "json", "token": tiingo_key()},
                         headers={"Content-Type": "application/json",
                                  "User-Agent": "retailAPOLLO/1.0"},
                         timeout=self.TIMEOUT_S)
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            raise RuntimeError("tiingo: request quota reached for this key "
                               f"({r.text[:120]}); wait for the window to "
                               "reset or raise the plan")
        if r.status_code in (401, 403):
            raise RuntimeError(f"tiingo: key rejected (HTTP {r.status_code}): "
                               f"{r.text[:120]}")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    @staticmethod
    def _to_long(rows: list, symbol: str) -> pd.DataFrame:
        """Tiingo rows -> long frame with a SPLIT-adjusted close.

        Tiingo's ``close`` is the raw print and ``adjClose`` is adjusted
        for splits AND dividends. ``PX_LAST`` is adjusted for splits
        only, so the stored value is ``close`` divided by the product of
        every later ``splitFactor`` (a 4-for-1 split carries factor 4 on
        the split day and applies to all prior prints).
        """
        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(rows)
        if "date" not in df.columns or "close" not in df.columns:
            return pd.DataFrame(columns=COLUMNS)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(df["close"], errors="coerce")
        if "splitFactor" in df.columns:
            sf = pd.to_numeric(df["splitFactor"], errors="coerce").fillna(1.0)
            # product of split factors strictly AFTER each row
            later = sf[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
            close = close / later
        out = pd.DataFrame({"date": df["date"], "symbol": symbol,
                            "px_last": close.values, "source": "tiingo"})
        return out.dropna(subset=["px_last"])[COLUMNS]

    def fetch(self, symbols: Iterable[str], start: str, end: str,
              log=print) -> pd.DataFrame:
        """Download closes for ``symbols`` (``YYYYMMDD`` window, inclusive).
        Unmappable symbols are reported and skipped; a quota answer
        raises so the caller's fallback logic sees a failure."""
        symbols = list(symbols)
        back: Dict[str, str] = {}
        skipped = []
        for s in symbols:
            t = to_tiingo(s)
            if t is None:
                skipped.append(s)
            else:
                back[t] = s
        if skipped:
            log(f"  tiingo: no Tiingo symbol for {len(skipped)} "
                f"instrument(s), skipped: {', '.join(skipped[:12])}"
                f"{' ...' if len(skipped) > 12 else ''}  "
                "(add a 'tiingo' column value in "
                "config/approved_instruments.csv)")
        parts = []
        n = len(back)
        for i, (t_sym, plain) in enumerate(back.items(), start=1):
            got = self._to_long(self._get(t_sym, start, end), plain)
            if len(got):
                parts.append(got)
            if n >= 20 and i % 50 == 0:
                log(f"  tiingo: {i}/{n} symbols")
            if i < n:
                time.sleep(self.PAUSE_S)
        if not parts:
            return pd.DataFrame(columns=COLUMNS)
        return pd.concat(parts, ignore_index=True)


_REGISTRY = {"bloomberg": BloombergProvider, "tiingo": TiingoProvider}


def provider_chain(preference: str, fallback: bool = True,
                   log=print) -> List[object]:
    """Ordered providers to try for ``preference``.

    Args:
        preference: ``auto``, ``bloomberg`` or ``tiingo``.
        fallback: For ``bloomberg``, whether Tiingo is appended as the
            fallback when the Bloomberg pull fails.
        log: Line sink for the availability report.

    Returns:
        Provider instances in the order to attempt them.

    Raises:
        ValueError: On an unknown preference.
        ProviderUnavailable: When no provider in the chain is usable.
    """
    pref = (preference or "auto").strip().lower()
    if pref not in PROVIDERS:
        raise ValueError(f"price provider must be one of {PROVIDERS}, "
                         f"not {preference!r}")
    bbg, tii = BloombergProvider(), TiingoProvider()
    if pref == "tiingo":
        chain = [tii]
    elif pref == "bloomberg":
        chain = [bbg] + ([tii] if fallback else [])
    else:                                               # auto
        ok, why = bbg.available()
        log(f"  price provider auto: bloomberg {'available' if ok else 'unavailable'} ({why})")
        chain = [bbg, tii] if ok else [tii]
    usable = []
    for prov in chain:
        ok, why = prov.available()
        if ok:
            usable.append(prov)
        else:
            log(f"  price provider {prov.name}: unavailable - {why}")
    if not usable:
        raise ProviderUnavailable(
            "no price provider is usable on this machine: install "
            "set TIINGO_API_KEY in .env (free key at tiingo.com) or run "
            "with a Bloomberg Terminal open")
    return usable


def fetch_with_fallback(chain: List[object], symbols: List[str],
                        start: str, end: str, log=print
                        ) -> Tuple[pd.DataFrame, str]:
    """Try each provider in ``chain`` until one answers.

    An empty answer is an answer: a provider that connected and returned
    no rows (a symbol that did not trade yet, a span of holidays) is not
    a failure, and falling through to the next provider on it would
    splice sources. Only an exception - no session, no network, a
    request error - moves on to the next provider.

    Returns:
        ``(frame, provider_name)``; the frame may be empty.

    Raises:
        ProviderUnavailable: When every provider raised.
    """
    last_err = None
    for prov in chain:
        try:
            log(f"  pulling {len(symbols)} symbol(s) {start} -> {end} "
                f"via {prov.name}")
            df = prov.fetch(symbols, start, end, log=log)
            if df is None:
                df = pd.DataFrame(columns=COLUMNS)
            if not len(df):
                log(f"  {prov.name}: no rows for this span "
                    "(not listed yet, or no trading days)")
            return df, prov.name
        except Exception as exc:                           # noqa: BLE001
            last_err = f"{prov.name}: {type(exc).__name__}: {exc}"
            log(f"  {prov.name} FAILED - {last_err}")
        if prov is not chain[-1]:
            log(f"  falling back to {chain[chain.index(prov) + 1].name}")
    raise ProviderUnavailable(f"every price provider failed ({last_err})")


# ---------------------------------------------------------------------------
# self-test: exercise one provider end to end, write nothing
# ---------------------------------------------------------------------------
def selftest(provider: str = "tiingo", symbols: Iterable[str] = None,
             days: int = 14, log=print) -> int:
    """Pull a few symbols over the last ``days`` days from one provider
    and print what came back. Touches no store. Returns 0 when every
    requested symbol produced rows, 1 otherwise.

    Usage::

        python -m src.prices --provider tiingo
        python -m src.prices --provider bloomberg --symbols SPY,GLD
    """
    symbols = list(symbols or ["SPY", "GLD", "NVDA", "BRK.B", "1622 JT"])
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)
    a, b = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    log(f"price provider self-test: {provider} | {', '.join(symbols)} | "
        f"{a} -> {b}")
    prov = {"tiingo": TiingoProvider, "bloomberg": BloombergProvider}.get(
        provider)
    if prov is None:
        log(f"unknown provider {provider!r}; use tiingo or bloomberg")
        return 1
    prov = prov()
    ok, why = prov.available()
    log(f"  available: {ok} ({why})")
    if not ok:
        return 1
    try:
        df = prov.fetch(symbols, a, b, log=log)
    except Exception as exc:                               # noqa: BLE001
        log(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1
    if df is None or not len(df):
        log("  no rows returned")
        # show what the service actually answered, so an HTML page, a
        # quota notice or a proxy interstitial is visible instead of
        # being read as "no data"
        if isinstance(prov, TiingoProvider) and symbols:
            try:
                raw = prov._get(to_tiingo(symbols[0]) or symbols[0], a, b)
                log(f"  raw answer for {symbols[0]}: {str(raw)[:300]}")
            except Exception as exc:                       # noqa: BLE001
                log(f"  raw request failed: {type(exc).__name__}: {exc}")
        return 1
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    got = df.groupby("symbol").agg(rows=("px_last", "size"),
                                   first=("date", "min"), last=("date", "max"),
                                   last_close=("px_last", "last"))
    log(got.to_string())
    missing = [s for s in symbols if s not in got.index]
    if missing:
        log(f"  no rows for: {', '.join(missing)}")
    log(f"  {len(df)} rows via {prov.name}; nothing written")
    return 0 if not missing else 1


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(description="Price provider self-test "
                                             "(writes nothing).")
    _p.add_argument("--provider", default="tiingo",
                    choices=("tiingo", "bloomberg"))
    _p.add_argument("--symbols", nargs="*", default=[],
                    help="comma-separated pipeline symbols; a symbol with "
                         "a space (1622 JT) may be typed unquoted (default: "
                         "a small mixed set incl. a dotted class and a "
                         "foreign line)")
    _p.add_argument("--days", type=int, default=14)
    _a = _p.parse_args()
    # the shell splits on spaces, so rejoin the tokens and split on commas:
    # "SPY,GLD,1622 JT" arrives as ["SPY,GLD,1622", "JT"]
    _syms = [s.strip() for s in " ".join(_a.symbols).split(",") if s.strip()] or None
    raise SystemExit(selftest(_a.provider, _syms, _a.days))
