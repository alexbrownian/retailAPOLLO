"""Price providers: Bloomberg and yfinance behind one interface.

The pipeline needs daily closes for every priced instrument in
``data/prices/prices.parquet`` (long form: ``date``, ``symbol``,
``px_last``, ``source``). Two providers can fill it:

``bloomberg``
    ``blpapi`` ``HistoricalDataRequest`` for ``PX_LAST`` against a
    running, logged-in Terminal. Securities are addressed by the code in
    ``config/approved_instruments.csv`` (``bloomberg`` column) or
    ``"<SYMBOL> US Equity"`` by default.

``yfinance``
    Yahoo Finance via the ``yfinance`` package; no credentials. Closes
    are split- and dividend-adjusted (``auto_adjust=True``). Non-US
    instruments need a Yahoo symbol in the ``yfinance`` column of
    ``config/approved_instruments.csv`` (for example ``1622.T``); a
    symbol with no mapping is skipped and reported.

Selection and fallback
----------------------
:func:`provider_chain` turns a preference into an ordered list:

* ``auto``       -> Bloomberg if it can connect, otherwise yfinance;
* ``bloomberg``  -> Bloomberg, falling back to yfinance if the pull
  fails (``fallback=False`` disables the fallback);
* ``yfinance``   -> yfinance only.

The preference comes from ``--provider`` on ``pull_prices.py`` /
``update_data.py`` or ``price_provider`` in ``config/settings.csv``.

One symbol, one source
----------------------
Bloomberg ``PX_LAST`` and Yahoo's adjusted close differ after a split
or a large dividend, so a symbol's history is never spliced across
providers: when the provider used for a symbol changes, its whole window
is re-pulled and the old rows are replaced. The ``source`` column in
``prices.parquet`` is what makes that check possible.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROVIDERS = ("auto", "bloomberg", "yfinance")
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


def to_yfinance(symbol: str) -> Optional[str]:
    """Yahoo Finance symbol for a plain symbol, or ``None`` if unmappable.

    Plain US tickers map to themselves. Instruments listed in
    ``approved_instruments.csv`` use the ``yfinance`` column if filled.
    A symbol whose Bloomberg code is not a US equity and that has no
    ``yfinance`` mapping returns ``None`` rather than guessing.
    """
    row = _approved_rows().get(symbol)
    if row:
        yf_sym = (row.get("yfinance") or "").strip()
        if yf_sym:
            return yf_sym
        bbg = (row.get("bloomberg") or "").strip()
        if bbg and not bbg.endswith("US Equity") and not bbg.endswith(
                "UQ Equity") and not bbg.endswith("UW Equity"):
            return None                       # foreign line, no mapping
    if " " in symbol:
        return None                           # e.g. "1622 JT" with no map
    return symbol.replace(".", "-")           # BRK.B -> BRK-B


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
        from pull_prices import bloomberg_request
        df = bloomberg_request(list(symbols), start, end, log=log)
        if df is None or df.empty:
            return pd.DataFrame(columns=COLUMNS)
        df = df.copy()
        df["source"] = self.name
        return df[COLUMNS]


class YFinanceProvider:
    """Daily adjusted closes from Yahoo Finance via ``yfinance``."""

    name = "yfinance"
    CHUNK = 100

    def available(self) -> Tuple[bool, str]:
        try:
            import yfinance                                # noqa: F401
        except ImportError:
            return False, "yfinance is not installed (pip install yfinance)"
        return True, "yfinance installed"

    @staticmethod
    def _to_long(raw: pd.DataFrame, requested: List[str],
                 back: Dict[str, str]) -> pd.DataFrame:
        """Reshape a ``yfinance.download`` result to the long schema.

        Handles both the multi-ticker layout (columns ``(field,
        ticker)``) and the single-ticker layout (columns ``field``).
        ``back`` maps a Yahoo symbol to the plain pipeline symbol.
        """
        if raw is None or raw.empty:
            return pd.DataFrame(columns=COLUMNS)
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                return pd.DataFrame(columns=COLUMNS)
            close = raw["Close"]
        else:
            if "Close" not in raw.columns or len(requested) != 1:
                return pd.DataFrame(columns=COLUMNS)
            close = raw[["Close"]].rename(columns={"Close": requested[0]})
        long = (close.stack(future_stack=True).rename("px_last")
                .reset_index())
        long.columns = ["date", "yf_symbol", "px_last"]
        long = long.dropna(subset=["px_last"])
        long["symbol"] = long["yf_symbol"].map(back)
        long = long.dropna(subset=["symbol"])
        long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
        long["source"] = "yfinance"
        return long[COLUMNS].reset_index(drop=True)

    def fetch(self, symbols: Iterable[str], start: str, end: str,
              log=print) -> pd.DataFrame:
        """Download adjusted closes for ``symbols`` (``YYYYMMDD`` window).

        Yahoo's ``end`` is exclusive, so one day is added to match the
        inclusive window the rest of the pipeline uses. Unmappable
        symbols are reported and skipped.
        """
        import yfinance as yf
        symbols = list(symbols)
        back: Dict[str, str] = {}
        skipped = []
        for s in symbols:
            y = to_yfinance(s)
            if y is None:
                skipped.append(s)
            else:
                back[y] = s
        if skipped:
            log(f"  yfinance: no Yahoo symbol for {len(skipped)} "
                f"instrument(s), skipped: {', '.join(skipped[:12])}"
                f"{' ...' if len(skipped) > 12 else ''}  "
                "(add a 'yfinance' column value in "
                "config/approved_instruments.csv)")
        if not back:
            return pd.DataFrame(columns=COLUMNS)
        start_d = _dt.datetime.strptime(start, "%Y%m%d").date()
        end_d = _dt.datetime.strptime(end, "%Y%m%d").date() + _dt.timedelta(days=1)
        parts = []
        ys = list(back)
        for i in range(0, len(ys), self.CHUNK):
            chunk = ys[i:i + self.CHUNK]
            raw = yf.download(chunk, start=start_d.isoformat(),
                              end=end_d.isoformat(), auto_adjust=True,
                              progress=False, threads=True,
                              group_by="column")
            parts.append(self._to_long(raw, chunk, back))
        out = pd.concat(parts, ignore_index=True) if parts else \
            pd.DataFrame(columns=COLUMNS)
        return out


_REGISTRY = {"bloomberg": BloombergProvider, "yfinance": YFinanceProvider}


def provider_chain(preference: str, fallback: bool = True,
                   log=print) -> List[object]:
    """Ordered providers to try for ``preference``.

    Args:
        preference: ``auto``, ``bloomberg`` or ``yfinance``.
        fallback: For ``bloomberg``, whether yfinance is appended as the
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
    bbg, yfp = BloombergProvider(), YFinanceProvider()
    if pref == "yfinance":
        chain = [yfp]
    elif pref == "bloomberg":
        chain = [bbg] + ([yfp] if fallback else [])
    else:                                               # auto
        ok, why = bbg.available()
        log(f"  price provider auto: bloomberg {'available' if ok else 'unavailable'} ({why})")
        chain = [bbg, yfp] if ok else [yfp]
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
            "yfinance (pip install yfinance) or run with a Bloomberg "
            "Terminal open")
    return usable


def fetch_with_fallback(chain: List[object], symbols: List[str],
                        start: str, end: str, log=print
                        ) -> Tuple[pd.DataFrame, str]:
    """Try each provider in ``chain`` until one returns rows.

    Returns:
        ``(frame, provider_name)``. A provider that raises or returns no
        rows is logged and the next one is tried.

    Raises:
        ProviderUnavailable: When every provider failed.
    """
    last_err = None
    for prov in chain:
        try:
            log(f"  pulling {len(symbols)} symbol(s) {start} -> {end} "
                f"via {prov.name}")
            df = prov.fetch(symbols, start, end, log=log)
            if df is not None and len(df):
                return df, prov.name
            log(f"  {prov.name} returned no rows")
            last_err = f"{prov.name} returned no rows"
        except Exception as exc:                           # noqa: BLE001
            last_err = f"{prov.name}: {type(exc).__name__}: {exc}"
            log(f"  {prov.name} FAILED - {last_err}")
        if prov is not chain[-1]:
            log(f"  falling back to {chain[chain.index(prov) + 1].name}")
    raise ProviderUnavailable(f"every price provider failed ({last_err})")
