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

Why an API and not a free site: free price sites rate-limit by address
or front their endpoints with a bot challenge, and refuse scripted
access from shared networks. A key is the permission.

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
Two vendors' closes can differ after a corporate action, so a full pull
keeps one provider per symbol inside the window it pulls: when the
provider used for a symbol changes, that window is re-pulled and the
stored rows sitting in it are replaced. The replacement is what earns
the deletion, so the old rows go only for a symbol the new provider
actually answered for - a provider with nothing for a symbol leaves that
symbol's series exactly as it stands. Rows dated before the window start
are outside what was re-pulled and are kept, which puts a seam at the
window start rather than throwing away the years behind it. The
``source`` column in ``prices.parquet`` records the vendor of every row,
so ``tools/data_health.py`` names each series that carries a seam.

``pull_prices.py --daily`` suspends that rule to keep a daily run down
to a handful of requests: it extends a stored series with whichever
provider answers and tags the rows it adds, so a series can carry one
vendor's past and another's present. The seam sits in the ``source``
column, and ``tools/data_health.py`` names every series that has one.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROVIDERS = ("auto", "bloomberg", "tiingo")
COLUMNS = ["date", "symbol", "px_last", "source"]


class ProviderUnavailable(RuntimeError):
    """The provider cannot run on this machine (missing package, no
    Terminal, no network)."""


# ---------------------------------------------------------------------------
# store schema
# ---------------------------------------------------------------------------
def with_source(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` carrying every column in :data:`COLUMNS`, in that order.

    Rows whose columns predate provider tracking hold no ``source``.
    They are labelled ``bloomberg`` - the same convention
    ``pull_prices._current_sources`` reads such a store under - so the
    label is written down once rather than inferred again by every
    reader that opens the file. A blank label counts as absent: the
    readers group by this column - ``_current_sources`` to decide what
    to re-pull, the health report to count series carrying two vendors -
    and an empty string groups as a vendor of its own, so one unlabelled
    row would read as a second source for its symbol.
    """
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = "bloomberg"
    out["source"] = (out["source"].fillna("bloomberg")
                     .astype(str).str.strip()
                     .replace("", "bloomberg"))
    return out[COLUMNS]


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


def is_us_listing(bloomberg_code: str) -> bool:
    """Whether a Bloomberg code names a US-listed equity.

    A code reads ``"<TICKER> <EXCH> Equity"``, and every US exchange
    token starts with ``U``: ``US`` composite, ``UN`` NYSE, ``UQ`` and
    ``UW`` Nasdaq, ``UP`` NYSE Arca, ``UA`` NYSE American, ``UR``,
    ``UF``, ``UV``, ``UD``, ``UC``, ``UB``, ``UT``, ``UX``, ``UU``. A
    foreign line carries its own market's token instead (``JT``, ``CS``,
    ``CH``, ``LN``, ``GR``, ``HK``, ``T``, ``AU``), none of which begins
    with ``U``. The test reads the exchange token rather than matching
    whole suffixes, because a whitelist of suffixes silently calls every
    venue it has not been told about foreign - and an ETF declared
    foreign is an ETF never requested.
    """
    parts = bloomberg_code.split()
    return (len(parts) == 3 and parts[2].lower() == "equity"
            and parts[1].upper().startswith("U"))


def to_tiingo(symbol: str) -> Optional[str]:
    """Tiingo ticker for a plain symbol, or ``None`` if unmappable.

    Plain US tickers map to themselves with ``.`` share classes written
    as ``-`` (``BRK.B`` -> ``BRK-B``). Instruments listed in
    ``approved_instruments.csv`` use the ``tiingo`` column if filled. A
    symbol whose Bloomberg code is not a US equity (see
    :func:`is_us_listing`) and that has no ``tiingo`` mapping returns
    ``None`` rather than guessing.
    """
    row = _approved_rows().get(symbol)
    if row:
        t_sym = (row.get("tiingo") or "").strip()
        if t_sym:
            return t_sym
        bbg = (row.get("bloomberg") or "").strip()
        if bbg and not is_us_listing(bbg):
            return None                       # foreign line, no mapping
    if " " in symbol:
        return None                           # e.g. "1622 JT" with no map
    return symbol.replace(".", "-")


def _env_value(name: str) -> str:
    """``name`` from the environment, else from the project ``.env``.

    A ``.env`` line is ``NAME=value`` or ``NAME = value``; a comment line
    is skipped and surrounding quotes are dropped. The fetchers read
    their keys this way, so a key that lives only in the file is as
    visible here as one exported into the environment.
    """
    val = os.environ.get(name, "").strip()
    if val:
        return val
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
    return ""


def tiingo_key() -> str:
    """``TIINGO_API_KEY`` from the environment or the project ``.env``."""
    return _env_value("TIINGO_API_KEY")


def redact(text: str) -> str:
    """``text`` with any API key in it replaced by ``***``.

    Every message that reports a failed request is written to the
    console and to ``Reports/logs/``, and a request library puts the
    URL it tried into the exception it raises. Keys are kept out of the
    URL at the call site; this is the second line, for a message that
    reaches a log by some other route.

    Every credential the shipped code carries is masked, whichever of
    the two names FetchLayer's is stored under. A value issued in spaced
    blocks is masked in both spellings, because such a value is as often
    stored with the spaces taken out as with them left in.
    """
    out = str(text)
    keys = [tiingo_key()] + [_env_value(n) for n in
                             ("FETCHLAYER_KEY", "FETCHLAYER_API_KEY",
                              "X_BEARER_TOKEN",
                              "ANTHROPIC_API_KEY", "REDDIT_SECRET",
                              "REDDIT_PASSWORD")]
    for key in keys:
        if key and len(key) >= 8:
            out = out.replace(key, "***")
            if " " in key:
                out = out.replace(key.replace(" ", ""), "***")
    return re.sub(r"((?:token|apikey|api_key|key)=)[^&\s\"']+", r"\1***",
                  out, flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
# Repeated Terminal failures take the Terminal out of the chain for the
# rest of the process. A session can START while //blp/refdata cannot be
# OPENED - the Terminal is running but not entitled, not logged in, or
# unreachable through the network - and openService only discovers that by
# blocking until it times out, which is about two minutes. A price pull
# asks for one span after another, so without this tally every span pays
# that wait again before falling back to the same place it fell back to
# last time.
#
# The allowance is a setting rather than 1 because a single failure can be
# a blip - a request that raced a Terminal restart - and giving up on the
# first one would send a healthy Terminal to the fallback for the whole
# run. The tally counts CONSECUTIVE failures: a span the Terminal answers
# clears it, so blips scattered over a long multi-span pull never add up
# to a verdict against a Terminal that is working. It is counted per
# PROCESS: the next run tries again, which is the right granularity,
# since a Terminal that comes back deserves a fresh chance at the next
# run rather than mid-run.
_BLOOMBERG_FAILURES: List[str] = []


def bloomberg_max_failures() -> int:
    """How many consecutive Terminal failures this process tolerates
    before it stops trying (``bloomberg_max_failures`` in
    ``config/settings.csv``)."""
    try:
        from src import settings
        return max(1, settings.get_int("bloomberg_max_failures"))
    except Exception:                                      # noqa: BLE001
        return 2


def bloomberg_failed_this_process() -> str:
    """Why the Terminal is being skipped, or ``""`` while it is still
    inside its allowance of consecutive failures."""
    n, allowed = len(_BLOOMBERG_FAILURES), bloomberg_max_failures()
    if n < allowed:
        return ""
    return (f"{n} consecutive failed attempt(s) this run, last: "
            f"{_BLOOMBERG_FAILURES[-1]}")


def reset_bloomberg_failure() -> None:
    """Forget the failures (tests, and a caller that has fixed the Terminal)."""
    _BLOOMBERG_FAILURES.clear()


class BloombergProvider:
    """Daily closes from a running Bloomberg Terminal via ``blpapi``."""

    name = "bloomberg"

    def down_reason(self) -> str:
        """Why this provider is being skipped, or ``""``."""
        return bloomberg_failed_this_process()

    def available(self) -> Tuple[bool, str]:
        """Whether ``blpapi`` imports and a session can start."""
        down = self.down_reason()
        if down:
            return False, f"{down} - not retried this run"
        try:
            import blpapi
        except ImportError:
            return False, "blpapi is not installed"
        try:
            session = blpapi.Session()
            if not session.start():
                return False, "could not start a blpapi session (is the Terminal running?)"
            session.stop()
        except Exception as exc:                           # noqa: BLE001
            return False, (f"blpapi session failed: {type(exc).__name__}: "
                           f"{redact(exc)}")
        # A session that starts is NOT a Terminal that answers: opening
        # //blp/refdata is the real test, and it is left to the first
        # fetch rather than paid here, because it is the slow half.
        return True, "Terminal reachable"

    def fetch(self, symbols: Iterable[str], start: str, end: str,
              log=print) -> pd.DataFrame:
        """Pull ``PX_LAST`` for ``symbols`` between ``start`` and ``end``
        (``YYYYMMDD``, inclusive). Delegates to the request loop in
        ``pull_prices.py``."""
        from ingestion.pull_prices import bloomberg_request
        try:
            df = bloomberg_request(list(symbols), start, end, log=log)
        except Exception as exc:                           # noqa: BLE001
            _BLOOMBERG_FAILURES.append(f"{type(exc).__name__}: {redact(exc)}")
            n, allowed = len(_BLOOMBERG_FAILURES), bloomberg_max_failures()
            if n >= allowed:
                log(f"  bloomberg: {n} consecutive failure(s) this run - the "
                    f"Terminal is skipped for the rest of it")
            else:
                log(f"  bloomberg: failure {n} of {allowed} before the "
                    f"Terminal is skipped for this run")
            raise
        # An answered span is the evidence that the Terminal is alive, so
        # it retires the failures ahead of it; the allowance covers a run
        # of failures, not a total spread over an entire pull.
        _BLOOMBERG_FAILURES.clear()
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

    def down_reason(self) -> str:
        """Tiingo is stateless across spans: one failed request says
        nothing about the next."""
        return ""

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
        such ticker (HTTP 404). A quota answer (HTTP 429) raises.

        The key travels in the ``Authorization`` header, never as a
        query parameter: a failed request carries its URL into the
        exception text, the console and the run log, and a key in that
        URL is a key written to disk in clear.
        """
        import requests
        r = requests.get(self.URL.format(sym=sym),
                         params={"startDate": f"{start[:4]}-{start[4:6]}-{start[6:]}",
                                 "endDate": f"{end[:4]}-{end[4:6]}-{end[6:]}",
                                 "format": "json"},
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Token {tiingo_key()}",
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
            "no price provider is usable on this copy: set TIINGO_API_KEY "
            "in .env (a free key from tiingo.com), or run with a Bloomberg "
            "Terminal open and blpapi installed")
    return usable


def provider_status(preference: str = "auto", fallback: bool = True):
    """``(usable, one-line reason)`` for the configured chain.

    The question :func:`provider_chain` answers by raising, for callers
    that need to explain the situation rather than stop: a display panel
    on a copy with no Terminal and no key wants the sentence, not the
    traceback.

    Returns:
        ``(True, "bloomberg, tiingo")`` naming the providers that would be
        tried, or ``(False, why)``.
    """
    try:
        chain = provider_chain(preference, fallback=fallback,
                               log=lambda *_: None)
    except (ProviderUnavailable, ValueError) as e:
        return False, str(e)
    return True, ", ".join(p.name for p in chain)


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
        # The chain is built once per run and reused for every span, so a
        # provider that has since been taken out of service is skipped
        # here rather than tried again. Without this the tally in
        # BloombergProvider would be written and never read.
        down = getattr(prov, "down_reason", lambda: "")()
        if down:
            log(f"  skipping {prov.name} - {down}")
            last_err = f"{prov.name}: {down}"
            if prov is not chain[-1]:
                log(f"  falling back to {chain[chain.index(prov) + 1].name}")
            continue
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
            last_err = f"{prov.name}: {type(exc).__name__}: {redact(exc)}"
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

        cd Code
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
        log(f"  FAILED: {type(exc).__name__}: {redact(exc)}")
        return 1
    if df is None or not len(df):
        log("  no rows returned")
        # show what the service actually answered, so an HTML page, a
        # quota notice or a proxy interstitial is visible instead of
        # being read as "no data"
        if isinstance(prov, TiingoProvider) and symbols:
            try:
                raw = prov._get(to_tiingo(symbols[0]) or symbols[0], a, b)
                log(f"  raw answer for {symbols[0]}: {redact(raw)[:300]}")
            except Exception as exc:                       # noqa: BLE001
                log(f"  raw request failed: {type(exc).__name__}: "
                    f"{redact(exc)}")
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
