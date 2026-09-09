"""US-listed ticker universe from the Nasdaq Trader symbol directories.

``load_us_ticker_universe()`` returns the set of valid symbols (letters
only, length 1-5) from nasdaqlisted.txt and otherlisted.txt, cached on
disk under a caller-supplied directory and re-downloaded when stale, plus
a hand-curated supplement of delisted retail favourites so their history
keeps counting. ``load_etf_symbols()`` reads the ETF flag from the same
cached files. File format reference:
https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefinitions
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_SYMBOL_OK = re.compile(r"^[A-Z]{1,5}$")

# ---------------------------------------------------------------------------
# Survivorship supplement. The Nasdaq Trader files list today's symbols,
# so tickers that were delisted (bankruptcy, buyout, deregistration)
# vanish from the universe and their historical mentions silently stop
# counting. That flatters any backtest: the casualties are exactly the
# names retail piled into before they died. This hand-curated supplement
# re-adds known dead retail favourites so their history counts again.
# Extend it whenever a name known to have been loud on Reddit is missing
# from the mention counts. The complete fix is a point-in-time universe
# (archived nasdaqlisted.txt snapshots or a Bloomberg/CRSP export).
# ---------------------------------------------------------------------------
DELISTED_TICKERS: frozenset[str] = frozenset({
    "BBBY",  # Bed Bath & Beyond - bankrupt 2023
    "WISH",  # ContextLogic - delisted 2024
    "EXPR",  # Express - bankrupt 2024
    "NAKD",  # Naked Brand - merged into CENN 2021
    "SPRT",  # Support.com - merged into GREE 2021 (huge squeeze)
    "ATER",  # Aterian - squeeze-era favourite, reverse-split casualty
    "MULN",  # Mullen Automotive - reverse splits into oblivion
    "RDBX",  # Redbox - 2022 squeeze, acquired
    "CTRM",  # Castor Maritime - 2021 penny favourite
    "GNUS",  # Genius Brands - renamed 2023
    "CLVS",  # Clovis Oncology - bankrupt 2022
    "SDC",   # SmileDirectClub - bankrupt 2023
    "APRN",  # Blue Apron - acquired 2023
    "FSR",   # Fisker - bankrupt 2024
})


def _fetch_text(url: str, timeout: int = 120) -> str:
    """Downloads one symbol directory file as text."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _parse_nasdaq_listed(text: str) -> set[str]:
    """Parses nasdaqlisted.txt (Symbol|...|Test Issue|...), skipping test
    issues and symbols that are not 1-5 letters."""
    out: set[str] = set()
    for line in text.splitlines():
        if not line or line.startswith("Symbol|"):
            continue
        if "File Creation Time" in line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        sym, test = parts[0].strip(), parts[3].strip()
        if test == "Y":
            continue
        sym = sym.upper()
        if _SYMBOL_OK.fullmatch(sym):
            out.add(sym)
    return out


def _parse_other_listed(text: str) -> set[str]:
    """Parses otherlisted.txt (ACT Symbol|...|Test Issue|...), skipping
    test issues and symbols that are not 1-5 letters."""
    out: set[str] = set()
    for line in text.splitlines():
        if not line or line.startswith("ACT Symbol|"):
            continue
        if "File Creation Time" in line:
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        sym, test = parts[0].strip(), parts[6].strip()
        if test == "Y":
            continue
        sym = sym.upper()
        if _SYMBOL_OK.fullmatch(sym):
            out.add(sym)
    return out


def load_us_ticker_universe(
    cache_dir: Path,
    *,
    max_cache_age_days: float = 7.0,
    force_refresh: bool = False,
) -> set[str]:
    """Returns the uppercase ticker universe.

    Combines nasdaqlisted + otherlisted (excluding test issues) with the
    DELISTED_TICKERS supplement. The raw .txt files are cached under
    cache_dir and re-downloaded when missing or older than
    max_cache_age_days.

    Args:
        cache_dir: Directory for the cached symbol files; created if
            needed.
        max_cache_age_days: Cache age beyond which the files are
            re-downloaded.
        force_refresh: Re-download regardless of age.

    Returns:
        Set of uppercase symbols.

    Raises:
        requests.HTTPError: If a download is needed and fails.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "nasdaqlisted.txt": NASDAQ_LISTED_URL,
        "otherlisted.txt": OTHER_LISTED_URL,
    }

    max_age_s = max_cache_age_days * 86400.0
    now = time.time()

    for fname, url in paths.items():
        dest = cache_dir / fname
        need = force_refresh or not dest.exists()
        if not need:
            age = now - dest.stat().st_mtime
            if age > max_age_s:
                need = True
        if need:
            logger.info("Downloading %s", url)
            dest.write_text(_fetch_text(url), encoding="utf-8")

    nasdaq = _parse_nasdaq_listed((cache_dir / "nasdaqlisted.txt").read_text(encoding="utf-8"))
    other = _parse_other_listed((cache_dir / "otherlisted.txt").read_text(encoding="utf-8"))
    merged = nasdaq | other | DELISTED_TICKERS   # Survivorship supplement.
    logger.info(
        "Ticker universe: %s unique symbols (nasdaqlisted %s, otherlisted %s, "
        "delisted supplement %s)",
        len(merged),
        len(nasdaq),
        len(other),
        len(DELISTED_TICKERS),
    )
    return merged


# ---------------------------------------------------------------------------
# ETF vs single name. Both Nasdaq files carry an ETF column ('Y'/'N'), so
# the distinction is available from files this project already caches:
# no new dependency, no hand list to maintain. The euphoria single-name
# universe uses it to keep SPY, QQQ, VXUS and SCHD out of a ranking whose
# premise is single names; their mentions stay in the counts, they simply
# are not single names.
# ---------------------------------------------------------------------------
def load_etf_symbols(cache_dir: Path) -> set[str]:
    """Returns the symbols flagged as ETFs by the Nasdaq symbol directories.

    Returns an empty set if the cached files are missing or have an
    unexpected layout, so every caller degrades to "cannot tell" rather
    than to a wrong answer.

    Args:
        cache_dir: Directory holding the cached symbol files.
    """
    out: set[str] = set()
    for fname, sym_col in (("nasdaqlisted.txt", "Symbol"),
                           ("otherlisted.txt", "ACT Symbol")):
        path = Path(cache_dir) / fname
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        head = lines[0].split("|")
        try:
            sym_i, etf_i = head.index(sym_col), head.index("ETF")
        except ValueError:
            continue                     # Unexpected layout: skip, never guess.
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) <= max(sym_i, etf_i):
                continue                 # The trailing "File Creation Time" row.
            if parts[etf_i].strip().upper() == "Y":
                sym = parts[sym_i].strip().upper()
                if sym:
                    out.add(sym)
    return out
