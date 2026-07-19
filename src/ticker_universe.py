"""
Load a US-listed equity ticker universe (letters only, length 1–5).

Uses Nasdaq Trader symbol directory files (cached on disk). See:
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
# SURVIVORSHIP FIX: the Nasdaq Trader files list TODAY'S symbols, so tickers
# that were delisted (bankruptcy, buyout, deregistration) vanish from the
# universe - and their historical mentions silently stop counting. That
# flatters any backtest: the casualties are exactly the names retail piled
# into before they died. This hand-curated supplement re-adds known dead
# retail favourites so their history counts again. Extend it whenever a
# name you KNOW was loud on Reddit fails to appear in notebook 02.
# Proper long-term fix: a point-in-time universe (archived nasdaqlisted.txt
# snapshots or a Bloomberg/CRSP export) - see README live-data checklist.
# ---------------------------------------------------------------------------
DELISTED_TICKERS: frozenset[str] = frozenset({
    "BBBY",  # Bed Bath & Beyond - bankrupt 2023, THE meme casualty
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
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _parse_nasdaq_listed(text: str) -> set[str]:
    """Symbol|...|Test Issue|... — skip Test Issue == Y."""
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
    """ACT Symbol|...|Test Issue|... — skip Test Issue == Y."""
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
    """
    Return uppercase tickers from nasdaqlisted + otherlisted, excluding test issues.

    Caches raw .txt files under cache_dir. Re-downloads if missing or older than
    max_cache_age_days (unless force_refresh is False and you want always refresh—
    use force_refresh=True to ignore age).
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
    merged = nasdaq | other | DELISTED_TICKERS   # survivorship supplement
    logger.info(
        "Ticker universe: %s unique symbols (nasdaqlisted %s, otherlisted %s, "
        "delisted supplement %s)",
        len(merged),
        len(nasdaq),
        len(other),
        len(DELISTED_TICKERS),
    )
    return merged
