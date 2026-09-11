"""Ticker extraction from post text (title + selftext).

Every ticker mention counted anywhere in the project comes through
``extract_tickers_from_text()``. A mention is validated against a
US-listed symbol universe (Nasdaq Trader files, see ticker_universe.py)
and three stop lists, in two passes:

1. Cashtags (``$GME``), which are high precision.
2. Bare all-caps words of 4-5 letters, scanned on the original text so
   only words the poster typed in capitals match. Bare words collide with
   English even when they are valid tickers (YOU, FOR), so a prose stoplist
   and a data-driven word screen apply; a configurable allowlist restores
   real tickers the screens would otherwise drop and extends the bare pass
   to 1-3 letter symbols.

Pass ``cashtags_only=True`` (or ``--mode cashtag-only`` on the command
line) for maximum precision at lower recall.

The command-line entry point reads a posts parquet and emits long-format
rows (post_id, date, ticker, source)::

    python -m src.extract_tickers \\
      --in Data/raw/wsb_posts_2021-01.parquet \\
      --out Data/processed/wsb_ticker_mentions_2021-01.parquet

Optional daily counts (for date x ticker matrices)::

    python -m src.extract_tickers --in ... --out ... --daily-out Data/processed/wsb_daily_counts.parquet
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from .ticker_universe import load_us_ticker_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jargon, not tickers: symbols the crowd uses as words.
#
# Maintained in config/ticker_stoplist.csv so the list can change without
# touching code. The frozenset below is the fallback seed that keeps a
# fresh clone working if the CSV is missing.
#
# How a symbol earns a place: additions use a falsifiable test rather
# than taste. Many finance abbreviations have been issued to a real ETF
# (HYSA, DRAM, BTC, REIT, NASA, DJIA, HVAC, EBIT, ROPE), so "is it a
# listed symbol?" cannot separate them from real instruments. What can: a
# fund cannot be discussed before it is listed. HYSA carries hundreds of
# mentions per year before its fund launched; DRAM likewise. Those
# mentions are the words. Symbols that pass the test (IBIT, with no
# mentions before its launch year; QQQI, SNDK, SPCX) are left alone, and
# so are old, genuinely traded ETFs like SPY, ARKK and GLD. The reason
# column in the CSV records the evidence per row.
# ---------------------------------------------------------------------------
_STOP_TICKERS_SEED: frozenset[str] = frozenset(
    {
        "DD", "YOLO", "USA", "CEO", "CFO", "IPO", "ATH", "ITM", "OTM",
        "FOMO", "FUD", "ER", "EOD", "IMO", "IRL", "AI", "IT", "OK", "ALL",
        "NEW", "NOW", "BIG", "LOL", "WSB", "IV", "EOY", "ETF", "ROI",
        "FBI", "NYSE", "NASDAQ", "SEC", "EV", "USD", "GDP", "CPI",
    }
)

STOPLIST_CSV = (
    Path(__file__).resolve().parent.parent / "config" / "ticker_stoplist.csv"
)


def load_stop_tickers(path: Path = STOPLIST_CSV) -> frozenset[str]:
    """Loads the user-editable jargon stoplist.

    Args:
        path: CSV with a 'symbol' column.

    Returns:
        Upper-cased symbols from the CSV, or the built-in seed when the
        file is absent.

    Raises:
        ValueError: When the CSV lacks a 'symbol' column or has no usable
            rows. A stoplist that silently loaded empty would let 'CEO'
            back into the mention counts.
    """
    if not Path(path).is_file():
        return _STOP_TICKERS_SEED
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path} needs a 'symbol' column")
    syms = {str(s).strip().upper() for s in df["symbol"] if str(s).strip()}
    if not syms:
        raise ValueError(f"{path} has no usable rows")
    return frozenset(syms)


STOP_TICKERS: frozenset[str] = load_stop_tickers()


# ---------------------------------------------------------------------------
# The allowlist: real tickers the bare-word pass would otherwise never see.
#
# It closes two separate gaps with one mechanism:
#
#   1. Too short. WORD_BARE is [A-Z]{4,5}, so every 1-3 letter ticker is
#      invisible in bare form and counted only when someone types a $
#      sign. Reddit barely uses cashtags, so for names like MU, AMD, IBM
#      and QQQ "cashtag only" means "almost never counted" (bare CAPS
#      mentions outnumber cashtags by well over an order of magnitude).
#   2. English-word collision. META, SOFI, HOOD, COIN, UBER, SHOP and
#      similar names are classified `cashtag_only` by the word-frequency
#      screen because their lower-case forms are common English. That is
#      right for "meta" and wrong for "META".
#
# The safety property both rely on: the bare pass is case-sensitive (it
# scans the original text, never the uppercased copy), so an allowlisted
# symbol only matches when the poster actually typed it in capitals.
# "coin" stays a word; "COIN" becomes Coinbase.
#
# Deliberately not on the list: AI (the technology), PE (price/earnings),
# EV, IT, all finance or tech abbreviations already in the jargon
# stoplist; the single letters X, T, C, V and F, which collide with
# everything; and CAT, KO, GOLD, COST, LOW, NOW and TEAM, where the
# measured CAPS share showed the English word winning.
#
# Changing this file changes every mention count, and it takes effect at
# ingestion: run a full rebuild for history to re-count under it.
# ---------------------------------------------------------------------------
ALLOWLIST_CSV = (
    Path(__file__).resolve().parent.parent / "config" / "ticker_allowlist.csv"
)


def load_allow_tickers(path: Path = ALLOWLIST_CSV) -> frozenset[str]:
    """Loads the user-editable allowlist.

    Args:
        path: CSV with a 'symbol' column.

    Returns:
        Upper-cased symbols from the CSV. A missing file means no
        allowlist, which is the previous behaviour.

    Raises:
        ValueError: When the CSV lacks a 'symbol' column.
    """
    if not Path(path).is_file():
        return frozenset()
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path} needs a 'symbol' column")
    return frozenset(str(x).strip().upper() for x in df["symbol"]
                     if str(x).strip())


ALLOW_TICKERS: frozenset[str] = load_allow_tickers()
# The symbols WORD_BARE cannot reach on its own (1-3 letters) get their
# own case-sensitive alternation; the 4-5 letter ones are already matched
# and only need the stoplist override in extract_tickers_from_text().
_SHORT_ALLOW = sorted((t for t in ALLOW_TICKERS if len(t) < 4), key=len,
                      reverse=True)
WORD_ALLOW_SHORT = (re.compile(r"\b(" + "|".join(_SHORT_ALLOW) + r")\b")
                    if _SHORT_ALLOW else None)

# Bare-word-only stoplist: common Reddit / finance prose that is also a
# valid 4-5 letter symbol. Cashtags for these symbols still count.
BARE_PROSE_STOP: frozenset[str] = frozenset(
    {
        "ABOUT",
        "AFTER",
        "AGAIN",
        "ALSO",
        "BACK",
        "BEEN",
        "BEFORE",
        "BEING",
        "BEST",
        "BOTH",
        "CALL",
        "CAME",
        "CASE",
        "COME",
        "CORP",
        "COST",
        "DAYS",
        "DOES",
        "DONE",
        "DOWN",
        "DRUG",
        "EACH",
        "EDIT",
        "ELSE",
        "EVEN",
        "EVER",
        "FEEL",
        "FIND",
        "FIRST",
        "FIVE",
        "FOUR",
        "FROM",
        "FULL",
        "GAIN",
        "GAVE",
        "GIVE",
        "GOOD",
        "GONE",
        "HARD",
        "HALF",
        "HAVE",
        "HELP",
        "HERE",
        "HIGH",
        "HOLD",
        "HOUR",
        "HOPE",
        "INTO",
        "JUST",
        "KEEP",
        "KEPT",
        "KNOW",
        "LAST",
        "LEFT",
        "LIFE",
        "LIKE",
        "LINE",
        "LONG",
        "LOOK",
        "LOSS",
        "LOW",
        "LOVE",
        "LUCK",
        "MADE",
        "MAKE",
        "MANY",
        "MEAN",
        "MORE",
        "MOST",
        "MUCH",
        "MUST",
        "MOVE",
        "NEAR",
        "NEED",
        "NEXT",
        "NICE",
        "ONCE",
        "ONLY",
        "OPEN",
        "OVER",
        "PART",
        "PICK",
        "PLAY",
        "POST",
        "PUT",
        "READ",
        "REAL",
        "RIGHT",
        "SAID",
        "SAME",
        "SEEN",
        "SELL",
        "SHOW",
        "SOME",
        "SUCH",
        "TAKE",
        "TALK",
        "TELL",
        "THAN",
        "THAT",
        "THEM",
        "THEN",
        "THEY",
        "THIS",
        "TIME",
        "TOLD",
        "TOOK",
        "TURN",
        "VERY",
        "WAIT",
        "WANT",
        "WELL",
        "WENT",
        "WERE",
        "WHAT",
        "WHEN",
        "WILL",
        "WITH",
        "WORD",
        "WORK",
        "YEAR",
        "YOUR",
        "ZERO",
        "AREA",
        "BASE",
        "CARE",
        "CASH",
        "DATA",
        "FACT",
        "FAST",
        "FLOW",
        "FORM",
        "FREE",
        "FUND",
        "GAME",
        "GROW",
        "HEAD",
        "HOME",
        "IDEA",
        "INFO",
        "KIND",
        "LIST",
        "LIVE",
        "MAIN",
        "MIND",
        "NAME",
        "NEWS",
        "NOTE",
        "PLAN",
        "POINT",
        "RATE",
        "REST",
        "RULE",
        "SAFE",
        "SIDE",
        "SURE",
        "TEAM",
        "TOLD",
        "TRUE",
        "TURN",
        "TYPE",
        "USED",
        "WAYS",
        "WEEK",
        # Words Redditors often type in ALL CAPS that are also real
        # tickers/ETFs; cashtags ($HODL) still count, bare caps do not.
        "AWAY",
        "CASH",
        "EASY",
        "EDGE",
        "FREE",
        "GOLD",
        "HODL",
        "HUGE",
        "LOAN",
        "LOSS",
        "MEME",
        "MOON",
        "NICE",
        "PLAN",
        "PLAY",
        "PUMP",
        "REAL",
        "RIDE",
        "SAFE",
        "SAVE",
        "SEEM",
        "SIZE",
        "TEST",
        "TLDR",
    }
)

CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")
# Bare caps: 4-5 letters only (avoids YOU, FOR, ARE, ON, ... as tickers).
WORD_BARE = re.compile(r"\b([A-Z]{4,5})\b")

# Data-driven word-ticker screening (see src/screen_tickers.py): tickers
# classified 'cashtag_only' there are English words in disguise (EDGE,
# LOAN, RENT ...). Their bare-caps mentions are ignored; $CASHTAG mentions
# still count. Regenerate the CSV with `python -m src.screen_tickers`.
from src.config import DATA_DIR  # noqa: E402
CLASSIFICATION_CSV = Path(DATA_DIR) / "reference" / "ticker_classification.csv"


def load_cashtag_only_tickers(path: Path = CLASSIFICATION_CSV) -> frozenset[str]:
    """Reads the 'cashtag_only' tickers from screen_tickers.py's output.

    Args:
        path: The classification CSV.

    Returns:
        The tickers classified 'cashtag_only', or an empty set when the
        CSV has not been generated yet, so extraction works without it.
    """
    if not Path(path).is_file():
        return frozenset()
    df = pd.read_csv(path)
    return frozenset(df.loc[df["classification"] == "cashtag_only", "ticker"])


# Loaded once at import time.
SCREENED_STOP: frozenset[str] = load_cashtag_only_tickers()


def _strip_cashtags_for_word_pass(text_upper: str) -> str:
    """Removes $TICKER spans so the bare-word pass does not double-count
    GME from $GME."""
    return CASHTAG.sub(" ", text_upper)


def extract_tickers_from_text(
    text: str,
    universe: set[str],
    *,
    cashtags_only: bool,
) -> list[str]:
    """Extracts the tickers mentioned in one piece of text.

    Args:
        text: The text to scan.
        universe: Valid symbols; anything outside it is ignored.
        cashtags_only: Skip the bare-word passes.

    Returns:
        Tickers in order of appearance, all cashtags first, then 4-5
        letter bare words, then short allowlisted bare words. Duplicates
        in the text are kept so mention counts reflect frequency.
    """
    if not text or not isinstance(text, str):
        return []

    t = text.upper()
    out: list[str] = []

    for m in CASHTAG.finditer(t):
        sym = m.group(1)
        if sym in STOP_TICKERS:
            continue
        if sym in universe:
            out.append(sym)

    if cashtags_only:
        return out

    # Scan the original text, not the uppercased copy. Only words the
    # poster actually wrote in ALL CAPS ("bought NVDA calls") can be bare
    # tickers. Uppercasing first would turn every ordinary word ("edge",
    # "loan", "meme") into a fake all-caps match and make EDGE and LOAN
    # look like top-mentioned tickers.
    stripped = _strip_cashtags_for_word_pass(text)
    for m in WORD_BARE.finditer(stripped):
        sym = m.group(1)
        if sym in STOP_TICKERS:
            continue                      # Jargon is never a ticker.
        if sym not in ALLOW_TICKERS and (sym in BARE_PROSE_STOP
                                         or sym in SCREENED_STOP):
            continue                      # An English word in disguise.
        if sym in universe:
            out.append(sym)

    # The 1-3 letter allowlist (MU, AMD, IBM ...), which WORD_BARE's
    # [A-Z]{4,5} cannot reach. Same case-sensitive text, same universe check.
    if WORD_ALLOW_SHORT is not None:
        for m in WORD_ALLOW_SHORT.finditer(stripped):
            sym = m.group(1)
            if sym in STOP_TICKERS:
                continue
            if sym in universe:
                out.append(sym)

    return out


def mentions_for_post(
    title: str,
    selftext: str,
    universe: set[str],
    *,
    cashtags_only: bool,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Extracts mentions from a post's title and body separately.

    Returns:
        A pair (title_rows, body_rows), each a list of (ticker, source)
        where source is "title" or "body".
    """
    title_hits = extract_tickers_from_text(title, universe, cashtags_only=cashtags_only)
    body_hits = extract_tickers_from_text(selftext, universe, cashtags_only=cashtags_only)
    t_rows = [(sym, "title") for sym in title_hits]
    b_rows = [(sym, "body") for sym in body_hits]
    return t_rows, b_rows


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring for usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Extract tickers from WSB post Parquet")
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Input posts .parquet")
    p.add_argument("--out", type=Path, required=True, help="Output long mentions .parquet")
    p.add_argument(
        "--mode",
        choices=("default", "cashtag-only"),
        default="default",
        help="default: $TICKER + 4–5 letter bare caps (minus prose stops). "
        "cashtag-only: $TICKER only (cleaner, lower recall).",
    )
    p.add_argument(
        "--universe-cache",
        type=Path,
        default=Path(DATA_DIR) / "reference" / "nasdaq_trader",
        help="Directory to cache nasdaqlisted.txt / otherlisted.txt",
    )
    p.add_argument(
        "--universe-max-age-days",
        type=float,
        default=7.0,
        help="Re-download symbol files if cache is older than this",
    )
    p.add_argument(
        "--force-refresh-universe",
        action="store_true",
        help="Always re-download Nasdaq Trader symbol files",
    )
    p.add_argument(
        "--daily-out",
        type=Path,
        default=None,
        help="Optional: write date,ticker,mention_count (all sources combined)",
    )
    args = p.parse_args(argv)

    if not args.inp.is_file():
        print(f"Input not found: {args.inp}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    universe = load_us_ticker_universe(
        args.universe_cache,
        max_cache_age_days=args.universe_max_age_days,
        force_refresh=args.force_refresh_universe,
    )

    cashtags_only = args.mode == "cashtag-only"

    def _process_frame(df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for pid, date, title, body in zip(
            df["id"],
            df["date"],
            df["title"].fillna("").astype(str),
            df["selftext"].fillna("").astype(str),
        ):
            if pd.isna(pid) or pd.isna(date):
                continue
            t_rows, b_rows = mentions_for_post(
                title, body, universe, cashtags_only=cashtags_only
            )
            for sym, src in t_rows + b_rows:
                rows.append(
                    {
                        "post_id": pid,
                        "date": str(date),
                        "ticker": sym,
                        "source": src,
                    }
                )
        return pd.DataFrame(rows)

    df = pd.read_parquet(args.inp, columns=["id", "date", "title", "selftext"])
    long_df = _process_frame(df)

    if long_df.empty:
        logger.warning("No ticker mentions found (empty input or no matches).")
        long_df.to_parquet(args.out, index=False)
        return 0

    long_df.to_parquet(args.out, index=False)
    logger.info("Wrote %s mention rows to %s", len(long_df), args.out)

    if args.daily_out:
        daily = (
            long_df.groupby(["date", "ticker"], as_index=False)
            .size()
            .rename(columns={"size": "mention_count"})
        )
        args.daily_out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(args.daily_out, index=False)
        logger.info("Wrote %s daily ticker rows to %s", len(daily), args.daily_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
