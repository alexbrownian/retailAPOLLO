"""
Extract stock tickers from WSB post Parquet (title + selftext).

Validates against a US-listed symbol universe (Nasdaq Trader files) and stop lists.
Emits long-format rows: post_id, date, ticker, source.

Cashtags ($GME) are high precision. Bare all-caps words collide with English even when
they are valid tickers (e.g. YOU, FOR); default rules use 4–5 letter bare matches plus
a prose stoplist. Use --mode cashtag-only for maximum precision at lower recall.

Example:
  python -m src.extract_tickers \\
    --in data/raw/wsb_posts_2021-01.parquet \\
    --out data/processed/wsb_ticker_mentions_2021-01.parquet

Optional daily counts (for date × ticker matrices):
  python -m src.extract_tickers --in ... --out ... --daily-out data/processed/wsb_daily_counts.parquet
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

# WSB / finance jargon that is not a tradeable ticker in this context.
STOP_TICKERS: frozenset[str] = frozenset(
    {
        "DD",
        "YOLO",
        "USA",
        "CEO",
        "CFO",
        "IPO",
        "ATH",
        "ITM",
        "OTM",
        "FOMO",
        "FUD",
        "ER",
        "EOD",
        "IMO",
        "IRL",
        "AI",
        "IT",
        "OK",
        "ALL",
        "NEW",
        "NOW",
        "BIG",
        "LOL",
        "WSB",
        "IV",
        "EOY",
        "ETF",
        "ROI",
        "FBI",
        "NYSE",
        "NASDAQ",
        "SEC",
        "EV",
        "USD",
        "GDP",
        "CPI",
    }
)

# Bare-word-only: common Reddit / finance prose that is also a valid 4–5 letter symbol.
# Cashtags for these symbols still count. Extend as you see false positives in your slice.
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
        # --- words Redditors often type in ALL CAPS that are also real
        # --- tickers/ETFs; cashtags ($HODL) still count, bare caps don't.
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
# Bare caps: 4–5 letters only (avoids YOU, FOR, ARE, ON, … as tickers).
WORD_BARE = re.compile(r"\b([A-Z]{4,5})\b")

# Data-driven word-ticker screening (see src/screen_tickers.py): tickers
# classified 'cashtag_only' there are English words in disguise (EDGE, LOAN,
# RENT ...). Their bare-caps mentions are ignored; $CASHTAG mentions still
# count. Regenerate the CSV via notebook 01 or `python -m src.screen_tickers`.
CLASSIFICATION_CSV = (
    Path(__file__).resolve().parent.parent
    / "data" / "reference" / "ticker_classification.csv"
)


def load_cashtag_only_tickers(path: Path = CLASSIFICATION_CSV) -> frozenset[str]:
    """Read screen_tickers.py's output. Returns an empty set if the CSV
    hasn't been generated yet, so everything still works without it."""
    if not Path(path).is_file():
        return frozenset()
    df = pd.read_csv(path)
    return frozenset(df.loc[df["classification"] == "cashtag_only", "ticker"])


# Loaded once at import time.
SCREENED_STOP: frozenset[str] = load_cashtag_only_tickers()


def _strip_cashtags_for_word_pass(text_upper: str) -> str:
    """Remove $TICKER spans so bare-word pass does not double-count GME from $GME."""
    return CASHTAG.sub(" ", text_upper)


def extract_tickers_from_text(
    text: str,
    universe: set[str],
    *,
    cashtags_only: bool,
) -> list[str]:
    """
    Return tickers in order (all cashtags first, then bare words if enabled).
    Duplicates in the text are kept so mention counts reflect frequency.
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

    # IMPORTANT: scan the ORIGINAL text, not the uppercased copy. Only words
    # the poster actually wrote in ALL CAPS ("bought NVDA calls") can be bare
    # tickers. Uppercasing first would turn every ordinary word ("edge",
    # "loan", "meme") into a fake all-caps match - that bug once made EDGE
    # and LOAN look like top-mentioned tickers.
    stripped = _strip_cashtags_for_word_pass(text)
    for m in WORD_BARE.finditer(stripped):
        sym = m.group(1)
        if sym in STOP_TICKERS or sym in BARE_PROSE_STOP or sym in SCREENED_STOP:
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
    """Returns (title_rows, body_rows) as list of (ticker, source)."""
    title_hits = extract_tickers_from_text(title, universe, cashtags_only=cashtags_only)
    body_hits = extract_tickers_from_text(selftext, universe, cashtags_only=cashtags_only)
    t_rows = [(sym, "title") for sym in title_hits]
    b_rows = [(sym, "body") for sym in body_hits]
    return t_rows, b_rows


def main(argv: list[str] | None = None) -> int:
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
        default=Path("data/reference/nasdaq_trader"),
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
