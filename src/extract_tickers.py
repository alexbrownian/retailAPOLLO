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

# ---------------------------------------------------------------------------
# JARGON, NOT TICKERS.  Symbols the crowd uses as words.
#
# Maintained in config/ticker_stoplist.csv so the list can change
# without touching code - the
# same treatment every other mapping in this project already had.  The
# frozenset below is the FALLBACK seed: it keeps a fresh clone working if
# the CSV is missing, and it is the list as it stood before the move.
#
# How a symbol earns a place here: additions use a
# falsifiable test rather than taste: many finance abbreviations have since
# been issued to a real ETF (HYSA, DRAM, BTC, REIT, NASA, DJIA, HVAC, EBIT,
# ROPE), so "is it a listed symbol?" cannot separate them from real
# instruments.  What can: A FUND CANNOT BE DISCUSSED BEFORE IT IS LISTED.
# HYSA carries 792 mentions in 2019 and 1,141 in 2020 against a fund
# launched in 2023; DRAM carries 116 in 2018 against a fund launched in
# 2025.  Those mentions are the words.  Symbols that pass the test - IBIT
# (0 before 2024, its launch year), QQQI, SNDK, SPCX - are left alone, and
# so are old, genuinely-traded ETFs like SPY, ARKK and GLD.  The reason
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
    """The desk-editable jargon list. Falls back to the built-in seed when
    the CSV is absent; raises on a malformed one, because a stoplist that
    silently loads empty would let 'CEO' back into the mention counts."""
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
# THE ALLOWLIST - real tickers the bare-word pass would otherwise never see.
#
# Rationale: short all-caps symbols (e.g. MU) were not being
# counted. Two separate gaps, one mechanism:
#
#   1. TOO SHORT. `WORD_BARE` is [A-Z]{4,5}, so every 1-3 letter ticker was
#      invisible in bare form and counted ONLY when someone typed a $ sign.
#      Measured over six days of comments: MU appears 265 times in bare CAPS
#      against SIX $MU cashtags - a ~44x undercount on the very name behind
#      the June-2026 memory GET OUT. AMD 58 vs 0, IBM 48 vs 0, QQQ 109 vs 2.
#      Reddit barely uses cashtags at all, so "cashtag only" means "almost
#      never counted".
#   2. ENGLISH-WORD COLLISION. META, SOFI, HOOD, COIN, UBER, SHOP and friends
#      were classified `cashtag_only` by the word-frequency screen because
#      their lower-case forms are common English. That is right for "meta"
#      and wrong for "META".
#
# The safety property both rely on: THE BARE PASS IS ALREADY CASE-SENSITIVE
# (it scans the original text, never the uppercased copy), so an allowlisted
# symbol only matches when the poster actually typed it in capitals. "coin"
# stays a word; "COIN" becomes Coinbase.
#
# WHAT IS DELIBERATELY NOT HERE, and why - the exclusions are the evidence
# that this list is judged rather than stuffed: AI (1,468 CAPS hits, the
# technology), PE (125, price/earnings), EV, IT - all finance or tech
# abbreviations already in the jargon stoplist; the single letters X, T, C, V
# and F, which collide with everything; and CAT, KO, GOLD, COST, LOW, NOW and
# TEAM, where the measured CAPS share showed the English word winning.
#
# CHANGING THIS FILE CHANGES EVERY MENTION COUNT, so it takes effect at
# INGESTION: run a FULL rebuild for history to re-count under it.
# ---------------------------------------------------------------------------
ALLOWLIST_CSV = (
    Path(__file__).resolve().parent.parent / "config" / "ticker_allowlist.csv"
)


def load_allow_tickers(path: Path = ALLOWLIST_CSV) -> frozenset[str]:
    """Desk-editable. An empty/missing file simply means no allowlist, which
    is the legacy behaviour retained for comparison."""
    if not Path(path).is_file():
        return frozenset()
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path} needs a 'symbol' column")
    return frozenset(str(x).strip().upper() for x in df["symbol"]
                     if str(x).strip())


ALLOW_TICKERS: frozenset[str] = load_allow_tickers()
# the ones WORD_BARE cannot reach on its own (1-3 letters) get their own
# case-sensitive alternation; the 4-5 letter ones are already matched and
# only needed the stoplist override below
_SHORT_ALLOW = sorted((t for t in ALLOW_TICKERS if len(t) < 4), key=len,
                      reverse=True)
WORD_ALLOW_SHORT = (re.compile(r"\b(" + "|".join(_SHORT_ALLOW) + r")\b")
                    if _SHORT_ALLOW else None)

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
        if sym in STOP_TICKERS:
            continue                      # jargon is never a ticker
        if sym not in ALLOW_TICKERS and (sym in BARE_PROSE_STOP
                                         or sym in SCREENED_STOP):
            continue                      # an English word in disguise
        if sym in universe:
            out.append(sym)

    # the 1-3 letter allowlist (MU, AMD, IBM ...), which WORD_BARE's
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
