"""Theme definitions and the two theme-mention signals.

The theme definitions (keywords, anchor ETFs, ticker membership, the
approved instrument list) are read from config/*.csv at import time and
exposed as module constants (THEME_KEYWORDS, THEME_ETFS,
THEME_ETF_FALLBACKS, HIDDEN_THEMES, THEME_TICKERS, APPROVED_INSTRUMENTS).
On top of them the module builds two complementary theme signals:

Signal 1, keyword themes (direct mentions)
  Scan each post's raw text for curated words and phrases. A post about
  "HBM memory pricing" and "DRAM capacity" rolls up into the `memory`
  theme even if it never names a ticker.

  Entry point: ``build_daily_theme_counts(posts_df)`` returning
  DataFrame(date, theme, keyword_count, keyword_weighted); the per-post
  matcher ``themes_in_text()`` is what the live fold calls.

  Matching is one tokenisation pass per post plus hash lookups, roughly
  20x faster than a per-theme regex scan. Single-word keywords must match
  a whole token; multi-word phrases are matched as substrings of the
  lowercased text. Each post counts at most once per theme, the same
  per-post dedupe rule the ticker pipeline uses.

  Keyword lists contain words and phrases only, never bare ticker
  symbols: matching is case-insensitive, so a symbol like C (Citigroup)
  or O (Realty Income) would match every ordinary "c"/"o" in prose.
  Ticker exposure to a theme is Signal 2's job.

Signal 2, inferred themes (ticker -> theme)
  Map the already-extracted daily ticker counts (which carry the
  stop-list and screening precision of the ticker pipeline) onto theme
  buckets: NVDA implies the AI trade, SHEL implies energy, and so on. A
  ticker may belong to several themes.

  Entry point: ``build_inferred_theme_counts(daily_ticker_counts_df)``
  returning DataFrame(date, theme, inferred_count, inferred_weighted).
  It is a groupby over a file that already exists.

``combine_theme_signals()`` outer-joins the two into one row per
(date, theme) with all four count columns, zeros where a signal is silent.

The ``*_weighted`` columns are always 0. Archived post scores are final
scores, so weighting day-t mentions by them leaks future information into
any backtest (see build_mentions.py); the columns are emitted only so
older readers keep working.

Every theme is tradeable by design: it is anchored to a liquid instrument
in THEME_ETFS (semiconductors -> SMH, gold_metals -> GLD, europe_defense
-> EUAD ...), so a spike in a theme's mentions has a concrete thing to
back-test against and trade. Vague non-tradeable themes (options chatter,
earnings chatter, IPO chatter) are excluded; short_squeeze / meme_stocks
stay because their proxy (GME) is tradeable.

Command line (Signal 2 only)::

    python -m src.themes --in daily_ticker_counts.parquet --out daily_theme_counts.parquet
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Editable config: the theme definitions live in config/*.csv, not in
# this file. Edit the CSVs (a spreadsheet editor is fine), rerun the
# pipeline.
#
#   config/theme_keywords.csv       theme,keyword        (Signal 1 wording)
#   config/theme_etfs.csv           theme,etf,fallbacks,note
#                                   fallbacks are pipe-separated, anchor
#                                   first; an empty etf cell means tracked
#                                   but not tradeable (crypto, cannabis...)
#   config/theme_tickers.csv        theme,ticker,source  (Signal 2 mapping;
#                                   source records why: "curated" or the
#                                   ETF whose constituent list it came from)
#   config/approved_instruments.csv symbol,bloomberg,name,note: the
#                                   approved tradeable list. Every anchor
#                                   and fallback must appear here, so a
#                                   typo fails loudly at import, not
#                                   silently at the Bloomberg pull.
#
# Keyword rules: words and phrases only, no bare ticker symbols (matching
# is case-insensitive; a symbol like C or O would match every ordinary
# "c"/"o" in prose). Ticker exposure is Signal 2's job.
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _config_rows(fname: str, required: tuple) -> list:
    """Reads one config CSV, strips whitespace, fails loudly on a bad header.
    Decoded as utf-8-sig so a file saved with a BOM still parses."""
    path = _CONFIG_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The theme definitions live in config/*.csv "
            "(see src/themes.py header). Restore the file from git.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in required if c not in (rdr.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; "
                             f"expected header {list(required)}")
        return [{k: (row.get(k) or "").strip() for k in rdr.fieldnames}
                for row in rdr]


def _load_theme_keywords() -> dict:
    """Returns {theme: [keyword, ...]} from config/theme_keywords.csv."""
    out: dict = {}
    for row in _config_rows("theme_keywords.csv", ("theme", "keyword")):
        if row["theme"] and row["keyword"]:
            out.setdefault(row["theme"], []).append(row["keyword"])
    if not out:
        raise ValueError("config/theme_keywords.csv has no keyword rows")
    return out


def _load_approved_instruments() -> dict:
    """Returns {symbol: row} from config/approved_instruments.csv."""
    out = {}
    for row in _config_rows("approved_instruments.csv",
                            ("symbol", "bloomberg", "name")):
        if row["symbol"]:
            out[row["symbol"]] = row
    return out


def _load_theme_etfs() -> tuple:
    """Returns (THEME_ETFS, THEME_ETF_FALLBACKS) from config/theme_etfs.csv.

    A row with an empty etf cell is a tracked-only theme (no approved
    instrument) and is deliberately absent from THEME_ETFS, which is what
    keeps it out of every tradeable list on the dashboard. Every symbol
    used must be on the approved list; the anchor always leads its own
    fallback chain so the two files cannot disagree about it.

    Raises:
        ValueError: When a symbol is not on the approved list or no
            tradeable theme rows exist.
    """
    approved = _load_approved_instruments()
    etfs: dict = {}
    fallbacks: dict = {}
    for row in _config_rows("theme_etfs.csv", ("theme", "etf", "fallbacks")):
        theme = row["theme"]
        if not theme or not row["etf"]:
            continue                      # Tracked-only theme: no anchor.
        anchor = row["etf"]
        chain = [s.strip() for s in row["fallbacks"].split("|") if s.strip()]
        if anchor not in chain:
            chain.insert(0, anchor)
        for sym in chain:
            if sym not in approved:
                raise ValueError(
                    f"config/theme_etfs.csv: '{sym}' (theme {theme}) is not "
                    "in config/approved_instruments.csv - add it there "
                    "first (symbol + exact Bloomberg code), or fix the typo.")
        etfs[theme] = anchor
        fallbacks[theme] = chain
    if not etfs:
        raise ValueError("config/theme_etfs.csv has no tradeable theme rows")
    return etfs, fallbacks


def _load_theme_tickers() -> dict:
    """Returns {theme: {TICKER, ...}} from config/theme_tickers.csv."""
    out: dict = {}
    for row in _config_rows("theme_tickers.csv", ("theme", "ticker")):
        if row["theme"] and row["ticker"]:
            out.setdefault(row["theme"], set()).add(row["ticker"].upper())
    if not out:
        raise ValueError("config/theme_tickers.csv has no rows")
    return out


def _load_hidden_themes() -> frozenset:
    """Themes whose ``show_on_dashboard`` column in theme_etfs.csv is no.

    Display only: a hidden theme is still tracked, scored and written to
    every store, so hiding one changes no computed result and cannot make
    the frozen thresholds stale. To remove a theme from the detector's
    universe, delete its row (which requires a research pass).
    """
    out = set()
    for row in _config_rows("theme_etfs.csv", ("theme",)):
        if row["theme"] and row.get("show_on_dashboard", "yes").strip(
                ).lower() in ("no", "false", "0"):
            out.add(row["theme"])
    return frozenset(out)


# Loaded once at import and exposed under stable public names so every
# importer (dashboard, analytics, tests) reads the same definitions.
THEME_KEYWORDS = _load_theme_keywords()
THEME_ETFS, THEME_ETF_FALLBACKS = _load_theme_etfs()
HIDDEN_THEMES = _load_hidden_themes()
THEME_TICKERS = _load_theme_tickers()
APPROVED_INSTRUMENTS = _load_approved_instruments()


# ---------------------------------------------------------------------------
# International coverage (Europe / Japan). Retail posts refer to foreign
# companies by name ("Rheinmetall", "Fanuc"), almost never by local
# ticker ("RHM.DE", "6954.T"), so the counting side is handled by the
# company names in THEME_KEYWORDS; this map provides the pricing side,
# the US-listed ADR that proxies each name, so overlays and backtests
# have a price line. The Bloomberg puller requests every symbol here.
# Only liquid, verified ADR symbols belong: a wrong symbol silently pulls
# nothing.
# ---------------------------------------------------------------------------
INTERNATIONAL_ADRS: dict[str, str] = {
    # robotics / bearings / motion control (Japan + Europe)
    "Fanuc": "FANUY",
    "Yaskawa": "YASKY",
    "Keyence": "KYCCY",
    "THK": "THKLY",
    "Nabtesco": "NCTKY",
    "SKF": "SKFRY",
    # semiconductor equipment (Japan)
    "Tokyo Electron": "TOELY",
    "Advantest": "ATEYY",
    "Lasertec": "LSRCY",
    "Disco Corp": "DSCSY",
    "SUMCO": "SUOPY",
    # European defense
    "Rheinmetall": "RNMBY",
    "BAE Systems": "BAESY",
    "Thales": "THLLY",
    "Saab": "SAABY",
    "Leonardo": "FINMY",
    "Airbus": "EADSY",
    # EV / batteries / Japan majors
    "BYD": "BYDDY",
    "Softbank": "SFTBY",
}

# Tokens are runs of letters/digits in the lowercased text, so "0DTE" and
# "3nm" survive as single tokens. Anything with a space, hyphen or dot in
# the keyword is treated as a phrase and substring-matched instead.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Built lazily from THEME_KEYWORDS on first use.
_WORD_TO_THEMES: dict[str, tuple[str, ...]] = {}
_PHRASE_TO_THEMES: dict[str, tuple[str, ...]] = {}


def _get_keyword_lookup():
    """Returns (word -> themes, phrase -> themes) lookup dicts.

    Single-word keywords go in the first, multi-word phrases in the
    second. Hash lookups replace a per-theme regex scan of every post,
    which is where the speedup comes from.
    """
    if _WORD_TO_THEMES or _PHRASE_TO_THEMES:
        return _WORD_TO_THEMES, _PHRASE_TO_THEMES
    word_acc: dict[str, list[str]] = {}
    phrase_acc: dict[str, list[str]] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        for kw in keywords:
            k = kw.lower()
            acc = phrase_acc if (" " in k or "-" in k or "." in k) else word_acc
            acc.setdefault(k, [])
            if theme not in acc[k]:
                acc[k].append(theme)
    _WORD_TO_THEMES.update({w: tuple(t) for w, t in word_acc.items()})
    _PHRASE_TO_THEMES.update({p: tuple(t) for p, t in phrase_acc.items()})
    return _WORD_TO_THEMES, _PHRASE_TO_THEMES


def themes_in_text(text: str) -> set[str]:
    """Returns the set of themes whose keywords appear in one post's text.

    The text is lowercased once; single-word keywords must match a whole
    token, phrases are substring-matched.
    """
    word_map, phrase_map = _get_keyword_lookup()
    lowered = text.lower()
    found: set[str] = set()
    for token in set(_TOKEN_RE.findall(lowered)):
        hit = word_map.get(token)
        if hit:
            found.update(hit)
    for phrase, themes in phrase_map.items():
        if phrase in lowered:
            found.update(themes)
    return found


def build_daily_theme_counts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Signal 1: scans each post's title + selftext for theme keywords.

    Args:
        posts_df: Posts frame with date, title and selftext columns.

    Returns:
        DataFrame(date, theme, keyword_count, keyword_weighted).
        keyword_count is the number of posts that day mentioning the
        theme; each post counts once per theme however many keywords it
        contains, the same dedupe rule as the ticker pipeline.
        keyword_weighted is always 0 (see the module docstring).
    """
    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str)
    dates = posts_df["date"].astype(str)

    rows = []
    for date, title, body in zip(dates, titles, bodies):
        found = themes_in_text(title + " " + body)
        for theme in found:
            rows.append((date, theme))

    if not rows:
        return pd.DataFrame(columns=["date", "theme", "keyword_count", "keyword_weighted"])

    long_df = pd.DataFrame(rows, columns=["date", "theme"])
    daily = (
        long_df.groupby(["date", "theme"])
        .agg(keyword_count=("theme", "size"))
        .reset_index()
    )
    daily["keyword_weighted"] = 0   # Deprecated; see the module docstring.
    return daily


def build_ticker_to_themes(theme_tickers=THEME_TICKERS):
    """Inverts {theme: tickers} into {ticker: [theme, ...]}."""
    lookup: dict[str, list[str]] = {}
    for theme, tickers in theme_tickers.items():
        for ticker in tickers:
            lookup.setdefault(ticker, []).append(theme)
    return lookup


def build_inferred_theme_counts(daily_ticker_counts: pd.DataFrame,
                                theme_tickers=THEME_TICKERS) -> pd.DataFrame:
    """Signal 2: rolls the daily ticker counts up into themes.

    A ticker in several themes counts toward each: NVDA mentions count
    toward semiconductors, ai and ai_megacap.

    Args:
        daily_ticker_counts: Frame with date, ticker and mention_count.
        theme_tickers: {theme: tickers} mapping; defaults to THEME_TICKERS.

    Returns:
        DataFrame(date, theme, inferred_count, inferred_weighted).
        inferred_weighted is always 0 (see the module docstring).
    """
    df = daily_ticker_counts

    lookup = build_ticker_to_themes(theme_tickers)
    rows = []
    for date, ticker, count in zip(df["date"], df["ticker"], df["mention_count"]):
        for theme in lookup.get(ticker, ()):
            rows.append((date, theme, count))

    if not rows:
        return pd.DataFrame(columns=["date", "theme", "inferred_count", "inferred_weighted"])

    long_df = pd.DataFrame(rows, columns=["date", "theme", "inferred_count"])
    out = (
        long_df.groupby(["date", "theme"], as_index=False)[["inferred_count"]]
        .sum()
    )
    out["inferred_weighted"] = 0   # Deprecated; see the module docstring.
    return out


def combine_theme_signals(keyword_df: pd.DataFrame,
                          inferred_df: pd.DataFrame) -> pd.DataFrame:
    """Outer-joins the two signals on (date, theme).

    A theme silent in one signal gets 0 there.

    Returns:
        DataFrame with columns date, theme, keyword_count,
        keyword_weighted, inferred_count, inferred_weighted, sorted by
        (date, theme).
    """
    merged = keyword_df.merge(inferred_df, on=["date", "theme"], how="outer")
    count_cols = ["keyword_count", "keyword_weighted", "inferred_count", "inferred_weighted"]
    for col in count_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged[count_cols] = merged[count_cols].fillna(0).astype("int64")
    return merged.sort_values(["date", "theme"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Command line (Signal 2 only).
# ---------------------------------------------------------------------------
def main(argv=None):
    """Rolls a daily ticker counts file up into inferred theme counts."""
    parser = argparse.ArgumentParser(description="Roll ticker mentions up into inferred themes")
    parser.add_argument("--in", dest="inp", required=True, help="daily ticker counts .parquet/.csv")
    parser.add_argument("--out", required=True, help="output daily inferred theme counts path")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.inp) if args.inp.endswith(".parquet") else pd.read_csv(args.inp)
    theme_df = build_inferred_theme_counts(df)

    if theme_df.empty:
        print("No tickers matched any theme - check THEME_TICKERS.")
        return 1

    if args.out.endswith(".parquet"):
        theme_df.to_parquet(args.out, index=False)
    else:
        theme_df.to_csv(args.out, index=False)

    totals = theme_df.groupby("theme")["inferred_count"].sum().sort_values(ascending=False)
    print("Saved", len(theme_df), "rows to", args.out)
    print("\nTotal inferred mentions per theme:")
    for theme, total in totals.items():
        print("  ", theme, ":", int(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
