"""
themes.py
=========
Two complementary theme signals, built to be compared side by side:

SIGNAL 1 - KEYWORD themes (direct mentions)
  Scan each post's raw text for curated words/phrases. A post about
  "HBM memory pricing" and "DRAM capacity" rolls up into the `memory`
  theme even if it never names a ticker.

  Entry point: build_daily_theme_counts(posts_df)
  Returns: DataFrame(date, theme, keyword_count, keyword_weighted)

  Matching is one tokenisation pass per post + hash lookups (fast: ~12k
  posts/sec vs ~600/sec for the old per-theme regex scan). Single-word
  keywords must match a whole token; multi-word phrases are matched as
  substrings of the lowercased text. Each post counts AT MOST ONCE per
  theme (same per-post dedupe rule as the ticker pipeline).

  NOTE (2026-07-06): the score**2 upvote weighting was REMOVED everywhere -
  archived scores are FINAL scores, so weighting day-t mentions by them
  leaks future information into any backtest (see build_mentions.py and
  design_decisions.xlsx #30). The *_weighted columns are still emitted for
  notebook compatibility but are ALWAYS 0 now.

  IMPORTANT: keyword lists contain WORDS AND PHRASES ONLY - no bare ticker
  symbols. Matching is case-insensitive, so a symbol like C (Citigroup) or
  O (Realty Income) would match every ordinary "c"/"o" in prose. Ticker
  exposure to a theme is Signal 2's job.

SIGNAL 2 - INFERRED themes (ticker -> theme)
  Map the already-extracted daily ticker counts (notebook 02's output,
  which had all the stop-list/screening precision applied) onto theme
  buckets: NVDA implies the AI trade, SHEL implies energy, and so on.
  A ticker may belong to several themes.

  Entry point: build_inferred_theme_counts(daily_ticker_counts_df)
  Returns: DataFrame(date, theme, inferred_count, inferred_weighted)

  This signal costs nothing to compute - it is a groupby over a file that
  already exists.

Combine both with combine_theme_signals() -> one row per (date, theme)
with all four count columns, zeros where a signal is silent.

TRADEABLE BY DESIGN: every theme is anchored to a liquid instrument in
THEME_ETFS (semiconductors -> SMH, gold_metals -> GLD, europe_defense ->
EUAD ...). If a theme's mentions spike, there is a concrete thing to
back-test it against and, eventually, trade. Vague non-tradeable themes
(options chatter, earnings chatter, IPO chatter) were deliberately removed;
short_squeeze / meme_stocks stay because their proxy (GME) is tradeable
and they are the project's home turf.

CLI (signal 2 only - signal 1 is called from notebook 04 directly):
  python -m src.themes --in daily_ticker_counts.parquet --out daily_theme_counts.parquet
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# EDITABLE CONFIG (2026-07-31): the theme definitions live in config/*.csv,
# NOT in this file.  Desk instruction: "make an easy to edit file with each
# theme ... i want the word mapping to a theme and the etf list to be both
# easy to edit."  Edit the CSVs (Excel is fine), rerun the pipeline, done.
#
#   config/theme_keywords.csv       theme,keyword        (Signal 1 wording)
#   config/theme_etfs.csv           theme,etf,fallbacks,note
#                                   fallbacks are pipe-separated, anchor
#                                   first;  an EMPTY etf cell = tracked but
#                                   NOT tradeable (crypto, cannabis...)
#   config/theme_tickers.csv        theme,ticker,source  (Signal 2 mapping;
#                                   source records WHY - "curated" or the
#                                   ETF whose constituent list it came from)
#   config/approved_instruments.csv symbol,bloomberg,name,note - the firm-
#                                   approved tradeable list.  Every anchor
#                                   and fallback must appear here, so a typo
#                                   fails LOUDLY at import, not silently at
#                                   the Bloomberg pull.
#
# The keyword rules are unchanged: WORDS AND PHRASES ONLY, no bare ticker
# symbols (matching is case-insensitive - a symbol like C or O would match
# every ordinary "c"/"o" in prose).  Ticker exposure is Signal 2's job.
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _config_rows(fname: str, required: tuple) -> list:
    """Read one config CSV, strip whitespace, fail loudly on a bad header.
    utf-8-sig so a file saved from Excel (BOM) still parses."""
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
    out: dict = {}
    for row in _config_rows("theme_keywords.csv", ("theme", "keyword")):
        if row["theme"] and row["keyword"]:
            out.setdefault(row["theme"], []).append(row["keyword"])
    if not out:
        raise ValueError("config/theme_keywords.csv has no keyword rows")
    return out


def _load_approved_instruments() -> dict:
    out = {}
    for row in _config_rows("approved_instruments.csv",
                            ("symbol", "bloomberg", "name")):
        if row["symbol"]:
            out[row["symbol"]] = row
    return out


def _load_theme_etfs() -> tuple:
    """(THEME_ETFS, THEME_ETF_FALLBACKS) from config/theme_etfs.csv.
    A row with an empty etf cell is a TRACKED-ONLY theme (no approved
    instrument) and is deliberately absent from THEME_ETFS - that is what
    keeps it out of every tradeable list on the dashboard.  Every symbol
    used must be on the approved list; the anchor always leads its own
    fallback chain so the two files cannot disagree about it."""
    approved = _load_approved_instruments()
    etfs: dict = {}
    fallbacks: dict = {}
    for row in _config_rows("theme_etfs.csv", ("theme", "etf", "fallbacks")):
        theme = row["theme"]
        if not theme or not row["etf"]:
            continue                      # tracked-only theme - no anchor
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
    out: dict = {}
    for row in _config_rows("theme_tickers.csv", ("theme", "ticker")):
        if row["theme"] and row["ticker"]:
            out.setdefault(row["theme"], set()).add(row["ticker"].upper())
    if not out:
        raise ValueError("config/theme_tickers.csv has no rows")
    return out


# Loaded ONCE at import.  Same public names, same types as the old in-file
# literals, so every existing importer (dashboard, analytics, notebooks,
# tests) works unchanged.
THEME_KEYWORDS = _load_theme_keywords()
THEME_ETFS, THEME_ETF_FALLBACKS = _load_theme_etfs()
THEME_TICKERS = _load_theme_tickers()
APPROVED_INSTRUMENTS = _load_approved_instruments()


# ---------------------------------------------------------------------------
# INTERNATIONAL COVERAGE (Europe / Japan) - retail posts refer to foreign
# companies by NAME ("Rheinmetall", "Fanuc"), almost never by local ticker
# ("RHM.DE", "6954.T"). So the counting side is handled by the company names
# in THEME_KEYWORDS above; this map provides the PRICING side - the US-listed
# ADR that proxies each name, so overlays and backtests have a price line.
# The Bloomberg puller requests every symbol here. Only liquid, verified ADR
# symbols - a wrong symbol silently pulls nothing.
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
    """word -> themes dict for single-word keywords, phrase -> themes dict
    for multi-word ones. Hash lookups replace the old 20-regexes-per-post
    scan - that is where the ~25x speedup comes from."""
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
    """All themes whose keywords appear in one post's text (lowercased once)."""
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
    """
    Signal 1: scan each post's title + selftext for theme keywords.

    posts_df must have columns: date, title, selftext (score optional).

    Returns DataFrame(date, theme, keyword_count, keyword_weighted) where
      keyword_count    = number of posts that day mentioning the theme
                         (each post counts once per theme, however many
                         keywords it contains - same dedupe rule as the
                         ticker pipeline)
      keyword_weighted = ALWAYS 0 - the score**2 weighting was removed
                         because archived scores leak future information
                         (kept as a column only so older notebooks run)
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
    daily["keyword_weighted"] = 0   # deprecated, see docstring
    return daily


def build_ticker_to_themes(theme_tickers=THEME_TICKERS):
    lookup: dict[str, list[str]] = {}
    for theme, tickers in theme_tickers.items():
        for ticker in tickers:
            lookup.setdefault(ticker, []).append(theme)
    return lookup


def build_inferred_theme_counts(daily_ticker_counts: pd.DataFrame,
                                theme_tickers=THEME_TICKERS) -> pd.DataFrame:
    """
    Signal 2: roll the daily ticker counts (notebook 02's output) up into
    themes. NVDA mentions count toward semiconductors, ai AND ai_megacap.

    daily_ticker_counts must have columns: date, ticker, mention_count.

    Returns DataFrame(date, theme, inferred_count, inferred_weighted).
    inferred_weighted is ALWAYS 0 (score**2 weighting removed - archived
    scores leak future information; column kept for notebook compatibility).
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
    out["inferred_weighted"] = 0   # deprecated, see docstring
    return out


def combine_theme_signals(keyword_df: pd.DataFrame,
                          inferred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join the two signals on (date, theme); a theme silent in one signal
    gets 0 there. Result columns: date, theme, keyword_count,
    keyword_weighted, inferred_count, inferred_weighted.
    """
    merged = keyword_df.merge(inferred_df, on=["date", "theme"], how="outer")
    count_cols = ["keyword_count", "keyword_weighted", "inferred_count", "inferred_weighted"]
    for col in count_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged[count_cols] = merged[count_cols].fillna(0).astype("int64")
    return merged.sort_values(["date", "theme"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI (signal 2 only — signal 1 is called from notebook 04 directly)
# ---------------------------------------------------------------------------
def main(argv=None):
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
