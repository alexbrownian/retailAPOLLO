"""Validate every file in ``config/`` before the pipeline runs.

Checks the schema of each CSV and the references between them, so a
typo in one file is caught here rather than as a stack trace three
stages into a data refresh. Run it directly, or let ``tools/preflight.py``
and the test suite call :func:`validate`.

    python Code/tools/validate_config.py           # exit 1 on any error
    python Code/tools/validate_config.py --quiet   # errors only

The checks, in order:

1. Every required file exists and has its required columns.
2. ``settings.csv`` values parse as the type each key expects.
3. ``forums.csv`` has at least one enabled forum and no duplicates.
4. ``theme_etfs.csv``: every anchor and fallback symbol is in
   ``approved_instruments.csv``; the anchor leads its fallback chain;
   no duplicate theme rows.
5. ``theme_keywords.csv`` and ``theme_tickers.csv``: every theme they
   mention exists in ``theme_etfs.csv``; no empty cells.
6. ``ticker_allowlist.csv`` and ``ticker_stoplist.csv`` do not overlap.
7. ``single_name_overrides.csv``: actions are ``include`` or ``exclude``.
8. ``agentic_terms.csv``: every regular expression compiles.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT)

CONFIG_DIR = os.path.join(ROOT, "config")

# file -> required columns
SCHEMA = {
    "settings.csv": ("key", "value"),
    "forums.csv": ("forum", "source", "enabled"),
    "theme_etfs.csv": ("theme", "etf", "fallbacks"),
    "theme_keywords.csv": ("theme", "keyword"),
    "theme_tickers.csv": ("theme", "ticker"),
    "approved_instruments.csv": ("symbol", "bloomberg"),
    "ticker_allowlist.csv": ("symbol",),
    "ticker_stoplist.csv": ("symbol",),
    "etf_constituents.csv": ("etf", "ticker"),
    "agentic_terms.csv": ("category", "pattern"),
    "ai_poll_prompts.csv": ("prompt_id", "prompt"),
    "single_name_overrides.csv": ("symbol", "action"),
}

# settings.csv keys -> parser
_SETTING_TYPES = {
    "show_pipeline_controls": "bool",
    "bot_screen_enabled": "bool",
    "single_name_top_n": "int",
    "single_name_window_days": "int",
    "bot_screen_burst_posts_per_day": "int",
    "bot_screen_threshold": "float",
    "bot_screen_duplicate_jaccard": "float",
}


def _rows(name: str) -> List[dict]:
    path = os.path.join(CONFIG_DIR, name)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        out = []
        for row in reader:
            if (row.get(fields[0]) or "").strip().startswith("#"):
                continue
            out.append({k: (v or "").strip() for k, v in row.items()
                        if k is not None})
        return out


def validate() -> List[str]:
    """Run every check and return the list of error messages (empty = OK)."""
    errors: List[str] = []

    # 1. presence + columns
    for name, required in SCHEMA.items():
        path = os.path.join(CONFIG_DIR, name)
        if not os.path.exists(path):
            errors.append(f"{name}: file is missing")
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
        missing = [c for c in required if c not in header]
        if missing:
            errors.append(f"{name}: missing column(s) {missing}")
    if errors:
        return errors                 # nothing below can run safely

    # 2. settings types
    from src import settings
    seen = set()
    for row in _rows("settings.csv"):
        key = row["key"]
        if not key:
            continue
        if key in seen:
            errors.append(f"settings.csv: duplicate key {key!r}")
        seen.add(key)
        kind = _SETTING_TYPES.get(key)
        try:
            if kind == "bool":
                settings.get_bool(key)
            elif kind == "int":
                settings.get_int(key)
            elif kind == "float":
                settings.get_float(key)
        except (settings.ConfigError, KeyError) as exc:
            errors.append(f"settings.csv: {exc}")
    prov = settings.get("price_provider").strip().lower()
    if prov not in ("auto", "bloomberg", "tiingo"):
        errors.append(f"settings.csv: price_provider must be auto, bloomberg "
                      f"or tiingo, got {prov!r}")
    thr = settings.get_float("bot_screen_threshold")
    if not 0.0 <= thr <= 1.0:
        errors.append(f"settings.csv: bot_screen_threshold must be in "
                      f"[0, 1], got {thr}")
    jac = settings.get_float("bot_screen_duplicate_jaccard")
    if not 0.0 < jac <= 1.0:
        errors.append(f"settings.csv: bot_screen_duplicate_jaccard must be "
                      f"in (0, 1], got {jac}")

    # 3. forums
    forums = _rows("forums.csv")
    names = [r["forum"].lower() for r in forums if r["forum"]]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        errors.append(f"forums.csv: duplicate forum(s) {sorted(dupes)}")
    if not any(r["enabled"].lower() in ("yes", "true", "1")
               for r in forums):
        errors.append("forums.csv: no forum is enabled")
    for r in forums:
        if r["forum"] and r["enabled"].lower() not in (
                "yes", "no", "true", "false", "1", "0"):
            errors.append(f"forums.csv: {r['forum']}: enabled must be "
                          f"yes/no, got {r['enabled']!r}")

    # 4. theme_etfs vs approved_instruments
    approved = {r["symbol"] for r in _rows("approved_instruments.csv")
                if r["symbol"]}
    themes_seen = set()
    tracked_themes = set()
    for r in _rows("theme_etfs.csv"):
        t = r["theme"]
        if not t:
            continue
        if t in themes_seen:
            errors.append(f"theme_etfs.csv: duplicate theme {t!r}")
        themes_seen.add(t)
        tracked_themes.add(t)
        if not r["etf"]:
            continue                    # tracked-only theme
        chain = [x.strip() for x in r["fallbacks"].split("|") if x.strip()]
        if chain and chain[0] != r["etf"]:
            errors.append(f"theme_etfs.csv: {t}: anchor {r['etf']} must "
                          f"lead its fallback chain ({chain[0]} does)")
        for sym in [r["etf"]] + chain:
            if sym not in approved:
                errors.append(f"theme_etfs.csv: {t}: {sym!r} is not in "
                              "approved_instruments.csv")
        flag = r.get("show_on_dashboard", "yes").lower()
        if flag not in ("yes", "no", "true", "false", "1", "0", ""):
            errors.append(f"theme_etfs.csv: {t}: show_on_dashboard must "
                          f"be yes/no, got {flag!r}")

    # 5. keywords / tickers reference known themes
    for fname, col in (("theme_keywords.csv", "keyword"),
                       ("theme_tickers.csv", "ticker")):
        for i, r in enumerate(_rows(fname), start=2):
            if not r["theme"] or not r[col]:
                errors.append(f"{fname}: row {i}: empty theme or {col}")
            elif r["theme"] not in tracked_themes:
                errors.append(f"{fname}: row {i}: theme {r['theme']!r} is "
                              "not in theme_etfs.csv")

    # 6. allow/stop lists disjoint
    allow = {r["symbol"].upper() for r in _rows("ticker_allowlist.csv")
             if r["symbol"]}
    stop = {r["symbol"].upper() for r in _rows("ticker_stoplist.csv")
            if r["symbol"]}
    both = allow & stop
    if both:
        errors.append(f"ticker_allowlist.csv and ticker_stoplist.csv both "
                      f"list {sorted(both)}")

    # 7. overrides
    for r in _rows("single_name_overrides.csv"):
        if r["symbol"] and r["action"].lower() not in ("include", "exclude"):
            errors.append(f"single_name_overrides.csv: {r['symbol']}: "
                          f"action must be include/exclude, got "
                          f"{r['action']!r}")

    # 8. agentic patterns compile
    for i, r in enumerate(_rows("agentic_terms.csv"), start=2):
        try:
            re.compile(r["pattern"], re.IGNORECASE)
        except re.error as exc:
            errors.append(f"agentic_terms.csv: row {i}: bad regex ({exc})")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true",
                    help="print errors only")
    args = ap.parse_args()
    errors = validate()
    if errors:
        print(f"config: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    if not args.quiet:
        n = sum(1 for _ in SCHEMA)
        print(f"config: OK ({n} files validated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
