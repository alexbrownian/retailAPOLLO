"""One command that answers: is this project still sound?

    python tools/preflight.py

Run it after editing a config, before a commit, and on any machine that
has just pulled. It exits 0 when everything a maintainer could plausibly
break is intact, 1 when something is FAILING, and prints WARNINGs for
things that are merely owed.

WHY THIS EXISTS
---------------
Most of what goes wrong in this project does not raise an exception. It
degrades quietly, and the screen keeps looking fine:

* a theme whose anchor lost its price history is silently drawn on a
  fallback - `europe_defense` spent a week priced off a US aerospace ETF
  standing in for a European one;
* a renamed ticker counts zero forever - nine of them were dead before
  the 2026-08-05 audit found them;
* the euphoria universe quietly holds fewer instruments than the config
  defines, because one theme has no priced line anywhere in its chain;
* a config CSV edit that fails validation only shows up at the next full
  run, which may be days away.

Every check below corresponds to something that actually happened. The
point is to make the failure LOUD and EARLY rather than clever.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It never writes, never fetches, never calls the LLM gateway and never
re-fits anything. It is safe to run at any time, including mid-pipeline.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAIL, WARN, OK = "FAIL", "WARN", "ok  "
_results: list[tuple[str, str, str]] = []


def _say(level: str, check: str, detail: str = "") -> None:
    _results.append((level, check, detail))
    mark = {FAIL: "FAIL", WARN: "WARN", OK: "ok  "}[level]
    print(f"  [{mark}] {check}" + (f" - {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def check_configs() -> None:
    """The CSVs load and validate.

    `src/themes.py` raises on a typo'd symbol or an unapproved anchor, so
    importing it IS the check - that is why the loaders were written to
    fail loudly rather than to skip bad rows."""
    try:
        import importlib
        import src.themes as th
        importlib.reload(th)
        _say(OK, "config CSVs parse and validate",
             f"{len(th.THEME_ETFS)} tradeable themes, "
             f"{len(th.APPROVED_INSTRUMENTS)} approved instruments")
    except Exception as e:                              # noqa: BLE001
        _say(FAIL, "config CSVs", f"{type(e).__name__}: {e}")


def check_universe_matches_config() -> None:
    """Does the scored universe hold what the config defines?

    The 36-vs-37 case: a theme with no priced line anywhere in its chain
    is counted in the crowd data and then dropped before scoring. The
    two numbers disagreeing is the only symptom, and nothing printed it
    until this check existed."""
    try:
        import pandas as pd
        from src.themes import THEME_ETFS, THEME_ETF_FALLBACKS
        from src.config import EUPHORIA_EXCLUDED_THEMES, PRICES_PATH
        if not os.path.exists(PRICES_PATH):
            _say(WARN, "universe vs config", "no price store yet")
            return
        priced = set(pd.read_parquet(PRICES_PATH,
                                     columns=["symbol"])["symbol"].unique())
        dark = []
        for t, a in THEME_ETFS.items():
            if t in EUPHORIA_EXCLUDED_THEMES:
                continue
            chain = [a] + [s for s in THEME_ETF_FALLBACKS.get(t, [])
                           if s != a]
            if not any(s in priced for s in chain):
                dark.append(t)
        if dark:
            _say(WARN, "themes that cannot be scored",
                 ", ".join(dark) + " - no priced line in the chain; "
                 "one Bloomberg pull fixes it")
        else:
            _say(OK, "every tradeable theme has a priced line")
    except Exception as e:                              # noqa: BLE001
        _say(FAIL, "universe vs config", f"{type(e).__name__}: {e}")


def check_anchor_substitutions() -> None:
    """Is any theme being DRAWN on something other than the instrument it
    names? Correct behaviour for an old window, wrong to leave unsaid."""
    try:
        import pandas as pd
        from src.themes import THEME_ETFS, THEME_ETF_FALLBACKS
        from src.config import PRICES_PATH
        if not os.path.exists(PRICES_PATH):
            return
        priced = set(pd.read_parquet(PRICES_PATH,
                                     columns=["symbol"])["symbol"].unique())
        subs = []
        for t, a in THEME_ETFS.items():
            if a in priced:
                continue
            for s in THEME_ETF_FALLBACKS.get(t, []):
                if s in priced:
                    subs.append(f"{t} ({a} -> {s})")
                    break
        if subs:
            _say(WARN, "anchors substituted", "; ".join(subs))
        else:
            _say(OK, "every theme is drawn on its named anchor")
    except Exception as e:                              # noqa: BLE001
        _say(FAIL, "anchor substitutions", f"{type(e).__name__}: {e}")


def check_mapped_tickers_still_exist() -> None:
    """A renamed ticker counts zero forever and raises nothing. Nine were
    dead before the 2026-08-05 audit; this is what stops the tenth."""
    try:
        import csv
        from pathlib import Path
        from src.config import REFERENCE_DIR
        from src.ticker_universe import load_us_ticker_universe
        uni = load_us_ticker_universe(Path(REFERENCE_DIR))
        if not uni:
            _say(WARN, "ticker liveness", "no symbol directory cached")
            return
        with open(os.path.join(ROOT, "config", "theme_tickers.csv"),
                  newline="", encoding="utf-8-sig") as fh:
            mapped = {r["ticker"].strip().upper()
                      for r in csv.DictReader(fh)}
        # OTC ADRs (5 letters ending Y) and dotted class shares can never
        # be in the listed file - they reach their theme by NAME instead,
        # so their absence here is by design, not a defect.
        suspect = sorted(s for s in mapped
                         if s not in uni and "." not in s
                         and not (len(s) == 5 and s.endswith("Y")))
        if suspect:
            _say(WARN, "mapped tickers not in the listed universe",
                 ", ".join(suspect) + " - check for a rename or delisting")
        else:
            _say(OK, f"all {len(mapped)} mapped tickers resolve")
    except Exception as e:                              # noqa: BLE001
        _say(FAIL, "ticker liveness", f"{type(e).__name__}: {e}")


def check_stores() -> None:
    """The files everything downstream reads."""
    from src.config import PROCESSED_DIR
    need = ["daily_ticker_counts.parquet", "daily_theme_counts.parquet",
            "daily_ticker_sentiment.parquet", "daily_theme_sentiment.parquet"]
    missing = [n for n in need
               if not os.path.exists(os.path.join(PROCESSED_DIR, n))]
    if missing:
        _say(FAIL, "core stores", "missing: " + ", ".join(missing))
    else:
        _say(OK, "core stores present")


def check_no_dangling_paths() -> None:
    """Cited files that are not there. See tools/verify_deps.py - and
    note its own warning about partial clones."""
    r = subprocess.run([sys.executable, "tools/verify_deps.py", "--quiet"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        _say(OK, "no dangling file references")
    else:
        _say(WARN, "dangling file references",
             "run tools/verify_deps.py for the list")


def check_frozen_constants() -> None:
    """The values the walk-forward record was earned at. If one of these
    has moved, the stored scorecards describe a different detector than
    the one now running, and every number on the dashboard is quoting a
    record it no longer belongs to."""
    try:
        from src import config as C
        # ground-truth bars updated 2026-08-07 (Class 14b sweep - the
        # walk-forward record was RE-FROZEN under these values the same
        # day, so record and constants moved together, as required)
        expect = {"EUPHORIA_HYPE_MULT": 2.0, "EUPHORIA_MIN_HISTORY": 180,
                  "EUPHORIA_ONSET_HYPE_MIN": 1.10,
                  "EUPHORIA_BOOM_MIN_ETF": 0.20,
                  "EUPHORIA_BOOM_MIN_SINGLE": 0.40,
                  "EUPHORIA_CRASH_MIN_ETF": 0.12,
                  "EUPHORIA_CRASH_MIN_SINGLE": 0.25,
                  "EUPHORIA_COOLDOWN_DAYS": 21,
                  "EUPHORIA_FA_BUDGET_PER_IY": 0.23,
                  "EUPHORIA_BOOM_LOOKBACK_D": 120}
        moved = [f"{k}: {getattr(C, k, '?')} (record: {v})"
                 for k, v in expect.items()
                 if getattr(C, k, None) != v]
        if moved:
            _say(WARN, "FROZEN constants have moved", "; ".join(moved)
                 + " - this is a re-validation event: re-run "
                   "--what phases --research and compare the record")
        else:
            _say(OK, "frozen constants match the stored record")
    except Exception as e:                              # noqa: BLE001
        _say(FAIL, "frozen constants", f"{type(e).__name__}: {e}")


def check_text_free_boundary() -> None:
    """Committed aggregates must never contain crowd text. This is the
    invariant that lets them be committed at all."""
    try:
        import pandas as pd
        from src.config import PROCESSED_DIR
        bad = []
        for n in ("daily_ticker_counts.parquet",
                  "daily_theme_sentiment.parquet"):
            p = os.path.join(PROCESSED_DIR, n)
            if not os.path.exists(p):
                continue
            cols = set(pd.read_parquet(p).columns)
            for leak in ("text", "body", "title", "author", "username",
                         "selftext"):
                if leak in cols:
                    bad.append(f"{n}:{leak}")
        if bad:
            _say(FAIL, "TEXT-FREE BOUNDARY BREACHED", ", ".join(bad))
        else:
            _say(OK, "committed aggregates are text-free")
    except Exception as e:                              # noqa: BLE001
        _say(FAIL, "text-free boundary", f"{type(e).__name__}: {e}")


CHECKS = [check_configs, check_stores, check_universe_matches_config,
          check_anchor_substitutions, check_mapped_tickers_still_exist,
          check_frozen_constants, check_text_free_boundary,
          check_no_dangling_paths]


def main() -> int:
    print("PREFLIGHT - is this project still sound?\n")
    for fn in CHECKS:
        try:
            fn()
        except Exception as e:                          # noqa: BLE001
            _say(FAIL, fn.__name__, f"{type(e).__name__}: {e}")
    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print(f"\n{len(_results)} checks - {len(fails)} failing, "
          f"{len(warns)} warning.")
    if fails:
        print("\nFAILING means something downstream is already wrong. "
              "Fix before running the pipeline.")
    elif warns:
        print("\nNothing is broken. The warnings are things that are "
              "OWED - mostly a price pull or a rebuild.")
    else:
        print("\nAll clear.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
