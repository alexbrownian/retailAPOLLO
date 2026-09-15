"""Dynamic panel: monthly crowd-referral subreddit discovery.

Every knob lives in ``src/config.py`` with its derivation.

When retail migrates between communities, the migration is visible in the
text of the subreddits already tracked before it is visible anywhere
else: people write ``r/<newplace>``. So the panel expands where the crowd
itself points, the same approach as the data-chosen single-name universe.

The rules (all documented in ``config.py``):

qualify
    At least ``PANEL_MIN_REFERRERS`` unique panel authors refer to
    ``r/<name>`` within ``PANEL_REFERRAL_WINDOW`` days.
screen
    The candidate must talk like the panel: one sampled page (100) of its
    newest comments must show a ticker-mention rate of at least
    ``PANEL_SCREEN_FRACTION`` times the tracked panel's own average rate,
    both measured identically in this run. Popularity without tickers
    never passes.
cap
    ``PANEL_ADD_CAP`` auto-adds per review, in the exploration tier; the
    founding subreddits are the frozen core tier. One denominator step
    per month is what the 365-day percentile normalisation absorbs
    gracefully.
audit
    Every review writes ``Reports/panel_review_latest.md`` and every add is
    logged in ``Data/reference/subreddit_panel.json`` (tier, date,
    referral count, measured rates). The manifest is what lets any
    analysis be re-cut excluding young additions.

The panel itself is ``config/forums.csv``, read through
``src.settings.load_forums``; an auto-add appends a row to it.

``Data/processed/daily_ticker_counts_by_subreddit.parquet`` is a
local-only aggregate, extended incrementally from the raw live post files
scanned here. It stays local (gitignored) because the committed contract
bans subreddit columns (``FORBIDDEN_COLS``). It exists so panel-step
artifacts are measurable on this copy.

Usage::

    python Code/ingestion/discover_subreddits.py                 # full review now
    python Code/ingestion/discover_subreddits.py --if-due        # only if the cadence has elapsed
    python Code/ingestion/discover_subreddits.py --report-only   # never adds
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys
import time

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import (PANEL_REVIEW_DAYS, PANEL_REFERRAL_WINDOW,   # noqa: E402
                        PANEL_MIN_REFERRERS, PANEL_SCREEN_FRACTION,
                        PANEL_ADD_CAP, REFERENCE_DIR, PROCESSED_DIR,
                        RAW_DIR)
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402

# The panel itself is config/forums.csv (read through src.settings). The
# manifest below is the detailed audit trail of automatic additions.
MANIFEST = os.path.join(DATA_DIR, "reference",
                        "subreddit_panel.json")
WM_FILE = os.path.join(REFERENCE_DIR, "panel_review_watermark.json")
SCAN_LEDGER = os.path.join(REFERENCE_DIR, "panel_scan_ledger.json")
REFERRALS = os.path.join(REFERENCE_DIR, "subreddit_referrals.parquet")
BY_SUB_COUNTS = os.path.join(PROCESSED_DIR,
                             "daily_ticker_counts_by_subreddit.parquet")
REPORT = os.path.join(REPORTS_DIR, "panel_review_latest.md")
API_COMMENTS = "https://arctic-shift.photon-reddit.com/api/comments/search"

# r/Name referrals in free text. 3-21 chars is Reddit's own name rule;
# the leading boundary stops "for/sale" style false matches.
REFERRAL_RE = re.compile(r"(?:^|[\s(\[])/?r/([A-Za-z0-9_]{3,21})\b")

# structural non-communities that can never be candidates (not a topic
# judgement - these are site mechanics, not forums)
NEVER = {"all", "popular", "askreddit", "announcements"}


# ---------------------------------------------------------------------------
# small IO helpers (same conventions as the fetchers)
# ---------------------------------------------------------------------------
def _load_json(path, default):
    """Read a JSON file, returning ``default`` when absent or unreadable."""
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return default


def _save_json(path, obj):
    """Write ``obj`` as JSON atomically (write beside, then replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def _save_parquet(df, path):
    """Write a frame as parquet atomically (write beside, then replace).

    Both stores here are read-modify-written in place, so an interrupted
    write would leave a truncated file in place of the whole referral or
    count history.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def read_panel() -> list:
    """Return the enabled forums from ``config/forums.csv``, lower-cased.

    The settings cache is reloaded first so an add made earlier in this
    process is seen.
    """
    from src.settings import load_forums, reload
    reload()                         # an add in this process must be seen
    return [s.lower() for s in load_forums()]


def load_manifest() -> dict:
    """Load the panel manifest (the audit trail of additions).

    Self-creates on first run: every subreddit already in the panel
    becomes core tier (the frozen founders).

    Returns:
        Dict mapping subreddit to its manifest record.
    """
    man = _load_json(MANIFEST, {})
    changed = False
    for sub in read_panel():
        if sub not in man:
            man[sub] = {"tier": "core", "added": "founding",
                        "source": "original panel"}
            changed = True
    if changed:
        _save_json(MANIFEST, man)
    return man


# ---------------------------------------------------------------------------
# 1. scan new raw files: referrals + local by-subreddit ticker counts
# ---------------------------------------------------------------------------
def _iter_raw(paths):
    """Yield ``(created_utc, author, subreddit, text, record)`` per raw line.

    Reads raw post/comment ``jsonl.zst`` files (completed files only;
    ``.tmp`` files are in-flight). Unreadable files and malformed lines
    are skipped.
    """
    import zstandard
    for path in paths:
        try:
            with open(path, "rb") as fh:
                reader = zstandard.ZstdDecompressor().stream_reader(fh)
                buf = b""
                while True:
                    chunk = reader.read(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        text = " ".join(str(rec.get(k) or "") for k in
                                        ("title", "selftext", "body"))
                        yield (int(rec.get("created_utc", 0) or 0),
                               str(rec.get("author") or ""),
                               str(rec.get("subreddit") or "").lower(),
                               text, rec)
        except OSError:
            continue


def _ledger_key(path: str) -> str:
    """Return the ledger key for a raw file.

    The key is the path relative to ``RAW_DIR`` with forward slashes on
    every OS. A relative key survives a repo folder rename or move; an
    absolute key would make every file look unseen after the checkout is
    relocated.
    """
    return os.path.relpath(path, RAW_DIR).replace(os.sep, "/")


def _migrate_ledger(ledger: dict) -> dict:
    """Fold legacy entries keyed by absolute path onto the current
    relative-path key scheme, so a prior scan is not repeated after a
    folder rename. Matches everything after the last Data/raw/ segment
    of the old key. Idempotent: a ledger already on relative keys passes
    through unchanged."""
    migrated = {}
    for old_key, size in ledger.items():
        marker = "/Data/raw/"
        norm = old_key.replace("\\", "/")
        idx = norm.rfind(marker)
        new_key = norm[idx + len(marker):] if idx != -1 else old_key
        if new_key not in migrated or size > migrated[new_key]:
            migrated[new_key] = size
    return migrated


def scan_new_raw(verbose=True) -> int:
    """Incrementally scan raw live files the ledger has not seen.

    Extracts ``r/<name>`` referrals (from panel subreddits only) into the
    local referral store, and extends the local by-subreddit ticker counts
    from post files. A file is rescanned when its size changes.

    Args:
        verbose: Print a one-line summary.

    Returns:
        The number of newly scanned files.
    """
    ledger = _migrate_ledger(_load_json(SCAN_LEDGER, {}))
    candidates_files = sorted(
        glob.glob(os.path.join(RAW_DIR, "RedditLive", "*.jsonl.zst"))
        + glob.glob(os.path.join(RAW_DIR, "RedditComments", "*.jsonl.zst")))
    new_files = [p for p in candidates_files
                 if ledger.get(_ledger_key(p)) != os.path.getsize(p)]
    if not new_files:
        _save_json(SCAN_LEDGER, ledger)   # persist the migration itself
        return 0

    panel = set(read_panel())
    ref_rows = []
    post_rows = []
    for created, author, sub, text, rec in _iter_raw(new_files):
        if not created or not author or author in ("[deleted]",):
            continue
        date = datetime.datetime.fromtimestamp(
            created, datetime.timezone.utc).date().isoformat()
        # referrals: only from PANEL subs (the crowd we already trust)
        if sub in panel and text:
            for m in REFERRAL_RE.finditer(text):
                target = m.group(1).lower()
                if target not in panel and target not in NEVER:
                    ref_rows.append({"date": date, "candidate": target,
                                     "author": author})
        # local by-subreddit ticker counts: POST records only (they carry
        # title/selftext; comments would double-count threads)
        if sub and ("title" in rec or "selftext" in rec):
            post_rows.append({"date": date, "subreddit": sub,
                              "text": text})

    if ref_rows:
        new_refs = pd.DataFrame(ref_rows).drop_duplicates()
        if os.path.exists(REFERRALS):
            new_refs = (pd.concat([pd.read_parquet(REFERRALS), new_refs])
                        .drop_duplicates())
        _save_parquet(new_refs, REFERRALS)

    if post_rows:
        _extend_by_sub_counts(pd.DataFrame(post_rows))

    for p in new_files:
        ledger[_ledger_key(p)] = os.path.getsize(p)
    _save_json(SCAN_LEDGER, ledger)
    if verbose:
        print(f"  panel scan: {len(new_files)} new raw files, "
              f"{len(ref_rows)} referral events")
    return len(new_files)


def _extend_by_sub_counts(posts: pd.DataFrame):
    """Extend the local-only daily ticker counts per subreddit.

    Rows for a ``(date, subreddit)`` pair present in ``posts`` replace the
    stored ones, so re-scanning a grown file replaces those days.

    Args:
        posts: Frame with ``date``, ``subreddit`` and ``text`` columns.
    """
    from src.abstracted_data import load_universe
    from src.extract_tickers import extract_tickers_from_text
    universe = load_universe()
    rows = []
    for (date, sub), g in posts.groupby(["date", "subreddit"]):
        counts = {}
        for text in g["text"]:
            for t in set(extract_tickers_from_text(text, universe,
                                                   cashtags_only=False)):
                counts[t] = counts.get(t, 0) + 1
        rows += [{"date": date, "subreddit": sub, "ticker": t,
                  "mention_count": n} for t, n in counts.items()]
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(BY_SUB_COUNTS):
        old = pd.read_parquet(BY_SUB_COUNTS)
        keys = set(zip(new["date"], new["subreddit"]))
        old = old[~old.apply(lambda r: (r["date"], r["subreddit"]) in keys,
                             axis=1)]
        new = pd.concat([old, new], ignore_index=True)
    _save_parquet(new, BY_SUB_COUNTS)


# ---------------------------------------------------------------------------
# 2. the finance screen (one polite API page per measured community)
# ---------------------------------------------------------------------------
def ticker_rate(sub: str, universe, sampler=None) -> float | None:
    """Measure how often a community's newest comments mention a ticker.

    Args:
        sub: Subreddit name.
        universe: The ticker universe from ``load_universe``.
        sampler: Callable returning a list of comment bodies for a
            subreddit; defaults to one Arctic Shift page of 100.

    Returns:
        Fraction of sampled comments that mention at least one
        extractable ticker, or ``None`` when the sample is unavailable
        (offline or API refusal). An unmeasurable candidate is never
        auto-added.
    """
    from src.extract_tickers import extract_tickers_from_text
    if sampler is None:
        import requests

        def sampler(s):
            # ANY failure -> None ("unmeasurable"), and unmeasurable is
            # never auto-added; a review run offline degrades to
            # report-only rather than crashing the pipeline
            try:
                r = requests.get(API_COMMENTS,
                                 params={"subreddit": s, "limit": 100},
                                 timeout=(10, 60))
                if r.status_code != 200:
                    return None
                return [str(rec.get("body") or "")
                        for rec in r.json().get("data", [])]
            except requests.RequestException:
                return None
    texts = sampler(sub)
    if not texts:
        return None
    hits = sum(1 for t in texts
               if extract_tickers_from_text(t, universe,
                                            cashtags_only=False))
    return hits / len(texts)


# ---------------------------------------------------------------------------
# 3. the review
# ---------------------------------------------------------------------------
def qualify(referrals: pd.DataFrame, panel: list,
            asof: datetime.date) -> pd.DataFrame:
    """Rank candidates by unique referring authors within the window.

    A pure function of the referral store.

    Args:
        referrals: Frame with ``date``, ``candidate`` and ``author``.
        panel: Current panel (lower-cased); its members are excluded.
        asof: End of the ``PANEL_REFERRAL_WINDOW`` window.

    Returns:
        Frame with ``candidate`` and ``referrers`` columns, sorted by
        ``referrers`` descending.
    """
    if referrals.empty:
        return pd.DataFrame(columns=["candidate", "referrers"])
    lo = (asof - datetime.timedelta(days=PANEL_REFERRAL_WINDOW)).isoformat()
    win = referrals[(referrals["date"] >= lo)
                    & ~referrals["candidate"].isin(panel)
                    & ~referrals["candidate"].isin(NEVER)]
    ranked = (win.groupby("candidate")["author"].nunique()
              .rename("referrers").sort_values(ascending=False)
              .reset_index())
    return ranked


def run_review(report_only: bool = False, sampler=None,
               verbose: bool = True) -> dict:
    """Run the monthly review: scan, qualify, screen, add, report.

    Args:
        report_only: Rank, screen and write the report but never add.
        sampler: Passed to ``ticker_rate``; tests inject one to avoid
            network calls.
        verbose: Print progress lines.

    Returns:
        Summary dict with ``date``, ``panel_size``, ``candidates_seen``,
        ``qualified``, ``added`` and ``screen``; the report is also
        written to ``Reports/panel_review_latest.md``.
    """
    from src.abstracted_data import load_universe

    scan_new_raw(verbose=verbose)
    panel = read_panel()
    man = load_manifest()
    refs = (pd.read_parquet(REFERRALS) if os.path.exists(REFERRALS)
            else pd.DataFrame(columns=["date", "candidate", "author"]))
    today = datetime.date.today()
    ranked = qualify(refs, panel, today)
    qualified = ranked[ranked["referrers"] >= PANEL_MIN_REFERRERS]

    summary = {"date": today.isoformat(), "panel_size": len(panel),
               "candidates_seen": int(len(ranked)),
               "qualified": qualified.to_dict(orient="records"),
               "added": [], "screen": {}}

    if len(qualified):
        universe = load_universe()
        # the panel's own average rate, measured identically (5 seeded
        # panel subs keep this to a handful of polite calls)
        import random
        rng = random.Random(42)
        probe = rng.sample(panel, min(5, len(panel)))
        panel_rates = []
        for s in probe:
            r = ticker_rate(s, universe, sampler)
            if r is not None:
                panel_rates.append(r)
            time.sleep(0 if sampler else 1)
        panel_avg = (sum(panel_rates) / len(panel_rates)
                     if panel_rates else None)
        summary["screen"]["panel_avg_rate"] = panel_avg

        added = 0
        for row in qualified.itertuples():
            if added >= PANEL_ADD_CAP or report_only:
                break
            rate = ticker_rate(row.candidate, universe, sampler)
            summary["screen"][row.candidate] = rate
            time.sleep(0 if sampler else 1)
            if rate is None or panel_avg is None:
                continue          # unmeasurable is never auto-added
            if rate >= PANEL_SCREEN_FRACTION * panel_avg:
                _add_to_panel(row.candidate, int(row.referrers), rate,
                              panel_avg, man)
                summary["added"].append(row.candidate)
                added += 1

    _write_report(summary, ranked)
    _save_json(WM_FILE, {"last_review": today.isoformat()})
    if verbose:
        print(f"  panel review {today}: {len(ranked)} candidates, "
              f"{len(qualified)} qualified, added {summary['added'] or 'none'}"
              f"{' (report-only)' if report_only else ''}")
    return summary


def _add_to_panel(sub: str, referrers: int, rate: float,
                  panel_avg: float, man: dict):
    """Append an exploration-tier row to ``forums.csv`` and log it in the manifest."""
    from src.settings import FORUMS_FILE
    with open(FORUMS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            sub, "reddit", "exploration", "yes",
            datetime.date.today().isoformat(),
            f"auto-added: {referrers} referrers/28d, ticker rate "
            f"{rate:.0%} vs panel {panel_avg:.0%}"])
    man[sub] = {"tier": "exploration",
                "added": datetime.date.today().isoformat(),
                "source": "crowd-referral auto-add",
                "referrers_28d": referrers,
                "ticker_rate": round(rate, 3),
                "panel_avg_rate": round(panel_avg, 3)}
    _save_json(MANIFEST, man)
    print(f"  PANEL ADD: r/{sub} (exploration tier - {referrers} unique "
          f"referrers/28d, ticker rate {rate:.0%} vs panel avg "
          f"{panel_avg:.0%}). Fetchers pick it up next run; the manifest "
          "logs the denominator step.")


def _write_report(summary: dict, ranked: pd.DataFrame):
    """Write the Markdown review report for the top-ranked candidates."""
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    lines = [
        f"# Panel review — {summary['date']}",
        "",
        f"Panel size: {summary['panel_size']} | candidates seen: "
        f"{summary['candidates_seen']} | qualification bar: "
        f"≥{PANEL_MIN_REFERRERS} unique referrers/"
        f"{PANEL_REFERRAL_WINDOW}d | added: "
        f"{', '.join('r/' + s for s in summary['added']) or 'none'}",
        "",
        "| candidate | unique referrers (28d) | screen rate |",
        "|---|---|---|",
    ]
    screen = summary.get("screen", {})
    for row in ranked.head(15).itertuples():
        rate = screen.get(row.candidate)
        lines.append(f"| r/{row.candidate} | {row.referrers} | "
                     f"{'' if rate is None else f'{rate:.0%}'} |")
    lines += ["",
              f"_Panel average screen rate: "
              f"{screen.get('panel_avg_rate')}_ — a candidate auto-adds "
              f"only at ≥ {PANEL_SCREEN_FRACTION:.0%} of it (and only "
              f"{PANEL_ADD_CAP}/review). Manifest: "
              "`Data/reference/subreddit_panel.json`."]
    # Staged and renamed, like the panel manifest above: the previous
    # review is the only record until this one finishes, and a write cut
    # short would replace it with half a table that still reads as a
    # complete report.
    with open(REPORT + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(REPORT + ".tmp", REPORT)


def main() -> int:
    """Run the review, or only the incremental scan when it is not due."""
    p = argparse.ArgumentParser(description="Dynamic subreddit panel: "
                                            "monthly crowd-referral review")
    p.add_argument("--if-due", action="store_true",
                   help="exit quietly unless >= PANEL_REVIEW_DAYS have "
                        "passed since the last review (the update_data "
                        "hook uses this)")
    p.add_argument("--report-only", action="store_true",
                   help="rank + screen + write the report, never add")
    args = p.parse_args()

    if args.if_due:
        wm = _load_json(WM_FILE, {})
        last = wm.get("last_review")
        if last:
            age = (datetime.date.today()
                   - datetime.date.fromisoformat(last)).days
            if age < PANEL_REVIEW_DAYS:
                print(f"  panel review not due ({age}d since last, "
                      f"cadence {PANEL_REVIEW_DAYS}d)")
                # still scan incrementally so referral history has no gaps
                scan_new_raw()
                return 0
    run_review(report_only=args.report_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
