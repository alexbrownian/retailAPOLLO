"""Publishes the display bundle the hosted dashboard reads.

The hosted dashboard (Streamlit Community Cloud) is a VIEW of this
project, not a copy of it: it never fetches, never recomputes, and never
reaches a Bloomberg Terminal. It renders whatever this script last
published to `DASHBOARD_DATA/`, which is committed to the repository the
same way `ABSTRACTED_DATA/` is.

Run this after a pipeline run, on either machine, then commit and push:

    python update_data.py                 # the ordinary refresh
    python tools/publish_dashboard.py     # stage the display bundle
    git add ABSTRACTED_DATA DASHBOARD_DATA
    git commit -m "publish dashboard"
    git push

Streamlit Cloud redeploys on the push, so the public dashboard tracks
whatever was last published.

WHY A BUNDLE AND NOT A CLOUD RECOMPUTE
    `data/processed/` is gitignored and absent on a fresh clone. The
    alternative - having the hosted app rebuild it - would run
    `analytics.run_analytics` on the host, which (a) needs
    `prices.parquet` anyway, (b) forks a process pool sized to the
    stage count, and (c) auto-opens a full walk-forward research pass
    whenever the frozen record is missing, which is exactly the state a
    fresh clone is in. Publishing the finished frames instead keeps the
    host doing one thing: drawing them.

TEXT-FREE GUARD
    Every published parquet is checked against `FORBIDDEN_COLS` before
    it is written, the same rule that protects `ABSTRACTED_DATA/`. A
    file carrying post text, authors, ids, or subreddit names is refused
    and the run fails. This is why the influence board is opt-in
    (`--with-influence`): its frames are keyed by author handle, so
    publishing them puts Reddit usernames on a public page.

USAGE
    python tools/publish_dashboard.py                # stage the bundle
    python tools/publish_dashboard.py --dry-run      # report, write nothing
    python tools/publish_dashboard.py --with-influence
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.config import (FORBIDDEN_COLS, PROCESSED_DIR,          # noqa: E402
                        PRICES_PATH, REFERENCE_DIR, SNAPSHOT_DIR)

BUNDLE_DIR = os.path.join(PROJECT_ROOT, "DASHBOARD_DATA")
LOCAL_CONTROLS_MARKER = os.path.join(PROJECT_ROOT, ".local_controls")

# Derived frames the dashboard draws. Absent files are skipped: a machine
# that has not run a given stage publishes the rest rather than failing.
PROCESSED_FILES = [
    "daily_theme_conviction.parquet",
    "trade_signals.parquet",
    "trade_signals_tickers.parquet",
    "euphoria_levels.parquet",
    "euphoria_onset.parquet",
    "euphoria_desk.parquet",
    "euphoria_desk_components.parquet",
    "episodes.parquet",
    "phase_day_frame.parquet",
    "daily_agentic_counts.parquet",
    "ai_poll.parquet",
    "euphoria_report.json",
    "euphoria_desk_report.json",
    "euphoria_onset_report.json",
    "desk_model_insight.json",
    "readiness_alerts.json",
    "ai_pulse.json",
]

# Symbol directories used only to label tickers with company names. The
# dashboard degrades to bare tickers without them.
REFERENCE_FILES = ["nasdaqlisted.txt", "otherlisted.txt"]

DOCS_RESEARCH_FILES = ["nb05_influence.json"]

INFLUENCE_FILES = ["author_scores.parquet", "calls.parquet",
                   "reply_edges.parquet"]

# Never publish: local-only aggregates that carry subreddit identity.
NEVER_PUBLISH = {"daily_ticker_counts_by_subreddit.parquet"}

# The publish guard, derived from FORBIDDEN_COLS.
#
# FORBIDDEN_COLS protects ABSTRACTED_DATA, whose files are aggregations of
# raw post ROWS - there, `score` and `num_comments` can only mean a post's
# Reddit score and comment count, so refusing them is right. This bundle
# carries DERIVED frames instead, where the same two names carry unrelated
# meanings: `trade_signals.score` is the 5-check signal score (4 or 5, the
# SIG_MIN_SCORE floor). Refusing on the name alone would block the signal
# tables over a spelling coincidence.
#
# The remaining names cannot collide: they identify a post, its author, or
# its community, and no derived frame has any reason to carry one. Those are
# refused unconditionally, which is what the guard is actually for. The
# ABSTRACTED_DATA rule in src/config.py is unchanged.
AMBIGUOUS_COLS = {"score", "num_comments"}
IDENTITY_COLS = FORBIDDEN_COLS - AMBIGUOUS_COLS


def _forbidden_columns(path: str) -> list:
    """Returns any identity columns present in a parquet file's schema.

    Reads the schema only, never the row groups, so a large frame costs
    the same as a small one.
    """
    import pyarrow.parquet as pq
    try:
        names = set(pq.ParquetFile(path).schema.names)
    except Exception as exc:                              # noqa: BLE001
        # FAIL CLOSED. Returning [] here meant an unreadable footer - a
        # half-written snapshot from a concurrent run, a truncated file -
        # was copied into the publicly served bundle with no guard
        # applied at all. A file whose schema cannot be read cannot be
        # cleared, so it is refused.
        raise SystemExit(
            f"REFUSED: cannot read the schema of "
            f"{os.path.relpath(path, PROJECT_ROOT)} ({exc}). A file that "
            "cannot be checked is never published.")
    # Case-insensitive, matching verify_abstracted in update_data.py: a
    # column arriving as "Author" or "Title" is the same disclosure as
    # the lower-case spelling.
    lowered = {n.lower() for n in names}
    return sorted(c for c in IDENTITY_COLS if c.lower() in lowered)


def _copy(src: str, dst: str, dry_run: bool) -> int:
    """Copies one file, returning its size in bytes (0 when absent)."""
    if not os.path.exists(src):
        return 0
    if os.path.basename(src) in NEVER_PUBLISH:
        raise SystemExit(f"REFUSED: {os.path.basename(src)} is never "
                         "publishable (carries subreddit identity).")
    if src.endswith(".parquet"):
        bad = _forbidden_columns(src)
        if bad:
            raise SystemExit(
                f"REFUSED: {os.path.relpath(src, PROJECT_ROOT)} carries "
                f"forbidden column(s) {bad}. Publishing it would put raw "
                "post text or identities on a public page.")
    if not dry_run:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return os.path.getsize(src)


def publish(with_influence: bool = False, dry_run: bool = False,
            log=print, per_file: bool = True) -> int:
    """Stages the display bundle. Returns the number of files published.

    Args:
      with_influence: also publish the author-keyed influence frames.
      dry_run: report only; write nothing.
      log: line sink. update_data.py passes its own so the bundle step
        lands in the run log with everything else.
      per_file: list every file. False gives the one-line summary the
        pipeline wants.
    """
    staged, total_bytes, missing = 0, 0, []

    def stage(src, rel_dst):
        nonlocal staged, total_bytes
        size = _copy(src, os.path.join(BUNDLE_DIR, rel_dst), dry_run)
        if size:
            staged += 1
            total_bytes += size
            if per_file:
                log(f"  {rel_dst:<48} {size / 1024:>9,.0f} KB")
        else:
            missing.append(rel_dst)

    if per_file:
        log(f"publish -> {os.path.relpath(BUNDLE_DIR, PROJECT_ROOT)}"
            + ("  (DRY RUN)" if dry_run else ""))

    for name in PROCESSED_FILES:
        stage(os.path.join(PROCESSED_DIR, name), name)

    stage(PRICES_PATH, "prices.parquet")

    for name in REFERENCE_FILES:
        stage(os.path.join(REFERENCE_DIR, name), name)

    for name in DOCS_RESEARCH_FILES:
        stage(os.path.join(PROJECT_ROOT, "docs", "research", name), name)

    # Signal snapshots drive the "what did it say back then" history.
    for path in sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.parquet"))):
        stage(path, os.path.join("signal_snapshots",
                                 os.path.basename(path)))

    if with_influence:
        log("  -- influence board (author handles; opt-in) --")
        for name in INFLUENCE_FILES:
            stage(os.path.join(REFERENCE_DIR, "influence", name),
                  os.path.join("influence", name))

    log(f"{'' if per_file else 'dashboard bundle: '}{staged} file(s), "
        f"{total_bytes / 1024 / 1024:,.1f} MB"
        + (f", {len(missing)} not on this machine" if missing else ""))
    if missing and per_file:
        for name in missing:
            log(f"  - {name}")
    if not with_influence and per_file:
        log("influence board not published (use --with-influence to "
            "include it; it carries Reddit author handles).")

    if not dry_run:
        # The marker enables the pipeline-run controls in the sidebar. It
        # is gitignored, so it exists on this machine and never on the
        # host - which is the whole point: the hosted app must not offer
        # buttons that fetch, spend API credit, or call a Terminal.
        # Written on every publish path: a machine that publishes is by
        # definition a workstation.
        if not os.path.exists(LOCAL_CONTROLS_MARKER):
            with open(LOCAL_CONTROLS_MARKER, "w", encoding="utf-8") as f:
                f.write("Local machine marker; see tools/publish_dashboard"
                        ".py. Gitignored on purpose.\n")
            log("created .local_controls (enables the sidebar pipeline "
                "controls on this machine only)")
    if not dry_run and per_file:
        log("\nnext:")
        log("  git add ABSTRACTED_DATA DASHBOARD_DATA")
        log('  git commit -m "publish dashboard"')
        log("  git push")
    return staged


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage the display bundle the hosted dashboard reads.")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be published, write nothing")
    p.add_argument("--with-influence", action="store_true",
                   help="also publish the influence board. Its frames are "
                        "keyed by Reddit author handle; only use this if a "
                        "public page showing those handles is intended.")
    args = p.parse_args()
    publish(with_influence=args.with_influence, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
