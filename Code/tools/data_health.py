"""Read-only health check of the data stores.

Fetcher warnings such as ``3 subreddit(s) gave up`` or ``NO DATA for 8 of
267 symbols`` print once into a long log and are gone. This answers "is
the data healthy?" without reading that log::

    python Code/tools/data_health.py
    python Code/tools/data_health.py --json      # machine-readable only

It writes nothing except ``Data/reference/ingestion_status.json`` and
opens every other file read-only, so it is safe to run at any time,
including while the pipeline is running. The exit code is ``1`` when any
check reports ``BAD``.

Checks:

freshness
    How stale each aggregate and the price store are.
corporate actions
    Single-session price moves consistent with an unadjusted split.
coverage
    Per-source span, and days missing inside that span.
dedup
    Size of the durable seen-id set, and whether the legacy JSON ledger
    holds ids the parquet does not.
ledgers
    Every ``Data/reference`` JSON: present, parseable, size.
backups
    Whether there are recent snapshots of ``Data/reference``.
raw
    How much raw is on disk, and dead ``.tmp`` files.
integrity
    Aggregate merges are additive, so a duplicate fold is permanent and
    invisible. This looks for the signature: a month whose mention total
    is a near multiple of its neighbours.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import datetime

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
from src.config import DATA_DIR  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ABS = os.path.join(PROJECT_ROOT, "Data/abstracted")
REF = os.path.join(DATA_DIR, "reference")
RAW = os.path.join(DATA_DIR, "raw")
PRICES = os.path.join(DATA_DIR, "prices", "prices.parquet")
STATUS = os.path.join(REF, "ingestion_status.json")

OK, WARN, BAD = "ok", "warn", "BAD"


def _age_days(path):
    """Return whole days since ``path`` was modified, or ``None`` if absent."""
    if not os.path.exists(path):
        return None
    return (datetime.datetime.now()
            - datetime.datetime.fromtimestamp(os.path.getmtime(path))).days


def check_freshness(out, say):
    """Report how old the newest aggregate day and the newest close are.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("FRESHNESS")
    tc = os.path.join(ABS, "daily_ticker_counts.parquet")
    if os.path.exists(tc):
        d = pd.read_parquet(tc, columns=["date"])
        newest = pd.to_datetime(d["date"]).max()
        lag = (pd.Timestamp.today().normalize() - newest.normalize()).days
        st = OK if lag <= 4 else (WARN if lag <= 10 else BAD)
        say(f"  aggregates newest day : {newest.date()}  ({lag}d old)", st)
        out["aggregates_newest"] = str(newest.date())
        out["aggregates_lag_days"] = int(lag)
    if os.path.exists(PRICES):
        p = pd.read_parquet(PRICES, columns=["date"])
        newest = pd.to_datetime(p["date"]).max()
        lag = (pd.Timestamp.today().normalize() - newest.normalize()).days
        # a close is only expected on trading days; 5d covers a long weekend
        st = OK if lag <= 5 else (WARN if lag <= 12 else BAD)
        say(f"  prices newest close   : {newest.date()}  ({lag}d old)", st)
        out["prices_newest"] = str(newest.date())
        out["prices_lag_days"] = int(lag)
        if st != OK:
            say("     -> signals cannot be judged past the last close",
                WARN)


# Single-session close-to-close moves beyond these bars are reported for
# a corporate-action check. Most are stock splits or reverse splits the
# price source did not back-adjust (a 30x jump in one session on a
# sub-$1 name); left in place they create spurious boom-bust episodes in
# the ground truth. A few are real - the largest retail squeezes have
# printed +400% in a session - so the check reports and never edits.
SPLIT_JUMP_UP = 4.0       # +400% in one session
SPLIT_JUMP_DOWN = -0.70   # -70% in one session


def check_corporate_actions(out, say):
    """Flag single-session price jumps consistent with an unadjusted split.

    Reports every ``(symbol, date)`` whose close moved by more than
    ``SPLIT_JUMP_UP`` or below ``SPLIT_JUMP_DOWN`` against the previous
    close. Read-only: the fix is to re-pull the symbol with adjusted
    prices.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("EXTREME SINGLE-SESSION MOVES (possible unadjusted splits)")
    if not os.path.exists(PRICES):
        say("  no prices.parquet", WARN)
        return
    px = pd.read_parquet(PRICES, columns=["date", "symbol", "px_last"])
    px = px.dropna(subset=["px_last"])
    px["date"] = pd.to_datetime(px["date"])
    hits = []
    for sym, g in px.sort_values("date").groupby("symbol"):
        r = g["px_last"].pct_change()
        bad = g[(r > SPLIT_JUMP_UP) | (r < SPLIT_JUMP_DOWN)]
        for _, row in bad.iterrows():
            prev = g["px_last"].shift(1).loc[row.name]
            hits.append({"symbol": sym, "date": str(row["date"].date()),
                         "prev_close": float(prev),
                         "close": float(row["px_last"]),
                         "move": float(row["px_last"] / prev - 1)})
    out["split_jumps"] = hits
    if not hits:
        say(f"  no single-session move beyond +{SPLIT_JUMP_UP:.0%} / "
            f"{SPLIT_JUMP_DOWN:.0%} across {px['symbol'].nunique()} symbols",
            OK)
        return
    for h in hits:
        say(f"  {h['symbol']:<8} {h['date']}  {h['prev_close']:.4f} -> "
            f"{h['close']:.4f}  ({h['move']:+.0%})", WARN)
    say(f"  {len(hits)} move(s) to verify: a genuine squeeze needs no "
        "action; a split needs the symbol re-pulled with adjusted prices",
        WARN)


def check_coverage(out, say):
    """Report each source's date span and how densely it is populated.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("COVERAGE BY SOURCE")
    f = os.path.join(ABS, "daily_ticker_counts_by_source.parquet")
    if not os.path.exists(f):
        say("  (no by-source file)", WARN)
        return
    d = pd.read_parquet(f)
    d["date"] = pd.to_datetime(d["date"])
    out["sources"] = {}
    for src, g in d.groupby("source"):
        lo, hi = g["date"].min(), g["date"].max()
        span = (hi - lo).days + 1
        days = g["date"].nunique()
        dense = 100.0 * days / max(span, 1)
        # WARN at worst: a young or intermittent source is a known
        # state, not a fault. reddit is the spine (100% dense); X and
        # StockTwits are recent additions and being thin is expected.
        # BAD is reserved for things that are actually broken.
        st = OK if dense >= 90 else WARN
        say(f"  {src:<12} {lo.date()} -> {hi.date()}  "
            f"{days:>5} days of {span:>5} ({dense:.0f}% dense)", st)
        if dense < 50:
            say(f"     -> thin: treat {src} history as indicative. "
                f"robust_share already stratifies by source so this "
                f"cannot dilute the shares.", WARN)
        out["sources"][src] = dict(first=str(lo.date()), last=str(hi.date()),
                                   days=int(days), span=int(span),
                                   dense_pct=round(dense, 1))


def check_dedup(out, say):
    """Report the size and readability of the seen-id set and ``LIVE_START``.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("DEDUP SET (the only thing preventing permanent double counting)")
    seen_p = os.path.join(REF, "abstracted_seen_ids.parquet")
    meta_p = os.path.join(REF, "abstracted_live_meta.json")
    n_par = n_json = 0
    if os.path.exists(seen_p):
        try:
            n_par = len(pd.read_parquet(seen_p))
            say(f"  durable parquet       : {n_par:,} ids", OK)
        except Exception as e:                                # noqa: BLE001
            say(f"  durable parquet       : UNREADABLE ({e})", BAD)
    else:
        say("  durable parquet       : MISSING - run the pipeline once to "
            "migrate", WARN)
    if os.path.exists(meta_p):
        try:
            meta = json.load(open(meta_p, encoding="utf-8"))
            n_json = len(meta.get("seen_ids", []))
            say(f"  legacy json ledger    : {n_json:,} ids (fallback only)")
            say(f"  LIVE_START            : {meta.get('live_start')}")
            out["live_start"] = meta.get("live_start")
        except Exception as e:                                # noqa: BLE001
            say(f"  legacy json ledger    : UNREADABLE ({e})", BAD)
    if n_par and n_json > n_par:
        say(f"  -> the json holds MORE ids than the parquet; the next run "
            f"merges them", WARN)
    out["seen_ids_parquet"] = n_par
    out["seen_ids_legacy_json"] = n_json


def check_ledgers(out, say):
    """Parse every ``Data/reference`` JSON file and report its size.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("LEDGERS")
    bad = []
    for f in sorted(glob.glob(os.path.join(REF, "*.json"))):
        name = os.path.basename(f)
        try:
            json.load(open(f, encoding="utf-8"))
            kb = os.path.getsize(f) / 1024
            say(f"  {name:<38} {kb:>8,.0f} KB")
        except Exception as e:                                # noqa: BLE001
            say(f"  {name:<38} UNPARSEABLE ({e})", BAD)
            bad.append(name)
    out["unparseable_ledgers"] = bad


def check_backups(out, say):
    """Report how many reference snapshots exist and how old the newest is.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("BACKUPS of Data/reference")
    bdir = os.path.join(REF, "_backups")
    if not os.path.isdir(bdir):
        say("  none yet - the next successful fold writes one", WARN)
        out["backups"] = 0
        return
    snaps = sorted(d for d in os.listdir(bdir)
                   if os.path.isdir(os.path.join(bdir, d)))
    age = _age_days(os.path.join(bdir, snaps[-1])) if snaps else None
    st = OK if snaps and (age is not None and age <= 7) else WARN
    say(f"  {len(snaps)} snapshot(s), newest {snaps[-1] if snaps else '-'}"
        f"{f' ({age}d old)' if age is not None else ''}", st)
    out["backups"] = len(snaps)


def check_raw(out, say):
    """Report raw bytes on disk and count dead ``.tmp`` files.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("RAW STORE")
    total = tmp_bytes = tmp_n = 0
    for root, _dirs, files in os.walk(RAW):
        if "_to_delete" in root:
            continue
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            total += sz
            if fn.endswith(".tmp"):
                tmp_bytes += sz
                tmp_n += 1
    say(f"  raw on disk           : {total / 2**30:.2f} GiB")
    st = OK if tmp_n == 0 else WARN
    say(f"  dead .tmp files       : {tmp_n} ({tmp_bytes / 2**20:,.0f} MiB)",
        st)
    if tmp_n:
        say("     -> killed runs; safe to delete, the glob ignores them",
            WARN)
    out["raw_gib"] = round(total / 2**30, 2)
    out["dead_tmp_files"] = tmp_n


def check_double_count(out, say):
    """The signature of a duplicate fold, checked at MONTH granularity.

    Folds operate per (file, month), so the unit of duplication is a
    month. Each month is compared with the median of the six months on
    either side, excluding itself: a self-excluding baseline is
    required because a doubled month doubles its own median, and a
    per-day check against low-volume history flags ordinary variance.

    Args:
        out: Dict the JSON status is assembled in (mutated).
        say: Line reporter ``say(msg, status=None)``.
    """
    say("DOUBLE-COUNT SCAN (additive merges make a duplicate permanent)")
    f = os.path.join(ABS, "daily_ticker_counts.parquet")
    if not os.path.exists(f):
        say("  (no aggregates)", WARN)
        return
    d = pd.read_parquet(f)
    d["date"] = pd.to_datetime(d["date"])
    m = (d.groupby(d["date"].dt.to_period("M"))["mention_count"].sum()
         .sort_index())
    if len(m) < 8:
        say("  (too little history to compare months)", WARN)
        return

    vals, idx = m.values.astype(float), list(m.index)
    flagged = []
    for i, per in enumerate(idx):
        lo, hi = max(0, i - 6), min(len(vals), i + 7)
        neigh = [vals[j] for j in range(lo, hi) if j != i]
        if len(neigh) < 4:
            continue
        base = float(pd.Series(neigh).median())
        if base <= 0:
            continue
        r = vals[i] / base
        # 1.7x is comfortably above normal month-to-month variation and
        # below the 2.0x a clean duplicate produces
        if r >= 1.70:
            flagged.append((per, vals[i], base, r))

    if flagged:
        # WARN, never BAD. The scan flags mania months and the month a
        # new source came online just as readily as an injected
        # duplicate. It cannot tell a mania or a new source from a
        # double fold - only a human with the ledger can - so it hands
        # over candidates rather than pretending to a verdict.
        say(f"  {len(flagged)} month(s) at >=1.7x the median of their "
            f"neighbours - CANDIDATES, not findings:", WARN)
        for per, v, base, r in flagged[:12]:
            say(f"     {per}  {v:>9,.0f} vs neighbour median "
                f"{base:>9,.0f}   ({r:.2f}x)", WARN)
        say("     Three innocent explanations, check them first:", WARN)
        say("       a real mania  |  a new SOURCE switched on that month  "
            "|  a backfill", WARN)
        say("     Then look for two ledger entries covering the same "
            "month in", WARN)
        say("       Data/reference/historical_fold_ledger.json - that is "
            "the guilty signature.", WARN)
    else:
        say("  no month sits at >=1.7x its neighbours", OK)
    out["suspect_double_count_months"] = [
        dict(month=str(p), total=int(v), neighbour_median=int(b),
             ratio=round(r, 2)) for p, v, b, r in flagged]


def main() -> int:
    """Run every check, write the status JSON and print the report.

    Returns:
        ``0`` unless any check reported ``BAD``, then ``1``.
    """
    p = argparse.ArgumentParser(description="Read-only data health check.")
    p.add_argument("--json", action="store_true",
                   help="print the JSON only (for scripting)")
    args = p.parse_args()

    out, lines, worst = {}, [], OK

    def say(msg, status=None):
        nonlocal worst
        if status in (WARN, BAD) and (worst == OK or status == BAD):
            worst = status
        tag = {OK: "  [ok] ", WARN: "  [!]  ", BAD: "  [XX] "}.get(status, "")
        lines.append(f"{tag}{msg}" if tag else msg)

    for fn in (check_freshness, check_corporate_actions, check_coverage,
               check_dedup, check_ledgers, check_backups, check_raw,
               check_double_count):
        try:
            fn(out, say)
        except Exception as exc:                              # noqa: BLE001
            say(f"  {fn.__name__} failed: {type(exc).__name__}: {exc}", BAD)
        say("")

    out["overall"] = worst
    out["checked_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
    try:
        os.makedirs(REF, exist_ok=True)
        tmp = STATUS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, STATUS)
    except Exception:
        pass

    if args.json:
        print(json.dumps(out, indent=1))
        return 0 if worst != BAD else 1
    print("=" * 62)
    print("DATA HEALTH")
    print("=" * 62)
    print("\n".join(lines))
    print("=" * 62)
    print(f"OVERALL: {worst.upper()}")
    print(f"written: {os.path.relpath(STATUS, PROJECT_ROOT)}")
    print("=" * 62)
    return 0 if worst != BAD else 1


if __name__ == "__main__":
    raise SystemExit(main())
