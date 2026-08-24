"""
data_health.py - is the data actually healthy?  (READ-ONLY)
==========================================================
recorded decision: `!! 3 subreddit(s) gave up` and `NO DATA for 8
of 267 symbols` print once, into a 200-line log, and are gone. This
answers "is my data healthy?" without reading that log.

    python tools/data_health.py
    python tools/data_health.py --json      # machine-readable only

IT WRITES NOTHING except data/reference/ingestion_status.json, and it
opens every other file read-only. Safe to run at any time, including
while the pipeline is running.

WHAT IT CHECKS
    freshness   how stale each aggregate and the price store are
    coverage    per-source span, and days missing inside that span
    dedup       size of the durable seen-id set, and whether the legacy
                JSON ledger holds ids the parquet does not
    ledgers     every data/reference JSON: present, parseable, size
    backups     are there recent snapshots of data/reference
    raw         how much raw is on disk, and dead .tmp files
    integrity   the one that matters: aggregate merges are ADDITIVE, so
                a duplicate fold is permanent and invisible. This looks
                for the signature - a day whose mention total is a near
                exact multiple of its neighbours.
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ABS = os.path.join(PROJECT_ROOT, "ABSTRACTED_DATA")
REF = os.path.join(PROJECT_ROOT, "data", "reference")
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
PRICES = os.path.join(PROJECT_ROOT, "data", "prices", "prices.parquet")
STATUS = os.path.join(REF, "ingestion_status.json")

OK, WARN, BAD = "ok", "warn", "BAD"


def _age_days(path):
    if not os.path.exists(path):
        return None
    return (datetime.datetime.now()
            - datetime.datetime.fromtimestamp(os.path.getmtime(path))).days


def check_freshness(out, say):
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


def check_coverage(out, say):
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
    say("BACKUPS of data/reference")
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
        # WARN, never BAD. Verified on this store: the scan flags
        # 2021-01/02 (GameStop), 2021-06 (AMC) and 2026-07 (the month
        # StockTwits and X came online) alongside a deliberately injected
        # duplicate. It cannot tell a mania or a new source from a double
        # fold - only a human with the ledger can - so it hands over
        # candidates rather than pretending to a verdict.
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
        say("       data/reference/historical_fold_ledger.json - that is "
            "the guilty signature.", WARN)
    else:
        say("  no month sits at >=1.7x its neighbours", OK)
    out["suspect_double_count_months"] = [
        dict(month=str(p), total=int(v), neighbour_median=int(b),
             ratio=round(r, 2)) for p, v, b, r in flagged]


def main() -> int:
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

    for fn in (check_freshness, check_coverage, check_dedup, check_ledgers,
               check_backups, check_raw, check_double_count):
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
