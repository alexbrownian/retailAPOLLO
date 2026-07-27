# helper/tune_thresholds.py
# =========================
# Tune notebook 10's signal thresholds across ALL market regimes.
#
#     python helper/tune_thresholds.py               # full grid (~27 runs)
#     python helper/tune_thresholds.py --quick       # 8-combo grid, faster
#
# HOW IT WORKS
#   1. For every (K, MIN_SCORE, COOLDOWN) combo, notebook 10 is executed for
#      real (via the SIG_* env overrides), so the evaluated engine is EXACTLY
#      the production engine - no reimplementation drift.
#   2. Every signal is scored by its SIGNED forward return over the next
#      FWD_DAYS: BUY wants price up, SELL wants price down.
#   3. Results split into ERAS (different regimes). A combo's score is its
#      WORST-era average signed return - robustness first, so the winner is
#      the combo that never falls apart, not the one that aced one mania.
#
# PREREQUISITE: data/prices/prices.parquet must span the whole backtest
#   range. Set the window in update_data.py to the full range once and run
#   pull_bloomberg_prices.py (Terminal open) before tuning.
#
# The last grid run leaves notebook 10's outputs in a tuned state; the
# script finishes by re-running notebook 10 with the RECOMMENDED combo so
# the signal files on disk match the leaderboard's winner.

import argparse
import itertools
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

P = os.path.join(ROOT, "data", "processed")
PRICES = os.path.join(ROOT, "data", "prices", "prices.parquet")
FWD_DAYS = 10                     # forward-return horizon per signal
MIN_TRADES_PER_ERA = 4            # eras with fewer signals than this are skipped
ERAS = [("2017-2019", "2017-01-01", "2020-01-01"),
        ("2020-2021", "2020-01-01", "2022-01-01"),
        ("2022-2023", "2022-01-01", "2024-01-01"),
        ("2024+",     "2024-01-01", "2099-01-01")]


def run_notebook_10(env_overrides):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})
    r = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook",
                        "--execute", "notebooks/10_trading_signals.ipynb",
                        "--output", "10_trading_signals.tune.ipynb",
                        "--output-dir", "notebooks",
                        "--ExecutePreprocessor.timeout=1800"],
                       env=env, capture_output=True, text=True)
    tmp = os.path.join(ROOT, "notebooks", "10_trading_signals.tune.ipynb")
    try:
        if os.path.exists(tmp):
            os.remove(tmp)                    # scratch output, never kept
    except OSError:
        pass                                  # harmless leftover; gitignored
    return r.returncode == 0, (r.stderr or "")[-400:]


def load_signals():
    frames = []
    for fname, sym_col in [("trade_signals.parquet", "etf"),
                           ("trade_signals_tickers.parquet", "ticker")]:
        path = os.path.join(P, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if len(df):
            df = df.rename(columns={sym_col: "symbol"})
            frames.append(df[["symbol", "action", "action_date"]])
    if not frames:
        return pd.DataFrame(columns=["symbol", "action", "action_date"])
    out = pd.concat(frames, ignore_index=True)
    out["action_date"] = pd.to_datetime(out["action_date"])
    return out


def evaluate(signals, price_series_by_symbol):
    """Signed forward return per signal, then per-era stats."""
    rows = []
    for _, s in signals.iterrows():
        px = price_series_by_symbol.get(s["symbol"])
        if px is None or px.empty:
            continue
        p0 = px.asof(s["action_date"])
        p1 = px.asof(s["action_date"] + pd.Timedelta(days=FWD_DAYS))
        if pd.isna(p0) or pd.isna(p1):
            continue
        direction = 1 if s["action"] == "BUY" else -1
        rows.append({"date": s["action_date"],
                     "signed_ret": (p1 / p0 - 1) * 100 * direction})
    if not rows:
        return {}
    perf = pd.DataFrame(rows)
    era_stats = {}
    for era, lo, hi in ERAS:
        grp = perf[(perf["date"] >= lo) & (perf["date"] < hi)]
        if len(grp) >= MIN_TRADES_PER_ERA:
            era_stats[era] = {"n": len(grp),
                              "avg": grp["signed_ret"].mean(),
                              "hit": (grp["signed_ret"] > 0).mean() * 100}
    return era_stats


def main():
    ap = argparse.ArgumentParser(description="Tune notebook 10's thresholds across regimes.")
    ap.add_argument("--quick", action="store_true", help="smaller grid (8 combos)")
    args = ap.parse_args()

    if not os.path.exists(PRICES):
        print("no data/prices/prices.parquet - pull full-range prices first "
              "(window = whole backtest range, Terminal open).")
        return 1
    prices = pd.read_parquet(PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    span_lo, span_hi = prices["date"].min(), prices["date"].max()
    print(f"prices span {span_lo.date()} -> {span_hi.date()} "
          f"({prices['symbol'].nunique()} symbols)")
    if span_lo > pd.Timestamp("2018-01-01"):
        print("WARNING: prices start late - eras before that cannot be scored. "
              "Pull prices over the full range for a fair tuning.")
    by_symbol = {}
    for sym, grp in prices.groupby("symbol"):
        by_symbol[sym] = grp.sort_values("date").set_index("date")["px_last"].asfreq("D").ffill()

    if args.quick:
        grid = list(itertools.product([2.0, 2.5], [4, 5], [7, 14]))
    else:
        grid = list(itertools.product([1.5, 2.0, 2.5], [3, 4, 5], [7, 14, 21]))
    print(f"grid: {len(grid)} combos x 1 notebook run each (expect ~20s per run)\n")

    results = []
    for i, (k, ms, cd) in enumerate(grid, 1):
        overrides = {"SIG_K": k, "SIG_MIN_SCORE": ms,
                     "SIG_MIN_SCORE_SELL": max(ms - 1, 2), "SIG_COOLDOWN": cd}
        ok, err = run_notebook_10(overrides)
        if not ok:
            print(f"[{i:>2}/{len(grid)}] K={k} MS={ms} CD={cd}  NOTEBOOK FAILED: {err[:120]}")
            continue
        era_stats = evaluate(load_signals(), by_symbol)
        if not era_stats:
            print(f"[{i:>2}/{len(grid)}] K={k} MS={ms} CD={cd}  no scoreable signals")
            continue
        worst = min(s["avg"] for s in era_stats.values())
        mean = sum(s["avg"] for s in era_stats.values()) / len(era_stats)
        total_n = sum(s["n"] for s in era_stats.values())
        results.append({"K": k, "MIN_SCORE": ms, "COOLDOWN": cd,
                        "eras_scored": len(era_stats), "signals": total_n,
                        "worst_era_avg": worst, "mean_avg": mean,
                        "detail": era_stats})
        print(f"[{i:>2}/{len(grid)}] K={k} MS={ms} CD={cd}  "
              f"signals={total_n:>4} | worst-era avg {worst:+.2f}% | mean {mean:+.2f}%")

    if not results:
        print("nothing scoreable - check price coverage and the signal files.")
        return 1

    # robustness ranking: best WORST-era average first, mean as tiebreak
    results.sort(key=lambda r: (r["worst_era_avg"], r["mean_avg"]), reverse=True)
    print("\n" + "=" * 74)
    print("LEADERBOARD (ranked by worst-era average signed return - robustness)")
    print("=" * 74)
    for r in results[:8]:
        print(f"K={r['K']:<4} MIN_SCORE={r['MIN_SCORE']} COOLDOWN={r['COOLDOWN']:<3} | "
              f"signals {r['signals']:>4} over {r['eras_scored']} eras | "
              f"worst {r['worst_era_avg']:+.2f}% | mean {r['mean_avg']:+.2f}%")
    best = results[0]
    print("\nRECOMMENDED:", f"K={best['K']}  MIN_SCORE={best['MIN_SCORE']}  "
          f"COOLDOWN={best['COOLDOWN']}")
    for era, s in best["detail"].items():
        print(f"  {era:<10} {s['n']:>4} signals | hit {s['hit']:3.0f}% | avg {s['avg']:+.2f}%")
    print("\nApply permanently by editing notebook 10's parameter cell, or set")
    print("SIG_K / SIG_MIN_SCORE / SIG_COOLDOWN in the environment.")

    print("\nre-running notebook 10 with the recommended combo so the signal "
          "files on disk match...")
    ok, _ = run_notebook_10({"SIG_K": best["K"], "SIG_MIN_SCORE": best["MIN_SCORE"],
                             "SIG_MIN_SCORE_SELL": max(best["MIN_SCORE"] - 1, 2),
                             "SIG_COOLDOWN": best["COOLDOWN"]})
    print("done." if ok else "final rerun failed - rerun notebook 10 manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
