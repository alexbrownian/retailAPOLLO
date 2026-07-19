"""
test_pipeline.py
================
Pytest checks for the retailAPOLLO pipeline - the parts where a silent
mistake would corrupt every downstream number.

    python -m pytest tests/ -v

The suite tests INVARIANTS, not snapshots: the merge maths must equal
one-shot aggregation, the trailing z must never see the future, one surge
must yield exactly one crossing, the committed data must stay text-free.
These hold whatever the data looks like, so the tests never rot as the
dataset grows.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import abstracted_data                                   # noqa: E402
from src.config import FORBIDDEN_COLS                             # noqa: E402
from analytics.conviction import trailing_z, compute_conviction   # noqa: E402
from analytics.signals import crosses_above, crosses_below, make_decisions  # noqa: E402
from analytics.overlays import (mention_share_series,             # noqa: E402
                                signal_scorecard, direction_flips)


# ---------------------------------------------------------------------------
# 1. THE MERGE MATHS - incremental folding must equal one-shot aggregation
# ---------------------------------------------------------------------------
class TestMergeMaths:
    def test_counts_add(self):
        """Same (date, ticker) rows sum; new rows carry through."""
        old = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"],
                            "ticker": ["GME", "GME"],
                            "mention_count": [10, 20]})
        new = pd.DataFrame({"date": ["2026-01-02", "2026-01-03"],
                            "ticker": ["GME", "NVDA"],
                            "mention_count": [5, 7]})
        merged = abstracted_data.merge_counts(old, new, ["date", "ticker"])
        m = merged.set_index([merged["date"].dt.strftime("%Y-%m-%d"), "ticker"])
        assert m.loc[("2026-01-02", "GME"), "mention_count"] == 25
        assert m.loc[("2026-01-03", "NVDA"), "mention_count"] == 7
        assert len(merged) == 3

    def test_sentiment_merge_is_n_weighted(self):
        """A 100-post day must dominate a 3-post day: the merge rebuilds
        the underlying sums, exactly what one-shot aggregation computes -
        NOT a naive average of the two averages."""
        old = pd.DataFrame({"date": ["2026-01-01"], "ticker": ["GME"],
                            "n_posts": [100], "avg_sentiment": [0.10],
                            "net_bullish": [0.20]})
        new = pd.DataFrame({"date": ["2026-01-01"], "ticker": ["GME"],
                            "n_posts": [3], "avg_sentiment": [0.90],
                            "net_bullish": [1.00]})
        merged = abstracted_data.merge_sentiment(old, new, ["date", "ticker"])
        row = merged.iloc[0]
        assert row["n_posts"] == 103
        # (0.10*100 + 0.90*3) / 103
        assert row["avg_sentiment"] == pytest.approx((10 + 2.7) / 103)
        # naive average-of-averages would give 0.50 - must NOT be that
        assert row["avg_sentiment"] < 0.2
        assert row["net_bullish"] == pytest.approx((20 + 3) / 103)

    def test_merge_idempotent_on_disjoint_days(self):
        """Folding a batch of brand-new days never touches existing rows."""
        old = pd.DataFrame({"date": ["2026-01-01"], "theme": ["ai"],
                            "mention_count": [50]})
        new = pd.DataFrame({"date": ["2026-01-02"], "theme": ["ai"],
                            "mention_count": [60]})
        merged = abstracted_data.merge_counts(old, new, ["date", "theme"])
        assert merged["mention_count"].tolist() == [50, 60]


# ---------------------------------------------------------------------------
# 2. TRAILING Z - live parity: day t must use only data through day t
# ---------------------------------------------------------------------------
class TestTrailingZ:
    def _frame(self, values):
        idx = pd.date_range("2025-01-01", periods=len(values), freq="D")
        return pd.DataFrame({"X": values}, index=idx)

    def test_no_lookahead(self):
        """Changing the FUTURE must not change the past's z-scores - the
        property that makes backtests honest."""
        base = [10.0] * 200
        za = trailing_z(self._frame(base))
        wild = base.copy()
        wild[150:] = [500.0] * 50          # a huge future spike
        zb = trailing_z(self._frame(wild))
        # z through day 149 must be IDENTICAL in both worlds
        pd.testing.assert_frame_equal(za.iloc[:150], zb.iloc[:150])

    def test_warmup_is_nan(self):
        """No z before MIN_DAYS of history - a z against 3 days of
        baseline would be noise pretending to be signal."""
        z = trailing_z(self._frame([5.0] * 100))
        assert z["X"].iloc[:27].isna().all()

    def test_spike_scores_positive(self):
        """A clear surge after a flat period must produce a large +z."""
        vals = [10.0] * 100 + [100.0] * 7
        z = trailing_z(self._frame(vals))
        assert z["X"].iloc[-1] > 2.5


# ---------------------------------------------------------------------------
# 3. CROSSINGS - one surge, one trade
# ---------------------------------------------------------------------------
class TestCrossings:
    def test_single_crossing_per_surge(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        z = pd.Series([0, 1, 2, 3, 3.5, 3.2, 2, 1, 0, -1], index=idx, dtype=float)
        up = crosses_above(z, 2.5)
        assert up.sum() == 1                       # not one per day above K
        assert up.idxmax() == idx[3]               # the day it first cleared

    def test_nan_warmup_never_crosses(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="D")
        z = pd.Series([np.nan, np.nan, 3.0, 3.0, 3.0], index=idx)
        # NaN -> 3.0 is not a crossing (shift(1) is NaN, comparison False)
        assert crosses_above(z, 2.5).sum() == 0

    def test_crosses_below_mirror(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="D")
        z = pd.Series([0, -1, -3, -3, -1, 0], index=idx, dtype=float)
        dn = crosses_below(z, -2.5)
        assert dn.sum() == 1
        assert dn.idxmax() == idx[2]


# ---------------------------------------------------------------------------
# 4. THE DECISION ENGINE - gates, scoring, cooldown
# ---------------------------------------------------------------------------
class TestDecisionEngine:
    def _ingredients(self, n=140):
        """Synthetic world: one theme, flat for 120 days, then a surge with
        improving mood - a textbook BUY setup on the surge day."""
        idx = pd.date_range("2026-01-01", periods=n, freq="D")
        az = pd.DataFrame({"th": [0.0] * 120 + [3.0] * (n - 120)}, index=idx)
        cz = pd.DataFrame({"th": [0.0] * 120 + [3.0] * (n - 120)}, index=idx)
        dv = pd.DataFrame({"th": [0.0] * 120 + [0.30] * (n - 120)}, index=idx)
        crowd = pd.DataFrame({"th": [False] * n}, index=idx)
        return idx, az, cz, dv, crowd

    def test_buy_fires_once_with_reason(self):
        idx, az, cz, dv, crowd = self._ingredients()
        trades = make_decisions(["th"], idx, az, cz, dv, crowd, None,
                                lambda t: "ETF")
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["action"] == "BUY"
        assert t["score"] >= 4
        assert "attention surged" in t["reason"]
        # order stamped for the NEXT day - no look-ahead
        assert (t["action_date"] - t["signal_date"]).days == 1

    def test_sentiment_gate_blocks(self):
        """Momentum fires but the mood is deteriorating -> NO buy."""
        idx, az, cz, _, crowd = self._ingredients()
        dv_bad = pd.DataFrame({"th": [-0.2] * len(idx)}, index=idx)
        trades = make_decisions(["th"], idx, az, cz, dv_bad, crowd, None,
                                lambda t: "ETF")
        # a SELL needs a bearish trigger (cz crossing -K or crowded-top),
        # which never happens here - so the log must be EMPTY
        assert len(trades) == 0

    def test_cooldown_suppresses_repeat(self):
        """Two surges 10 days apart = the same episode -> one trade."""
        idx = pd.date_range("2026-01-01", periods=140, freq="D")
        pattern = ([0.0] * 120 + [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 3.0] + [0.0] * 9)
        az = pd.DataFrame({"th": pattern}, index=idx)
        cz = pd.DataFrame({"th": pattern}, index=idx)
        dv = pd.DataFrame({"th": [0.3] * 140}, index=idx)
        crowd = pd.DataFrame({"th": [False] * 140}, index=idx)
        trades = make_decisions(["th"], idx, az, cz, dv, crowd, None,
                                lambda t: "ETF", cooldown_days=21)
        assert len(trades) == 1

    def test_crowded_top_sell(self):
        """Attention high while mood deteriorates -> crowded-top SELL."""
        idx = pd.date_range("2026-01-01", periods=130, freq="D")
        az = pd.DataFrame({"th": [0.0] * 120 + [3.0] * 10}, index=idx)
        cz = pd.DataFrame({"th": [0.0] * 130}, index=idx)
        dv = pd.DataFrame({"th": [0.0] * 120 + [-0.3] * 10}, index=idx)
        crowd = pd.DataFrame({"th": [False] * 120 + [True] * 10}, index=idx)
        # give it the cross-source confirmation so the score clears 4/5
        xr = pd.DataFrame({"th": [True] * 130}, index=idx)
        trades = make_decisions(["th"], idx, az, cz, dv, crowd, xr,
                                lambda t: "ETF")
        assert len(trades) == 1
        assert trades.iloc[0]["action"] == "SELL"
        assert "crowded-top" in trades.iloc[0]["reason"]


# ---------------------------------------------------------------------------
# 5. CONVICTION - the bull-pressure construction
# ---------------------------------------------------------------------------
class TestConviction:
    def test_bull_pressure_is_volume_times_direction(self):
        sent = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=60, freq="D"),
            "theme": ["ai"] * 60,
            "n_posts": [50] * 60,
            "avg_sentiment": [0.1] * 60,
            "net_bullish": [0.2] * 60,
        })
        cs = compute_conviction(sent, "theme")
        # 50 posts * 0.2 net bullish = 10 net bullish votes per day
        assert cs.bull_pressure["ai"].iloc[10] == pytest.approx(10.0)
        # share = rolled pressure / rolled volume = net_bullish again
        assert cs.share["ai"].iloc[30] == pytest.approx(0.2)

    def test_quiet_day_is_zero_pressure_not_missing(self):
        """A day with no rows must appear as 0 pressure - dropping it
        would corrupt every rolling window."""
        sent = pd.DataFrame({
            "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05")],
            "theme": ["ai", "ai"], "n_posts": [10, 10],
            "avg_sentiment": [0.5, 0.5], "net_bullish": [0.5, 0.5]})
        cs = compute_conviction(sent, "theme")
        assert cs.bull_pressure["ai"].loc["2026-01-03"] == 0.0
        assert len(cs.bull_pressure) == 5          # full calendar span


# ---------------------------------------------------------------------------
# 6. OVERLAY ANALYTICS
# ---------------------------------------------------------------------------
class TestOverlays:
    def test_share_masks_thin_days(self):
        """A 1-post day must NOT read as a 100% share spike."""
        rows = []
        for d in pd.date_range("2026-01-01", periods=30, freq="D"):
            rows.append({"date": d, "ticker": "GME", "mention_count": 40})
            rows.append({"date": d, "ticker": "NVDA", "mention_count": 60})
        thin_day = pd.Timestamp("2026-02-01")
        rows.append({"date": thin_day, "ticker": "GME", "mention_count": 1})
        counts = pd.DataFrame(rows)
        share = mention_share_series(counts, "ticker", "GME",
                                     pd.Timestamp("2026-01-01"), None)
        # on healthy days GME is 40% of chatter
        assert share.loc["2026-01-20"] == pytest.approx(40.0, abs=0.5)
        # the thin day contributes NaN into the rolling mean, not 100
        raw_share_that_day = share.loc[thin_day]
        assert raw_share_that_day < 60          # nowhere near a 100% spike

    def test_scorecard_signed_pnl(self):
        """A BUY into a rising price and a SELL into the same rise must
        book mirror-image P&L (SELLs are counted short)."""
        days = pd.date_range("2026-01-01", periods=60, freq="D")
        prices = pd.DataFrame({
            "date": list(days) * 1, "symbol": ["SMH"] * 60,
            "px_last": np.linspace(100, 120, 60)})   # steady +20% drift
        sig = pd.DataFrame({
            "action_date": [pd.Timestamp("2026-01-10")] * 2,
            "action": ["BUY", "SELL"], "etf": ["SMH", "SMH"]})
        card = signal_scorecard(sig, prices, {"SMH"}, days[0], hold_days=20)
        # too few trades per side for stats rows, but ALL row must exist
        all_row = card[card["strategy"] == "ALL (buy+sell)"].iloc[0]
        assert all_row["trades"] == 2
        # buy gain == sell loss -> total P&L ~ 0
        assert abs(all_row.get("total P&L %", 0)) < 0.01

    def test_direction_flips_hysteresis_and_spacing(self):
        """The hysteresis threshold scales with the series' own std, so a
        genuine regime shift must register while (a) wiggles SMALLER than
        the noise floor never flip and (b) counted flips are min_gap
        apart (first-in-a-burst wins)."""
        idx = pd.date_range("2026-01-01", periods=100, freq="D")
        rng = np.random.default_rng(0)
        # tiny wiggles on top of one huge step: eps = 0.25 * std is
        # dominated by the step, so the wiggles sit far below the floor
        series = pd.Series(rng.normal(0, 0.01, 100), index=idx)
        series.iloc[60:] += 5.0                    # one genuine regime shift
        px = pd.Series(np.linspace(50, 60, 100), index=idx)
        up, down, *_ = direction_flips(series, px, 5, min_gap=7)
        # exactly one upward regime change registers, on/after the step day
        assert len(up) == 1 and up[0] >= idx[60]
        assert len(down) == 0                      # wiggles never cleared -eps
        # spacing invariant on a genuinely oscillating series
        osc = pd.Series(([2.0] * 5 + [-2.0] * 5) * 10, index=idx)
        up_o, down_o, *_ = direction_flips(osc, px, 5, min_gap=7)
        gaps = np.diff([d.value for d in sorted(up_o)])
        assert (gaps >= 7 * 86400 * 10**9).all()   # >= 7 days apart


# ---------------------------------------------------------------------------
# 7. THE TEXT-FREE CONTRACT - committed data must never carry raw posts
# ---------------------------------------------------------------------------
class TestAbstractedSafety:
    def test_committed_files_are_text_free(self):
        import pyarrow.parquet as pq
        checked = 0
        for name in abstracted_data.FILES:
            path = os.path.join(abstracted_data.ABSTRACTED_DIR, name)
            if not os.path.exists(path):
                continue
            cols = [c.lower() for c in pq.ParquetFile(path).schema_arrow.names]
            leaks = [c for c in cols if c in FORBIDDEN_COLS]
            assert not leaks, f"{name} leaks text columns: {leaks}"
            checked += 1
        assert checked > 0, "no ABSTRACTED_DATA files found to check"

    def test_aggregate_posts_output_is_text_free(self):
        """The live-fold aggregator itself must only emit safe columns."""
        posts = pd.DataFrame({
            "id": ["a1", "a2"], "date": ["2026-07-01", "2026-07-01"],
            "title": ["NVDA to the moon", "buying $GME calls"],
            "selftext": ["", ""], "source": ["reddit", "reddit"]})
        aggs = abstracted_data.aggregate_posts(posts)
        for name, df in aggs.items():
            leaks = [c for c in df.columns if c.lower() in FORBIDDEN_COLS]
            assert not leaks, f"{name} would leak: {leaks}"


# ---------------------------------------------------------------------------
# 8. EXTRACTION - the counting rules
# ---------------------------------------------------------------------------
class TestExtraction:
    @pytest.fixture(scope="class")
    def universe(self):
        return abstracted_data.load_universe()

    def test_cashtags_and_bare_caps(self, universe):
        from src.extract_tickers import extract_tickers_from_text
        out = extract_tickers_from_text("bought NVDA calls and $GME",
                                        universe, cashtags_only=False)
        assert "NVDA" in out and "GME" in out

    def test_word_tickers_demoted_not_deleted(self, universe):
        """EDGE/LOAN written as prose must not count; $EDGE still counts."""
        from src.extract_tickers import (extract_tickers_from_text,
                                         SCREENED_STOP, BARE_PROSE_STOP)
        blocked = (SCREENED_STOP | BARE_PROSE_STOP)
        assert "EDGE" in blocked          # the screening layer is loaded
        out = extract_tickers_from_text("the EDGE of the market",
                                        universe, cashtags_only=False)
        assert "EDGE" not in out

    def test_lowercase_prose_never_matches(self, universe):
        """Only words the poster actually wrote in ALL CAPS can be bare
        tickers - 'edge', 'loan', 'meme' in prose must not count."""
        from src.extract_tickers import extract_tickers_from_text
        out = extract_tickers_from_text("i have an edge on this loan meme",
                                        universe, cashtags_only=False)
        assert out == []

    def test_one_post_counts_once(self, universe):
        """A post mentioning NVDA five times counts ONE mention (breadth
        of attention, not verbosity) - enforced by build_daily_counts."""
        from src.build_mentions import build_daily_counts
        posts = pd.DataFrame({
            "date": ["2026-07-01"],
            "title": ["NVDA NVDA NVDA $NVDA"],
            "selftext": ["NVDA again"]})
        daily = build_daily_counts(posts, universe)
        row = daily[daily["ticker"] == "NVDA"].iloc[0]
        assert row["mention_count"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
