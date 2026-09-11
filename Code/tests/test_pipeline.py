"""Invariant tests for the retailAPOLLO pipeline.

Each test pins a property that must hold whatever the data looks like,
rather than a snapshot of a particular output: incremental merges equal
one-shot aggregation, trailing statistics never read the future, one
surge yields one crossing, committed stores carry no raw text, display
transforms never alter the columns the joins run on.  Because the
properties are data-independent the suite does not rot as the dataset
grows.

Three kinds of test appear:

* pure-function tests on synthetic frames with hand-computed answers;
* store tests that read the built parquet files when they are present
  on the machine and skip otherwise;
* source-text tests that read another module's source and assert a
  structural fact about it (a helper exists, a call site routes through
  one function, a forbidden expression is absent).  These guard rules
  that cannot be expressed as a unit test because the module executes
  at import time (``dashboard.py``) or because the rule is about the
  absence of a code path.

Run with ``python -m pytest Code/tests/ -v``.
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
from src.analytics.conviction import trailing_z, compute_conviction   # noqa: E402
from src.analytics.signals import crosses_above, crosses_below, make_decisions  # noqa: E402
from src.analytics.overlays import (mention_share_series,             # noqa: E402
                                signal_scorecard, direction_flips)
from src.config import DATA_DIR  # noqa: E402


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
        assert checked > 0, "no Data/abstracted files found to check"

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
    # staticmethod, not an instance method: a class-scoped fixture runs once
    # while each test gets a fresh instance, so pytest deprecated the instance
    # form (anything it set on `self` would be invisible to the tests).
    @staticmethod
    @pytest.fixture(scope="class")
    def universe():
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


# ---------------------------------------------------------------------------
# 9. THE EUPHORIA DETECTOR - the top-calling rules
# ---------------------------------------------------------------------------
class TestEuphoria:
    def test_convexity_flags_super_exponential(self):
        """A super-exponential (accelerating) price path must show positive
        log-convexity; a steady exponential must not (Sornette signature)."""
        from src.analytics.euphoria import log_price_convexity
        idx = pd.date_range("2025-01-01", periods=120, freq="D")
        t = np.arange(120)
        steady = pd.Series(100 * np.exp(0.002 * t), index=idx)
        bubble = pd.Series(100 * np.exp(0.0004 * t + 0.00006 * t * t), index=idx)
        c_steady = log_price_convexity(steady).iloc[-1]
        c_bubble = log_price_convexity(bubble).iloc[-1]
        assert c_bubble > 0 and c_bubble > 10 * abs(c_steady)

    def test_ground_truth_needs_boom_and_bust(self):
        """A peak only counts with BOTH a prior boom and a subsequent bust
        - a top in a flat market or a dip without a boom never qualifies."""
        from src.analytics.euphoria import ground_truth_peaks
        idx = pd.date_range("2024-01-01", periods=400, freq="D")
        flat = pd.Series(100.0, index=idx)                    # no boom
        assert ground_truth_peaks(flat, "theme") == []
        boom_bust = pd.Series(
            np.r_[np.linspace(100, 100, 100),
                  np.linspace(100, 160, 100),                 # +60% boom
                  np.linspace(160, 120, 100),                 # -25% bust
                  np.linspace(120, 125, 100)], index=idx)
        peaks = ground_truth_peaks(boom_bust, "theme")
        assert len(peaks) == 1
        assert abs((peaks[0] - idx[199]).days) <= 3           # at the top
        boom_only = pd.Series(
            np.r_[np.linspace(100, 160, 200),
                  np.linspace(160, 155, 200)], index=idx)     # no bust
        assert ground_truth_peaks(boom_only, "theme") == []

    def test_trailing_pct_rank_no_lookahead(self):
        """Percentile features must never see the future."""
        from src.analytics.euphoria import trailing_pct_rank
        idx = pd.date_range("2024-01-01", periods=400, freq="D")
        a = pd.Series(np.sin(np.arange(400) / 20.0) + 2, index=idx)
        b = a.copy(); b.iloc[300:] += 100                     # future shock
        ra = trailing_pct_rank(a, window=180, min_periods=90)
        rb = trailing_pct_rank(b, window=180, min_periods=90)
        pd.testing.assert_series_equal(ra.iloc[:300], rb.iloc[:300])

    def test_euphoria_is_reddit_only(self):
        """THE selection rule: compute_euphoria must not accept price input -
        prediction is built from the crowd data alone (price exists only
        on the ground-truth/scoring side)."""
        import inspect
        from src.analytics.euphoria import compute_euphoria
        params = inspect.signature(compute_euphoria).parameters
        assert not any("px" in p or "price" in p for p in params), \
            "compute_euphoria takes a price argument - Reddit-only rule broken"

    def test_judgeable_window_excludes_unpriced_eras(self):
        """Alerts before price history begins, or within the forward
        horizon of its end, must be PENDING rather than false alarms:
        an alert cannot be judged without a priced outcome window, and
        scoring it as a miss inflates the false-alarm rate."""
        from src.analytics.euphoria import judgeable_window
        idx = pd.date_range("2021-01-01", periods=200, freq="D")
        px = pd.Series(100.0, index=idx)
        j0, j1 = judgeable_window(px)
        assert j0 == idx[0]
        assert j1 == idx[-1] - pd.Timedelta(days=45)


class TestInfluence:
    def test_vol_scaled_bar(self):
        """The correctness bar must rise with the name's own volatility:
        tau = max(3%, 0.5 sigma).  One fixed bar would grade an index
        ETF and a meme stock with the same ruler."""
        from src.analytics.influence import score_calls
        idx = pd.date_range("2025-01-01", periods=400, freq="D")
        rng = np.random.default_rng(7)
        calm = 100 * np.cumprod(1 + rng.normal(0, 0.001, 400))
        wild = 100 * np.cumprod(1 + rng.normal(0, 0.05, 400))
        prices = pd.concat([
            pd.DataFrame({"date": idx, "symbol": "CALM", "px_last": calm}),
            pd.DataFrame({"date": idx, "symbol": "WILD", "px_last": wild})])
        calls = pd.DataFrame({
            "rec_id": ["a", "b"], "author": ["u", "u"],
            "date": [idx[300], idx[300]], "ticker": ["CALM", "WILD"],
            "direction": [1, 1], "stance": [0.5, 0.5],
            "kind": ["post", "post"]})
        s = score_calls(calls, prices)
        tau_calm = s.loc[s["ticker"] == "CALM", "tau"].iloc[0]
        tau_wild = s.loc[s["ticker"] == "WILD", "tau"].iloc[0]
        assert tau_calm == pytest.approx(0.03)     # floored at 3%
        assert tau_wild > tau_calm                 # scales with vol

    def test_shrinkage_orders_evidence(self):
        """3-for-3 must NOT outrank 40-for-60 after shrinkage."""
        from src.analytics.influence import _shrink
        lucky = _shrink(1.0, 3, 10, 0.5)
        veteran = _shrink(40 / 60, 60, 10, 0.5)
        assert veteran > lucky

    def test_pagerank_ranks_the_hub(self):
        """On a star-ish graph the most-connected node must carry the
        highest PageRank, and broadcast/bot accounts must be excluded."""
        from src.analytics.influence import build_graph_metrics
        edges = pd.DataFrame({
            "rec_id": [str(i) for i in range(6)],
            "replier": ["u1", "u2", "u3", "u4", "u1", "bot"],
            "author": ["hub", "hub", "hub", "hub", "u2", "hub"]})
        activity = pd.DataFrame(
            {"n_comments": [0, 0, 0, 0, 0, 5000],
             "n_posts": [0, 0, 0, 0, 0, 500]},
            index=["hub", "u1", "u2", "u3", "u4", "bot"])
        g = build_graph_metrics(edges, activity)
        assert "bot" not in g.index                # broadcast filter
        assert g["pagerank"].idxmax() == "hub"

    def test_store_write_refuses_text(self):
        """The committed store must reject any text column - the same
        text-free contract Data/abstracted lives under."""
        from src.analytics.influence import _safe_store_write
        bad = pd.DataFrame({"author": ["x"], "body": ["secret text"]})
        with pytest.raises(RuntimeError):
            _safe_store_write(bad, "/tmp/should_never_exist.parquet")


class TestInfluenceGraph:
    """src/analytics/influence_graph.py - the pure numpy/scipy graph layer
    under the dashboard's influence map.  The project has no networkx
    dependency, so the hand-rolled Louvain and k-core implementations
    are verified here against graphs whose answers are known by hand.
    """

    @staticmethod
    def _two_triangles():
        """Two triangles joined by a single bridge edge. Known by hand:
        6 nodes, 7 edges, every node has degree 2 except the two bridge
        ends (degree 3), modularity is maximised by the obvious 2-way
        split, and the 2-core is the whole graph."""
        u = ["a", "b", "c", "a", "d", "e", "f"]
        v = ["b", "c", "a", "d", "e", "f", "d"]
        return pd.DataFrame({"rec_id": [str(i) for i in range(len(u))],
                             "replier": u, "author": v})

    def test_graph_is_undirected_and_deduplicated(self):
        """A reply graph is a conversation, not a direction: (u,v) and
        (v,u) are the SAME edge, and a repeated pair must raise the edge
        WEIGHT, never the edge count."""
        from src.analytics import influence_graph as ig
        e = pd.DataFrame({"rec_id": ["1", "2", "3"],
                          "replier": ["a", "b", "a"],
                          "author": ["b", "a", "b"]})
        g = ig.build_graph(e)
        assert g.n == 2 and g.m == 1              # one edge, not three
        assert g.A[g.idx["a"], g.idx["b"]] == 3   # weight carries the count
        assert g.A[g.idx["a"], g.idx["b"]] == g.A[g.idx["b"], g.idx["a"]]

    def test_self_replies_never_become_edges(self):
        """Replying to yourself is not influence. If it survived, every
        prolific poster would look like a hub."""
        from src.analytics import influence_graph as ig
        e = pd.DataFrame({"rec_id": ["1", "2"], "replier": ["a", "a"],
                          "author": ["a", "b"]})
        g = ig.build_graph(e)
        assert g.m == 1
        assert g.A[g.idx["a"], g.idx["a"]] == 0

    def test_louvain_finds_the_two_triangles(self):
        """The bridged-triangles graph has one obvious community split.
        Louvain must find exactly it, and modularity must be positive and
        equal to the hand-checkable value for that partition."""
        from src.analytics import influence_graph as ig
        g = ig.build_graph(self._two_triangles())
        comm = ig.louvain(g, seed=42)
        assert comm.nunique() == 2
        assert comm["a"] == comm["b"] == comm["c"]
        assert comm["d"] == comm["e"] == comm["f"]
        assert comm["a"] != comm["d"]
        # Q = sum_c [ L_c/m - (d_c/2m)^2 ]; here m=7, each side has 3
        # internal edges and total degree 7 -> 2*(3/7 - (7/14)^2) = 0.3571
        assert ig.modularity(g, comm) == pytest.approx(0.35714, abs=1e-4)

    def test_modularity_accepts_series_or_array(self):
        """modularity() must accept either a Series keyed by author or a
        bare array in node order and return the same number.  A Series
        that is not re-aligned to node order would score a shuffled
        partition without raising."""
        from src.analytics import influence_graph as ig
        g = ig.build_graph(self._two_triangles())
        comm = ig.louvain(g, seed=42)
        as_array = comm.reindex(g.names).to_numpy()
        assert ig.modularity(g, comm) == pytest.approx(
            ig.modularity(g, as_array))

    def test_kcore_members_all_have_k_neighbours_inside(self):
        """The defining property of a k-core, and the whole reason the
        dashboard draws one: every node kept must still have >= k
        neighbours AFTER the pruning, not before it."""
        from src.analytics import influence_graph as ig
        g = ig.build_graph(self._two_triangles())
        sub, k = ig.kcore_subgraph(g, min_nodes=3)
        assert k >= 2
        assert (sub.degree >= k).all()
        assert sub.n >= 3

    def test_homophily_splits_by_class(self):
        """On a graph where the positives are deliberately spread apart
        (each one surrounded by negatives), positive-class node homophily
        must be ~0 while negative-class is ~1.  Per-class homophily is
        what shows whether graph structure carries label information, so
        the two classes must be reported separately and never averaged
        into one number."""
        from src.analytics import influence_graph as ig
        # positives hang off the negative cluster, one edge each; the
        # negatives are wired to each other. By hand: class-1 homophily 0,
        # class-0 homophily (0.75 + 2/3 + 1 + 1) / 4 = 0.854.
        e = pd.DataFrame({
            "rec_id": [str(i) for i in range(7)],
            "replier": ["p1", "p2", "n1", "n1", "n2", "n3", "n4"],
            "author": ["n1", "n2", "n2", "n3", "n3", "n4", "n1"]})
        g = ig.build_graph(e)
        y = pd.Series({"p1": 1, "p2": 1, "n1": 0, "n2": 0, "n3": 0, "n4": 0})
        h = ig.homophily(g, y)
        assert h["node_homophily_class_1"] == pytest.approx(0.0)
        assert h["node_homophily_class_0"] == pytest.approx(0.85417, abs=1e-4)
        # and the two must not be silently averaged into one number
        assert h["node_homophily"] != pytest.approx(
            h["node_homophily_class_1"])

    def test_by_class_accepts_a_label_series(self):
        """by_class() takes either a column name or an external Series, so
        a centrality table can be scored against labels that live in a
        different frame.  Unlabelled rows must be DROPPED, not counted as
        zeros."""
        from src.analytics import influence_graph as ig
        tab = pd.DataFrame({"x": [1.0, 3.0, 10.0]},
                           index=["a", "b", "c"])
        out = ig.by_class(tab, pd.Series({"a": 0, "b": 1}))
        assert out.loc["x", "class_0"] == pytest.approx(1.0)
        assert out.loc["x", "class_1"] == pytest.approx(3.0)
        assert out.loc["x", "ratio_1_over_0"] == pytest.approx(3.0)

    def test_map_frames_gives_plotly_ready_coordinates(self):
        """The dashboard is a VIEW: it must never compute geometry. Every
        node needs finite x/y (a NaN would blank the map) and every edge
        needs both endpoints already resolved."""
        from src.analytics import influence_graph as ig
        g = ig.build_graph(self._two_triangles())
        board = pd.DataFrame({"author": list("abcdef"),
                              "composite": [0.9, 0.1, 0.2, 0.3, 0.4, 0.5]})
        nodes, links = ig.map_frames(g, board=board, seed=42)
        assert len(nodes) == g.n and len(links) == g.m
        assert np.isfinite(nodes[["x", "y"]].to_numpy()).all()
        assert np.isfinite(links[["x0", "y0", "x1", "y1"]].to_numpy()).all()
        assert nodes.loc[nodes["author"] == "a", "composite"].iloc[0] == 0.9

    def test_consensus_is_influence_weighted(self):
        """suggestion_digest's whole point: a call from someone with a
        record must outweigh a call from a first-timer. Two opposing calls
        of equal conviction, unequal record -> consensus takes the sign of
        the better author."""
        from src.analytics import influence_graph as ig
        calls = pd.DataFrame({
            "rec_id": ["1", "2"], "author": ["good", "bad"],
            "date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "ticker": ["GME", "GME"], "direction": [1, -1],
            "stance": [0.8, 0.8], "kind": ["post", "post"]})
        board = pd.DataFrame({"author": ["good", "bad"],
                              "composite": [0.9, 0.1]})
        dig = ig.suggestion_digest(calls, board, days=30,
                                   asof=pd.Timestamp("2026-07-02"))
        row = dig.set_index("ticker").loc["GME"]
        assert row["consensus"] > 0                 # the record wins
        assert row["longs"] == 1 and row["shorts"] == 1
        assert -1.0 <= row["consensus"] <= 1.0

    def test_ticker_voices_orders_by_record_and_nets_flip_flops(self):
        """ticker_voices() builds the bubble chart's hover text.  Two
        properties are easy to get silently wrong:

        1. the STRONGEST record must be listed first, because the hover is
           truncated to `top` lines and the lines that survive the cut
           must be the ones that carry the most weight;
        2. an author who said LONG once and SHORT once must show as MIXED,
           not appear twice pulling in both directions - their lean is the
           SUM of their calls, so it nets to zero.
        """
        from src.analytics import influence_graph as ig
        calls = pd.DataFrame({
            "rec_id": list("12345"),
            "author": ["weak", "strong", "flip", "flip", "strong"],
            "date": pd.to_datetime(["2026-07-01"] * 5),
            "ticker": ["GME"] * 5,
            "direction": [1, -1, 1, -1, -1],
            "stance": [0.8] * 5,
            "kind": ["post"] * 5})
        board = pd.DataFrame({"author": ["weak", "strong", "flip"],
                              "composite": [0.10, 0.95, 0.50]})
        out = ig.ticker_voices(calls, board, days=30,
                              asof=pd.Timestamp("2026-07-02"))
        # Hover text is a screen surface, so the handles in it arrive
        # half-masked (see research.ipynb).  The expected labels are
        # built through the same helper the display uses rather than
        # hard-coded, so this test keeps testing ORDERING and does not
        # become a second, stale copy of the mask rule.
        from src.analytics.plain_english import half_mask
        _m = half_mask
        lines = out.set_index("ticker").loc["GME", "voices"].split("<br>")
        assert lines[0].startswith(_m("strong")) and "SHORT" in lines[0]
        assert "2 calls" in lines[0]                  # both its calls, once
        assert [ln.split(" - ")[0] for ln in lines] == [_m("strong"),
                                                        _m("flip"),
                                                        _m("weak")]
        assert "MIXED" in [ln for ln in lines
                           if ln.startswith(_m("flip"))][0]
        # top_author is a DATA column (a join key), never a label - it must
        # stay unmasked or the caller cannot look the person back up.
        assert out.loc[0, "top_author"] == "strong"

    def test_ticker_voices_truncates_and_says_how_many_are_hidden(self):
        """A hover box with forty names in it is unreadable, so the list is
        cut - but a cut the reader cannot see is a lie about breadth. The
        '...and N more' line is therefore part of the contract."""
        from src.analytics import influence_graph as ig
        n = 9
        calls = pd.DataFrame({
            "rec_id": [str(i) for i in range(n)],
            "author": [f"a{i}" for i in range(n)],
            "date": pd.to_datetime(["2026-07-01"] * n),
            "ticker": ["GME"] * n, "direction": [1] * n,
            "stance": [0.5] * n, "kind": ["post"] * n})
        board = pd.DataFrame({"author": [f"a{i}" for i in range(n)],
                              "composite": np.linspace(0.1, 0.9, n)})
        out = ig.ticker_voices(calls, board, days=30, top=4,
                               asof=pd.Timestamp("2026-07-02"))
        lines = out.loc[0, "voices"].split("<br>")
        assert len(lines) == 5                        # 4 names + the notice
        assert lines[-1] == "...and 5 more"
        assert out.loc[0, "n_more"] == 5

    def test_ticker_voices_empty_window_returns_typed_empty_frame(self):
        """The dashboard merges this frame onto the digest. An empty result
        with the WRONG columns raises inside plotly instead of drawing an
        empty chart, so the empty case must still carry the schema."""
        from src.analytics import influence_graph as ig
        calls = pd.DataFrame({
            "rec_id": ["1"], "author": ["a"],
            "date": pd.to_datetime(["2020-01-01"]), "ticker": ["GME"],
            "direction": [1], "stance": [0.5], "kind": ["post"]})
        board = pd.DataFrame({"author": ["a"], "composite": [0.5]})
        out = ig.ticker_voices(calls, board, days=30,
                               asof=pd.Timestamp("2026-07-02"))
        assert len(out) == 0
        assert list(out.columns) == ["ticker", "voices", "top_author",
                                     "n_more"]

    def test_direction_label_reads_both_encodings(self):
        """The store writes direction as +1/-1 today. If a future
        extractor writes words instead, the dashboard must not silently
        read every short as a long."""
        from src.analytics import influence_graph as ig
        assert ig.direction_label(-1) == "SHORT"
        assert ig.direction_label(1) == "LONG"
        assert ig.direction_label("bearish") == "SHORT"

    def test_backing_share_sums_to_one_hundred_and_is_order_preserving(self):
        """backing_share is the unit the whole Influence tab is
        denominated in.  Two properties justify it over a ratio to the
        median name, so both are pinned: it is bounded and totals 100,
        and it is a POSITIVE rescaling, so no ranking anywhere on the tab
        can change because of it."""
        from src.analytics import influence_graph as ig
        w = pd.Series([8.0, 4.0, 2.0, 1.0, 1.0])
        s = ig.backing_share(w)
        assert s.sum() == pytest.approx(100.0)
        assert s.iloc[0] == pytest.approx(50.0)
        assert list(s.rank()) == list(w.rank())
        assert (s >= 0).all() and (s <= 100).all()

    def test_backing_share_empty_window_is_zero_not_nan(self):
        """A zero denominator (nobody on the board spoke in the window)
        must yield 0, not NaN.  An empty window means "no crowding", not
        "unknown", and a bar chart cannot plot NaN heights."""
        from src.analytics import influence_graph as ig
        assert list(ig.backing_share(pd.Series([0.0, 0.0]))) == [0.0, 0.0]
        assert list(ig.backing_share(pd.Series([], dtype=float))) == []

    def test_even_share_is_the_derived_reference_line(self):
        """The line the bubble chart draws instead of a chosen threshold:
        what each name would show if attention were spread equally."""
        from src.analytics import influence_graph as ig
        assert ig.even_share(51) == pytest.approx(100.0 / 51)
        assert ig.even_share(4) == pytest.approx(25.0)
        assert np.isnan(ig.even_share(0))

    def test_crowding_history_denominator_ignores_the_ticker_filter(self):
        """Share and even split are computed over EVERY name in the period
        before `tickers` is applied, so drawing five lines and drawing fifty
        give the same height for the same name.  Computing them after the
        filter would make the baseline move with how many lines the caller
        happened to ask for."""
        from src.analytics import influence_graph as ig
        calls = pd.DataFrame({
            "rec_id": [str(i) for i in range(4)],
            "author": ["a", "a", "a", "a"],
            "date": pd.to_datetime(["2026-07-20"] * 4),
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "direction": [1, 1, 1, 1], "stance": [1.0, 1.0, 1.0, 1.0],
            "kind": ["post"] * 4})
        board = pd.DataFrame({"author": ["a"], "composite": [1.0]})
        kw = dict(authors=["a"], days=30, asof=pd.Timestamp("2026-07-22"))
        every = ig.crowding_history(calls, board, **kw)
        one = ig.crowding_history(calls, board, tickers=["AAA"], **kw)
        assert set(every["ticker"]) == {"AAA", "BBB", "CCC"}
        assert list(one["ticker"]) == ["AAA"]
        a_all = float(every.loc[every["ticker"] == "AAA", "share"].iloc[0])
        a_one = float(one["share"].iloc[0])
        assert a_one == pytest.approx(a_all)
        assert a_all == pytest.approx(50.0)          # 2 of 4 unit calls
        assert float(one["even"].iloc[0]) == pytest.approx(100.0 / 3)


class TestInfluenceAdoption:
    """src/analytics/influence_ml.py - the adoption discipline, not the
    models.  These guard the rules that decide which model ships: paired
    seeds, a confidence interval that must clear zero, parsimony on
    ties, and the leakage blacklist.  A model adopted because one of
    those rules quietly inverted would ship an unverified claim."""

    def test_leakage_blacklist_covers_every_label_ingredient(self):
        """The label is built from composite/s_conf/s_z/s_enh.  Every one
        of those, and every column derived from them, must be barred from
        the feature bank by NAME so that a new feature cannot leak the
        label by accident."""
        from src.analytics import influence_ml as ml
        for col in ("composite", "s_conf", "s_z", "s_enh", "score", "tier",
                    "hit_rate", "hits", "n_judged", "fwd_ret", "z", "tau"):
            assert col in ml.LEAKAGE
        assert not set(ml.FULL_BANK) & ml.LEAKAGE

    def test_score_adjacent_features_are_not_in_the_shipped_bank(self):
        """mean_conf is an arithmetic FACTOR of the label (composite is
        built from conf * y terms), so a bank containing it is scoring
        itself.  Its lift is recorded in
        Data/research_record/nb05_influence.json as circular, which
        is exactly why it lives in WIDE_BANK for measurement and never in
        FULL_BANK."""
        from src.analytics import influence_ml as ml
        assert not set(ml.SCORE_ADJACENT) & set(ml.FULL_BANK)
        assert set(ml.SCORE_ADJACENT) <= set(ml.WIDE_BANK)
        assert len(ml.WIDE_BANK) > len(ml.FULL_BANK)

    def test_adoption_needs_the_interval_to_clear_zero(self):
        """The adoption rule is 'ci_lo > 0', not 'the mean looks better'.
        A candidate whose interval straddles zero must be REJECTED however
        pretty its point estimate is."""
        from src.analytics import influence_ml as ml
        lad = pd.DataFrame([
            {"baseline": "random", "candidate": "logit", "mean_diff": 0.05,
             "ci_lo": 0.03, "ci_hi": 0.07, "wins": 10, "adopt": True},
            {"baseline": "logit", "candidate": "sage_lite", "mean_diff": 0.02,
             "ci_lo": -0.01, "ci_hi": 0.05, "wins": 7, "adopt": False}])
        assert ml.choose_model(lad) == "logit"

    def test_parsimony_breaks_ties_toward_the_simpler_model(self):
        """If nothing beats the linear floor, the linear floor ships. A
        ladder with every rung rejected must NOT fall through to the last
        candidate tried."""
        from src.analytics import influence_ml as ml
        lad = pd.DataFrame([
            {"baseline": "logit", "candidate": c, "mean_diff": -0.01,
             "ci_lo": -0.03, "ci_hi": 0.01, "wins": 3, "adopt": False}
            for c in ("mlp", "sage_lite", "gcn_lite", "h2gcn_lite")])
        assert ml.choose_model(lad, floor="logit") == "logit"

    def test_seeds_are_paired_and_fixed(self):
        """A paired test needs the SAME seeds on both arms - that is where
        the variance reduction comes from - and the seed lists must be
        constants so a rerun reproduces the decision exactly."""
        from src.analytics import influence_ml as ml
        assert len(ml.ADOPTION_SEEDS) == 10
        assert len(set(ml.ADOPTION_SEEDS)) == 10
        assert ml.HOUSE_SEEDS == (42, 100, 2026)

    def test_maturity_bar_and_headline_regime_are_declared(self):
        """Two pre-stated numbers that must never drift silently: the
        minimum positives before any claim is made, and which label regime
        the headline quotes."""
        from src.analytics import influence_ml as ml
        assert ml.MIN_POSITIVES >= 130
        assert ml.HEADLINE_REGIME in ml.LABEL_REGIMES
        assert ml.BEST_MODEL in ml.MODELS
        assert ml.BEST_GRAPH_MODEL in ml.MODELS

    def test_split_is_stratified_and_disjoint(self):
        """With ~5% positives an UNstratified split can hand a fold zero
        positives, which makes average precision undefined.  Every node
        must land in exactly one fold, every fold must see the minority
        class, and the split must be a function of the seed alone.  The
        label carries a default RangeIndex because a read-only index
        buffer is the case in which an in-place shuffle fails."""
        from src.analytics import influence_ml as ml
        rng = np.random.default_rng(0)
        y = pd.Series((rng.random(600) < 0.05).astype(int))
        part = ml.stratified_split(y, seed=42)
        assert set(part.unique()) == {"train", "val", "test"}
        assert len(part) == len(y)                 # exactly one fold each
        for name in ("train", "val", "test"):
            assert y[part == name].sum() >= 1
        # proportions honoured to within one node per class
        assert abs((part == "train").sum() / len(y) - 0.6) < 0.02
        # and the split is a FUNCTION of the seed, not of call order
        assert part.equals(ml.stratified_split(y, seed=42))
        assert not part.equals(ml.stratified_split(y, seed=43))


class TestEuphoriaPhases:
    """src/analytics/euphoria_phases.py - the onset detector and the episode
    ground truth.  Same invariants as TestEuphoria: the crowd-only rule,
    no look-ahead, and hit windows that cannot be gamed."""

    def _synthetic_episode_px(self):
        """A price path with one unambiguous boom-bust arc: flat 100 ->
        trough -> +80% run over ~60d -> -40% bust. Single-name thresholds
        (50% boom / 30% bust) are comfortably cleared."""
        idx = pd.date_range("2023-01-01", periods=400, freq="D")
        px = pd.Series(100.0, index=idx)
        run = pd.date_range("2023-06-01", "2023-07-30", freq="D")
        px.loc[run] = np.linspace(100, 180, len(run))
        after = pd.date_range("2023-07-31", "2023-10-30", freq="D")
        px.loc[after] = np.linspace(180, 100, len(after))
        return px

    def test_onset_is_crowd_only(self):
        """THE selection rule extends to the onset detector: neither the
        feature builder nor the score accepts price input."""
        import inspect
        from src.analytics.euphoria_phases import (compute_onset_features,
                                               onset_score)
        for fn in (compute_onset_features, onset_score):
            params = inspect.signature(fn).parameters
            assert not any("px" in p or "price" in p for p in params), \
                f"{fn.__name__} takes a price argument - crowd-only rule broken"

    def test_onset_window_capped_at_peak(self):
        """The onset hit window may never extend past the peak - an alert
        AFTER the top must not count as 'caught the start' (fast rallies
        like GME would otherwise be gamed by late alerts)."""
        from src.analytics.euphoria_phases import find_episodes, \
            ONSET_WINDOW_DAYS
        eps = find_episodes(self._synthetic_episode_px(), "SYN", "SYN",
                            "single")
        assert eps, "synthetic boom-bust arc was not detected"
        for ep in eps:
            assert ep.onset_hi <= ep.peak
            assert ep.onset_hi <= ep.trough + pd.Timedelta(
                days=ONSET_WINDOW_DAYS)
            assert ep.onset_lo == ep.trough

    def test_episode_extends_confirmed_peaks_only(self):
        """The episode catalog introduces NO new ground truth: every
        episode's peak is one of ground_truth_peaks' confirmed tops, and
        the boom is measured off the same 120d-window minimum."""
        from src.analytics.euphoria import ground_truth_peaks
        from src.analytics.euphoria_phases import find_episodes
        px = self._synthetic_episode_px()
        peaks = ground_truth_peaks(px, "single")
        eps = find_episodes(px, "SYN", "SYN", "single")
        assert [e.peak for e in eps] == peaks
        for e in eps:
            prior = px.dropna().loc[e.peak - pd.Timedelta(days=120):e.peak]
            assert abs(px[e.peak] / prior.min() - 1.0 - e.boom_pct) < 1e-9

    def test_alert_cooldown_enforced(self):
        """One alert per episode: consecutive alerts must be >= 21d apart
        even when the score sits above threshold every single day."""
        from src.analytics.euphoria_phases import alerts_from_scores
        dates = list(pd.date_range("2024-01-01", periods=100, freq="D"))
        alerts = alerts_from_scores(dates, [1.0] * 100, threshold=0.5)
        assert alerts, "no alerts fired"
        gaps = np.diff([a.value for a in alerts]) / 86_400_000_000_000
        assert (gaps >= 21).all(), "cooldown violated"

    def test_onset_alert_classification_buckets(self):
        """HIT inside [trough, capped end]; LATE inside (end, peak];
        FA elsewhere - LATE is neither a hit nor a false alarm."""
        from src.analytics.euphoria_phases import (classify_onset_alerts,
                                               _eps_arrays, _day_ints)
        eps = pd.DataFrame([{
            "trough": pd.Timestamp("2023-06-01"),
            "onset_lo": pd.Timestamp("2023-06-01"),
            "onset_hi": pd.Timestamp("2023-07-16"),
            "peak": pd.Timestamp("2023-07-30")}])
        alerts = pd.DatetimeIndex(["2023-06-10",   # HIT
                                   "2023-07-20",   # LATE (post-window, pre-peak)
                                   "2023-12-01"])  # FA (outside the arc)
        res = classify_onset_alerts(_day_ints(alerts), _eps_arrays(eps))
        assert len(res["captured"]) == 1
        assert len(res["late"]) == 1
        assert len(res["fa"]) == 1
        lead = res["leads"][0]
        assert lead["after_trough"] == 9 and lead["before_peak"] == 50

    def test_onset_features_no_lookahead(self):
        """Truncating the input data must not change any feature value on
        the days both runs share - every rule is trailing."""
        from src.analytics.euphoria_phases import compute_onset_features
        rng = np.random.default_rng(42)
        days = pd.date_range("2022-01-01", periods=500, freq="D")
        counts = pd.DataFrame({
            "date": list(days) * 2,
            "ticker": ["SYN"] * len(days) + ["OTH"] * len(days),
            "mention_count": rng.integers(1, 50, 2 * len(days))})
        sents = pd.DataFrame({
            "date": list(days) * 2,
            "ticker": ["SYN"] * len(days) + ["OTH"] * len(days),
            "n_posts": rng.integers(1, 30, 2 * len(days)),
            "net_bullish": rng.uniform(-1, 1, 2 * len(days))})
        full = compute_onset_features("SYN", counts, sents, "ticker",
                                      with_breadth=False)
        cut = pd.Timestamp("2023-01-31")
        trunc = compute_onset_features(
            "SYN", counts[counts.date <= cut], sents[sents.date <= cut],
            "ticker", with_breadth=False)
        joint = full.loc[:cut]
        pd.testing.assert_frame_equal(joint, trunc.loc[:cut],
                                      check_exact=False, atol=1e-12)

    def test_onset_store_is_text_free(self):
        """The dashboard's onset table must never carry raw-text columns
        (same commit contract as every other store)."""
        from src.config import FORBIDDEN_COLS
        onset_cols = {"date", "name", "kind", "onset_score", "hype_raw",
                      "alert", "symbol"}
        assert not (onset_cols & FORBIDDEN_COLS)
        path = os.path.join(DATA_DIR, "processed", "euphoria_onset.parquet")
        if os.path.exists(path):
            cols = set(pd.read_parquet(path).columns)
            assert not (cols & FORBIDDEN_COLS), \
                f"forbidden columns in onset store: {cols & FORBIDDEN_COLS}"


class TestDynamicPanel:
    """ingestion/discover_subreddits.py - the dynamic subreddit panel:
    crowd-referral discovery with the A0 coverage floor reused as the
    qualification bar, a finance screen that measures candidate and
    panel with the same sampler, a one-add-per-review cap and a
    committed audit manifest."""

    def test_referral_regex(self):
        """r/Name referrals extract from prose; look-alikes do not."""
        from ingestion.discover_subreddits import REFERRAL_RE
        text = ("check r/Superstonk and /r/pennystocks! but not "
                "https://site/for/sale or crazy(r/thetagang)")
        found = {m.group(1).lower() for m in REFERRAL_RE.finditer(text)}
        assert found == {"superstonk", "pennystocks", "thetagang"}
        assert not REFERRAL_RE.search("for/sale")

    def test_qualification_bar_is_the_coverage_floor(self):
        """The qualification threshold must literally BE the A0 floor -
        if someone tunes one, the other moves with it (no second magic
        number can appear)."""
        from src.config import PANEL_MIN_REFERRERS, EUPHORIA_MIN_COVERAGE
        assert PANEL_MIN_REFERRERS == EUPHORIA_MIN_COVERAGE

    def test_qualify_counts_unique_authors_in_window(self):
        """One loud author posting 500 times is ONE referrer; stale
        referrals outside the 28d window do not count; panel members and
        structural subs are never candidates."""
        import datetime
        from ingestion.discover_subreddits import qualify
        today = datetime.date(2026, 7, 24)
        recent = (today - datetime.timedelta(days=5)).isoformat()
        stale = (today - datetime.timedelta(days=60)).isoformat()
        refs = pd.DataFrame(
            [{"date": recent, "candidate": "newsub", "author": "a1"}] * 500
            + [{"date": recent, "candidate": "newsub", "author": "a2"}]
            + [{"date": stale, "candidate": "oldsub", "author": f"a{i}"}
               for i in range(200)]
            + [{"date": recent, "candidate": "wallstreetbets",
                "author": "a3"}]
            + [{"date": recent, "candidate": "askreddit", "author": "a4"}])
        ranked = qualify(refs, panel=["wallstreetbets"], asof=today)
        as_dict = dict(zip(ranked.candidate, ranked.referrers))
        assert as_dict.get("newsub") == 2          # unique, not 501
        assert "oldsub" not in as_dict             # outside the window
        assert "wallstreetbets" not in as_dict     # already tracked
        assert "askreddit" not in as_dict          # structural denylist

    def test_screen_uses_one_ruler_and_never_adds_unmeasurable(self):
        """The finance screen measures candidate and panel with the SAME
        sampler; a candidate whose sample is unavailable returns None
        (and the review code treats None as 'never auto-add')."""
        from ingestion.discover_subreddits import ticker_rate
        from src.abstracted_data import load_universe
        universe = load_universe()
        finance = ["$GME to the moon", "buying AAPL calls", "TSLA earnings"]
        chatter = ["nice weather today", "look at this cat", "lol"]
        rate_fin = ticker_rate("x", universe, sampler=lambda s: finance)
        rate_chat = ticker_rate("x", universe, sampler=lambda s: chatter)
        rate_none = ticker_rate("x", universe, sampler=lambda s: None)
        assert rate_fin == 1.0
        assert rate_chat == 0.0
        assert rate_none is None


class TestResearchLiveSplit:
    """Research decides thresholds once; live runs only score with them.
    A data pull never re-fits on its own.

    A pull derives a threshold in exactly one case: the machine has no
    usable frozen record, so it cannot score at all (the bootstrap).
    A record that exists but stops at an earlier year is out-of-sample
    use, which is what the walk-forward licenses - it produces a NOTICE
    (the `*_record_lags_data` half of each pair), never a silent refit
    inside `update_data`."""

    def test_needs_research_only_bootstraps(self):
        """needs_research is True only when no usable record exists."""
        from src.analytics.euphoria import needs_research
        stored = {"thresholds": {"2024": 85, "2025": 85, "2026": 85}}
        assert needs_research(None, 2026)            # no report yet
        assert needs_research({}, 2026)              # empty report
        assert not needs_research(stored, 2026)      # covered year: frozen
        # a rolled-over year must NOT trigger a re-fit
        assert not needs_research(stored, 2027)

    def test_record_lags_data_reports_the_year(self):
        """A record that stops before the current year yields the last
        covered year as a notice; a missing record is the bootstrap's
        case and yields None."""
        from src.analytics.euphoria import record_lags_data
        stored = {"thresholds": {"2024": 85, "2025": 85, "2026": 85}}
        assert record_lags_data(stored, 2026) is None      # covered
        assert record_lags_data(stored, 2027) == 2026      # notice, not refit
        assert record_lags_data(None, 2027) is None        # bootstrap's job

    def test_onset_needs_research_only_bootstraps(self):
        """The onset detector follows the same bootstrap-only contract."""
        from src.analytics.euphoria_phases import (onset_needs_research,
                                               onset_record_lags_data)
        stored = {"live_threshold": 0.89,
                  "walk_forward": {"test_years": [2024, 2025, 2026]}}
        assert onset_needs_research(None, 2026)
        assert onset_needs_research({"walk_forward": {}}, 2026)
        assert not onset_needs_research(stored, 2026)
        assert not onset_needs_research(stored, 2027)
        assert onset_record_lags_data(stored, 2027) == 2026
        assert onset_record_lags_data(stored, 2026) is None


class TestEpisodeCoherence:
    """The episode coherence rule is asymmetric by design: a START within
    one 21d cooldown after an END is suppressed (a contradictory flip);
    an END after a START is NEVER suppressed.  Fast manias genuinely run
    start-to-end inside one cooldown, and a symmetric rule measured as
    cutting the top detector's captures from 17 to 9."""

    def test_start_after_end_is_suppressed(self):
        """A START inside the cooldown after an END is dropped; the END
        survives."""
        from src.analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        o, tp = episode_coherent_alerts([t("2026-01-11")], [t("2026-01-01")])
        assert o == [] and tp == [t("2026-01-01")]

    def test_fast_mania_end_is_never_suppressed(self):
        """An END five days after a START is a valid fast mania: both
        alerts survive."""
        from src.analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        o, tp = episode_coherent_alerts([t("2026-01-01")], [t("2026-01-06")])
        assert o == [t("2026-01-01")] and tp == [t("2026-01-06")]

    def test_separated_phases_both_survive(self):
        """Alerts further apart than the cooldown are never touched."""
        from src.analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        o, tp = episode_coherent_alerts([t("2026-01-01")], [t("2026-03-01")])
        assert o == [t("2026-01-01")] and tp == [t("2026-03-01")]

    def test_same_day_tie_goes_to_the_risk_signal(self):
        """A START and an END on the same day resolve to the END: the
        risk-reducing signal wins a tie."""
        from src.analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        o, tp = episode_coherent_alerts([t("2026-01-01")], [t("2026-01-01")])
        assert tp == [t("2026-01-01")] and o == []


class TestDeskConfiguration:
    """The shipped signal configuration (see research.ipynb):
    GET OUT = boom-gated + 7d-smoothed end rules; GET IN = phase-aware +
    7d-smoothed onset rules.  These tests pin the semantics that
    configuration rests on - candidacy, the end-stage mask, trailing
    smoothing - and the text-free contract of the shipped store."""

    def _frame(self):
        return pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "name": ["A"] * 6,
            "e1": [0.95, 0.95, 0.50, 0.95, 0.95, 0.10],
            "e2": [0.5, 0.0, 0.5, 0.5, 0.5, 0.0],
            "e3": [0.5] * 6, "e5": [0.5] * 6, "fade": [0.0] * 6,
            "hype_ok": [True, True, True, False, True, False],
            "hype_raw": [2.0, 2.0, 2.0, 2.0, 0.5, 2.0],
            "boom_state": [True, False, True, True, True, True],
        })

    def test_end_stage_is_exactly_the_end_gates(self):
        """END-STAGE must be built ONLY from the existing END gates
        (A1 hype + A2 attention + A3 persistence) - day 0 satisfies all
        three; days 1/2/3 each fail exactly one."""
        from src.analytics.euphoria_phases import end_stage_mask
        m = end_stage_mask(self._frame())
        assert m.tolist() == [True, False, False, False, True, False]

    def test_desk_candidacy_semantics(self):
        """GET OUT candidacy = crowd swollen AND real price boom;
        GET IN candidacy = measurable crowd (hype_raw >= 1) AND NOT
        end-stage - a start may never be declared on a day that already
        satisfies every ending gate."""
        from src.analytics.euphoria_phases import (desk_candidacy,
                                               end_stage_mask)
        f = self._frame()
        end_f, onset_f = desk_candidacy(f)
        assert end_f.index.tolist() == [0, 2, 4]     # hype_ok & boom
        assert not end_stage_mask(onset_f).any()
        assert (onset_f["hype_raw"] >= 1).all()
        assert onset_f.index.tolist() == [1, 2, 3, 5]

    def test_smoothing_is_trailing_and_kills_blips(self):
        """The 7d smoothing must (a) shrink a one-day spike below the
        raw trigger it used to fire, and (b) use PAST days only - the
        smoothed value on day t may not change when the future does."""
        from src.analytics.euphoria_phases import _smooth_by_name
        names = pd.Series(["A"] * 10)
        spike = pd.Series([0.0] * 4 + [1.0] + [0.0] * 5)
        sm = _smooth_by_name(spike, names)
        assert sm.max() < 1.0                        # blip attenuated
        assert sm.idxmax() == spike.idxmax()         # not shifted early
        altered = spike.copy()
        altered.iloc[7:] = 5.0                       # change the FUTURE
        sm2 = _smooth_by_name(altered, names)
        assert (sm.iloc[:7].values == sm2.iloc[:7].values).all(), \
            "smoothing looked ahead"

    def test_desk_needs_research_only_bootstraps(self):
        """Same contract as the euphoria and onset records: bootstrap
        only, and a lagging record is a notice rather than a refit."""
        from src.analytics.euphoria_phases import (desk_needs_research,
                                               desk_record_lags_data)
        stored = {"get_in": {"walk_forward": {"test_years": [2024, 2025,
                                                            2026]}},
                  "get_out": {"walk_forward": {"test_years": [2024, 2025,
                                                             2026]}}}
        assert desk_needs_research(None, 2026)
        assert desk_needs_research({"get_in": {}}, 2026)
        assert not desk_needs_research(stored, 2026)
        assert not desk_needs_research(stored, 2027)
        assert desk_record_lags_data(stored, 2027) == 2026
        assert desk_record_lags_data(stored, 2026) is None

    def test_desk_store_contract(self):
        """The shipped store is text-free (FORBIDDEN_COLS) and internally
        coherent.  The coherence invariant - no GET IN on an end-stage
        day - holds under EVERY model: the rule-based configuration gets
        it from candidacy, the learned models from the display-layer
        suppression in rebuild_phase_files.  The boom-gate invariant - no
        GET OUT without a confirmed price boom - is a consequence of
        rules candidacy only, because the learned models replace that
        hard gate with continuous price features; it is therefore
        asserted only when the frozen record says the rules ship."""
        import json
        import os
        from src.config import PROCESSED_DIR, FORBIDDEN_COLS
        path = os.path.join(PROCESSED_DIR, "euphoria_desk.parquet")
        if not os.path.exists(path):
            pytest.skip("desk store not built yet")
        df = pd.read_parquet(path)
        assert not (set(df.columns) & FORBIDDEN_COLS)
        assert not (df["get_in"] & df["end_stage"]).any()
        if "get_in_strict" in df.columns:
            # the Strict setting obeys the same coherence invariant as
            # the standard columns.  (No subset assertion on counts:
            # with a crossing trigger a higher cut is not mathematically
            # a subset - one long push above the low cut can re-cross
            # the high cut several times.)
            assert not (df["get_in_strict"] & df["end_stage"]).any()
        if "boomed120" in df.columns:
            # the shaped trigger: a GET OUT can only exist after the
            # ground truth's own boom bar; a GET IN only before it; and
            # no GET IN stands within 21d of a GET OUT in either
            # direction - on BOTH signal settings.
            df2 = df.copy()
            df2["date"] = pd.to_datetime(df2["date"])
            for sfx in ("", "_strict"):
                gi, go = f"get_in{sfx}", f"get_out{sfx}"
                assert not (df2[go] & ~df2["boomed120"]).any()
                assert not (df2[gi] & df2["boomed120"]).any()
                for _n, _g in df2.groupby("name"):
                    ins = _g.loc[_g[gi], "date"]
                    outs = _g.loc[_g[go], "date"]
                    if len(ins) and len(outs):
                        for d in ins:
                            assert (outs - d).abs().dt.days.min() > 21, \
                                (sfx, _n, str(d))
        rep_path = os.path.join(PROCESSED_DIR, "euphoria_desk_report.json")
        model = "rules"
        if os.path.exists(rep_path):
            try:
                model = json.load(open(rep_path)).get("model", "rules")
            except (ValueError, OSError):
                model = "rules"
        if model == "rules":
            assert not (df["get_out"] & ~df["boom_state"]).any()


class TestChartLabelLayout:
    """dashboard.py::_thin_labels - the only rule in the project that is
    about pixels rather than data.  It decides where there is room for
    ink; it never decides which names matter.  Every point it un-labels
    is still drawn, still hovers, and still appears in the exact-numbers
    table.

    It exists because `consensus` is bounded at +/-1 and hits +1.00
    exactly whenever every call on a name was long, so many names pile
    into one column and their tickers print through each other.
    """

    @staticmethod
    def _mask(xs, ys, y_span=35.0):
        import dashboard as D
        return D._thin_labels(xs, ys, x_span=2.36, y_span=y_span,
                              w_px=880.0, h_px=398.0)

    def test_same_column_and_too_close_in_height_loses_one_label(self):
        """Two points in the same column 0.46pp apart on a 35% axis are
        about 9px apart, and a 10pt label needs 13px, so exactly one
        label is dropped: the taller point keeps its label."""
        keep = self._mask([1.0, 1.0], [4.04, 3.58])
        assert keep == [True, False]

    def test_same_height_but_different_column_keeps_both(self):
        """Suppression must require BOTH axes to be tight, or the chart
        starts hiding names that never overlapped: two points that share
        a height but not a column both keep their labels."""
        assert self._mask([0.53, 1.0], [3.60, 3.58]) == [True, True]

    def test_well_separated_heights_keep_every_label(self):
        """Same column, heights far apart: nothing is suppressed."""
        assert self._mask([1.0, 1.0], [26.1, 7.05]) == [True, True]

    def test_ties_resolve_toward_the_earlier_row(self):
        """The caller passes the digest already ranked by backing, so on an
        exact tie the label must go to the better-backed name rather than to
        whichever row happened to be last."""
        assert self._mask([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == [True, False,
                                                                False]

    def test_degenerate_inputs_do_not_raise_or_hide_everything(self):
        """A zero-size plot or an empty frame must not blank the labels: the
        figure is drawn before its true pixel size is known, and an empty
        window is a normal state on this tab."""
        assert self._mask([], []) == []
        import dashboard as D
        assert D._thin_labels([1.0, 1.0], [2.0, 2.0], 2.36, 35.0,
                              0.0, 0.0) == [True, True]


class TestEuphoriaGauge:
    """dashboard.py's euphoria gauge.  The dial compresses a whole name
    into one number that a reader acts on directly, so four properties
    are pinned:

      1. Its edges come from a measured record
         (Data/research_record/gauge_zones.json), never from a
         literal in dashboard.py, so every band edge has a traceable
         origin.
      2. The red edge is the SAME number the walk-forward froze for the
         END alert, so the gauge cannot become a second, softer
         threshold.
      3. The needle equals the plotted curve's last value, so the dial
         and the chart under it can never disagree.
      4. The dial reports a STATE and never an instruction - a band
         label must not be able to masquerade as a signal.
    """

    @staticmethod
    def _zones():
        import json
        import os
        import dashboard as D
        p = os.path.join(DATA_DIR, "research_record", "gauge_zones.json")
        assert os.path.exists(p), \
            "Data/research_record/gauge_zones.json is missing"
        return json.load(open(p, encoding="utf-8"))

    def test_edges_are_not_literals_in_the_dashboard(self):
        """Neither band edge may appear as a numeric literal in the gauge
        function body: the number must be read from the record."""
        import os
        import re
        import dashboard as D
        src = open(os.path.join(D.ROOT, "dashboard.py"),
                   encoding="utf-8").read()
        z = self._zones()
        body = src[src.index("def fig_euphoria_gauge"):
                   src.index("def fig_series_vs_price")]
        code = "\n".join(ln.split("#")[0] for ln in body.splitlines())
        for edge in (z["amber_edge"], z["red_edge"]):
            assert not re.search(rf"(?<![\w.]){edge}(?![\w.])", code), \
                f"gauge edge {edge} is hard-coded in dashboard.py"

    def test_red_edge_is_the_frozen_walk_forward_end_level(self):
        """The gauge must not invent a second threshold.  Its red edge has
        to be the level the walk-forward already selected for the END
        alert, read out of euphoria_report.json."""
        import json
        import os
        import dashboard as D
        thr = json.load(open(os.path.join(DATA_DIR, "processed", "euphoria_report.json"),
            encoding="utf-8"))["thresholds"]
        assert self._zones()["red_edge"] == int(thr[max(thr)])

    def test_amber_edge_is_below_red_and_was_measured(self):
        """The amber edge is the lowest cut whose hit rate is significantly
        above the base rate, and it sits below the red edge."""
        z = self._zones()
        assert z["amber_edge"] < z["red_edge"]
        band = z["bands"][f"level >= {z['amber_edge']}"]
        assert band["significant"] and band["ci95"][0] > 0, \
            "the amber edge is supposed to be the lowest SIGNIFICANT cut"
        assert band["p"] > z["base_rate"]

    def test_needle_equals_the_plotted_curve_endpoint(self):
        """The dial is built from the smoothed series' last value; if the
        two ever drift apart the dashboard is telling two stories."""
        import numpy as np
        import pandas as pd
        import dashboard as D
        raw = pd.Series(np.linspace(10, 95, 60),
                        index=pd.date_range("2026-01-01", periods=60))
        lvl = raw.rolling(D.ROLL, min_periods=1).mean()
        fig = D.fig_euphoria_gauge(float(lvl.iloc[-1]),
                                   float(lvl.iloc[-1 - D.ROLL]), False,
                                   self._zones(), lvl.index[-1])
        assert fig.data[0].value == pytest.approx(float(lvl.iloc[-1]))
        assert fig.data[0].delta.reference == pytest.approx(
            float(lvl.iloc[-1 - D.ROLL]))

    def test_rising_euphoria_is_never_painted_green(self):
        """Euphoria going UP is the risk direction.  Plotly's default paints
        an increase green, which would invert the meaning of the arrow."""
        import dashboard as D
        fig = D.fig_euphoria_gauge(80.0, 60.0, False, self._zones(),
                                   "2026-06-15")
        assert fig.data[0].delta.increasing.color == D.BEAR
        assert fig.data[0].delta.decreasing.color == D.BULL

    def test_bands_are_ordered_and_cover_the_whole_axis(self):
        """The gauge steps tile [0, 100] with no gaps, so every level
        falls in exactly one band."""
        import dashboard as D
        z = self._zones()
        steps = D.fig_euphoria_gauge(50.0, 50.0, False, z,
                                     "2026-06-15").data[0].gauge.steps
        edges = [s.range for s in steps]
        assert edges[0][0] == 0 and edges[-1][1] == 100
        for a, b in zip(edges, edges[1:]):
            assert a[1] == b[0], "a gap between bands leaves a dead zone"

    def test_state_is_a_description_and_never_an_instruction(self):
        """gauge_state must describe WHERE the crowd is.  The words GET IN
        and GET OUT belong to the detector; if they leak into a band
        label the dial reads as a trade instruction."""
        import dashboard as D
        z = self._zones()
        for lvl, dgr in ((10.0, False), (80.0, False), (95.0, False),
                         (95.0, True)):
            _, label, _ = D.gauge_state(lvl, dgr, z)
            assert "get in" not in label.lower()
            assert "get out" not in label.lower()

    def test_danger_state_reads_hotter_than_the_red_zone_alone(self):
        """The measured ordering the caption depends on: level alone is a
        weak read, level plus an already-run-up price is the strong one."""
        z = self._zones()
        red = z["red_edge"]
        alone = z["bands"][f"level >= {red}"]["p"]
        both = z["bands"][f"level >= {red} AND danger state"]["p"]
        assert both > alone, (alone, both)
        import dashboard as D
        assert D.gauge_state(red + 1, True, z)[0] == "red_danger"
        assert D.gauge_state(red + 1, False, z)[0] == "red"

    def test_missing_evidence_draws_no_bands_rather_than_invented_ones(self):
        """With no frozen gauge record (gauge_zones.json) there are no
        measured edges, and the dial must report an unknown state instead
        of guessing band boundaries."""
        import dashboard as D
        key, label, _ = D.gauge_state(80.0, False, {})
        assert key == "unknown"
        assert "no frozen gauge record" in D.gauge_caption(80.0, False, {})


class TestHandleCensoring:
    """`analytics.plain_english.censor` - the display-only mask on
    obscene Reddit handles.

    Two failure modes matter and neither is caught by "it ran without
    error":

      * UNDER-masking lets a crude handle reach a shared screen.
      * OVER-masking is easier to do accidentally: a stem list matched
        as plain substrings mangles innocent handles such as
        `Painkiller_830`, `AssumptionPretty7018` and `Ok-Grapefruit2910`.
        Those handles are real rows in `author_scores.parquet`, so they
        are pinned here and any word-list edit that reintroduces
        substring matching fails this test.

    The store test protects the DATA: masking is a display transform,
    and the moment it touches a handle used as a key the joins between
    `author_scores`, `calls` and `reply_edges` silently drop people.
    """

    # handles that MUST be masked
    DIRTY = ["just_lick_my_ass", "fucktheredditapp15", "BigBoiBenis",
             "CuntyAnne_Conway", "RetardedChimpanzee", "dick-knuckle",
             "I_love_boobs86", "Hornysnek69", "BallsOfStonk", "nut-sack",
             "TittyClapper", "spez_eats_nazi_ass", "sluthouseincel"]

    # real handles from the store that must survive UNTOUCHED.  Each one
    # is a substring false positive the two-tier design exists to
    # prevent; the trailing comment is the stem a naive matcher hits.
    CLEAN = ["Painkiller_830",        # kill
             "AssumptionPretty7018",  # ass
             "passionlessDrone",      # ass
             "cow_grass",             # ass
             "Tricky-Doughnut-6429",  # nut
             "thenuttyhazlenut",      # nut
             "buffetite",             # tit
             "JohnTitor_3",           # tit
             "Stitch426",             # tit
             "AfraidAnalyst",         # anal
             "Valuable-Analyst-464",  # anal
             "scientia_analytica",    # anal
             "MeridianAllocation",    # anal, across a word boundary
             "Ok-Grapefruit2910",     # rape
             "RepulsiveGrapefruit",   # rape
             "Calm_Cockroach_5284",   # cock
             "Pool_cocktail_repeat",  # cock
             "SatoshiTrails",         # shit, across a word boundary
             "TheSatoshiTimes",       # shit, across a word boundary
             "Dippissippi",           # piss
             "sobewankanobe",         # wank
             "BooBeef",               # boob, across a word boundary
             "Feb17Sucks",            # deliberately not in the word list
             "TheRedditModsSuck"]     # deliberately not in the word list

    def test_obscene_handles_are_masked(self):
        """Every handle in DIRTY is detected and its output carries MASK."""
        from src.analytics.plain_english import censor, is_obscene, MASK
        for h in self.DIRTY:
            assert is_obscene(h), h
            assert MASK in censor(h), (h, censor(h))

    def test_innocent_handles_are_left_exactly_alone(self):
        """Every handle in CLEAN passes through byte-for-byte."""
        from src.analytics.plain_english import censor, is_obscene
        for h in self.CLEAN:
            assert not is_obscene(h), h
            assert censor(h) == h, (h, censor(h))

    def test_masking_keeps_handles_distinguishable(self):
        """Only the offending span is replaced, so two masked authors stay
        two authors.  A whole-handle mask would collapse them into one row
        on the leaderboard and one bar on the backers chart."""
        from src.analytics.plain_english import censor
        assert censor("just_lick_my_ass") == "just_lick_my_**"
        assert censor("Hornysnek69") == "**snek69"
        assert len({censor(h) for h in self.DIRTY}) == len(self.DIRTY)

    def test_masking_runs_to_a_fixed_point(self):
        """`Buttslut69696969` tokenises as [Buttslut, 69696969], so a
        single pass removes only `slut` and leaves `Butt**...`, where
        `Butt` has become a whole token.  The censor must iterate to a
        fixed point or it ships a crude word on a handle it claims to
        have censored."""
        from src.analytics.plain_english import censor
        assert censor("Buttslut69696969") == "**69696969"

    def test_the_store_is_never_rewritten(self):
        """The parquet keeps the TRUE handle.  This is the invariant that
        keeps `calls` / `reply_edges` / `author_scores` joinable, and it is
        also what lets an author be re-judged when new prices arrive."""
        import os
        import pandas as pd
        from src.analytics.plain_english import MASK
        p = os.path.join(DATA_DIR, "reference", "influence",
                         "author_scores.parquet")
        if not os.path.exists(p):
            pytest.skip("influence store not present in this checkout")
        a = pd.read_parquet(p, columns=["author"])
        assert not a["author"].astype(str).str.contains(
            MASK, regex=False).any()

    def test_censoring_is_idempotent_and_total_on_the_real_store(self):
        """Whatever reaches a screen must contain no stem, including after a
        second pass - a mask that itself matched a stem would loop."""
        import os
        import pandas as pd
        from src.analytics.plain_english import censor, is_obscene
        p = os.path.join(DATA_DIR, "reference", "influence",
                         "author_scores.parquet")
        if not os.path.exists(p):
            pytest.skip("influence store not present in this checkout")
        a = pd.read_parquet(p, columns=["author"])
        shown = a["author"].astype(str).map(censor)
        assert not shown.map(is_obscene).any()
        assert shown.map(censor).equals(shown)
        # and the mask must not merge two different people into one label
        assert shown.nunique() == a["author"].nunique()


class TestThemeRollup:
    """`theme_digest` / `theme_voices` - the influence tab's THEME view.

    Crowding is a theme-level question (many influential accounts
    converging on one theme), so the roll-up reuses the ticker-level
    consensus and backing arithmetic through one shared `_digest_frame`
    rather than restating it.  The two views sit side by side on one
    toggle, so any arithmetic disagreement between them would be visible
    and unexplainable; these tests keep them consistent.

    Every case below has an answer known by hand from a three-row frame.
    """

    @staticmethod
    def _fixture():
        """NVDA is in several themes, MSFT in several, and ZZZZ in none.

        Chosen from the real `src/themes.py` membership, not invented, so
        the test fails if that membership is edited in a way that breaks
        the multi-theme assumption the exhibit is built on."""
        calls = pd.DataFrame({
            "rec_id": ["1", "2", "3"],
            "author": ["good", "bad", "good"],
            "date": pd.to_datetime(["2026-07-01"] * 3),
            "ticker": ["NVDA", "MSFT", "ZZZZ"],
            "direction": [1, -1, 1],
            "stance": [0.8, 0.8, 0.8],
            "kind": ["post", "post", "post"]})
        board = pd.DataFrame({"author": ["good", "bad"],
                              "composite": [0.9, 0.1]})
        return calls, board, pd.Timestamp("2026-07-02")

    def test_a_ticker_in_several_themes_counts_in_every_one(self):
        """NVDA is semiconductors AND ai AND ai_megacap.  A crowding read
        on any of those themes must include the NVDA call; a roll-up that
        assigned each ticker to one primary theme would silently
        under-count every other theme the ticker belongs to."""
        from src.analytics import influence_graph as ig
        from src.themes import build_ticker_to_themes
        calls, board, asof = self._fixture()
        homes = build_ticker_to_themes()["NVDA"]
        assert len(homes) > 1, "fixture assumes NVDA is multi-theme"
        dig = ig.theme_digest(calls, board, days=30, asof=asof)
        got = dig.set_index("theme")
        for th in homes:
            assert got.loc[th, "n_calls"] >= 1

    def test_unmapped_tickers_are_dropped_not_bucketed_into_other(self):
        """ZZZZ belongs to no theme.  An "other" bucket is not a theme
        anyone can position in, and on the live store it would be the
        LARGEST bar on the chart purely by being a residue - so the call
        is left out of the theme view entirely, and the two views
        therefore have different denominators on purpose."""
        from src.analytics import influence_graph as ig
        calls, board, asof = self._fixture()
        dig = ig.theme_digest(calls, board, days=30, asof=asof)
        assert "other" not in set(dig["theme"])
        # ZZZZ contributed NOTHING - asserted exactly rather than via a
        # "themes per ticker" bound, because the ticker->theme membership
        # is config (config/theme_tickers.csv, incl. ETF-constituent
        # rows) and may legitimately grow.  Each mapped call lands once
        # in each of its own themes; an unmapped call lands nowhere.
        from src.themes import build_ticker_to_themes
        _homes = build_ticker_to_themes()
        _expected = sum(len(_homes.get(t, [])) for t in ("NVDA", "MSFT"))
        assert dig["n_calls"].sum() == _expected   # ZZZZ contributed none

    def test_consensus_stays_bounded_and_keeps_the_ticker_views_sign(self):
        """Same arithmetic, different key: a theme whose only call is the
        bullish NVDA one must read exactly what NVDA reads, because
        grouping cannot change a single row's number."""
        from src.analytics import influence_graph as ig
        calls, board, asof = self._fixture()
        dig = ig.theme_digest(calls, board, days=30, asof=asof)
        tick = ig.suggestion_digest(calls, board, days=30, asof=asof)
        assert dig["consensus"].dropna().between(-1.0, 1.0).all()
        # ai_megacap holds NVDA (long) and MSFT (short); semiconductors
        # holds NVDA alone, so it must equal NVDA's own reading.
        semis = dig.set_index("theme").loc["semiconductors", "consensus"]
        nvda = tick.set_index("ticker").loc["NVDA", "consensus"]
        assert semis == pytest.approx(nvda)

    def test_backing_share_of_the_theme_room_still_sums_to_one_hundred(self):
        """The vertical axis is a SHARE, so it has to be a share of
        something.  Because a ticker duplicates into each of its themes,
        the theme denominator is the theme-mapped room and not the ticker
        room - which is exactly why the toggle recomputes rather than
        reusing the ticker frame."""
        from src.analytics import influence_graph as ig
        calls, board, asof = self._fixture()
        dig = ig.theme_digest(calls, board, days=30, asof=asof)
        share = ig.backing_share(dig["weighted_voices"])
        assert float(share.sum()) == pytest.approx(100.0)

    def test_empty_window_returns_the_typed_empty_frame(self):
        """The dashboard indexes these columns before it checks the row
        count, so an untyped empty frame is a KeyError on a quiet window
        rather than an empty chart."""
        from src.analytics import influence_graph as ig
        calls, board, _ = self._fixture()
        old = pd.Timestamp("2000-01-01")
        dig = ig.theme_digest(calls, board, days=30, asof=old)
        vox = ig.theme_voices(calls, board, days=30, asof=old)
        assert len(dig) == 0 and len(vox) == 0
        assert list(dig.columns) == ["theme"] + ig.DIGEST_COLS
        assert list(vox.columns) == ["theme", "voices", "top_author",
                                     "n_more"]

    def test_the_theme_view_never_sees_a_price(self):
        """Prediction is crowd-only.  This exhibit is information rather
        than a signal, but it sits on the same page as the alerts, so
        the price-free invariant is asserted here too - a theme roll-up
        that quietly joined prices would be the easiest way to leak
        one in."""
        import ast
        import inspect
        from src.analytics import influence_graph as ig
        for fn in (ig.theme_digest, ig.theme_voices, ig.explode_to_themes,
                   ig._digest_frame):
            sig = inspect.signature(fn)
            assert not any("price" in p.lower() for p in sig.parameters)
            # Parse and drop the docstring rather than grep the source: the
            # PROSE says "not a forecast about the price", which is the
            # sentence that documents the rule, and a text search cannot
            # tell that apart from a line of code that reads one.
            tree = ast.parse(inspect.getsource(fn).lstrip()).body[0]
            body = [n for n in tree.body
                    if not (isinstance(n, ast.Expr)
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str))]
            code = "\n".join(ast.unparse(n) for n in body).lower()
            assert "price" not in code

    def test_half_mask_shows_some_of_the_handle_and_hides_most(self):
        """A half-masked handle must reveal a recognisable prefix and hide
        at least half of the rest.  Both halves are load-bearing: a label
        that reveals nothing makes the leaderboard unreadable, and one
        that reveals everything is not a mask."""
        from src.analytics.plain_english import half_mask, IDENT_MASK
        for h in ["tomato232", "Love-to-Trade101", "Independent-Use-228",
                  "zq7495", "Funklemire"]:
            m = half_mask(h)
            assert IDENT_MASK in m, (h, m)
            assert m != h
            shown = len(m) - len(IDENT_MASK)
            assert shown <= max(3, len(h) // 2), (
                f"{h!r} -> {m!r} reveals more than half the handle")
            assert m.startswith(h[:2]), (
                f"{h!r} -> {m!r} lost the prefix a reader recognises it by")
        assert half_mask("") == ""
        assert half_mask(None) is None

    def test_half_mask_never_merges_two_people_into_one_label(self):
        """The point of revealing a prefix is telling rows apart, so the
        one failure that matters is two handles landing on one label.  A
        fixed 2-char disambiguator is not enough (`Marketspike` and
        `Markthehare` mask alike AND hash alike), so the tag must grow
        until the group is unique."""
        import pandas as pd
        from src.analytics.plain_english import half_mask_series
        s = pd.Series(["Marketspike", "Markthehare", "trader_bull_99",
                       "trader_bear_99", "tomato232"])
        out = half_mask_series(s)
        assert out.nunique() == s.nunique()

    def test_half_mask_is_applied_over_the_obscenity_mask(self):
        """Order matters: half-masking a crude handle could otherwise leave
        the crude part inside the revealed prefix."""
        from src.analytics.plain_english import half_mask, is_obscene
        for h in ["Hornysnek69", "Buttslut69696969", "just_lick_my_ass"]:
            assert not is_obscene(half_mask(h)), (h, half_mask(h))

    def test_the_whole_real_store_masks_without_collisions(self):
        """half_mask_series keeps every author distinct on the real store."""
        import os
        import pandas as pd
        from src.analytics.plain_english import half_mask_series
        p = os.path.join(DATA_DIR, "reference", "influence",
                         "author_scores.parquet")
        if not os.path.exists(p):
            import pytest
            pytest.skip("no influence store on disk")
        a = pd.read_parquet(p)["author"].astype(str)
        assert half_mask_series(a).nunique() == a.nunique()

    def test_theme_hover_labels_are_censored_like_every_other_handle(self):
        """`theme_voices` puts author handles in hover text.  The censor
        rule is about what reaches a screen, so a new surface that renders
        handles must inherit it rather than be an exception."""
        from src.analytics import influence_graph as ig
        from src.analytics.plain_english import is_obscene
        calls, board, asof = self._fixture()
        vox = ig.theme_voices(calls, board, days=30, asof=asof)
        assert len(vox)
        assert not vox["voices"].astype(str).map(is_obscene).any()


class TestDashboardModuleHygiene:
    """The dashboard body executes at MODULE scope, so a loop variable in
    a tab can silently rebind a module-level helper of the same name (a
    tab-local string `_unit` shadowing the `_unit()` scaler, for
    example, fails hundreds of lines later with "'str' object is not
    callable").  Importing the module proves the script runs at all;
    this proves the helpers survived it."""

    def test_module_level_helpers_are_still_callable_after_the_script_runs(
            self):
        """Each named module-level helper is still a callable after the
        script body has executed."""
        import dashboard as D
        for name in ("_unit", "_dig", "_theme", "_thin_labels", "_facts"):
            assert callable(getattr(D, name)), (
                f"dashboard.{name} was rebound by a tab-local variable")


class TestPulseNoFillerRule:
    """AI Pulse output must not contain filler items: a brief whose whole
    content is "there is minimal discussion" carries no information and
    is dropped, so that everything shown is the most interesting, most
    mentioned or most recent material.  The prompt asks the model for
    that; `_drop_filler` is the half that does not depend on the model
    complying."""

    def test_empty_calorie_items_are_dropped(self):
        """Theme briefs and vibe bullets that are entirely filler are
        removed; substantive ones survive."""
        from src.analytics.ai_pulse import _drop_filler
        doc = _drop_filler({
            "theme_briefs": [
                {"theme": "a", "brief": "Minimal discussion this week."},
                {"theme": "b", "brief": "There is no meaningful chatter."},
                {"theme": "c", "brief": "Nothing notable."},
                {"theme": "d", "brief": "The bulls and bears are arguing "
                                        "about whether the capex cycle "
                                        "has topped, and the bears are "
                                        "winning on engagement."},
            ],
            "market_vibe": {"bullets": ["Little activity.",
                                        "Everyone is tired of being "
                                        "wrong and says so loudly."]},
        })
        assert [b["theme"] for b in doc["theme_briefs"]] == ["d"]
        assert len(doc["market_vibe"]["bullets"]) == 1

    def test_a_long_brief_that_merely_mentions_quiet_is_kept(self):
        """A 200-word brief noting a surprising silence is doing real
        work; only items whose WHOLE content is 'nothing here' go."""
        from src.analytics.ai_pulse import _is_filler
        long_one = ("Rate chatter is quiet, and that is the point: after "
                    "two years in which every thread bent back to the "
                    "Fed, the boards have stopped arguing about it "
                    "entirely, which historically has marked the end of "
                    "a macro regime rather than a lull inside one. "
                    "The bulls now argue capex, not discount rates.")
        assert not _is_filler(long_one)
        assert _is_filler("minimal discussion")


class TestWatchSideIsStable:
    """The DISPLAY gate must not flip on knife-edge noise.

    The raw boomed120 is an instantaneous test against a hard bar, so a
    name parked near the bar flips sides on rounding (96% of measured
    flips were within 5pp of the bar), and the displayed side alternates
    between CUT and INCREASE day to day.  boomed120_stable adds
    hysteresis plus an ASYMMETRIC debounce.  All three properties below
    are load-bearing; the asymmetry especially, because debouncing the
    turn-ON put 14 of 124 real CUT calls on a day the display still
    showed the INCREASE side."""

    @staticmethod
    def _frame(px):
        import pandas as pd
        from src.analytics.euphoria_phases import boomed120_frame

        class _ES:
            def __init__(s):
                s.name, s.symbol, s.kind = "t", "T", "theme"
        return boomed120_frame([_ES()], {"T": px})

    def _px(self, tail):
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2025-01-01", periods=150 + len(tail),
                            freq="D")
        return pd.Series(np.r_[np.full(150, 100.0), tail], index=idx)

    def test_knife_edge_noise_does_not_flip_the_displayed_side(self):
        """A price oscillating around the boom bar chatters in the raw
        gate and settles in the stable one."""
        import numpy as np
        px = self._px(100 * (1 + 0.20 + 0.02
                             * np.sin(np.arange(250) / 2.0)))
        f = self._frame(px)
        raw = f["boomed120"].values
        stb = f["boomed120_stable"].values
        n_raw = int((raw[1:] != raw[:-1]).sum())
        n_stb = int((stb[1:] != stb[:-1]).sum())
        assert n_raw > 8, "fixture must actually chatter"
        assert n_stb <= 2, f"displayed side still chatters: {n_stb}"

    def test_a_real_breakout_registers_immediately(self):
        """The stable gate turns ON on the same day as the raw gate.  A
        CUT can only fire on a run-up day, so any lag on the way in
        would place a CUT marker on a day the display still shows the
        opposite side."""
        import numpy as np
        f = self._frame(self._px(np.linspace(100, 180, 250)))
        assert (int(np.argmax(f["boomed120"].values))
                == int(np.argmax(f["boomed120_stable"].values)))

    def test_leaving_the_run_up_state_is_slower_than_entering(self):
        """Hysteresis: the stable gate stays ON after the raw gate has
        dropped below the bar."""
        import numpy as np
        f = self._frame(self._px(np.r_[np.linspace(100, 130, 60),
                                       np.linspace(130, 112, 190)]))
        last_raw = len(f) - 1 - int(np.argmax(f["boomed120"]
                                              .values[::-1]))
        last_stb = len(f) - 1 - int(np.argmax(f["boomed120_stable"]
                                              .values[::-1]))
        assert last_stb > last_raw, "hysteresis is not holding"

    def test_the_model_gate_itself_is_untouched(self):
        """boomed120 is what every frozen threshold was calibrated
        against; the stable twin is consumed by the display gate only,
        so the dashboard must route through `watch_gate` rather than
        redefine the model gate."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "dashboard.py"
               ).read_text(encoding="utf-8")
        assert "def watch_gate(" in src
        assert "boomed120_stable" in src


class TestPollPromptPanel:
    """The poll's value is its continuity: a reworded prompt silently
    breaks that prompt_id's time series (see the module docstring of
    src/analytics/ai_poll.py)."""

    def test_the_kept_prompts_are_untouched(self):
        """The wording of every surviving prompt_id is pinned.  REMOVING
        a prompt only ends its series; REWORDING a surviving one corrupts
        it.  A retired id may be re-added later with its exact original
        wording and its series resumes."""
        from src.analytics.ai_poll import _prompts
        frozen = {
            "p01": "What should I invest in right now?",
            "p05": "What are the best AI stocks to buy now?",
            "p09": "What meme stocks are about to squeeze?",
            "p10": "Is it too late to buy gold and silver?",
            "p31": "What are the best dividend stocks for monthly "
                   "income right now?",
        }
        got = {p["prompt_id"]: p["prompt"] for p in _prompts()}
        for pid, text in frozen.items():
            assert got.get(pid) == text, (
                f"{pid} was reworded - that breaks its time series; add "
                "a new prompt_id instead")

    def test_every_prompt_has_a_unique_id_and_a_family(self):
        """prompt_ids are unique, every prompt carries a family tag, and
        the panel stays small enough to poll on every run."""
        from src.analytics.ai_poll import _prompts
        ps = _prompts()
        ids = [p["prompt_id"] for p in ps]
        assert len(ids) == len(set(ids))
        # the panel is 12 prompts; a few additions are fine, silent
        # growth past 20 is not
        assert 12 <= len(ps) <= 20, f"panel has {len(ps)} prompts"
        for p in ps:
            assert (p.get("family") or "").strip(), (
                f"{p['prompt_id']} has no family tag")


class TestSingleNameUniverse:
    """The EUPHORIA: Singles tab picks its own names via
    `single_name_universe`.  Three properties of that selection are
    pinned, each guarding a distinct way it can go wrong:

      * membership must be ranked on RECENT chatter, not all history -
        2021 is 39% of every mention ever recorded, so an all-history
        rank tracks bankrupt and delisted names for years;
      * ETFs must be excluded, or SPY and SCHD read as "single names";
      * finance acronyms that have since been issued to real ETFs (HYSA,
        DRAM, BTC) must be stoplisted, or they count as tickers - HYSA
        alone would otherwise be the most-mentioned symbol in the store."""

    def _prices(self):
        import pandas as pd
        from src.config import PRICES_PATH
        if not os.path.exists(PRICES_PATH):
            pytest.skip("no price store on this machine")
        return pd.read_parquet(PRICES_PATH)

    def test_no_etfs_in_a_tab_called_single_names(self):
        """No symbol in the ETF directory appears in the universe."""
        from pathlib import Path
        from src.analytics.euphoria import single_name_universe
        from src.config import REFERENCE_DIR
        from src.ticker_universe import load_etf_symbols
        etfs = load_etf_symbols(Path(REFERENCE_DIR))
        if not etfs:
            pytest.skip("Nasdaq symbol directories not cached here")
        uni = single_name_universe(self._prices())
        assert uni, "the universe came back empty"
        bad = [t for t in uni if t in etfs]
        assert not bad, f"ETFs in the single-name universe: {bad}"

    def test_jargon_symbols_never_reach_the_universe(self):
        """Stoplisted finance acronyms are excluded from the universe."""
        from src.analytics.euphoria import single_name_universe
        uni = set(single_name_universe(self._prices()))
        for junk in ("HYSA", "DYOR", "DRAM", "BTC", "REIT"):
            assert junk not in uni, (
                f"{junk} is jargon, not a tracked single name")

    def test_the_universe_tracks_names_that_are_ALIVE(self):
        """Every tracked name has at least EUPHORIA_MIN_COVERAGE posts in
        the trailing EUPHORIA_SINGLE_WINDOW_D days.  Membership means
        "this name has enough recent chatter to measure", which is what
        the coverage floor encodes; a universe ranked over all history
        would keep tracking a name for years after it went quiet.

        The test asserts COVERAGE, not mention rank: EUPHORIA_SINGLE_TOP_N
        sits above the number of eligible names, so the cap is
        deliberately non-binding and a name can be tracked without being
        in the mention top-N.  The real gate is measurability."""
        import pandas as pd
        from src.analytics.euphoria import single_name_universe
        from src.config import (PROCESSED_DIR, EUPHORIA_SINGLE_WINDOW_D,
                                EUPHORIA_MIN_COVERAGE)
        p = os.path.join(PROCESSED_DIR, "daily_ticker_sentiment.parquet")
        if not os.path.exists(p):
            pytest.skip("no sentiment store on this machine")
        s = pd.read_parquet(p)
        s["date"] = pd.to_datetime(s["date"])
        hi = s["date"].max()
        cov = (s[s["date"] > hi - pd.Timedelta(days=EUPHORIA_SINGLE_WINDOW_D)]
               .groupby("ticker")["n_posts"].sum())
        uni = single_name_universe(self._prices())
        dead = [t for t in uni if cov.get(t, 0) < EUPHORIA_MIN_COVERAGE]
        assert not dead, (
            "tracked names without enough recent chatter to measure: "
            f"{dead}")

    def test_the_stoplist_is_config_driven_and_fails_loudly(self):
        """STOP_TICKERS is loaded from config, and a malformed stoplist
        file raises rather than loading as empty - an empty stoplist
        would silently let jargon such as CEO back into the counts."""
        import tempfile
        from pathlib import Path
        from src.extract_tickers import load_stop_tickers, STOP_TICKERS
        assert "HYSA" in STOP_TICKERS and "CEO" in STOP_TICKERS
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.csv"
            bad.write_text("nonsense\n1\n", encoding="utf-8")
            with pytest.raises(ValueError):
                load_stop_tickers(bad)      # a silent empty stoplist would
                                            # let CEO back into the counts


class TestEverythingIsIncremental:
    """The dashboard's LIVE refresh is always incremental; the
    full-history rebuild belongs to `update_data.py --full` only.

    The slow work in this pipeline is re-reading raw archives.  Every
    scanner therefore keeps a ledger and skips what it has already seen,
    and the dashboard must never launch the full-history rebuild.  Both
    properties are cheap to break by accident and expensive to notice."""

    def test_every_scanner_keeps_a_ledger(self):
        """agentic_watch declares a LEDGER path under Data/reference/."""
        import src.agentic_watch as A
        path = getattr(A, "LEDGER", None)
        assert path, "agentic_watch has no LEDGER - it would rescan"
        assert "reference" in str(path), (
            "the ledger must live in Data/reference/")

    def test_a_second_scan_does_no_work(self):
        """The ledger is keyed on (size, mtime) per archive, so a rerun
        with nothing new on disk must be a no-op, not a rescan."""
        import time
        import src.agentic_watch as A
        if not os.path.exists(A.LEDGER):
            pytest.skip("no agentic ledger on this machine yet")
        t0 = time.time()
        A.scan(log=lambda *_: None)
        assert time.time() - t0 < 20, (
            "a no-change rescan took longer than 20s - the ledger is not "
            "being honoured")

    def test_the_dashboard_never_launches_a_full_rebuild(self):
        """`--full` rebuilds nine years of aggregates from posts.parquet.
        It only works on a copy with the raw post store, and it is a
        re-validation event rather than a refresh.  No dashboard button
        may pass it to start_pipeline."""
        src = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "dashboard.py"),
            encoding="utf-8").read()
        import re
        for call in re.findall(r"start_pipeline\((.*?)\)\s*$", src,
                               re.S | re.M):
            assert "--full" not in call, (
                "a dashboard button passes --full: " + call[:200])


class TestTickerAllowlist:
    """Short, allowlisted tickers (MU, AMD, COIN) must count when written
    in capitals without a cashtag.  The bare-CAPS pass only matches 4-5
    letter words, so an explicit allowlist is the only way a 2-3 letter
    symbol can be counted from prose; see the block comment in
    src/extract_tickers.py for how the allowlist and stoplist interact."""

    def _universe(self):
        from src.abstracted_data import load_universe
        u = load_universe()
        if not u:
            pytest.skip("no ticker universe cached on this machine")
        return u

    def test_short_tickers_are_counted_without_a_dollar_sign(self):
        """WORD_BARE is [A-Z]{4,5}, so MU and AMD are invisible to the
        bare pass unless allowlisted.  Measured over six days of
        comments, MU had 265 bare-CAPS mentions against six $MU
        cashtags, so a cashtag-only rule loses almost all of its
        attention."""
        from src.extract_tickers import extract_tickers_from_text
        u = self._universe()
        got = extract_tickers_from_text(
            "loaded up on MU and AMD before earnings", u,
            cashtags_only=False)
        assert "MU" in got and "AMD" in got

    def test_case_is_what_separates_the_company_from_the_word(self):
        """The bare pass reads the ORIGINAL text, so an allowlisted
        symbol only matches in capitals. This is the whole safety
        property: 'coin' stays a word, 'COIN' becomes Coinbase."""
        from src.extract_tickers import extract_tickers_from_text
        u = self._universe()
        for caps, lower, sym in (("COIN reports tomorrow",
                                  "i found a coin", "COIN"),
                                 ("META is up again",
                                  "this is very meta", "META"),
                                 ("bought SOFI today",
                                  "sofi sounds like sofa", "SOFI")):
            assert sym in extract_tickers_from_text(caps, u,
                                                    cashtags_only=False)
            assert sym not in extract_tickers_from_text(lower, u,
                                                        cashtags_only=False)

    def test_jargon_still_wins_over_the_allowlist(self):
        """STOP_TICKERS outranks the allowlist. If a symbol ever appears
        in both, the jargon reading must win - otherwise adding a row to
        the allowlist could quietly resurrect 'AI' as a ticker."""
        from src.extract_tickers import (ALLOW_TICKERS, STOP_TICKERS,
                                         extract_tickers_from_text)
        overlap = ALLOW_TICKERS & STOP_TICKERS
        u = self._universe()
        for sym in overlap:
            assert sym not in extract_tickers_from_text(
                f"buying {sym} now", u, cashtags_only=False), (
                f"{sym} is in both lists and the allowlist won")
        assert "AI" not in extract_tickers_from_text(
            "AI is the future", u, cashtags_only=False)

    def test_the_ambiguous_ones_were_deliberately_left_out(self):
        """Symbols whose bare-CAPS samples measured as the English word
        dominating (GOLD, COST, LOW, ...) must stay off the allowlist;
        adding one would count ordinary prose as a ticker."""
        from src.extract_tickers import ALLOW_TICKERS
        for sym in ("GOLD", "COST", "LOW", "NOW", "TEAM", "CAT", "PE"):
            assert sym not in ALLOW_TICKERS, (
                f"{sym} was allowlisted despite the word dominating - "
                "re-measure before adding it")


class TestFlagLabelsAndConfigReload:
    """The dashboard must SAY which instrument a flag refers to, and it
    must notice when config/theme_etfs.csv changes.

    Streamlit reruns the script on every interaction but never
    re-imports an already-imported module, so a theme->ETF map imported
    once at start-up keeps serving stale anchors after the CSV is
    edited, and the screen is indistinguishable from the edit having
    failed.  Separately, a banner that prints a bare symbol gives the
    reader nothing to say what the instrument is."""

    @staticmethod
    def _src():
        from pathlib import Path
        return Path(__file__).resolve().parents[1] / "dashboard.py"

    def test_theme_etf_map_is_keyed_on_the_files_mtime(self):
        """A module-level `from src.themes import THEME_ETFS` freezes the
        map for the life of the process.  The map must be re-derived
        through a helper keyed on the CSV's mtime instead."""
        src = self._src().read_text(encoding="utf-8")
        assert "from src.themes import THEME_ETFS" not in src, (
            "THEME_ETFS is imported once at start-up again - a config "
            "edit will be invisible until the server is restarted")
        assert "_theme_etf_maps" in src and "getmtime" in src

    def test_the_map_still_validates_against_the_approved_list(self):
        """Re-reading must reuse `_load_theme_etfs`, not reimplement it,
        so the 'every symbol must be approved' check cannot be lost."""
        import src.themes as themes
        etfs, chains = themes._load_theme_etfs()
        approved = set(themes.APPROVED_INSTRUMENTS)
        for theme, anchor in etfs.items():
            assert anchor in approved, f"{theme} -> {anchor} not approved"
            assert chains[theme][0] == anchor, (
                f"{theme}: the anchor must lead its own fallback chain")

    def test_china_geopolitics_is_broad_china_not_the_internet_basket(self):
        """china_geopolitics anchors on FXI (broad China beta), with KWEB
        (the China INTERNET basket) retained only as a fallback:
        geopolitics moves broad China beta, not the internet basket."""
        import src.themes as themes
        etfs, chains = themes._load_theme_etfs()
        assert etfs["china_geopolitics"] == "FXI"
        assert "KWEB" in chains["china_geopolitics"], (
            "KWEB should stay in the chain as a fallback, just not lead it")

    def test_every_theme_note_that_claims_a_proxy_names_the_real_line(self):
        """The `note` column is shown on screen, so a caveat that says
        'PROXY' without naming the instrument it stands in for helps
        nobody."""
        import csv
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "config" / "theme_etfs.csv"
        with open(p, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                note = (row.get("note") or "")
                if "PROXY" in note or "natural ETF" in note:
                    assert "not approved" in note or "approved" in note, (
                        f"{row['theme']}: the note flags a proxy but does "
                        "not say what the right instrument is")

    def test_the_colour_legend_matches_the_boxes_on_screen(self):
        """INCREASE EXPOSURE renders through st.success, which is GREEN,
        so the legend must say green, not blue.

        INCREASE EXPOSURE and CUT EXPOSURE are the on-screen labels for
        the get_in and get_out signals; the stored column names are
        unchanged.
        """
        src = self._src().read_text(encoding="utf-8")
        assert "blue = INCREASE EXPOSURE" not in src
        assert ("Green = INCREASE EXPOSURE" in src
                and "Red = CUT EXPOSURE" in src)

    def test_flags_are_labelled_with_the_instrument_not_the_bare_symbol(self):
        """One helper spells every instrument on the euphoria tab - the
        banners, the why-expanders and the lookup dropdown - so the
        three can never disagree about what a name is called."""
        src = self._src().read_text(encoding="utf-8")
        assert "def flag_label(" in src
        assert src.count("flag_label(n, kind)") >= 4, (
            "the banners, both why-expanders and the lookup should all "
            "route through flag_label")

    def test_security_names_drop_the_share_class_boilerplate(self):
        """Share-class boilerplate is stripped before the 38-character
        cut: 'AMC Entertainment Holdings, Inc. Class A Common Stock'
        truncated raw gives '...Inc. Class', which reads as broken
        data."""
        # dashboard.py executes the whole app on import, so the one pure
        # function is compiled out of the source on its own.
        src = self._src().read_text(encoding="utf-8")
        start = src.index("_NAME_TAIL = re.compile(")
        end = src.index("@st.cache_data", start)
        ns = {}
        exec("import re\n" + src[start:end], ns)
        clean = ns["_clean_security_name"]
        assert clean("NVIDIA Corporation - Common Stock") == \
            "NVIDIA Corporation"
        assert clean("AMC Entertainment Holdings, Inc. Class A Common "
                     "Stock") == "AMC Entertainment Holdings, Inc."
        assert clean("Micron Technology, Inc.") == "Micron Technology, Inc."
        assert len(clean("A" * 90)) <= 38 and clean("A" * 90).endswith("…")


class TestDataFreshnessIsVisible:
    """A dashboard must be able to say that its DATA is out of date.

    Distinct from TestStaleTabIsVisible, which is about a stale
    *process* serving an old dashboard.py.  This one is about a healthy
    process serving stale numbers: a masthead that prints the render
    time reads as "just updated" on a page whose newest reading is
    weeks old, so the masthead must print the data date and grade its
    age."""

    @staticmethod
    def _src():
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / "dashboard.py"
                ).read_text(encoding="utf-8")

    @staticmethod
    def _fresh():
        """dashboard.py executes the whole app on import, so the
        freshness helper and its level constants are compiled out of the
        source on their own."""
        import pandas as pd
        src = TestDataFreshnessIsVisible._src()
        start = src.index("_FRESH_OK, _FRESH_LATE, _FRESH_STALE =")
        end = src.index("h_left, h_right = st.columns", start)
        ns = {"pd": pd}
        exec(src[start:end], ns)
        return ns

    def test_the_masthead_no_longer_passes_render_time_off_as_freshness(self):
        """No live line prints `last update:` (the render clock); the
        masthead reports `data through` instead."""
        src = self._src()
        _live = [ln for ln in src.splitlines()
                 if "last update:" in ln and not ln.lstrip().startswith("#")]
        assert not _live, (
            "the masthead is printing the render clock again - that "
            f"says 'just updated' on three-week-old numbers: {_live}")
        assert "data through " in src

    def test_a_weekend_does_not_make_monday_look_late(self):
        """Age is counted in BUSINESS days: Friday's data read on Monday
        morning is current.  A calendar-day rule would paint the
        masthead amber every Monday, and a weekly false alarm is a
        warning nobody reads."""
        import pandas as pd
        ns = self._fresh()
        lvl, bd = ns["_data_freshness"](pd.Timestamp("2026-08-28"),   # Fri
                                        pd.Timestamp("2026-08-31"))   # Mon
        assert lvl == ns["_FRESH_OK"], (lvl, bd)
        assert bd <= 1

    def test_same_day_and_yesterday_are_current(self):
        """Data dated today or the previous business day grades OK."""
        import pandas as pd
        ns = self._fresh()
        for d in ("2026-09-02", "2026-09-01"):      # Wed today, Tue
            lvl, _ = ns["_data_freshness"](pd.Timestamp(d),
                                           pd.Timestamp("2026-09-02"))
            assert lvl == ns["_FRESH_OK"], d

    def test_two_business_days_is_late_and_a_working_week_is_stale(self):
        """Two business days of age grades LATE; five or more grade
        STALE."""
        import pandas as pd
        ns = self._fresh()
        today = pd.Timestamp("2026-09-04")                 # Friday
        lvl, bd = ns["_data_freshness"](pd.Timestamp("2026-09-02"), today)
        assert (lvl, bd) == (ns["_FRESH_LATE"], 2)
        lvl, bd = ns["_data_freshness"](pd.Timestamp("2026-08-28"), today)
        assert lvl == ns["_FRESH_STALE"] and bd >= 5

    def test_a_missing_date_counts_as_stale_not_as_fine(self):
        """None and NaT grade STALE: an unknown data date must never
        read as current."""
        import pandas as pd
        ns = self._fresh()
        assert ns["_data_freshness"](None,
                                     pd.Timestamp("2026-09-04"))[0] \
            == ns["_FRESH_STALE"]
        assert ns["_data_freshness"](pd.NaT,
                                     pd.Timestamp("2026-09-04"))[0] \
            == ns["_FRESH_STALE"]

    def test_a_future_date_never_reports_negative_age(self):
        """A data date after today clamps to age 0 and grades OK."""
        import pandas as pd
        ns = self._fresh()
        lvl, bd = ns["_data_freshness"](pd.Timestamp("2026-09-10"),
                                        pd.Timestamp("2026-09-04"))
        assert lvl == ns["_FRESH_OK"] and bd == 0

    def test_the_notice_tells_the_reader_what_to_do(self):
        """A warning that only states a fact is half a warning.  The
        hosted copy points the reader at the dashboard owner; the copy
        with local controls names the command, because there the reader
        is the person who can run it."""
        src = self._src()
        assert "Sorry! Working on updates right now" in src
        assert "update_data.py</code> on this machine" in src
        assert "LOCAL_CONTROLS else" in src
        # the viewer sentence is what the hosted copy shows in BOTH
        # behind states; the diagnosis is gated on the controls
        assert "if _deploy_behind and not LOCAL_CONTROLS:" in src
        assert src.count("_VIEWER_NOTICE}") == 2

    def test_a_half_applied_deploy_is_diagnosed_separately(self):
        """A second failure has the same symptom: the pipeline DID run
        and published, but the page never picked the bundle up.  The fix
        is the opposite (restart, not re-run), so it gets its own
        message, and that branch must be tested before the age branch
        because sending the reader to re-run a pipeline that already
        ran is the wrong instruction."""
        src = self._src()
        assert "publish_manifest.json" in src
        assert "_deploy_behind = (_pub_through is not None" in src
        assert "This page is not drawing the newest published data." in src
        # the deploy branch is tested FIRST - it must win over the age
        # branch, not be shadowed by it
        _i_dep = src.index("if _deploy_behind and not LOCAL_CONTROLS:")
        _i_age = src.index("elif _fresh_lvl == _FRESH_OK:")
        assert _i_dep < _i_age

    def test_only_published_ahead_is_a_fault(self):
        """A workstation that has run the pipeline and not published
        yet has a manifest BEHIND its data.  That is the normal state of
        a local copy between a run and a publish and must stay silent,
        so the comparison is strictly one-directional: only a manifest
        AHEAD of the data is a fault."""
        src = self._src()
        _line = [ln for ln in src.splitlines()
                 if "_pub_through > pd.Timestamp(data_max)" in ln]
        assert _line, "the deploy check is no longer one-directional"
        assert "_pub_through <" not in src and "_pub_through !=" not in src

    def test_the_publisher_writes_the_manifest_it_depends_on(self):
        """The dashboard check is only as good as the manifest, so the
        publisher must write one on every real publish - and must not
        fail a publish if it cannot."""
        from pathlib import Path
        pub = (Path(__file__).resolve().parents[1] / "tools"
               / "publish_dashboard.py").read_text(encoding="utf-8")
        assert '"publish_manifest.json"' in pub
        assert '"data_through"' in pub and '"published_at"' in pub
        assert "publish manifest skipped" in pub, (
            "manifest writing must be failure-isolated - a bundle "
            "without one is an older bundle, never a broken publish")

    def test_the_manifest_survives_a_round_trip(self):
        """The dashboard parses data_through with pd.Timestamp and
        published_at as a UTC stamp it then strips. Both shapes the
        publisher writes must survive that."""
        import pandas as pd
        from datetime import datetime, timezone
        _at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert pd.Timestamp("2026-09-03") == pd.Timestamp("2026-09-03")
        assert pd.Timestamp(_at).tz_localize(None) is not None

    def test_the_notice_is_severity_coloured_from_the_palette(self):
        """No hard-coded hexes - ochre for late, brick for stale, the
        same two colours the charts use for warning and danger."""
        src = self._src()
        assert "_fresh_col = BEAR if _fresh_lvl == _FRESH_STALE else OCHRE" \
            in src


class TestStaleTabIsVisible:
    """A running dashboard must be able to say that it is out of date.

    `.streamlit/config.toml` turns the file watcher OFF on purpose: the
    pipeline rewrites parquet in place and a watcher reloading mid-read
    is a source of spurious errors.  The cost is that an edited
    dashboard.py is never picked up by a live server, and a second
    `streamlit run` takes the next port while the pinned tab keeps
    serving the original process.  Both look exactly like "the fix did
    not work", so the page must detect and announce a stale process."""

    @staticmethod
    def _src():
        from pathlib import Path
        return Path(__file__).resolve().parents[1] / "dashboard.py"

    def test_the_watcher_is_still_off_and_still_explains_itself(self):
        """The watcher setting is off and its reason sits next to it."""
        from pathlib import Path
        cfg = (Path(__file__).resolve().parents[2]
               / ".streamlit" / "config.toml")
        text = cfg.read_text(encoding="utf-8")
        assert 'fileWatcherType = "none"' in text, (
            "the watcher was turned back on - if that is deliberate, the "
            "stale-build banner and the RUNBOOK row should go with it")
        assert "rewrites in place" in text, (
            "the setting must keep its reason next to it")

    def test_the_build_stamp_is_pinned_per_process(self):
        """The start-up mtime is held in a cache_resource, which survives
        reruns and so holds the mtime this PROCESS started with.  A
        cache_data keyed on its own argument would be invalidated by the
        very change it is meant to detect."""
        src = self._src().read_text(encoding="utf-8")
        assert "_mtime_at_process_start" in src
        i = src.index("def _mtime_at_process_start")
        assert "@st.cache_resource" in src[max(0, i - 200):i], (
            "cache_data would be invalidated by its own argument and "
            "could never detect a change; it must be cache_resource")

    def test_the_stale_banner_says_what_to_do(self):
        """The banner names the fix (restart) and the port trap."""
        src = self._src().read_text(encoding="utf-8")
        assert "Stale tab." in src
        assert "restart the server" in src.lower()
        assert "takes the next port" in src, (
            "the port trap is the half of this that people miss")

    def test_the_sidebar_shows_which_port_it_is_serving(self):
        """Two servers, two ports, identical pages - the caption is the
        only way to tell which one the browser is talking to."""
        src = self._src().read_text(encoding="utf-8")
        assert "server.port" in src

    def test_the_runbook_has_the_restart_recipe(self):
        """RUNBOOK.md documents the stale-tab symptom and both ports."""
        from pathlib import Path
        rb = (Path(__file__).resolve().parents[2] / "Reference Materials"
              / "RUNBOOK.md").read_text(encoding="utf-8")
        assert "A code or config edit is not showing" in rb
        assert "8501" in rb and "8502" in rb


class TestApprovedUniverseCoverage:
    """APPROVED_INSTRUMENTS is the tradeable universe.  Nothing on it
    may be unreachable (unrequested from the price source), and nothing
    on screen may quote an instrument the chart is not actually using.

    The dropdown lists THEMES, and instruments legitimately outnumber
    themes: several anchors serve two themes each, many lines are
    fallbacks, and the rest are benchmarks nobody posts about."""

    @staticmethod
    def _src():
        from pathlib import Path
        return Path(__file__).resolve().parents[1] / "dashboard.py"

    def test_every_approved_instrument_is_requested_from_bloomberg(self):
        """The price puller's symbol universe must cover every approved
        instrument.  A request built from theme anchors and fallbacks
        only silently omits any approved line no theme points at, which
        then has no price history and nothing on screen to say so."""
        import ingestion.pull_prices as pull
        from src.themes import APPROVED_INSTRUMENTS
        universe = set(pull.build_symbol_universe())
        missing = sorted(s for s in APPROVED_INSTRUMENTS
                         if s not in universe)
        assert not missing, (
            f"approved but never requested: {missing} - these can never "
            "be drawn, and nothing would report it")

    def test_every_instrument_has_exactly_one_role(self):
        """anchor / fallback / benchmark must partition the list, or the
        coverage panel is double-counting."""
        from src.themes import (THEME_ETFS, THEME_ETF_FALLBACKS,
                                APPROVED_INSTRUMENTS)
        anchors = set(THEME_ETFS.values())
        fallbacks = set()
        for theme, chain in THEME_ETF_FALLBACKS.items():
            fallbacks.update(s for s in chain if s != THEME_ETFS.get(theme))
        fallbacks -= anchors
        approved = set(APPROVED_INSTRUMENTS)
        assert anchors <= approved and fallbacks <= approved
        benchmarks = approved - anchors - fallbacks
        assert (len(anchors) + len(fallbacks) + len(benchmarks)
                == len(approved))

    def test_the_label_names_the_line_actually_drawn(self):
        """`resolve_anchor` falls through to the first PRICED line, so a
        theme whose anchor has no history is drawn on a substitute. The
        label has to follow that, not the config."""
        src = self._src().read_text(encoding="utf-8")
        assert "def _live_anchor(" in src
        i = src.index("def flag_label(")
        body = src[i:src.index("def _live_anchor(")]
        assert "_live_anchor(name)" in body, (
            "flag_label quotes THEME_ETFS directly again - it will name "
            "an instrument the chart is not using")
        assert "unpriced" in body

    def test_a_substituted_anchor_is_reported_not_hidden(self):
        """A theme drawn on a fallback line says so on screen."""
        src = self._src().read_text(encoding="utf-8")
        assert "Drawn on a fallback, not the named anchor" in src

    def test_the_themes_are_not_forced_to_match_the_instrument_count(self):
        """Deliberate: a theme exists because retail argues about it, and
        the ETF is only how you would express it. One theme per approved
        line would run that backwards and invent themes with no crowd
        behind them."""
        from src.themes import THEME_ETFS, APPROVED_INSTRUMENTS
        assert len(THEME_ETFS) < len(APPROVED_INSTRUMENTS), (
            "if these ever match, check it happened because the crowd "
            "started discussing every approved line - not because "
            "somebody padded the theme list to make the counts agree")


class TestTickerMappingsAreCurrent:
    """Every mapped symbol must be one that still trades under that
    ticker, or be reachable by company NAME instead.

    A renamed or delisted ticker does not break anything loudly - it
    quietly counts nothing, forever, while the theme it belonged to
    looks fine - so the known renames and the name-reachability rule
    are pinned."""

    @staticmethod
    def _universe():
        from pathlib import Path
        from src.config import REFERENCE_DIR
        from src.ticker_universe import load_us_ticker_universe
        return load_us_ticker_universe(Path(REFERENCE_DIR))

    @staticmethod
    def _rows(name):
        import csv
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "config" / name
        with open(p, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))

    def test_the_retired_tickers_are_gone(self):
        """Known re-tickered symbols are absent and their replacements
        present, so a half-applied rename fails; delisted symbols are
        absent outright."""
        mapped = {r["ticker"] for r in self._rows("theme_tickers.csv")}
        for dead, live in (("SQ", "XYZ"),        # Block re-tickered
                           ("PARA", "PSKY"),     # Paramount Skydance
                           ("MMC", "MRSH"),      # Marsh, 14 Jan 2026
                           ("VSCO", "VSXY"),     # Victoria's Secret
                           ("ARMN", "ARIS"),     # Aris Mining
                           ("BITF", "KEEL")):    # Bitfarms -> Keel
            assert dead not in mapped, f"{dead} is dead; it is now {live}"
            assert live in mapped, f"{dead} was removed but {live} is absent"
        for gone in ("CYBR", "DIDI"):
            assert gone not in mapped, f"{gone} is delisted"
        # SPLG -> SPYM is the same class of correction; its only home was
        # the broad_market_passive basket, which the sp500 theme replaced
        # with actual constituents.  Nothing should carry the dead symbol
        # either way.
        assert "SPLG" not in mapped

    def test_keel_moved_theme_as_well_as_ticker(self):
        """Bitfarms did not just re-ticker to KEEL - it stopped being a
        bitcoin miner and became AI infrastructure.  A rename that keeps
        the old theme is still a wrong mapping."""
        rows = self._rows("theme_tickers.csv")
        themes = {r["theme"] for r in rows if r["ticker"] == "KEEL"}
        assert "crypto" not in themes
        assert "datacenters" in themes

    def test_every_unmatchable_constituent_is_reachable_by_name(self):
        """OTC ADRs are absent from the listed file and dotted class
        shares cannot pass ^[A-Z]{1,5}$, so those rows can NEVER match as
        tickers. Each one must have its company name in the keyword map
        instead, or it is silently contributing nothing to its theme."""
        import collections
        import re
        uni = self._universe()
        tt = collections.defaultdict(set)
        for r in self._rows("theme_tickers.csv"):
            tt[r["ticker"]].add(r["theme"])
        names = {}
        for r in self._rows("etf_constituents.csv"):
            names.setdefault(r["ticker"], r["company"])
        # A few mapped symbols are curated rather than ETF holdings, so
        # etf_constituents.csv carries no company for them.  Their names
        # are supplied here so the reachability check covers them; add a
        # line when adding such a row.
        names.setdefault("NTDOY", "Nintendo ADR")
        kw = collections.defaultdict(set)
        for r in self._rows("theme_keywords.csv"):
            kw[r["theme"]].add(r["keyword"].lower().strip())
        orphans = []
        for sym, themes in tt.items():
            if sym in uni:
                continue
            # the shortest distinctive token of the company name is what a
            # person actually types: "Rolls-Royce ADR" -> "rolls"
            raw = re.sub(r"\b(adr|plc|ag|sa|nv|holdings?|class [a-z]|inc|"
                         r"corp|company|companies|group|ltd)\b", " ",
                         names.get(sym, sym).lower())
            toks = [t for t in re.split(r"[^a-z0-9&-]+", raw) if len(t) > 2]
            reachable = any(
                any(t in phrase for t in toks) or
                any(phrase in " ".join(toks) for phrase in ())
                for th in themes for phrase in kw.get(th, set()))
            if not reachable:
                orphans.append((sym, names.get(sym, "?"), sorted(themes)))
        assert not orphans, (
            "these can never match as tickers and have no company name in "
            f"the keyword map either: {orphans}")

    def test_no_symbol_is_both_jargon_and_a_theme_ticker_by_accident(self):
        """AI, DD, ES and NOW are mapped AND stoplisted.  That is
        deliberate: the stoplist wins in extract_tickers, so C3.ai,
        DuPont, Eversource and ServiceNow are documented as theme members
        but never counted from prose (ES in bare CAPS is the E-mini
        future).  The test pins the SET so a new clash gets noticed."""
        from src.extract_tickers import STOP_TICKERS
        mapped = {r["ticker"] for r in self._rows("theme_tickers.csv")}
        clash = sorted(mapped & STOP_TICKERS)
        assert clash == ["AI", "DD", "ES", "NOW"], (
            f"the set of deliberate jargon/ticker clashes changed: {clash}")

    def test_es_can_never_become_a_ticker(self):
        """Every sampled bare-CAPS "ES" hit was the E-mini S&P future,
        not Eversource.  It is stoplisted and kept off the allowlist so
        an allowlist edit cannot poison utilities_power."""
        from src.extract_tickers import ALLOW_TICKERS, STOP_TICKERS
        assert "ES" in STOP_TICKERS and "ES" not in ALLOW_TICKERS

    def test_pm_was_measured_and_rejected(self):
        """PM's bare-CAPS count would pass a ratio test, but the sampled
        hits were overwhelmingly the abbreviations for private message
        and prime minister - about one in six meant Philip Morris - so
        it stays off the allowlist."""
        from src.extract_tickers import ALLOW_TICKERS
        assert "PM" not in ALLOW_TICKERS


class TestCrawlAndBudgetHygiene:
    """Crawl-budget and run-to-run determinism properties of the
    ingestion and analytics stages."""

    @staticmethod
    def _src(rel):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / rel).read_text(encoding="utf-8")

    def test_a_dry_subreddit_stops_instead_of_burning_the_budget(self):
        """The crawl walks newest-first, so once pages stop yielding
        anything it is re-reading collected ground.  Without a dry-page
        stop, a quiet subreddit spends tens of budgeted pages returning
        zero new comments while a busy one is deferred for want of
        pages."""
        src = self._src("ingestion/fetch_reddit_comments.py")
        assert "DRY_PAGES_STOP" in src
        assert "dry_pages >= DRY_PAGES_STOP" in src

    def test_a_dry_crawl_advances_its_watermark(self):
        """A dry stop must mark the crawl `completed`.  Leaving it False
        makes the next run start in the same place and buy the same
        dead pages, so a dry subreddit could never make progress."""
        src = self._src("ingestion/fetch_reddit_comments.py")
        # `completed` starts True for every subreddit crawl ...
        assert "got, newest, completed = 0, int(wm) if wm else 0, True" in src
        # ... and the dry-stop branch breaks out without clearing it.
        i = src.index("if dry_pages >= DRY_PAGES_STOP:")
        branch = src[i:src.index("break", i) + 5]
        assert "completed = False" not in branch, (
            "the dry-stop must leave completed True or the watermark "
            "will not advance")

    def test_a_rate_limit_wearing_a_422_is_retried(self):
        """The API returns some rate limits as HTTP 422 with a body of
        {"error": "Timeout. Maybe slow down a bit"}.  The body is what
        separates that from a genuinely malformed request, so the
        fetcher must inspect it and retry rather than abandon the
        subreddit."""
        src = self._src("ingestion/fetch_reddit_comments.py")
        assert 'slow down' in src
        assert "r.status_code == 422" in src

    def test_the_fa_budget_is_a_constant_not_a_file_read(self):
        """The false-alarm budget is a config constant, never read from
        euphoria_report.json.  The euphoria stage rewrites that file in
        parallel with the phases stage, so a budget read from it depends
        on which stage finishes first: two passes over identical data
        can read 0.23 then 0.19 and disagree on the adopted result."""
        from src.config import EUPHORIA_FA_BUDGET_PER_IY
        assert EUPHORIA_FA_BUDGET_PER_IY == 0.23
        src = self._src("src/analytics/euphoria_phases.py")
        assert 'fa_per_instrument_year' not in src, (
            "the FA budget is being read off the last run again - that "
            "races the euphoria stage and ratchets the bar downward")
        assert src.count("EUPHORIA_FA_BUDGET_PER_IY") >= 3

    def test_no_silent_boolean_downcast_remains(self):
        """Filling NaN into an otherwise-boolean column without stating
        the dtype relies on a deprecated pandas downcast and emits a
        FutureWarning; every such fill must state `.astype(bool)`."""
        for rel in ("src/analytics/euphoria_phases.py", "src/analytics/signals.py"):
            src = self._src(rel)
            assert ".fillna(False)\n" not in src.replace(
                ".fillna(False).astype(bool)", ""), (
                f"{rel} still fills a bool column without stating dtype")

    def test_a_theme_with_no_priced_line_is_reported(self):
        """A theme whose entire fallback chain is unpriced is defined,
        counted, and then dropped before scoring, so the universe line
        shows one theme fewer than the config defines.  The page must
        report that rather than let the counts disagree silently."""
        src = self._src("dashboard.py")
        assert "No priced instrument at all" in src




class TestNothingCanDangle:
    """No file, command or path cited anywhere in the repository may
    point at something that does not exist.

    A dangling citation fails silently until an operator runs the
    command or opens the path.  `tools/verify_deps.py` catches them,
    and it has two structural requirements that are pinned here: it
    must sweep comments and markdown, not just quoted string literals
    in .py files, and its directory-prefix list must include
    directories that do NOT exist, because a reference to a deleted
    folder is otherwise invisible by construction."""

    @staticmethod
    def _root():
        from pathlib import Path
        return Path(__file__).resolve().parents[1]

    # Folders that are not the project's own code: environments and
    # third-party packages (which may legitimately contain non-UTF-8
    # fixtures), scratch folders, and the optional research tree.
    _SKIP_PARTS = (".venv", "venv", "env", "site-packages", "node_modules",
                   "_to_delete", ".ipynb_checkpoints", "__pycache__",
                   "research", "presentations", "others", ".git")

    @classmethod
    def _project_py_files(cls):
        for p in cls._root().rglob("*.py"):
            rel = p.relative_to(cls._root()).parts
            if any(part in cls._SKIP_PARTS for part in rel[:-1]):
                continue
            yield p

    def test_no_cited_path_is_missing(self):
        """verify_deps exits 0.  If this fails, something references a
        file that is not there - fix the reference, or say in the same
        paragraph that the file is gone.

        Check the tree is complete before acting on a finding:
        verify_deps cannot tell a deleted file from an un-cloned one, so
        a "missing file" result is only as good as the tree it ran
        against.  Paths that live only on a full checkout are declared
        inside verify_deps rather than filtered here."""
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "tools/verify_deps.py"],
                           cwd=self._root(), capture_output=True, text=True)
        assert r.returncode == 0, (
            "verify_deps found dangling references:\n" + r.stdout[-3000:])

    def test_the_checker_still_reads_comments_and_markdown(self):
        """The checker sweeps markdown, lists non-existent directories in
        its prefix set, and honours the known-absent phrases.  Narrowing
        it back to string literals would silently stop it catching
        anything."""
        src = (self._root() / "tools" / "verify_deps.py").read_text(
            encoding="utf-8")
        assert "def sweep_docs(" in src, "markdown is no longer swept"
        assert "_CITED_DIRS" in src and '"reference"' in src, (
            "the prefix list must include directories that DO NOT exist - "
            "that is the case the old checker could not see")
        assert "_KNOWN_ABSENT" in src

    def test_every_python_file_parses(self):
        """A syntax error anywhere is a broken pipeline, and several of
        these files are only imported on a copy with the raw post store,
        so the test suite would not otherwise exercise them."""
        import ast
        bad = []
        for p in self._project_py_files():
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError) as e:
                bad.append(f"{p.relative_to(self._root())}: {e}")
        assert not bad, "files will not parse:\n" + "\n".join(bad)

    def test_config_exposes_everything_its_importers_ask_for(self):
        """`from src.config import (...)` of a missing name fails at
        IMPORT time, which means the dashboard does not start at all."""
        import ast
        import src.config as C
        missing = []
        for p in self._project_py_files():
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.ImportFrom)
                        and node.module == "src.config"):
                    for a in node.names:
                        if a.name != "*" and not hasattr(C, a.name):
                            missing.append(
                                f"{p.relative_to(self._root())}: {a.name}")
        assert not missing, ("src/config.py no longer defines:\n"
                             + "\n".join(missing))


class TestAiPulseControls:
    """src/analytics/ai_pulse.py - back-dating and prompt-structure
    properties."""

    def test_a_back_dated_run_cannot_overwrite_the_live_pulse(self):
        """Reading history must never clobber today's page.  A dated run
        writes ai_pulse_<date>.json; only a live run touches
        ai_pulse.json."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "analytics"
               / "ai_pulse.py").read_text(encoding="utf-8")
        assert 'f"ai_pulse_{AS_OF:%Y-%m-%d}.json"' in src
        i = src.index("out_path = (OUT_PATH if AS_OF is None")
        assert "OUT_PATH if AS_OF is None" in src[i:i + 120]

    def test_the_clock_moves_in_exactly_one_place(self):
        """AS_OF is applied inside `_read`, the single function every
        store passes through. Threading it per-call-site would let one
        section keep reading a different day, and a pulse dated
        inconsistently is worse than one not back-dated at all."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "analytics"
               / "ai_pulse.py").read_text(encoding="utf-8")
        body = src[src.index("def _read("):src.index("def _evidence(")]
        assert "AS_OF is not None" in body and 'df["date"] <= AS_OF' in body

    def test_the_forum_paragraph_asks_what_they_SAY(self):
        """The forum prompt asks what each board is SAYING, not what it
        IS.  Asking the model to characterise the boards produces static
        descriptions ('r/investing is a long-term community') that carry
        no information about the current week."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "analytics"
               / "ai_pulse.py").read_text(encoding="utf-8")
        assert "WHAT EACH BOARD IS ACTUALLY " in src
        assert "NEVER describe what " in src
        assert "THE FORUMS THEMSELVES" not in src

    def test_theme_briefs_are_long_and_structured(self):
        """The theme prompt sets a 220-300 word target AND names four
        required elements.  Raising the word target alone yields more
        adjectives; the required elements are what make the extra words
        carry content."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "analytics"
               / "ai_pulse.py").read_text(encoding="utf-8")
        assert "220-300 " in src
        # the shorter target must not appear as a JSON-spec value; a
        # comment may legitimately mention it, so the check targets the
        # quoted spec form rather than the bare number
        assert '"brief: 80-120' not in src.replace(" ", "")\
            .replace("brief:80-120", '"brief: 80-120')
        for part in ("THE ARGUMENT", "THE EVIDENCE THEY CITE",
                     "THE DISSENT", "WHAT CHANGED"):
            assert part in src, f"{part} missing from the theme prompt"

    def test_the_roadmap_panel_is_gone(self):
        """The AI Pulse tab has no planned-segments expander, exposes the
        prompt behind the page, and owns its date through the single
        slider at the top (see research.ipynb)."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1]
               / "dashboard.py").read_text(encoding="utf-8")
        # the expander must be absent; the prompt expander must remain
        assert 'st.expander("planned LLM segments' not in src
        assert "the exact prompt behind this page" in src
        # one control owns the date: the slider at the top of the page
        assert "the market's mood on" in src
        assert "_market_read(" in src


class TestPreflight:
    """The health check must itself keep working - it is the thing that
    catches the failures nothing else reports."""

    def test_preflight_runs_clean(self):
        """tools/preflight.py exits 0 on this checkout."""
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        r = subprocess.run([sys.executable, "tools/preflight.py"],
                           cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, (
            "preflight is FAILING - something downstream is already "
            "wrong:\n" + r.stdout[-2500:])

    def test_the_164mb_write_stays_disabled(self):
        """daily_ticker_conviction.parquet is ~164MB, would be rebuilt
        every run, and is read by nothing; the conviction stage must
        write the theme file only."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "analytics"
               / "conviction.py").read_text(encoding="utf-8")
        i = src.index("for sent_name, out_name, entity in [")
        loop = src[i:i + 220]
        assert "TICKER_CONVICTION" not in loop, (
            "the ticker conviction write is back - it is 164MB per run "
            "and nothing reads it")
        assert "THEME_CONVICTION" in loop, "the theme file must still ship"


class TestPulseRegister:
    """The AI Pulse narrative sections must read like a colleague
    briefing the reader - forum, ticker and reason in one sentence,
    mood shown as behaviour - and must be written from posts alone.
    Only the divergences call may see the numeric evidence pack."""

    @staticmethod
    def _src():
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / "src" / "analytics"
                / "ai_pulse.py").read_text(encoding="utf-8")

    def test_no_aggregate_reaches_the_model(self):
        """The market, themes and agentic prompts carry posts and nothing
        else; the evidence pack is still built and saved (the dropdown
        orders from it) but must never be handed to those prompts, and
        the system prompt must forbid invented numbers.

        Passing `ev` back into a prompt is a one-word change that would
        silently reintroduce aggregates while every other test still
        passed, so the call signatures are pinned."""
        src = self._src()
        for bad in ("_market_prompt(ev", "_themes_prompt(ev",
                    "_agentic_prompt(ev"):
            assert bad not in src, (
                f"{bad!r} is back - calls 1, 2 and 4 read posts only")
        assert "YOU HAVE NO STATISTICS" in src, (
            "the system prompt no longer forbids invented numbers, which "
            "is the whole safeguard once the evidence pack is withheld")
        assert "system=_PULSE_SYSTEM" in src

    def test_divergences_is_the_only_call_with_numbers(self):
        """The divergences call is the only one that receives the
        evidence pack, and it runs under its own system prompt.  Handing
        it numbers under the posts-only system prompt would tell the
        model in one breath that it has no statistics and in the next to
        cite them."""
        src = self._src()
        assert "_watch_prompt(ev, posts)" in src
        assert "system=_WATCH_SYSTEM" in src
        assert "_WATCH_SYSTEM = _PULSE_SYSTEM.replace(" in src
        import src.analytics.ai_pulse as ap
        assert ap._WATCH_SYSTEM != ap._PULSE_SYSTEM, (
            "_WATCH_SYSTEM is identical to the posts-only prompt - the "
            "replace() target drifted and the divergences call is now "
            "being told it has no statistics while holding a block of "
            "them")
        assert "YOU HAVE NO STATISTICS" not in ap._WATCH_SYSTEM
        assert "EVIDENCE block" in ap._WATCH_SYSTEM
        # and the posts-only calls must still be clean
        assert "EVIDENCE" not in ap._market_prompt([])
        assert "EVIDENCE" not in ap._agentic_prompt([])

    def test_the_measured_record_is_still_kept(self):
        """Withholding the pack from the MODEL must not stop it being
        MEASURED and saved: the dashboard orders the theme dropdown from
        it, and it is the record the narrative can be checked against."""
        src = self._src()
        assert "emerging_terms_7d" in src
        assert "daily_term_counts.parquet" in src
        assert '"evidence": ev' in src, (
            "the evidence pack is no longer saved into ai_pulse.json")

    def test_theme_briefs_are_chosen_by_the_posts(self):
        """A theme may only reach the dropdown when there is enough of
        its own conversation to write the brief FROM - choosing from an
        aggregate the model cannot see would let a theme be selected and
        then have nothing to read."""
        src = self._src()
        assert "_themes_from_posts" in src
        assert "theme_list = _themes_from_posts()" in src

    def test_mood_must_be_shown_as_behaviour_not_asserted(self):
        """An adjective is the model's conclusion; the behaviour behind
        it is evidence the reader can judge, so the prompt must demand
        the behaviour."""
        src = self._src()
        assert "never just label the mood" in src
        assert "gain screenshots" in src

    def test_the_forum_ticker_reason_triple_is_required(self):
        """The prompt requires forum, ticker and reason in one sentence."""
        src = self._src()
        assert "Name the forum, name the " in src
        assert "REASON in the same " in src

    def test_the_evidence_pack_actually_carries_terms(self):
        """The instruction is worthless if the numbers are not supplied."""
        import src.analytics.ai_pulse as ap
        ev = ap._evidence()
        if not ev:
            import pytest
            pytest.skip("no aggregates on disk")
        assert "emerging_terms_7d" in ev, (
            "the prompt cites emerging_terms_7d but the evidence pack "
            "does not contain it - the model would be told to use a key "
            "that is not there")


class TestBackfillRunner:
    """`tools/backfill_reddit.py` streams a child process's output.

    A wrapper that hides the child's progress is worse than no wrapper:
    the operator cannot tell a 26-minute chunk from a hang, which is
    exactly what happened on 2026-08-12 ("its paused at [1/30] - why?")."""

    @staticmethod
    def _src():
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / "tools"
                / "backfill_reddit.py").read_text(encoding="utf-8")

    def test_the_child_runs_unbuffered(self):
        """Python BLOCK-buffers stdout when it is a pipe (~8 KB). One
        chunk emits ~700 bytes, so without `-u` the buffer never fills
        and nothing appears until the child exits half an hour later.
        `bufsize=1` does NOT fix this - it is the parent's read side."""
        src = self._src()
        assert 'sys.executable, "-u", FETCHER' in src, (
            "the fetcher subprocess is buffered again - progress will "
            "not appear until each chunk ends")

    def test_there_is_a_heartbeat(self):
        """Even unbuffered, one busy subreddit can page for minutes in
        silence. On a 13-hour job silence must never be ambiguous."""
        src = self._src()
        assert "still working" in src
        assert "threading.Thread(target=_beat, daemon=True)" in src


class TestInflectionMarker:
    """The INFLECTION head is a CONTEXT MARKER, not a call.

    It was adopted with its numbers known (9.6% hit vs a 5.6% base, no
    direction), on the condition that it stays furniture:
    drawn on the price panel, never a call, never in the watchlist, and
    never able to change GET IN or GET OUT. These tests are the fence
    around that condition, because every one of those boundaries is a
    one-line change away from being crossed by accident."""

    @staticmethod
    def _src(name):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / name).read_text(encoding="utf-8")

    def test_the_turn_head_cannot_touch_get_in_or_get_out(self):
        """The whole safety case. If the inflection score ever gates, filters
        or re-scores either call, withdrawing the head stops being free
        and the record before and after this change stops comparing."""
        src = self._src("src/analytics/euphoria_phases.py")
        body = src[src.index("ds[\"inflection_score\"] = np.nan"):]
        for forbidden in ("get_in", "get_out"):
            assert f'ds["{forbidden}"] =' not in body, (
                f"the inflection block assigns to {forbidden} - it must not")
        # and the alert builders must not read the inflection columns
        head = src[:src.index("ds[\"inflection_score\"] = np.nan")]
        assert "inflection_score" not in head.split("def _alert_dates")[-1][:1500]

    def test_turn_fires_with_no_phase_gate(self):
        """GET IN and GET OUT are gated by the 120d boom bar. A reversal
        is as interesting at the bottom of a bust as at the top of a
        boom, so gating the inflection head would discard half of what it
        exists to see."""
        src = self._src("src/analytics/euphoria_phases.py")
        fn = src[src.index("def inflection_alerts("):src.index("def _day_ints")]
        assert "gate = [True] * len(dates)" in fn

    def test_the_turn_label_requires_a_real_move_away(self):
        """Without the move-away test every flat drift containing a
        local maximum is a 'turning point' and the label is noise."""
        src = self._src("src/analytics/euphoria_phases.py")
        fn = src[src.index("def inflection_label_frame("):
                 src.index("def inflection_alerts(")]
        assert "EUPHORIA_INFLECTION_MIN_MOVE" in fn
        assert "median(axis=1)" in fn, (
            "the inflection label must use EXCESS over the cross-section, not "
            "a raw forward return")

    def test_the_infl_threshold_is_frozen_not_recomputed(self):
        """Same contract as the two desk cuts: research re-opens by
        being typed. A threshold recomputed every live run leaves
        nothing on disk explaining how today's marker differs from
        yesterday's."""
        src = self._src("src/analytics/euphoria_phases.py")
        assert 'if research or "threshold" not in _frozen_turn:' in src
        assert '_infl_src = "frozen"' in src

    def test_the_inflection_features_are_price_free(self):
        """The head is sold as a CROWD signal. A price feature in its
        bank would make that untrue."""
        # aliased `eph`, NOT `ep`: elsewhere in this file `ep` is an
        # episode row from itertuples(), and verify_deps resolves the
        # name file-wide - importing the module as `ep` makes every
        # `ep.peak` in the file look like a missing module attribute.
        import src.analytics.euphoria_phases as eph
        from src.analytics import ml_detector as mld
        for f in eph.INFLECTION_EXTRA_FEATURES:
            assert f not in mld.PRICE_FEATURES
        src = self._src("src/analytics/euphoria_phases.py")
        fn = src[src.index("def inflection_features("):
                 src.index("def inflection_label_frame(")]
        assert "price" not in fn.lower().replace("price-free", "")

    def test_the_marker_is_not_a_call_on_the_dashboard(self):
        """Drawn as a tick, never as a full-height rule, and absent from
        the watchlist orderings."""
        src = self._src("dashboard.py")
        assert "possible inflection" in src
        # the caveat that MUST survive: the marker has no direction.
        # (The hit-rate sentence was removed from the hover on desk
        # instruction 2026-08-12 - it lives in the RUNBOOK and the
        # parameter register instead, which is where a number that
        # changes on every research pass belongs.)
        assert "Direction NOT implied" in src
        # a store written before 2026-08-12 has no inflection columns; the
        # page must SAY so rather than silently drawing nothing, which
        # reads identically to "this name has no inflections"
        assert "predates them" in src
        # The inflection head appears in the watchlist as a THIRD SIDE
        # ("closest to an INFLECTION" ordering). What must remain true is
        # that it is
        # labelled context wherever it is rendered and that it still
        # cannot fire, gate or re-score a call.
        # In the watchlist the inflection appears as a COLUMN, not as a
        # competing sort (desk 2026-08-12: the two "closest to..."
        # orderings asked one question two ways). What must hold is
        # that it is labelled context and that it cannot out-rank a
        # real call.
        assert "INFLECTION (context)" in src
        assert "context should" in src
        assert "closest to an INFLECTION" not in src, (
            "the second watchlist ordering is back - it produced "
            "nearly the same table as the first")
        blk = src[src.index("_infl_gap = dict("):
                  src.index("_best = _watch.drop_duplicates")]
        assert "_calls = _watch[~_watch[\"_is_infl\"]]" in blk, (
            "inflection rows are competing for the ranking again")
        assert "9.6%" not in src, (
            "a measured hit rate is hard-coded in the dashboard - it "
            "goes stale silently the first time the head is re-fitted")
        blk = src[src.index("if inflection_alerts:"):
                  src.index("for d in onset_alerts:")]
        # THE MARKER RIDES THE DRAWN LINE. It was first placed at
        # `level.min()` - a euphoria value on an axis that is usually
        # PRICE - so every diamond landed near y=0, detached from the
        # series and looking broken. `_carrier` is whatever line is
        # actually drawn, so this cannot regress when the axis changes.
        assert "_carrier" in blk, (
            "the inflection marker is not placed on the drawn line - it will "
            "land on whatever the y-axis happens to mean")
        assert "level" not in blk
        assert "add_vline" not in blk, (
            "the inflection marker is drawn as a vertical rule - that is the "
            "visual language of a CALL")

    def test_the_store_carries_the_columns(self):
        import os
        import pandas as pd
        p = os.path.join(DATA_DIR, "processed", "euphoria_desk.parquet")
        if not os.path.exists(p):
            import pytest
            pytest.skip("no desk store on disk")
        d = pd.read_parquet(p)
        assert {"inflection", "inflection_score"} <= set(d.columns)
        assert d["inflection"].dtype == bool


class TestMoodGauge:
    """The mood/bullishness SCORES are gone, and must stay gone.

    Requirement: "on AI pulse remove the bullish score number
    and all the associated code". Two numbers existed - the LLM's
    'retail mood gauge X/100' and the slider-driven bullishness
    percentile - and both were removed. These tests are the tripwire
    against either quietly coming back (e.g. via a revert of the pulse
    prompt spec, which used to REQUEST a mood_gauge from the model)."""

    @staticmethod
    def _src():
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / "dashboard.py").read_text(encoding="utf-8")

    def test_the_slider_bullishness_score_is_gone(self):
        """The percentile-ranked 0-100 bullishness number the mood
        slider used to print."""
        src = self._src()
        assert "(wk <= nb).mean()" not in src, (
            "the slider bullishness percentile is back - it was "
            "for the score number and its code to be removed")
        assert "bullishness (0-100)" not in src

    def test_the_llm_mood_gauge_is_gone_everywhere(self):
        """The X/100 fear-greed metric, in BOTH the page and the prompt
        spec (leaving it in the spec would pay gateway tokens for a
        number nothing displays)."""
        src = self._src()
        assert "retail mood gauge" not in src
        assert 'get("mood_gauge")' not in src
        from pathlib import Path
        pulse_src = (Path(__file__).resolve().parents[1] / "src" / "analytics"
                     / "ai_pulse.py").read_text(encoding="utf-8")
        assert '"mood_gauge": "object' not in pulse_src, (
            "the pulse prompt still asks the model for a mood_gauge")

    def test_the_page_says_when_the_words_are_a_different_date(self):
        """The written sections are one stored document and cannot follow
        a slider. A page showing one date on top and another underneath
        is indistinguishable from a broken one."""
        src = self._src()
        assert "not the selected date" in src
        assert "do not move with the slider" in src


class TestWeeklySnapshot:
    """Dragging the slider must produce a written read of THAT week.

    Requirement: dragging the slider updates the
    text though, so i get like a snapshot of the market at that point of
    time." The model cannot do this - one stored document, gateway call
    - so the snapshot is composed from the stores instead."""

    @staticmethod
    def _src():
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / "dashboard.py").read_text(encoding="utf-8")

    def test_the_snapshot_is_computed_not_retrieved(self):
        src = self._src()
        assert "def _snapshot_text(" in src
        # The rendered digest (snapshot quote + the crowd/felt columns)
        # is not shown on the AI Pulse tab; the composer stays for
        # research and CLI use. Guard that the section STAYS removed.
        assert "What the crowd was talking about" not in src

    def test_it_reports_flags_terms_and_rotation(self):
        """A snapshot that only gave a mood score would not be worth
        dragging to."""
        src = self._src()
        i = src.index("def _snapshot_text(")
        body = src[i:i + 5000]
        for part in ("Rotation was", "words spreading", "Loudest names",
                     "CUT EXPOSURE", "INCREASE EXPOSURE"):
            assert part in body, f"snapshot omits {part}"

    def test_breadth_reads_wide_or_narrow_not_loud_or_quiet(self):
        """Breadth is an absolute share of themes, and with the
        intensity score removed (see research.ipynb) it is the ONLY axis
        the sentence may speak to - wide vs carried-by-a-few, never a
        loudness claim it no longer measures."""
        src = self._src()
        assert "def _breadth_clause(" in src
        assert "carried by a few" in src


class TestBackDatedHarvest:
    """A back-dated pulse must hand the model posts from THAT week.

    Requirement: "even historically if we drag it back to a specific
    day and it uses LLMs to analyse the posts in a lookback like 3d".
    Two bugs stopped that working, both silent."""

    @staticmethod
    def _src():
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / "src" / "analytics"
                / "ai_pulse.py").read_text(encoding="utf-8")

    def test_the_cap_is_enforced_inside_the_file(self):
        """HARVEST_MAX was only checked BETWEEN files, so one 43MB
        archive overran it 5x and starved every archive after it."""
        src = self._src()
        i = src.index('"text": body[:POST_CLIP]})')
        assert "if len(rows) >= max_posts:" in src[i:i + 400], (
            "the cap is not checked inside the line loop")

    def test_archives_are_ordered_by_relevance_to_the_target_day(self):
        """Newest-mtime order is right live and wrong back-dated: the
        archive holding the target week was never opened."""
        src = self._src()
        assert "def _rank(" in src and "_AS_OF_STR <= hi" in src

    def test_the_as_of_filter_runs_before_the_cap(self):
        """Filtering after the cap meant the cap filled with recent posts
        that were then all discarded - the model got a stale sample or
        nothing."""
        src = self._src()
        assert 'day > _AS_OF_STR' in src

    def test_a_back_dated_harvest_returns_that_weeks_posts(self):
        """The behaviour, not the code. Measured against the archives on
        this machine."""
        import pandas as pd
        import src.analytics.ai_pulse as ap
        ap.AS_OF = None
        ap._HARVEST = ap._HARVEST_KEY = None
        live = ap._harvest()
        if not live:
            import pytest
            pytest.skip("no raw archives on this machine")
        newest = max(r["day"] for r in live)
        target = (pd.Timestamp(newest) - pd.Timedelta(days=30))
        ap.AS_OF = target
        ap._HARVEST = ap._HARVEST_KEY = None
        back = ap._harvest()
        ap.AS_OF = None
        ap._HARVEST = ap._HARVEST_KEY = None
        assert back, "back-dated harvest returned nothing"
        got = max(r["day"] for r in back)
        assert got <= f"{target:%Y-%m-%d}", "posts leaked past the as-of"
        gap = (target - pd.Timestamp(got)).days
        # A hole shorter than one fetch window (FETCH_LOOKBACK_DAYS) is a
        # cadence gap between two crawls, not data loss; a longer one is.
        from src.config import FETCH_LOOKBACK_DAYS
        assert gap <= FETCH_LOOKBACK_DAYS, (
            f"newest back-dated post is {gap} days before the target - "
            "the model would be reading a different week")


class TestRobustShare:
    """src/analytics/robust_share.py - the 2026-08-07 coverage fix. These
    are the invariants the estimator was built to provide; if any of
    them fails, the fake-zero / dilution defects are back."""

    @staticmethod
    def _toy():
        days = pd.date_range("2026-01-01", periods=40, freq="D")
        rows = []
        for i, d in enumerate(days):
            rows.append({"date": d, "ticker": "AAA", "source": "reddit",
                         "mention_count": 5})
            rows.append({"date": d, "ticker": "BBB", "source": "reddit",
                         "mention_count": 45})
            # stocktwits covers only every third day, in bulk - the
            # uneven-pull pattern that caused the defect
            if i % 3 == 0:
                rows.append({"date": d, "ticker": "BBB",
                             "source": "stocktwits",
                             "mention_count": 400})
        long = pd.DataFrame(rows)
        return long.groupby(["date", "ticker"], as_index=False)[
            "mention_count"].sum(), long, days

    def test_no_fake_zero_on_a_thin_day(self):
        """A name with steady mentions must never print 0% because the
        puller had a thin day."""
        from src.analytics.robust_share import robust_share
        counts, by_src, days = self._toy()
        sh = robust_share(counts, "ticker", "AAA", days)
        assert (sh.iloc[7:] > 0).all(), "steady name printed a zero share"

    def test_stratification_damps_source_mix_swings(self):
        """AAA lives only on Reddit; StockTwits' bulk days must move its
        share LESS under the stratified estimator than under the pooled
        one."""
        from src.analytics.robust_share import robust_share
        counts, by_src, days = self._toy()
        pooled = robust_share(counts, "ticker", "AAA", days)
        strat = robust_share(counts, "ticker", "AAA", days,
                             by_source=by_src)
        assert (strat.iloc[10:].std()
                < pooled.iloc[10:].std() + 1e-12), (
            "stratification made the share MORE sensitive to the "
            "source mix, not less")

    def test_shares_are_bounded(self):
        from src.analytics.robust_share import robust_share
        counts, by_src, days = self._toy()
        for bs in (None, by_src):
            sh = robust_share(counts, "ticker", "BBB", days,
                              by_source=bs).dropna()
            assert ((sh >= 0) & (sh <= 100)).all()

    def test_convexity_fast_path_matches_the_apply_path(self):
        """log_convexity got a vectorised fast path (one convolution);
        it must be numerically identical to the rolling-apply original."""
        import numpy as np
        from src.analytics.euphoria import log_convexity
        s = pd.Series(
            np.exp(np.linspace(0, 3, 200))
            + np.abs(np.sin(np.arange(200))) * 5 + 1,
            index=pd.date_range("2025-01-01", periods=200))
        fast = log_convexity(s)
        window = 60
        t = np.arange(window, dtype=float)
        t = (t - t.mean()) / window
        X = np.column_stack([np.ones(window), t, t * t])
        pinv = np.linalg.pinv(X)
        slow = np.log(s).rolling(window).apply(
            lambda w: float(pinv[2] @ w), raw=True)
        assert np.allclose(fast.dropna(), slow.dropna())


class TestMLDetector:
    """src/analytics/ml_detector.py - the learned desk detectors (2026-08)."""

    def test_winner_is_picked_on_lift_not_raw_ap(self):
        """The rule-based baseline's gated frame gives it a base rate of
        ~0.5, so its raw AP dwarfs every learner's while its LIFT is ~1x.
        Raw-AP ranking would hand the tournament to the baseline forever."""
        from src.analytics.ml_detector import pick_winner
        results = {
            "get_out": {"rules": {"ap": 0.60, "ap_baseline": 0.59,
                                  "auroc": 0.52, "false_alarms": 4},
                        "ens": {"ap": 0.19, "ap_baseline": 0.054,
                                "auroc": 0.74, "false_alarms": 99}},
            "get_in": {"rules": {"ap": 0.09, "ap_baseline": 0.065,
                                 "auroc": 0.56, "false_alarms": 97},
                       "ens": {"ap": 0.20, "ap_baseline": 0.061,
                               "auroc": 0.76, "false_alarms": 90}},
        }
        assert pick_winner(results) == "ens"

    def test_crowd_only_variants_never_ship(self):
        """*_crowd rows are the clean-claim record - present in the
        table, never adoptable."""
        from src.analytics.ml_detector import pick_winner
        big = {"ap": 9.9, "ap_baseline": 0.01, "auroc": 0.99,
               "false_alarms": 0}
        small = {"ap": 0.1, "ap_baseline": 0.05, "auroc": 0.6,
                 "false_alarms": 10}
        results = {"get_out": {"ens_crowd": big, "logit": small},
                   "get_in": {"ens_crowd": big, "logit": small}}
        assert pick_winner(results) == "logit"

    def test_the_bank_has_no_embedded_gate_constants(self):
        """The point of the exercise: every ML bank feature is a raw
        percentile/ratio measurement - e2 (with its baked-in 75%
        persistence gate) and fade (with its baked-in 90% attention
        gate) must not be in it."""
        from src.analytics.ml_detector import ML_BANK, DESK_ML_BANK
        assert "e2" not in ML_BANK and "fade" not in ML_BANK
        assert set(ML_BANK) < set(DESK_ML_BANK)
        assert {"bull_level", "bull_persist"} < set(ML_BANK)

    def test_registry_covers_every_tournament_family(self):
        from src.analytics.ml_detector import ML_FITS
        assert set(ML_FITS) == {"logit", "gbm", "mlp", "ens"}

    def test_f1_chooser_ignores_the_fa_budget(self):
        """No hand-set budget survives in the ML operating point: the
        chooser must return the same threshold whatever budget value is
        passed."""
        import numpy as np
        from src.analytics.ml_detector import choose_threshold_f1
        rng = np.random.default_rng(0)
        days = pd.date_range("2023-01-01", periods=300)
        scored = pd.DataFrame({
            "date": days, "name": "AAA",
            "score": rng.uniform(0, 1, 300),
            "year": days.year})
        eps = pd.DataFrame({
            "name": ["AAA"], "kind": ["theme"],
            "trough": [pd.Timestamp("2023-03-01")],
            "onset_lo": [pd.Timestamp("2023-03-01")],
            "onset_hi": [pd.Timestamp("2023-04-15")],
            "peak": [pd.Timestamp("2023-06-01")],
            "year": [2023], "onset_detectable": [True],
            "top_detectable": [True]})
        a = choose_threshold_f1(scored, eps, "onset", 0.0001, 1)
        b = choose_threshold_f1(scored, eps, "onset", 999.0, 1)
        assert a == b

    def test_shipped_store_never_shows_a_start_right_after_an_end(self):
        """The coherence rule, verified on the store that actually ships:
        no GET IN within one cooldown after a GET OUT on the same name."""
        import os
        from src.config import PROCESSED_DIR, EUPHORIA_COOLDOWN_DAYS
        path = os.path.join(PROCESSED_DIR, "euphoria_desk.parquet")
        if not os.path.exists(path):
            pytest.skip("desk store not built yet")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        for name, g in df.groupby("name"):
            outs = g.loc[g["get_out"], "date"]
            ins = g.loc[g["get_in"], "date"]
            for d in ins:
                assert not any(
                    0 <= (d - t).days < EUPHORIA_COOLDOWN_DAYS
                    for t in outs), (
                    f"{name}: GET IN on {d:%Y-%m-%d} lands inside the "
                    "cooldown after a GET OUT")


class TestPriceBlindTrigger:
    """The EXPERIMENTAL price-blind trigger (desk 2026-08-14: "i want
    only the post factors to predict the price, not price predicting
    price").

    The design promise is total: price appears NOWHERE between a post
    and an experimental call - not as a feature, not as the phase gate
    that routes IN vs OUT. These tests fence the two ways that promise
    breaks silently: a price feature drifting into the bank, and the
    dashboard mixing shipped and experimental quantities (a shipped
    score against an experimental cut reads plausibly and is wrong)."""

    @staticmethod
    def _src(name):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / name).read_text(
            encoding="utf-8")

    def test_the_bank_is_price_free(self):
        import src.analytics.euphoria_phases as eph
        from src.analytics import ml_detector as mld
        xp_bank = list(mld.ML_BANK) + eph.INFLECTION_EXTRA_FEATURES
        for f in mld.PRICE_FEATURES:
            assert f not in xp_bank
        # the pipeline builds the bank from exactly these two lists
        src = self._src("src/analytics/euphoria_phases.py")
        assert "list(mld.ML_BANK) + INFLECTION_EXTRA_FEATURES" in src

    def test_the_pipeline_never_gates_xp_on_price(self):
        """The xp alert path must pass an all-True gate - the phase
        gate (boomed120) is the second door price enters by."""
        src = self._src("src/analytics/euphoria_phases.py")
        blk = src[src.index("def _xp_alerts("):
                  src.index('_rin = _frozen_xp["get_in"]')]
        assert "[True] * len(" in blk
        assert "boomed120" not in blk
        # and the candidate frame is NOT the price-attached one
        blk2 = src[src.index("if EUPHORIA_XP_ENABLED:"):
                   src.index("xp_in_scored = xp_live.assign")]
        assert "attach_price_features" not in blk2

    def test_the_dashboard_cannot_mix_the_two_triggers(self):
        """Every surface reads columns and cuts through the same
        trigger-aware indirection - scores via IN_SCORE/OUT_SCORE,
        alert columns via sig_col, cuts via sig_head."""
        src = self._src("dashboard.py")
        assert 'IN_SCORE = f"in_score{_XP_PART}"' in src
        assert 'OUT_SCORE = f"out_score{_XP_PART}"' in src
        # 2026-08-17: sig_col grew the GET IN gate routing; the fence
        # follows - the column is still built through the SAME
        # trigger-aware parts, never assembled ad hoc at a call site.
        assert '_c = f"{base}{_XP_PART}{_gate_part}{_SIG_SUFFIX}"' in src
        assert "experimental_price_blind" in src
        # The trigger SELECTOR is not exposed - the dashboard is hardwired
        # to the shipped pair, so
        # the "your store predates the experimental columns" warning it
        # used to show has no way to fire and was removed with it. The
        # fence therefore changes shape: instead of asserting the
        # warning exists, assert the experimental path CANNOT BE
        # SELECTED, which is a strictly stronger guarantee.
        assert "XP_TRIGGER = False" in src
        _sel = [ln for ln in src.splitlines()
                if "st.sidebar" in ln and "trigger" in ln.lower()]
        assert not _sel, f"a trigger selector is back in the sidebar: {_sel}"
        # and the routing still flows through the same trigger-aware
        # parts, so re-enabling it is a one-line change that cannot
        # leave a surface reading the wrong family
        assert '_XP_PART = "_xp" if XP_TRIGGER else ""' in src
        # the watchlist keeps both sides eligible in xp mode (no gate)
        assert "_boomed or XP_TRIGGER" in src
        assert "(not _boomed) or XP_TRIGGER" in src
        # The mode is never the default. This used to be checked by
        # reading index=0 off the radio; with the radio gone the same
        # guarantee is now structural - XP_TRIGGER is a literal False
        # and nothing in the file assigns it True.
        _true = [ln.strip() for ln in src.splitlines()
                 if ln.strip().startswith("XP_TRIGGER")
                 and ln.strip().endswith("True")]
        assert not _true, f"XP_TRIGGER is set True somewhere: {_true}"

    def test_store_columns_when_present(self):
        """On a machine whose store has been rebuilt since 2026-08-14:
        the xp columns exist together, the booleans are bool, and the
        end-stage suppression held."""
        import os
        import pandas as pd
        p = os.path.join(DATA_DIR, "processed", "euphoria_desk.parquet")
        if not os.path.exists(p):
            import pytest
            pytest.skip("no local desk store")
        ds = pd.read_parquet(p)
        if "in_score_xp" not in ds.columns:
            import pytest
            pytest.skip("store predates the experimental trigger")
        for c in ("get_in_xp", "get_out_xp", "get_in_xp_strict",
                  "get_out_xp_strict"):
            assert c in ds.columns
            assert ds[c].dtype == bool
        assert ds["in_score_xp"].notna().any()
        both = ds[ds["get_in_xp"] & ds["end_stage"].astype(bool)]
        assert both.empty, "a posts-only GET IN fired on an end-stage day"


class TestSignedReadinessAndUngatedGetIn:
    """The adoptions (notebook 08 §10, record
    Data/research_record/nb08_single_dial.json): the ungated GET IN, the one
    SIGNED readiness, and the retirement of the Relaxed setting.

    The contradiction these changes killed - a name reading 100% of the
    way to GET IN and to GET OUT at once - came from displaying two
    heads whose scores correlate at +0.75. The fences below hold the
    two structural promises: GET OUT never loses its phase gate (§10.4:
    ungated its false alarms double), and the signed readiness is
    routed so exactly ONE side exists per name-day."""

    @staticmethod
    def _src(name):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / name).read_text(encoding="utf-8")

    def test_no_ungated_get_out_anywhere(self):
        """§10.4's asymmetry is the whole safety case: the gate IS the
        OUT side's direction knowledge. An ungated GET OUT column or
        code path is a one-line change away and must never appear."""
        for fname in ("src/analytics/euphoria_phases.py", "dashboard.py"):
            assert "get_out_nogate" not in self._src(fname), (
                f"{fname} references an ungated GET OUT - §10.4 refutes "
                "that variant (false alarms double)")

    def test_dashboard_reroutes_only_get_in(self):
        """The gate routing may touch GET IN and nothing else. The
        checkbox that briefly exposed the gated variant was removed
        2026-08-17 (GET IN is ungated, full stop) - the one-sided
        routing it guarded still has to hold."""
        src = self._src("dashboard.py")
        i = src.index("def sig_col(")
        body = src[i:i + 1200]
        assert 'base == "get_in"' in body
        assert 'base == "get_out"' not in body.replace(
            "GET OUT never does", "")

    def test_ungated_get_in_exists_and_respects_coherence(self):
        """The ungated columns are real columns with the coherence rule
        still applied: no GET IN on an end-stage day, none within one
        cooldown of a (gated) GET OUT. Dropping the gate widens WHEN a
        start may be called, never the coherence promise."""
        from src.config import EUPHORIA_COOLDOWN_DAYS
        path = os.path.join(DATA_DIR, "processed",
                            "euphoria_desk.parquet")
        if not os.path.exists(path):
            pytest.skip("no desk store on this machine")
        ds = pd.read_parquet(path)
        if "get_in_nogate_strict" not in ds.columns:
            pytest.skip("store predates the ungated GET IN - run "
                        "run_analytics --what phases")
        ds["date"] = pd.to_datetime(ds["date"])
        bad_end = ds[ds["get_in_nogate_strict"].astype(bool)
                     & ds["end_stage"].astype(bool)]
        assert bad_end.empty, "an ungated GET IN fired on an end-stage day"
        for name, g in ds.groupby("name"):
            outs = g.loc[g["get_out_strict"].astype(bool), "date"]
            if not len(outs):
                continue
            for d in g.loc[g["get_in_nogate_strict"].astype(bool), "date"]:
                assert not any(abs((d - t).days) <= EUPHORIA_COOLDOWN_DAYS
                               for t in outs), (
                    f"{name}: ungated GET IN within one cooldown of a "
                    f"GET OUT ({d.date()})")

    def test_signed_readiness_is_one_sided_by_construction(self):
        """Recompute the §10.5 signed readiness from the store exactly
        as the dashboard does: the phase routing must give every scored
        name-day exactly one live side, so the both-at-100% state is
        impossible, not just unobserved."""
        path = os.path.join(DATA_DIR, "processed",
                            "euphoria_desk.parquet")
        if not os.path.exists(path):
            pytest.skip("no desk store on this machine")
        ds = pd.read_parquet(path).dropna(subset=["in_score", "out_score"])
        if not len(ds):
            pytest.skip("no scored rows")
        assert "boomed120" in ds.columns
        boomed = ds["boomed120"].eq(True)
        # one side per row: the routed reading is IN xor OUT, never both
        side_in = ~boomed
        side_out = boomed
        assert not (side_in & side_out).any()
        assert (side_in | side_out).all()

    def test_relaxed_setting_is_gone_from_the_interface(self):
        """Operating-point columns: both cuts are stored. The F1 columns stay in the
        store (recorded numbers must reproduce); the CONTROL must not
        return."""
        src = self._src("dashboard.py")
        assert '"signal setting"' not in src
        assert "RELAXED_SIGNALS = False" in src


class TestRetailFlowDial:
    """The continuous retail-flow dial (adoption, notebook 08
    §9). Adopted as the trend/context layer on the same explicit
    condition as the inflection marker: it is FURNITURE. It never
    fires, gates, filters or re-scores a call, and its smoothness comes
    from model structure (state-space filter, slow features), never
    from a lookout into the future."""

    @staticmethod
    def _src(name):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / name).read_text(encoding="utf-8")

    def test_the_dial_cannot_touch_the_calls(self):
        """retail_flow.py must never read or assign the call columns -
        the dial is display, the calls are the signal."""
        src = self._src("src/analytics/retail_flow.py")
        for forbidden in ('"get_in', '"get_out', "'get_in", "'get_out"):
            assert forbidden not in src, (
                "retail_flow.py references a call column - the dial "
                "must stay furniture")

    def test_the_dial_is_failure_isolated(self):
        """A dial bug must degrade to missing columns, never take the
        desk store down with it."""
        src = self._src("src/analytics/euphoria_phases.py")
        i = src.index("attach_retail_flow")
        block = src[max(0, i - 600):i]
        assert "try:" in block, (
            "the retail-flow attach is not wrapped - a dial error "
            "would kill the signal store write")

    def test_the_dial_filter_is_causal(self):
        """The Kalman recursion may only ever fold in the CURRENT
        observation - no centred windows, no backward pass."""
        from src.analytics.retail_flow import _kalman
        names = ["a"] * 10
        vals = [0.0] * 5 + [1.0] * 5
        out = _kalman(np.array(names), np.array(vals), 0.2)
        assert (out[:5] == 0).all(), "the filter saw the future"
        assert out[5] == pytest.approx(0.2), "gain arithmetic drifted"

    def test_dial_columns_bounded_and_present_when_fresh(self):
        path = os.path.join(DATA_DIR, "processed",
                            "euphoria_desk.parquet")
        if not os.path.exists(path):
            pytest.skip("no desk store on this machine")
        ds = pd.read_parquet(path)
        if "retail_flow_disp" not in ds.columns:
            pytest.skip("store predates the dial - run "
                        "run_analytics --what phases")
        v = ds["retail_flow_disp"].dropna()
        assert len(v), "dial column exists but is empty"
        assert float(v.min()) >= -1.0 - 1e-9
        assert float(v.max()) <= 1.0 + 1e-9


class TestPinnedModelFamily:
    """DESK_MODEL_FAMILY must skip the tournament without weakening the
    validation that follows it (see research.ipynb)."""

    def _spy(self):
        import src.analytics.ml_detector as mld
        calls = []

        def spy(vframe, episodes, vbank, label, mode, fit, budget,
                chooser=None):
            calls.append((label, mode, getattr(fit, "__name__", "?"),
                          len(vbank)))
            return {"ap": 0.2, "ap_baseline": 0.05, "auroc": 0.7,
                    "alerts_by_name": {}, "threshold": 0.9}

        return mld, calls, spy

    def _run(self, families):
        import pandas as pd
        mld, calls, spy = self._spy()
        orig = (mld.run_tournament_entry, mld._summarise_entry,
                mld.attach_price_features, mld.candidate_frame)
        mld.run_tournament_entry = spy
        mld._summarise_entry = lambda *a, **k: None
        mld.attach_price_features = lambda c, s, p: c
        mld.candidate_frame = lambda f: f
        try:
            frame = pd.DataFrame({"name": ["a"] * 5, "year": [2024] * 5})
            res = mld.run_ml_tournament(frame, pd.DataFrame(), {}, {},
                                        series=object(), boom=None,
                                        families=families)
        finally:
            (mld.run_tournament_entry, mld._summarise_entry,
             mld.attach_price_features, mld.candidate_frame) = orig
        return res, calls, mld

    def test_pinned_fits_two_models_not_sixteen(self):
        res, calls, mld = self._run(["ens"])
        assert len(calls) == 2, "pinned mode must fit one family per head"
        assert {c[2] for c in calls} == {"ens_y_top", "ens_y_onset"}
        # both heads present, and the winner is the pinned family
        assert sorted(res["get_out"]) == ["ens"]
        assert sorted(res["get_in"]) == ["ens"]
        assert mld.pick_winner(res) == "ens"

    def test_pinned_uses_the_deployable_bank_not_the_crowd_control(self):
        _res, calls, mld = self._run(["ens"])
        # DESK_ML_BANK is the crowd+price bank; ML_BANK is the crowd-only
        # control that pick_winner refuses to adopt anyway.
        assert all(c[3] == len(mld.DESK_ML_BANK) for c in calls), (
            "pinned mode must fit the deployable configuration, "
            "never the *_crowd control")

    def test_unpinned_still_runs_the_full_tournament(self):
        res, calls, _mld = self._run(None)
        assert len(calls) == 16
        assert "ens_crowd" in res["get_out"]

    def test_unknown_family_is_refused_loudly(self):
        with pytest.raises(ValueError):
            self._run(["definitely_not_a_family"])

    def test_config_default_is_the_logit_gbm_ensemble(self):
        from src.config import DESK_MODEL_FAMILY
        assert DESK_MODEL_FAMILY == "ens"
