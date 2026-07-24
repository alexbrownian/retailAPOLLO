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


# ---------------------------------------------------------------------------
# 9. THE EUPHORIA DETECTOR - the top-calling rules
# ---------------------------------------------------------------------------
class TestEuphoria:
    def test_convexity_flags_super_exponential(self):
        """A super-exponential (accelerating) price path must show positive
        log-convexity; a steady exponential must not (Sornette signature)."""
        from analytics.euphoria import log_price_convexity
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
        from analytics.euphoria import ground_truth_peaks
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
        from analytics.euphoria import trailing_pct_rank
        idx = pd.date_range("2024-01-01", periods=400, freq="D")
        a = pd.Series(np.sin(np.arange(400) / 20.0) + 2, index=idx)
        b = a.copy(); b.iloc[300:] += 100                     # future shock
        ra = trailing_pct_rank(a, window=180, min_periods=90)
        rb = trailing_pct_rank(b, window=180, min_periods=90)
        pd.testing.assert_series_equal(ra.iloc[:300], rb.iloc[:300])

    def test_euphoria_is_reddit_only(self):
        """THE desk rule: compute_euphoria must not accept price input -
        prediction is built from the crowd data alone (price exists only
        on the ground-truth/scoring side)."""
        import inspect
        from analytics.euphoria import compute_euphoria
        params = inspect.signature(compute_euphoria).parameters
        assert not any("px" in p or "price" in p for p in params), \
            "compute_euphoria takes a price argument - Reddit-only rule broken"

    def test_judgeable_window_excludes_unpriced_eras(self):
        """Alerts before price history (or too close to its end) must be
        PENDING, not false alarms - scoring them was the 2018-20 FA bug."""
        from analytics.euphoria import judgeable_window
        idx = pd.date_range("2021-01-01", periods=200, freq="D")
        px = pd.Series(100.0, index=idx)
        j0, j1 = judgeable_window(px)
        assert j0 == idx[0]
        assert j1 == idx[-1] - pd.Timedelta(days=45)


class TestInfluence:
    def test_vol_scaled_bar(self):
        """The correctness bar must rise with the name's own volatility
        (thesis 4.5): tau = max(3%, 0.5 sigma) - one fixed bar would
        misgrade an index ETF and a meme stock with the same ruler."""
        from analytics.influence import score_calls
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
        from analytics.influence import _shrink
        lucky = _shrink(1.0, 3, 10, 0.5)
        veteran = _shrink(40 / 60, 60, 10, 0.5)
        assert veteran > lucky

    def test_pagerank_ranks_the_hub(self):
        """On a star-ish graph the most-connected node must carry the
        highest PageRank, and broadcast/bot accounts must be excluded."""
        from analytics.influence import build_graph_metrics
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
        text-free contract ABSTRACTED_DATA lives under."""
        from analytics.influence import _safe_store_write
        bad = pd.DataFrame({"author": ["x"], "body": ["secret text"]})
        with pytest.raises(RuntimeError):
            _safe_store_write(bad, "/tmp/should_never_exist.parquet")


class TestEuphoriaPhases:
    """The July-2026 phases study (onset detector + episode ground truth,
    analytics/euphoria_phases.py). Same invariants philosophy as
    TestEuphoria: the crowd-only rule, no look-ahead, honest windows."""

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
        """THE desk rule extends to the onset detector: neither the
        feature builder nor the score accepts price input."""
        import inspect
        from analytics.euphoria_phases import (compute_onset_features,
                                               onset_score)
        for fn in (compute_onset_features, onset_score):
            params = inspect.signature(fn).parameters
            assert not any("px" in p or "price" in p for p in params), \
                f"{fn.__name__} takes a price argument - crowd-only rule broken"

    def test_onset_window_capped_at_peak(self):
        """The onset hit window may never extend past the peak - an alert
        AFTER the top must not count as 'caught the start' (fast rallies
        like GME would otherwise be gamed by late alerts)."""
        from analytics.euphoria_phases import find_episodes, \
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
        from analytics.euphoria import ground_truth_peaks
        from analytics.euphoria_phases import find_episodes
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
        from analytics.euphoria_phases import alerts_from_scores
        dates = list(pd.date_range("2024-01-01", periods=100, freq="D"))
        alerts = alerts_from_scores(dates, [1.0] * 100, threshold=0.5)
        assert alerts, "no alerts fired"
        gaps = np.diff([a.value for a in alerts]) / 86_400_000_000_000
        assert (gaps >= 21).all(), "cooldown violated"

    def test_onset_alert_classification_buckets(self):
        """HIT inside [trough, capped end]; LATE inside (end, peak];
        FA elsewhere - LATE is neither a hit nor a false alarm."""
        from analytics.euphoria_phases import (classify_onset_alerts,
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
        from analytics.euphoria_phases import compute_onset_features
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
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "processed",
            "euphoria_onset.parquet")
        if os.path.exists(path):
            cols = set(pd.read_parquet(path).columns)
            assert not (cols & FORBIDDEN_COLS), \
                f"forbidden columns in onset store: {cols & FORBIDDEN_COLS}"


class TestDynamicPanel:
    """The dynamic subreddit panel (ingestion/discover_subreddits.py,
    desk decisions 2026-07-24): crowd-referral discovery with the A0
    coverage floor reused as the qualification bar, a same-ruler finance
    screen, a 1-add/review cap and a committed audit manifest."""

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
    """Desk decision 2026-07-24: research decides once, live scores.
    The walk-forward trains each year's threshold on STRICTLY EARLIER
    years, so within a calendar year a daily recompute is a no-op - the
    pipeline must therefore run at frozen thresholds and re-derive them
    only when the data rolls into an uncovered year (or on --research)."""

    def test_needs_research_triggers(self):
        from analytics.euphoria import needs_research
        stored = {"thresholds": {"2024": 85, "2025": 85, "2026": 85}}
        assert needs_research(None, 2026)            # no report yet
        assert needs_research({}, 2026)              # empty report
        assert not needs_research(stored, 2026)      # covered year: frozen
        assert needs_research(stored, 2027)          # year rolled over

    def test_onset_needs_research_triggers(self):
        from analytics.euphoria_phases import onset_needs_research
        stored = {"live_threshold": 0.89,
                  "walk_forward": {"test_years": [2024, 2025, 2026]}}
        assert onset_needs_research(None, 2026)
        assert onset_needs_research({"walk_forward": {}}, 2026)
        assert not onset_needs_research(stored, 2026)
        assert onset_needs_research(stored, 2027)


class TestEpisodeCoherence:
    """Desk rule 2026-07-24, ASYMMETRIC by measurement: a START within
    one 21d cooldown after an END is suppressed (contradictory flip);
    an END after a START is NEVER suppressed - fast manias genuinely
    run start-to-end inside a cooldown, and the symmetric rule cost the
    top detector half its captures (17->9) when tested."""

    def test_start_after_end_is_suppressed(self):
        from analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        o, tp = episode_coherent_alerts([t("2026-01-11")], [t("2026-01-01")])
        assert o == [] and tp == [t("2026-01-01")]

    def test_fast_mania_end_is_never_suppressed(self):
        from analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        # end fires 5d after start: a violent mania - BOTH survive
        o, tp = episode_coherent_alerts([t("2026-01-01")], [t("2026-01-06")])
        assert o == [t("2026-01-01")] and tp == [t("2026-01-06")]

    def test_separated_phases_both_survive(self):
        from analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        o, tp = episode_coherent_alerts([t("2026-01-01")], [t("2026-03-01")])
        assert o == [t("2026-01-01")] and tp == [t("2026-03-01")]

    def test_same_day_tie_goes_to_the_risk_signal(self):
        from analytics.euphoria_phases import episode_coherent_alerts
        t = pd.Timestamp
        o, tp = episode_coherent_alerts([t("2026-01-01")], [t("2026-01-01")])
        assert tp == [t("2026-01-01")] and o == []
