"""On-demand ETF lookup: resolve, holdings, words, assembly, scoring.

Every network call is injected, so the suite runs offline. The assembly
tests use a small synthetic aggregate set shaped like the real stores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import lookup as L                                   # noqa: E402


def _terminal(query, limit):
    return [L.Candidate("EWZ", "iShares MSCI Brazil ETF", "search"),
            L.Candidate("BRZU", "Direxion Daily Brazil Bull 2X", "search")]


def _holdings(symbol):
    return [{"ticker": "VALE", "name": "Vale S.A.", "weight": 0.12},
            {"ticker": "PBR", "name": "Petroleo Brasileiro S.A.", "weight": 0.10},
            {"ticker": "BRK-B", "name": "Berkshire Hathaway", "weight": 0.02},
            {"ticker": "PETR4.SA", "name": "Petrobras", "weight": 0.05}]


class TestResolve:
    def test_configured_theme_matches_before_the_terminal(self):
        out = L.search("gold", remote=lambda q, l: [])
        assert out and out[0].symbol == "GLD" and out[0].origin == "approved"

    def test_terminal_results_are_appended_and_deduplicated(self):
        out = L.search("brazil", remote=_terminal)
        syms = [c.symbol for c in out]
        assert "EWZ" in syms and "BRZU" in syms
        assert len(syms) == len(set(syms))

    def test_offline_search_still_returns_local_matches(self):
        def boom(q, l):
            raise ConnectionError("no network")
        out = L.search("semi", remote=boom)
        assert any(c.symbol == "SMH" for c in out)

    def test_empty_query(self):
        assert L.search("   ", remote=_terminal) == []


class TestHoldingsAndWords:
    def test_known_flag_uses_universe_or_aggregates(self):
        hs = L.holdings("EWZ", remote=_holdings, universe={"VALE"},
                        present={"PBR"})
        known = {h.ticker for h in hs if h.known}
        assert known == {"VALE", "PBR"}
        assert any(h.ticker == "PETR4.SA" and not h.known for h in hs)

    def test_vendor_dash_class_becomes_dot(self):
        hs = L.holdings("X", remote=_holdings, universe={"BRK.B"})
        assert any(h.ticker == "BRK.B" and h.known for h in hs)

    def test_holdings_failure_is_empty_not_fatal(self):
        def boom(s):
            raise RuntimeError("terminal down")
        assert L.holdings("EWZ", remote=boom, universe=set()) == []

    def test_words_drop_issuer_noise_and_keep_theme_tokens(self):
        hs = L.holdings("EWZ", remote=_holdings, universe=set())
        w = L.words("iShares MSCI Brazil ETF", hs)
        assert w[0] == "brazil"
        assert "ishares" not in w and "etf" not in w and "msci" not in w
        assert "vale" in w and "petroleo" in w

    def test_ai_expansion_is_a_noop_without_a_provider(self, monkeypatch):
        from src import ai
        monkeypatch.setattr(ai, "available", lambda: False)
        assert L.expand_words_with_ai("x", ["brazil"]) == ["brazil"]


def _aggregates():
    days = pd.date_range("2026-01-01", periods=400, freq="D")
    tc = pd.concat([
        pd.DataFrame({"date": days, "ticker": "VALE", "mention_count": 3}),
        pd.DataFrame({"date": days, "ticker": "PBR", "mention_count": 2}),
        pd.DataFrame({"date": days, "ticker": "NVDA", "mention_count": 50}),
    ], ignore_index=True)
    ts = pd.concat([
        pd.DataFrame({"date": days, "ticker": "VALE", "n_posts": 3,
                      "avg_sentiment": 0.2, "net_bullish": 0.1}),
        pd.DataFrame({"date": days, "ticker": "PBR", "n_posts": 1,
                      "avg_sentiment": -0.4, "net_bullish": -0.5}),
    ], ignore_index=True)
    term = pd.concat([
        pd.DataFrame({"date": days, "term": "__TOTAL__", "mention_count": 500}),
        pd.DataFrame({"date": days, "term": "brazil", "mention_count": 4}),
        pd.DataFrame({"date": days, "term": "nuclear", "mention_count": 9}),
    ], ignore_index=True)
    thc = pd.concat([
        pd.DataFrame({"date": days, "theme": "semiconductors", "mention_count": 100}),
        pd.DataFrame({"date": days, "theme": "gold_metals", "mention_count": 20}),
    ], ignore_index=True)
    ths = pd.concat([
        pd.DataFrame({"date": days, "theme": "semiconductors", "n_posts": 100,
                      "avg_sentiment": 0.1, "net_bullish": 0.05}),
        pd.DataFrame({"date": days, "theme": "gold_metals", "n_posts": 20,
                      "avg_sentiment": 0.0, "net_bullish": 0.0}),
    ], ignore_index=True)
    return tc, ts, term, thc, ths


class TestAssembleAndScore:
    def _spec(self):
        hs = L.holdings("EWZ", remote=_holdings, universe={"VALE", "PBR"})
        return L.LookupSpec("EWZ", "iShares MSCI Brazil ETF", holdings=hs,
                            words=["brazil"])

    def test_assembly_sums_holdings_and_words(self):
        tc, ts, term, thc, ths = _aggregates()
        fr = L.assemble(self._spec(), tc, ts, term)
        row = fr.iloc[0]
        assert row["ticker_mentions"] == 5          # VALE 3 + PBR 2
        assert row["term_mentions"] == 4            # brazil only
        assert row["mention_count"] == 9
        assert row["n_posts"] == 4
        # n_posts-weighted: (3*0.2 + 1*-0.4) / 4
        assert abs(row["avg_sentiment"] - 0.05) < 1e-9

    def test_unknown_holdings_and_unlisted_words_contribute_nothing(self):
        tc, ts, term, thc, ths = _aggregates()
        spec = L.LookupSpec("X", "x", holdings=[
            L.Holding("PETR4.SA", "Petrobras", 0.1, known=False)],
            words=["unicorn"])
        fr = L.assemble(spec, tc, ts, term)
        assert fr.empty or fr["mention_count"].sum() == 0

    def test_scoring_runs_and_share_is_against_theme_total(self):
        tc, ts, term, thc, ths = _aggregates()
        spec = self._spec()
        fr = L.assemble(spec, tc, ts, term)
        es = L.euphoria_series(spec, fr, thc, ths)
        assert es is not None and es.name == "lookup:EWZ"
        assert es.level.dropna().between(0, 100).all()
        share = L.discussion_share(fr, thc)
        # 9 a day vs 120 a day of configured themes, over 7 days
        assert abs(share - 100 * 9 / (9 + 120)) < 0.05

    def test_price_is_cached_and_prefers_the_store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(L, "LOOKUP_PRICE_DIR", str(tmp_path / "lookups"))
        # a symbol NOT in the store goes through fetch, once
        calls = []

        def fetch(sym, a, b):
            calls.append(sym)
            return pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]),
                                 "symbol": [sym], "px_last": [1.0],
                                 "source": ["tiingo"]})
        # Saturday 3 Jan 2026: Friday's close (2 Jan) is the last one
        now = pd.Timestamp("2026-01-03 10:00")
        a = L.price("ZZZZ", "20260101", "20260110", fetch=fetch, now=now)
        b = L.price("ZZZZ", "20260101", "20260110", fetch=fetch, now=now)
        assert len(a) == 1 and len(b) == 1 and calls == ["ZZZZ"]

    def test_the_dashboard_panel_reads_its_title_from_settings(self):
        from src import settings
        assert settings.get("lookup_section_title")
        src = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        assert "def render_etf_lookup(" in src
        assert "applied read-only at its frozen cuts" in src
        assert "capture rate or lead time is claimed" in src


class TestOfflineResolution:
    def test_catalogue_answers_by_name_keyword_and_symbol(self):
        def down(q, l):
            raise RuntimeError("rate limited")
        assert L.search("INDA", remote=down)[0].symbol == "INDA"
        assert {c.symbol for c in L.search("india", remote=down)} >= {"INDA", "INDY"}
        assert any(c.symbol == "URA" for c in L.search("uranium", remote=down))

    def test_remote_failure_is_reported_not_hidden(self):
        def down(q, l):
            raise RuntimeError("YFRateLimitError")
        st = {}
        L.search("INDA", remote=down, status=st)
        assert "YFRateLimitError" in st["remote_error"]
        st = {}
        L.search("INDA", remote=lambda q, l: [], status=st)
        assert st["remote_error"] is None

    def test_unknown_text_that_looks_like_a_ticker_is_offered_as_one(self):
        out = L.search("XYZQ", remote=lambda q, l: [])
        assert [(c.symbol, c.origin) for c in out] == [("XYZQ", "ticker")]
        # a word that matches by name is NOT also offered as a ticker
        assert not any(c.origin == "ticker" for c in L.search("india", remote=lambda q, l: []))

    def test_manual_holdings_are_equal_weight_and_flagged(self):
        hs = L.manual_holdings("INFY, hdb ,IBN", universe={"INFY", "IBN"})
        assert [h.ticker for h in hs] == ["INFY", "HDB", "IBN"]
        assert all(abs(h.weight - 1 / 3) < 1e-9 for h in hs)
        assert [h.known for h in hs] == [True, False, True]

    def test_catalogue_file_is_well_formed(self):
        cat = L._catalogue()
        assert {"symbol", "name", "keywords"} <= set(cat.columns)
        assert cat["symbol"].is_unique
        assert len(cat) > 100


# ---------------------------------------------------------------------------
# the persisted production model
# ---------------------------------------------------------------------------
def _tiny_train(n=400, seed=0):
    import numpy as np
    from src.analytics.ml_detector import DESK_ML_BANK
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(DESK_ML_BANK))), columns=DESK_ML_BANK)
    heat = X[DESK_ML_BANK].mean(axis=1)
    X["y_top"] = (heat + 0.15 * rng.standard_normal(n) > 0.6).astype(int)
    X["y_onset"] = (heat + 0.15 * rng.standard_normal(n) > 0.55).astype(int)
    X["name"] = "n"
    X["date"] = pd.date_range("2024-01-01", periods=n)
    return X


class TestModelBundle:
    def test_round_trip_reproduces_the_live_scores(self, tmp_path):
        import numpy as np
        from src.analytics import ml_detector as mld
        train = _tiny_train()
        live = _tiny_train(seed=1)
        fits = {"y_top": mld.make_ens_fit("y_top"),
                "y_onset": mld.make_ens_fit("y_onset")}
        live_scores = {k: f(train, live, mld.DESK_ML_BANK) for k, f in fits.items()}
        path = str(tmp_path / "bundle.joblib")
        mld.save_desk_bundle(path, fits, mld.DESK_ML_BANK,
                             {"thr_in": 0.9, "thr_out": 0.9})
        b = mld.load_desk_bundle(path)
        assert b["feats"] == mld.DESK_ML_BANK and set(b["heads"]) == {"y_top", "y_onset"}
        for k in fits:
            again = mld.score_with_bundle(b, k, live)
            # the ensemble score is a rank within the scored population;
            # re-ranking each row against the stored population must give
            # the same number (up to tie handling) as the live pass did
            assert np.nanmax(np.abs(again - live_scores[k])) < 0.02

    def test_a_new_row_is_ranked_against_the_stored_population(self):
        import numpy as np
        from src.analytics import ml_detector as mld
        pop = np.sort(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        r = mld._rank_within(pop, np.array([0.05, 0.35, 0.9]))
        assert r[0] < r[1] < r[2] and 0 < r[0] and r[2] <= 1.0

    def test_unfitted_or_non_ensemble_fits_are_refused(self, tmp_path):
        from src.analytics import ml_detector as mld
        with pytest.raises(ValueError):
            mld.save_desk_bundle(str(tmp_path / "x.joblib"),
                                 {"y_top": mld.make_logit_fit("y_top")},
                                 mld.DESK_ML_BANK, {})

    def test_model_score_without_a_bundle_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(L, "model_bundle_path",
                            lambda: str(tmp_path / "absent.joblib"))
        r = L.model_score(L.LookupSpec("X", "x"), None, pd.DataFrame(), None,
                          pd.DataFrame(), pd.DataFrame())
        assert r["status"] == "no_model"

    def test_the_pipeline_saves_the_bundle_and_publishes_it(self):
        src = (ROOT / "src" / "analytics" / "euphoria_phases.py").read_text(encoding="utf-8")
        assert "mld.save_desk_bundle(" in src
        pub = (ROOT / "tools" / "publish_dashboard.py").read_text(encoding="utf-8")
        assert "euphoria_desk_model.joblib" in pub


class TestBloombergFirst:
    def test_bulk_rows_are_mapped_to_holdings(self):
        rows = [{"Member Ticker and Exchange Code": "INFY US", "Percentage Weight": "7.9"},
                {"Member Ticker and Exchange Code": "RELIANCE IN", "Percentage Weight": "6.2"},
                {"Holding Ticker": "HDB US Equity", "Weight": "0.045", "Name": "HDFC Bank"}]
        hs = L._bbg_rows_to_holdings(rows)
        assert [h["ticker"] for h in hs] == ["INFY", "RELIANCE IN", "HDB"]
        assert abs(hs[0]["weight"] - 0.079) < 1e-9        # percent -> fraction
        assert abs(hs[2]["weight"] - 0.045) < 1e-9        # already a fraction
        assert hs[2]["name"] == "HDFC Bank"

    def test_no_terminal_means_search_is_local_only(self, monkeypatch):
        """Without a Terminal the remote search raises (so the panel can
        say so) and the local sources still answer."""
        monkeypatch.setattr(L, "bloomberg_available", lambda refresh=False: False)
        with pytest.raises(RuntimeError, match="local only"):
            L._remote_search("india", 5)
        st = {}
        out = L.search("india", status=st)
        assert any(c.symbol == "INDA" for c in out)
        assert "local only" in st["remote_error"]

    def test_bloomberg_failure_is_reported_not_hidden(self, monkeypatch):
        monkeypatch.setattr(L, "bloomberg_available", lambda refresh=False: True)

        def bbg(q, l):
            raise RuntimeError("session down")
        monkeypatch.setattr(L, "_bloomberg_search", bbg)
        st = {}
        L.search("india", status=st)
        assert "session down" in st["remote_error"]

    def test_bloomberg_answer_wins_when_it_has_rows(self, monkeypatch):
        monkeypatch.setattr(L, "bloomberg_available", lambda refresh=False: True)
        monkeypatch.setattr(L, "_bloomberg_holdings",
                            lambda s: [{"ticker": "INFY", "name": "Infosys", "weight": 0.08}])
        assert L._remote_holdings("INDA")[0]["ticker"] == "INFY"

    def test_no_terminal_means_holdings_must_be_typed(self, monkeypatch):
        monkeypatch.setattr(L, "bloomberg_available", lambda refresh=False: False)
        st = {}
        assert L.holdings("INDA", universe=set(), status=st) == []
        assert "type the holdings" in st["remote_error"]


class TestPriceCacheFreshness:
    def test_cache_that_stops_before_the_last_close_is_refreshed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(L, "LOOKUP_PRICE_DIR", str(tmp_path / "lookups"))
        calls = []

        def fetch_through(last):
            def fetch(sym, a, b):
                calls.append(last)
                return pd.DataFrame({"date": pd.to_datetime([last]), "symbol": [sym],
                                     "px_last": [1.0], "source": ["tiingo"]})
            return fetch
        # Wednesday 10 Sep 2026, 09:00: the cache holds Monday's close only
        now = pd.Timestamp("2026-09-10 09:00")
        L.price("ZZZZ", "20260901", "20260910", fetch=fetch_through("2026-09-08"), now=now)
        # a second look the same morning must re-pull (Tuesday's close exists)
        out = L.price("ZZZZ", "20260901", "20260910", fetch=fetch_through("2026-09-09"), now=now)
        assert calls == ["2026-09-08", "2026-09-09"]
        assert pd.to_datetime(out["date"]).max() == pd.Timestamp("2026-09-09")
        # and now the cache is trusted
        L.price("ZZZZ", "20260901", "20260910", fetch=fetch_through("x"), now=now)
        assert len(calls) == 2

    def test_no_price_provider_returns_nothing_rather_than_raising(
            self, tmp_path, monkeypatch):
        """A copy with no Terminal and no API key is the normal hosted
        state. The lookup must degrade to an unpriced panel; a raise here
        takes the whole page down."""
        from src.prices import ProviderUnavailable
        monkeypatch.setattr(L, "LOOKUP_PRICE_DIR", str(tmp_path / "lookups"))

        def boom(sym, a, b):
            raise ProviderUnavailable("no price provider is usable")

        out = L.price("ZZZZ", "20260901", "20260910", fetch=boom)
        assert out.empty
        assert list(out.columns) == ["date", "symbol", "px_last", "source"]

    def test_provider_status_reports_instead_of_raising(self, monkeypatch):
        from src import prices as P
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (False, "no Terminal"))
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (False, "no key"))
        ok, why = P.provider_status()
        assert ok is False and "TIINGO_API_KEY" in why
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (True, "key present"))
        ok, why = P.provider_status()
        assert ok is True and "tiingo" in why

    def test_the_panel_cannot_take_the_page_down(self):
        """Streamlit gives a script ONE error boundary: an exception
        anywhere ends the whole page. The lookup reaches outside the
        committed stores (a typed symbol, a Terminal, a price provider),
        so its call site keeps its own guard."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1]
               / "dashboard.py").read_text(encoding="utf-8")
        i = src.rindex("render_etf_lookup(key_prefix)")   # the CALL, not the def
        assert "def render_etf_lookup" not in src[i - 40:i]
        block = src[i - 500:i + 500]
        assert "try:" in block and "except Exception" in block, (
            "the ETF lookup call site lost its guard - one failure there "
            "would take the landing page with it")
        assert "st.error(" in block

    def test_last_trading_day_skips_weekends(self):
        assert L.last_trading_day(pd.Timestamp("2026-09-14 09:00")) == pd.Timestamp("2026-09-11")  # Mon -> Fri
        assert L.last_trading_day(pd.Timestamp("2026-09-10 09:00")) == pd.Timestamp("2026-09-09")  # Wed -> Tue
