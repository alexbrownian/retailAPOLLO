"""Price-provider layer: selection, fallback, reshaping, source tracking.

The Bloomberg path needs a Terminal and the Tiingo path needs a key and
network, so these tests stub both at the boundary (``blpapi`` import and
the Tiingo HTTP call) and check the logic around them: which provider is
chosen for each preference, that a failing provider falls through to
the next, that Tiingo's JSON becomes the long ``date, symbol, px_last,
source`` schema with a split-adjusted close, and that a symbol whose
stored rows came from another provider is re-pulled in full rather than
spliced.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import prices as P                                   # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tiingo_rows(days=3, start="2026-08-03"):
    idx = pd.date_range(start, periods=days, freq="B")
    return [{"date": f"{d:%Y-%m-%d}T00:00:00.000Z", "close": 10.0 + i,
             "adjClose": 9.0 + i, "splitFactor": 1.0} for i, d in enumerate(idx)]


def _stub_tiingo(monkeypatch, answer):
    """Replace the HTTP call: ``answer(sym, start, end) -> json rows``."""
    monkeypatch.setattr(P.TiingoProvider, "_get",
                        lambda self, sym, a, b: answer(sym, a, b))
    monkeypatch.setattr(P.TiingoProvider, "PAUSE_S", 0.0)
    monkeypatch.setattr(P, "tiingo_key", lambda: "test-key")


class _Quiet:
    """A log sink that records lines."""
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(str(msg))


# ---------------------------------------------------------------------------
# symbol mapping
# ---------------------------------------------------------------------------
class TestSymbolMapping:
    def test_us_ticker_maps_to_itself_for_both_providers(self):
        assert P.to_bloomberg("NVDA") == "NVDA US Equity"
        assert P.to_tiingo("NVDA") == "NVDA"

    def test_dotted_share_class_becomes_dashed_for_tiingo(self):
        assert P.to_tiingo("BRK.B") == "BRK-B"

    def test_foreign_line_uses_the_csv_mapping(self):
        assert P.to_bloomberg("1622 JT") == "1622 JT Equity"
        # a foreign line with no tiingo mapping is skipped, not guessed
        assert P.to_tiingo("1622 JT") is None

    def test_foreign_line_without_a_mapping_is_skipped_not_guessed(self,
                                                                    monkeypatch):
        monkeypatch.setattr(P, "_approved_rows", lambda: {
            "XYZ LN": {"symbol": "XYZ LN", "bloomberg": "XYZ LN Equity",
                       "tiingo": ""}})
        assert P.to_tiingo("XYZ LN") is None


# ---------------------------------------------------------------------------
# tiingo parsing and fetching
# ---------------------------------------------------------------------------
class TestTiingo:
    def test_rows_become_long(self):
        out = P.TiingoProvider._to_long(_tiingo_rows(3), "SPY")
        assert list(out.columns) == P.COLUMNS
        assert out["px_last"].tolist() == [10.0, 11.0, 12.0]
        assert (out["source"] == "tiingo").all() and (out["symbol"] == "SPY").all()
        assert out["date"].dt.tz is None
        assert out["date"].iloc[0] == pd.Timestamp("2026-08-03")

    def test_close_is_split_adjusted_but_not_dividend_adjusted(self):
        """A 4-for-1 split on day 2 must divide day 1's print by 4 and
        leave later prints alone; adjClose (which also strips dividends)
        is deliberately not used."""
        rows = [{"date": "2026-08-03T00:00:00.000Z", "close": 100.0, "adjClose": 24.0, "splitFactor": 1.0},
                {"date": "2026-08-04T00:00:00.000Z", "close": 25.5, "adjClose": 25.0, "splitFactor": 4.0},
                {"date": "2026-08-05T00:00:00.000Z", "close": 26.0, "adjClose": 25.5, "splitFactor": 1.0}]
        out = P.TiingoProvider._to_long(rows, "X")
        assert out["px_last"].tolist() == [25.0, 25.5, 26.0]

    def test_empty_and_garbage_give_empty_frames(self):
        assert P.TiingoProvider._to_long([], "SPY").empty
        assert P.TiingoProvider._to_long([{"foo": 1}], "SPY").empty

    def test_fetch_maps_symbols_and_skips_unmappable(self, monkeypatch):
        asked = []

        def answer(sym, a, b):
            asked.append((sym, a, b))
            return _tiingo_rows(2)
        _stub_tiingo(monkeypatch, answer)
        log = _Quiet()
        out = P.TiingoProvider().fetch(["SPY", "BRK.B", "1622 JT", "XYZ LN"],
                                       "20260803", "20260805", log=log)
        assert [a[0] for a in asked] == ["SPY", "BRK-B"]
        assert asked[0][1:] == ("20260803", "20260805")
        assert set(out["symbol"]) == {"SPY", "BRK.B"}
        assert any("1622 JT" in l for l in log.lines)

    def test_quota_answer_raises_instead_of_storing_it(self, monkeypatch):
        def quota(sym, a, b):
            raise RuntimeError("tiingo: request quota reached for this key")
        _stub_tiingo(monkeypatch, quota)
        with pytest.raises(RuntimeError, match="quota"):
            P.TiingoProvider().fetch(["SPY"], "20260803", "20260805", log=_Quiet())

    def test_unavailable_without_a_key(self, monkeypatch):
        monkeypatch.setattr(P, "tiingo_key", lambda: "")
        ok, why = P.TiingoProvider().available()
        assert not ok and "TIINGO_API_KEY" in why


# ---------------------------------------------------------------------------
# provider selection and fallback
# ---------------------------------------------------------------------------
class TestProviderChain:
    def test_tiingo_preference_is_tiingo_only(self, monkeypatch):
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("tiingo", log=_Quiet())
        assert [c.name for c in chain] == ["tiingo"]

    def test_bloomberg_preference_falls_back_to_tiingo(self, monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (True, "ok"))
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("bloomberg", log=_Quiet())
        assert [c.name for c in chain] == ["bloomberg", "tiingo"]
        chain = P.provider_chain("bloomberg", fallback=False, log=_Quiet())
        assert [c.name for c in chain] == ["bloomberg"]

    def test_auto_skips_bloomberg_when_no_terminal(self, monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (False, "no terminal"))
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("auto", log=_Quiet())
        assert [c.name for c in chain] == ["tiingo"]

    def test_auto_prefers_bloomberg_when_reachable(self, monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (True, "ok"))
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("auto", log=_Quiet())
        assert [c.name for c in chain] == ["bloomberg", "tiingo"]

    def test_bloomberg_only_and_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (False, "no terminal"))
        with pytest.raises(P.ProviderUnavailable):
            P.provider_chain("bloomberg", fallback=False, log=_Quiet())

    def test_unknown_preference_is_rejected(self):
        with pytest.raises(ValueError):
            P.provider_chain("alphavantage")

    def test_fetch_falls_through_on_exception(self, monkeypatch):
        class Boom:
            name = "bloomberg"

            def fetch(self, *a, **k):
                raise RuntimeError("Terminal not running")

        class Good:
            name = "tiingo"

            def fetch(self, symbols, start, end, log=print):
                return pd.DataFrame({"date": pd.to_datetime(["2026-08-03"]),
                                     "symbol": ["SPY"], "px_last": [1.0],
                                     "source": ["tiingo"]})
        log = _Quiet()
        df, used = P.fetch_with_fallback([Boom(), Good()], ["SPY"],
                                         "20260803", "20260803", log=log)
        assert used == "tiingo" and len(df) == 1
        assert any("falling back to tiingo" in l for l in log.lines)

    def test_an_empty_answer_is_not_a_failure(self):
        """A symbol that did not trade over the span comes back with no
        rows from a provider that connected fine. That must not abort the
        pull or fall through to the next provider (which would splice
        sources): the empty frame is returned under the same provider."""
        calls = []

        class Empty:
            name = "bloomberg"

            def fetch(self, symbols, start, end, log=print):
                calls.append("bloomberg")
                return pd.DataFrame(columns=P.COLUMNS)

        class Never:
            name = "tiingo"

            def fetch(self, *a, **k):
                calls.append("tiingo")
                raise AssertionError("fallback must not be consulted")
        df, used = P.fetch_with_fallback([Empty(), Never()], ["NEWCO"],
                                         "20210101", "20210111", log=_Quiet())
        assert used == "bloomberg" and df.empty and calls == ["bloomberg"]
        assert list(df.columns) == P.COLUMNS

    def test_every_provider_failing_raises(self):
        class Boom:
            name = "x"

            def fetch(self, *a, **k):
                raise RuntimeError("down")
        with pytest.raises(P.ProviderUnavailable):
            P.fetch_with_fallback([Boom()], ["SPY"], "20260803",
                                  "20260803", log=_Quiet())


# ---------------------------------------------------------------------------
# one symbol, one source
# ---------------------------------------------------------------------------
class TestTerminalIsNotRetriedAfterItFails:
    """A session can start while //blp/refdata cannot be opened, and
    openService only discovers that by blocking for about two minutes. A
    multi-span pull must pay that once, not once per span."""

    @staticmethod
    def _always_fails(monkeypatch):
        monkeypatch.setattr(
            "ingestion.pull_prices.bloomberg_request",
            lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("could not open //blp/refdata service")))

    def test_the_terminal_is_dropped_after_the_allowed_failures(
            self, monkeypatch):
        import src.prices as P
        P.reset_bloomberg_failure()
        monkeypatch.setattr(P, "bloomberg_max_failures", lambda: 2)
        self._always_fails(monkeypatch)
        bbg = P.BloombergProvider()
        assert P.bloomberg_failed_this_process() == ""
        with pytest.raises(RuntimeError):                  # first failure
            bbg.fetch(["SPY"], "20260101", "20260201", log=lambda *_: None)
        assert P.bloomberg_failed_this_process() == "", \
            "one failure is a blip, not a verdict"
        with pytest.raises(RuntimeError):                  # second failure
            bbg.fetch(["SPY"], "20260101", "20260201", log=lambda *_: None)
        assert "refdata" in P.bloomberg_failed_this_process()
        ok, why = bbg.available()
        assert ok is False and "not retried" in why
        P.reset_bloomberg_failure()

    def test_the_allowance_comes_from_settings(self, monkeypatch):
        import src.prices as P
        P.reset_bloomberg_failure()
        monkeypatch.setattr("src.settings.get_int", lambda k: 1)
        assert P.bloomberg_max_failures() == 1
        self._always_fails(monkeypatch)
        with pytest.raises(RuntimeError):
            P.BloombergProvider().fetch(["SPY"], "20260101", "20260201",
                                        log=lambda *_: None)
        assert P.bloomberg_failed_this_process() != "", \
            "with an allowance of 1 the first failure is the last"
        P.reset_bloomberg_failure()

    def test_a_reused_chain_skips_the_dropped_provider(self, monkeypatch):
        """pull_prices builds the chain ONCE and reuses it for every span,
        so the skip has to happen inside fetch_with_fallback."""
        import src.prices as P
        P.reset_bloomberg_failure()
        monkeypatch.setattr(P, "bloomberg_max_failures", lambda: 2)
        self._always_fails(monkeypatch)
        monkeypatch.setattr(P.TiingoProvider, "fetch",
                            lambda self, s, a, b, log=print: pd.DataFrame(
                                {"date": pd.to_datetime(["2026-01-02"]),
                                 "symbol": list(s)[:1], "px_last": [1.0],
                                 "source": ["tiingo"]}))
        chain = [P.BloombergProvider(), P.TiingoProvider()]
        lines = []
        for _ in range(3):            # three spans, as a real pull has
            df, used = P.fetch_with_fallback(chain, ["SPY"], "20260101",
                                             "20260201", log=lines.append)
            assert used == "tiingo" and len(df) == 1
        tried = sum(1 for line in lines if "via bloomberg" in line)
        skipped = sum(1 for line in lines if "skipping bloomberg" in line)
        assert (tried, skipped) == (2, 1), (tried, skipped, lines)
        P.reset_bloomberg_failure()

    def test_the_chain_drops_the_terminal_after_that(self, monkeypatch):
        import src.prices as P
        P.reset_bloomberg_failure()
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (True, "key present"))
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (True, "Terminal reachable"))
        chain = P.provider_chain("auto", log=lambda *_: None)
        assert [p.name for p in chain] == ["bloomberg", "tiingo"]
        # once a fetch has failed, availability is False and auto skips it
        monkeypatch.undo()
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (True, "key present"))
        monkeypatch.setattr(
            "ingestion.pull_prices.bloomberg_request",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no refdata")))
        with pytest.raises(RuntimeError):
            P.BloombergProvider().fetch(["SPY"], "20260101", "20260201",
                                        log=lambda *_: None)
        chain = P.provider_chain("auto", log=lambda *_: None)
        assert [p.name for p in chain] == ["tiingo"]
        P.reset_bloomberg_failure()


class TestOneSymbolOneSource:
    def test_source_map_defaults_to_bloomberg_for_legacy_stores(self):
        import ingestion.pull_prices as pp
        legacy = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"] * 2),
                               "symbol": ["SPY", "GLD"],
                               "px_last": [1.0, 2.0]})
        assert pp._current_sources(legacy) == {"SPY": "bloomberg",
                                               "GLD": "bloomberg"}

    def test_source_map_reads_the_column(self):
        import ingestion.pull_prices as pp
        store = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"] * 2),
                              "symbol": ["SPY", "GLD"],
                              "px_last": [1.0, 2.0],
                              "source": ["bloomberg", "tiingo"]})
        assert pp._current_sources(store) == {"SPY": "bloomberg",
                                              "GLD": "tiingo"}

    def test_switched_symbol_is_planned_as_a_full_window(self):
        """plan_requests treats a symbol absent from ``existing`` as new,
        and main() removes switched symbols from ``existing`` first - so
        a provider change re-pulls the whole window for that symbol."""
        import ingestion.pull_prices as pp
        existing = pd.DataFrame({
            "date": pd.date_range("2026-01-02", periods=100, freq="B"),
            "symbol": ["SPY"] * 100, "px_last": 1.0, "source": "bloomberg"})
        # simulate main()'s removal of a switched symbol
        remaining = existing[~existing["symbol"].isin(["SPY"])]
        buckets = pp.plan_requests(["SPY"], "20260102", "20260601", remaining)
        assert list(buckets) == [("20260102", "20260601")]
        assert buckets[("20260102", "20260601")] == ["SPY"]

    def test_pull_script_and_shim_parse_and_expose_provider_flag(self):
        src = (ROOT / "ingestion" / "pull_prices.py").read_text(encoding="utf-8")
        assert '"--provider"' in src and '"--no-fallback"' in src
        shim = (ROOT / "ingestion" / "pull_bloomberg_prices.py").read_text(
            encoding="utf-8")
        assert "pull_prices.py" in shim and '"--provider", "bloomberg"' in shim

    def test_update_data_passes_the_provider_through(self):
        src = (ROOT / "update_data.py").read_text(encoding="utf-8")
        assert '"--provider"' in src
        assert '"ingestion/pull_prices.py", "--provider", _prov' in src


# ---------------------------------------------------------------------------
# end to end: fill from the fallback, then keep extending from it
# ---------------------------------------------------------------------------
class TestFillFromFallback:
    """A symbol the primary has no data for is filled in full from the
    fallback and stored under that source; on the next run its new days
    are fetched from the fallback, not asked of the primary again."""

    @staticmethod
    def _rows(symbols, start, end, source):
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
        out = []
        for s in symbols:
            out.append(pd.DataFrame({"date": idx, "symbol": s,
                                     "px_last": 1.0, "source": source}))
        return (pd.concat(out, ignore_index=True) if out
                else pd.DataFrame(columns=P.COLUMNS))

    def _run(self, monkeypatch, tmp_path, window, calls):
        import ingestion.pull_prices as pp
        rows = self._rows

        class BBG:
            name = "bloomberg"

            def fetch(self, symbols, start, end, log=print):
                calls.append(("bloomberg", tuple(symbols), start, end))
                return rows([s for s in symbols if s == "AAA"], start, end,
                            "bloomberg")

        class YF:
            name = "tiingo"

            def fetch(self, symbols, start, end, log=print):
                calls.append(("tiingo", tuple(symbols), start, end))
                return rows(symbols, start, end, "tiingo")
        monkeypatch.setattr(pp, "provider_chain", lambda *a, **k: [BBG(), YF()])
        monkeypatch.setattr(pp, "build_symbol_universe", lambda: ["AAA", "BBB"])
        monkeypatch.setattr(pp, "window_dates", lambda: window)
        monkeypatch.setattr(pp, "OUT_PATH", str(tmp_path / "prices.parquet"))
        monkeypatch.setattr(pp, "PRICES_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["ingestion/pull_prices.py", "--provider", "auto"])
        assert pp.main() == 0
        return pd.read_parquet(tmp_path / "prices.parquet")

    def test_no_data_symbol_is_filled_then_extended(self, monkeypatch, tmp_path):
        calls = []
        store = self._run(monkeypatch, tmp_path, ("20260105", "20260109"), calls)
        src = store.groupby("symbol")["source"].agg(lambda s: set(s))
        assert src["AAA"] == {"bloomberg"} and src["BBB"] == {"tiingo"}
        # the fill asked tiingo for BBB over the FULL window
        assert ("tiingo", ("BBB",), "20260105", "20260109") in calls

        calls.clear()
        store = self._run(monkeypatch, tmp_path, ("20260105", "20260116"), calls)
        # BBB's new days came from tiingo; bloomberg was never asked for BBB
        assert not any(c[0] == "bloomberg" and "BBB" in c[1] for c in calls)
        assert any(c[0] == "tiingo" and c[1] == ("BBB",) for c in calls)
        src = store.groupby("symbol")["source"].agg(lambda s: set(s))
        assert src["AAA"] == {"bloomberg"} and src["BBB"] == {"tiingo"}
        assert store[store.symbol == "BBB"]["date"].max() == pd.Timestamp("2026-01-16")
        assert store[store.symbol == "AAA"]["date"].max() == pd.Timestamp("2026-01-16")


class TestSelfTest:
    def test_selftest_reports_per_symbol_and_writes_nothing(self, monkeypatch):
        _stub_tiingo(monkeypatch, lambda s, a, b: _tiingo_rows(3))
        log = _Quiet()
        rc = P.selftest("tiingo", ["SPY", "GLD"], days=5, log=log)
        assert rc == 0
        text = "\n".join(log.lines)
        assert "SPY" in text and "GLD" in text and "nothing written" in text

    def test_selftest_fails_loudly_when_nothing_comes_back(self, monkeypatch):
        _stub_tiingo(monkeypatch, lambda s, a, b: [])
        log = _Quiet()
        assert P.selftest("tiingo", ["SPY"], days=5, log=log) == 1
        assert any("no rows" in l for l in log.lines)
