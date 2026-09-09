"""Price-provider layer: selection, fallback, reshaping, source tracking.

The Bloomberg path needs a Terminal and the Yahoo path needs network, so
these tests stub both at the boundary (``blpapi`` import and
``yfinance.download``) and check the logic around them: which provider
is chosen for each preference, that a failing provider falls through to
the next, that a ``yfinance.download`` frame in either of its two
layouts becomes the long ``date, symbol, px_last, source`` schema, and
that a symbol whose stored rows came from another provider is re-pulled
in full rather than spliced.
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
def _multi_ticker_frame(tickers, days=3):
    idx = pd.date_range("2026-08-03", periods=days, freq="B")
    cols = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], tickers])
    data = np.arange(len(idx) * len(cols), dtype=float).reshape(
        len(idx), len(cols)) + 100.0
    return pd.DataFrame(data, index=idx, columns=cols)


def _single_ticker_frame(days=3):
    idx = pd.date_range("2026-08-03", periods=days, freq="B")
    return pd.DataFrame({"Close": [10.0, 11.0, 12.0][:days],
                         "Open": [9.0, 10.0, 11.0][:days]}, index=idx)


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
        assert P.to_yfinance("NVDA") == "NVDA"

    def test_dotted_share_class_becomes_dashed_for_yahoo(self):
        assert P.to_yfinance("BRK.B") == "BRK-B"

    def test_foreign_line_uses_the_csv_mapping(self):
        # config/approved_instruments.csv carries 1622 JT -> 1622.T
        assert P.to_bloomberg("1622 JT") == "1622 JT Equity"
        assert P.to_yfinance("1622 JT") == "1622.T"

    def test_foreign_line_without_a_mapping_is_skipped_not_guessed(self,
                                                                    monkeypatch):
        monkeypatch.setattr(P, "_approved_rows", lambda: {
            "XYZ LN": {"symbol": "XYZ LN", "bloomberg": "XYZ LN Equity",
                       "yfinance": ""}})
        assert P.to_yfinance("XYZ LN") is None


# ---------------------------------------------------------------------------
# yfinance reshaping
# ---------------------------------------------------------------------------
class TestYFinanceReshape:
    def test_multi_ticker_layout_becomes_long(self):
        raw = _multi_ticker_frame(["SPY", "GLD"])
        out = P.YFinanceProvider._to_long(raw, ["SPY", "GLD"],
                                          {"SPY": "SPY", "GLD": "GLD"})
        assert list(out.columns) == P.COLUMNS
        assert set(out["symbol"]) == {"SPY", "GLD"}
        assert len(out) == 6
        assert (out["source"] == "yfinance").all()
        assert out["date"].dt.tz is None

    def test_single_ticker_layout_becomes_long(self):
        raw = _single_ticker_frame()
        out = P.YFinanceProvider._to_long(raw, ["1622.T"], {"1622.T": "1622 JT"})
        assert list(out["symbol"].unique()) == ["1622 JT"]
        assert out["px_last"].tolist() == [10.0, 11.0, 12.0]

    def test_empty_download_gives_empty_frame_with_schema(self):
        out = P.YFinanceProvider._to_long(pd.DataFrame(), ["SPY"], {"SPY": "SPY"})
        assert list(out.columns) == P.COLUMNS and out.empty

    def test_fetch_calls_download_with_inclusive_end(self, monkeypatch):
        calls = {}

        def fake_download(tickers, start, end, **kw):
            calls["tickers"], calls["start"], calls["end"] = tickers, start, end
            return _multi_ticker_frame(list(tickers))
        fake_yf = types.SimpleNamespace(download=fake_download)
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
        log = _Quiet()
        out = P.YFinanceProvider().fetch(["SPY", "GLD", "1622 JT"],
                                         "20260803", "20260805", log=log)
        assert calls["start"] == "2026-08-03"
        assert calls["end"] == "2026-08-06"          # exclusive end, +1 day
        assert "1622.T" in calls["tickers"]
        assert set(out["symbol"]) == {"SPY", "GLD", "1622 JT"}


# ---------------------------------------------------------------------------
# provider selection and fallback
# ---------------------------------------------------------------------------
class TestProviderChain:
    def test_yfinance_preference_is_yfinance_only(self, monkeypatch):
        monkeypatch.setattr(P.YFinanceProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("yfinance", log=_Quiet())
        assert [c.name for c in chain] == ["yfinance"]

    def test_bloomberg_preference_falls_back_to_yfinance(self, monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (True, "ok"))
        monkeypatch.setattr(P.YFinanceProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("bloomberg", log=_Quiet())
        assert [c.name for c in chain] == ["bloomberg", "yfinance"]
        chain = P.provider_chain("bloomberg", fallback=False, log=_Quiet())
        assert [c.name for c in chain] == ["bloomberg"]

    def test_auto_skips_bloomberg_when_no_terminal(self, monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (False, "no terminal"))
        monkeypatch.setattr(P.YFinanceProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("auto", log=_Quiet())
        assert [c.name for c in chain] == ["yfinance"]

    def test_auto_prefers_bloomberg_when_reachable(self, monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (True, "ok"))
        monkeypatch.setattr(P.YFinanceProvider, "available",
                            lambda self: (True, "ok"))
        chain = P.provider_chain("auto", log=_Quiet())
        assert [c.name for c in chain] == ["bloomberg", "yfinance"]

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
            name = "yfinance"

            def fetch(self, symbols, start, end, log=print):
                return pd.DataFrame({"date": pd.to_datetime(["2026-08-03"]),
                                     "symbol": ["SPY"], "px_last": [1.0],
                                     "source": ["yfinance"]})
        log = _Quiet()
        df, used = P.fetch_with_fallback([Boom(), Good()], ["SPY"],
                                         "20260803", "20260803", log=log)
        assert used == "yfinance" and len(df) == 1
        assert any("falling back to yfinance" in l for l in log.lines)

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
class TestOneSymbolOneSource:
    def test_source_map_defaults_to_bloomberg_for_legacy_stores(self):
        import pull_prices as pp
        legacy = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"] * 2),
                               "symbol": ["SPY", "GLD"],
                               "px_last": [1.0, 2.0]})
        assert pp._current_sources(legacy) == {"SPY": "bloomberg",
                                               "GLD": "bloomberg"}

    def test_source_map_reads_the_column(self):
        import pull_prices as pp
        store = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"] * 2),
                              "symbol": ["SPY", "GLD"],
                              "px_last": [1.0, 2.0],
                              "source": ["bloomberg", "yfinance"]})
        assert pp._current_sources(store) == {"SPY": "bloomberg",
                                              "GLD": "yfinance"}

    def test_switched_symbol_is_planned_as_a_full_window(self):
        """plan_requests treats a symbol absent from ``existing`` as new,
        and main() removes switched symbols from ``existing`` first - so
        a provider change re-pulls the whole window for that symbol."""
        import pull_prices as pp
        existing = pd.DataFrame({
            "date": pd.date_range("2026-01-02", periods=100, freq="B"),
            "symbol": ["SPY"] * 100, "px_last": 1.0, "source": "bloomberg"})
        # simulate main()'s removal of a switched symbol
        remaining = existing[~existing["symbol"].isin(["SPY"])]
        buckets = pp.plan_requests(["SPY"], "20260102", "20260601", remaining)
        assert list(buckets) == [("20260102", "20260601")]
        assert buckets[("20260102", "20260601")] == ["SPY"]

    def test_pull_script_and_shim_parse_and_expose_provider_flag(self):
        src = (ROOT / "pull_prices.py").read_text(encoding="utf-8")
        assert '"--provider"' in src and '"--no-fallback"' in src
        shim = (ROOT / "pull_bloomberg_prices.py").read_text(encoding="utf-8")
        assert "pull_prices.py" in shim and '"--provider", "bloomberg"' in shim

    def test_update_data_passes_the_provider_through(self):
        src = (ROOT / "update_data.py").read_text(encoding="utf-8")
        assert '"--provider"' in src
        assert '"pull_prices.py", "--provider", _prov' in src
