"""Price-provider layer: selection, fallback, reshaping, source tracking.

The Bloomberg path needs a Terminal and the Tiingo path needs a key and
network, so these tests stub both at the boundary (``blpapi`` import and
the Tiingo HTTP call) and check the logic around them: which provider is
chosen for each preference, that a failing provider falls through to
the next, that Tiingo's JSON becomes the long ``date, symbol, px_last,
source`` schema with a split-adjusted close, and that a symbol whose
stored rows came from another provider is re-pulled in full rather than
spliced.

The pull script's modes are driven the same way: ``main()`` runs against
a stub provider and a store in ``tmp_path`` (``OUT_PATH`` and
``PRICES_DIR`` are module-level, so the run never sees ``Data/``), which
pins what ``--daily``, ``--force`` and ``--check`` each do to the rows
already on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


class _Stub:
    """A provider that answers every symbol it is asked for, over the
    whole span, at one price - so a fetched row is told apart from a
    stored one by its ``px_last`` as well as by its ``source``.

    Symbols named in ``silent`` come back with no rows, as they do from a
    vendor that does not carry the line or cannot address it."""

    def __init__(self, name="tiingo", px=2.0, silent=()):
        self.name = name
        self.px = px
        self.silent = set(silent)
        self.calls = []

    def fetch(self, symbols, start, end, log=print):
        self.calls.append((tuple(symbols), start, end))
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
        parts = [pd.DataFrame({"date": idx, "symbol": s, "px_last": self.px,
                               "source": self.name}) for s in symbols
                 if s not in self.silent]
        return (pd.concat(parts, ignore_index=True) if parts and len(idx)
                else pd.DataFrame(columns=P.COLUMNS))


def _store_without_sources(path, symbols, start, end, px=1.0):
    """Write a store whose rows carry no ``source`` column, as one built
    before the column existed does, and return what was written."""
    idx = pd.date_range(start, end, freq="B")
    df = pd.concat([pd.DataFrame({"date": idx, "symbol": s, "px_last": px})
                    for s in symbols], ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def _store_from(path, symbols, start, end, px=1.0, source="tiingo"):
    """Write a store whose rows all carry ``source``, as one built by a
    pull on that provider does, and return what was written."""
    idx = pd.date_range(start, end, freq="B")
    df = pd.concat([pd.DataFrame({"date": idx, "symbol": s, "px_last": px,
                                  "source": source}) for s in symbols],
                   ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def _run_pull(monkeypatch, tmp_path, argv, window, provider,
              anchors=None, universe=None):
    """Drive ``pull_prices.main()`` against ``provider`` and a store in
    ``tmp_path``; returns its exit code."""
    import ingestion.pull_prices as pp
    monkeypatch.setattr(pp, "provider_chain", lambda *a, **k: [provider])
    monkeypatch.setattr(pp, "theme_anchor_symbols",
                        lambda: list(anchors if anchors is not None else []))
    monkeypatch.setattr(pp, "build_symbol_universe",
                        lambda: list(universe if universe is not None else []))
    monkeypatch.setattr(pp, "window_dates", lambda: window)
    monkeypatch.setattr(pp, "OUT_PATH", str(tmp_path / "prices.parquet"))
    monkeypatch.setattr(pp, "PRICES_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["ingestion/pull_prices.py"] + argv)
    return pp.main()


# ---------------------------------------------------------------------------
# store schema
# ---------------------------------------------------------------------------
class TestStoreSchema:
    """``with_source`` is what every writer puts a row through, so the
    store carries the column the readers key on rather than leaving each
    of them to infer it."""

    @staticmethod
    def _frame(sources=None):
        df = pd.DataFrame({"symbol": ["AAA", "BBB"],
                           "date": pd.to_datetime(["2026-01-05"] * 2),
                           "px_last": [1.0, 2.0]})
        if sources is not None:
            df["source"] = sources
        return df

    def test_a_frame_with_no_source_column_reads_as_bloomberg(self):
        out = P.with_source(self._frame())
        assert list(out.columns) == P.COLUMNS
        assert (out["source"] == "bloomberg").all()

    def test_stored_sources_are_kept(self):
        out = P.with_source(self._frame(["tiingo", "bloomberg"]))
        assert out["source"].tolist() == ["tiingo", "bloomberg"]
        assert list(out.columns) == P.COLUMNS

    def test_a_missing_source_fills_to_bloomberg(self):
        """A frame that carries the column can still hold rows without a
        label - a part-filled store, or a splice of one frame that has
        the column with one that does not."""
        out = P.with_source(self._frame(["tiingo", None]))
        assert out["source"].tolist() == ["tiingo", "bloomberg"]

    def test_the_columns_come_back_in_store_order(self):
        df = self._frame(["tiingo", "tiingo"])[
            ["source", "px_last", "symbol", "date"]]
        assert list(P.with_source(df).columns) == P.COLUMNS


# ---------------------------------------------------------------------------
# symbol mapping
# ---------------------------------------------------------------------------
class TestSymbolMapping:
    #: US ETFs the approved config spells with a venue code rather than
    #: the ``US`` composite: NYSE Arca lines, ordinary US listings.
    US_VENUE_ETFS = ("MOO", "XRT", "KRE", "VTV", "VUG", "IVE", "IVW")
    #: Lines listed outside the US, which carry their own market's token.
    FOREIGN_LINES = ("1622 JT", "159915 CS", "588000 CH")

    @staticmethod
    def _approved_config():
        """The real ``config/approved_instruments.csv`` as rows of str."""
        return (pd.read_csv(ROOT / "config" / "approved_instruments.csv",
                            dtype=str).fillna("")
                .to_dict("records"))

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

    # -- which venues read as US ------------------------------------------
    def test_every_us_exchange_token_reads_as_a_us_listing(self):
        """A US line reaches the market through one of a dozen venue
        codes - the composite, the two Nasdaq tapes, Arca, American -
        and the ticker is the same ticker on each of them."""
        for exch in ("US", "UN", "UQ", "UW", "UP", "UA"):
            assert P.is_us_listing(f"XRT {exch} Equity"), exch

    def test_a_foreign_exchange_token_reads_as_a_foreign_listing(self):
        for exch in ("JT", "CS", "CH", "LN", "GR", "HK"):
            assert not P.is_us_listing(f"1234 {exch} Equity"), exch

    def test_a_code_that_is_not_an_equity_line_is_not_a_us_listing(self):
        """The test reads a three-part ``<TICKER> <EXCH> Equity`` code.
        An index line and a bare ticker are neither, so neither is a US
        equity listing."""
        assert not P.is_us_listing("CSIN0852 Index")
        assert not P.is_us_listing("UX Index")
        assert not P.is_us_listing("XRT")
        assert not P.is_us_listing("")

    # -- what the approved config maps to --------------------------------
    def test_a_us_listing_on_a_venue_code_maps_to_its_plain_ticker(self):
        """These seven are ordinary US ETFs written with their venue's
        code, and a provider addressed by plain ticker takes the ticker."""
        for sym in self.US_VENUE_ETFS:
            assert P.to_tiingo(sym) == sym

    def test_a_foreign_line_with_no_mapping_has_no_tiingo_symbol(self):
        for sym in self.FOREIGN_LINES:
            assert P.to_tiingo(sym) is None

    def test_every_us_listing_in_the_approved_config_is_mappable(self):
        """The approved config is the tradeable universe, and a US line
        in it that no provider can be asked for is a line that is never
        requested - and so a series that is never filled."""
        rows = self._approved_config()
        us = [r for r in rows if P.is_us_listing(r["bloomberg"])]
        assert {r["symbol"] for r in us} >= set(self.US_VENUE_ETFS), (
            "a US venue code in the approved config reads as foreign")
        unmappable = sorted(r["symbol"] for r in us
                            if not P.to_tiingo(r["symbol"]))
        assert unmappable == [], (
            f"{len(unmappable)} US-listed approved instrument(s) have no "
            f"tiingo symbol: {', '.join(unmappable)}")

    def test_a_foreign_line_in_the_approved_config_maps_or_is_skipped(self):
        """A line listed outside the US is asked for only through an
        explicit ``tiingo`` column; without one it is skipped rather than
        guessed at, and stays a Bloomberg-only series."""
        rows = self._approved_config()
        foreign = [r for r in rows if not P.is_us_listing(r["bloomberg"])]
        assert foreign, "the approved config holds no non-US line at all"
        for row in foreign:
            mapped = P.to_tiingo(row["symbol"])
            assert mapped == (row["tiingo"] or None), row["symbol"]


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

    def test_an_answered_span_clears_the_run_of_failures(self, monkeypatch):
        """The allowance covers a run of CONSECUTIVE failures, not a
        total spread over a whole pull. A long pull asks for one span
        after another over an hour or more, and a Terminal that is
        working still drops the odd request; counted as a total, two
        blips an hour apart would send a healthy Terminal to the
        fallback for the rest of the run - and splice the sources of
        every symbol it then filled."""
        import src.prices as P
        P.reset_bloomberg_failure()
        monkeypatch.setattr(P, "bloomberg_max_failures", lambda: 2)
        answered = {"20260201"}

        def _request(symbols, start, end, log=print):
            if start not in answered:
                raise RuntimeError("could not open //blp/refdata service")
            return pd.DataFrame({"date": pd.to_datetime(["2026-02-02"]),
                                 "symbol": ["SPY"], "px_last": [1.0]})
        monkeypatch.setattr("ingestion.pull_prices.bloomberg_request",
                            _request)
        bbg = P.BloombergProvider()
        with pytest.raises(RuntimeError):                  # span 1: fails
            bbg.fetch(["SPY"], "20260101", "20260131", log=lambda *_: None)
        assert P.bloomberg_failed_this_process() == ""

        got = bbg.fetch(["SPY"], "20260201", "20260228",   # span 2: answers
                        log=lambda *_: None)
        assert len(got) == 1 and got["source"].iloc[0] == "bloomberg"

        with pytest.raises(RuntimeError):                  # span 3: fails
            bbg.fetch(["SPY"], "20260301", "20260331", log=lambda *_: None)
        assert P.bloomberg_failed_this_process() == "", (
            "two failures with an answered span between them were counted "
            "as a run of two - the answer in the middle clears the tally")
        assert bbg.down_reason() == ""
        assert "not retried" not in bbg.available()[1]

        with pytest.raises(RuntimeError):        # now two in a row
            bbg.fetch(["SPY"], "20260401", "20260430", log=lambda *_: None)
        assert "2 consecutive" in P.bloomberg_failed_this_process()
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
# the daily universe
# ---------------------------------------------------------------------------
class TestThemeAnchorUniverse:
    """``--daily`` prices the theme anchors, which is a small corner of
    the priced universe: one ETF per theme, with the fallback anchors,
    the benchmarks, the overlays and the single names left to a full
    pull. Every symbol in it costs one request on a per-symbol API, so
    what the set holds is what the light run costs."""

    def test_the_anchors_are_a_small_fraction_of_the_priced_universe(self):
        import ingestion.pull_prices as pp
        anchors = pp.theme_anchor_symbols()
        universe = pp.build_symbol_universe()
        assert anchors, "no theme anchors - a daily run would price nothing"
        assert len(anchors) * 3 < len(universe), (
            f"{len(anchors)} anchors against {len(universe)} priced "
            "symbols - the daily universe is not the light set")
        assert set(anchors) <= set(universe), (
            "an anchor that a full pull never asks for would only ever "
            "be priced on a daily run")

    def test_every_anchor_is_a_theme_etf(self):
        import ingestion.pull_prices as pp
        from src.themes import THEME_ETFS
        values = {str(v).strip().upper() for v in THEME_ETFS.values()}
        assert set(pp.theme_anchor_symbols()) <= values

    def test_every_anchor_is_priceable_by_at_least_one_provider(self):
        """An anchor is the line a whole theme is drawn against, so an
        anchor no provider can be asked for is a theme that goes
        unpriced, and the only sign of it is an empty line on the
        dashboard. Each anchor therefore either maps to a Tiingo symbol
        or is a non-US line with a Bloomberg security behind it."""
        import ingestion.pull_prices as pp
        from src.themes import APPROVED_INSTRUMENTS
        unreachable, bloomberg_only = [], []
        for sym in pp.theme_anchor_symbols():
            if P.to_tiingo(sym):
                continue
            if (APPROVED_INSTRUMENTS.get(sym) or {}).get("bloomberg"):
                bloomberg_only.append(sym)
            else:
                unreachable.append(sym)
        assert unreachable == [], (
            f"{len(unreachable)} theme anchor(s) no provider can be asked "
            f"for: {', '.join(unreachable)} - add a tiingo symbol or a "
            "bloomberg security to config/approved_instruments.csv")
        assert bloomberg_only == ["1622 JT"], (
            "1622 JT is the one anchor priced on the Terminal alone (a "
            f"Tokyo line with no Tiingo symbol); {bloomberg_only} means "
            "a theme goes unpriced on every run that is not Bloomberg's")

    def test_the_anchors_are_upper_cased_sorted_and_free_of_placeholders(self):
        """A theme with no approved anchor stores ``"?"`` as its
        instrument, and a blank is a row that was never filled in.
        Neither is a symbol any provider can be asked for."""
        import ingestion.pull_prices as pp
        anchors = pp.theme_anchor_symbols()
        assert "?" not in anchors and "" not in anchors
        assert all(s == s.strip().upper() for s in anchors)
        assert anchors == sorted(anchors)
        assert len(anchors) == len(set(anchors))


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


# ---------------------------------------------------------------------------
# end to end: what each mode does to the rows already on disk
# ---------------------------------------------------------------------------
class TestDailyModeExtendsTheStore:
    """A store whose rows predate provider tracking reads as bloomberg,
    so a run on another provider finds every symbol switching. In daily
    mode the stored rows stay and the run's provider fills the trailing
    gap: nine years of closes are worth more than one vendor per series,
    and the seam is written into the ``source`` column of the rows the
    run adds rather than smoothed over."""

    WINDOW = ("20260105", "20260116")
    OLD_END = pd.Timestamp("2026-01-09")

    def test_a_daily_run_extends_a_stored_series(self, monkeypatch, tmp_path):
        path = tmp_path / "prices.parquet"
        before = _store_without_sources(path, ["AAA", "BBB"],
                                        "2026-01-05", "2026-01-09")
        rc = _run_pull(monkeypatch, tmp_path,
                       ["--daily", "--provider", "tiingo"], self.WINDOW,
                       _Stub("tiingo"), anchors=["AAA", "BBB"])
        assert rc == 0
        after = pd.read_parquet(path)

        kept = set(zip(before["symbol"], before["date"]))
        assert kept <= set(zip(after["symbol"], after["date"])), (
            "a daily run dropped stored closes it was asked to extend")
        old = after[after["date"] <= self.OLD_END]
        assert len(old) == len(before)
        assert set(old["source"]) == {"bloomberg"}
        assert (old["px_last"] == 1.0).all(), "a stored close was overwritten"

        new = after[after["date"] > self.OLD_END]
        assert len(new) and set(new["source"]) == {"tiingo"}
        one = after[after["symbol"] == "AAA"]
        assert one["date"].min() == pd.Timestamp("2026-01-05")
        assert one["date"].max() == pd.Timestamp("2026-01-16")

    def test_the_run_says_how_many_series_carry_two_vendors(
            self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA", "BBB"], "2026-01-05",
                               "2026-01-09")
        assert _run_pull(monkeypatch, tmp_path,
                         ["--daily", "--provider", "tiingo"], self.WINDOW,
                         _Stub("tiingo"), anchors=["AAA", "BBB"]) == 0
        out = capsys.readouterr().out
        assert "2 series carry two vendors" in out, out[-800:]
        assert "AAA" in out and "BBB" in out


class TestAFullRunKeepsOneSymbolOnOneSource:
    """Outside daily mode the one-symbol-one-source rule stands: a
    symbol whose stored rows came from a provider outside this run's
    chain is re-pulled over the whole window and its old rows go, so no
    series holds two vendors' closes."""

    def test_a_provider_change_re_pulls_the_stored_symbols_in_full(
            self, monkeypatch, tmp_path):
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA", "BBB"], "2026-01-05",
                               "2026-01-09", px=1.0)
        rc = _run_pull(monkeypatch, tmp_path, ["--provider", "tiingo"],
                       ("20260105", "20260116"), _Stub("tiingo", px=2.0),
                       universe=["AAA", "BBB"])
        assert rc == 0
        after = pd.read_parquet(path)
        assert not (after["px_last"] == 1.0).any(), (
            "stored rows survived a provider change outside daily mode")
        assert set(after["source"]) == {"tiingo"}
        assert (after.groupby("symbol")["source"].nunique() == 1).all()
        assert after["date"].min() == pd.Timestamp("2026-01-05"), (
            "the re-pull covers the whole window, not just the gap")


class TestARePullReplacesOnlyWhatComesBack:
    """A re-pull hands a symbol's stored rows over to what the run got
    back for it, which is bounded twice: by the symbols the provider
    answered for, and by the window that was asked for. So a symbol the
    provider has nothing for keeps every close it had, and a series that
    reaches back beyond the window start keeps the half in front of it -
    the half no request in this run covers.
    """

    WINDOW = ("20260105", "20260116")
    WINDOW_START = pd.Timestamp("2026-01-05")

    def test_a_symbol_the_provider_cannot_answer_for_keeps_its_series(
            self, monkeypatch, tmp_path):
        path = tmp_path / "prices.parquet"
        before = _store_without_sources(path, ["AAA", "BBB", "CCC"],
                                        "2026-01-05", "2026-01-09", px=1.0)
        rc = _run_pull(monkeypatch, tmp_path, ["--provider", "tiingo"],
                       self.WINDOW, _Stub("tiingo", px=2.0, silent=("CCC",)),
                       universe=["AAA", "BBB", "CCC"])
        assert rc == 0
        after = pd.read_parquet(path)
        assert "CCC" in set(after["symbol"]), (
            "a symbol the run's provider returned nothing for was dropped "
            "from the store, and its history exists nowhere else")
        kept = after[after["symbol"] == "CCC"]
        was = before[before["symbol"] == "CCC"]
        assert kept["date"].tolist() == was["date"].tolist()
        assert kept["px_last"].tolist() == was["px_last"].tolist()
        assert set(kept["source"]) == {"bloomberg"}, (
            "the stored rows are the ones that were there, under the "
            "provider that wrote them")
        answered = after[after["symbol"].isin(["AAA", "BBB"])]
        assert set(answered["source"]) == {"tiingo"}
        assert (answered["px_last"] == 2.0).all()

    def test_the_run_names_the_symbols_that_kept_their_stored_rows(
            self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA", "BBB", "CCC"], "2026-01-05",
                               "2026-01-09", px=1.0)
        assert _run_pull(monkeypatch, tmp_path, ["--provider", "tiingo"],
                         self.WINDOW,
                         _Stub("tiingo", px=2.0, silent=("CCC",)),
                         universe=["AAA", "BBB", "CCC"]) == 0
        out = capsys.readouterr().out
        assert "keep every stored row" in out, out[-800:]
        assert "CCC" in out

    def test_a_re_pull_keeps_the_closes_dated_before_the_window(
            self, monkeypatch, tmp_path):
        path = tmp_path / "prices.parquet"
        before = _store_without_sources(path, ["AAA"], "2025-12-01",
                                        "2026-01-09", px=1.0)
        rc = _run_pull(monkeypatch, tmp_path, ["--provider", "tiingo"],
                       self.WINDOW, _Stub("tiingo", px=2.0),
                       universe=["AAA"])
        assert rc == 0
        after = pd.read_parquet(path)
        old = after[after["date"] < self.WINDOW_START]
        was = before[before["date"] < self.WINDOW_START]
        assert old["date"].tolist() == was["date"].tolist(), (
            "the re-pull replaced the whole series while refetching the "
            "window alone, so the closes in front of it are gone")
        assert old["px_last"].tolist() == was["px_last"].tolist()
        assert set(old["source"]) == {"bloomberg"}
        assert after["date"].min() == before["date"].min()

        new = after[after["date"] >= self.WINDOW_START]
        assert set(new["source"]) == {"tiingo"}
        assert (new["px_last"] == 2.0).all()
        assert new["date"].max() == pd.Timestamp("2026-01-16")

    def test_a_series_that_predates_the_window_carries_two_vendors(
            self, monkeypatch, tmp_path):
        """The window start is where the two halves meet, and the seam is
        written into the ``source`` column so the health check can name
        it rather than a step in the line reading as a price move."""
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA"], "2025-12-01", "2026-01-09",
                               px=1.0)
        assert _run_pull(monkeypatch, tmp_path, ["--provider", "tiingo"],
                         self.WINDOW, _Stub("tiingo", px=2.0),
                         universe=["AAA"]) == 0
        after = pd.read_parquet(path)
        assert after.groupby("symbol")["source"].nunique().tolist() == [2]
        import ingestion.pull_prices as pp
        assert pp._two_vendor_symbols(after) == ["AAA"]


class TestForceKeepsWhatTheRunDidNotAskFor:
    """``--force`` re-downloads the window for the symbols THIS run
    asks for. The store holds every symbol a full pull ever priced, so
    a narrowed universe - a daily run is a couple of dozen symbols -
    must leave the rest of the store exactly where it is."""

    def test_symbols_outside_the_run_keep_every_stored_row(
            self, monkeypatch, tmp_path):
        path = tmp_path / "prices.parquet"
        before = _store_without_sources(path, ["AAA", "BBB", "CCC"],
                                        "2026-01-05", "2026-01-09", px=1.0)
        rc = _run_pull(monkeypatch, tmp_path,
                       ["--daily", "--force", "--provider", "tiingo"],
                       ("20260105", "20260116"), _Stub("tiingo", px=2.0),
                       anchors=["AAA"])
        assert rc == 0
        after = pd.read_parquet(path)
        for sym in ("BBB", "CCC"):
            kept = after[after["symbol"] == sym]
            was = before[before["symbol"] == sym]
            assert kept["date"].tolist() == was["date"].tolist(), (
                f"--force wrote {sym} out of the store")
            assert (kept["px_last"] == 1.0).all()
            assert set(kept["source"]) == {"bloomberg"}

    def test_the_run_universe_is_re_pulled_in_full(self, monkeypatch,
                                                   tmp_path):
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA", "BBB", "CCC"], "2026-01-05",
                               "2026-01-09", px=1.0)
        assert _run_pull(monkeypatch, tmp_path,
                         ["--daily", "--force", "--provider", "tiingo"],
                         ("20260105", "20260116"), _Stub("tiingo", px=2.0),
                         anchors=["AAA"]) == 0
        one = pd.read_parquet(path)
        one = one[one["symbol"] == "AAA"]
        assert set(one["source"]) == {"tiingo"}
        assert (one["px_last"] == 2.0).all()
        assert one["date"].min() == pd.Timestamp("2026-01-05")
        assert one["date"].max() == pd.Timestamp("2026-01-16")

    def test_a_run_symbol_keeps_the_closes_dated_before_the_window(
            self, monkeypatch, tmp_path):
        """What ``--force`` re-downloads is the window, so the window is
        the whole of what it gives up: a symbol inside the run's universe
        whose series starts before the window start keeps that older
        half, which no request in this run covers."""
        path = tmp_path / "prices.parquet"
        before = _store_without_sources(path, ["AAA", "BBB"], "2025-12-01",
                                        "2026-01-09", px=1.0)
        assert _run_pull(monkeypatch, tmp_path,
                         ["--daily", "--force", "--provider", "tiingo"],
                         ("20260105", "20260116"), _Stub("tiingo", px=2.0),
                         anchors=["AAA"]) == 0
        after = pd.read_parquet(path)
        one = after[after["symbol"] == "AAA"]
        old = one[one["date"] < pd.Timestamp("2026-01-05")]
        was = before[(before["symbol"] == "AAA")
                     & (before["date"] < pd.Timestamp("2026-01-05"))]
        assert old["date"].tolist() == was["date"].tolist(), (
            "--force dropped closes dated before the window it "
            "re-downloads, and nothing it asked for replaces them")
        assert old["px_last"].tolist() == was["px_last"].tolist()
        assert set(old["source"]) == {"bloomberg"}

        new = one[one["date"] >= pd.Timestamp("2026-01-05")]
        assert set(new["source"]) == {"tiingo"}
        assert (new["px_last"] == 2.0).all()
        assert new["date"].max() == pd.Timestamp("2026-01-16")


class TestASameProviderRunOnlyAppends:
    """A run against a store its own provider wrote has nothing to
    replace. Every stored close stands as it is and the run asks for the
    days after the newest one alone - which is what makes a nightly pull
    cost a handful of requests and leave nine years of closes alone."""

    def test_a_stored_close_is_untouched_by_a_run_on_its_own_provider(
            self, monkeypatch, tmp_path):
        path = tmp_path / "prices.parquet"
        before = _store_from(path, ["AAA", "BBB"], "2026-01-05",
                             "2026-01-09", px=1.0, source="tiingo")
        stub = _Stub("tiingo", px=2.0)
        assert _run_pull(monkeypatch, tmp_path, ["--provider", "tiingo"],
                         ("20260105", "20260116"), stub,
                         universe=["AAA", "BBB"]) == 0
        after = pd.read_parquet(path)
        merged = before.merge(after, on=["symbol", "date"], how="left",
                              suffixes=("_was", "_is"))
        assert merged["px_last_is"].notna().all(), "a stored close went missing"
        assert (merged["px_last_is"] == merged["px_last_was"]).all(), (
            "a stored close was overwritten by a run that was only asked "
            "to extend the series")
        assert (merged["source_is"] == merged["source_was"]).all()
        assert len(after) > len(before), "the run appended nothing"
        assert set(after["symbol"]) == {"AAA", "BBB"}
        assert after["date"].max() == pd.Timestamp("2026-01-16")

    def test_only_the_days_after_the_newest_close_are_requested(
            self, monkeypatch, tmp_path):
        path = tmp_path / "prices.parquet"
        _store_from(path, ["AAA", "BBB"], "2026-01-05", "2026-01-09", px=1.0,
                    source="tiingo")
        stub = _Stub("tiingo", px=2.0)
        assert _run_pull(monkeypatch, tmp_path, ["--provider", "tiingo"],
                         ("20260105", "20260116"), stub,
                         universe=["AAA", "BBB"]) == 0
        assert stub.calls == [(("AAA", "BBB"), "20260110", "20260116")], (
            "the run re-requested days the store already holds")


class TestCheckMeasuresWithoutWriting:
    """``--check`` is run to find out whether a pull is worth starting,
    which means it has to be safe to run at any time and to report
    rather than fail when nothing answers."""

    @staticmethod
    def _no_provider(monkeypatch):
        monkeypatch.setattr(P.BloombergProvider, "available",
                            lambda self: (False, "no Terminal on this machine"))
        monkeypatch.setattr(P.TiingoProvider, "available",
                            lambda self: (False, "TIINGO_API_KEY is not set"))

    def test_the_store_is_byte_for_byte_unchanged(self, monkeypatch,
                                                  tmp_path):
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA", "BBB"], "2026-01-05",
                               "2026-01-09")
        self._no_provider(monkeypatch)
        was, mtime = path.read_bytes(), path.stat().st_mtime_ns
        assert _run_pull(monkeypatch, tmp_path,
                         ["--check", "--provider", "auto"],
                         ("20260105", "20260116"), _Stub("tiingo"),
                         anchors=["AAA"], universe=["AAA", "BBB"]) == 0
        assert path.read_bytes() == was
        assert path.stat().st_mtime_ns == mtime
        assert sorted(p.name for p in tmp_path.iterdir()) == \
            ["prices.parquet"], "a check run left a file behind"

    def test_no_usable_provider_is_a_report_not_a_failure(self, monkeypatch,
                                                          tmp_path, capsys):
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA", "BBB"], "2026-01-05",
                               "2026-01-09")
        self._no_provider(monkeypatch)
        rc = _run_pull(monkeypatch, tmp_path,
                       ["--check", "--provider", "auto"],
                       ("20260105", "20260116"), _Stub("tiingo"),
                       anchors=["AAA"], universe=["AAA", "BBB"])
        assert rc == 0, "a machine with no provider must still get a report"
        out = capsys.readouterr().out
        assert "no Terminal on this machine" in out, out
        assert "TIINGO_API_KEY is not set" in out, out
        assert "no usable provider" in out, out

    def test_the_report_names_the_store_and_both_plans(self, monkeypatch,
                                                       tmp_path, capsys):
        path = tmp_path / "prices.parquet"
        _store_without_sources(path, ["AAA", "BBB"], "2026-01-05",
                               "2026-01-09")
        self._no_provider(monkeypatch)
        assert _run_pull(monkeypatch, tmp_path,
                         ["--check", "--provider", "auto"],
                         ("20260105", "20260116"), _Stub("tiingo"),
                         anchors=["AAA"], universe=["AAA", "BBB"]) == 0
        out = capsys.readouterr().out
        assert "rows        : 10" in out, out
        assert "symbols     : 2" in out, out
        assert "two-vendor  : 0 series" in out, out
        assert "daily :    1 symbols" in out, out
        assert "full  :    2 symbols" in out, out


# ---------------------------------------------------------------------------
# the health check reads the same seam
# ---------------------------------------------------------------------------
class TestTwoVendorSeriesAreReported:
    """Two vendors' closes differ after a corporate action, and such a
    step reads exactly like a price move downstream. The health check
    names the series that hold one, which is what keeps the seam
    readable from outside the pull that made it."""

    @staticmethod
    def _health(monkeypatch, tmp_path, frame):
        import tools.data_health as H
        path = tmp_path / "prices.parquet"
        frame.to_parquet(path, index=False)
        monkeypatch.setattr(H, "PRICES", str(path))
        out, lines = {}, []
        H._mixed_sources(out, lambda msg, status=None: lines.append(msg))
        return out, "\n".join(lines)

    @staticmethod
    def _rows(symbol, source, start, end):
        idx = pd.date_range(start, end, freq="B")
        return pd.DataFrame({"date": idx, "symbol": symbol, "px_last": 1.0,
                             "source": source})

    def test_a_symbol_with_two_sources_is_named(self, monkeypatch, tmp_path):
        frame = pd.concat([
            self._rows("AAA", "bloomberg", "2026-01-05", "2026-01-09"),
            self._rows("AAA", "tiingo", "2026-01-12", "2026-01-16"),
            self._rows("BBB", "bloomberg", "2026-01-05", "2026-01-16")],
            ignore_index=True)
        out, text = self._health(monkeypatch, tmp_path, frame)
        assert out["mixed_source_symbols"] == ["AAA"]
        assert "1 of 2 symbols carry more than one source" in text
        assert "AAA" in text

    def test_a_single_vendor_store_reports_none(self, monkeypatch, tmp_path):
        frame = pd.concat([
            self._rows("AAA", "bloomberg", "2026-01-05", "2026-01-16"),
            self._rows("BBB", "tiingo", "2026-01-05", "2026-01-16")],
            ignore_index=True)
        out, text = self._health(monkeypatch, tmp_path, frame)
        assert out["mixed_source_symbols"] == []
        assert "0 of 2 symbols carry more than one source" in text


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
