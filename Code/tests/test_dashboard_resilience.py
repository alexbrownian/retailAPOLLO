"""The dashboard against stores it cannot read.

Streamlit gives a script ONE error boundary: an exception anywhere ends
the whole page, so a single damaged file is the difference between one
blank panel and a site that shows a stack trace. These tests damage a
COPY of the tree and render the page headless against it, then assert
two things about every case: the page comes up, and it says which file
is at fault.

The tests here never touch the real ``Data/``. ``src/config.py``
anchors every path on the repository root, so pointing the dashboard at
a different store means giving it a different tree: each test copies
``Code/`` and the data folders it needs into ``tmp_path``, damages the
copy, and renders that.

Each render runs in its own process. ``dashboard.py`` resolves its data
folder from the location of the module file and imports ``src.config``
off ``sys.path``; importing the copy in the test process would find the
real ``src.config`` already in ``sys.modules`` and read the real store
instead. A whole render costs three to eight seconds, which is why the
per-file sweep is parametrised rather than repeated by hand.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]          # <project>/Code
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

# The data folders a render reads. Data/raw carries no store the page
# opens but ensure_dirs() expects the shape, and the whole set copies in
# well under a second.
DATA_FOLDERS = ("abstracted", "dashboard", "prices", "processed",
                "reference", "research_record", "raw")

# The tab bar is a segmented control, and only the active tab's body
# runs, so a store read by one tab is only reached with that tab
# selected.
INFLUENCE_TAB = "Influence tracker"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _project_copy(tmp_path):
    """A whole working copy of the project, returned as its root.

    The copy carries a local settings override so the render behaves
    like a machine that runs the pipeline whatever the machine under
    the test is set to: the influence tab tells a missing store from an
    unpublished one by that flag, and a test should not read differently
    on two copies.
    """
    dst = tmp_path / "project"
    shutil.copytree(ROOT, dst / "Code",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in DATA_FOLDERS:
        src = PROJECT / "Data" / name
        if src.is_dir():
            shutil.copytree(src, dst / "Data" / name)
    if (PROJECT / ".streamlit").is_dir():
        shutil.copytree(PROJECT / ".streamlit", dst / ".streamlit")
    (dst / "Code" / "config" / "settings.local.csv").write_text(
        "key,value,description\n"
        "show_pipeline_controls,true,pipeline host\n", encoding="utf-8")
    return dst


_RENDER = r'''
import json, sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=600)
if sys.argv[2]:
    at.session_state["active_tab"] = sys.argv[2]
at.run()
sys.stdout.write("<<<PAGE " + json.dumps({
    "exception": [e.message for e in at.exception],
    "warning": [w.value for w in at.warning],
    "error": [w.value for w in at.error],
    "info": [w.value for w in at.info],
}))
'''


def _render(tree, tab=""):
    """Renders the copied dashboard headless and returns what it said.

    The answer is ``{"exception": [...], "warning": [...],
    "error": [...], "info": [...]}`` - the four channels a reader would
    see, plus the exceptions Streamlit caught.
    """
    r = subprocess.run(
        [sys.executable, "-c", _RENDER,
         str(tree / "Code" / "dashboard.py"), tab],
        cwd=str(tree), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    marker = "<<<PAGE "
    assert marker in r.stdout, (
        "the render process produced no result:\n"
        + r.stdout[-2000:] + "\n" + r.stderr[-3000:])
    return json.loads(r.stdout.split(marker, 1)[1])


def _copies(tree, name):
    """Every path called ``name`` under the copy's Data/ folder.

    A store can sit in more than one place: the six aggregates travel in
    Data/abstracted and are hydrated into Data/processed on every
    render, and the published bundle holds a second copy of the price
    and euphoria stores. Damaging one copy and leaving the other would
    only test the bootstrap.
    """
    return [Path(dirpath) / name
            for dirpath, _dirs, files in os.walk(tree / "Data")
            if name in files]


def _truncate(tree, name, size=100):
    """Cuts every copy of ``name`` to ``size`` bytes, as a killed write
    would leave it. Returns the paths damaged."""
    hit = _copies(tree, name)
    assert hit, f"{name} is not in the copied tree"
    for path in hit:
        with open(path, "r+b") as fh:
            fh.truncate(size)
    return hit


def _said(page, text):
    """Whether any message on the page contains ``text``."""
    return any(text in line
               for channel in ("warning", "error", "info")
               for line in page[channel])


# ---------------------------------------------------------------------------
# a store that is present and unreadable
# ---------------------------------------------------------------------------
# The file, and the tab whose body reads it.
DAMAGED_STORES = [
    ("daily_theme_counts.parquet", ""),
    ("prices.parquet", ""),
    ("euphoria_desk.parquet", ""),
    ("euphoria_levels.parquet", ""),
    ("author_scores.parquet", INFLUENCE_TAB),
    ("reply_edges.parquet", INFLUENCE_TAB),
]


class TestADamagedStoreLeavesThePageUp:
    """A committed store cut short by a killed write is present and
    unreadable, which is the case an ``os.path.exists`` guard cannot
    see. Every one of these files is read at module scope or in a tab
    body, so an unhandled parse error there takes the whole page with
    it - and takes it with a pyarrow message about a magic number,
    which tells a reader nothing about what to do."""

    @pytest.mark.parametrize("name,tab", DAMAGED_STORES)
    def test_a_truncated_store_renders_and_the_page_names_it(
            self, tmp_path, name, tab):
        tree = _project_copy(tmp_path)
        _truncate(tree, name)
        page = _render(tree, tab)
        assert not page["exception"], (
            f"a truncated {name} took the page down:\n"
            + "\n".join(page["exception"])[:3000])
        assert _said(page, name), (
            f"the page came up but never says {name} is the problem - a "
            "silent gap reads as 'nothing collected yet', which sends the "
            f"reader to re-run a pipeline that is not the fault: {page}")

    def test_an_aggregate_with_no_rows_stops_with_the_empty_store_sentence(
            self, tmp_path):
        """A valid parquet holding zero rows parses, so every read
        succeeds and the page goes on to derive its date window from an
        empty column. The masthead's freshness line is the first thing
        to reach for that date, and a max over nothing is NaT, which has
        no strftime. The page has to stop at the store instead, with the
        sentence an absent store gets."""
        tree = _project_copy(tmp_path)
        name = "daily_theme_counts.parquet"
        for path in _copies(tree, name):
            pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
        page = _render(tree)
        assert not page["exception"], (
            "an empty aggregate took the page down:\n"
            + "\n".join(page["exception"])[:3000])
        assert _said(page, "No aggregate data"), page


class TestADamagedRecordCostsItsFigureNotThePage:
    """The JSON verdict files are quoted by the plain-English panels. A
    hand-edited or half-written one must cost the panel its number."""

    def test_unparseable_report_files_leave_the_page_up(self, tmp_path):
        tree = _project_copy(tmp_path)
        for name in ("euphoria_report.json", "euphoria_desk_report.json"):
            paths = _copies(tree, name)
            assert paths, f"{name} is not in the copied tree"
            for path in paths:
                path.write_text('{"thresholds": ', encoding="utf-8")
        page = _render(tree)
        assert not page["exception"], (
            "an unparseable report file took the page down:\n"
            + "\n".join(page["exception"])[:3000])

    def test_an_unparseable_report_reads_as_no_report(self, tmp_path):
        """``euph_report`` and ``desk_report`` are ``_read_json`` of the
        two files, and every reader of both tests them against None, so
        the read has to answer None rather than raise."""
        import dashboard as D
        bad = tmp_path / "euphoria_report.json"
        bad.write_text('{"thresholds": ', encoding="utf-8")
        assert D._read_json(str(bad), D._mtime(str(bad))) is None

    def test_an_unparseable_research_record_reads_as_no_record(
            self, tmp_path, monkeypatch):
        """``_research`` hands every quoting panel a dict, so a damaged
        record leaves one figure as a dash instead of raising under the
        panel that asked for it."""
        import dashboard as D
        monkeypatch.setattr(D, "RESEARCH_DIR", str(tmp_path))
        (tmp_path / "nb01_episode_stats.json").write_text(
            "{not json", encoding="utf-8")
        assert D._research("nb01_episode_stats") == {}
        assert D._research("no_such_record") == {}


class TestAnAbsentStoreAndADamagedStoreReadDifferently:
    """The influence tab has three states - never built, built and
    damaged, built and readable - and the first two are told apart by an
    ``elif`` chain whose order decides which sentence a reader gets.
    Written the other way round, a damaged store is reported as one that
    was never built and the reader is sent to run a pull that will not
    fix it."""

    def test_a_store_that_was_never_built_says_so(self, tmp_path):
        tree = _project_copy(tmp_path)
        for path in _copies(tree, "author_scores.parquet"):
            path.unlink()
        page = _render(tree, INFLUENCE_TAB)
        assert not page["exception"], "\n".join(page["exception"])[:3000]
        assert _said(page, "no influence store on this machine yet"), page
        assert not _said(page, "does not parse"), (
            "an absent store is being reported as a damaged one", page)

    def test_a_store_that_does_not_parse_says_that_instead(self, tmp_path):
        tree = _project_copy(tmp_path)
        _truncate(tree, "author_scores.parquet")
        page = _render(tree, INFLUENCE_TAB)
        assert not page["exception"], "\n".join(page["exception"])[:3000]
        assert _said(page, "does not parse"), page
        assert not _said(page, "no influence store on this machine yet"), (
            "a damaged store is being reported as one that was never "
            "built, which sends the reader to a pull that rebuilds "
            "nothing", page)

    def test_the_map_tells_an_absent_edge_store_from_a_damaged_one(
            self, tmp_path):
        """The map runs the same three-state chain over the edge store,
        and the map is the only panel that reads it, so the rest of the
        tab has to stay up either way."""
        tree = _project_copy(tmp_path)
        for path in _copies(tree, "reply_edges.parquet"):
            path.unlink()
        page = _render(tree, INFLUENCE_TAB)
        assert not page["exception"], "\n".join(page["exception"])[:3000]
        assert _said(page, "no reply_edges.parquet in the store yet"), page
        assert not _said(page, "reply_edges.parquet does not parse"), page


class TestTheStoreReadersAnswerRatherThanRaise:
    """The helpers each panel reaches through, exercised directly. The
    renders above prove the page survives; these say which read is
    responsible, so a failure names the helper rather than a tab."""

    @staticmethod
    def _broken(tmp_path, name="store.parquet"):
        path = tmp_path / name
        shutil.copy2(PROJECT / "Data" / "abstracted"
                     / "daily_theme_counts.parquet", path)
        with open(path, "r+b") as fh:
            fh.truncate(100)
        return path

    def test_read_answers_none_for_a_file_that_does_not_parse(self, tmp_path):
        import dashboard as D
        path = self._broken(tmp_path)
        assert D._read(str(path), D._mtime(str(path))) is None

    def test_load_warns_and_answers_none(self, tmp_path):
        import dashboard as D
        self._broken(tmp_path, "daily_theme_counts.parquet")
        assert D.load("daily_theme_counts.parquet",
                      folder=str(tmp_path)) is None

    def test_priced_symbols_are_empty_when_the_price_store_is_damaged(
            self, tmp_path, monkeypatch):
        import dashboard as D
        path = self._broken(tmp_path, "prices.parquet")
        monkeypatch.setattr(D, "PRICES_PATH", str(path))
        assert D._cached_priced_symbols(D._mtime(str(path))) == frozenset()

    def test_the_momentum_gates_are_empty_when_the_price_store_is_damaged(
            self, tmp_path, monkeypatch):
        import dashboard as D
        path = self._broken(tmp_path, "prices.parquet")
        monkeypatch.setattr(D, "PRICES_PATH", str(path))
        assert D._momentum_gates(D._mtime(str(path))) == {}

    def test_the_level_outcome_frame_is_none_when_either_store_is_damaged(
            self, tmp_path, monkeypatch):
        import dashboard as D
        px = self._broken(tmp_path, "prices.parquet")
        lv = self._broken(tmp_path, "euphoria_levels.parquet")
        monkeypatch.setattr(D, "PRICES_PATH", str(px))
        monkeypatch.setattr(D, "PROCESSED_DIR", str(tmp_path))
        assert D._level_outcome_frame(D._mtime(str(lv)),
                                      D._mtime(str(px))) is None

    def test_the_reply_graph_reads_the_stores_it_is_handed(self, tmp_path,
                                                           monkeypatch):
        """``_reply_graph`` is reached only once the map's own guard has
        established that both stores parse, so it is handed frames, not
        paths. What it owes is a graph over the SCORED authors: the raw
        edge store holds every account, and drawing all of them gives a
        picture made of the people the tab has nothing to say about."""
        import dashboard as D
        board = tmp_path / "author_scores.parquet"
        pd.DataFrame({"author": ["a", "b", "c"],
                      "composite": [1.0, 2.0, None],
                      "n_judged": [9, 9, 0]}).to_parquet(board, index=False)
        edges = tmp_path / "reply_edges.parquet"
        pd.DataFrame({"replier": ["a", "c"],
                      "author": ["b", "a"]}).to_parquet(edges, index=False)
        monkeypatch.setattr(D, "_INFL_SCORES", str(board))
        monkeypatch.setattr(D, "_INFL_EDGES", str(edges))
        g = D._reply_graph(D._mtime(str(edges)), D._mtime(str(board)))
        assert sorted(g.names) == ["a", "b"]
        assert g.m == 1                      # the c->a edge is off the board
