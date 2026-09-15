"""Production hygiene: guards that keep the shipped tree fit to publish.

These tests do not exercise the model. They check the properties a
repository needs before it is handed to someone who was not part of
building it:

* the frozen ground-truth constants are pinned, so an edit to
  ``src/config.py`` cannot silently change what counts as an episode;
* every config CSV validates;
* the shipped code and documentation contain no project-internal
  vocabulary (people's names, organisation names, references to
  internal machines or to decisions made "by the desk");
* nothing runnable depends on the untracked ``Reference Materials/archive/`` folder;
* no shipped module writes a store in place: every write to ``Data/`` or
  ``Reports/`` is staged beside its target and swapped in;
* every text file is opened with an explicit encoding, so the shipped
  code reads the same on a machine whose console is not UTF-8.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]          # <project>/Code
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

# Folders that make up the shipped product, relative to the project root.
# Reference Materials/archive/ (long-form docs, research notebooks) is
# deliberately absent: it is not part of the product.
SHIPPED_DIRS = ("Code/src", "Code/ingestion", "Code/tools", "Code/tests",
                "Code/config", "Data/abstracted", "Data/dashboard",
                "Data/research_record")
SHIPPED_ROOT_FILES = ("Code/dashboard.py", "Code/update_data.py",
                      "Code/example.env", "README.md", ".gitignore",
                      "requirements.txt", "Reference Materials/RUNBOOK.md",
                      "Reference Materials/research.ipynb",
                      "Reference Materials/ice_source_review.ipynb")

# Words that must not appear in shipped code or docs. Matched
# case-insensitively as whole words; each entry names the class of
# reference it catches.
# The two settings that carry the credit and acknowledgement shown on the
# dashboard. They name people by design and are the only shipped lines
# allowed to.
CREDIT_KEYS = ("app_credit", "app_thanks")

BANNED = {
    r"\bdesk decision\b": "decision attributed to an internal group",
    r"\bthe desk\b": "reference to an internal group",
    r"\bdesk's\b": "reference to an internal group",
    r"\bincumbent\b": "internal history vocabulary",
    r"\binternal machine\b": "machine-specific wording",
    r"\bexternal machine\b": "machine-specific wording",
    r"\bAlex\b": "a person's name",
    r"\bShawn\b": "a person's name",
    r"\bWang Han\b": "a person's name",
    r"\bJonathan\b": "a person's name",
    r"\bGIC\b": "an organisation name",
    r"\bMAARS\b": "an organisation name",
    r"\bFIMA\b": "an organisation name",
    r"\bGIP\b": "an organisation name",
    r"\bintern\b": "role-specific wording",
    r"\ba PM\b": "role-specific wording",
    r"\bthe PM\b": "role-specific wording",
}


def _shipped_text_files():
    for d in SHIPPED_DIRS:
        base = PROJECT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix in (".py", ".md", ".txt", ".csv", ".env") and \
                    "__pycache__" not in p.parts and \
                    not p.name.endswith(".local.csv") and \
                    not (d.startswith("Data/") and p.suffix == ".txt"):
                # (.local.csv is git-ignored, per machine; the .txt files
                # under data/ are exchange symbol lists)
                yield p
    for name in SHIPPED_ROOT_FILES:
        p = PROJECT / name
        if p.exists():
            yield p


class TestFrozenConstantsArePinned:
    """The ground-truth definition is evidence; changing it is a research
    pass, never a config edit. These pins make an accidental edit fail
    loudly. Update them only together with a re-run of the research
    record and the values stored in ``Data/research_record``."""

    def test_episode_bars(self):
        from src import config as c
        assert c.EUPHORIA_BOOM_MIN_ETF == 0.20
        assert c.EUPHORIA_BOOM_MIN_SINGLE == 0.40
        assert c.EUPHORIA_CRASH_MIN_ETF == 0.12
        assert c.EUPHORIA_CRASH_MIN_SINGLE == 0.25

    def test_episode_windows(self):
        from src import config as c
        assert c.EUPHORIA_BOOM_LOOKBACK_D == 120
        assert c.EUPHORIA_CRASH_WINDOW_D == 90
        assert c.EUPHORIA_PEAK_LOCAL_MAX_D == 21
        assert c.EUPHORIA_PEAK_MERGE_D == 30

    def test_percentile_and_coverage_windows(self):
        from src import config as c
        assert c.EUPHORIA_PCT_WINDOW == 365
        assert c.EUPHORIA_MIN_HISTORY == 180
        assert c.EUPHORIA_MIN_COVERAGE == 100

    def test_find_episodes_uses_the_config_windows(self):
        """The episode extender must read its windows from config, not
        repeat them as literals that can drift from the peak finder."""
        src = (ROOT / "src" / "analytics" / "euphoria_phases.py").read_text(
            encoding="utf-8")
        body = src[src.index("def find_episodes("):
                   src.index("def episode_catalog(")]
        assert "EUPHORIA_BOOM_LOOKBACK_D" in body
        assert "EUPHORIA_CRASH_WINDOW_D" in body
        assert "days=120" not in body and "days=90" not in body


class TestConfigValidates:
    def test_validator_passes_on_the_committed_config(self):
        from tools.validate_config import validate
        errors = validate()
        assert not errors, "\n".join(errors)

    def test_validator_catches_an_unknown_theme(self, tmp_path, monkeypatch):
        import shutil
        from tools import validate_config as vc
        cfg = tmp_path / "config"
        shutil.copytree(ROOT / "config", cfg)
        with open(cfg / "theme_keywords.csv", "a", encoding="utf-8") as f:
            f.write("no_such_theme,word\n")
        monkeypatch.setattr(vc, "CONFIG_DIR", str(cfg))
        errors = vc.validate()
        assert any("no_such_theme" in e for e in errors)

    def test_settings_types(self):
        from src import settings
        assert isinstance(settings.get_bool("bot_screen_enabled"), bool)
        assert 0.0 <= settings.get_float("bot_screen_threshold") <= 1.0
        assert settings.get_int("single_name_top_n") > 0
        assert len(settings.load_forums()) >= 1


class TestNoInternalVocabulary:
    """Shipped text reads as product documentation, not as a diary."""

    @pytest.mark.parametrize("pattern,why", sorted(BANNED.items()))
    def test_banned_word_is_absent(self, pattern, why):
        rx = re.compile(pattern, re.IGNORECASE)
        hits = []
        for p in _shipped_text_files():
            if p.name == "test_production_hygiene.py":
                continue
            for i, line in enumerate(
                    p.read_text(encoding="utf-8", errors="replace")
                    .splitlines(), start=1):
                if p.name == "settings.csv" and \
                        line.split(",", 1)[0] in CREDIT_KEYS:
                    continue  # the credit and acknowledgement lines
                if rx.search(line):
                    hits.append(f"{p.relative_to(PROJECT)}:{i}: {line.strip()[:90]}")
                    if len(hits) >= 12:
                        break
            if len(hits) >= 12:
                break
        assert not hits, (f"{why} ({pattern}) found in shipped text:\n  "
                          + "\n  ".join(hits))

    def test_no_quoted_chat_requests_in_comments(self):
        """Comments that quote a chat message ('remove this', 'make it
        bigger') describe history, not behaviour, and were the largest
        source of internal vocabulary. None should remain."""
        rx = re.compile(r'#.*\b(on request|defect report|request \d{4}-\d{2})\b',
                        re.IGNORECASE)
        hits = []
        for p in _shipped_text_files():
            if p.suffix != ".py" or p.name == "test_production_hygiene.py":
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8",
                                                 errors="replace")
                                     .splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(PROJECT)}:{i}")
        assert not hits, "history comments remain:\n  " + "\n  ".join(hits[:20])


class TestRuntimeDoesNotNeedResearch:
    def test_no_shipped_module_imports_research(self):
        rx = re.compile(r"^\s*(from|import)\s+(research|notebooks|others)\b",
                        re.M)
        for p in _shipped_text_files():
            if p.suffix == ".py":
                assert not rx.search(p.read_text(encoding="utf-8",
                                                 errors="replace")), \
                    f"{p.relative_to(PROJECT)} imports from Reference Materials/archive/"

    def test_archive_is_gitignored(self):
        gi = (PROJECT / ".gitignore").read_text(encoding="utf-8")
        assert re.search(r"^Reference Materials/archive/\s*$", gi, re.M), \
            ".gitignore must ignore Reference Materials/archive/"

    def test_dashboard_reads_the_research_record_from_data(self):
        src = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        assert 'os.path.join(DATA_DIR, "research_record")' in src
        assert "docs/research" not in src

    def test_config_validator_is_a_shipped_tool(self):
        r = subprocess.run([sys.executable,
                            str(ROOT / "tools" / "validate_config.py"),
                            "--quiet"], capture_output=True, text=True,
                           encoding="utf-8", cwd=str(ROOT))
        assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------------------
# Source-level sweeps: the whole shipped tree, one AST walk each.
# ---------------------------------------------------------------------------
# The pipeline: the code that writes to Data/ and Reports/. The tests
# themselves are left out - a test writes into pytest's tmp_path, which
# is the whole point of the rule below and not an exception to it.
PIPELINE_DIRS = ("src", "ingestion", "tools")
PIPELINE_FILES = ("dashboard.py", "update_data.py")


def _pipeline_modules():
    for d in PIPELINE_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" not in p.parts:
                yield p
    for name in PIPELINE_FILES:
        if (ROOT / name).is_file():
            yield ROOT / name


def _shipped_modules():
    """Every .py under Code/, the tests included."""
    for p in sorted(ROOT.rglob("*.py")):
        if "__pycache__" not in p.parts:
            yield p


def _parents(tree):
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def _mode_of(call):
    """The mode string of an ``open()`` call, or None."""
    if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
        return call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


class TestNothingIsWrittenInPlace:
    """Data/ and Reports/ hold the only copy of histories no rebuild can
    recover, and a write straight onto a file truncates it at its first
    byte. Every write therefore lands beside its target and is swapped
    in with os.replace, so the name a reader opens is either the old
    whole file or the new whole file and never a fragment.

    The sweep reads the destination of every write in the shipped
    pipeline and asks one question: is it staged? A destination is
    staged when its expression ends in ``.tmp`` or is built from a name
    that says tmp. ``json.dump`` writes to a handle rather than a path,
    so the handle is resolved back to the ``with open(...)`` that made
    it and that path is read instead."""

    # Writes that are not staged, with the reason each is exempt. Keyed
    # by the module and the destination exactly as it is written, so a
    # NEW in-place write anywhere fails this test even in a file that
    # already has an entry.
    ALLOWED = {
        # --- operator-facing CLIs. The destination is a path the person
        # running the command typed, never a store under Data/: these
        # three exist to produce a file somewhere for inspection.
        ("src/clean_data.py", "output_path"):
            "--out, a path given on the command line",
        ("src/extract_tickers.py", "args.out"):
            "--out, a path given on the command line",
        ("src/extract_tickers.py", "args.daily_out"):
            "--daily-out, a path given on the command line",
        ("src/themes.py", "args.out"):
            "--out, a path given on the command line",
        # --- append-only journals. An append cannot truncate what is
        # already in the file, and both are read a line at a time, so an
        # interrupted write costs the last record rather than the
        # journal.
        ("src/agentic_watch.py", "SAMPLES"):
            "Data/reference/agentic_samples.jsonl, appended one line "
            "per sample",
        ("src/analytics/ai_poll.py", "ANSWERS"):
            "Data/reference/ai_poll_answers.jsonl, appended one line "
            "per answer",
        # --- run logs, under Reports/logs. Appended line by line as the
        # run progresses, which is what makes them readable while it is
        # still going; staging would hold the whole log until the end.
        ("update_data.py", "os.path.join(LOG_DIR, f'run_{today}.log')"):
            "the run log, appended as the run goes",
        ("dashboard.py", "p['log']"):
            "the same run log, appended by the sidebar's pipeline runner",
        # --- not under Data/ or Reports/ at all.
        ("ingestion/discover_subreddits.py", "FORUMS_FILE"):
            "Code/config/forums.csv, a config file appended to",
        ("tools/publish_dashboard.py", "SETTINGS_LOCAL"):
            "Code/config/settings.local.csv, written once when absent",
        # --- staged by a route the expression cannot show.
        ("src/pipeline_budget.py", "os.fdopen(fd, 'w', encoding='utf-8')"):
            "a descriptor from tempfile.mkstemp(suffix='.tmp'), swapped "
            "in with os.replace",
        ("ingestion/append_live_abstracted.py", "os.path.join(dest, fn)"):
            "a backup copied into a directory created for this run, so "
            "the destination cannot already exist",
    }

    @staticmethod
    def _staged(expr):
        """Whether a destination expression names a staging file."""
        if expr is None:
            return False
        text = ast.unparse(expr)
        return text.endswith('".tmp"') or text.endswith("'.tmp'") \
            or "tmp" in text.lower()

    @staticmethod
    def _handle_source(fn, name):
        """The ``with <expr> as name:`` a file handle came from."""
        for node in ast.walk(fn):
            if isinstance(node, ast.With):
                for item in node.items:
                    var = item.optional_vars
                    if isinstance(var, ast.Name) and var.id == name:
                        return item.context_expr
        return None

    @classmethod
    def _destinations(cls, path):
        """Every write in one module, as (lineno, destination expr)."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = _parents(tree)

        def enclosing(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node
            return tree

        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            dest = None
            if name in ("to_parquet", "to_csv") and node.args:
                dest = node.args[0]
            elif (name == "dump" and isinstance(fn, ast.Attribute)
                    and getattr(fn.value, "id", "") == "json"
                    and len(node.args) > 1):
                dest = node.args[1]
                if isinstance(dest, ast.Name):
                    dest = cls._handle_source(enclosing(node),
                                              dest.id) or dest
            elif name == "copy2" and len(node.args) > 1:
                dest = node.args[1]
            elif name == "open" and isinstance(fn, ast.Name) and node.args:
                mode = _mode_of(node)
                if mode and ("w" in mode or "a" in mode):
                    dest = node.args[0]
            if dest is not None:
                found.append((node.lineno, dest))
        return found

    def test_every_write_is_staged_or_explicitly_exempt(self):
        unstaged = {}
        for path in _pipeline_modules():
            rel = path.relative_to(ROOT).as_posix()
            for lineno, dest in self._destinations(path):
                if self._staged(dest):
                    continue
                key = (rel, ast.unparse(dest))
                if key not in self.ALLOWED:
                    unstaged[key] = f"{rel}:{lineno}"
        assert not unstaged, (
            "a write goes straight onto its destination - a run killed "
            "part-way leaves a fragment under the real name:\n  "
            + "\n  ".join(f"{v}: {k[1]}" for k, v in sorted(
                unstaged.items(), key=lambda kv: kv[1])))

    def test_the_exempt_list_still_describes_writes_that_exist(self):
        """An entry that no longer matches anything is a rule nobody is
        reading: it would go on excusing a line that has moved."""
        live = set()
        for path in _pipeline_modules():
            rel = path.relative_to(ROOT).as_posix()
            for _lineno, dest in self._destinations(path):
                live.add((rel, ast.unparse(dest)))
        stale = sorted(set(self.ALLOWED) - live)
        assert not stale, f"exemptions for writes that are gone: {stale}"

    def test_the_committed_aggregates_are_written_through_one_helper(self):
        """``_safe_write`` is the staging route, and every builder that
        touches Data/abstracted goes through it rather than repeating
        the stage-and-swap by hand."""
        for rel in ("ingestion/build_term_counts.py",
                    "ingestion/build_aggregates.py"):
            src = (ROOT / rel).read_text(encoding="utf-8")
            assert "_safe_write(" in src, f"{rel} writes its own way"


class TestTextIsReadTheSameOnEveryMachine:
    """Python opens a text file in the machine's locale encoding when it
    is not told otherwise, so the same file reads differently on a
    cp1252 console and a UTF-8 one - and a single character the locale
    has no mapping for aborts whatever was reading it. Every text read
    and write in the tree names its encoding, and so does every child
    process whose output is decoded."""

    def test_every_text_open_names_an_encoding(self):
        bare = []
        for path in _shipped_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "open" and node.args):
                    continue
                mode = _mode_of(node)
                if mode and "b" in mode:
                    continue                      # bytes carry no encoding
                if not any(kw.arg == "encoding" for kw in node.keywords):
                    bare.append(
                        f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
        assert not bare, ("these open a text file in whatever the console "
                          "is set to:\n  " + "\n  ".join(bare))

    def test_every_decoded_subprocess_names_an_encoding(self):
        """``text=True`` decodes the child's output with the locale
        encoding. git and the tools here report paths as UTF-8, so a
        non-ASCII filename in that output becomes a decode error on a
        machine whose console is not - and the caller fails over a name
        it only had to read."""
        bare = []
        for path in _shipped_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "run"):
                    continue
                kw = {k.arg: k.value for k in node.keywords}
                text = kw.get("text")
                if not (isinstance(text, ast.Constant) and text.value is True):
                    continue
                if "encoding" not in kw:
                    bare.append(
                        f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
        assert not bare, ("these decode a child process with the console's "
                          "encoding:\n  " + "\n  ".join(bare))
