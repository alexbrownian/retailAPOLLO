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
* nothing runnable depends on the untracked ``Reference Materials/archive/`` folder.
"""

from __future__ import annotations

import os
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
                      "Reference Materials/research.ipynb")

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
                           cwd=str(ROOT))
        assert r.returncode == 0, r.stdout + r.stderr
