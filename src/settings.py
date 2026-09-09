"""Editable runtime settings, loaded from ``config/*.csv``.

Everything a user is expected to change without editing Python lives in
``config/`` as CSV. This module is the single reader for the three files
that are not theme definitions (those are read by :mod:`src.themes`):

``config/settings.csv``
    ``key,value,description`` rows. Typed accessors below convert the
    string ``value`` to bool/int/float and fall back to a documented
    default when a key is absent, so an older ``settings.csv`` never
    breaks a newer build.

``config/settings.local.csv`` (optional, not committed)
    Same schema; its rows override ``settings.csv``. Use it for values
    that differ per copy, such as ``show_pipeline_controls``, so the
    committed file stays safe for a hosted deployment.

``config/forums.csv``
    The forums (subreddits) the ingestion layer crawls. One row per forum
    with an ``enabled`` flag, so a forum can be paused without deleting
    its history from the panel.

``config/single_name_overrides.csv``
    Optional force-include / force-exclude rows for the single-name
    universe, which is otherwise chosen from the data (see
    :func:`analytics.euphoria.single_name_universe`).

All readers accept a UTF-8 BOM so files saved from Excel parse cleanly.
Results are cached per process; call :func:`reload` after editing a file
in a long-running process.

Typical use::

    from src import settings
    if settings.get_bool("bot_screen_enabled"):
        threshold = settings.get_float("bot_screen_threshold")
    forums = settings.load_forums()          # enabled reddit forums only
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(_ROOT, "config")

SETTINGS_FILE = os.path.join(CONFIG_DIR, "settings.csv")
# Same schema as settings.csv, applied on top of it, never committed
# (see .gitignore). Holds per-copy values such as show_pipeline_controls.
SETTINGS_LOCAL_FILE = os.path.join(CONFIG_DIR, "settings.local.csv")
FORUMS_FILE = os.path.join(CONFIG_DIR, "forums.csv")
SINGLE_NAME_OVERRIDES_FILE = os.path.join(CONFIG_DIR,
                                          "single_name_overrides.csv")

# Defaults used when a key is missing from settings.csv. Keep this table
# in sync with config/README.md.
DEFAULTS: Dict[str, str] = {
    "app_title": "RetailRadar",
    "app_tagline": ("A real-time read on where retail attention is "
                    "building, and where it is ending."),
    "app_credit": "",
    "show_pipeline_controls": "false",
    "price_provider": "auto",
    "single_name_top_n": "25",
    "single_name_window_days": "365",
    "bot_screen_enabled": "true",
    "bot_screen_threshold": "0.6",
    "bot_screen_duplicate_jaccard": "0.85",
    "bot_screen_burst_posts_per_day": "12",
}

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off", ""}


class ConfigError(ValueError):
    """A config CSV is missing, malformed, or holds an invalid value."""


def _read_csv(path: str, required: tuple) -> List[dict]:
    """Read a config CSV into a list of stripped-string dicts.

    Args:
        path: Absolute path of the CSV file.
        required: Column names that must be present in the header.

    Returns:
        One dict per data row. Rows whose first cell starts with ``#``
        are treated as comments and skipped.

    Raises:
        ConfigError: If the file is absent or a required column is
            missing.
    """
    if not os.path.exists(path):
        raise ConfigError(f"{path} is missing; see config/README.md")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in required if c not in fields]
        if missing:
            raise ConfigError(f"{path}: missing column(s) {missing}; "
                              f"expected at least {list(required)}")
        rows = []
        for row in reader:
            first = (row.get(fields[0]) or "").strip()
            if first.startswith("#"):
                continue
            rows.append({k: (v or "").strip() for k, v in row.items()
                         if k is not None})
        return rows


@lru_cache(maxsize=None)
def _settings() -> Dict[str, str]:
    """All ``settings.csv`` values as strings, defaults filled in."""
    values = dict(DEFAULTS)
    for path in (SETTINGS_FILE, SETTINGS_LOCAL_FILE):
        if os.path.exists(path):
            for row in _read_csv(path, ("key", "value")):
                if row["key"]:
                    values[row["key"]] = row["value"]
    return values


def get(key: str) -> str:
    """Return a setting as a string.

    Raises:
        KeyError: If ``key`` is neither in ``settings.csv`` nor in
            :data:`DEFAULTS`.
    """
    values = _settings()
    if key not in values:
        raise KeyError(f"unknown setting {key!r}; add it to "
                       f"config/settings.csv or src/settings.DEFAULTS")
    return values[key]


def get_bool(key: str) -> bool:
    """Return a setting as a boolean (accepts true/false, yes/no, 1/0)."""
    raw = get(key).lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"settings.csv: {key}={raw!r} is not a boolean")


def get_int(key: str) -> int:
    """Return a setting as an integer."""
    try:
        return int(float(get(key)))
    except ValueError as exc:
        raise ConfigError(f"settings.csv: {key}={get(key)!r} is not an "
                          "integer") from exc


def get_float(key: str) -> float:
    """Return a setting as a float."""
    try:
        return float(get(key))
    except ValueError as exc:
        raise ConfigError(f"settings.csv: {key}={get(key)!r} is not a "
                          "number") from exc


def show_pipeline_controls() -> bool:
    """Whether the dashboard exposes the pipeline-run buttons.

    True when ``show_pipeline_controls`` is set in ``settings.csv`` or the
    environment variable ``RETAILAPOLLO_CONTROLS`` is ``1``. The buttons
    fetch data and call paid APIs, so the default is off.
    """
    if os.environ.get("RETAILAPOLLO_CONTROLS") == "1":
        return True
    return get_bool("show_pipeline_controls")


@lru_cache(maxsize=None)
def load_forums(source: str = "reddit",
                enabled_only: bool = True) -> tuple:
    """Return the forums to crawl, in file order.

    Args:
        source: Filter to one data source (``reddit``); ``None`` returns
            every source.
        enabled_only: Skip rows whose ``enabled`` column is false.

    Returns:
        A tuple of forum names (for example subreddit names without the
        ``r/`` prefix).
    """
    rows = _read_csv(FORUMS_FILE, ("forum", "source", "enabled"))
    out = []
    for row in rows:
        if not row["forum"]:
            continue
        if source and row["source"].lower() != source.lower():
            continue
        if enabled_only and row["enabled"].lower() not in _TRUE:
            continue
        out.append(row["forum"])
    if not out:
        raise ConfigError(f"{FORUMS_FILE}: no enabled forums for "
                          f"source={source!r}")
    return tuple(out)


def load_forum_rows() -> List[dict]:
    """Every row of ``forums.csv`` as dicts (all sources, all states)."""
    return _read_csv(FORUMS_FILE, ("forum", "source", "enabled"))


@lru_cache(maxsize=None)
def single_name_overrides() -> Dict[str, str]:
    """Force-include / force-exclude rules for the single-name universe.

    Returns:
        ``{symbol: action}`` where action is ``include`` or ``exclude``.
        Empty when the file is absent or holds no rows.
    """
    if not os.path.exists(SINGLE_NAME_OVERRIDES_FILE):
        return {}
    out = {}
    for row in _read_csv(SINGLE_NAME_OVERRIDES_FILE, ("symbol", "action")):
        sym, action = row["symbol"].upper(), row["action"].lower()
        if not sym:
            continue
        if action not in ("include", "exclude"):
            raise ConfigError(f"{SINGLE_NAME_OVERRIDES_FILE}: action for "
                              f"{sym} must be include or exclude, "
                              f"not {action!r}")
        out[sym] = action
    return out


def reload() -> None:
    """Drop every cached read so the next call re-reads the CSV files."""
    _settings.cache_clear()
    load_forums.cache_clear()
    single_name_overrides.cache_clear()
