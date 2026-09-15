"""Normalisation of raw live-Reddit dumps into the standard posts schema.

Turns the raw records written by ingestion/fetch_reddit_live.py into the
project's standard 9-column posts shape, so live posts merge into
posts.parquet exactly like the historical Pushshift data.

Each raw record is one post's JSON as the backend returned it, tagged with
a "_backend" field that says which shape to expect:

``"_backend" == "official"``
    A standard Reddit API `data` object (id, created_utc, author,
    subreddit, title, selftext, num_comments, score), identical to the
    Pushshift shape, so src.clean_data.normalise() is reused.

``"_backend" == "fetchlayer"``
    FetchLayer's community-posts shape. Field names are not guaranteed
    stable, so every field is looked up defensively across the names
    FetchLayer has been seen to use.

Output columns (the one schema shared with clean_data / x_data /
stocktwits_data)::

    id, date, author, score, subreddit, title, selftext, num_comments, source

Live Reddit posts keep source='reddit', their real subreddit (for example
'wallstreetbets') and their real base36 id, so a live post that later
shows up in a Pushshift dump dedupes against it automatically (first seen
wins). Entry point: ``normalise_reddit_live_records()``.
"""

from __future__ import annotations

import datetime

import pandas as pd

from src.clean_data import normalise as normalise_official  # Official API == Pushshift shape.

OUTPUT_COLUMNS = ["id", "date", "author", "score", "subreddit",
                  "title", "selftext", "num_comments", "source"]


def _first(record: dict, *names, default=""):
    """Returns the first non-empty value among the named keys."""
    for name in names:
        v = record.get(name)
        if v not in (None, ""):
            return v
    return default


def _date_of(record: dict) -> str:
    """Returns a best-effort 'YYYY-MM-DD' from whatever timestamp FetchLayer
    supplies: a unix epoch (int/float/str) or an ISO/RFC date string."""
    raw = _first(record, "created_utc", "createdUtc", "created", "createdAt",
                 "created_at", "date", default="")
    if raw in (None, ""):
        return ""
    # Unix seconds?
    try:
        secs = float(raw)
        if secs > 1_000_000_000:                 # Sane epoch (>= 2001).
            return datetime.datetime.fromtimestamp(
                secs, datetime.timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        pass
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _author_of(record: dict) -> str:
    """FetchLayer's 'author' is sometimes a plain handle, sometimes an object."""
    a = _first(record, "author", "authorName", "user", default="")
    if isinstance(a, dict):
        return str(a.get("handle") or a.get("username") or a.get("name") or "")
    return str(a or "")


def _normalise_fetchlayer(record: dict) -> dict:
    """Maps one FetchLayer record onto the standard schema."""
    title = str(_first(record, "title", "postTitle") or "")
    selftext = str(_first(record, "selftext", "previewText", "text", "body", "content") or "")
    return {
        "id": str(_first(record, "id", "postId", "name") or ""),
        "date": _date_of(record),
        "author": _author_of(record),
        "score": int(_first(record, "score", "upvotes", "ups", default=0) or 0),
        "subreddit": str(_first(record, "subreddit", "community") or "").lower(),
        "title": title,
        "selftext": selftext,
        "num_comments": int(_first(record, "num_comments", "numComments",
                                   "commentCount", "comments", default=0) or 0),
        "source": "reddit",
    }


def normalise_reddit_live_records(records: list[dict]) -> pd.DataFrame:
    """Normalises raw live-Reddit records into the standard posts schema.

    Records without an id, a date or a title cannot be placed in the
    timeline and are dropped.

    Args:
        records: Raw post dicts, any mix of backends.

    Returns:
        DataFrame with OUTPUT_COLUMNS, deduped on id (first seen wins),
        sorted by date.
    """
    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        backend = rec.get("_backend", "official")
        row = _normalise_fetchlayer(rec) if backend == "fetchlayer" else normalise_official(rec)
        # A post with no id or no date can't be placed in the timeline.
        if row["id"] and row["date"] and str(row["title"]).strip():
            rows.append({c: row[c] for c in OUTPUT_COLUMNS})
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="first")
    return df.sort_values("date").reset_index(drop=True)[OUTPUT_COLUMNS]
