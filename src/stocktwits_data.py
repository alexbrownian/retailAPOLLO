"""
stocktwits_data.py
==================
StockTwits as a THIRD social source (adopted from the fintwit-bot project,
reworked for our pipeline). Two reasons it earns a place:

1. It is the only mainstream finance-social platform whose users LABEL
   THEIR OWN POSTS bullish or bearish. That gives us ground truth to
   CALIBRATE our VADER+WSB sentiment against (how often does our lexicon
   agree with the author's own label?) - no other source offers that.
2. The public JSON API needs NO key for read-only symbol streams.

API surface used (see docs/LIVE_INGESTION.md for rate limits):
    https://api.stocktwits.com/api/2/streams/symbol/{SYM}.json
        -> ~30 most recent messages for one ticker, JSON
Message shape (the fields we keep):
    id, body, created_at ('2024-01-05T14:31:22Z'),
    user.username, entities.sentiment.basic ('Bullish'/'Bearish'/None)

Normalisation to the standard 9-column schema:
    id           <- 'st_' + message id  (own prefix, no collisions)
    date         <- created_at day
    author       <- user.username
    score        <- 0 (likes exist but are sparse; not a counting signal anyway)
    subreddit    <- 'stocktwits'  (its own pseudo-subreddit, like x_twitter)
    title        <- body (the message text)
    selftext     <- ''
    num_comments <- 0
    source       <- 'stocktwits'

The author's own Bullish/Bearish label does NOT fit the 9-column schema -
it is kept ONLY in the raw .jsonl.zst files that fetch_stocktwits.py
writes. The calibration notebook reads the raw files directly.
"""

from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = ["id", "date", "author", "score", "subreddit",
                  "title", "selftext", "num_comments", "source"]


def normalise_stocktwits(messages: list[dict]) -> pd.DataFrame:
    """messages: list of raw message dicts from the symbol-stream API.
    Returns rows in the standard schema, deduped on id (first seen wins)."""
    rows = []
    for m in messages:
        msg_id = m.get("id")
        body = (m.get("body") or "").strip()
        created = m.get("created_at") or ""
        if not msg_id or not body or len(created) < 10:
            continue
        user = m.get("user") or {}
        rows.append({
            "id": f"st_{msg_id}",
            "date": created[:10],
            "author": str(user.get("username") or ""),
            "score": 0,
            "subreddit": "stocktwits",
            "title": body,
            "selftext": "",
            "num_comments": 0,
            "source": "stocktwits",
        })
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.DataFrame(rows).drop_duplicates(subset="id", keep="first")
    return df.sort_values("date").reset_index(drop=True)[OUTPUT_COLUMNS]


def author_label(message: dict) -> str | None:
    """The author's OWN sentiment label ('Bullish'/'Bearish') or None.
    Only available in the raw messages - used by the calibration notebook."""
    entities = message.get("entities") or {}
    sentiment = entities.get("sentiment") or {}
    return sentiment.get("basic")
