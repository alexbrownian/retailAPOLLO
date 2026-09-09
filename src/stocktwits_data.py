"""Normalisation of StockTwits messages into the standard posts schema.

StockTwits is the third social source. Two properties earn it a place:

1. It is the only mainstream finance-social platform whose users label
   their own posts bullish or bearish, which gives ground truth to
   calibrate the lexicon sentiment against (how often does the lexicon
   agree with the author's own label?).
2. The public JSON API needs no key for read-only symbol streams.

API surface used (rate limits and the 429 rule are documented at the top
of ingestion/fetch_stocktwits.py, this module's only caller)::

    https://api.stocktwits.com/api/2/streams/symbol/{SYM}.json
        -> ~30 most recent messages for one ticker, JSON

Message fields kept::

    id, body, created_at ('2024-01-05T14:31:22Z'),
    user.username, entities.sentiment.basic ('Bullish'/'Bearish'/None)

Normalisation to the standard 9-column schema::

    id           <- 'st_' + message id  (own prefix, no collisions)
    date         <- created_at day
    author       <- user.username
    score        <- 0 (likes exist but are sparse; not a counting signal)
    subreddit    <- 'stocktwits'  (its own pseudo-subreddit, like x_twitter)
    title        <- body (the message text)
    selftext     <- ''
    num_comments <- 0
    source       <- 'stocktwits'

The author's own Bullish/Bearish label does not fit the 9-column schema;
it is kept only in the raw .jsonl.zst files that fetch_stocktwits.py
writes, which the calibration tooling reads directly. Entry point:
``normalise_stocktwits()``.
"""

from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = ["id", "date", "author", "score", "subreddit",
                  "title", "selftext", "num_comments", "source"]


def normalise_stocktwits(messages: list[dict]) -> pd.DataFrame:
    """Normalises raw symbol-stream messages into the standard posts schema.

    Messages without an id, a body or a parseable created_at are dropped.

    Args:
        messages: Raw message dicts from the symbol-stream API.

    Returns:
        DataFrame with OUTPUT_COLUMNS, deduped on id (first seen wins),
        sorted by date.
    """
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


