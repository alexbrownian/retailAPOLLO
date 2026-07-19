"""
terms.py
========
Shared vocabulary machinery for EMERGING-TERM detection - the counting side
of "what is retail suddenly talking about that no theme covers yet?".

Two consumers:
  * ingestion/build_term_counts.py and the live fold build
    ABSTRACTED_DATA/daily_term_counts.parquet from post text
    (date, term, mention_count - plus one __TOTAL__ row per day holding the
    day's post count, so shares can be computed without the raw store).
  * helper/find_emerging_terms.py runs the spike test - from raw text on the
    external machine, or from the counts file on EITHER machine.

WHY THIS FILE IS SAFE TO SHARE: a table of daily word frequencies contains
no post text, no authors, no ids. No post can be reconstructed from it -
the same abstraction class as the theme/ticker counts already committed.

Terms are single words (3+ chars) and consecutive two-word phrases. Function
words and finance boilerplate are dropped at BUILD time (they carry no
signal and bloat the file); everyday-English filtering (wordfreq) happens at
SCAN time, so tightening that filter never requires a rebuild.
"""

from __future__ import annotations

import re

import pandas as pd

TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")   # words of 3+ chars, letter-first

TOTAL_MARKER = "__TOTAL__"    # per-day row carrying the total post count

# finance boilerplate - present in every period, so never "emerging"
EXTRA_STOPWORDS = {
    "https", "http", "www", "com", "amp", "quot", "gt", "lt",
    "stock", "stocks", "market", "markets", "share", "shares", "price",
    "buy", "sell", "hold", "calls", "puts", "earnings", "portfolio",
    "trading", "trade", "invest", "investing", "investors", "investor",
    "money", "today", "tomorrow", "week", "year", "think", "thoughts",
}

# function words - single-word zipf filtering catches these, but word PAIRS
# like "from the" slip through unless each half is checked
FUNCTION_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "all", "any",
    "can", "had", "has", "have", "him", "her", "his", "its", "our", "out",
    "she", "they", "them", "their", "this", "that", "these", "those",
    "was", "were", "will", "with", "would", "could", "should", "what",
    "when", "where", "which", "who", "why", "how", "than", "then", "there",
    "here", "from", "into", "onto", "over", "under", "about", "after",
    "before", "between", "through", "during", "against", "above", "below",
    "again", "once", "also", "just", "only", "even", "still", "while",
    "now", "right", "more", "most", "much", "many", "some", "same",
    "other", "another", "such", "very", "too", "been", "being", "does",
    "did", "doing", "because", "until", "both", "each", "own", "off",
    "doesn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
    "wouldn", "couldn", "shouldn", "won", "don", "didn", "ain", "lot",
}
# spam vocabulary - scam/promo posts ("join my whatsapp group for signals")
# spike hard and would otherwise become auto-themes. Platform names and
# promo words are never a tradeable theme, so they are dropped outright.
SPAM_WORDS = {
    "whatsapp", "telegram", "discord", "instagram", "tiktok", "youtube",
    "facebook", "snapchat", "linkedin", "twitter", "gmail", "email",
    "inbox", "website", "webinar", "zoom", "click", "link", "links",
    "subscribe", "follow", "followers", "join", "joined", "group",
    "groups", "channel", "channels", "community", "admin", "moderator",
    "giveaway", "promo", "promotion", "referral", "bonus",
    "signup", "register", "registration", "mentor", "mentorship", "guru",
    "coach", "coaching", "masterclass", "course", "courses", "ebook",
    "vip", "casino", "jackpot", "lottery", "winner",
    "congratulations", "guaranteed", "risk-free", "dm", "dms",
}

DROP_ALWAYS = EXTRA_STOPWORDS | FUNCTION_WORDS | SPAM_WORDS

MIN_PER_DAY_WORD = 3    # a word must appear in >= this many posts that day
MIN_PER_DAY_PAIR = 5    # pairs are noisier and more numerous - higher bar
RETAIN_DAYS = 365       # rolling window the counts file keeps (the spike
                        # test needs ~200 days; a year gives headroom)


def terms_in_text(text: str):
    """One post's candidate terms: unique filtered words + unique filtered
    two-word phrases ('harmonic drive'). Lowercases once."""
    words = TOKEN_RE.findall(text.lower())
    keep = set()
    for w in words:
        if w not in DROP_ALWAYS:
            keep.add(w)
    for w1, w2 in zip(words, words[1:]):
        if w1 not in DROP_ALWAYS and w2 not in DROP_ALWAYS:
            keep.add(w1 + " " + w2)
    return keep


def count_daily_terms(posts_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame(date, term, mention_count) for one batch of posts, with a
    __TOTAL__ row per day (total posts that day, mention or not). Each post
    counts each term AT MOST once - share of posts, not raw frequency.
    Per-day minimums (MIN_PER_DAY_*) keep the table small; they are applied
    per BATCH here, and the additive merge preserves correctness because
    live batches arrive day-aligned."""
    counts: dict = {}
    day_totals: dict = {}
    titles = posts_df["title"].fillna("").astype(str)
    bodies = posts_df["selftext"].fillna("").astype(str)
    dates = posts_df["date"].astype(str).str.slice(0, 10)
    for date, title, body in zip(dates, titles, bodies):
        day_totals[date] = day_totals.get(date, 0) + 1
        for term in terms_in_text(title + " " + body):
            key = (date, term)
            counts[key] = counts.get(key, 0) + 1

    rows = []
    for (date, term), n in counts.items():
        floor = MIN_PER_DAY_PAIR if " " in term else MIN_PER_DAY_WORD
        if n >= floor:
            rows.append((date, term, n))
    for date, n in day_totals.items():
        rows.append((date, TOTAL_MARKER, n))
    out = pd.DataFrame(rows, columns=["date", "term", "mention_count"])
    return out.sort_values(["date", "term"]).reset_index(drop=True)


def trim_to_retention(df: pd.DataFrame, retain_days: int = RETAIN_DAYS) -> pd.DataFrame:
    """Keep only the rolling window - the file must stay small enough to
    commit, and the spike test never looks further back anyway."""
    dates = pd.to_datetime(df["date"])
    floor = dates.max() - pd.Timedelta(days=retain_days)
    return df[dates >= floor].reset_index(drop=True)
