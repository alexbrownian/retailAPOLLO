"""Vocabulary counting for emerging-term detection.

This module is the counting side of "what is retail suddenly talking about
that no theme covers yet?". It tokenises post text into candidate terms and
tallies how many posts mention each term per day.

Producers and consumers:

* ingestion/build_term_counts.py and the live fold build
  Data/abstracted/daily_term_counts.parquet from post text. The file holds
  (date, term, mention_count) plus one ``__TOTAL__`` row per day carrying
  the day's post count, so shares can be computed without the raw store.
* The dashboard's weekly context and tools/ai_keyword_audit.py read the
  counts file: the former ranks terms whose 7d count spiked against the
  prior four weeks, the latter surfaces high-frequency terms the theme
  keyword map does not cover. Both work in either mode.

The counts file is safe to commit: a table of daily word frequencies
contains no post text, no authors and no ids, so no post can be
reconstructed from it. It is the same abstraction class as the committed
theme and ticker counts.

Terms are single words (3+ chars) and consecutive two-word phrases.
Function words and finance boilerplate are dropped at build time (they
carry no signal and bloat the file); everyday-English filtering (wordfreq)
happens at scan time, so tightening that filter never requires a rebuild.
"""

from __future__ import annotations

import re

import pandas as pd

TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")   # Words of 3+ chars, letter-first.

TOTAL_MARKER = "__TOTAL__"    # Per-day row carrying the total post count.

# Finance boilerplate: present in every period, so never "emerging".
EXTRA_STOPWORDS = {
    "https", "http", "www", "com", "amp", "quot", "gt", "lt",
    "stock", "stocks", "market", "markets", "share", "shares", "price",
    "buy", "sell", "hold", "calls", "puts", "earnings", "portfolio",
    "trading", "trade", "invest", "investing", "investors", "investor",
    "money", "today", "tomorrow", "week", "year", "think", "thoughts",
}

# Function words. Single-word zipf filtering catches these at scan time,
# but word pairs like "from the" slip through unless each half is checked.
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
# Spam vocabulary. Scam and promo posts ("join my whatsapp group for
# signals") spike hard and would otherwise surface as emerging terms.
# Platform names and promo words are never a tradeable theme, so they are
# dropped outright.
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

MIN_PER_DAY_WORD = 3    # A word must appear in >= this many posts that day.
MIN_PER_DAY_PAIR = 5    # Pairs are noisier and more numerous: higher bar.
RETAIN_DAYS = 365       # Rolling window the counts file keeps. The spike
                        # test needs ~200 days; a year gives headroom.


def terms_in_text(text: str):
    """Returns one post's candidate terms.

    Args:
        text: The post's title and body.

    Returns:
        Set of unique filtered words and unique filtered two-word phrases
        (for example 'harmonic drive'), lowercased.
    """
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
    """Counts term mentions per day for one batch of posts.

    Each post counts each term at most once, so the figures are shares of
    posts, not raw frequencies. Per-day minimums (MIN_PER_DAY_*) keep the
    table small; they are applied per batch here, and the additive merge
    preserves correctness because live batches arrive day-aligned.

    Args:
        posts_df: Posts frame with date, title and selftext columns.

    Returns:
        DataFrame(date, term, mention_count) with one ``__TOTAL__`` row per
        day holding that day's total post count, mention or not.
    """
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
    """Keeps only the trailing retain_days of rows.

    The file must stay small enough to commit, and the spike test never
    looks further back.

    Args:
        df: Term counts frame with a date column.
        retain_days: Window length measured back from the newest date.

    Returns:
        The trimmed frame with a fresh RangeIndex.
    """
    dates = pd.to_datetime(df["date"])
    floor = dates.max() - pd.Timedelta(days=retain_days)
    return df[dates >= floor].reset_index(drop=True)
