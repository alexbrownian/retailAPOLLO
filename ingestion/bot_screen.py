"""Screen posts for automated or duplicated content before aggregation.

Every post gets a ``bot_score`` in ``[0, 1]`` and a ``bot_reasons``
string. Posts at or above ``bot_screen_threshold`` (``config/settings.csv``)
are excluded from the sentiment and attention aggregates; they stay in
the raw store, so the decision is reversible and auditable.

The screen is deterministic and needs no model. Five signals, each with
a weight; the score is the capped sum, so one strong signal or several
weak ones reach the threshold:

``near_duplicate``  (0.55)
    The post's text is a near-copy of another post by a *different*
    author, or of one of the author's own earlier posts. Copy-pasted
    promotion is the most common bot signature. Detected with MinHash
    locality-sensitive hashing (the ``datasketch`` package) over word
    3-shingles, Jaccard similarity at or above
    ``bot_screen_duplicate_jaccard``. Very short texts (under 8 tokens)
    are never treated as duplicates - "to the moon" is not a bot.

``burst``  (0.45)
    The author posted at least ``bot_screen_burst_posts_per_day`` times
    in one calendar day within the batch.

``disclosed``  (1.00)
    The text carries a bot self-description ("I am a bot", "this action
    was performed automatically", "beep boop") or the author is a known
    automated account (``AutoModerator``, names ending in ``bot``).

``low_diversity``  (0.35)
    An author with at least 5 posts whose texts are near-identical to
    each other (Jaccard of the author's shingle sets against their own
    first post), which catches templated posting that does not repeat
    exactly.

``deleted_author``  (0.10)
    The author field is ``[deleted]`` or empty. Weak on its own - many
    real posts lose their author - so it only tips a borderline case.

Typical use::

    from ingestion.bot_screen import screen_posts, apply_screen
    scored = screen_posts(posts)          # adds bot_score, bot_reasons
    kept, report = apply_screen(posts)    # kept = rows below threshold

The report dictionary (rows in, rows excluded, rate, top reasons) is
printed by the callers and stored in ``data/reference/bot_screen_last.json``
so the exclusion rate is visible after every run.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd

WEIGHTS = {
    "near_duplicate": 0.55,
    "burst": 0.45,
    "disclosed": 1.00,
    "low_diversity": 0.35,
    "deleted_author": 0.10,
}

MIN_TOKENS_FOR_DUP = 8        # shorter texts are never "duplicates"
SHINGLE = 3                   # word n-gram size for MinHash
NUM_PERM = 64                 # MinHash permutations (speed vs accuracy)
DIVERSITY_MIN_POSTS = 5       # posts before low_diversity can fire
DIVERSITY_JACCARD = 0.60      # own-post similarity that counts as templated

_DISCLOSED = re.compile(
    r"\b(i am a bot|i'm a bot|this (action|comment) was performed "
    r"automatically|beep boop|bot account|automated (post|message))\b",
    re.IGNORECASE)
_BOT_NAME = re.compile(r"(^automoderator$|bot$|_bot$|-bot$|bot\d+$)",
                       re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9$]+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _shingles(tokens: List[str]) -> set:
    if len(tokens) < SHINGLE:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + SHINGLE])
            for i in range(len(tokens) - SHINGLE + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _texts(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").astype(str) if "title" in df else ""
    body = (df["selftext"].fillna("").astype(str).str.slice(0, 2000)
            if "selftext" in df else "")
    return (title + " " + body).str.strip()


# ---------------------------------------------------------------------------
# the five signals
# ---------------------------------------------------------------------------
def _near_duplicates(texts: List[str], authors: List[str],
                     jaccard: float) -> List[bool]:
    """Flag texts that near-match another post (MinHash LSH).

    Two posts are duplicates when their estimated Jaccard similarity on
    word 3-shingles is at or above ``jaccard``. Both members of a pair
    are flagged unless they share an author and the pair is the only
    match - one person cross-posting once is not automation - but an
    author whose text matches two or more others is flagged regardless.
    """
    from datasketch import MinHash, MinHashLSH

    n = len(texts)
    flagged = [False] * n
    if n < 2:
        return flagged
    lsh = MinHashLSH(threshold=jaccard, num_perm=NUM_PERM)
    hashes: Dict[int, MinHash] = {}
    shingle_sets: Dict[int, set] = {}
    for i, t in enumerate(texts):
        toks = _tokens(t)
        if len(toks) < MIN_TOKENS_FOR_DUP:
            continue
        sh = _shingles(toks)
        m = MinHash(num_perm=NUM_PERM)
        for s in sh:
            m.update(s.encode("utf-8"))
        hashes[i] = m
        shingle_sets[i] = sh
        lsh.insert(str(i), m)
    for i, m in hashes.items():
        cands = [int(c) for c in lsh.query(m) if int(c) != i]
        if not cands:
            continue
        # confirm with exact Jaccard to cut LSH false positives
        real = [j for j in cands
                if _jaccard(shingle_sets[i], shingle_sets[j]) >= jaccard]
        if not real:
            continue
        others = [j for j in real if authors[j] != authors[i]]
        if others or len(real) >= 2:
            flagged[i] = True
    return flagged


def _bursts(dates: Iterable, authors: Iterable, per_day: int) -> List[bool]:
    key = Counter()
    pairs = list(zip(pd.to_datetime(list(dates)).date, authors))
    for d, a in pairs:
        if a and a != "[deleted]":
            key[(d, a)] += 1
    return [bool(a) and a != "[deleted]" and key[(d, a)] >= per_day
            for d, a in pairs]


def _disclosed(texts: Iterable[str], authors: Iterable[str]) -> List[bool]:
    return [bool(_DISCLOSED.search(t or "")) or bool(_BOT_NAME.search(a or ""))
            for t, a in zip(texts, authors)]


def _low_diversity(texts: List[str], authors: List[str]) -> List[bool]:
    by_author: Dict[str, List[int]] = defaultdict(list)
    for i, a in enumerate(authors):
        if a and a != "[deleted]":
            by_author[a].append(i)
    flagged = [False] * len(texts)
    for a, idx in by_author.items():
        if len(idx) < DIVERSITY_MIN_POSTS:
            continue
        sets = {i: _shingles(_tokens(texts[i])) for i in idx}
        ref = sets[idx[0]]
        if len(ref) < MIN_TOKENS_FOR_DUP:
            continue
        sims = [_jaccard(ref, sets[i]) for i in idx[1:]]
        if sims and sum(s >= DIVERSITY_JACCARD for s in sims) / len(sims) >= 0.6:
            for i in idx:
                flagged[i] = True
    return flagged


def _deleted(authors: Iterable[str]) -> List[bool]:
    return [(not a) or a == "[deleted]" for a in authors]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def screen_posts(posts: pd.DataFrame, jaccard: float = None,
                 burst_per_day: int = None) -> pd.DataFrame:
    """Return ``posts`` with ``bot_score`` and ``bot_reasons`` columns added.

    Args:
        posts: Frame with at least ``title``; ``selftext``, ``author``
            and ``date`` are used when present. Missing columns disable
            the signals that need them.
        jaccard: Near-duplicate similarity cut; default from settings.
        burst_per_day: Burst threshold; default from settings.

    Returns:
        A copy of ``posts`` with the two new columns. Rows are not
        removed; see :func:`apply_screen`.
    """
    from src import settings
    if jaccard is None:
        jaccard = settings.get_float("bot_screen_duplicate_jaccard")
    if burst_per_day is None:
        burst_per_day = settings.get_int("bot_screen_burst_posts_per_day")

    out = posts.copy()
    n = len(out)
    if n == 0:
        out["bot_score"] = pd.Series(dtype=float)
        out["bot_reasons"] = pd.Series(dtype=str)
        return out
    texts = _texts(out).tolist()
    authors = (out["author"].fillna("").astype(str).tolist()
               if "author" in out else [""] * n)
    dates = out["date"].tolist() if "date" in out else [pd.NaT] * n

    flags = {
        "near_duplicate": _near_duplicates(texts, authors, jaccard),
        "burst": (_bursts(dates, authors, burst_per_day)
                  if "date" in out and "author" in out else [False] * n),
        "disclosed": _disclosed(texts, authors),
        "low_diversity": (_low_diversity(texts, authors)
                          if "author" in out else [False] * n),
        "deleted_author": _deleted(authors) if "author" in out else [False] * n,
    }
    scores, reasons = [], []
    for i in range(n):
        hit = [k for k, v in flags.items() if v[i]]
        scores.append(min(1.0, sum(WEIGHTS[k] for k in hit)))
        reasons.append("|".join(hit))
    out["bot_score"] = scores
    out["bot_reasons"] = reasons
    return out


def apply_screen(posts: pd.DataFrame, threshold: float = None,
                 enabled: bool = None) -> Tuple[pd.DataFrame, dict]:
    """Score ``posts`` and drop rows at or above the threshold.

    Returns:
        ``(kept, report)``. ``kept`` is the frame of surviving rows
        (original columns only). ``report`` holds ``rows_in``,
        ``rows_excluded``, ``exclusion_rate``, ``threshold`` and
        ``top_reasons`` (reason combination -> count). When the screen
        is disabled in settings, nothing is dropped and the report says
        so.
    """
    from src import settings
    if enabled is None:
        enabled = settings.get_bool("bot_screen_enabled")
    if threshold is None:
        threshold = settings.get_float("bot_screen_threshold")
    report = {"enabled": bool(enabled), "threshold": float(threshold),
              "rows_in": int(len(posts)), "rows_excluded": 0,
              "exclusion_rate": 0.0, "top_reasons": {}}
    if not enabled or len(posts) == 0:
        return posts, report
    scored = screen_posts(posts)
    bad = scored["bot_score"] >= threshold
    report["rows_excluded"] = int(bad.sum())
    report["exclusion_rate"] = float(bad.mean())
    report["top_reasons"] = dict(Counter(
        scored.loc[bad, "bot_reasons"]).most_common(8))
    kept = scored.loc[~bad, list(posts.columns)].reset_index(drop=True)
    return kept, report


def write_report(report: dict, path: str) -> None:
    """Persist ``report`` as JSON (creating the folder), with a timestamp."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rep = dict(report)
    rep["checked_utc"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    os.replace(tmp, path)


def format_report(report: dict) -> str:
    """One log line summarising a report."""
    if not report.get("enabled", True):
        return "bot screen: disabled (config/settings.csv)"
    n, x = report["rows_in"], report["rows_excluded"]
    top = ", ".join(f"{k or 'none'}:{v}" for k, v in
                    list(report.get("top_reasons", {}).items())[:4])
    return (f"bot screen: excluded {x:,} of {n:,} posts "
            f"({report['exclusion_rate']:.1%}) at threshold "
            f"{report['threshold']:.2f}" + (f" | {top}" if top else ""))
