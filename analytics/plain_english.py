"""One glossary for the whole project: research notation -> desk English.

WHY THIS MODULE EXISTS
----------------------
The research half of this repo uses research notation, and that is the right
notation for a research record: ``s_conf``, ``s_z``, ``s_enh``, ``e1..e5``, AP,
AUROC, homophily, DICE.  It is the wrong notation for a portfolio manager
reading a chart between meetings.  The recorded decision (2026-07-27) was
therefore:

* the METHODS keep their formal definitions, unchanged - the research must
  stay reproducible;
* every LABEL a human reads is translated into the words that human would
  use;
* the translation lives in exactly ONE file, imported by the dashboard and by
  every notebook, so the screen and the research record can never drift apart
  and start calling the same quantity two different things.

WHY NOT JUST RENAME THE COLUMNS
-------------------------------
Because the stored parquet column names are an interface.  Renaming ``e1`` to
``attention_vs_its_own_year`` inside ``data/`` would invalidate every cached
frame, every test that asserts on a schema, and every notebook output already
saved - for a purely cosmetic gain.  Translation belongs at the display layer,
which is here.

WHY A DICT AND NOT f-STRINGS AT EACH SITE
-----------------------------------------
There were 30-odd places where these names reach a human.  Spelled out inline,
a wording change means 30 edits and a guaranteed miss, and the two that get
missed are the ones a reader trips over.

USAGE
-----
    from analytics.plain_english import plain, relabel, glossary_md

    plain("s_conf")            -> 'how reliably their calls worked'
    relabel(df)                -> the same frame with human column headers
    print(glossary_md())       -> a markdown definition list for a notebook
"""
from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# THE GLOSSARY
#
# Keys are the names as STORED or as the research paper writes them.  Values
# are what a PM would say.  Keep values lower-case and short enough to fit a
# chart axis; the longer explanation belongs in DEFINITIONS below.
# ---------------------------------------------------------------------------
PLAIN: dict[str, str] = {
    # --- euphoria components (stored as e1/e2/e3/e5 since the first build) --
    "e1": "attention vs its own year",
    "e2": "how long the mood stayed bullish",
    "e3": "rate of new people arriving",
    "e5": "attention going near-vertical",
    "hype_ok": "crowd big enough to signal",
    # --- onset bank (notebook 02) ------------------------------------------
    "attention_accel": "this week busier than this month",
    "hype_ratio": "crowd size vs its own normal",
    "bull_inflection": "mood turning up",
    "influx_speed": "new arrivals, at double speed",
    "attention_convexity": "attention going near-vertical",
    "source_breadth": "how many platforms are talking",
    "fade": "mood rolling over while the crowd is still large",
    "level": "euphoria level",
    "boom_state": "price already run up from its own low",
    "onset": "euphoria starting (CONSIDER)",
    "top": "euphoria ending (WARNING)",
    "lead_days": "days of warning before the drop",
    "hit": "the drop actually came",
    # --- influence scoring ------------------------------------------------
    "composite": "usefulness score",
    "s_conf": "how reliably their calls worked",
    "s_z": "how much better than chance",
    "s_enh": "how much they move the room",
    "consensus": "net direction",
    "weighted_voices": "influence behind it",
    "n_calls": "calls",
    "n_authors": "people",
    "n_judged": "calls judged against prices",
    "hit_rate": "share of their calls that worked",
    "base_rate": "what an average call scores",
    "tier": "usefulness band",
    "called_tops": "called tops",
    "bought_tops": "bought tops",
    "loud_but_wrong": "loud but wrong",
    "stance": "how strongly it was worded",
    "direction": "long or short",
    # --- model metrics -----------------------------------------------------
    "AP": "precision when it ranks (average precision)",
    "AUROC": "how well it sorts good from bad",
    "prevalence": "base rate - what guessing would score",
    "lift": "how many times better than guessing",
    "homophily": "how often connected people share a label",
    "k-core": "the densely-replying core of the graph",
    "DICE": "deliberately rewiring edges to test if they matter",
    "modularity": "how cleanly the graph splits into groups",
    "betweenness": "how often someone sits on the path between others",
    "pagerank": "how much of the conversation flows through them",
    "degree": "how many people they exchange replies with",
    "ci_lo": "worst case of the confidence interval",
    "ci_hi": "best case of the confidence interval",
    "p_value": "chance of seeing this by luck alone",
    # --- episode ground truth (notebook 01) --------------------------------
    "episode": "one boom-then-bust arc",
    "trough": "the measured low the run-up started from",
    "peak": "the confirmed top",
    "boom_pct": "how far it ran up",
    "bust_pct": "how far it fell afterwards",
    "run_days": "days from the low to the top",
    "onset_detectable": "enough posts to be judged on the start",
    "top_detectable": "enough posts to be judged on the top",
}

# ---------------------------------------------------------------------------
# THE DEFINITIONS
#
# One or two sentences each, written to answer "what IS that" for a reader who
# has never seen the notebooks.  The notebooks print this; the dashboard uses the
# short PLAIN form in help= tooltips.  Anything a defence panel could ask
# "define that" about should have an entry.
# ---------------------------------------------------------------------------
DEFINITIONS: dict[str, str] = {
    "euphoria level": (
        "One 0-100 number per name per day, combining four measured "
        "ingredients of crowd behaviour. It is a LEVEL, not a forecast - high "
        "means the crowd is already hot, not that a fall is due tomorrow."),
    "attention vs its own year": (
        "Today's posting volume compared with this name's own past year, in "
        "standard deviations. Compared with ITSELF because 400 posts a day is "
        "quiet for TSLA and a mania for a small ETF."),
    "how long the mood stayed bullish": (
        "How many of the recent days ran bullish rather than bearish. A crowd "
        "that has been one-sided for weeks is different from a single loud "
        "day."),
    "rate of new people arriving": (
        "How fast previously-unseen authors are joining the conversation. "
        "New entrants, not the regulars talking more, are what marks a mania."),
    "attention going near-vertical": (
        "The slope of attention - how fast the volume itself is accelerating. "
        "This is the ingredient that separates a steady high level from a "
        "blow-off."),
    "crowd big enough to signal": (
        "A gate, not a score: the name must actually be busy before any "
        "signal is allowed to fire. It stops a thin, quiet name producing a "
        "signal off five posts."),
    "average precision": (
        "Average precision (AP) asks: when the model RANKS candidates, how "
        "much of the top of that ranking is genuinely positive? It is the "
        "metric to use when positives are rare - with 5% positives, a model "
        "that predicts 'no' for everything scores 95% accuracy and is "
        "useless, but scores an AP at the random floor, which is honest."),
    "the random floor": (
        "The AP a coin-flip ranking would get, which equals the share of "
        "positives in the data. Every AP in this project is reported next to "
        "it, because an AP of 0.10 is excellent at a 1% base rate and "
        "worthless at a 20% one."),
    "how well it sorts good from bad": (
        "AUROC: pick one positive and one negative at random - the "
        "probability the model scores the positive higher. 0.50 is a "
        "coin-flip, 1.00 is perfect. It is readable but flattering on rare "
        "classes, which is why AP leads and AUROC follows."),
    "homophily": (
        "The share of connections that join two people with the SAME label. "
        "High homophily is the assumption every graph neural network is built "
        "on; measuring it low for the class you care about predicts, in "
        "advance, that message-passing will fail."),
    "DICE": (
        "A deliberate attack on the graph: delete edges that join same-label "
        "people, add edges that join different-label people. If a model's "
        "score DROPS, the graph structure was carrying real signal. If the "
        "score RISES, the structure was noise the model was being misled by."),
    "confidence interval": (
        "The range the measurement would plausibly fall in if the experiment "
        "were repeated. This project adopts nothing whose interval includes "
        "zero, because 'better on average' with an interval spanning zero is "
        "indistinguishable from luck."),
    "paired seeds": (
        "Running challenger and champion on the SAME random splits, then "
        "comparing them split by split. Pairing removes the split-to-split "
        "luck that otherwise swamps a small real difference."),
    "walk-forward": (
        "Fit on the past, score the next unseen stretch, roll forward, "
        "repeat. Every number in this project is out-of-sample by "
        "construction; nothing is scored on data used to choose it."),
    "lead time": (
        "How many days before the actual drop the WARNING fired. A signal "
        "that is right but simultaneous is a description, not a warning."),
    "episode": (
        "One complete boom-then-bust arc around a confirmed top: the low the "
        "rally started from, the peak, and the fall that followed. Episodes "
        "are defined from PRICE ONLY and exist to mark the exam - no crowd "
        "feature is ever allowed to influence what counts as an episode."),
    "the onset window": (
        "The stretch of days in which a CONSIDER alert counts as having caught "
        "the start: from the trough to 45 days later, and never past the peak. "
        "An alert after the top is not an early call however close it lands."),
    "detectable": (
        "Whether the archive was thick enough on that stretch for ANY "
        "crowd-only detector to have had a chance - at least 100 tagged posts "
        "in 28 days. An episode in a thin period is one the detector was blind "
        "to, which is different from one it got wrong, so the two are counted "
        "in separate columns and never merged."),
    "mention share": (
        "Of everything posted today, the fraction that talks about this name. "
        "A share rather than a count, so a name does not look euphoric merely "
        "because the whole platform had a busy day."),
    "percentile rank": (
        "Where today sits inside this name's OWN last 365 days: 0 = the "
        "quietest it has been, 1 = the most extreme. Every feature is ranked "
        "against itself because 400 posts a day is quiet for TSLA and a mania "
        "for a small ETF."),
    "this week busier than this month": (
        "attention_accel - the 7-day mention share minus the 28-day share, "
        "ranked. Positive means the crowd arriving now is bigger than the "
        "crowd that was already there."),
    "crowd size vs its own normal": (
        "hype_ratio - the 7-day share divided by this name's own 120-day "
        "median, ranked. It is the existing hype gate made continuous: that "
        "gate fires at 2x, this feature reports the ratio itself."),
    "mood turning up": (
        "bull_inflection - the 14-day change in the 14-day net-bullish share. "
        "It is the fade rule pointed the other way: fade catches mood rolling "
        "over, this catches mood rolling up."),
    "new arrivals, at double speed": (
        "influx_speed - the 14-day change in mention share, which is the "
        "crowd-influx feature measured over half the horizon. Starts are about "
        "rate of change, so the shorter window is the point."),
    "how many platforms are talking": (
        "source_breadth - the count of distinct posting sources active in 7 "
        "days. REJECTED from the bank: the archive only gained its second and "
        "third source in 2026, so before then the feature is a constant and "
        "afterwards it mostly encodes which year it is."),
    "boom and bust thresholds": (
        "How far a name must run up, and then fall, before the arc counts as "
        "an episode at all: up 25% (theme ETF) or 50% (single name) off its "
        "own 120-day low, then down 15% / 30% within 90 days. Single names are "
        "held to the harder bar because they move further on nothing."),
}


def plain(name: str, default: str | None = None) -> str:
    """Translate one stored/research name into desk English.

    Unknown names come back UNCHANGED rather than raising: the glossary is a
    courtesy layer, and a missing entry should print an untranslated label,
    never break a chart mid-render.
    """
    return PLAIN.get(name, name if default is None else default)


def relabel(df: pd.DataFrame, extra: dict[str, str] | None = None
            ) -> pd.DataFrame:
    """Return a COPY of ``df`` with human-readable column headers.

    A copy, not a rename in place, because the caller almost always still
    needs the machine names for the next computation - silently mutating the
    frame it just handed us is how display code corrupts analysis code.
    """
    mapping = {c: plain(c) for c in df.columns if c in PLAIN}
    if extra:
        mapping.update(extra)
    return df.rename(columns=mapping)


def glossary_md(keys: list[str] | None = None) -> str:
    """A markdown definition list, for printing at the top of a notebook.

    ``keys`` selects and ORDERS the entries, so each notebook can define only
    the terms it actually uses - a wall of 40 definitions is skipped, six
    relevant ones are read.
    """
    keys = list(DEFINITIONS) if keys is None else keys
    lines = []
    for k in keys:
        body = DEFINITIONS.get(k)
        if body:
            lines.append(f"- **{k}** - {body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DISPLAY-ONLY MASKING OF OBSCENE HANDLES
#
# WHY THIS IS HERE AND NOT IN THE STORE
# -------------------------------------
# The influence board is built from real Reddit handles, and a meaningful
# minority of them are unprintable.  The requirement (2026-07-28) was to
# "censor the innapropriate stuff with **".  That is a DISPLAY concern, so it
# belongs in this module for exactly the reason the module docstring gives: the
# stored parquet is an interface.  Rewriting handles inside
# `data/reference/influence/` would (a) silently change the join key that
# `calls.parquet`, `reply_edges.parquet` and `author_scores.parquet` share,
# (b) make two different authors collide the moment their masked forms match,
# and (c) destroy the ability to re-judge an author against new price data.
# Nothing below touches storage; it rewrites a STRING on its way to a screen.
#
# WHY TWO TIERS AND NOT ONE WORD LIST
# -----------------------------------
# The obvious implementation - one list of stems, matched as substrings - was
# BUILT AND MEASURED AGAINST ALL 12,528 REAL HANDLES FIRST, and it is wrong.
# It flagged 384, and the false positives were not marginal: `Painkiller_830`
# (kill), `AssumptionPretty7018` and `passionlessDrone` and `cow_grass` (ass),
# `Tricky-Doughnut-6429` (nut), `buffetite` and `Stitch426` (tit),
# `BlownCamaro` (blow).  Masking an innocent handle is not a harmless error:
# the whole point of keeping the un-offending part of a handle visible is that
# a PM can tell two authors apart, and a wrongly-mangled name breaks that
# while also looking careless in front of the desk.
#
# The split that fixes it follows from WHERE each word can legitimately occur:
#
#   TIER A - stems that never appear inside an innocent English word.  Matched
#            as a SUBSTRING, because obfuscated spellings
#            (`fucktheredditapp15`, `RHfuckedup`) run the stem straight into
#            other characters with no separator to key off.
#
#   TIER B - words that are crude alone but sit inside longer innocent words
#            (`ass` in assumption, `tit` in title, `nut` in doughnut, `anal`
#            in analyst, `rape` in grapefruit, `cock` in cockroach).  Matched
#            only when the handle, split into tokens, contains the word AS A
#            WHOLE TOKEN.
#
# WHICH TIER EACH STEM LANDS IN WAS DECIDED BY COUNTING, NOT BY INTUITION.
# Every candidate stem was run over the whole corpus and its hits read by eye.
# Six stems that intuition puts in Tier A had to be demoted, because as
# substrings they hit innocent words far more often than obscene handles:
#   anal -> AfraidAnalyst, Band10_Analyst, Valuable-Analyst-464,
#           scientia_analytica         (3 innocent vs 2 genuine)
#   rape -> every one of Ok-Grapefruit2910, Own_Grapefruit8839,
#           RepulsiveGrapefruit, SpecialGrapefruit208, GrapefruitOrganic741,
#           Foreign_GrapeStorage, _grapevan  (7 innocent vs 0 genuine - the
#           corpus contains no handle using the word at all)
#   cock -> Calm_Cockroach_5284, Winter_Cockroach_753, Pool_cocktail_repeat
#   boob -> BooBeef, lulubooboo28   (the stem straddles `Boo`+`B`)
#   piss -> Dippissippi            (1 innocent vs 0 genuine)
#   wank -> sobewankanobe          (an Obi-Wan pun; 1 innocent vs 0 genuine)
# Demoting them costs six genuinely crude handles that hide the word inside a
# longer run (`Cockballzz`, `redditsuckscockss`, `TRASHTALKINGCOCKSTAR`,
# `3boobsarenice`, `eskimoboob`, `analbuttlick`) and buys back 15 innocent
# ones.  That is the trade this module deliberately takes: see the note on the
# direction of error below.
#
# A SECOND RULE, WHICH COSTS NOTHING: a Tier A match must lie INSIDE ONE
# TOKEN.  Several false positives were the stem straddling a word boundary -
# `SatoshiTrails` and `TheSatoshiTimes` ("...oshi|Trails" -> shit),
# `MeridianAllocation` ("Meridian|Allocation" -> anal).  A stem spanning two
# words is by construction not the word being written, so requiring
# containment removes those with no list to maintain and no genuine loss.
#
# WHY THE TOKENISER LOOKS LIKE THAT
# ---------------------------------
# Reddit handles carry their word boundaries in four different notations at
# once, so all four have to be honoured or Tier B does nothing:
#   `just_lick_my_ass`   -> underscores
#   `dick-knuckle`       -> hyphens
#   `AssumptionPretty`   -> camelCase
#   `Painkiller_830`     -> a letter/digit boundary
# `[A-Z]+(?![a-z])` comes first so an acronym run stays one token and the word
# after it still splits: `RHfuckedup` -> [RH, fuckedup] rather than [R, Hfuck...].
#
# WHAT IS DELIBERATELY *NOT* MASKED
# ---------------------------------
# `suck`, `kill`, `damn`, `hell` and `crap` were tested and left out.  They are
# not obscene, and their hits are handles no desk would blink at
# (`Feb17Sucks`, `TheRedditModsSuck`, `ISuckAtJavaScript12`, `p8inKill3r`).
# Masking them would make the board look bowdlerised without hiding anything
# anyone objects to.  `ball` singular is out for the same reason `nut` is
# token-only: it would mask innocent tokens, and it costs one handle
# (`BallSmashingForever`).
#
# HONEST CLASSIFICATION
# ---------------------
# These lists are a CONVENTION (Class 3), not a learned or derived parameter.
# There is no ground truth for "offensive" to fit against, so no amount of
# bootstrapping would make them evidence-backed.  What IS evidence is the
# measured behaviour, recorded in `docs/RESEARCH_RECORD.md`: how many of
# the 12,528 real handles are masked, which stem fires each one, and the
# residual errors named individually.  Under-masking is the deliberate
# direction of the error: a missed handle is one embarrassing name on a board
# the desk already knows is scraped from Reddit, while over-masking corrupts
# identity for every reader of the leaderboard.
# ---------------------------------------------------------------------------

#: Matched as substrings (within a single token), case-insensitively.
_OBSCENE_SUBSTRING = (
    "fuck", "cunt", "shit", "nigg", "fag", "pussy", "whore", "slut",
    "jizz", "penis", "benis", "vagina", "dick", "milf", "horny", "turd",
    "bitch", "bastard", "analingus",
    # PROMOTED to substring matching after counting: every hit in the corpus
    # is genuine, so the safer token rule would only lose coverage.
    #   retard -> 10 hits, 10 genuine (it also picks up `Retardation-Syndrome`,
    #             which whole-token matching misses)
    #   boobs  ->  3 hits,  3 genuine (`3boobsarenice`, `PlzSendCDKeysNBoobs`,
    #             `I_love_boobs86`) - note the SINGULAR `boob` stays a token,
    #             because that is the spelling that straddles `Boo`+`B`
    "retard", "boobs",
    # `tits` was tested for promotion and REJECTED: 2 hits, one genuine
    # (`Murrrtits`) and one not (`Iplayminecraftitsfun` - "minecraft its
    # fun"), and the innocent one is a single token so the containment rule
    # cannot separate them. A 50% error rate is not worth one handle.
)

#: Matched only as a WHOLE TOKEN after the handle is split on separators,
#: digit boundaries and camelCase transitions.
_OBSCENE_TOKEN = frozenset({
    "ass", "asses", "arse", "tit", "tits", "titty", "nut", "nuts",
    "balls", "ballsack", "butt", "sex", "sexy", "cum", "cums", "hole",
    "nazi", "hitler", "retard", "retards", "retarded",
    # demoted from substring matching on the measured evidence above
    "anal", "cock", "cocks", "boob", "boobs", "piss", "wank", "rape",
})

# One compiled alternation for Tier A. Longest-first so that when two stems
# overlap the wider span wins and the mask does not leave half a word behind.
_SUB_RE = re.compile(
    "|".join(sorted((re.escape(w) for w in _OBSCENE_SUBSTRING),
                    key=len, reverse=True)),
    re.IGNORECASE)

# Acronym run | Capitalised word | lower run | digit run - see the note above.
_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")

MASK = "**"
_MASK_RUN_RE = re.compile("(?:" + re.escape(MASK) + ")+")


def _mask_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of ``text`` that the two tiers want hidden.

    Returned spans are sorted and non-overlapping: adjacent or overlapping
    hits are merged so `fuckshit` becomes one `**` rather than two, which is
    both shorter to read and stops the mask itself from looking like a word.
    """
    toks = [m.span() for m in _TOKEN_RE.finditer(text)]
    spans = [m.span() for m in _SUB_RE.finditer(text)
             if any(lo <= m.start() and m.end() <= hi for lo, hi in toks)]
    spans += [(lo, hi) for lo, hi in toks
              if text[lo:hi].lower() in _OBSCENE_TOKEN]
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1]:          # overlapping or touching
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def is_obscene(handle: str) -> bool:
    """True when ``handle`` contains anything the display layer should hide.

    Exposed separately from `censor` so callers can COUNT without rewriting -
    the parameter-register audit and the unit tests both need that, and
    inferring it from `censor(h) != h` would be a subtler test than the thing
    it is testing.
    """
    return bool(handle) and bool(_mask_spans(str(handle)))


def censor(handle: str) -> str:
    """A handle safe to put on a screen, with only the offending spans hidden.

    Each offending span collapses to ``**`` and EVERYTHING ELSE SURVIVES
    (`just_lick_my_ass` -> `just_**_my_**`), because the surrounding
    characters are what let a reader tell two authors apart on a leaderboard.
    Fully replacing the handle with `**` would make every masked author look
    like the same author.

    Non-strings and blanks pass through untouched: this sits on the hot path
    of every influence chart, and a missing author must render as a gap, not
    raise mid-figure.
    """
    if handle is None or not isinstance(handle, str) or not handle:
        return handle
    out = handle
    # ITERATE TO A FIXED POINT, and the reason is a real case in the store,
    # not defensiveness. `Buttslut69696969` tokenises as [Buttslut, 69696969]
    # - `butt` is not a whole token there, so only Tier A's `slut` fires and
    # the first pass yields `Butt**69696969`. NOW `Butt` IS a whole token, so
    # a single pass would leave a crude word on screen having "censored" the
    # handle. Two passes reach `**69696969`. The bound is small because each
    # pass strictly shortens the letters available to match; four is slack
    # over the deepest case observed (two).
    for _ in range(4):
        spans = _mask_spans(out)
        if not spans:
            break
        parts, prev = [], 0
        for lo, hi in spans:
            parts.append(out[prev:lo])
            parts.append(MASK)
            prev = hi
        parts.append(out[prev:])
        # Collapse runs of masks: two adjacent hits should read as one gap.
        # Without this the second pass above produces `****69696969`, which
        # looks like a rendering bug rather than a redaction.
        out = _MASK_RUN_RE.sub(MASK, "".join(parts))
    return out


# ---------------------------------------------------------------------------
# HALF-MASKING - the identity layer that sits on top of the obscenity layer
#
# Desk 2026-08-12: "can you abstract the names with like *** but can kinda
# see? like half abstract it".
#
# `censor` above hides only the OFFENSIVE spans of a handle and is not an
# anonymity measure - `tomato232` reaches the screen intact. This pair
# hides most of the IDENTITY while leaving enough of it that a reader can
# still tell two rows apart, follow one person down a page, and recognise
# a name they already know. The two run in order: obscenity first (so a
# crude word cannot survive inside a revealed prefix), identity second.
#
# WHY NOT JUST `***`. A leaderboard where every row reads `***` is not a
# leaderboard - the reader cannot tell whether row 3 and row 9 are the same
# person, and the ego-network picker becomes unusable. Same reasoning
# `censor` already records for why it does not replace whole handles.
#
# THE REVEAL IS PROPORTIONAL, not a fixed prefix: a fixed 3 characters
# leaves `Bob` fully exposed and `Independent-Use-2281` almost entirely
# hidden. Roughly a third at the front and a sixth at the end, and never
# more than half the handle in total.
REVEAL_FRONT = 0.34         # share of the handle shown at the start
REVEAL_BACK = 0.17          # share shown at the end
IDENT_MASK = "***"


def half_mask(handle: str) -> str:
    """A handle abstracted for display: recognisable, not identifying.

    `tomato232` -> `tom***2`,  `Love-to-Trade101` -> `Love***01`.

    Obscenity is masked FIRST (via `censor`), so a revealed prefix can
    never carry a crude word through. Non-strings and blanks pass through
    untouched - this sits on the hot path of every influence chart and a
    missing author must render as a gap, not raise mid-figure.
    """
    if handle is None or not isinstance(handle, str) or not handle:
        return handle
    h = censor(handle)
    n = len(h)
    if n <= 4:
        # Nothing to hide usefully: reveal the first character only. Short
        # handles are the one case where proportional reveal degenerates.
        return h[:1] + IDENT_MASK
    front = max(2, round(n * REVEAL_FRONT))
    back = max(1, round(n * REVEAL_BACK))
    budget = max(3, n // 2)             # never show more than half
    while front + back > budget and (front > 2 or back > 1):
        if back > 1:
            back -= 1
        else:
            front -= 1
    return f"{h[:front]}{IDENT_MASK}{h[n - back:]}"


def half_mask_series(s: pd.Series) -> pd.Series:
    """Vectorised `half_mask`, COLLISION-SAFE, returned as a COPY.

    Hiding the middle of a handle can map two genuinely different people
    onto one label - `trader_bull_99` and `trader_bear_99` both become
    `trad***99` - and a leaderboard that shows one person twice under the
    same name is worse than one that shows nothing. Where a mask is shared
    by more than one TRUE handle, a stable two-character tag is appended
    so the rows stay distinguishable: `trad***99·a7`.

    The tag comes from md5 of the true handle, not Python's `hash`, which
    is salted per process - the same person must carry the same label
    across runs or the board cannot be read week to week.

    A copy, for the same reason `censor_series` copies: the caller nearly
    always still needs the true handle as a join key one line later, and
    display code that mutates the analysis frame in place is how a masked
    name ends up written back into the store.
    """
    masked = s.map(half_mask)
    true = s.astype(str)
    # a mask is ambiguous when it stands for more than one distinct handle
    pairs = pd.DataFrame({"m": masked, "t": true}).dropna()
    groups = pairs.groupby("m")["t"].unique()
    clashing = {m: list(ts) for m, ts in groups.items() if len(ts) > 1}
    if not clashing:
        return masked
    import hashlib

    def _digest(t):
        return hashlib.md5(str(t).encode("utf-8")).hexdigest()

    # THE TAG LENGTH IS CHOSEN PER GROUP, not fixed. Two hex characters is
    # 256 buckets, and over ~19k handles the birthday collisions are not
    # hypothetical - `Marketspike` and `Markthehare` both mask to `Mark***e`
    # and both hashed to `68`, so a fixed 2-char tag reunited exactly the
    # two rows it was added to separate. Grow the prefix until the group is
    # unique, which is 2 characters for essentially every real group.
    tag_of: dict[str, str] = {}
    for m, ts in clashing.items():
        digs = {t: _digest(t) for t in ts}
        k = 2
        while k < 32 and len({d[:k] for d in digs.values()}) < len(digs):
            k += 1
        for t, d in digs.items():
            tag_of[t] = f"{m}\u00b7{d[:k]}"

    return pd.Series([tag_of.get(t, m) if m in clashing else m
                      for m, t in zip(masked, true)],
                     index=s.index, name=s.name)


def censor_series(s: pd.Series) -> pd.Series:
    """Vectorised `censor` for a column of handles, returned as a COPY.

    A copy for the same reason `relabel` copies: the caller nearly always
    still needs the true handle as a join key one line later, and display
    code that mutates the analysis frame in place is how a masked name ends
    up written back into the store.
    """
    return s.map(censor)


# ---------------------------------------------------------------------------
# THEME SLUGS -> DESK ENGLISH
#
# `src/themes.py` spells a theme as a snake_case slug because it is a dict
# key: `ev_clean_energy` is a good identifier and a bad chart label.  The
# translation lives HERE, with the rest of the display layer, for the same
# reason the column glossary does - the stored spelling never changes, only
# what a human reads, so nothing that joins on a theme can be broken by a
# relabelling.
#
# Only two rules, and both are mechanical rather than a per-theme dictionary
# (39 hand-written labels is 39 chances to let one drift out of sync with
# `THEME_TICKERS`): underscores become spaces, and a word in _ACRONYMS gets
# its house spelling.  A new theme added to src/themes.py therefore gets a
# sensible label with no edit here at all, which is the point - the map holds
# only the words that plain capitalisation would get WRONG.
# ---------------------------------------------------------------------------
_ACRONYMS = {"ai": "AI", "ev": "EV", "saas": "SaaS", "glp1": "GLP-1"}


# Whole-slug display overrides, for the few themes whose slug is not the
# name to put on screen. DISPLAY ONLY: the slug stays the key in every
# store, config file and model, so nothing here can move a number or
# orphan history.
#
# short_squeeze: anchored to ARKK, which config/theme_etfs.csv itself
# records as a proxy ("no squeeze ETF exists"). A squeeze name against a
# fund holding none of those positions invites the reader to take the
# price panel as the theme's own line, so the label names the instrument
# actually drawn.
#
# meme_stocks is the SECOND theme anchored to ARKK, and config records
# the same caveat for it ("ARKK holds none of the meme names"). It is
# labelled the same way for the same reason.
#
# CONSEQUENCE, recorded because it is visible on screen: two themes now
# render under one name. They remain separate series with separate
# crowd signals - only the label collides - so any list that shows both
# (the instrument dropdown, the ETF radar) will carry two rows reading
# "ARK Innovation (ARKK)". Merging or hiding one is the fix if that
# ambiguity matters; renaming alone cannot resolve it.
#
# quantum_computing: anchored to IYW, which config/theme_etfs.csv records
# as "a broad-tech proxy; QTUM is the natural line (not approved)". A
# quantum name against a broad US technology fund invites the reader to
# take the price panel as the theme's own line, so the label names the
# instrument actually drawn. The crowd signal underneath is unchanged -
# it still measures quantum-computing chatter.
_LABEL_OVERRIDES = {"short_squeeze": "ARK Innovation",
                    "meme_stocks": "ARK Innovation",
                    "quantum_computing": "US Technology"}


def theme_label(slug: str) -> str:
    """`ai_megacap` -> `AI megacap`, `ev_clean_energy` -> `EV clean energy`.

    Sentence case, not Title Case: the labels sit inside chart captions and
    hover text as ordinary nouns, and Title Case on a scatter reads as a
    proper name ("Gold Metals" looks like a company). A slug listed in
    _LABEL_OVERRIDES returns that name verbatim instead."""
    key = str(slug).strip().lower()
    if key in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[key]
    words = [w for w in str(slug).split("_") if w]
    if not words:
        return str(slug)
    out = [_ACRONYMS.get(w, w) for w in words]
    if words[0] not in _ACRONYMS:
        out[0] = out[0][:1].upper() + out[0][1:]
    return " ".join(out)
