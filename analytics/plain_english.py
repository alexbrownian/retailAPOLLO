"""One glossary for the whole project: research notation -> desk English.

WHY THIS MODULE EXISTS
----------------------
The methods in this repo follow Chan (2026) closely, and her notation is the
right notation for a thesis: ``s_conf``, ``s_z``, ``s_enh``, ``e1..e5``, AP,
AUROC, homophily, DICE.  It is the wrong notation for a portfolio manager
reading a chart between meetings.  The desk decision (2026-07-27) was
therefore:

* the METHODS keep Chan's definitions, unchanged - the research must stay
  replicable against her thesis;
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
    "onset": "euphoria starting (GET IN)",
    "top": "euphoria ending (GET OUT)",
    "lead_days": "days of warning before the drop",
    "hit": "the drop actually came",
    # --- influence scoring (Chan section 4.6) ------------------------------
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
# has never seen the thesis.  The notebooks print this; the dashboard uses the
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
        "How many days before the actual drop the GET OUT fired. A signal "
        "that is right but simultaneous is a description, not a warning."),
    "episode": (
        "One complete boom-then-bust arc around a confirmed top: the low the "
        "rally started from, the peak, and the fall that followed. Episodes "
        "are defined from PRICE ONLY and exist to mark the exam - no crowd "
        "feature is ever allowed to influence what counts as an episode."),
    "the onset window": (
        "The stretch of days in which a GET IN alert counts as having caught "
        "the start: from the trough to 45 days later, and never past the peak. "
        "An alert after the top is not an early call however close it lands."),
    "detectable": (
        "Whether the archive was thick enough on that stretch for ANY "
        "crowd-only detector to have had a chance - at least 100 scored posts "
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
