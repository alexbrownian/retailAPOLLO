"""Comment-fetch budget: how much comment data one live run may fetch.

Why a budget exists: the 17-subreddit panel produces a measured ~14,000
comments/day, about 140 API pages/day. At the archive's rate contract of
1 request per second, a 7-day gap costs ~16.5 minutes of pure fetching
against a wall-clock ceiling of 10 minutes (PIPELINE_BUDGET_S) for the
entire refresh. Weekly runs, the full panel and the ceiling cannot all
three hold, so the run plans what it can afford from measured quantities
rather than from a typed constant.

Approaches deliberately not used, and why:

* A stopwatch (kill the crawl at N minutes) makes the amount of data
  collected a function of network luck on the day; no two runs are
  comparable.
* W parallel workers each pausing a second is an aggregate W req/s and
  breaks the archive's rate contract. The request rate is not a knob.
* Narrowing the window uniformly advances watermarks past uncollected
  days, which is silent data loss: nothing in the output looks wrong.
* Ranking subreddits by measured yield would require a `subreddit` column
  in the committed influence store.

What runs instead is arithmetic over quantities measured on the machine
that runs the pipeline:

1. A stage ledger (`Data/reference/pipeline_stage_times.json`) times every
   non-fetch stage, so the fetch budget is the ceiling minus what this
   machine actually spends on analytics, folding and prices.
2. A cost ledger (`Data/reference/reddit_comments_cost.json`) records
   pages per day and comments per page per subreddit. A large community
   in a mania and a small one on a quiet day differ by two orders of
   magnitude, so one panel average would misplan both.
3. An allocation that spends the page allowance in proportion to what
   each subreddit owes (its watermark gap x its own measured pages/day),
   cutting every subreddit by the same proportion when the allowance
   falls short, with a floor of one page each so no community is starved
   off the board and then falsely reads as "no influential users here".

Both ledgers are EWMA-smoothed with alpha = 2/(N+1), the standard
EWMA-to-SMA span identity, where N = PANEL_REFERRAL_WINDOW / run cadence
= 28 / 3.2 ~= 9 runs, hence alpha = 0.2. Reusing the project's own 28-day
measurement window (the one E2/E3/A0 use) means the cost estimate tracks
regime change on the same clock the features do.

Both ledgers are machine-local and gitignored: they measure one machine's
speed. Each falls back to a bootstrap prior in `src/config.py` and
re-measures itself within one run.

Running short is a deferral, not data loss. The crawl walks newest-first
and a watermark only advances over ground the run fully covered. A
subreddit that hits its cap keeps its old watermark, so the next run
resumes exactly where this one stopped. Every deferral is printed.

Nothing here touches a signal: these numbers decide how much data a run
fetches, never how anything is scored. Rationale: research.ipynb
sections 3.1b and 3.1b-i.

Public functions: ``record_stage()`` / ``stage_seconds()`` /
``non_fetch_seconds()`` for the stage ledger; ``allowance_pages()`` for
the page allowance; ``record_pages()`` / ``sub_pages_per_day()`` /
``panel_pages_per_day()`` for the cost ledger; ``derived_cadence_days()``
and ``live_lookback_days()`` for the derived cadence; ``allocate()`` for
the per-subreddit split.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

from . import config

# The two machine-local ledgers.
STAGE_LEDGER = os.path.join(config.REFERENCE_DIR, "pipeline_stage_times.json")
COST_LEDGER = os.path.join(config.REFERENCE_DIR, "reddit_comments_cost.json")

# Stages that are not subtracted from the ceiling. "fetch" is the thing
# being budgeted: subtracting it would make the allowance shrink every
# time the crawl used it, a feedback loop that converges on zero. It is
# still recorded, just never spent twice.
_NOT_A_COST = ("fetch",)


# ---------------------------------------------------------------------------
# IO helpers. A ledger is an optimisation, never a blocker.
# ---------------------------------------------------------------------------
def _load(path: str) -> dict:
    """Reads a ledger, tolerating every way it can be unusable.

    A ledger is an accelerator, not a source of truth: if it is missing,
    half-written by a killed run, or hand-edited into invalid JSON, the
    caller falls back to the bootstrap prior and re-measures rather than
    failing over a cache file."""
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: str, obj: dict) -> None:
    """Writes a ledger atomically (temp file in the same directory, then
    os.replace). A run killed mid-write must not leave a truncated JSON
    file behind: _load would then silently discard a real measurement
    history and re-plan from the prior."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ewma(old, new, alpha: float = None) -> float:
    """One EWMA step. The first observation is taken whole rather than
    blended with the prior: the prior is a stand-in for a measurement,
    and once a real measurement of this machine exists there is no reason
    to keep 80% of a number that describes a different machine."""
    a = config.COMMENT_EWMA_ALPHA if alpha is None else alpha
    if old is None:
        return float(new)
    return float(a) * float(new) + (1.0 - float(a)) * float(old)


def fmt_minutes(seconds: float) -> str:
    """Formats seconds for the run summary. Sub-minute costs are printed
    in seconds because '0.0 min' reads as 'free' when it is not."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.0f}s"
    return f"{s / 60.0:.1f} min"


# ---------------------------------------------------------------------------
# 1. The stage ledger: what this machine spends on everything except fetching.
# ---------------------------------------------------------------------------
def record_stage(stage: str, seconds: float) -> None:
    """Folds one stage's wall clock into the ledger.

    Called by update_data.py for every timed step. Only successful runs
    are recorded upstream: a stage that crashed after two seconds is not
    evidence that it takes two seconds. Negative, non-finite or missing
    values are ignored.

    Args:
        stage: Stage name (for example "analytics").
        seconds: Wall-clock seconds the stage took.

    Raises:
        OSError: If the ledger cannot be written; callers treat that as
            non-fatal.
    """
    if not stage or seconds is None:
        return
    sec = float(seconds)
    if sec < 0 or not math.isfinite(sec):
        return
    led = _load(STAGE_LEDGER)
    stages = led.setdefault("stages", {})
    row = stages.setdefault(stage, {})
    row["ewma"] = _ewma(row.get("ewma"), sec)
    row["last"] = sec
    row["n"] = int(row.get("n", 0)) + 1
    led["alpha"] = config.COMMENT_EWMA_ALPHA
    _save(STAGE_LEDGER, led)


def stage_seconds() -> dict:
    """Returns {stage: smoothed seconds} as measured on this machine
    (may be empty)."""
    led = _load(STAGE_LEDGER)
    out = {}
    for name, row in (led.get("stages") or {}).items():
        try:
            v = float(row.get("ewma"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v >= 0:
            out[name] = v
    return out


def non_fetch_seconds(skip_prices: bool = False) -> tuple:
    """Returns what one run costs before any comment fetching.

    A stage the ledger has never seen falls back to its prior
    individually rather than dropping to zero: a machine that has run
    three stages is not told the fourth is free.

    Args:
        skip_prices: Leave the price pull out of the total.

    Returns:
        A pair (seconds, measured). `measured` is False while every stage
        is still on the bootstrap prior; the caller prints which one it
        used, because a budget derived from a prior is a plan and one
        derived from the ledger is an observation.
    """
    have = stage_seconds()
    prior = config.PIPELINE_STAGE_PRIOR_S
    names = set(prior) | set(have)
    total, measured = 0.0, False
    for name in names:
        if name in _NOT_A_COST:
            continue
        if skip_prices and name == "prices":
            continue
        if name in have:
            total += have[name]
            measured = True
        else:
            total += float(prior.get(name, 0.0))
    return total, measured


# ---------------------------------------------------------------------------
# 2. The allowance: the ceiling minus everything else, converted to pages.
# ---------------------------------------------------------------------------
def allowance_pages(skip_prices: bool = False) -> int:
    """Returns how many comment pages this run may spend::

        (PIPELINE_BUDGET_S - measured non-fetch seconds) x COMMENT_RATE_PER_S

    Floored at a single page: a run that can afford nothing still makes
    one request's worth of progress rather than silently becoming
    --skip-comments, because a run that fetches nothing while claiming to
    fetch is the failure this module exists to prevent.
    """
    other_s, _ = non_fetch_seconds(skip_prices=skip_prices)
    left = float(config.PIPELINE_BUDGET_S) - float(other_s)
    return max(1, int(left * float(config.COMMENT_RATE_PER_S)))


# ---------------------------------------------------------------------------
# 3. The cost ledger: what each subreddit actually costs.
# ---------------------------------------------------------------------------
def record_pages(sub: str, pages: int, comments: int, days: float) -> None:
    """Folds one subreddit's observed cost into the ledger.

    Args:
        sub: Subreddit name.
        pages: API pages fetched.
        comments: Comments returned across those pages.
        days: The span the crawl actually covered, so pages/day is
            comparable across a 1-day incremental run and a 30-day
            backfill. A run that covered no measurable span teaches
            nothing and is dropped rather than divided by zero.
    """
    if not sub or not days or float(days) <= 0 or int(pages) <= 0:
        return
    led = _load(COST_LEDGER)
    subs = led.setdefault("subs", {})
    row = subs.setdefault(str(sub), {})
    row["pages_per_day"] = _ewma(row.get("pages_per_day"),
                                 float(pages) / float(days))
    if int(pages) > 0:
        row["comments_per_page"] = _ewma(row.get("comments_per_page"),
                                         float(comments) / float(pages))
    row["n"] = int(row.get("n", 0)) + 1
    led["alpha"] = config.COMMENT_EWMA_ALPHA
    _save(COST_LEDGER, led)


def sub_pages_per_day(sub: str, panel_size: int = 17) -> float:
    """Returns this subreddit's measured pages/day, or its share of the
    panel prior.

    The prior is a panel total, not a per-subreddit number:
    COMMENT_PAGES_PER_DAY_PRIOR = 140 is the measured cost of the whole
    17-subreddit panel (~14,000 comments/day at 100 rows/page), so it is
    divided by the panel size here. This is the only reading under which
    the documented cadence reproduces (465 pages / 140 pages-per-day =
    3.3 days); treating 140 as per-subreddit would imply 2,380 pages/day
    and a 0.2-day cadence.

    Args:
        sub: Subreddit name.
        panel_size: Number of subreddits the prior is shared across.
    """
    led = _load(COST_LEDGER)
    row = (led.get("subs") or {}).get(str(sub)) or {}
    try:
        v = float(row.get("pages_per_day"))
        if math.isfinite(v) and v > 0:
            return v
    except (TypeError, ValueError):
        pass
    n = max(1, int(panel_size))
    return float(config.COMMENT_PAGES_PER_DAY_PRIOR) / n


def panel_pages_per_day(subs) -> float:
    """Returns what the whole panel costs in pages per calendar day."""
    subs = list(subs or [])
    if not subs:
        return float(config.COMMENT_PAGES_PER_DAY_PRIOR)
    n = len(subs)
    return sum(sub_pages_per_day(s, panel_size=n) for s in subs)


def derived_cadence_days(subs, skip_prices: bool = False) -> float:
    """Returns the run cadence (days) at which nothing is ever deferred:
    allowance pages / panel pages-per-day.

    This is the number in the run summary, and it is entirely measured: a
    consequence of this machine's speed and this panel's volume.
    """
    per_day = panel_pages_per_day(subs)
    if per_day <= 0:
        return float(config.COMMENT_CADENCE_DAYS)
    return allowance_pages(skip_prices=skip_prices) / per_day


def live_lookback_days() -> int:
    """Returns the comment fetcher's live window in days::

        ceil(COMMENT_CADENCE_DAYS) + LATE_ARRIVAL_DAYS

    The late-arrival day is one day of deliberate overlap for comments
    posted just behind the watermark; the seen-file dedups the re-read,
    so the overlap costs pages but can never double-count a comment.
    """
    return int(math.ceil(float(config.COMMENT_CADENCE_DAYS))
               + int(config.LATE_ARRIVAL_DAYS))


# ---------------------------------------------------------------------------
# 4. The allocation: who gets the pages when there are not enough.
# ---------------------------------------------------------------------------
def allocate(owed_days: dict, allowance: int, panel_size: int = None) -> dict:
    """Splits the page allowance across subreddits.

    Proportional to what each subreddit owes (its watermark gap x its own
    measured pages/day), so a stale subreddit is not starved by a fresh
    one and a cheap subreddit does not consume the budget of an expensive
    one.

    One-page floor: the trim loop stops while any subreddit still sits at
    1, so a tight allowance spreads thin instead of dropping communities
    entirely. A subreddit that is never crawled reads downstream as a
    community with no influential users, which is a false statement
    rather than a missing one.

    Surplus is deliberately not redistributed mid-run: handing an early
    subreddit's unspent pages to a later one would make coverage depend
    on crawl order, the position-dependence this allocation removes. Any
    surplus goes unspent and shows up as a shorter run.

    Args:
        owed_days: {sub: days it must cover}.
        allowance: Total pages the run may spend.
        panel_size: Subreddits the prior is shared across; defaults to
            the number of keys in owed_days.

    Returns:
        {sub: pages it may spend}.
    """
    owed = {}
    subs = list(owed_days or {})
    n = panel_size or max(1, len(subs))
    for s in subs:
        try:
            d = max(0.0, float(owed_days[s]))
        except (TypeError, ValueError):
            d = 0.0
        owed[s] = d * sub_pages_per_day(s, panel_size=n)
    total = sum(owed.values())
    budget = max(0, int(allowance))
    if not subs:
        return {}
    if total <= 0:
        # Nothing owed anywhere (every watermark is current): one page
        # each, so a genuinely up-to-date panel still checks for late
        # arrivals.
        return {s: 1 for s in subs}
    if total <= budget:
        # Everybody gets what they owe, rounded up so a fractional page is
        # still fetched rather than dropped.
        return {s: max(1, int(math.ceil(owed[s]))) for s in subs}

    # Short: cut everyone by the same proportion, then repair the floor.
    scale = budget / total
    plan = {s: max(1, int(math.floor(owed[s] * scale))) for s in subs}
    # The floor can push the plan back over budget; trim the biggest
    # allocation one page at a time, never below 1, until it fits or
    # nothing is left that can be trimmed.
    while sum(plan.values()) > budget:
        big = [s for s in subs if plan[s] > 1]
        if not big:
            break                     # Every sub is at the floor.
        plan[max(big, key=lambda s: plan[s])] -= 1
    return plan
