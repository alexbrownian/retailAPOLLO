"""How much comment data ONE live run is allowed to buy.

WHY THIS MODULE EXISTS
----------------------
The 17-subreddit panel produces a measured ~14,000 comments/day, i.e. about
140 API pages/day.  At the Arctic Shift politeness contract of 1 request per
second, a 7-day gap costs ~16.5 minutes of pure fetching - against the
stated ceiling of "~10 minutes" for the ENTIRE refresh.  Weekly runs, the full
panel and a 10-minute ceiling cannot all three hold, and the project does not
resolve that with a number somebody types into a constant.

WHAT WAS REJECTED, AND WHY (kept here so the choice stays auditable)
--------------------------------------------------------------------
* A STOPWATCH - start the crawl, kill it at N minutes.  Rejected: it makes the
  amount of data collected a function of network luck on the day.  No two runs
  are comparable and "why did this run stop there?" has no answer.
* W PARALLEL WORKERS, each pausing a second.  Rejected: that is an aggregate
  W req/s and breaks the contract the project accepted when it chose a free
  public API.  The request rate is not a knob.
* NARROWING THE WINDOW UNIFORMLY.  Rejected: it advances watermarks past
  uncollected days, which is silent data loss - the worst failure mode
  available, because nothing in the output looks wrong.
* RANKING SUBREDDITS BY MEASURED YIELD.  Rejected for now: it would require
  adding a `subreddit` column to the committed influence store.

WHAT SHIPS INSTEAD
------------------
Arithmetic over quantities MEASURED on the machine that runs the pipeline:

1. A STAGE LEDGER (`data/reference/pipeline_stage_times.json`) times every
   non-fetch stage, so the fetch budget is the ceiling minus what THIS machine
   actually spends on analytics, folding and prices - not minus a guess.
2. A COST LEDGER (`data/reference/reddit_comments_cost.json`) records pages
   per day and comments per page PER SUBREDDIT.  r/wallstreetbets in a mania
   and r/Bogleheads on a quiet Tuesday differ by two orders of magnitude, so
   one panel average would misplan both.
3. An ALLOCATION that spends the page allowance in proportion to what each
   subreddit OWES (its watermark gap x its own measured pages/day), cutting
   every subreddit by the SAME proportion when the allowance falls short, with
   a floor of one page each so no community is starved off the board and then
   falsely reads as "no influential users here".

Both ledgers are EWMA-smoothed with alpha = 2/(N+1) - the standard
EWMA-to-SMA span identity - where N = PANEL_REFERRAL_WINDOW / run cadence =
28 / 3.2 ~= 9 runs, hence alpha = 0.2.  Reusing the project's OWN 28-day
measurement window (the one E2/E3/A0 already use) rather than inventing a
timescale means the cost estimate tracks regime change on the same clock the
features do.  It is not a tuned number.

Both ledgers are machine-local and gitignored: they measure one machine's
speed, so committing them would plan the laptop's run with the desktop's
numbers.  Each falls back to a bootstrap prior in `src/config.py` and
re-measures itself within one run.

RUNNING SHORT IS A DEFERRAL, NOT DATA LOSS
------------------------------------------
The crawl walks newest-first and a watermark only advances over ground the run
FULLY covered (`if incremental and completed and newest`).  A subreddit that
hits its cap keeps its old watermark, so the next run resumes exactly where
this one stopped.  Every deferral is printed.  The pipeline never silently
collects less than it claims.

NOTHING HERE TOUCHES A SIGNAL.  These numbers decide how much data a run
FETCHES, never how anything is scored.

Evidence: ARCHITECTURE.md 3.1b and 3.1b-i, docs/RESEARCH_RECORD.md Class 7.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

from . import config

# --- the two machine-local ledgers ---------------------------------------
STAGE_LEDGER = os.path.join(config.REFERENCE_DIR, "pipeline_stage_times.json")
COST_LEDGER = os.path.join(config.REFERENCE_DIR, "reddit_comments_cost.json")

# Stages that are NOT subtracted from the ceiling.  "fetch" is the thing being
# budgeted - subtracting it would make the allowance shrink every time the
# crawl used it, a feedback loop that converges on zero.  It is still RECORDED
# (it is useful evidence), just never spent twice.
_NOT_A_COST = ("fetch",)


# ---------------------------------------------------------------------------
# small IO helpers - a ledger is an OPTIMISATION, never a blocker
# ---------------------------------------------------------------------------
def _load(path: str) -> dict:
    """Read a ledger, tolerating every way it can be unusable.

    A ledger is an accelerator, not a source of truth: if it is missing,
    half-written by a killed run, or hand-edited into invalid JSON, the right
    behaviour is to fall back to the bootstrap prior and re-measure - NOT to
    take the pipeline down over a cache file."""
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: str, obj: dict) -> None:
    """Write a ledger ATOMICALLY (temp file in the same directory, then
    os.replace).  A run killed mid-write must not be able to leave a truncated
    JSON file behind: _load would then silently discard a real measurement
    history and quietly re-plan from the prior."""
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
    """One EWMA step.  The FIRST observation is taken whole rather than blended
    with the prior: the prior is a stand-in for a measurement, and once a real
    measurement of THIS machine exists there is no reason to keep 80% of a
    number that describes a different machine."""
    a = config.COMMENT_EWMA_ALPHA if alpha is None else alpha
    if old is None:
        return float(new)
    return float(a) * float(new) + (1.0 - float(a)) * float(old)


def fmt_minutes(seconds: float) -> str:
    """Seconds -> a human string.  Sub-minute costs are printed in seconds
    because '0.0 min' reads as 'free' when it is not."""
    s = max(0.0, float(seconds))
    if s < 60:
        return f"{s:.0f}s"
    return f"{s / 60.0:.1f} min"


# ---------------------------------------------------------------------------
# 1. THE STAGE LEDGER - what this machine spends on everything except fetching
# ---------------------------------------------------------------------------
def record_stage(stage: str, seconds: float) -> None:
    """Fold one stage's wall clock into the ledger.

    Called by update_data.py for every timed step.  ONLY successful runs are
    recorded upstream - a stage that crashed after two seconds is not evidence
    that it takes two seconds.

    Raises OSError if the ledger cannot be written; callers treat that as
    non-fatal (a cost ledger is an optimisation, never a blocker)."""
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
    """{stage: smoothed seconds} as measured on THIS machine (may be {})."""
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
    """(seconds, measured) - what one run costs BEFORE any comment fetching.

    `measured` is False while the machine is still running on the bootstrap
    prior, and the caller prints which one it used.  That distinction matters:
    a budget derived from a prior is a plan, a budget derived from the ledger
    is an observation, and the run summary should not present one as the
    other.

    A stage the ledger has never seen falls back to its prior individually
    rather than dropping to zero - a machine that has run three stages should
    not be told the fourth is free."""
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
# 2. THE ALLOWANCE - the ceiling minus everything else, converted to pages
# ---------------------------------------------------------------------------
def allowance_pages(skip_prices: bool = False) -> int:
    """How many comment pages this run may spend.

        (PIPELINE_BUDGET_S - measured non-fetch seconds) x COMMENT_RATE_PER_S

    Floored at zero and at a single page: a run that can afford nothing should
    still make one request per subreddit's worth of progress rather than
    silently becoming --skip-comments, because a run that fetches nothing while
    claiming to fetch is the failure this whole module exists to prevent."""
    other_s, _ = non_fetch_seconds(skip_prices=skip_prices)
    left = float(config.PIPELINE_BUDGET_S) - float(other_s)
    return max(1, int(left * float(config.COMMENT_RATE_PER_S)))


# ---------------------------------------------------------------------------
# 3. THE COST LEDGER - what each subreddit actually costs, per subreddit
# ---------------------------------------------------------------------------
def record_pages(sub: str, pages: int, comments: int, days: float) -> None:
    """Fold one subreddit's observed cost into the ledger.

    `days` is the span the crawl actually covered, so pages/day is comparable
    across a 1-day incremental run and a 30-day backfill.  A run that covered
    no measurable span teaches nothing and is dropped rather than divided by
    zero."""
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
    """This subreddit's measured pages/day, or its share of the panel prior.

    THE PRIOR IS A PANEL TOTAL, NOT A PER-SUBREDDIT NUMBER.
    COMMENT_PAGES_PER_DAY_PRIOR = 140 is the measured cost of the WHOLE
    17-subreddit panel (~14,000 comments/day at 100 rows/page).  It is divided
    by the panel size here.  This is the only reading under which the
    documented cadence reproduces: 465 pages / 140 pages-per-day = 3.3 days.
    Treating 140 as per-subreddit would imply 2,380 pages/day and a 0.2-day
    cadence, i.e. running the pipeline five times a day, which is not what the
    project accepted."""
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
    """What the whole panel costs per day of calendar it must cover."""
    subs = list(subs or [])
    if not subs:
        return float(config.COMMENT_PAGES_PER_DAY_PRIOR)
    n = len(subs)
    return sum(sub_pages_per_day(s, panel_size=n) for s in subs)


def derived_cadence_days(subs, skip_prices: bool = False) -> float:
    """Run cadence at which nothing is ever deferred.

    allowance pages / panel pages-per-day.  This is the number in the RUN
    SUMMARY, and it is entirely measured: it is what turns "run about twice a
    week" into a consequence of this machine's speed and this panel's volume
    rather than a preference someone expressed."""
    per_day = panel_pages_per_day(subs)
    if per_day <= 0:
        return float(config.COMMENT_CADENCE_DAYS)
    return allowance_pages(skip_prices=skip_prices) / per_day


def live_lookback_days() -> int:
    """The comment fetcher's live window, DERIVED rather than typed:

        ceil(cadence) + LATE_ARRIVAL_DAYS

    so the window moves when the real run cadence moves.  The late-arrival
    day is one day of deliberate overlap for comments posted just behind the
    watermark; the seen-file dedups the re-read, so the overlap costs pages
    but can never double-count a comment."""
    return int(math.ceil(float(config.COMMENT_CADENCE_DAYS))
               + int(config.LATE_ARRIVAL_DAYS))


# ---------------------------------------------------------------------------
# 4. THE ALLOCATION - who gets the pages when there are not enough
# ---------------------------------------------------------------------------
def allocate(owed_days: dict, allowance: int, panel_size: int = None) -> dict:
    """{sub: days it must cover} -> {sub: pages it may spend}.

    PROPORTIONAL TO WHAT EACH SUBREDDIT OWES (its watermark gap x its own
    measured pages/day), so a stale subreddit is not starved by a fresh one and
    a cheap subreddit does not consume the budget of an expensive one.

    ONE-PAGE FLOOR.  The trim loop stops while any subreddit still sits at 1,
    so a tight allowance spreads thin instead of dropping communities
    entirely.  This is anti-starvation, not politeness: a subreddit that is
    never crawled reads downstream as a community with no influential users,
    which is a false statement rather than a missing one.

    SURPLUS IS DELIBERATELY NOT REDISTRIBUTED MID-RUN.  Handing an early
    subreddit's unspent pages to a later one would make coverage depend on
    crawl ORDER - exactly the position-dependence this allocation removes.  Any
    surplus simply goes unspent and shows up as a shorter run."""
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
        # nothing owed anywhere (every watermark is current): one page each, so
        # a genuinely up-to-date panel still checks for late arrivals.
        return {s: 1 for s in subs}
    if total <= budget:
        # the happy path - everybody gets what they owe, rounded up so a
        # fractional page is still fetched rather than dropped.
        return {s: max(1, int(math.ceil(owed[s]))) for s in subs}

    # SHORT: cut everyone by the SAME proportion, then repair the floor.
    scale = budget / total
    plan = {s: max(1, int(math.floor(owed[s] * scale))) for s in subs}
    # the floor can push the plan back over budget; trim the biggest
    # allocation one page at a time, never below 1, until it fits or
    # nothing is left that can be trimmed.
    while sum(plan.values()) > budget:
        big = [s for s in subs if plan[s] > 1]
        if not big:
            break                     # every sub is at the floor - stop here
        plan[max(big, key=lambda s: plan[s])] -= 1
    return plan
