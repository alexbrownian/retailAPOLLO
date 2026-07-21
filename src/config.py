"""
config.py
=========
ONE place for every path and every tunable number in retailAPOLLO.

WHY THIS FILE EXISTS
--------------------
In the original RetailFlow1 project the knobs lived scattered across the
tops of a dozen notebooks and scripts (ROLL=7 here, BASELINE=84 there,
K=2.5 somewhere else). That worked, but it meant a change to one number
had to be hunted down in several files, and two files could silently
disagree. Here every module imports its constants from this one file, so:

  * the pipeline, the analytics and the dashboard are guaranteed to use
    the SAME parameters (a signal shown on the dashboard is computed with
    exactly the numbers the pipeline used), and
  * a re-tune is a one-line edit.

Environment-variable overrides are kept for the handful of knobs the old
project exposed that way (SIG_K, SIG_MIN_SCORE, SIG_MIN_SCORE_SELL,
SIG_COOLDOWN, PIPELINE_START_DATE, PIPELINE_END_DATE) so existing habits
and the dashboard's "rebuild this window" button keep working.

LAYOUT OF THIS FILE
    1. paths            - where every folder / file lives
    2. the window       - the one knob most runs touch
    3. fetch settings   - live-ingestion behaviour
    4. analytics knobs  - rolling windows, z baselines, signal thresholds
    5. safety settings  - the text-free commit guard
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 1. PATHS - everything is anchored on the project root (the folder that
#    contains this src/ package), so the project can be cloned anywhere.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ABSTRACTED_DIR = os.path.join(ROOT, "ABSTRACTED_DATA")   # committed to git
DATA_DIR = os.path.join(ROOT, "data")                    # gitignored
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")      # working aggregates
PRICES_DIR = os.path.join(DATA_DIR, "prices")            # Bloomberg closes
RAW_DIR = os.path.join(DATA_DIR, "raw")                  # raw live fetches
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")      # universe cache, ledgers
LOG_DIR = os.path.join(ROOT, "logs")
SNAPSHOT_DIR = os.path.join(PROCESSED_DIR, "signal_snapshots")

PRICES_PATH = os.path.join(PRICES_DIR, "prices.parquet")
POSTS_PATH = os.path.join(PROCESSED_DIR, "posts.parquet")   # external machine only

# The list of subreddits the Reddit fetchers cover (one name per line).
SUBREDDITS_FILE = os.path.join(ROOT, "ingestion", "finance_subreddits.txt")

# ---------------------------------------------------------------------------
# 2. THE WINDOW - the one knob. START/END select the VIEW (what the
#    dashboard and the price pull cover); they never change what is stored.
#
#    END_DATE = ""            -> LIVE mode: fetch fresh posts, view to today
#    END_DATE = "2021-11-01"  -> BACKTEST: frozen view, no fetching
#
#    update_data.py --start / --end override these for a single run, and
#    export them to child processes through PIPELINE_START_DATE /
#    PIPELINE_END_DATE (read back below, so every script agrees).
# ---------------------------------------------------------------------------
START_DATE = os.environ.get("PIPELINE_START_DATE") or "2021-01-01"
END_DATE = os.environ.get("PIPELINE_END_DATE")
if END_DATE is None:
    END_DATE = ""            # "" = live (to the newest data)

# How many of the top-mentioned tickers the Bloomberg pull covers.
PRICE_TOP_N = 150

# The range a --full aggregate rebuild covers (external machine only).
# The aggregates are WINDOW-INDEPENDENT: build once over full history,
# then any view window renders instantly.
BUILD_START_DATE = "2017-01-01"

# ---------------------------------------------------------------------------
# 3. FETCH SETTINGS - live ingestion behaviour
# ---------------------------------------------------------------------------
FETCH_LOOKBACK_DAYS = 7      # how far back each live fetch reaches. Ran late?
                             # Set 14-30 to fill the gap - overlap NEVER
                             # duplicates (dedup on post id, first seen wins).
FETCH_MAX_CREDITS = 90       # FetchLayer credit cap PER SOURCE per run.
FETCH_TIMEOUT_S = 600        # hard wall-clock budget per fetcher subprocess,
                             # so one hung API can never freeze the pipeline.

# ---------------------------------------------------------------------------
# 4. ANALYTICS KNOBS - shared by analytics/ and the dashboard.
#    These numbers ARE the model; change them consciously.
# ---------------------------------------------------------------------------
ROLL = 7             # rolling window (days) for mention / bull-pressure sums.
                     # One loud afternoon is not a trend; a sustained week is.
BASELINE = 84        # trailing z-score baseline (days). Every z compares
                     # today against the SAME name's PRECEDING 84 days only -
                     # no future information ever leaks into a backtest.
MIN_DAYS = 28        # warm-up: days of history required before a z exists.
DERIV_SMOOTH = 5     # moving average over the day-to-day change (kills the
                     # sawtooth while still reacting within a week).
MIN_TOTAL = 30       # mask days with fewer total posts than this - a 1-post
                     # day would otherwise read as a fake 100% mention share.

# --- signal engine (the 5-check BUY/SELL scorer) ---
SIG_K = float(os.environ.get("SIG_K", 2.5))
#   ^ z threshold for the momentum triggers. 2.5 = only the top ~1% most
#     abnormal days for that name even qualify.
SIG_MIN_SCORE = int(os.environ.get("SIG_MIN_SCORE", 4))        # BUY floor /5
SIG_MIN_SCORE_SELL = int(os.environ.get("SIG_MIN_SCORE_SELL", 4))  # SELL floor
SIG_COOLDOWN_DAYS = int(os.environ.get("SIG_COOLDOWN", 21))
#   ^ once a name signals, the SAME side is suppressed for this many days:
#     one episode, one trade (kills clusters of near-identical signals).
MIN_DAILY_MENTIONS = 5    # theme volume floor (mean mentions/day)
MIN_TICKER_MENTIONS = 10  # ticker volume floor - single names are noisier
SENT_CHANGE_HORIZON = 5   # days over which the sentiment change is measured
CROWDED_ATT_Z = 1.0       # attention z above this counts as "crowd surging"
CROWDED_SENT_DROP = -0.10  # 5d sentiment change below this = "mood souring"

# --- conviction engine (validated in the July-2026 conviction study:
#     helper lab, real Bloomberg prices, per-year cross-validation) ---
CONV_BASELINE = "ewm"     # "ewm" = EWM mean/std baseline (winner: +1.36%/trade
                          # at K=2.5 long, 63% hit, 5/6 years positive, and it
                          # absorbs coverage cliffs within weeks instead of an
                          # 84-day hangover). "rolling" = the old fixed window.
CONV_EWM_HALFLIFE = 42    # baseline memory (~rolling-84 equivalent). The edge
                          # was stable across 28/42/60 - not a knife-edge fit.
CONV_EXIT_LEVEL = 0.5     # display: after a +/-CROSS_AT crossing, a grey
                          # marker shows where z reverts inside +/-this level
                          # ("signal expired - exit"). Validated: exiting longs
                          # on reversion frees capital ~2x faster at a better
                          # %/day (0.080 vs 0.065) than holding the full 20d.
DESK_EXIT_Z = 1.0         # trade desk hint: an OPEN BUY whose theme conviction
                          # has dropped back below this is flagged "REVERTED -
                          # consider exit" instead of waiting out the 20d cap.

# --- trade bookkeeping (dashboard + report card) ---
HOLD_DAYS = 20       # every suggestion is a 20-day hold (the edge peaks and
                     # plateaus around 3-4 weeks in the horizon analysis)
CROSS_AT = 1.5       # conviction crossing level drawn/marked on charts
MIN_GAP = 10         # days between counted crossings on a chart

# ---------------------------------------------------------------------------
# 5. SAFETY - the text-free commit guard. Any of these column names in an
#    ABSTRACTED_DATA file means raw posts leaked into the committed data;
#    the pipeline refuses to bless the commit.
# ---------------------------------------------------------------------------
FORBIDDEN_COLS = {"title", "selftext", "author", "id", "subreddit",
                  "score", "num_comments", "body", "text", "permalink"}
MAX_ABSTRACTED_MB = 25      # per-file size guard for the committed aggregates


def ensure_dirs() -> None:
    """Create every runtime folder that is gitignored (so a fresh clone
    works immediately). Called by update_data.py at startup."""
    for d in (PROCESSED_DIR, PRICES_DIR, RAW_DIR, REFERENCE_DIR,
              LOG_DIR, SNAPSHOT_DIR):
        os.makedirs(d, exist_ok=True)
