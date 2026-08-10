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
# THE FIVE INHERITED KNOBS. Carried over from RetailFlow1, and until
# 2026-08-04 they carried a sentence of reasoning and no measurement -
# which the desk called out ("we need to make everything have reason and
# testing"). Each was then put through `tools/sweep_config.py`, which
# re-runs the FULL walk-forward with one number changed. The results are
# quoted below and stored in docs/research/config_sweep.json.
#
# The headline finding was not what anyone expected: FOUR OF THE FIVE DO
# NOT REACH THE GET IN / GET OUT FLAGS AT ALL. They are display and
# legacy-engine numbers. That is why they were never swept, and saying so
# here is a better answer than inventing a justification for them.
ROLL = 7             # rolling window (days) for mention / bull-pressure sums.
                     # One loud afternoon is not a trend; a sustained week is.
                     # SWEPT 2026-08-04 (3/5/7/10/14): GET OUT utility
                     # 5/3/4/4/5 on a constant 79 detectable peaks - the
                     # detector is FLAT across the whole range. 7 is not the
                     # argmax; the point is that no value is, so nothing here
                     # is a knife-edge fit on 7. GET IN mildly prefers longer
                     # (-94/-96/-78/-72/-71), which is a reason to revisit
                     # this under a proper nested walk-forward, not a reason
                     # to move it on a full-record sweep.
BASELINE = 84        # trailing z-score baseline (days). Every z compares
                     # today against the SAME name's PRECEDING 84 days only -
                     # no future information ever leaks into a backtest.
                     # SWEPT 2026-08-04 (42..180): IDENTICAL scorecard at
                     # every value, because it does not reach the desk
                     # detector. It feeds `trailing_z`, used by the legacy
                     # 5-check signals engine and by the NON-DEFAULT
                     # conviction branch (CONV_BASELINE is "ewm"). Live
                     # euphoria features are percentile ranks, not z-scores.
MIN_DAYS = 28        # warm-up: days of history required before a z exists.
                     # SWEPT 2026-08-04 (14..56): identical scorecard, same
                     # reason as BASELINE above - same code path.
DERIV_SMOOTH = 5     # moving average over the day-to-day change (kills the
                     # sawtooth while still reacting within a week).
                     # DISPLAY ONLY - used exclusively by analytics/overlays.py
                     # for the chart derivative. Touches no signal, so there
                     # is nothing for a walk-forward to say about it.
MIN_TOTAL = 30       # mask days with fewer total posts than this - a 1-post
                     # day would otherwise read as a fake 100% mention share.
                     # DISPLAY ONLY (overlays.py chart masking + a dashboard
                     # caption). SWEPT 2026-08-04 (10..100) to confirm:
                     # identical scorecard at every value. The euphoria
                     # detector has its own coverage gate for this job -
                     # EUPHORIA_MIN_COVERAGE, which IS a live signal number.

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

# --- EUPHORIA DETECTOR (the project's aim since the July-2026 re-aim:
#     detect retail euphoria -> call price TOPS). Full rule definitions
#     + research grounding: analytics/euphoria.py docstring. ---
EUPHORIA_ATT_GATE = 0.90        # A2: attention must be >= this trailing
                                # percentile - you cannot be euphoric quietly.
                                # SWEPT 2026-08-04 (0.80..0.98) at a constant
                                # 79 detectable: GET OUT utility 4/6/4/5/9,
                                # and 0.98 is strictly better on the raw rule
                                # (10 captures at ONE false alarm, vs 9 at 5)
                                # and on GET IN (-71 vs -78). It also HALVES
                                # the warning: median lead 7 days vs 16. The
                                # utility rule cannot see that cost, a desk
                                # can, and a top called 7 days out is a
                                # different product. CANDIDATE - not adopted
                                # on a full-record sweep; see nb04.
EUPHORIA_HYPE_MULT = 2.0        # A1 (Reddit-only): the 7d mention share must
                                # be >= this multiple of its own trailing 120d
                                # median - the crowd must have genuinely
                                # SWOLLEN before an alert is even possible
                                # ("something has to go euphoric first",
                                # measured in the crowd, never the chart)
                                # SWEPT 2026-08-04 (1.5/1.75/2.0/2.5/3.0).
                                # The first two rows are NOT comparable - a
                                # looser gate also enlarges the detectable
                                # set (106 and 98 peaks vs 79), so it moves
                                # the exam as well as the answer. Among the
                                # three that share 79 detectable, tightening
                                # helps monotonically: utility 4 / 6 / 6 at
                                # 2.0 / 2.5 / 3.0, with false alarms 5 / 3 / 1
                                # and lead 16 / 18 / 19 days; GET IN improves
                                # too (-78 / -71 / -64). 2.5 holds captures at
                                # 9 while halving the false alarms. CANDIDATE
                                # - not adopted here, for the multiple-
                                # comparisons reason recorded at
                                # EUPHORIA_MIN_HISTORY.
# THE FOUR BUBBLE/BUST SIZES WERE SWEPT 2026-08-05 AND NOTHING MOVED.
# Eight settings from 1.4x to 0.5x, walk-forward re-run at each. Two
# results worth knowing before touching any of them: hit rate is
# maximised by the TIGHTEST bars, purely because `detectable` collapses
# (115 -> 45) while captures fall too (18 -> 10) - and that same tightest
# setting is the only one that LOSES gold and GameStop from the ground
# truth. Improving the ratio and keeping the desk's named episodes are
# incompatible requests.
# Full tables, the reasoning and the one row worth revisiting if coverage
# ever binds (0.6x): docs/PARAMETER_REGISTER.md Class 4a.
# LOWERED 2026-08-07 (desk instruction: "you can try making it lower if
# the accuracy doesnt deteriorate too much ... more false alarms is ok").
# Swept at 25/50, 20/40 and 15/30 with crash bars scaled alongside, each
# variant re-run through the FULL walk-forward with the adopted desk
# model (docs/research/ml_tournament.json, ground_truth_sweep):
#   25/50: 292 episodes | GET OUT capture 36% | GET IN 54%
#   20/40: 494 episodes | GET OUT capture 47% | GET IN 69%  <- ADOPTED
#   15/30: 779 episodes | GET OUT capture 54% but FA 0.71/iy and ~1.5
#          episodes per instrument-year - at that rate an "episode" is
#          any ordinary correction and the word euphoria stops meaning
#          anything.
# 20/40 keeps every named desk episode (gold_metals, meme_stocks,
# semiconductors, ai - all present in its theme list) and ADDS the
# mid-sized manias 25/50 was blind to (ai_megacap, cybersecurity,
# robotics_automation ...), while accuracy IMPROVES rather than
# deteriorates (AP 0.168 -> 0.201 GET OUT, 0.211 -> 0.290 GET IN).
EUPHORIA_BOOM_MIN_ETF = 0.20    # G2: an ETF/theme peak must sit >= 20%
                                # above its trailing low. Used in TWO places
                                # with DIFFERENT windows - see the two
                                # constants below and read the note there
                                # before touching either.
EUPHORIA_BOOM_MIN_SINGLE = 0.40  # single names boom harder before they count

# --- THE TWO BOOM WINDOWS. They are not the same number and must not be
#     merged (desk decision 2026-07-29).
#
# G2 asks "did a boom precede this?" in two different jobs:
#
#   1. GROUND TRUTH - `ground_truth_peaks` / `find_episodes` use a 120-day
#      prior window to decide what COUNTS as a peak and where its trough is.
#      That window is the yardstick the detector is scored against, so
#      changing it changes the exam, not the answer. It stays 120 and is
#      NOT given a constant here deliberately: it lives inline in
#      euphoria.py / euphoria_phases.py where the episode definition is, so
#      that a future edit to the LIVE gate cannot move the yardstick by
#      accident. NB01 records its known cost ("run_days <= 120 by
#      construction"; "a boom off a 120d low admits crash-rebounds").
#
#   2. THE LIVE GATE - `boom_state_frame` decides which days the desk's
#      GET OUT rule is even allowed to judge. That is a PREDICTION-TIME
#      choice and it is what this constant sets.
#
# WHY 54 AND NOT 120 (swept 2026-07-29, from the screen: "why did this
# graph fire? the price did not move enough").  The complaint was correct in
# substance and the 120-day window was the cause: over 120 days a name that
# had crashed and merely bounced could clear a 25% "boom" bar. `semiconductors`
# (SMH) fired GET OUT on 2023-01-31 and 2023-03-16 at +36.9% and +27.5% above
# its 120-day low while still 25% and 20% BELOW its Dec-2021 peak. Across the
# 95 signals live at the time, 30 fired more than 10% below their own 1-year
# high.
#
# The window was then swept through the SAME walk-forward that produces the
# frozen record (`run_tournament_entry`, per-year thresholds chosen on train
# years only, the GROUND TRUTH held fixed so only the judgeable day set
# moves), with the NB07 §A3b frontier framing. Configurations differ in how
# many years they can score, so every row below is re-judged on the years they
# all share (2021/2022/2026, 93 detectable peaks) - capture counts on their
# own year sets are NOT comparable and reading them that way is the mistake
# the first pass made.
#
#   window  captured  FA/inst-yr  precision  warning   AP lift   AUROC
#     45d      16       0.036       0.640      5 d     -0.013    0.478
#     50d      15       0.035       0.577     16 d     -0.001    0.499
#     52d      21       0.081       0.538      8 d     +0.004    0.507
#  -> 54d      22       0.092       0.550    9.5 d     +0.005    0.506
#     56d      21       0.092       0.525      8 d     +0.006    0.499
#     58d      21       0.103       0.512      8 d     +0.007    0.501
#     60d      21       0.115       0.500      8 d     +0.014    0.503
#    120d      20       0.115       0.488      6 d     +0.082    0.546
#    150d      21       0.125       0.488      8 d     +0.086    0.549
#    252d      18       0.172       0.400     18 d     +0.097    0.557
#
# CAPTURE IS FLAT AT 21-22 ACROSS 52-60 DAYS WHILE FALSE ALARMS RISE
# MONOTONICALLY, so the efficient point is at the SHORT end of that plateau.
# Below 52d capture collapses 21 -> 15: a cliff, not a gradient. 54d is the
# max-capture point inside the FA budget - the project's own pre-stated
# selection rule (`choose_threshold`: inside budget, maximise capture) lifted
# from the threshold to the window. It dominates the previous 120d on BOTH
# axes (+2 captures, -20% false alarms) and dominates 60d as well. 52d is the
# lower-FA alternative (21 captures at 0.081) if false alarms are weighted
# harder than captures.
#
# WHAT THIS DOES NOT BUY, stated because an earlier draft of this comment
# claimed the opposite. A first pass read AP lift off each window's OWN test
# years and concluded that windows below ~60d destroy score quality. On the
# shared years AUROC is approximately 0.50 for EVERY window from 40 to 100
# days - including 60d, which reads 0.503 and a lift of +0.014, not the
# +0.042 that draft quoted. Lift only becomes clearly positive at 120d+,
# which is exactly where capture and false alarms both get worse. So the
# honest claim is: THE WINDOW TRADES CAPTURE AGAINST FALSE ALARMS AND DOES
# NOT BUY DETECTOR SKILL AT ANY SHORT SETTING. The gate does most of the work
# whichever short window is chosen. That is a live limitation (RESEARCH_REPORT
# section 8), not a settled result, and it is a bigger question than this
# constant.
#
# THE COST OF GOING SHORT, also stated. The walk-forward needs >= 3 positive
# train days before a test year; at 54d the pre-2020 candidate set no longer
# clears it, so the shipped record loses 2020 and its denominator falls from
# 122 detectable peaks to 98. Fewer captures get REPORTED not because the
# detector got worse but because it is examined on less. 60d keeps the fifth
# test year and remains defensible on that ground alone.
#
# NOISE. Three shared test years and 93 peaks; 21 versus 22 captures is one
# episode. 54d also sits two steps from the 50d cliff, so a different data
# vintage could move it - 56d or 58d buy margin for ~0.01 more FA/inst-yr.
#
# CHANGING THIS INVALIDATES THE FROZEN THRESHOLDS: it changes the candidate
# set they were selected on. Re-fit deliberately with
# `python -m analytics.run_analytics --what phases --research`.
# The sweep itself lives in notebook 07 section A3b and re-runs from data.
EUPHORIA_BOOM_WINDOW_D = 54     # LIVE GATE only (boom_state_frame)
EUPHORIA_BOOM_WINDOW_MIN_D = 27  # min_periods: half the window, as before
# ---- THE GRADING WINDOWS (the second half of the ground truth) ----
#
# Added to config 2026-08-05, desk question: "the comparison to trailing
# low (how many days now and why)?" The honest answer was that nobody
# could read it off this file, because these four numbers were typed
# INSIDE `ground_truth_peaks` and never surfaced. They are the shape of a
# top - what counts as a peak, how far back the run-up is measured, how
# long the bust has to arrive in - and they belong here beside the sizes.
#
# EUPHORIA_BOOM_LOOKBACK_D = 120 is the one that was actually asked about.
# It is NOT the same window as EUPHORIA_BOOM_WINDOW_D (54) above, and the
# difference matters:
#   * 54d  is the LIVE GATE - "is this name in a boom right now", the
#          condition the detector needs before it may fire. Short on
#          purpose, so a boom that ended months ago stops qualifying.
#   * 120d is the GRADING window - "was this peak the end of a real
#          run-up", asked once, after the fact, when the record is
#          scored. Long on purpose: a top that took four months to build
#          is still a top, and a 54d lookback would grade it as noise
#          because the run-up started outside the window.
# Measured on this store, moving the grading lookback to 54d empties the
# ground truth completely - see docs/PARAMETER_REGISTER.md Class 4b.
# SWEPT 2026-08-05. Captures are FLAT from 90d to 365d (18/18/19/19/19)
# - only the denominator moves - so there is nothing to win by changing
# this, and 120d sits mid-range. THE TRAP: at 54d (i.e. "harmonised" with
# the live gate) NOT ONE peak in the store qualifies, the record silently
# becomes empty and FA/instrument-year jumps to 0.260 because every alert
# is a false alarm by default. Table: docs/PARAMETER_REGISTER.md Class 4b.
EUPHORIA_BOOM_LOOKBACK_D = 120  # G2: run-up measured off the trailing
                                # 120d low. GRADING only - see above.
EUPHORIA_CRASH_WINDOW_D = 90    # G3: the bust has to arrive within 90d
                                # of the peak. A fall that takes a year
                                # is a bear market, not a bust, and the
                                # detector was never claiming to call it.
EUPHORIA_PEAK_LOCAL_MAX_D = 21  # G1: a peak is the highest close within
                                # +/-21d. One month either side - shorter
                                # and every wiggle is a peak, longer and
                                # two distinct tops merge into one.
EUPHORIA_PEAK_MERGE_D = 30      # two peaks closer than this are ONE
                                # episode (the higher close wins), so a
                                # jagged top is not counted three times.

EUPHORIA_CRASH_MIN_ETF = 0.12   # G3: >= 12% drawdown within 90d = ETF bust
EUPHORIA_CRASH_MIN_SINGLE = 0.25  # >= 25% for single names (structurally
                                  # more volatile - the desk's dual-threshold
                                  # call, July 2026). Lowered with the boom
                                  # bars 2026-08-07 - same sweep, same
                                  # record (see the block above
                                  # EUPHORIA_BOOM_MIN_ETF).
# --- THE GET IN CANDIDACY FLOOR (desk decision 2026-07-29, swept).
#
# The onset rule may only judge a day whose 7d mention share is at least this
# multiple of its own trailing 120d median. It USED TO BE 1.0, chosen as a
# definition rather than a fit - "the crowd is above its own normal",
# multiplier one, parameter-free - and that property was worth something.
#
# It is given up deliberately, because at 1.0 GET IN had been BREACHING ITS
# OWN FALSE-ALARM BUDGET since it shipped: 0.255 per instrument-year against
# the accepted 0.23, carried in the report as a stated limitation rather than
# fixed. Swept on the NB07 A3c frontier (same walk-forward, ground truth held
# fixed, judged on the years every configuration shares):
#
#   floor   captured/125   late   FA/inst-yr   precision
#    0.90        19          8       0.336        0.142
#    1.00        15          6       0.278        0.138   <- was
#    1.10        14          2       0.207        0.173   <- is
#    1.25        10          2       0.176        0.147
#
# 1.10 is the max-capture point INSIDE the budget - the project's own
# selection rule - and it is the first setting at which GET IN meets the
# budget at all. It costs one capture and buys: budget compliance, late
# starts 6 -> 2, precision 0.138 -> 0.173.
#
# SCOPE: this is the DESK candidacy floor (`desk_candidacy`) only. The
# crowd-only onset store (`frame_live[hype_raw >= 1]`, euphoria_phases 683 /
# 705) deliberately KEEPS 1.0 - it is a separate published detector, it was
# not swept here, and its record stands on the 1.0 frame. Changing a detector
# on evidence gathered about a different one is the error this note exists to
# prevent.
#
# Re-fit required: `python -m analytics.run_analytics --what phases --research`
EUPHORIA_ONSET_HYPE_MIN = 1.10

EUPHORIA_COOLDOWN_DAYS = 21     # A4: one alert per episode per name.
                                # SWEPT 2026-07-29 and KEPT. 7d captures 25
                                # vs 21d's 22 and stays inside the budget,
                                # but it fires 78 alerts against 42 and
                                # precision falls 0.52 -> 0.32: the extra
                                # captures are more shots at the SAME peak,
                                # which the capture count cannot see and a
                                # desk certainly can. 28d is the other side
                                # (21 captures, FA 0.086 -> 0.060, precision
                                # 0.60) and remains available.
EUPHORIA_FADE_DISCOUNT = 10     # A3: the fade flag (crowd maximal, mood
                                # rolling over) lowers the trigger by this
                                # many level-points - the fade is the LAST
                                # stage, so it may fire the alert earlier.
                                # SWEPT 2026-08-04 (0/5/10/15/20): IDENTICAL
                                # scorecard at every value, INCLUDING ZERO.
                                # The discount lives in euphoria.py's
                                # detect_alerts - the legacy TOP-alert path -
                                # and the desk GET IN / GET OUT detectors do
                                # not call it: they carry `fade` as a FEATURE
                                # inside the GET OUT bank instead. So this
                                # number currently moves nothing the desk
                                # looks at. Left in place because the legacy
                                # detector still uses it and notebook 03
                                # scores it, but it must not be described as
                                # part of the live flags.
EUPHORIA_FA_PENALTY = 0.5       # walk-forward threshold selection: a false
                                # alarm costs half a captured peak
EUPHORIA_PCT_WINDOW = 365       # "extreme" = vs this name's own last year.
                                # SWEPT 2026-08-04 (180/270/365/540/730) at a
                                # constant 79 detectable: GET OUT utility
                                # 2/3/4/5/5. A plateau from 365 up; the two
                                # longer windows differ by ONE capture, which
                                # is inside the noise of a 79-event sample.
                                # Kept - and now known not to be a spike.
EUPHORIA_MIN_HISTORY = 180      # days of history before percentiles exist.
                                # SWEPT 2026-08-04 (90..365) at a constant 79
                                # detectable: utility 6/5/4/4/2, i.e. 90 beats
                                # the frozen 180 on BOTH directions (11 vs 9
                                # captures at the same 5 false alarms; GET IN
                                # -72 vs -78). A CANDIDATE, deliberately NOT
                                # adopted here: this is a full-record sweep and
                                # ten constants were swept the same day, so
                                # picking each one's argmax is a multiple-
                                # comparisons trap. Adoption requires the
                                # nested walk-forward described in nb04.
EUPHORIA_SINGLE_TOP_N = 25      # how many single names the detector tracks
EUPHORIA_SINGLE_WINDOW_D = 365  # ...ranked over the TRAILING YEAR, not over
                                # all history. FIXED 2026-08-04: the ranking
                                # was cumulative, and 2021 alone is 39% of
                                # every mention ever recorded, so the tab
                                # tracked a 2021 list (BBBY - bankrupt - SNDL,
                                # CLOV, WKHS, NOK, MVIS) while MU missed the
                                # cut by 185 posts six weeks after the memory
                                # theme fired a GET OUT. 365d matches
                                # EUPHORIA_PCT_WINDOW, the project's existing
                                # "versus its own last year" convention, and
                                # is long enough that the universe does not
                                # churn week to week.
# (EUPHORIA_MIN_NAME_POSTS retired 2026-08-04. It was a CUMULATIVE floor -
#  3,000 tagged posts (posts naming the ticker/theme) over all history -
#  used to decide which single names
#  the detector tracks, and it had the same lookback flaw as the ranking
#  beside it: a 2021 relic with 8,000 posts from five years ago always
#  cleared it, while MU, SNDK, MSTR and SMCI never could. Its job is now
#  done by EUPHORIA_MIN_COVERAGE below, applied over the trailing 28 days -
#  the detector's own measurability rule, so the tab's membership test and
#  its contents can no longer disagree. No replacement constant.)
EUPHORIA_MIN_COVERAGE = 100     # A0: tagged posts needed in the last 28d
                                # before euphoria is measurable - percentile
                                # extremes on a handful of posts are noise
                                # (kills the thin-coverage 2023-25 FA storm)
                                # SWEPT 2026-08-04, twice (50/75/100/150/250
                                # then 100/125/150/175/200). Size-normalised,
                                # which is the only fair comparison here
                                # because this gate also decides how many
                                # instrument-days are scoreable at all:
                                #   bar  GETOUT rate  FA/iy   GETIN rate  FA/iy
                                #    50      0.131    0.150     0.156     0.283
                                #    75      0.115    0.149     0.144     0.256
                                #   100      0.114    0.057     0.141     0.222
                                #   125      0.127    0.071     0.157     0.174
                                #   150      0.156    0.053     0.157     0.152
                                #   175      0.119    0.028     0.162     0.182
                                #   200      0.121    0.028     0.171     0.264
                                # TWO CONCLUSIONS. (1) 150 is a genuine local
                                # peak - best GET OUT capture rate and best
                                # GET IN false-alarm rate at once - at the
                                # cost of median lead 12d vs 16d. CANDIDATE,
                                # not adopted: one full-record sweep, a
                                # one-capture difference, and the nested
                                # walk-forward has not been run.
                                # (2) LOOSENING IS INADMISSIBLE, not merely
                                # worse: at 75 and 50 the GET IN false-alarm
                                # rate reaches 0.256 and 0.283 against the
                                # desk's stated 0.23/instrument-year budget.
                                # This is the thin-coverage FA storm the gate
                                # was created to stop, and it is why "lower
                                # the bar to track more names" is not an
                                # option - see the note in POST_INTERN_HANDOVER
                                # about where the missing names actually are.
EUPHORIA_FA_PENALTY = 1.0       # (overrides above) a false alarm costs a
                                # FULL captured peak in threshold selection

# THE FALSE-ALARM BUDGET, FROZEN 2026-08-05.
#
# 0.23 false alarms per instrument-year: the incumbent top detector's
# documented, desk-accepted walk-forward rate. The rule it enforces is
# unchanged - a new detector may not be noisier than the noise already
# accepted. What changed is that the number is now a CONSTANT instead of
# being re-read from euphoria_report.json on every run.
#
# WHY, and it is not tidiness. `run_analytics` runs its stages in
# PARALLEL. The phases stage read this budget out of
# euphoria_report.json while the euphoria stage was rewriting that same
# file, so the budget depended on which stage happened to finish first.
# On the 2026-08-05 run two consecutive `--what phases` passes over
# IDENTICAL data printed "vs budget 0.23" and then "vs budget 0.19", and
# disagreed about the result: GET OUT captured 17 then 16, adjacency 4
# then 5. A detector whose adoption bar moves between two runs of the
# same data has no record anybody can defend.
#
# There was a second problem underneath the race. Reading the budget from
# the last run's realised FA rate makes it RATCHET: a quiet run lowers
# the bar, the lower bar changes what is adopted, and the next run is
# measured against that. The bar drifts downward on its own, and nothing
# on disk says by how much. The pre-stated rule in ARCHITECTURE.md §6 is
# pre-stated only if the number stops moving.
#
# Changing this is a RE-VALIDATION EVENT: re-run
# `--what phases --research` and compare the stored record before
# trusting any threshold selected under the old, moving value.
EUPHORIA_FA_BUDGET_PER_IY = 0.23
# themes OUTSIDE the euphoria universe - the desk trades equities and
# retail commodities only (gold/silver via GLD+fallbacks, oil via XLE,
# uranium via URA all remain through their theme anchors)
EUPHORIA_EXCLUDED_THEMES = {"rates_bonds", "real_estate"}

# --- trade bookkeeping (dashboard + report card) ---
HOLD_DAYS = 20       # every suggestion is a 20-day hold (the edge peaks and
                     # plateaus around 3-4 weeks in the horizon analysis)
CROSS_AT = 1.5       # conviction crossing level drawn/marked on charts
MIN_GAP = 10         # days between counted crossings on a chart

# --- DYNAMIC SUBREDDIT PANEL (desk decisions, 2026-07-24) ---
# The tracked-subreddit list is no longer fully static: a monthly review
# (ingestion/discover_subreddits.py, run inside update_data) mines the
# collected raw text for r/<name> referrals and can auto-add ONE new
# community per review into an EXPLORATION tier. The original 17 subs
# are the frozen CORE tier. Every addition is logged in
# ingestion/subreddit_panel.json (the audit trail), because panel changes
# step the mention-share denominator - the manifest is what lets any
# analysis be re-cut excluding young additions.
PANEL_REVIEW_DAYS = 30        # review cadence (monthly, watermarked)
PANEL_REFERRAL_WINDOW = 28    # referral lookback = the project's 28d
                              # measurement window (same as E2/E3/A0)
PANEL_MIN_REFERRERS = EUPHORIA_MIN_COVERAGE
                              # a community qualifies when >= this many
                              # UNIQUE panel authors referred to it in
                              # 28d - the SAME floor (100) that makes a
                              # name measurable at all (A0), reused
PANEL_SCREEN_FRACTION = 0.5   # finance screen: the candidate's sampled
                              # ticker-mention rate must be >= this
                              # fraction of the tracked panel's own
                              # average rate, both measured identically
                              # in the same review run
PANEL_ADD_CAP = 1             # max auto-adds per review - one step of
                              # the share denominator per month, so the
                              # 365d percentile normalisation absorbs it

# ---------------------------------------------------------------------------
# 4b. COMMENT INGESTION BUDGET (desk decision 2026-07-27)
#
# NOTHING IN THIS BLOCK TOUCHES A SIGNAL. These numbers decide how much data
# one run FETCHES, never how anything is scored. They exist because "just
# fetch everything" and "a full update shouldn't take more than ~10 minutes"
# (the desk's own sentence) are incompatible at the panel's real volume, and
# the project does not resolve that with a typed-in cap.
#
# EXACTLY ONE number below was chosen by a human: PIPELINE_BUDGET_S. Every
# other number here is measured, or derived from a measurement, or is a
# property of somebody else's API that is not ours to choose. The arithmetic
# lives in src/pipeline_budget.py; the evidence is in ARCHITECTURE 3.1b /
# 3.1b-i and PARAMETER_REGISTER Class 7.
# ---------------------------------------------------------------------------
PIPELINE_BUDGET_S = 600       # DESK DECISION 2026-07-27: "~10 minutes" for a
                              # full refresh. The only human-chosen number in
                              # this block; everything else is measured
                              # against it.
COMMENT_RATE_PER_S = 1.0      # CONTRACT, NOT A KNOB. What the project
                              # committed to when it chose a free public
                              # archive API. Raising it - or splitting the
                              # panel across W workers each pausing a second,
                              # an aggregate W req/s - breaks that contract,
                              # so it is not available as a speedup. Recorded
                              # as rejected rather than left as a temptation.
COMMENT_PAGE = 100            # GROUND TRUTH: the API's own page size. Not
                              # ours to choose; it is what one request
                              # returns.
COMMENT_PAGES_PER_DAY_PRIOR = 140
                              # MEASURED, and a PANEL TOTAL - not a
                              # per-subreddit figure. Intersecting
                              # comment-call rec_ids with reply_edges gives
                              # 12,010 / 417,208 = 2.879% call rate among
                              # comments; against 403 comment-calls/day
                              # (2026-06) and 414/day (2026-07) that implies
                              # ~14,000 comments/day = ~140 pages/day across
                              # the 17-sub panel. Caveat: the rate is measured
                              # on the edge-covered subpopulation (48% of
                              # comment-calls). Used only until the machine
                              # builds its own per-subreddit cost ledger, at
                              # which point measurement replaces it.
COMMENT_CADENCE_DAYS = 3.2    # DERIVED: allowance / panel pages-per-day =
                              # how many days of volume one run can buy
                              # (465 / 140 = 3.32 on the real panel). This is
                              # what makes "about twice a week" a CONSEQUENCE
                              # rather than a preference - the desk chose
                              # 2x/week with this number in front of it.
                              # pipeline_budget.derived_cadence_days()
                              # recomputes it live; this constant only seeds
                              # the EWMA span below.
COMMENT_EWMA_RUNS = max(1, round(PANEL_REFERRAL_WINDOW / COMMENT_CADENCE_DAYS))
                              # = round(28 / 3.2) = 9 runs. Reuse the
                              # project's OWN 28-day measurement window (the
                              # one E2/E3/A0 use) rather than invent a
                              # timescale, so the cost estimate tracks regime
                              # change on the same clock the features do.
COMMENT_EWMA_ALPHA = 2.0 / (COMMENT_EWMA_RUNS + 1)
                              # = 0.2. The standard EWMA-to-SMA span identity
                              # alpha = 2/(N+1). Derived, not tuned.
LATE_ARRIVAL_DAYS = 1         # one day of deliberate overlap so comments
                              # posted just behind the watermark are not
                              # missed. The seen-file dedups the re-read, so
                              # the overlap costs pages and can never
                              # double-count a comment.
PIPELINE_STAGE_PRIOR_S = {    # BOOTSTRAP ONLY - what a run is assumed to cost
    "analytics": 90.0,        # before this machine has measured itself. The
    "prices": 60.0,           # ledger (data/reference/pipeline_stage_times
    "fold": 0.5,              # .json) replaces each entry with a measurement
    "coverage": 0.4,          # after one run: on the reference machine the
    "hydrate": 0.1,           # prior's 151s became a measured 134.1s
}                             # (analytics 73.3, prices ~60, fold 0.44,
                              # coverage 0.31, hydrate 0.012), and the page
                              # allowance self-corrected 449 -> 465. That the
                              # loop closes at all is the point of measuring.
# Both ledgers are MACHINE-LOCAL and gitignored: they measure one machine's
# speed, so committing them would plan the laptop's run with the desktop's
# numbers.

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
