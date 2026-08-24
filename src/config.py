"""Central configuration for retailAPOLLO.

Every path and tunable constant lives in this module so that the
pipeline, the analytics package, and the dashboard are guaranteed to
read the same values. All parameters are final production values; each
was fixed by a recorded sweep or decision (see docs/DECISIONS.md and
docs/RESEARCH_RECORD.md for provenance and evidence files).

Environment-variable overrides are supported for the window and signal
knobs listed in section 4 (SIG_K, SIG_MIN_SCORE, SIG_MIN_SCORE_SELL,
SIG_COOLDOWN, PIPELINE_START_DATE, PIPELINE_END_DATE).

Sections:
    1. Paths.
    2. View window.
    3. Fetch settings.
    4. Analytics parameters.
    5. Commit safety guard.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 1. Paths. Anchored on the repository root so the project runs from any
#    clone location.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ABSTRACTED_DIR = os.path.join(ROOT, "ABSTRACTED_DATA")   # committed, text-free
DATA_DIR = os.path.join(ROOT, "data")                    # gitignored
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")      # working aggregates
PRICES_DIR = os.path.join(DATA_DIR, "prices")            # Bloomberg closes
RAW_DIR = os.path.join(DATA_DIR, "raw")                  # raw live fetches
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")      # ledgers, caches
LOG_DIR = os.path.join(ROOT, "logs")
SNAPSHOT_DIR = os.path.join(PROCESSED_DIR, "signal_snapshots")

PRICES_PATH = os.path.join(PRICES_DIR, "prices.parquet")
POSTS_PATH = os.path.join(PROCESSED_DIR, "posts.parquet")   # external machine only

# Subreddits covered by the Reddit fetchers, one name per line.
SUBREDDITS_FILE = os.path.join(ROOT, "ingestion", "finance_subreddits.txt")

# ---------------------------------------------------------------------------
# 2. View window. START/END select what the dashboard and price pull
#    cover; they never change what is stored.
#
#    END_DATE = ""            -> live mode: fetch fresh posts, view to today.
#    END_DATE = "YYYY-MM-DD"  -> backtest: frozen view, no fetching.
#
#    update_data.py --start/--end override these for a single run and
#    propagate to child processes via PIPELINE_START_DATE/PIPELINE_END_DATE.
# ---------------------------------------------------------------------------
START_DATE = os.environ.get("PIPELINE_START_DATE") or "2021-01-01"
END_DATE = os.environ.get("PIPELINE_END_DATE")
if END_DATE is None:
    END_DATE = ""            # "" = live (to the newest data)

# Number of top-mentioned tickers covered by the Bloomberg pull.
PRICE_TOP_N = 150

# Range covered by a --full aggregate rebuild (external machine only).
# Aggregates are window-independent: built once over full history, any
# view window renders without recomputation.
BUILD_START_DATE = "2017-01-01"

# ---------------------------------------------------------------------------
# 3. Fetch settings.
# ---------------------------------------------------------------------------
FETCH_LOOKBACK_DAYS = 7      # Days each live fetch reaches back. Overlap is
                             # safe: dedup is by post id, first seen wins.
FETCH_MAX_CREDITS = 90       # FetchLayer credit cap per source per run.
FETCH_TIMEOUT_S = 600        # Wall-clock budget per fetcher subprocess; a
                             # hung API cannot stall the pipeline.

# ---------------------------------------------------------------------------
# 4. Analytics parameters. Shared by analytics/ and the dashboard.
# ---------------------------------------------------------------------------
# Rolling/statistics windows. ROLL and BASELINE feed the legacy signal
# engine and display layers; the euphoria detector uses percentile ranks
# with its own gates. Sweep evidence: docs/research/config_sweep.json.
ROLL = 7             # Rolling window (days) for mention/bull-pressure sums.
BASELINE = 84        # Trailing z-score baseline (days); trailing-only, no
                     # future data enters any window.
MIN_DAYS = 28        # Warm-up days of history required before a z exists.
DERIV_SMOOTH = 5     # Moving average (days) over day-to-day change.
                     # Display only (analytics/overlays.py).
MIN_TOTAL = 30       # Mask chart days with fewer total posts than this.
                     # Display only; the detector's coverage gate is
                     # EUPHORIA_MIN_COVERAGE.

# Signal engine (5-check BUY/SELL scorer).
SIG_K = float(os.environ.get("SIG_K", 2.5))
#   Z threshold for the momentum triggers (~top 1% abnormal days).
SIG_MIN_SCORE = int(os.environ.get("SIG_MIN_SCORE", 4))        # BUY floor /5
SIG_MIN_SCORE_SELL = int(os.environ.get("SIG_MIN_SCORE_SELL", 4))  # SELL floor
SIG_COOLDOWN_DAYS = int(os.environ.get("SIG_COOLDOWN", 21))
#   Same-side suppression window after a signal: one episode, one trade.
MIN_DAILY_MENTIONS = 5    # Theme volume floor (mean mentions/day).
MIN_TICKER_MENTIONS = 10  # Ticker volume floor (single names are noisier).
SENT_CHANGE_HORIZON = 5   # Days over which sentiment change is measured.
CROWDED_ATT_Z = 1.0       # Attention z above this counts as crowd surging.
CROWDED_SENT_DROP = -0.10  # 5d sentiment change below this = mood souring.

# Conviction engine. Validated per-year against Bloomberg prices; the EWM
# baseline absorbs coverage cliffs within weeks (evidence: DECISIONS.md).
CONV_BASELINE = "ewm"     # "ewm" (production) or "rolling" (legacy window).
CONV_EWM_HALFLIFE = 42    # Baseline memory, ~rolling-84 equivalent.
CONV_EXIT_LEVEL = 0.5     # Display: reversion level marked "signal expired".
DESK_EXIT_Z = 1.0         # Open BUY below this conviction is flagged
                          # "REVERTED - consider exit".

# Euphoria detector. Rule definitions: analytics/euphoria.py docstring.
EUPHORIA_ATT_GATE = 0.90        # A2: minimum trailing attention percentile.
EUPHORIA_HYPE_MULT = 2.0        # A1: 7d mention share must be at least this
                                # multiple of its trailing 120d median.
# Episode (ground-truth) size bars. Fixed 2026-08-07 at 20/40 boom,
# 12/25 bust; sweep table and rationale: docs/DECISIONS.md.
EUPHORIA_BOOM_MIN_ETF = 0.20    # G2: minimum ETF/theme run-up above
                                # trailing low.
EUPHORIA_BOOM_MIN_SINGLE = 0.40  # Single names must boom harder to count.

# Boom windows. Two distinct jobs, deliberately not merged:
#   - Live gate (boom_state_frame): is the name in a boom right now.
#     Short window, set here.
#   - Grading (ground truth): was a peak the end of a real run-up.
#     120d, held inline beside the episode definition so a live-gate
#     edit cannot move the yardstick.
# The 54d live gate was selected by walk-forward sweep (max capture
# inside the false-alarm budget); table in docs/DECISIONS.md. Changing
# it invalidates the frozen thresholds: re-fit with
# `python -m analytics.run_analytics --what phases --research`.
EUPHORIA_BOOM_WINDOW_D = 54     # Live gate window (boom_state_frame only).
EUPHORIA_BOOM_WINDOW_MIN_D = 27  # min_periods: half the window.

# Grading windows (episode definition; scored after the fact).
EUPHORIA_BOOM_LOOKBACK_D = 120  # G2: run-up measured off the trailing
                                # 120d low. Grading only.
EUPHORIA_CRASH_WINDOW_D = 90    # G3: the bust must arrive within 90d of
                                # the peak.
EUPHORIA_PEAK_LOCAL_MAX_D = 21  # G1: a peak is the highest close within
                                # +/- this window.
EUPHORIA_PEAK_MERGE_D = 30      # Peaks closer than this merge into one
                                # episode (higher close wins).

EUPHORIA_CRASH_MIN_ETF = 0.12   # G3: minimum ETF drawdown within 90d.
EUPHORIA_CRASH_MIN_SINGLE = 0.25  # Single-name bust bar (structurally
                                  # more volatile).

# GET IN candidacy floor. The onset rule judges only days whose 7d
# mention share is at least this multiple of the trailing 120d median.
# 1.10 is the max-capture setting inside the false-alarm budget (sweep:
# docs/DECISIONS.md). Applies to desk candidacy only; the crowd-only
# onset store keeps 1.0 as a separately published detector.
EUPHORIA_ONSET_HYPE_MIN = 1.10

EUPHORIA_ALERT_SPACING_D = 63   # Minimum days between two calls of the
                                # same head on one name (one call per
                                # name per quarter).


# ---------------------------------------------------------------------------
# Inflection: reversal context marker.
# ---------------------------------------------------------------------------
# Context only, never a tradeable call: it fires at tops and bottoms with
# no direction (hit rate 9.6% vs 5.6% base). Drawn on the price panel;
# never enters the watchlist or touches GET IN / GET OUT. Evidence:
# docs/research/inflection_trigger_sweep.json.
#
# A day is an extremum when its close is the max (or min) within
# +/- WIN_D and the excess move away over the next HORIZON_D is at least
# MIN_MOVE. The label is 1 when such a day lands within LOOKAHEAD_D.
EUPHORIA_INFLECTION_ENABLED = True
EUPHORIA_INFLECTION_WIN_D = 21          # Half-window for the extremum test.
EUPHORIA_INFLECTION_MIN_MOVE = 0.08     # Excess move that qualifies a reversal.
EUPHORIA_INFLECTION_LOOKAHEAD_D = 10    # Label's forward window.
EUPHORIA_INFLECTION_HORIZON_D = 21      # Window the excess is measured over.

# Trigger operating point (sweep: docs/research/inflection_trigger_sweep
# .json). The cut is evidence-selected; re-arm and spacing follow the
# euphoria heads' convention.
EUPHORIA_INFLECTION_CUT_Q = 0.95        # Percentile of train-year scores.
EUPHORIA_INFLECTION_REARM_Q = 0.50      # Score must fall below this to re-arm.
EUPHORIA_INFLECTION_SPACING_D = 63      # One inflection call per name per quarter.

# Price-blind trigger: a second GET IN / GET OUT pair scored from crowd
# features only (no price features, no price-based phase gate). Research
# record (AUROC ~0.57 vs the production pair's ~0.73); columns *_xp.
# Computed on every run; not exposed on the dashboard.
EUPHORIA_XP_ENABLED = True

EUPHORIA_COOLDOWN_DAYS = 21     # A4: one alert per episode per name;
                                # also the IN-vs-OUT separation window.
EUPHORIA_FADE_DISCOUNT = 10     # A3: fade-flag trigger discount
                                # (level points). Legacy top-alert path
                                # only; the production detectors carry
                                # fade as a model feature instead.
EUPHORIA_FA_PENALTY = 0.5       # Superseded by the assignment below;
                                # retained for the legacy scorer.
EUPHORIA_PCT_WINDOW = 365       # "Extreme" = vs the name's own trailing year.
EUPHORIA_MIN_HISTORY = 180      # Days of history before percentiles exist.
EUPHORIA_SINGLE_TOP_N = 25      # Single names tracked by the detector,
EUPHORIA_SINGLE_WINDOW_D = 365  # ranked over the trailing year.
EUPHORIA_MIN_COVERAGE = 100     # A0: tagged posts required in the trailing
                                # 28d before euphoria is measurable.
                                # Loosening is inadmissible: below 100 the
                                # GET IN false-alarm rate breaches the
                                # 0.23/instrument-year budget.
EUPHORIA_FA_PENALTY = 1.0       # Threshold selection: a false alarm costs
                                # one full captured peak.

# False-alarm budget, frozen as a constant. 0.23 false alarms per
# instrument-year: the accepted walk-forward rate a detector may not
# exceed. Held as a constant (not re-read from run output) so the
# adoption bar cannot drift or race between parallel stages. Changing it
# is a re-validation event: re-run `--what phases --research` and compare
# the stored record.
EUPHORIA_FA_BUDGET_PER_IY = 0.23

# Production model family. "ens" (logistic regression + monotone gradient
# boosting, rank-averaged) won the model tournament on every research
# pass and is pinned as the final selection; a research pass fits and
# validates this family only. Set to None (or DESK_MODEL_FAMILY="" in
# the environment) to re-open the tournament — required if the feature
# bank, label definitions, or universe change, since the selection was
# made under the current ones. Pinning skips no validation: the family
# is still fitted walk-forward and judged against the episode ledger.
DESK_MODEL_FAMILY = os.environ.get("DESK_MODEL_FAMILY", "ens") or None

# Strict-cut operating point. The `*_strict` signal columns use the
# threshold maximising episode-level F-beta on train years; beta = 0.5
# weights precision twice as heavily as recall and is the final
# production value, validated against the false-alarm budget by
# tools/sweep_operating_point.py (the dashboard's default view reads the
# F1 columns; see dashboard.py). Changing beta is a re-validation event:
# re-run `--what phases --research` and record the change in
# docs/RESEARCH_RECORD.md.
EUPHORIA_STRICT_BETA = float(os.environ.get("EUPHORIA_STRICT_BETA", "0.5"))

# Themes outside the euphoria universe. The desk trades equities and
# retail commodities only; gold/silver, oil, and uranium remain covered
# through their theme anchors.
EUPHORIA_EXCLUDED_THEMES = {"rates_bonds", "real_estate"}

# Trade bookkeeping (dashboard + report card).
HOLD_DAYS = 20       # Suggested hold period; edge plateaus at 3-4 weeks.
CROSS_AT = 1.5       # Conviction crossing level drawn on charts.
MIN_GAP = 10         # Days between counted crossings on a chart.

# Dynamic subreddit panel. A monthly review
# (ingestion/discover_subreddits.py, run inside update_data) mines
# collected text for r/<name> referrals and may auto-add at most one
# community per review into an exploration tier; the original 17
# subreddits are the frozen core tier. Additions are logged in
# ingestion/subreddit_panel.json so any analysis can be re-cut excluding
# young additions.
PANEL_REVIEW_DAYS = 30        # Review cadence (monthly, watermarked).
PANEL_REFERRAL_WINDOW = 28    # Referral lookback; matches the project's
                              # 28d measurement window (E2/E3/A0).
PANEL_MIN_REFERRERS = EUPHORIA_MIN_COVERAGE
                              # Unique panel authors required in 28d;
                              # reuses the A0 measurability floor.
PANEL_SCREEN_FRACTION = 0.5   # Candidate's sampled ticker-mention rate
                              # must reach this fraction of the panel
                              # average, measured in the same review.
PANEL_ADD_CAP = 1             # Max auto-adds per review: one step of the
                              # share denominator per month, absorbed by
                              # the 365d percentile normalisation.

# ---------------------------------------------------------------------------
# 4b. Comment ingestion budget. Controls how much one run fetches; never
#     how anything is scored. PIPELINE_BUDGET_S is the only human-chosen
#     value; the rest are measured, derived, or API properties. The
#     arithmetic lives in src/pipeline_budget.py (evidence:
#     ARCHITECTURE 3.1b, the parameter register (docs/RESEARCH_RECORD.md §7, Class 7)).
# ---------------------------------------------------------------------------
PIPELINE_BUDGET_S = 600       # Wall-clock budget for a full refresh.
COMMENT_RATE_PER_S = 1.0      # API rate contract; not a tunable. Raising
                              # it (including via parallel workers) breaks
                              # the free-archive usage contract.
COMMENT_PAGE = 100            # API page size; not ours to choose.
COMMENT_PAGES_PER_DAY_PRIOR = 140
                              # Measured panel total (~14,000 comments/day
                              # across 17 subreddits). Used until the
                              # machine's own cost ledger replaces it.
COMMENT_CADENCE_DAYS = 3.2    # Derived: allowance / pages-per-day. Seeds
                              # the EWMA span below;
                              # pipeline_budget.derived_cadence_days()
                              # recomputes it live.
COMMENT_EWMA_RUNS = max(1, round(PANEL_REFERRAL_WINDOW / COMMENT_CADENCE_DAYS))
                              # Cost-estimate memory in runs, on the same
                              # 28d clock the features use.
COMMENT_EWMA_ALPHA = 2.0 / (COMMENT_EWMA_RUNS + 1)
                              # Standard EWMA-to-SMA span identity.
LATE_ARRIVAL_DAYS = 1         # Deliberate overlap behind the watermark;
                              # the seen-file dedups the re-read.
PIPELINE_STAGE_PRIOR_S = {    # Bootstrap costs, replaced by this
    "analytics": 90.0,        # machine's measurements after one run
    "prices": 60.0,           # (ledger:
    "fold": 0.5,              # data/reference/pipeline_stage_times.json).
    "coverage": 0.4,
    "hydrate": 0.1,
}
# Stage-time and cost ledgers are machine-local and gitignored: they
# measure one machine's speed.

# ---------------------------------------------------------------------------
# 5. Commit safety guard. Any of these column names in an ABSTRACTED_DATA
#    file means raw post text leaked into the committed data; the
#    pipeline refuses to bless the commit.
# ---------------------------------------------------------------------------
FORBIDDEN_COLS = {"title", "selftext", "author", "id", "subreddit",
                  "score", "num_comments", "body", "text", "permalink"}
MAX_ABSTRACTED_MB = 25      # Per-file size guard for committed aggregates.


def ensure_dirs() -> None:
    """Creates every gitignored runtime folder so a fresh clone runs.

    Called by update_data.py at startup.
    """
    for d in (PROCESSED_DIR, PRICES_DIR, RAW_DIR, REFERENCE_DIR,
              LOG_DIR, SNAPSHOT_DIR):
        os.makedirs(d, exist_ok=True)
