# Key parameters

Every constant that shapes a number the dashboard shows, with its value,
the test that chose it and the reason. The companion notebook
`reference/KEY_PARAMETERS.ipynb` (paired with `KEY_PARAMETERS.py`)
re-derives each table from the frozen record in
`reference/research_record/` and from the price store, so the claims
here can be checked rather than believed.

Values live in `src/config.py` (frozen constants) or
`config/settings.csv` (settings). The distinction matters: a setting can
be changed by editing a CSV; a frozen constant changes what the record
measures and must be changed together with a research pass
(`RUNBOOK.md` §6). `tests/test_production_hygiene.py` pins the
ground-truth constants so an accidental edit fails the suite.

Evidence files are named by their JSON in `reference/research_record/`.

## 1. Ground truth: what counts as an episode

An episode is defined from price alone, before any crowd data is
consulted. That order is the point: a definition that used crowd data
would make the detector's job circular.

| Parameter | Value | Evidence | Rationale |
|---|---|---|---|
| `EUPHORIA_BOOM_MIN_ETF` | 0.20 | `sweep_ground_truth` | Minimum run-up from the trailing low into a peak for a theme ETF. |
| `EUPHORIA_BOOM_MIN_SINGLE` | 0.40 | `sweep_ground_truth` | The same for a single name. |
| `EUPHORIA_CRASH_MIN_ETF` | 0.12 | `sweep_ground_truth` | Minimum drawdown after the peak for a theme ETF. |
| `EUPHORIA_CRASH_MIN_SINGLE` | 0.25 | `sweep_ground_truth` | The same for a single name. |
| `EUPHORIA_BOOM_LOOKBACK_D` | 120 | `sweep_lookback` | Window over which the run-up is measured. |
| `EUPHORIA_CRASH_WINDOW_D` | 90 | inline with the definition | The bust must arrive within 90 days of the peak or the peak is not a top. |
| `EUPHORIA_PEAK_LOCAL_MAX_D` | 21 | inline | A peak is the highest close within ±21 days. |
| `EUPHORIA_PEAK_MERGE_D` | 30 | inline | Peaks closer than 30 days merge into one episode. |

**Why single names carry a higher bar.** Over the full price history of
the tracked universe, the median trailing-120-day run-up from the window
low is about 15% for theme ETFs and about 30% for single names; the
median forward-90-day drawdown is about 6% and 14%. A single bar would
either call every ordinary single-name swing an episode or miss most ETF
manias. The boom bars sit at about 1.35× each group's own median run-up and
the bust bars at about 1.8–2× its median drawdown, so an episode is a
move a holder of that kind of instrument would notice, and the same
multiple applies to both kinds.
(The notebook recomputes these from `data/prices/prices.parquet`.)

**Why these multiples.** `sweep_ground_truth` varied all four bars
together from 0.5× to 1.4× of an earlier setting (0.25 / 0.15 / 0.50 /
0.30). Tighter bars produced too few episodes to learn from (45
detectable peaks at 1.4×); looser bars turned ordinary volatility into
episodes and pushed capture down. The adopted 0.8× setting gives 490
peaks, 214 detectable, and an episode on 52 of the 59 instruments, at
0.19 false alarms per instrument-year. The record holds 494 episodes
(median run 94 days, median boom +49%, median bust −27%;
`nb01_episode_stats`).

**Why 120 days.** `sweep_lookback` tried 54, 90, 120, 180, 250 and 365
days. At 54 nothing qualifies; above 120 the peak count keeps rising
while captures stay flat, so the extra episodes are slow drifts rather
than manias. The window must be at least as long as the typical run
(median 94 days from trough to peak) or it truncates the run-up it is
meant to measure, which rules out 90; 120 is the first grid point above
that and matches the 120-day norm `hype_ratio` uses, so a run-up and a
crowd surge are measured against the same horizon.

## 2. Candidacy: which days the detector may judge

| Parameter | Value | Evidence | Rationale |
|---|---|---|---|
| `EUPHORIA_MIN_COVERAGE` (A0) | 100 posts / 28 d | `nb06_strictness` | Below 100 tagged posts the INCREASE EXPOSURE false-alarm rate breaches the budget. Loosening is inadmissible. |
| `EUPHORIA_HYPE_MULT` (A1) | 2.0 | `nb06_strictness` | CUT EXPOSURE candidacy: 7-day mention share at least twice the name's own trailing-120-day median. |
| `EUPHORIA_ONSET_HYPE_MIN` | 1.10 | `nb06_strictness` | INCREASE EXPOSURE candidacy floor; the max-capture setting inside the false-alarm budget. |
| `EUPHORIA_ATT_GATE` (A2) | 0.90 | `nb06_strictness` | Attention must be in its own top decile. |
| `EUPHORIA_BOOM_WINDOW_D` | 54 (min 27) | `boom_min_sweep` | Live boom gate: is the name in a run-up now. Separate from the 120-day grading window so a live-gate edit cannot move the yardstick. |
| `EUPHORIA_PCT_WINDOW` / `MIN_HISTORY` | 365 / 180 d | inline | "Extreme" means extreme against the name's own trailing year; no percentile exists before 180 days of history. |
| `EUPHORIA_EXCLUDED_THEMES` | rates_bonds, real_estate | inline | Rate-driven instruments do not form retail manias; they only add false alarms. |

The near-miss census in `nb06_strictness` records why each candidate
day that cleared the raw trigger was not called: the 2× crowd gate,
then the attention gate, then the boom gate. Loosening any of them was
swept and the shipped values are on the Pareto front.

## 3. Features

Nine crowd features and two price features, all trailing windows and
all percentile-ranked against the same name's own history, so no gate
constant is embedded in a feature (`analytics/ml_detector.py`,
`ML_BANK` and `PRICE_FEATURES`; formulas in
`analytics/euphoria_phases.py` and `analytics/euphoria.py`).

| Feature | Meaning | Window |
|---|---|---|
| `e1` | attention level vs the name's own year | 365 d percentile |
| `e3` | attention change over one month | 21 d |
| `influx_speed` | attention change over two weeks | 14 d |
| `attention_accel` | this week vs this month | 7 / 28 d |
| `hype_ratio` | this week vs the name's own 120-day norm | 7 / 120 d |
| `attention_convexity` | is attention growth itself accelerating | second difference |
| `bull_level` | bullishness vs the name's own year | 365 d percentile |
| `bull_persist` | share of the last month that leaned bullish | 28 d |
| `bull_inflection` | is the mood turning: 14-day bullish share vs 14 days earlier | 14 d change |
| `price_runup` | run-up from the trailing low | 120 d |
| `price_ret21` | 21-day return | 21 d |

Per-feature separation is in `nb02_feature_stats`: individually every
crowd feature is weak (AUROC 0.51–0.55 with confidence intervals), and
Spearman correlations among the attention measures (`research_stats`)
are below 0.8 except `e1`/`e3` (0.80), which were both kept because
they answer different questions (level vs change) and a redundant
input cannot mislead a monotone model. The bank's value is in combination, which
is why the tournament judges banks and not features.

Supporting constants: `ROLL` 7, `BASELINE` 84, `MIN_DAYS` 28,
`DERIV_SMOOTH` 5, `SENT_CHANGE_HORIZON` 5, `CONV_EWM_HALFLIFE` 42
(`config_sweep` shows the scorecard is flat across `BASELINE` 42–120,
so 84 is a convention rather than a tuned value).

## 4. Model and selection

| Parameter | Value | Evidence | Rationale |
|---|---|---|---|
| `DESK_MODEL_FAMILY` | `ens` | `ml_tournament`, `euphoria_desk_report` | Rank ensemble of logistic regression and monotone gradient boosting. Won on combined test AP lift under the pre-stated rule on every research pass. |
| Selection rule | one family for both heads; combined test AP lift; ties on AUROC then fewer false alarms | `euphoria_desk_report.selection_rule` | Stated before the numbers were computed. |
| Walk-forward | fit on years < Y, score Y, for every test year | `ml_tournament.results.*.test_years` | No headline number was computed on the year it is scored against. |
| Monotone constraints | every feature non-decreasing toward its head's direction | `analytics/ml_detector.py` | A more crowded, more bullish, more extended day may never score *less* euphoric. Prevents the learner from fitting an artefact in one year that inverts a feature's sense. |

Tournament (walk-forward test years, from `ml_tournament`):

| Head | Family | Episodes caught | AUROC | AP lift | FA / instr-yr |
|---|---|---|---|---|---|
| CUT EXPOSURE | ensemble (adopted) | 42% | 0.72 | 2.5× | 0.35 |
| CUT EXPOSURE | logistic | 47% | 0.69 | 1.9× | 0.47 |
| CUT EXPOSURE | monotone GBM | 45% | 0.72 | 2.0× | 0.57 |
| CUT EXPOSURE | ensemble, crowd-only | 37% | 0.55 | 1.2× | 1.12 |
| INCREASE EXPOSURE | ensemble (adopted) | 57% | 0.75 | 2.5× | 0.24 |
| INCREASE EXPOSURE | logistic | 69% | 0.75 | 2.6× | 0.52 |
| INCREASE EXPOSURE | monotone GBM | 62% | 0.74 | 2.4× | 0.43 |
| INCREASE EXPOSURE | ensemble, crowd-only | 35% | 0.55 | 1.2× | 0.50 |

The production record (`data/processed/euphoria_desk_report.json`,
re-frozen on each research pass) is what the dashboard quotes: CUT
EXPOSURE 44% caught, median lead 14 days, AUROC 0.73, AP 2.7× base;
INCREASE EXPOSURE 57%, 19 days, 0.73, 2.5×.

**What the price features add.** The crowd-only rows above and
`nb08_price_blind` answer the question directly: without price the
ensemble's AUROC is about 0.55–0.57 on both heads. The crowd tells the
model *that* a name is crowded; price tells it *where in the run* it is.
Both are needed for a top call and the tournament is the evidence.

## 5. Operating point and alert shape

| Parameter | Value | Evidence | Rationale |
|---|---|---|---|
| `EUPHORIA_FA_BUDGET_PER_IY` | 0.23 | `nb04_final_eval`, `research_stats` | The accepted false-alarm rate: one false call per instrument every four years. Threaded through every tournament entry and shown on the methodology page. |
| `EUPHORIA_FA_PENALTY` | 1.0 | inline | A false alarm costs one captured peak in the utility used to place thresholds. |
| Production operating point | F1 (β = 1) | `operating_point_sweep` | β from 0.5 to 1.0 was swept; capture rises from 34% to 44% on CUT EXPOSURE as the false-alarm rate rises from 0.35 to 0.58. F1 is the shipped balance; the F0.5 twin is stored beside it as the strict columns. |
| `EUPHORIA_STRICT_BETA` | 0.5 | `operating_point_sweep` | The precision-weighted twin stored as `*_strict`. |
| Trigger | 7-day smoothed crossing with re-arm | `conditioning_sweep` | Among configurations that cut false alarms with capture within 5 points of the raw trigger, the smoothed crossing has the fewest flips within 30 days. Adoption rule stated in the record. |
| `EUPHORIA_COOLDOWN_DAYS` (A4) | 21 | `alert_shape_sweep` | One call per episode per name, and the minimum separation between an INCREASE and a CUT call on the same name. |
| `EUPHORIA_ALERT_SPACING_D` | 63 | `alert_shape_sweep` | One call per head per name per quarter; 21 vs 63 days swept, 63 chosen for fewer calls per alerted episode at ≥ 75% of the baseline's capture. |
| Phase gate | CUT only after a boom; INCREASE only pre-boom and not end-stage | `alert_shape_sweep`, `nb06_desk_config` | A top cannot be called on a name that has not run; a start cannot be called on one that has already topped. |

`nb06_desk_config` compares the raw and smoothed variants of each head
and records the production choice; `nb06_desk_signal` shows the
combined crowd-plus-price trigger against crowd-only, price-only,
logistic and GBM variants on the same test years, with the smoothed
combined trigger giving the best utility (captures minus false alarms).

## 6. Inflection marker

A reversal-context marker shown beside the two heads, not a third head.

| Parameter | Value | Evidence |
|---|---|---|
| `EUPHORIA_INFLECTION_WIN_D` | 21 | `inflection_trigger_sweep.label` |
| `EUPHORIA_INFLECTION_MIN_MOVE` | 0.08 excess | same |
| `EUPHORIA_INFLECTION_LOOKAHEAD_D` / `HORIZON_D` | 10 / 21 | same |
| `EUPHORIA_INFLECTION_CUT_Q` / `REARM_Q` / `SPACING_D` | 0.95 / 0.50 / 63 | `inflection_trigger_sweep.adopted` |

The adopted cut gives a 9.6% hit rate against a 5.6% base rate (lift
1.7×) over 230 flags; the configuration inherited from the two heads
(0.97 / 0.50 / 21) gave 1.5×. The by-year table in the record shows the
lift is concentrated in 2020–2021 and 2026, which is why it is a marker
and not a signal.

## 7. Display

| Parameter | Value | Evidence | Rationale |
|---|---|---|---|
| Gauge amber edge | 76 | `gauge_zones` | Lowest cut on a 2-point grid whose 95% paired cluster-bootstrap lower bound on the outcome ("a 10% fall over 7 days starts within 30 days") is positive under all five seeds. |
| Gauge red edge | 85 | `gauge_zones` | The walk-forward CUT level already frozen in the record; not a new constant. |
| Share of discussion | trailing 7 d, within kind | `dashboard.py` | Each name's share of theme or single-name mentions over `ROLL` days, ranked, so "80% of the way to a signal" is read next to "9% of discussion (#3)". |
| `HOLD_DAYS` | 20 | `nb06_signal_efficacy` | The horizon at which the CUT edge is measurable; edge plateaus at 3–4 weeks. |
| Hidden themes | `theme_etfs.csv → show_on_dashboard` | — | Display only; the detector universe is unchanged. |

## 8. Ingestion and budget

| Parameter | Value | Rationale |
|---|---|---|
| `FETCH_LOOKBACK_DAYS` | 7 | Overlap with the previous run; dedup makes overlap free. |
| `PIPELINE_BUDGET_S` | 600 | A full refresh under ten minutes at a twice-weekly cadence; `src/pipeline_budget.py` allocates it. |
| `FETCH_MAX_CREDITS` | 90 | Per-source credit cap per run on the paid API. |
| `PANEL_REVIEW_DAYS` / `PANEL_MIN_REFERRERS` / `PANEL_SCREEN_FRACTION` | 30 / 100 / 0.5 | A forum joins the panel only when the existing panel refers to it at least as often as the coverage floor and a sample of its posts mentions tickers at half the panel's rate. |
| `PRICE_TOP_N` | 150 | Single names priced beyond the tracked set, so a name that enters the universe already has history. |

## 9. Bot screen (settings)

| Setting | Value | Rationale |
|---|---|---|
| `bot_screen_threshold` | 0.6 | One strong signal (a disclosed bot, weight 1.0) or two moderate ones (near-duplicate 0.55 + deleted author 0.10; burst 0.45 + low diversity 0.35) exclude a post; a single moderate signal does not. |
| `bot_screen_duplicate_jaccard` | 0.85 | Word-3-shingle similarity at which two posts are near-copies. Texts under 8 tokens are never duplicates, so short exclamations are safe. |
| `bot_screen_burst_posts_per_day` | 12 | Twelve posts by one author in one day is beyond what a person writes by hand across the tracked forums. |

`tests/test_bot_screen.py` holds the synthetic corpus the weights were
checked against: copy-paste promotion across accounts, a burst
account, a disclosed bot and a templated account are excluded; sixty
varied human posts and twenty short exclamations are not.
`tools/bot_screen_report.py` shows the exclusion rate and examples on
real data.

## 10. Changing a parameter

1. Settings (`config/settings.csv`, section 9 and the display switches):
   edit the CSV, run `python tools/validate_config.py`, restart.
2. Frozen constants (sections 1–7): edit `src/config.py`, update the pin
   in `tests/test_production_hygiene.py`, run
   `python -m analytics.run_analytics --what phases --research`, compare
   the new `euphoria_desk_report.json` against the previous one, and
   re-run the notebook so this page's tables match the record. A change
   ships only if it beats the previous record under the selection rule
   in section 4.
