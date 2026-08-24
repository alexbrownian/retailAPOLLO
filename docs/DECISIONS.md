# Decision log

Chronological record of every design decision that shaped the production
system, with the evidence file for each. This document replaces the
narrative history previously embedded in source comments and in
`DECISIONS.xlsx`; the deep parameter-by-parameter register remains
`docs/RESEARCH_RECORD.md`. All parameters referenced here are final
production values.

Conventions: each entry lists the decision, the measured basis, and
where the evidence lives. "Walk-forward" always means thresholds fitted
on strictly earlier years and scored blind on the held-out year.

---

## July 2026 — project aim and core contract

| Decision | Basis / evidence |
|---|---|
| Aim fixed: detect retail euphoria from Reddit-derived data and call price tops; success = an alert inside [peak − 30d, peak + 1d]. | Aim definition precedes all validation. `docs/RESEARCH_RECORD.md`. |
| Prediction uses crowd data only in the euphoria level; price defines and scores ground-truth tops, nothing else. The later desk configuration (Aug 2026) relaxes this for the shipped model; the crowd-only record is retained separately. | Enforced by test (`compute_euphoria` accepts no price argument). |
| Ground truth is price-defined: peak = local 21d high above size bars, bust within 90d, peaks under 30d apart merge. Dual bars for ETFs vs single names (single names are structurally more volatile). | `analytics/euphoria.py`; `docs/RESEARCH_RECORD.md` Class 4. |
| Two-machine architecture: the external machine holds raw text (`posts.parquet`); the internal machine holds only text-free aggregates. `FORBIDDEN_COLS` guard runs on every commit. | `src/abstracted_data.py`; safety check in `update_data.py`. |
| BUY/SELL legacy engine retired from the dashboard (full-history BUY = −1.42%/trade); conviction EWM engine and reversion exits validated and kept. | DECISIONS archive, sheet 9. |

## 2026-07-24 — panel, coherence, cadence

| Decision | Basis / evidence |
|---|---|
| Dynamic subreddit panel: monthly watermarked review; a candidate needs ≥ 100 unique referring authors in 28d plus a same-ruler finance screen; at most one auto-add per review into an exploration tier; core 17 frozen; every addition logged. | `ingestion/discover_subreddits.py`; `ingestion/subreddit_panel.json`. |
| Episode coherence is asymmetric: a START within 21d after an END is suppressed; an END after a START never is. The symmetric rule cost half the walk-forward captures (17 → 9). | `TestEpisodeCoherence`. |
| Research/live split: live runs score at frozen thresholds; research re-opens only on explicit `--research`, year rollover, or a missing record. | `analytics/euphoria.py::needs_research`. |

## 2026-07-27 — comment budget

| Decision | Basis / evidence |
|---|---|
| Comments fetched on every run under a measured page allowance: (600s ceiling − measured non-fetch stage seconds) × 1 req/s, split across subreddits by days owed. Hitting the cap defers (watermark does not advance); never loses data. | `src/pipeline_budget.py`; `docs/RESEARCH_RECORD.md` Class 7. |
| Rejected alternatives, with reasons: wall-clock stopwatch (non-reproducible), parallel workers (breaks the 1 req/s API contract), uniform window narrowing (starves quiet subreddits). | ARCHITECTURE 3.1b. |
| Run cadence ~2×/week; a full refresh must stay under ~10 minutes (`PIPELINE_BUDGET_S = 600`, the one human-chosen number in the block). | `src/config.py` section 4b. |

## 2026-07-28 / 07-29 — live-run discipline and gates

| Decision | Basis / evidence |
|---|---|
| A live run never re-selects: no model choice, no threshold re-fit, no stats printed. Staleness is reported, not repaired (out-of-sample use is what the walk-forward licenses). | `docs/RESEARCH_RECORD.md` Class 9. |
| Live boom gate window set to 54d (was 120d): max capture inside the false-alarm budget; the 120d window let crash-rebounds qualify as booms. Grading window stays 120d — the yardstick is deliberately separate from the gate. | Sweep table in notebook 07 §A3b; formerly inlined at `EUPHORIA_BOOM_WINDOW_D`. |
| GET IN candidacy floor raised 1.0 → 1.10: first setting that meets the false-alarm budget (0.207 vs 0.278/instr-yr), late starts 6 → 2. | NB07 A3c frontier. |
| Alert cooldown kept at 21d: 7d adds captures but they are repeat shots at the same peak (precision 0.52 → 0.32). | Cooldown sweep. |

## 2026-08-04 / 08-05 — parameter sweeps and the frozen budget

| Decision | Basis / evidence |
|---|---|
| The five inherited knobs (ROLL, BASELINE, MIN_DAYS, DERIV_SMOOTH, MIN_TOTAL) swept: four do not reach the live flags at all (display/legacy paths); the detector is flat across the tested ranges. Left at inherited values. | `docs/research/config_sweep.json`. |
| Coverage gate 100 posts/28d confirmed; loosening is inadmissible (below 100 the GET IN false-alarm rate breaches the budget). 150 is a candidate under a nested walk-forward, not adopted. | Coverage sweep, twice. |
| False-alarm budget frozen as a constant, 0.23/instrument-year. Previously re-read from run output, which both raced parallel stages and ratcheted the bar. | `EUPHORIA_FA_BUDGET_PER_IY`. |
| Single-name universe ranked over the trailing 365d, not cumulative history (the cumulative ranking tracked a 2021 relic list). | `EUPHORIA_SINGLE_WINDOW_D`. |
| Grading windows surfaced into config; moving the grading lookback to 54d empties the ground truth entirely (recorded as the trap it is). | `docs/RESEARCH_RECORD.md` Class 4b. |

## 2026-08-07 — episode bars, robust share, model adoption

| Decision | Basis / evidence |
|---|---|
| Ground-truth bars lowered 25/50 → 20/40 (busts 15/30 → 12/25): 494 episodes (was 292), capture improves (GET OUT AP 0.168 → 0.201, GET IN 0.211 → 0.290), and every named episode is retained. 15/30 rejected (an "episode" becomes any correction). | `ground_truth_sweep`; `docs/research/ml_tournament.json`. |
| Share-of-chatter estimator rebuilt (ratio-of-sums, stratified by source, shrunk): the old mean-of-daily-ratios printed fake zeros on thin days and diluted shares when a large StockTwits/X pull landed. | `analytics/robust_share.py`. |
| The desk model becomes the tournament winner: logit + monotone-GBM rank ensemble ("ens"), selected under a pre-stated criterion (combined AP lift, one family for both heads). | `docs/research/nb03_tournament.json`. |

## 2026-08-09 — shaped calls and two operating points

| Decision | Basis / evidence |
|---|---|
| Shaped triggers: GET IN only pre-boom, GET OUT only post-boom, deep re-arm, one call per name per quarter per side (spacing 63d), no GET IN within 21d of a GET OUT. | `docs/research/alert_shape_sweep.json`. |
| Two operating points computed on every run: F1 (`get_in`/`get_out`) and F0.5 precision-weighted (`*_strict`). Column names are frozen; the display mapping is documented and fenced. | `analytics/ml_detector.py`. |

## 2026-08-12 — inflection marker

| Decision | Basis / evidence |
|---|---|
| Inflection = context, never a call: fires at tops and bottoms with no direction, hit rate 9.6% vs 5.6% base. Drawn on the price panel; cannot fire, gate, or re-score a call (fenced). | `docs/research/inflection_trigger_sweep.json`; `TestInflectionMarker`. |

## 2026-08-14 — price-blind pair

| Decision | Basis / evidence |
|---|---|
| Experimental crowd-only GET IN/GET OUT pair (13 crowd features, no phase gate): AUROC ~0.57 vs the shipped ~0.73. Kept as the "what can the crowd alone see" record; computed on every run; not a desk signal. | `docs/research/nb08_price_blind.json`. |

## 2026-08-17 — signed readiness, ungated GET IN, retail-flow dial

| Decision | Basis / evidence |
|---|---|
| Signed readiness replaces the two-dial display: one number in [−1, +1] whose sign follows the phase routing, so GET IN and GET OUT can never both read 100%. Impossible by construction, fenced by test. | `docs/research/nb08_single_dial.json`; `TestSignedReadinessAndUngatedGetIn`. |
| GET IN ungated in the desk view: the 120d phase gate blocked ~3/4 of correct GET IN calls (captures 28 → 116 at under double the false alarms). GET OUT keeps its gate — ungated its false alarms double. | Notebook 08 §10.4. |
| Retail-flow dial adopted: −1..+1 crowd-flow gauge, Kalman-smoothed with the gain chosen per fold on train-year IC; measured to lead price by 1–3 weeks. Display context only; cannot touch calls (fenced). | Notebook 08 §9; `TestRetailFlowDial`. |
| Alerts and the ETF radar are themes/ETFs only; single names never alert. | Dashboard display rules. |

## 2026-08-18 / 08-21 — historical restoration

| Decision | Basis / evidence |
|---|---|
| The 2023–2025 coverage gap (drought quarters ~1,000 mention rows vs a healthy 10–16k) was closed by `tools/fold_historical.py`: a deliberate second door that aggregates historical posts into the text-free store with a per-(file, month) ledger and a hard refusal to touch days ≥ LIVE_START. `update_data --skip-fetch` cannot do this on the internal machine (the live fold drops everything before LIVE_START by design). | `docs/RESEARCH_RECORD.md`, with the measured before/after per quarter. |
| Backfill fetcher rewritten for the archive API: `limit=auto`, minimal `fields=`, header-based pacing, 4xx classification. 6.5× fewer requests, 9.2× less transfer, byte-identical post set; a full 17-month backfill ran in 19.9 minutes. Backfill ledger keyed by window **and** subreddit set. | `ingestion/fetch_reddit_arctic.py`; `tools/backfill_reddit.py`. |
| Any fold is a re-validation event: `--what phases --research` must follow, because a fold rewrites the history the thresholds were fitted on. | `docs/RESEARCH_RECORD.md`, "Two explicit ways to re-open research". |

## 2026-08-21 — model family pinned

| Decision | Basis / evidence |
|---|---|
| `DESK_MODEL_FAMILY = "ens"` pinned: the tournament selected the same family on every research pass, so re-deciding it each fold spent ~24 of 27 minutes re-deriving a settled result. Pinning skips no validation (walk-forward fit and episode judging unchanged); re-open by setting it to None if the bank, labels, or universe change. | `src/config.py`; `TestPinnedModelFamily`. |

## 2026-08-23 — final display configuration and ingestion hardening

| Decision | Basis / evidence |
|---|---|
| Operating-point beta exposed as `EUPHORIA_STRICT_BETA` (final: 0.5) and validated by `tools/sweep_operating_point.py` under the pre-stated budget: on the OUT side no beta meets the budget; on the IN side only 0.5 does. The criterion, not any single instrument, decides. | `docs/research/operating_point_sweep.json`. |
| Dashboard hardwired: shipped trigger, wider (F1) operating point, ungated GET IN, % -of-signal dial. No user-facing mode selectors. Measured cost of the wider cut vs F0.5 (walk-forward): GET OUT capture 34→45% at precision 37→32%; GET IN 39→57% at 58→46%; FA/instr-yr roughly doubles. The research pack quotes the F0.5 figures. | `dashboard.py` signal-configuration block. |
| Dedup set moved to an uncapped, atomically-written parquet (`abstracted_seen_ids.parquet`); aggregate merges are additive and carry no post ids, so this set is the only guard against permanent double counting. The 300k-entry JSON cap and its eviction risk are retired; `data/reference/` is snapshotted (last 7) after every successful fold. | `ingestion/append_live_abstracted.py`. |
| Per-file skip ledger for the live fold: a raw file whose newest post predates LIVE_START is skipped without being opened (exact by construction). Read stage measured 48× faster on a 44-file fixture, byte-identical in-window rows. | `ingestion/append_live_abstracted.py`. |
| `tools/data_health.py` added (read-only): freshness, per-source coverage, ledger integrity, backups, and a month-level double-count scan against a self-excluding neighbour median (a doubled month doubles its own median, so a centered baseline cannot see it). Reports candidates, never verdicts: real manias and source onboarding look identical to a duplicate fold. | `tools/data_health.py`. |
| Vendor co-mention feed (ICE) evaluated and not adopted: the delivery is an entity co-mention edge list with no sentiment and no post text; usable only as parallel features pending ≥ 2 years of overlapping history. Evaluation notebook retained in `New_ICE_data/`. | `New_ICE_data/explore_ice_source.py` §7–§9. |

---

## Standing invariants

These are not dated decisions but permanent properties the test suite
fences:

- Committed aggregates are text-free (`FORBIDDEN_COLS`); raw post text
  never leaves the external machine.
- No lookahead: every rolling statistic is trailing; thresholds are
  fitted on strictly earlier years; a backfill forces re-research.
- A live run never re-selects a model or threshold.
- Signed readiness cannot indicate GET IN and GET OUT simultaneously.
- The retail-flow dial and the inflection marker cannot fire, gate, or
  re-score a call.
- Alerts and the ETF radar are themes/ETFs only.
