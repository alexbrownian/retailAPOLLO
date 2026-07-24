# ARCHITECTURE — how retailAPOLLO works, section by section

This document explains every layer of the system in enough depth to modify
it confidently: what each component does, WHY it is built that way, and the
contracts between layers. Read it top to bottom once; after that the
module docstrings carry the detail.

---

## 0. The one-paragraph version

Live posts from Reddit / X / StockTwits are fetched in parallel, normalised
into one 9-column shape, deduplicated by post id, and folded into six small
**text-free daily aggregates** (counts + sentiment per ticker/theme). Pure
pandas analytics turn those aggregates into **conviction z-scores** and
**5-check BUY/SELL signals with reasons**, all measured against trailing
baselines so backtests equal live behaviour. A Streamlit dashboard renders
everything interactively against Bloomberg closes. No notebooks exist
anywhere in the loop.

---

## 1. Why the notebook pipeline was slow — and what replaced it

The old flow executed Jupyter notebooks through nbconvert on every run.
That cost, per run:

1. **Kernel startup × N notebooks** — each notebook boots a fresh Python
   kernel (~5–10 s) before doing any work.
2. **Chart rendering into the file** — every matplotlib figure was drawn
   and base64-embedded into the `.ipynb` JSON (the notebooks grew to
   1–4 MB each). Rendering hundreds of static charts nobody looks at
   until they open the file is where most of the wall-clock went.
3. **Re-serialisation + atomicity machinery** — the whole notebook JSON is
   rewritten on every execution, which also created the truncated-notebook
   failure mode the old pipeline needed validate/repair/atomic-swap
   machinery to survive.

The replacement: each notebook's *mathematics* became a function in
`analytics/`, operating on wide `dates × entities` matrices in single
vectorised pandas calls. Charts are drawn ONLY when a human looks at
them — interactively, by the dashboard, from the saved outputs. The result
files (`daily_*_conviction.parquet`, `trade_signals*.parquet`) keep the
exact schemas the notebooks wrote, so the two projects are comparable
file-for-file.

Where parallelism buys real time, it is used:

| place | mechanism | why it is safe |
|---|---|---|
| the three live fetchers | `ThreadPoolExecutor` over subprocesses (`ingestion/fetch_all.py`) | network-bound, disjoint output folders |
| sentiment scoring | `joblib.Parallel` across all cores (`src/sentiment.py`) | VADER is embarrassingly parallel; a permanent id→score store means each post is ever scored once |
| the `--full` aggregate rebuild | `joblib.Parallel` batch extraction (`ingestion/build_aggregates.py`) | each batch of posts is independent; one text pass feeds all aggregates |
| conviction + signal recompute | `ProcessPoolExecutor` (`analytics/run_analytics.py`) | the two stages read different inputs and write different outputs |

What is deliberately NOT parallelised: the fold/merge steps (they are
read-modify-write on shared files — parallelism there would buy races, not
speed) and the per-name decision loop (it iterates only over trigger days,
a few hundred rows).

---

## 2. The data boundary: ABSTRACTED_DATA

The single most important design rule, inherited unchanged from
RetailFlow1: **raw post text never crosses the git boundary.**

- The raw stores (`posts.parquet`, raw `.jsonl.zst` files) reveal
  everything — title, body, author, id, subreddit. They live only on the
  external machine, gitignored.
- The committed folder `ABSTRACTED_DATA/` holds six parquet files carrying
  only `(date, entity, counts, sentiment aggregates)`. No individual post
  can be reconstructed from them.

| file | columns | merge rule |
|---|---|---|
| `daily_ticker_counts` | date, ticker, mention_count | counts ADD |
| `daily_ticker_counts_by_source` | + source | counts ADD |
| `daily_ticker_sentiment` | date, ticker, n_posts, avg_sentiment, net_bullish | n-weighted RECOMBINE |
| `daily_theme_counts` | date, theme, mention_count | counts ADD |
| `daily_theme_sentiment` | date, theme, n_posts, avg_sentiment, net_bullish | n-weighted RECOMBINE |
| `daily_term_counts` | date, term, mention_count (+ `__TOTAL__` rows) | counts ADD, rolls 365d |

**The merge maths** (`src/abstracted_data.py`): counts simply add. The
sentiment merge must NOT average two averages — a row built from 100 posts
must outweigh a row built from 3 — so it rebuilds the underlying sums
(`avg * n`), adds them, and divides back out. Both operations equal what
one-shot aggregation over the union of posts would produce, so history is
never revised, only extended. The tests assert this equivalence.

**Enforcement**: every `update_data.py` run ends with a schema scan of the
committed files against a forbidden-column list (`title`, `selftext`,
`author`, `id`, …) and a per-file size guard. FAIL means do not commit.

---

## 3. Ingestion

### 3.1 The fetchers (`ingestion/fetch_*.py`)

Each fetcher is a standalone script that appends RAW records to its own
folder under `data/raw/` (immutable-raw philosophy — nothing is ever
re-fetched or rewritten, dedup happens downstream):

- **`fetch_reddit_arctic.py`** (default Reddit source) — the Arctic Shift
  public API: complete per-subreddit coverage, near-real-time, free.
  Paginates each of the 17 tracked subreddits over the lookback window,
  keeps a rolling seen-ids file so re-runs write only new posts.
  **Watermarked incremental fetch**: Arctic archives by creation time
  with complete coverage, so once a subreddit is fetched through time T,
  older posts can never appear later — a per-subreddit watermark
  (`data/reference/reddit_arctic_watermark.json`) lets every run after
  the first fetch only what is new, minus a 1-day overlap for late
  arrivals. The watermark only advances when a subreddit's pagination
  completed cleanly, so an interrupted run re-covers its window next
  time. This turns a repeat run's Reddit pass from many minutes into
  about one.
- **`fetch_x_live.py`** — FetchLayer (or the official v2 API if a bearer
  token is configured). Two passes: broad DISCOVERY queries first (top
  finance chatter with engagement floors — catches tickers on nobody's
  watchlist, because the extractor later finds every valid symbol in post
  text), then targeted cashtag chunks over the theme anchors, in both
  Top-of-week and Latest products. Backs off on 429, stops on 402.
- **`fetch_stocktwits.py`** — public symbol streams for every theme anchor
  ETF + the most retail-heavy names; no key. The author's own
  Bullish/Bearish label is preserved in the raw lines (sentiment-engine
  calibration ground truth).

`fetch_all.py` orchestrates them: a key check first (a fetcher whose
credentials are missing is never started), then all enabled fetchers run
**concurrently** with a hard per-fetcher timeout, each one's output
printed as a single block on completion. Because block-on-completion
printing leaves the terminal silent while the slow fetchers grind, a
**heartbeat** line (`still fetching: Reddit, X`) prints whenever ~30s
pass with nothing finishing — the slowness itself is deliberate
rate-limit pacing (StockTwits ~1.5 s/symbol, X 5 s/request, Arctic Shift
1 s/page), so a full pull is minutes of politeness, not seconds of work,
and the total equals the slowest fetcher rather than the sum of all
three.

### 3.1b Comments are OPTIONAL and decoupled (desk decision 2026-07-24)

Comments are the slow species (10-50x post volume at the API's polite
1s/page), so the daily pipeline now fetches WITHOUT them
(`fetch_all.py --skip-comments` is what `update_data.py` passes by
default; `--with-comments` restores the bundle). The dedicated runner is
**`update_comments.py`**: watermark-aware time estimate up front, the
comment fetch (resumable — cancelling never loses work), then the
influence-board update, in one command. The dashboard has a matching
sidebar button ("pull comments + influence board") with the same
estimate. Nothing else changed: the influence store still self-creates
on the first run, and `run_analytics`'s influence stage remains a
zero-touch no-op when no new raw comments exist.

### 3.1c The DYNAMIC subreddit panel (desk decisions 2026-07-24)

The tracked-subreddit list self-expands where the crowd points. A
monthly, watermarked review (`ingestion/discover_subreddits.py`, run
with `--if-due` inside every live `update_data.py`) mines the collected
raw text for `r/<name>` referrals written by authors in the subs already
tracked — when retail migrates (WSB → Superstonk, 2021), the migration
is visible in panel text before anywhere else. Qualification reuses
existing constants: ≥ `PANEL_MIN_REFERRERS` (=100 — the A0 coverage
floor) unique referring authors within 28d; then a same-ruler finance
screen (one sampled page of the candidate's comments must show a
ticker-mention rate ≥ half the panel's own average, measured identically
in the same run — popularity without tickers never passes; an
unmeasurable candidate is never added). At most `PANEL_ADD_CAP` (=1)
auto-add per review, into an EXPLORATION tier — the founding 17 are the
frozen CORE tier. Every add is logged in `ingestion/subreddit_panel.json`
(committed audit trail) because panel changes step the mention-share
denominator; the manifest is what lets any analysis be re-cut excluding
young additions, and the 365d percentile normalisation absorbs one step
per month gracefully. The review also maintains a LOCAL-ONLY
`daily_ticker_counts_by_subreddit.parquet` (gitignored — the committed
contract bans subreddit columns) so panel-step artifacts are measurable,
not hidden.

### 3.2 Normalisation — one shape for everything

`src/clean_data.py`, `src/reddit_live_data.py`, `src/x_data.py`,
`src/stocktwits_data.py` map every source into ONE 9-column schema:

```
id, date, author, score, subreddit, title, selftext, num_comments, source
```

Tweets put their text in `title` with `subreddit='x_twitter'`; StockTwits
messages get `subreddit='stocktwits'`. Id prefixes (`x_`, `st_`, Reddit
base36) make cross-source collisions impossible, which is what lets one
global "first seen wins" dedup rule govern every path.

### 3.3 The two append destinations

- **Internal machine** (`ingestion/append_live_abstracted.py`): aggregates
  the new posts in memory and folds text-free deltas into
  `ABSTRACTED_DATA/`. A local gitignored ledger of folded ids plus a
  frozen `LIVE_START` date guarantee each post enters exactly once and
  live days never overlap the committed historical block.
- **External machine** (`ingestion/merge_live.py`): appends the raw posts
  into `posts.parquet` — streamed row-group by row-group (the multi-GB
  store is never loaded whole), with a fast id-only pre-check so the
  no-new-posts case costs seconds, count/schema verification, and an
  atomic swap. Then `ingestion/refresh_recent_aggregates.py` recomputes
  just the last ~45 days of the aggregates from the store and splices that
  tail onto untouched history — the store is the single source of truth,
  so this is always correct however often it runs.
- **`--full`** (`ingestion/build_aggregates.py`): the from-scratch rebuild
  over `BUILD_START_DATE` → today. One parallel text pass per post feeds
  all aggregates simultaneously; sentiment comes from the permanent
  id→score store so only never-seen posts are ever scored.

---

## 4. The extraction stack (what counts as a mention)

1. **Universe** (`src/ticker_universe.py`) — Nasdaq Trader symbol files
   (cached in `data/reference/`) plus a curated delisted supplement (BBBY,
   WISH, …) so the meme casualties keep counting historically.
2. **Extractor** (`src/extract_tickers.py`) — cashtags (`$GME`) always
   count; bare ALL-CAPS words count only if 4–5 letters, in the universe,
   and not on the stop lists. Crucially the bare-word pass scans the
   ORIGINAL text: only words the poster actually typed in caps can match,
   so "edge"/"loan" in prose never count.
3. **Word-ticker screening** (`src/screen_tickers.py`) — data-driven: a
   symbol that appears mostly lowercase in the corpus (EDGE ≈ 0.02 caps
   share) is an English word in disguise and is demoted to cashtag-only.
   The shipped `data/reference/ticker_classification.csv` was measured on
   the full RetailFlow1 corpus (433 symbols demoted).
4. **Themes** (`src/themes.py`) — two signals: keyword matching over post
   text (one tokenisation pass + hash lookups) and ticker→theme rollup.
   Every theme is anchored to a firm-approved liquid instrument
   (`THEME_ETFS`, with dated-fallback chains for young ETFs); themes with
   no approved instrument are tracked but excluded from trade signals.
5. **Counting rule** — one post = at most ONE mention per entity, and raw
   `mention_count` is the only counting signal (no score weighting:
   archived scores are final scores — using them would leak the future).

## 5. Sentiment

`src/sentiment.py`: VADER + a hand-tuned WSB/finance lexicon ("moon" +2.5,
"bagholder" −2.5, …), optionally layered with FinVADER's financial
dictionaries when installed. Per (day, entity): `n_posts`,
`avg_sentiment`, and the headline `net_bullish` = (bullish − bearish
posts) / n, using VADER's conventional ±0.05 cutoffs. Text is truncated at
300 chars (sentiment saturates; the tail adds cost, not signal), scoring
runs across all cores, and a **permanent id→score store** keyed by engine
name means a post is scored exactly once, ever.

## 6. The analytics layer

### 6.1 Conviction (`analytics/conviction.py`)

`bull_pressure = n_posts × net_bullish` per day → 7-day rolling sum →
z-score against an **EWM (exponentially-weighted) trailing baseline**
(half-life 42 d, 28-day warm-up). Both baseline styles are strictly
trailing (day *t* uses only data ≤ *t*); the EWM default was chosen by a
**July-2026 study** (temporary lab harness, real Bloomberg closes,
per-year cross-validation) over the notebook-08/09 rolling-84 window and
over share-normalised variants:

- **Chart sanity**: a one-off volume shock sits in a rolling window at
  full weight for 84 days then falls off a cliff — the cause of the
  "every theme reads negative" episodes after coverage drops. Under an
  EWM baseline the shock decays smoothly (half gone in 42 d) and the
  chart re-centres itself. It is also the least noisy variant tested.
- **PnL (long)**: trading its own +2.5 up-crossings (long the anchor
  ETF, 20 d hold) earned **+1.36 %/trade, 63 % hit rate, 299 trades,
  +0.78 %/trade above the ETFs' unconditional drift, positive in 5 of 6
  years** — stable across half-life 28/42/60 and holds 10/20/30.
- **SELL finding**: every short construction tested LOSES money (plain
  down-cross −1.33 %/trade, post-peak reversal −0.67, shallow −0.30;
  0–2 of 6 years positive). Retail conviction fading is not bearish
  price information — its value on the sell side is **exit timing**:
  leaving a long when z reverts to neutral returned +0.83 %/trade in
  ~10 days held vs +1.36 % in 20 (less per trade, ~2× better per day of
  capital, 0.080 vs 0.065 %/day). Hence the grey "back to neutral" exit
  markers and the trade desk's REVERTED hint.

Knobs in `src/config.py`: `CONV_BASELINE` ("ewm"/"rolling"),
`CONV_EWM_HALFLIFE`, `CONV_EXIT_LEVEL`, `DESK_EXIT_Z`;
`compute_conviction(normalise=True)` keeps the share-of-day's-posts
inputs as a research option. Supporting series: attention z, the rolled
net-bullish share, its 5-day change, crowded-top / swarm flags.

### 6.2 Signals (`analytics/signals.py`)

**Provenance and evidence (July-2026 study).** The engine is the ORIGINAL
RetailFlow1 notebook-10 logic, ported unchanged — verified by diffing the
two projects' signal files (every tradeable signal identical; only the
anchor-less `cannabis` theme differs, from data drift). Scored on real
prices over 2021-2026 it earns **−0.93%/trade** (BUY −1.42), driven by
2022 (−11%/trade): the checks buy retail enthusiasm into bear markets.
Running the engine on the EWM z made it *worse* (−1.80%/trade), so it
keeps its original rolling-84 ingredients. The validated positive edge
lives in the simpler EWM conviction-crossing longs (§6.1); the engine's
daily snapshots build its forward out-of-sample record next to that
benchmark. The dashboard's "MODEL DECISIONS & EVIDENCE" expander shows
this audit trail to every user.


The decision engine (see the dashboard's Trade-desk expander for the
trader-facing description): momentum crossing triggers (`crosses_above` —
one surge, one trade), a hard sentiment-agreement gate, the 5-check score
with a ≥4/5 floor, a 21-day same-side cooldown, next-day action stamping,
and a `reason` string reconstructing the whole scorecard. Themes trade
their anchor ETFs; the identical engine also runs per ticker.

### 6.3 Overlays (`analytics/overlays.py`)

Share-of-chatter normalisation everywhere (raw counts are not comparable
across the archive/live eras). The functions return data for: mention
share and its first derivative vs price, the forward-move decile
staircase, the lead/lag correlation scan (does chatter LEAD price?),
direction-flip evidence (state machine with hysteresis), conviction
crossings, the trade-desk ledger, the certainty ranking and the
hold-N-days report card.

### 6.4 The Euphoria Detector (`analytics/euphoria.py`) — THE AIM

The project's headline signal since the July-2026 re-aim. Everything is
in the module docstring (rules E1–E5, alert gates A0–A4, ground truth
G1–G3, scoring); the essentials below.

**Prediction is REDDIT-ONLY (desk rule, July 2026).** Price never enters
the euphoria level or the alert — it only defines and scores the
ground-truth tops. Four percentile-ranked ingredients (attention
extremity, sustained bullishness, crowd influx, and the LPPLS-inspired
super-exponential **attention** convexity — Sornette's bubble signature
applied to the mention count instead of the chart) average into a 0–100
euphoria level; an alert needs a swollen crowd (7d mention share ≥ 2×
its own 120d median — the "something must go euphoric first" rule,
measured in the crowd), extreme attention, persistent bullishness,
sufficient coverage and a threshold crossing (lowered when the
crowd-maximal/mood-fading divergence is active). The ONLY fitted number
is the alert threshold, learned walk-forward from past years only, with
a do-no-harm default (no training evidence → most conservative trigger).
Ground-truth tops are price-defined with dual thresholds (≥15% ETF /
≥30% single-name busts after a boom of ≥25%/≥50%), and alerts are only
judged inside the **judgeable window** — where price history exists at
the alert AND for 45 days after (earlier alerts are *pending*, not
false: scoring them as FAs was a measured bug).

Current validation (real closes, threshold always from past years):
**~23% of coverage-detectable peaks captured, median lead 4 days, ~0.11
FAs/instrument-year**. The earlier price-assisted variant captured 46% —
the delta is the documented cost of the Reddit-only claim, and the
comment backfill (≈10× post volume, feeding every ingredient) is the
identified path to recover it. Extending the price pull to 2017 adds the
dense 2018–2020 archive years to training and test.

**Ablation (thesis-style, §7.2.1 of Chan 2026):** each rule is knocked
out and the FULL walk-forward re-run (table on the dashboard). Headlines:
the hype gate is the precision lever (removing it: +83 FAs), the fade
trigger is the capture lever (removing it: −0.095 of detectable), E1/E3
carry the level, and single-feature deltas understate correlated
ingredients — the same caveat the thesis flags.

**ML challenger:** a walk-forward logistic regression on the same
features and prerequisites, its probability cut-off chosen on train
years by the same utility. Adoption criterion fixed before the numbers:
win utility AND capture at least as many peaks overall AND in the most
recent year (a near-silent model can win utility by never firing).
Verdict: **rules kept** — the learned coefficients rank the same
features top, independent evidence the hand-rules are not arbitrary.

Universe: themes minus rates_bonds/real_estate, plus the top-25
most-mentioned priced single names (data-chosen, not a hand list).

### 6.5 The Influence Tracker (`analytics/influence.py`) — committed

Method ported from Chan (Oxford M.Eng, 2026): predictive ability
concentrates in identifiable users — and NOT the loud ones (the thesis's
false-positive analysis: the structurally prominent accounts had 3× the
degree and barely-above-chance accuracy, 40% vs 79% for the quiet true
positives). Implementation:

- **Calls**: every authored post/comment mentioning a ticker with
  clearly-signed sentiment (|VADER| ≥ 0.20) is a directional call.
- **Volatility-aware judging** (thesis §4.5): a call is correct when the
  20d move clears `tau = max(3%, 0.5·sigma_90d)` for THAT name — one
  fixed bar would misgrade an index ETF and a meme stock with the same
  ruler. Each judged call gets an abnormal-return z; *enhanced* correct
  needs the move ≥ 1σ abnormal.
- **Usefulness scores** (thesis §4.6): stance-weighted accuracy,
  abnormal-return-weighted accuracy (w(z)=clip(1+|z|, 0.1, 2)), and
  enhanced accuracy — each Bayesian-shrunk (α = 10/5/10; the z-weighted
  score shrinks less because a big-|z| hit is itself evidence), min-max
  normalised, combined 0.4/0.4/0.2 into the COMPOSITE; ≥ 0.66 = HIGH
  tier. Bearish calls inside euphoria peak windows that the bust
  confirmed count as "called tops".
- **The interaction graph** (thesis ch. 4–5): undirected weighted author
  graph from the reply edges (comment→comment via parent ids AND
  comment→post via link ids), bot-filtered per the thesis's cleaning
  table (edge weight cap 100, broadcast accounts >1000 comments / >100
  posts excluded, star-topology filter on graphs ≥50 nodes), then
  degree, weighted degree and **PageRank** (power iteration — the
  thesis ablation's most beneficial structural feature; raw degree was
  *harmful* there, so the board never ranks by size).
- **Loud-but-wrong flag**: top-quartile PageRank + below-median
  composite — the thesis's false-positive profile as a column.

**Storage — committed and text-free** (July 2026, reversing the earlier
local-only rule): `data/reference/influence/` now crosses git so both
machines share one leaderboard. Author names are pseudonymous public
identifiers; a hard write-time check refuses any text column (same
contract as ABSTRACTED_DATA). Only `ingest_ledger.json` (which local
raw files were parsed, at what size) stays per-machine — it is how
`update()` (called by every `run_analytics` pass) parses only NEW raw
files and extends the store incrementally on live runs.

### 6.6 The Phases Study (`analytics/euphoria_phases.py`) — onset + episodes

The July-2026 extension of the aim: detect the START of euphoria, not
only its end. The module owns (1) the **episode ground truth** — every
confirmed top (the existing G1–G3 rules, unchanged) extended backward to
its trough (the same 120d low G2 measures the boom from — no new fitted
quantity) and forward to its bust date, with the onset hit window
`[trough, min(trough+45d, peak)]` and the LATE≠FALSE bucket for
mid-rally alerts; (2) the **onset feature bank** — five crowd-only
trailing percentile features whose windows are all derived from existing
constants (`source_breadth` was evaluated and REJECTED as a
coverage-regime artifact — notebook 02); (3) the **tournament
machinery** — walk-forward scoring, constrained threshold selection
(max capture s.t. the FA budget derived from the incumbent's accepted
0.23/instr-yr), int-day-space alert judging; and (4) the **live
detector** — `rebuild_phase_files()`, run by the `phases` stage of
`run_analytics`, writing `episodes.parquet`, `euphoria_onset.parquet`
(scored through TODAY — recent alerts are pending, not clipped) and
`euphoria_onset_report.json` for the dashboard's Start/End radar.

The research record lives in `notebooks/01–04` (ground truth → feature
battery → model tournament with a pre-stated criterion → final
evaluation). The notebooks import THIS module (a drift-guard assert in
notebook 02 enforces bank equality), so the numbers in the deck and the
numbers on the dashboard can never diverge. Tournament verdict: the
rules bank beat logistic regression, GBM and an MLP under the parsimony
rule (GBM tied on AP, not outside the bootstrap CI; the MLP finished
below random — the thesis's own small-label warning reproduced).
Trading translation (onset→BUY, top→SELL) was tested under a pre-stated
criterion and REJECTED — recorded in notebook 04, same treatment as the
retired BUY/SELL engine.

### 6.7 The Influential-Users Model (`analytics/influence_ml.py`)

Chan (2026) chapter 6 ported to the live influence store: semi-
supervised node classification — can HIGH-tier authors (composite ≥
0.66, labels from `influence.py`) be identified from behaviour and
reply-graph position alone? Models: random floor, feature-only MLP,
structure-only label propagation, and `sage_lite` (one GraphSAGE
mean-aggregation layer — the honest small-data version of the thesis's
winner). Discipline kept exactly: leakage guard (labelling-pipeline
columns are never features), stratified 60/20/20, seeds 42/100/2026,
class weights, threshold = max precision s.t. recall ≥ 0.05 on
validation, AP+AUROC on test, category ablation, random/DICE graph
perturbation. Driven by `notebooks/05` as a STANDING EXPERIMENT against
the live store (which seeds on the first live pull); a pre-stated
maturity criterion (≥130 labelled positives) gates any desk use.

## 7. Prices (`pull_bloomberg_prices.py`)

blpapi HistoricalDataRequest, PX_LAST daily. The symbol universe is the
union of: top-N mentioned tickers (whole window AND last 60 days), every
theme anchor + fallbacks, every international ADR, and anything named in
the signals. **Incremental**: per symbol, only the spans missing from
`prices.parquet` are requested (append-only store, atomic swap), so
switching windows back and forth costs nothing after the first pull. The
file stays local (licensing).

## 8. The dashboard (`dashboard.py` — "RetailRadar")

One Streamlit app (branded **RetailRadar**, with an animated radar-sweep
mark; credit line: Alex Brown — GIP 2026 Project — MAARS Global Macro),
dark terminal styling, every chart Plotly (hover/zoom/pan). The pipeline
buttons carry rough time estimates in their labels (tooltips explain
where the time goes), and the progress panel shows elapsed time. Loading is cached on (path, mtime) so
interaction is instant and the cache self-invalidates when the pipeline
rewrites a file. All computation is imported from `analytics/` — the
dashboard never re-implements a formula, so a number on screen is the
number on disk.

**The background pipeline runner.** The sidebar buttons launch pipelines
as background subprocesses (stdout to a temp log file) rather than
blocking the script — the design that makes CANCEL possible, since a
synchronous streaming loop would freeze Streamlit's event handling until
the run finished. A `@st.fragment(run_every="2s")` panel re-renders just
itself every 2 s: it advances a small state machine (step N done → launch
step N+1), draws a progress bar plus a plain-English stage checklist, and
keeps the raw output in a *technical log* expander (auto-expanded on
failure). Progress is derived by scanning the log for known **marker
lines** the pipeline scripts print ("DATA COVERAGE", "pulling Bloomberg
prices", …): each pipeline type declares its ordered stage plan
(`PLANS`), the furthest matched marker is the active stage, and the bar
fraction is kept monotonic so a marker scrolling past can never move it
backwards. Cancel kills the **whole process tree** (`taskkill /T` on
Windows, `killpg` elsewhere — `update_data.py` spawns fetcher/analytics
children a plain kill would orphan) and reaps the process. One pipeline
at a time: the buttons disable while one runs.

**Chart/UX conventions worth knowing:**

- **Masked ≠ missing**: days under the `MIN_TOTAL` mention floor are
  masked (NaN) as too thin to trust. The solid trace breaks at those
  stretches; a second **dotted, dimmed trace** bridges them (linear
  interpolation inside the data span only, hover disabled) with the
  legend key *"not enough posts that day"* — so a filled-in stretch is
  visibly different from measured data, and the mask still applies.
- **Themes only**: single-ticker overlay views were removed from the
  dashboard — the desk trades themes via their anchor ETFs. The ticker
  analytics (share/derivative/lead-lag functions in
  `analytics/overlays.py`, ticker signals in `analytics/signals.py`)
  remain importable for research and windowed backtests.
- **Emerging trends** exposes the growth lookback as a slider (3–30 d):
  a short lookback is an early-warning list, a long one demands a
  sustained build-up.
- **Conviction ranking** uses an EWMA of conviction z (half-life slider,
  default 10 d) instead of a flat 30-day mean — recent days weigh most,
  so the table tracks where crowds are *now*; the latest z and the old
  flat average are shown alongside for transparency. A board of negative
  values is meaningful (crowds quieter than their own trailing normal),
  not a bug, and the tab says so in a caption.

## 9. The orchestrator (`update_data.py`)

Auto-detects the machine (posts.parquet ⇒ external), picks the path
(live fast / backtest view / `--full` rebuild), and runs: parallel fetch →
append/fold → hydrate → coverage + window checks → analytics → signal
snapshots (never revised) → price pull → publish (external) → text-free
safety check → run summary. Guards ported from RetailFlow1: the stale-
aggregate abort (tail splice would leave a hole), the cross-machine
`--full` revert guard, and per-step environment pre-flight.

## 10. Testing (`tests/test_pipeline.py`)

Invariant-based (they never rot as data grows): merge maths ≡ one-shot
aggregation; trailing z sees no future; warm-up yields no z; one surge =
one crossing; the sentiment gate, cooldown and crowded-top rules; thin-day
masking; signed P&L symmetry; text-free schemas both on disk and at the
aggregator's output; extraction stop-list and one-post-one-mention rules.

## 11. Extension points

- **Better sentiment**: swap a transformer into
  `src/sentiment.py::score_text()` — the id→score store is keyed by engine
  name, so it re-scores automatically and nothing downstream changes.
- **New source**: write a fetcher (raw append) + a normaliser to the
  9-column shape with its own id prefix; add it to `fetch_all.fetch_plan()`.
- **New theme**: add keywords + tickers + an anchor in `src/themes.py`,
  then rerun `--full` on the external machine (keyword themes need raw
  text to backfill; the live fold picks it up from day one either way).
- **The AI Pulse tab** is a placeholder spec for an LLM layer that reads
  the freshly fetched posts during the live fold and writes market-pulse
  sections; only the finished text would be stored, consistent with the
  text-free boundary.
