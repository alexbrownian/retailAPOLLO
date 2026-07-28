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

### 3.1b Comments are BUDGETED, not optional (desk decision 2026-07-27)

**This reverses the 2026-07-24 decision** that took comments out of the
daily pipeline. That decision made runs fast but left the influence
board rescoring whatever comments happened to be on disk — so the
"most influential users right now" panel could be quoting a month-old
crowd. Freshness is the whole point of that panel, so comments rejoined
the live pipeline and their cost was made a budget instead of a switch.

**The problem.** The panel produces a measured ~14,000 comments/day
(derivation in §3.1b-i below), i.e. ~140 API pages/day. At the Arctic
Shift politeness contract of 1 request/second a **7-day** gap costs
~16.5 minutes of pure fetching, against a desk ceiling of ~10 minutes
for the entire refresh. Weekly runs, the full 17-subreddit panel, and a
10-minute ceiling cannot all three hold.

**What was rejected.** A stopwatch — start the crawl, kill it at N
minutes — was rejected because it makes the amount of data collected a
function of network luck on the day: no two runs are comparable and "why
did this run stop there?" has no answer. Splitting the panel across W
parallel workers was rejected because W workers each pausing a second is
an aggregate W req/s, which breaks the contract the project accepted
when it chose a free public API. Narrowing the window uniformly was
rejected because it advances watermarks past uncollected days, silently
losing data. Ranking subreddits by measured yield would require adding a
`subreddit` column to the committed influence store.

**What ships instead** (`src/pipeline_budget.py`) is arithmetic over
quantities measured on the machine that runs the pipeline:

1. A **stage ledger** (`data/reference/pipeline_stage_times.json`) times
   every non-fetch stage, so the fetch budget is the ceiling minus what
   this machine *actually* spends on analytics, folding and prices —
   not minus a number someone typed. Measured on the reference machine:
   analytics 73.3 s, fold 0.4 s, coverage 0.3 s, hydrate 0.01 s, prices
   ~60 s ⇒ residual ≈ 465 pages ≈ 7.8 min ≈ **3.3 days** of volume.
2. A **cost ledger** (`data/reference/reddit_comments_cost.json`) records
   pages/day and comments/page **per subreddit** — r/wallstreetbets in a
   mania and r/Bogleheads on a quiet Tuesday differ by two orders of
   magnitude, so one panel average would misplan both.
3. An **allocation** that spends the page allowance in proportion to what
   each subreddit owes (watermark gap × its measured pages/day), cutting
   every subreddit by the *same* proportion when the allowance falls
   short, with a floor of one page each so no community is starved off
   the board and then falsely reads as having no influential users.

Both ledgers are EWMA-smoothed with **α = 2/(N+1)**, the standard
EWMA-to-SMA span identity, where N = `PANEL_REFERRAL_WINDOW` / run
cadence = 28/3.2 ≈ 9 runs ⇒ **α = 0.2**. Reusing the project's own 28-day
measurement window (the same one E2/E3/A0 use) rather than inventing a
timescale means the cost estimate tracks regime change on the same clock
the features do. Both ledgers are gitignored: they measure one machine's
speed, so committing them would plan the laptop's run with the desktop's
numbers. Each falls back to a measured bootstrap prior in `src/config.py`
and re-measures itself in one run.

**Running short is a deferral, not data loss.** The crawl walks
newest-first and a watermark only advances over ground the run fully
covered (`if incremental and completed and newest`), so a capped
subreddit keeps its old watermark and the next run resumes exactly where
this one stopped. Every deferral is printed; the pipeline never silently
collects less than it claims.

**The one legitimate speedup** was removing dead time: the fetcher used
to `sleep(1.0)` *after* each round-trip, so its true period was RTT + 1 s
and it ran ~1.3-1.4× slower than the contract allows. A shared `Pacer`
now sleeps the **remainder** of the second — same request rate, no idle
gap.

**Consequence for the desk:** run the pipeline **about twice a week**
(every ≤3.3 days) and nothing is ever deferred. `update_data.py` prints
that cadence, computed from its own measurements, in every RUN SUMMARY.
`--skip-comments` opts out for one run; `--with-comments` is accepted and
ignored (kept so older command lines do not die on an unknown argument).
`update_comments.py` survives as the **unbudgeted** runner for backfills
and for catching up after the pipeline has been idle for weeks, and its
runtime estimate is now computed from the cost ledger rather than quoted
as a hand-written range. The dashboard's sidebar button still calls it.

#### 3.1b-i Where ~14,000 comments/day came from

Nothing was assumed. Using only the committed influence store: the
intersection of comment-sourced call `rec_id`s with `reply_edges`
`rec_id`s gives 12,010 of 417,208 comments that also produced a call, so
the **call rate among comments is 2.879%**. Two independent months of
observed comment-calls per day (2026-06: 403/day; 2026-07: 414/day)
divided by that rate give ~14,000 and ~14,400 comments/day — agreeing
within 3%. Caveat recorded for the defence: the call rate is measured on
the edge-covered subpopulation, since only 48% of comment-calls carry a
resolvable reply edge.

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

**The desk-signal study (2026-07-24).** With price permitted for a
second signal family, NB03/NB06 ran the price-assisted tests: the G2
boom gate passed (capture +62% on matched years); the combined
price+crowd ALERT bank failed its pre-stated cliff criterion against its
own candidate-day baseline; and the decisive product finding was that
the candidacy STATE (A1 2× hype AND G2 boom - existing constants only)
is itself the drop-warning: cliff-30 62% in-state vs 19% ordinary
(CI [+28pp, +50pp]). The dashboard renders this as the amber DANGER
STATE band on every price panel; the crowd-only detector remains the
thesis-headline claim.

**The adopted DESK CONFIGURATION (2026-07-24) — GET IN / GET OUT.** The
decision cycle closed with a production signal pair, selected by a rule
pre-stated in NB06 and productionised in this module's §6 (the notebook
imports the production functions, and a drift guard asserts its
recomputation equals the shipped record). GET OUT = the boom-gated END
rules with the trigger on the 7d-smoothed (ROLL) score — walk-forward
capture 24/122, FA 39 (0.195/instr-yr), AP 0.449, median warning 8d.
GET IN = the onset rules with PHASE-AWARE candidacy (a day satisfying
every END gate — A1 ∧ A2 ∧ A3-persistence, existing constants only —
is end-stage and cannot host a "start") plus the same smoothing —
adjacency 20→2, LATE 21→10, FA 169→124, at a RECORDED capture cost
29→20 (a desk decision: the thrice-stated adjacency priority overrules
the raw-capture utility rule). `rebuild_phase_files()` additionally
writes `euphoria_desk.parquet` (per-day scores, candidacy states,
get_in/get_out alerts) and `euphoria_desk_report.json` (frozen
thresholds GET IN 0.848 / GET OUT 0.630 + both walk-forward records),
honouring the research/live split: live runs score at the frozen
thresholds; `--research` (or year rollover) refreezes them. The
dashboard's EUPHORIA tabs are driven by this store — explicit GET IN /
GET OUT banners, chart labels, and a window-adaptive scorecard judged
by the same `classify_*` functions the research record uses.

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

**RUN AND CONCLUDED, 2026-07-27.** The store matured (5,071 authors with
at least one judged call), the experiment ran, and it returned a
**negative result that is now load-bearing for the design**. The model
bank grew to the thesis's eight architectures — `logit`, `mlp`,
`label_prop`, `gcn_lite`, `sage_lite`, `mixhop_lite`, `h2gcn_lite`, plus
the random floor. (GAT is deliberately NOT ported: attention has to
*learn* per-edge weights, and ~250 positives cannot support that. The
notebook records the refusal rather than shipping a layer it cannot
train.) Findings, in the order they constrain the product:

1. `random → logit` is a real gain (+0.0464 AP, CI [+0.0334, +0.0593],
   10/10 paired seeds) — behaviour alone carries signal.
2. **No graph layer earns its complexity.** mixhop +0.0011 CI
   [−0.0063, +0.0085]; sage_lite −0.0035; h2gcn −0.0078 CI [−0.0130,
   −0.0026] (significantly *worse*); gcn −0.0132; mlp −0.0326. Parsimony
   ships `logit`.
3. **Why, measured rather than asserted:** positive-class node homophily
   is **0.0948** against 0.9628 for negatives, and DICE perturbation
   *raises* AP (0.1031 → 0.2063 as 0→50% of edges are corrupted). Good
   callers do not cluster, so message-passing averages signal away. Chan
   found the same direction; our split is sharper.
4. **It does not generalise to new authors.** On a tenure/cohort split
   the shipped model sits at the random floor (lift −0.046). This is the
   finding that decides the dashboard: the influence tab ranks authors by
   their **measured** record and files the model as a research exhibit,
   because the one thing a model would be *for* — scoring a newcomer
   before they have a record — is exactly what it cannot do.

Two disciplines beyond the thesis: `mean_conf` and `stance_sd` are
refused as arithmetic factors of their own target, with the cost of that
refusal recorded (+0.0983 AP, CI [+0.0834, +0.1132]); and Bonferroni
within a round (6 candidates → conf 0.99167) turned the one nominally
significant bank change into **adopted: null**. Full record:
`notebooks/05_influence_users_model.py` and
`docs/research/nb05_influence.json`.

**The graph layer (`analytics/influence_graph.py`)** is pure
numpy/scipy — **no networkx anywhere**, deliberately, so the repo keeps
one dependency story: hand-rolled multi-level Louvain, Brandes sampled
betweenness (400 pivots), k-core decomposition, Fruchterman-Reingold
layout, DICE and degree-preserving double-edge-swap perturbation,
label-permutation significance, and the cohort/tenure split.

The same module also owns the **crowding read-out** the dashboard's
influence tab draws, and its shape is deliberate. `_weighted_calls`
attaches each author's `influence_index(board)/100` to their calls as a
continuous weight w in [0, 1] — **no tier cut anywhere**, because `tier`
is unusable as a population split (25 HIGH authors against 12,503 low;
only 234 of 27,881 live calls come from a HIGH author). From there one
private function per output serves **both** grains: `_digest_frame(c,
key)` is called with `key="ticker"` by `suggestion_digest` and with
`key="theme"` by `theme_digest`, and `_voices_frame(c, key, board)`
likewise backs `ticker_voices` / `theme_voices`. That is the invariant
worth defending — the accepted consensus formula (|Σ w·s| / Σ w·|s|,
the same arithmetic as the euphoria detector's `consensus`) and
`backing_share` exist in **exactly one place**, so the two toggle views
cannot drift apart under later edits; only the grouping key differs.

`explode_to_themes` sits between them and reuses the membership in
`src/themes.py`, so "semiconductors" means the same set of names in the
influence tab as in the euphoria Themes tab. Two consequences are
intended, not accidents. A ticker that belongs to several themes (NVDA →
semiconductors, ai, ai_megacap) **duplicates into every one of them**,
because a call on NVDA genuinely is a call on all three. And calls on
tickers in **no** theme are **dropped, not bucketed into "other"** — the
residue is 58.5% of live calls and would otherwise be the largest bar on
the chart purely by being a residue. The two views therefore have
different denominators and are not expected to agree name-for-name;
the radio's help text says so on screen.

What this layer is **not**: a signal. The pre-registered test of whether
influence convergence predicts a drawdown was run and **rejected** — on
the identical 1,552 name-days where the accepted euphoria level
separates 0.925 against 0.428, all three influence candidates read the
wrong way (0.237/0.482, 0.250/0.481, 0.282/0.477), because the panel
converges on the largest liquid names and those fall less often than the
small-cap tail. No code from the rejected half survives anywhere; the
record is `docs/PARAMETER_REGISTER.md` Class 6c and
`docs/RESEARCH_REPORT.md` §6.12. Eight tests fence the shipped half
(`tests/test_pipeline.py::TestThemeRollup`), including one that parses
the AST of every theme function and asserts the word "price" appears in
no line of code — dropping docstring nodes first, since the *prose*
legitimately says "not a forecast about the price".

### 6.8 The plain-English layer (`analytics/plain_english.py`)

Two jobs, both **display-only**, and neither of them ever touches a
stored value. `PLAIN` / `plain()` / `glossary_md()` translate the
project's internal names into the words a PM reads, which is why the
stored parquet column names never have to change — translation belongs at
the display layer, and a column name is an interface shared by
`author_scores.parquet`, `calls.parquet` and `reply_edges.parquet`.

`censor()` / `censor_series()` / `is_obscene()` mask offensive substrings
in Reddit handles with `**` (desk request 2026-07-28). The design is
two-tier and every tier assignment was decided by **counting hits over
the real corpus of 12,528 handles**, not by intuition: Tier A stems match
as substrings but only **inside a single token** (obfuscated handles run
the stem into other characters with no separator, e.g.
`fucktheredditapp15`), while Tier B words match only as a **whole token**
(crude alone, common inside innocent words). The tokeniser
`[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+` honours underscores, hyphens,
camelCase and letter/digit boundaries at once, with the acronym
alternative first so `RHfuckedup` splits as `[RH, fuckedup]`. Masking
**iterates to a fixed point** because removing one span can expose a new
whole token (`Buttslut69696969` → `Butt**69696969` → `**69696969`), and
runs of masks collapse so the output does not look like a rendering bug.

Two invariants matter architecturally. **Only the offending span is
replaced**, so authors stay distinguishable — measured, 12,528 unique
handles map to 12,528 unique censored strings, which is what makes this
safe on a plotly category axis where duplicate labels merge into one bar.
And **the store is never rewritten**: `author` is a join key, so every
call site censors at the point a handle becomes a string a human reads —
the influence map's hover and labels, the ticker-backers axis, both
warning boards, the leaderboard (censored *after* the `_push` merge), the
author-calls table, `ticker_voices`'s hover text, and the ego selector via
`format_func` so the widget still returns the true key. Measured rate 97 /
12,528 = 0.774%; full audit, known misses and the English-only limitation
in `docs/PARAMETER_REGISTER.md` Class 3b. Six tests in
`tests/test_pipeline.py::TestHandleCensoring`, one of which asserts no
mask ever reaches the parquet.

`theme_label()` is the third display-only translator, added with the
influence theme view (2026-07-28). It turns a `src/themes.py` slug into
the words on a chart — `ai_megacap` → `AI megacap`, `ev_clean_energy` →
`EV clean energy` — and follows the same rule as `PLAIN`: **the slug
stays the join key**, so nothing that groups on a theme can be broken by
a relabelling. The implementation is deliberately **mechanical rather
than a 39-entry hand-written dictionary**: underscores become spaces and
a word in `_ACRONYMS` (`ai`→`AI`, `ev`→`EV`, `saas`→`SaaS`,
`glp1`→`GLP-1`) gets its house spelling, so a theme added to
`src/themes.py` renders sensibly with **no edit here at all**. The map
holds only the words plain capitalisation would get wrong; 39 hand-typed
labels would be 39 chances to drift out of sync with `THEME_TICKERS`.
Output is **sentence case, not Title Case**, because the labels sit
inside captions and scatter hover as ordinary nouns and Title Case reads
as a proper name ("Gold Metals" looks like a company).

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

**A structural hazard: the dashboard body runs at MODULE scope.**
Streamlit executes `dashboard.py` top to bottom as a script, so a
variable assigned inside a tab is **not** local to that tab — it is a
module global, and it silently rebinds any module-level helper of the
same name. This is not hypothetical: a tab-local `_unit = "themes"`
in the influence tab rebound the module-level `_unit()` scaler and
killed `fig_influence_map` three hundred lines later with `'str' object
is not callable`, and `_dig` collided the same way with the `_dig()`
helper at line 712. The convention is therefore that **tab-local names
carry a grain-specific suffix** (`_grain`, `_digest`) rather than the
bare helper name, and the class of bug is fenced permanently by
`tests/test_pipeline.py::TestDashboardModuleHygiene`, which imports the
module and asserts `_unit`, `_dig`, `_theme` and `_thin_labels` are
still callable after the script has run. Adding a module-level helper
means adding its name to that tuple.

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

**The EUPHORIA GAUGE (added 2026-07-27) — one dial above every chart.**
`draw_chart` is called once per instrument in both the Themes and the
Singles tab, so wiring the dial there gives the desk's "for each theme /
ticker" for free. The dial sits in the left of a `[1, 2.1]` column pair;
the right column carries a one-sentence caption quoting the MEASURED risk
of the band the needle is in, plus a standing reminder that the dial is a
state and not an instruction.

Four functions, immediately before `fig_series_vs_price`:

- `gauge_zones()` — `_research("gauge_zones")`, i.e. reads
  `docs/research/gauge_zones.json`, which **notebook 06 writes**. Nothing
  in `dashboard.py` is a literal: an edge that lived in this file could
  drift away from the evidence justifying it, and "why 76?" is the first
  question a gauge invites. An empty dict means notebook 06 has not run,
  and the caller then draws **no dial** rather than a dial with invented
  bands.
- `gauge_state(level_now, in_danger, z)` — returns
  `(key, label, colour)`: `calm` / `amber` "warming" / `red` "RED ZONE" /
  `red_danger` "RED ZONE + already run up". Descriptive labels only; a
  unit test asserts "get in" and "get out" can never appear in one.
- `fig_euphoria_gauge(level_now, level_prev, in_danger, z, as_of)` — a
  plotly `Indicator`. Three construction details are load-bearing.
  (i) The **needle is the display curve's last value**, not a fresh
  calculation, so dial and chart cannot disagree. (ii) The band edges
  arrive from `gauge_zones()`. (iii) The **delta is inverted on purpose** —
  euphoria rising is the risk direction, so `increasing` is painted `BEAR`
  and `decreasing` `BULL`; plotly's default would paint a rise green and
  invert the meaning of the arrow. Two layout facts learned by rendering
  the PNG and looking at it: plotly draws an Indicator `title` inside the
  **same domain as the arc**, so the header is built as paper-space
  annotations in the top margin instead; and the value bar is
  `thickness=0.15`, because at 0.28 the navy sweep covered the very band
  colours it is meant to be read against. The as-of date is **on the dial**
  rather than in a caption, because the sidebar can select a historical
  window and a dial labelled "now" while showing March would be the worst
  kind of wrong.
- `gauge_caption(level_now, in_danger, z)` — one sentence, four branches,
  every percentage read out of the JSON. It closes with the base rate for
  scale, and in the red band it says plainly that the level alone is a weak
  read (~1.3x) while the level plus an already-run-up price is the strong
  one (~3.1x).

The two edges, and the fact that only ONE of them is a new number, are in
Class 1b of `docs/PARAMETER_REGISTER.md`; the derivation is §6.10 of the
research report and the code that produced it is in notebook 06. Nine tests
in `TestEuphoriaGauge` fence it.

**The INFLUENCE tab (rebuilt 2026-07-27, re-cut later the same day after
the charts were finally LOOKED at) — information only.** It opens with a
caption saying so: nothing on it feeds the euphoria level or the GET IN /
GET OUT alerts.

The tab now leads with **what the panel is pushing** rather than with a
list of usernames, and the reason is about what the tab is for: a PM does
not trade a list of accounts, they trade positioning. Sections, in order:
(1) *what the panel is pushing* — a bubble chart, x = net direction,
y = **share of the room's conviction** in per cent, area = number of
calls, colour = side; (2) *week by week* — the panel's tilt above the
per-name share history, so the tab has a time dimension at all;
(3) *who is behind one name* — the people pushing a chosen ticker, ranked
by influence 0–100 with their side beside each bar; (4) *the names* — the
board itself; (5) the influence map, either the k-core **backbone** or one
author's **ego** neighbourhood; and (6) *called the tops* and *loud but
wrong*, the two boards worth reading against the grain. A *why there is no
model on this tab* expander quotes four measured numbers straight from
`docs/research/nb05_influence.json`. The board (authors with ≥5 judged
calls, `INFL_MIN_JUDGED`) sits at §4, **under the exhibits it is the
evidence for** rather than at the top where it used to be, and shows
**influence 0–100** and each author's tickers — **not a per-author hit
rate**, for the two reasons recorded in Class 6b of the parameter
register.

**Everything on the tab is now denominated in a unit that can be spoken
aloud, and that was the substance of the re-cut**, not a restyling. Bare
sums were the concrete reason the desk said *"I still don't get it"*: a
min-max-normalised composite printed as "usefulness 0.987" reads as an
accuracy and is not one, and a Σ(influence × conviction) printed as
"3.42" cannot be compared between two windows. Both became rescalings of
themselves — **influence 0–100** and **share of the room's conviction
(%)** — which are rank-identical and therefore change no conclusion, only
whether a reader can state what they are looking at. The share replaced a
divide-by-the-median-name ratio that measured out at 141× and 171×; the
full argument, including why the euphoria detector's A1 convention is
*not* the precedent it looked like, is in Class 6b.

Two removals fall out of that. **`fig_consensus` is gone**: it plotted the
same per-ticker consensus the bubble chart carries on its x-axis with the
bubble chart's y-axis folded into bar opacity, and its own docstring said
so. Its encoding decision (bar length is direction, bar fade is weight of
evidence, because length saturates) is retained in the register as a
rejected row rather than deleted. **The HIGH-tier cut (≥0.66) is no
longer printed**, for the same reason the raw composite is not: a tier
boundary next to a normalised score invites the reader to treat it as an
accuracy.

Six engineering choices in that tab are worth knowing, each forced by
measurement rather than taste:

- **The weekly panels are computed over every recorded voice, not the
  top-N slider.** Measured, not assumed: cut the top 25 into weeks and the
  weeks hold 124, 25, 1, 23 and 9 calls — the one-call week has one name,
  which is 100% of that week by definition. The same weeks over all 340
  recorded voices hold 733, 404, 23, 98 and 155 calls across 14–171 names.
  The slider keeps the *name-level* exhibits legible; it was never meant
  to define the population whose mood is measured.
- **Thin weeks are ENCODED, not gated.** Weekly volume is wildly unequal
  under the ingestion budget (23 vs 7,260 calls across the store), and a
  23-call week can read tilt = +1.00 off four people. A minimum-calls
  cut-off would be exactly the arbitrary threshold this project refuses,
  so the tilt marker's *area* carries `n_calls` instead: no week is
  dropped, no number is invented, and a thin week looks thin.
- **The share denominator is fixed before the ticker filter.**
  `crowding_history` computes both `share` and `even` over every name in
  the period and only then applies `tickers`, so drawing five lines and
  drawing fifty give the same height for the same name. Computing them
  after the filter — the original bug — made the baseline move with how
  many lines the caller happened to ask for. Unit-tested directly.
  The bubble chart has the same property for the same reason: it receives
  the full digest plus a draw count, because when it received a
  pre-truncated frame the chart printed MSFT at 29.5% (of 30 names) while
  the KPI above it printed 26.1% (of 51) for the same name in the same
  window.

- **The map is drawn over the SCORED pool, not the raw graph.** The reply
  store holds ~107k accounts / 259k edges, of which only 5,071 have a
  usefulness colour, so the unrestricted picture took 39.5 s to lay out
  and 88.7% of its dots were colourless. Restricting to scored authors
  gives 100% colour coverage in 0.3 s, and the caption states that this
  pool is wider than the board's.
- **Betweenness is never computed here** (21 s for a number the tab does
  not use). It lives in the notebook, where a research reader can wait.
- **Labels park outside the hairball with a leader arrow.** Top authors
  by usefulness sit inside the same dense cluster — that is what a reply
  graph *is* — so names printed at their own dots overlap, and names
  merely stacked apart no longer say *which* dot they belong to. Each
  label is pushed away from the layout centroid into the empty space a
  force-directed layout always leaves at the edges, tries four parking
  spots, and is dropped (hover only) if none is free. The box geometry is
  derived from the font size, not tuned.
- **Point labels de-collide on geometry** (`dashboard.py::_thin_labels`).
  `consensus` is bounded at ±1 and hits +1.00 exactly whenever every call
  on a name was long, so names pile into one column: INTU (4.04%) and MELI
  (3.58%) are 0.46pp apart, ≈9px on a 35% axis, and a 10pt label needs 13.
  A greedy pass in descending height keeps the better-backed label and
  suppresses only text whose box would overlap in **both** axes, so GOOG
  at x=0.53 keeps its label against MELI at x=1.00. On the weekly chart
  the equivalent fix goes the other way — the end-of-line names are drawn
  as annotations and *nudged apart* rather than suppressed, because the
  legend was already dropped in favour of those labels and losing one to a
  collision would defeat the change (measured: MSTR and ADBE both finished
  the window at ~0.3% and printed as one smear). This is the only rule in
  the project that is about pixels rather than data; it is fenced in by
  five unit tests and decides where there is room for ink, never which
  names matter.

The Graph object holds a scipy sparse matrix and so is un-picklable:
it is cached with `@st.cache_resource` while the frames around it use
`@st.cache_data` keyed on (path, mtime), the same convention as the rest
of the app.

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

Two later classes fence the influence layer on the same principle.
`TestThemeRollup` runs a hand-computable three-row fixture through the
theme grain and asserts the properties that must hold for any data: a
ticker in several themes counts in every one, unmapped tickers are
dropped rather than bucketed, consensus stays bounded and keeps the
ticker view's sign, backing shares sum to 100, an empty window returns
the correctly *typed* empty frame, hover handles are censored like
everywhere else, and no theme function so much as mentions a price in
code. `TestDashboardModuleHygiene` imports `dashboard` and asserts its
module-level helpers survive the script run (see §8).

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
