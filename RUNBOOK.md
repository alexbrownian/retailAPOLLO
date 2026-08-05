# RUNBOOK — what to run, when

Every command runs from the project root (`retailAPOLLO/`).
Most days only one command is needed: `python update_data.py` — or none at
all: the dashboard's sidebar buttons run the same pipelines.

## Quick reference

| Task | Command |
|---|---|
| Refresh everything (LIVE, fast) | `python update_data.py` — parallel fetch of all sources, fold, recompute signals, pull prices |
| Open the terminal | `python -m streamlit run dashboard.py` |
| Re-read the AI Pulse as of an earlier day | `python -m analytics.ai_pulse --as-of 2026-06-01` — clips every store AND every raw post to that day, so the whole page is dated consistently. Writes `ai_pulse_<date>.json` and **never** touches the live `ai_pulse.json`. Needs the gateway, so desk machine only. `--dry --as-of <date>` prints just the evidence pack, no LLM call |
| **A code or config edit is not showing** | **Restart the server — the file watcher is OFF by design** (`.streamlit/config.toml`, so the pipeline rewriting parquet in place cannot trigger a mid-read reload). Stop EVERY running streamlit first, or the new one silently takes the next port (8501 → 8502) and your pinned tab keeps serving the old process. PowerShell, targeted so a running pipeline is not killed with it: `Get-NetTCPConnection -LocalPort 8501,8502,8503 -State Listen -ErrorAction SilentlyContinue \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` then `python -m streamlit run dashboard.py`. The sidebar shows the build time AND the port it is serving, and turns red if dashboard.py on disk is newer than the build this process started with. `config/theme_etfs.csv` is the exception — it re-reads on its own mtime, so a rerun is enough |
| Backtest / view a past window | set the window in `src/config.py` (or `--start/--end`), then `python update_data.py` — instant: the aggregates are window-independent |
| Build aggregates over ALL history | `python update_data.py --full` — external machine, run ONCE (and after changing BUILD_START_DATE or the theme definitions) |
| Recompute without API calls | `python update_data.py --skip-fetch` |
| Recompute ONLY the analytics | `python -m analytics.run_analytics` (seconds) |
| Windowed ticker-signal backtest | `python -m analytics.run_analytics --what signals --start 2026-01-01 --end 2026-06-01` |
| Check the FetchLayer key | `python ingestion/fetch_all.py --check` (no calls) / `--test` (1 credit) |
| See what live data landed | `python check_live_ingestion.py` |
| Pull Bloomberg prices | `python pull_bloomberg_prices.py` (`--dry-run` to preview) |
| Refresh euphoria/onset LIVE data | automatic in every pull - scores at the FROZEN walk-forward thresholds in seconds (intra-year recompute is a no-op: thresholds train on strictly earlier years) |
| Re-run the FULL euphoria/onset validation | `python -m analytics.run_analytics --research` (or `--what euphoria`/`--what phases` with `--research`) - walk-forward + ablation + ML challenger + threshold re-selection. Run after a backfill or rule change; auto-triggers on year rollover or a missing report; `update_data --full` forces it |
| Extend prices to full history | PowerShell: `$env:PIPELINE_START_DATE="2017-01-01"; python pull_bloomberg_prices.py` (incremental — pulls only the missing 2017-2020 spans, then rerun the euphoria stage) |
| Comments + Influence Tracker | **NOTHING TO RUN — `python update_data.py` now does it** (desk decision 2026-07-27, reversing 2026-07-24: an influence board that rescores month-old comments is not current, so comments rejoined the live pipeline). They are still the slow fetch, so the crawl gets a **measured page allowance** instead of being switched off: the desk's `PIPELINE_BUDGET_S = 600 s` ceiling minus what *this machine* actually spends on its other stages, converted to API pages at the contracted 1 req/s. The comment fetch runs **in parallel** with the post fetchers, so those pages cost no extra wall clock. Every run prints `comment budget: N pages … = ceiling − other stages` before fetching and a `run cadence` line after. **Run the pipeline at least every ~3 days** (≈2×/week) and no comments are ever deferred; run it less often and the oldest pages are deferred to the next run (never lost — a capped subreddit keeps its watermark). Opt out for one run with `python update_data.py --skip-comments`; `--with-comments` is still accepted and now does nothing. `python -m analytics.influence --top 20` prints the board |
| Catch up comments after a long gap, or backfill | `python update_comments.py` — the **unbudgeted** runner, for the crawls the 10-minute ceiling cannot afford. `--estimate` prints a computed (not hand-written) runtime from this machine's measured throughput and exits; `--backfill 2026-01-01 2026-07-01` for history. Watermarked, Ctrl-C-safe, resumable. Use this when the pipeline has been idle for weeks; for the ordinary refresh use `update_data.py` |
| Dynamic subreddit panel | NOTHING TO RUN — a monthly, watermarked review rides every live pull (`ingestion/discover_subreddits.py --if-due`): it mines collected text for r/NAME referrals, and a candidate with ≥100 unique panel referrers/28d (the A0 floor, reused) that passes the finance screen auto-joins the EXPLORATION tier (max 1/review). Audit trail: `ingestion/subreddit_panel.json` + `docs/panel_review_latest.md (absent until the first review runs)`. Force a review: `python ingestion/discover_subreddits.py` (`--report-only` to rank without adding) |
| Rebuild the ONSET detector + DESK signals (GET IN / GET OUT) | `python -m analytics.run_analytics --what phases` — LIVE mode: episode catalog + today's scores/alerts at the frozen thresholds (seconds), including `euphoria_desk.parquet` (the boom-gated smoothed GET OUT + phase-aware smoothed GET IN the dashboard shows); add `--research` for the full walk-forward scorecards + threshold re-freeze |
| Re-run the research notebooks | `cd notebooks` then `python -m jupyter nbconvert --to notebook --execute --inplace 04_*.ipynb 00_*.ipynb`. **Only 00, 04, 06-10 have `.ipynb`** — 01, 02, 03 and 05 are jupytext `.py` only, so the old glob failed on three unmatched patterns. Run those as scripts (`python notebooks/01_*.py`) or convert first with `python -m jupytext --to ipynb notebooks/01_*.py` — every figure/number re-renders from current data. The set (since 2026-07-31): **00** = the presentation-grade method walkthrough (raw post → features → thresholds → flag, one worked example, ~1 min); **01–03** = the research record (episode ground truth, feature battery, model tournament); **04** = the consolidated evaluation — every threshold justified with a sweep plot, the signal/feature justification, the walk-forward record with 5/20/84-trading-day outcome tables and median time-to-fall, plus a per-name sample section (`SAMPLE_NAME`) — it replaced the old 04/06/07 (retired to `notebooks/_to_delete_2026-07-31_merged_into_04/`) and writes `docs/research/nb04_evaluation.json`. Run 04 before 00 after a data refresh: 00 reads 04's saved sweep record |
| Influential-users model (notebook 05) | `python notebooks/05_influence_users_model.py` (**there is no `.ipynb` for 05** — corrected 2026-08-05; convert first with `python -m jupytext --to ipynb notebooks/05_influence_users_model.py` if you want figures saved) (~2 min; needs `scikit-learn` + `jupyter`, which the live pipeline and the dashboard do NOT). **No jupyter installed, or only want the numbers?** `python notebooks/05_influence_users_model.py` runs the identical analysis as a plain script in ~107 s and writes the same `nb05_influence.json` (verified byte-identical 2026-07-27) — it just does not save the figures back into the `.ipynb`. Set `MPLBACKEND=Agg` first (PowerShell: `$env:MPLBACKEND="Agg"`) so matplotlib does not try to open 17 plot windows. Runs the full Chan (2026) replication on the current store and rewrites `docs/research/nb05_influence.json (absent until notebook 05 runs)`, which the dashboard quotes. **Concluded 2026-07-27**: `logit` ships, every graph layer rejected, and the model does NOT generalise to unseen authors — so the dashboard ranks by the measured record. Re-run after any big comment pull to refresh the numbers; the adoption ladder re-tests itself and the notebook asserts that what it ships equals `influence_ml.BEST_MODEL` |
| Run the tests | `python -m pytest tests/ -v` |
| Rebuild the presentation evidence pack | **NO LONGER POSSIBLE — `helper/research_charts.py` is not in this repo.** `docs/research/` (figures, `research_stats.json`, README) is therefore a FROZEN artefact with no producer on disk: it can be read and cited, not regenerated. Found 2026-08-05 while auditing cited paths |

## The one switch: the window

```python
START_DATE = "2021-01-01"   # inclusive
END_DATE   = ""             # "" = LIVE (to today);  "2021-11-01" = backtest window
```

`END_DATE = ""` → live fast path (minutes). A date → a frozen backtest
VIEW — instant, because the aggregates are built once over
`BUILD_START_DATE` → today (`--full`) and every window is just a lens over
them. The window drives the Bloomberg pull and the dashboard;
`--start/--end` override it for a single run. A **WINDOW CHECK** in every
run's output flags, per source, whether the chosen window actually has
data — an empty chart is never a mystery.

Every run prints a **data coverage table** (posts per month, per source) so
gaps are visible immediately, and ends with the text-free **safety check**
on ABSTRACTED_DATA.

## Which pipeline steps run where

| Step | External machine (raw store) | Internal machine |
|---|---|---|
| fetch (3 sources, parallel) | every live run | every live run |
| append | `merge_live.py` → posts.parquet, then tail splice | `append_live_abstracted.py` → ABSTRACTED_DATA |
| `--full` rebuild from raw text | yes (`build_aggregates.py`, all cores) | never — no raw text exists there |
| analytics (conviction + signals) | every run | every run |
| Bloomberg pull + dashboard | yes | yes |

The machine is auto-detected (posts.parquet present = external); force
with `--external` / `--internal`.

## Initial setup (once per machine)

```powershell
pip install -r requirements.txt --user
```

Create `.env` in the project root with the FetchLayer key
(`FETCHLAYER_KEY=...`). `.env` is never committed. StockTwits and Arctic
Shift Reddit need no key.

**Internal machine, first time only:**

```powershell
git pull
python -c "from src import abstracted_data; abstracted_data.hydrate()"
```

`hydrate()` copies the committed aggregates from `ABSTRACTED_DATA/` into
`data/processed/`, where the analytics look. After this one step,
`update_data.py` keeps the two in sync automatically.

For the Bloomberg prices, install blpapi once per machine (Terminal running):

```powershell
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi --user
python -c "import blpapi; print('blpapi', blpapi.__version__)"
```

## Everyday live refresh

The same command on both machines — it auto-detects which one it is on:

```powershell
python update_data.py
```

**What this run does NOT do (desk decision 2026-07-28).** It does not
pick a model, re-select a threshold, or re-run the walk-forward,
ablation or ML challenger, and it prints no performance statistics. It
refreshes the data and scores it with the already-frozen winner, so the
job is the same job every time. Research is decided once, in the
notebooks, and frozen. If the frozen record stops at an earlier year
than the data, the run prints one notice line and **keeps scoring at
it** — that is out-of-sample use, which is exactly what the walk-forward
licenses. Re-open the question deliberately, never by drift:

```powershell
python -m analytics.run_analytics --what phases --research
python update_data.py --full     # a backfill IS new research: it rewrites
                                 # the history the thresholds were chosen on
```

Then commit the updated aggregates + the influence store (the store is
committed by design — text-free, pseudonymous; the safety check covers
both):

```powershell
git add ABSTRACTED_DATA data/reference/influence
git commit -m "live update"
git push
```

(Only commit when the run's safety line says **PASS**.)

## Backtest / study a past regime

1. Set the window: `--start 2021-01-01 --end 2021-11-01` (or edit
   `src/config.py`).
2. `python update_data.py --start 2021-01-01 --end 2021-11-01` — backtest
   mode skips fetching automatically; nothing rebuilds unless stale.
3. Open the dashboard, set the same window in the sidebar — every tab
   (overlays included) clips itself to it.

## The dashboard tabs (RetailRadar - all interactive, no notebooks)

- **EUPHORIA: Themes** / **EUPHORIA: Singles** — the headline signal,
  one tab per instrument kind (desk decision 2026-07-24: conclusions
  only on the terminal; evidence in the notebooks). Each tab: a sparse
  state strip (names STARTING = onset alert in the last 21d, names
  ENDING = top alert in the last 21d - empty is the radar working), then
  per-instrument charts with the state ON the chart: BLUE vertical line
  = euphoria starting, RED vertical line = euphoria ending, and a
  "EUPHORIA STARTING/ENDING NOW" badge on the header line.

  **Reading the lower panel (rebuilt 2026-07-28).** It now carries ONE
  line per firing rule, and that line is **how close the rule is to
  firing, as a percentage**: the deciding score divided by that rule's
  own frozen threshold, times 100. So the dotted line at **100 is the
  trigger, always** — every name, every rule, every window — and the
  question "what activated this signal?" has a visible answer: **the
  line touched the top.** It is a crossing, not an inflection. Nothing
  new was computed: same stored score, same frozen threshold, divided,
  and the alert dates are unchanged. A rule is drawn whenever its score
  exists in the window, not only when it fired, so a line that climbs to
  80 and turns over tells you why nothing fired.

  **SHADED DAYS = THE EXIT QUESTION ONLY, AND WHY BOTH LINES CAN LOOK HOT
  AT ONCE** (2026-07-29, seventh pass). A faint red band sits behind the
  days when the name was **end-stage** — the crowd had already cleared
  every exit gate. On those days the detector asks **only** the exit
  question: the entry rule leaves end-stage days out of its candidate set
  altogether. So inside a band the teal GET IN line is a
  *what-it-would-have-said* reading, drawn so the line stays continuous
  and readable, and it **cannot fire however high it goes**. That is why
  both lines can look hot at the same time without contradicting each
  other, and it is not a loophole: in the whole record, **no name-day has
  ever fired both**, and no end-stage day even carries a GET IN score.
  Outside a band, a teal line touching 100 is a real alert with a
  vertical mark under it.

  **THE RED LINE IS NOT THE SLOPE OF THE NAVY LINE.** It is the same
  blend of ingredients **plus one more** — mood rolling over while the
  crowd is still large — rescaled so 100 is its own trigger. They share
  four of five ingredients, which is why they move together. The
  **slope-like line is the teal one**: it is built from short-window
  versus long-window comparisons, so it asks *how fast* attention is
  climbing rather than how high it already is. Read the panel as: **navy
  = how high, red = high and rolling over, teal = climbing fast.**

  **TWO THINGS ON ONE 0-100 SCALE** (2026-07-29, sixth pass). The **navy
  line is the euphoria level, 7-day smoothed** — how hot the crowd is, the
  same number the dial reads, drawn on every calendar day so you can see
  the build-up rather than just today's reading. The **coloured lines are
  how close each rule is to firing**, as a percentage of its own trigger.
  Both run 0-100 and higher is hotter in both, but they are **not the same
  quantity**: a level of 70 is not "70% of the way to a signal", so read
  each line against its own legend entry. One axis rather than two is
  deliberate — with two axes the "fires here" rule could be drawn at any
  height relative to the level curve, which is exactly the *"why is the
  threshold there?"* question the 100 scaling exists to answer.

  **ONE SOLID LINE PER RULE, AND NOTHING ON IT IS INTERPOLATED**
  (2026-07-28, fifth pass — this replaced the grey/coloured two-state
  line the fourth pass added). Each rule is drawn once, solid, in its own
  colour, 7-day smoothed, as a percentage of its own trigger. There is no
  second style to decode: **the line is the score, 100 is the trigger, a
  crossing is the alert.** Whether the name was *eligible* to fire on a
  given day is no longer shown on the line — that is what the flags are
  for. The number plotted is the production scorer's own output, and on
  every day the detector actually judged, the **stored** value is the one
  you see, so the panel can never print a number that contradicts a flag.

  **GET OUT never breaks. A break in GET IN means the crowd was not
  building.** GET OUT inputs are recorded every calendar day, so that line
  runs edge to edge. GET IN breaks where there was no build-up to measure,
  and that splits two ways: either the 7-day chatter share sat **at or
  below this name's own normal level** (its 120-day median — 20.5% of days
  inside a name's span), or there was too little of it to measure at all
  (56.5% — the coverage gate, days before the price history is judgeable,
  or not enough history for the percentiles). A gap is **not** "nobody was
  talking" — one fifth of gap days do have chatter, just not above the
  name's own normal. Nothing is drawn across a gap.

  **A one-day spike is a real fire, and it is not less accurate.** If a
  line shoots through 100 and drops back, that is a fast build, not a
  glitch. Measured: 29 of 109 above-threshold stretches last a single day,
  and those thin alerts hit **37.9%** of the time against **36.4%** for
  alerts sitting on a full 7-day window — a difference a Fisher exact test
  puts at **p = 1.0**. Requiring a full window before firing would delete
  29 alerts, 11 hits and **10 of the 26 captured episodes**, so it was
  measured and rejected (research report §6.16).

  **If a reading looks different from last time you widened the window,
  that was a bug and it is fixed** (2026-07-29). The level curve used to be
  7-day-averaged *after* the window was cut, so the first six days of
  whatever window you had chosen were an average of one, two, … six days —
  meaning the same date could read differently depending on how far back
  you were looking. It is now averaged over the full history and then cut
  to the window. Checked on all 59 names: the **last** day is identical
  either way (so today's reading, the dial and the 7-day change never
  moved), at most six days per name change, the largest correction is 22
  level points, and five names had a *"Peaked at N"* figure that was an
  artefact of the window start. **No signal date changed** — the level is
  not an input to either rule.

  **Only one explainer is left on the page** (desk decision 2026-07-28):
  *"what is euphoria? (start here - plain English)"*. The four deeper
  expanders — the seven-decision summary, the long-form evidence log, the
  full method & measured record, and the printed parameter register — were
  removed from the dashboard. If someone challenges a threshold, the
  answer is in `docs/PARAMETER_REGISTER.md`, `docs/DECISIONS.xlsx` or the
  notebook that produced it, not on the screen.

  **No performance numbers appear on this page** (desk decision
  2026-07-28) — hit rate, lead time and false alarms live in notebook
  04 (§3, the consolidated evaluation), which prints the full scorecard
  with confidence intervals.
  An AMBER band on the price panel marks the DANGER STATE (crowd ≥2×
  its normal AND price ≥25/50% above its 60d low): sharp drops (≥10%
  in a week) begin within 30d on ~62% of these days vs ~19% of ordinary
  days (NB06) — the band is the standing PM warning, the red line the
  timing call. Above each chart sits a **EUPHORIA GAUGE**: a dial whose
  needle is the euphoria level you see plotted underneath (0-100, read at
  the last day of the selected window, with the change vs 7d earlier), and
  whose bands are **calm** below 76, **warming** 76-84 and **RED ZONE** at
  85+. The 85 is not a new number — it is the same level the walk-forward
  froze for the ENDING alert; the 76 is the lowest cut whose effect on
  "does a >=10% fall start within 30 days" stayed positive under all five
  bootstrap seeds (research report §6.10, parameter register Class 1b).
  Read it as a **state, not an instruction**: the sentence beside the dial
  quotes the measured drop risk of the band you are actually in, and warns
  that the level alone at 85+ is only ~1.3x the all-days rate while the
  level *plus* the amber danger band is ~3.1x. GET IN and GET OUT still come
  from the detector and can fire with the needle anywhere. If the dial is
  missing, `docs/research/gauge_zones.json` is absent on this machine —
  it was derived by the retired signal-efficacy notebook (kept, with its
  derivation, in `notebooks/_to_delete_2026-07-31_merged_into_04/`), and
  the dashboard refuses to invent bands without it. A single caption
  states the validated record; the walk-forward tables, sweeps, ablation,
  ML challenger and tournament live in `notebooks/01-04` (04 = the
  consolidated evaluation) + `docs/DECISIONS.xlsx`, not on the terminal
- **Trade desk** — the live ledger, scorecard, certainty ranking, signal
  charts with per-trade reasons; INSTRUMENT LOOKUP expander above the
  tabs shows every suggestion + reason for one tradeable instrument
- **Overlays: themes** — first derivative vs anchor ETF, conviction
  crossings, BUY/SELL on the anchor price + report card. (Single-ticker
  overlays were removed — the desk trades themes via anchor ETFs only;
  ticker research lives in `analytics/`, e.g. windowed backtests via
  `run_analytics --what signals`)
- **Top trends / Emerging trends** — most-mentioned and fastest-growing
  tradeable themes. Emerging has a **growth lookback slider (3–30d)**:
  7d = twitchy early-warning list, 21d = sustained build-ups only
- **Conviction** — ranked by an **EWMA of conviction z** (half-life
  slider, default 10d), with the latest z and the old flat 30d average
  shown alongside. The z uses an EWM trailing baseline (validated July
  2026 on real prices: best cross-validated PnL AND self-recentring
  after coverage shocks). Grey open triangles = the signal reverting to
  neutral, the validated early-exit point; the trade desk flags OPEN
  trades whose conviction has REVERTED ("consider exit" instead of
  waiting out the 20d cap). Conviction is computed live on the
  dashboard - no recompute needed to see engine changes
- **Influence** — *information only; nothing here feeds the euphoria
  signal or the GET IN / GET OUT alerts.* Who has actually been right on
  Reddit, and what they are saying now. **Re-cut 2026-07-27 so every
  number on it has a unit you can say out loud** — the previous version
  printed bare sums, which is why it did not read. **Re-laid-out
  2026-07-28 into four sub-tabs** — *What they are pushing*, *Building or
  fading?*, *The names*, *The map* — after the desk read a single
  scrolling column and asked *"why is everything crowded long?"*. Nothing
  was removed: the window and panel-size controls sit **above** the
  sub-tabs so one population feeds every view (a view whose population
  changed with the tab would not be comparable), and the "why there is no
  model on this tab" footnote stays at page level because it governs all
  four. Six sections, in the
  order a PM reads them: (1) **what the panel is pushing** — one bubble
  per ticker, left/right is net direction, **height is that name's share
  of the room's conviction in per cent**, dot area is how many calls, and
  the dashed line is the **even split**, `100 ÷ names in the window`
  (2.0% across 51 names on the 30-day view), so above the line means more
  crowded than even; (2) **is the crowding building or fading** — the same
  share week by week, over *all* recorded voices rather than the top-N
  panel, with the tilt marker's size showing how many calls that week
  rested on (a thin week looks thin — no week is ever filtered out);
  (3) **who is behind one name** — the people pushing a chosen ticker,
  ranked **influence 0–100** with their side beside each bar;
  (4) **the names** — the board of authors with **5+ judged calls**,
  showing influence 0–100 and their tickers. There is deliberately **no
  per-author hit rate** anywhere on this tab (the stored column still
  exists; it is just not displayed — at five judged calls it is too thin
  to read as skill); (5) the **influence map** — either the reply-graph
  backbone (the k-core: everyone with at least k neighbours inside the
  picture) or one author's neighbourhood, dots sized by who replies to
  them and coloured by usefulness, names arrowed to their dots; and
  (6) two contrarian boards, *called the tops* and *loud but wrong*.
  The "why there is no model on this tab" expander gives the measured
  reason: a model can rank authors it has seen, but on unseen authors it
  sits at the random floor, so the ranking you see is the record, not a
  prediction. **Reading the share**: it is a share, so the numbers on any
  one view add to 100 — that is the check. It is denominated on every name
  in the window, not just the ones drawn, so the chart's figure for a name
  always equals the KPI's figure for the same name. Populates itself from
  ordinary `update_data.py` runs (comments are part of the live pull); the
  map appears once `reply_edges.parquet` exists. Seeding from nothing takes
  two or three runs because the comment fetch is budgeted — `python
  update_comments.py` does it in one uncapped sitting instead
- **Historical checker** — any window, any theme

Every theme list and picker shows the TRADEABLE universe only (themes
with a firm-approved instrument in `THEME_ETFS`).
**Which names get a chart, and why the count is often small.** Three
filters stack on the euphoria tabs, and only the last is a preference.
(1) The name must have euphoria rows inside the selected window. (2) **It
must have ALERTED inside the window** — the tabs show names with a
detected episode, not a padded top-N, and this is what usually binds: in
the default window (1 Jan → 21 Jul 2026) 8 of 34 themes and **3 of the 25
singles present in the window** alerted (AAPL, MSFT, PLTR), while over
all history 30/31 themes and 22/23 singles have alerted at some point. So
"only 3 charts" is a statement about the window, not about the cap —
widen the window and the count rises. (3) The sidebar's `items per
section` slider (3–60, default 6), newest alert first. A line above the
charts always states **how many names had data in the window against how
many of those alerted** — coverage and alerting are reported as two
separate numbers, so an empty tab can never be mistaken for a quiet
universe — and it says so explicitly when the slider is what is holding
the rest back, so the cap is never silent. A name whose history ends
before the window is counted out loud in the same line rather than
folded into the denominator, because it is *absent*, not *calm*. The window
controls at the top drive every tab at once. Chart
note: days with under `MIN_TOTAL` total mentions are masked as too thin;
those stretches draw as a dotted, dimmed bridge with the legend key
"not enough posts that day" — visibly different from real data, and no
values are invented.

## Running pipelines from the dashboard

The sidebar buttons launch the pipelines as BACKGROUND processes: the
app stays responsive, a panel shows a progress bar + plain-English stage
checklist (markers parsed from the run log every 2s), and **cancel**
kills the whole process tree (fetchers and analytics included). The raw
output lives in the panel's *technical log* expander — it auto-opens
when a run fails. Buttons grey out while a pipeline runs (one at a
time); hit *dismiss* on the finished panel to refresh the data views.

In a terminal fetch, a `still fetching: Reddit, X` heartbeat line prints
every ~30s — the pauses are deliberate API rate-limit pacing
(StockTwits ~1.5s/symbol, X 5s/request, Arctic Shift 1s/page), so a
quiet minute is normal, not a hang.

**How long should things take?** (also shown on the dashboard buttons)

| Run | Typical time | Where it goes |
|---|---|---|
| LIVE pull, first run of the day | ~3–10 min | X rate-limit pacing dominates; Reddit paginates only what is NEW since the last run (per-subreddit watermark, 1-day overlap) |
| LIVE pull, repeat run | ~2–5 min | mostly the X pass |
| window rebuild (prices + signals) | ~1–3 min | Bloomberg pull is incremental — covered spans are skipped |
| analytics only | ~1 min | pure local compute |
| FULL rebuild | 30 min – hours | raw-text extraction + sentiment over the whole build range (external machine) |

## Transfer: external → internal

External machine (after `update_data.py`):
`git add ABSTRACTED_DATA && git commit && git push`.

Internal machine: `git pull`, then `python update_data.py` (it detects the
fresh aggregates and recomputes the derived outputs once).

## If something looks off

- `python check_live_ingestion.py` — freshness of every layer, in flow order.
- The **DATA COVERAGE** table in the run output shows exactly which months
  have data, per source ('.' = a real gap).
- The safety line at the end of every run must say **PASS** before
  committing ABSTRACTED_DATA.
- Close Excel/viewers before a run (Windows locks the parquet files); if a
  write fails, the script prints the manual rename fix.
- `python -m pytest tests/ -v` — the invariants that would catch silent
  corruption (merge maths, look-ahead, dedup, text-leaks).
- **`python tools/verify_deps.py`** — does anything reference code that is not
  here? Static, no imports, no network, under a second; exits 1 on a finding.
  Catches missing modules, imported names that do not exist, attributes on
  local modules, repo paths named in strings, and CLI flags handed to a script
  whose argparse rejects them. **Run it after any change that spans module
  boundaries, and before pushing.** It exists because on 2026-07-29 a routine
  `update_data.py` died on `ImportError: cannot import name
  'pipeline_budget'` — a module that ARCHITECTURE §3.1b and PARAMETER_REGISTER
  Class 7 both described in full, that four files called into, and that had
  never existed on disk. Three quieter gaps rode along with it, one of them
  hidden behind a bare `except` in the dashboard. Documentation asserting that
  code exists is not evidence that it does.

## Housekeeping scripts (tools/)

Not part of any pipeline; each is run by hand when you want what it makes.

- `python tools/verify_deps.py` — the dangling-reference sweep above.
- `python tools/export_figures.py` — regenerates the figures the research
  report and the decks embed. Run it after re-running the notebooks, or the
  documents will keep showing the previous fit's charts.
- `python tools/dashboard_shots.py` — headless screenshots of the dashboard
  tabs (Playwright), for the decks.
- `python tools/contrast_audit.py` — checks the dashboard palette against
  WCAG contrast ratios.
- `python tools/nb_codehash.py` — hashes the notebooks so a drifted notebook
  is visible before it is quietly re-run.
