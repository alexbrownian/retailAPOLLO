# OPEN ITEMS — what is outstanding

*The live to-do list, and the only document in this project that is
expected to go out of date. Everything permanent lives elsewhere (see
the documentation map at the end of `docs/ARCHITECTURE.md`). Delete a
line when it is done; do not archive it here.*

Last reviewed: **2026-08-07**

## From the 2026-08-07 rebuild (robust shares · looser ground truth · learned desk model)

| Item | What to do | Why |
|---|---|---|
| Re-execute notebook 04 on the desk machine when convenient | standard jupytext → nbconvert loop | 00 (new), 01, 02, 03 and the presentation pack (11) were re-executed under the new ground truth / estimator / model on 2026-08-07; 04 is the heavy rules-era record and re-runs cleanly but its stored figures still quote pre-change numbers until re-run. (Notebooks 06–10 were RETIRED the same day — see `notebooks/_retired_2026-08-07_presentation_refactor/README.md`; their frozen JSONs remain live.) |
| Refresh the deck figure pack | `python tools/build_deck_figures.py` after the notebook re-run | F10/F11/F15/F19 quote the old record; the new comparison table is `docs/research/ml_tournament.md` + `docs/figures/deck/F20_model_tournament.png` |
| Watch the first live weeks of the ens model | nothing to run — read the signal snapshots | the frozen cut (~0.92 both heads) was chosen by train-year F1 under the new ground truth; the forward snapshots are its real out-of-sample test |

> **Before you change anything, and after: `python tools/preflight.py`.**
> Eight checks, no writes, no network, safe to run at any time. It exists
> because almost nothing in this project fails loudly — a renamed ticker
> counts zero forever, a theme whose anchor lost its prices is silently
> drawn on a fallback, and a config edit that fails validation only
> surfaces at the next full run. Every check corresponds to something
> that actually happened. Exit 1 = something downstream is already
> wrong; warnings = something is owed.

## Needs the desk machine (Bloomberg / the VPN)

| Item | What to do | Why it is blocked |
|---|---|---|
| Prices for the new instruments | run a QUICK UPDATE with the Terminal open | EUAD, MOO, XRT, DTCR, BBH, KRE, 159915 CS, 588000 CH, MTUM have no price history yet; until then charts fall back down the ETF fallback chains |
| First real AI pulse and poll | run any update on the VPN | the committed `ai_pulse.json` is MOCK and `ai_poll.parquet` does not exist yet — the first live run writes both |
| Notebook 10 (AI sentiment) — RETIRED 2026-08-07, verdict still pending | resurrect from `notebooks/_retired_2026-08-07_presentation_refactor/`, re-execute on the desk machine with `AI_MAX_CALLS=80` | the FinBERT and LLM scoring cells were proxy-blocked in the cloud. Each caches its scores, so re-execution completes the verdict |
| Notebook 07 (index composite) — RETIRED 2026-08-07 | resurrect from `notebooks/_retired_2026-08-07_presentation_refactor/` and re-execute once MTUM is priced | MTUM judging auto-activates then; nothing to edit |
| `.env` | fill `APOLLO_AUTH_PASSWORD`, then `python -m src.ai --selftest` (expect Paris) | the gateway cannot authenticate without it |

## Needs a human decision

| Item | What to check |
|---|---|
| MTUM Bloomberg code | stored as `MTUM TF Equity` as provided; Cboe BZX is usually `UF`. One cell in `config/approved_instruments.csv` |

| CSIN0852 | the CSI 1000 INDEX, not a fund; priced for reference only. Flagged in the instruments CSV — decide whether to keep pulling it |

## Housekeeping (safe, unglamorous)

| Item | What to do |
|---|---|
| ~~`daily_ticker_conviction.parquet` (164MB)~~ | **DONE 2026-08-05** — the write is disabled in `analytics/conviction.py` and the file deleted. It was read by nothing; the dashboard computes ticker conviction live from the sentiment store. A 164MB write every run on a finite disk is a failure waiting for a quiet week |
| Git history (~85MB) | junk blobs committed inside old `_to_delete` folders. `git gc` locally, or `git filter-repo` on the `_to_delete*` paths before sharing the repo |
| Stale folders | `data/raw/RedditComments/_salvaged_originals/…tmp` — delete when convenient. **`notebooks/_to_delete_2026-07-31_merged_into_04/` is NOT safe to delete**: it holds the only notebook that writes `docs/research/nb04_final_eval.json`, which `dashboard.py` reads live. Deleting it orphans a working dashboard read. A test now fails if it disappears while that read exists |

## Sealed until their gate opens — do not peek

*Both are pre-registered forward tests. Running a partial version of one
is how a pre-registration stops being one.*

| Test | Gate | Where |
|---|---|---|
| Does the AI poll LEAD our flags? | ≥60 distinct poll days | notebook 09 §2b (retired 2026-08-07 — resurrect from `notebooks/_retired_2026-08-07_presentation_refactor/` when the data threshold is met) |

## The one big open question (raised 2026-08-04)

**Reddit comments are fetched but never counted.** `build_aggregates.py`
builds every mention and sentiment aggregate from Reddit SUBMISSIONS
only. The 765,000 comments on disk feed the influence board, the AI
pulse and the agentic scan — but not the signal.
Measured over 2026-07-23→29: the pipeline counted **1,111** Reddit
ticker-mentions across 360 symbols; the raw comment archives for the same
week hold **9,629** across 923 symbols.

It is why coverage is thin enough that only 27 single names clear the
28-day firing gate. (Separating universe MEMBERSHIP from that firing gate
later the same day took the tracked universe from 27 to 69 without
loosening anything — see PARAMETER_REGISTER Class 12 — but the
underlying scarcity, and the ceiling it puts on everything, is still
this.)

Folding comments in would multiply coverage roughly tenfold and is the
**largest re-validation event this project could undertake** — every
count, z-score, euphoria level and frozen threshold moves while the
ground-truth episodes do not. It needs a decision before any work:
  * is comment sentiment trustworthy enough to enter the signal, or does
    it belong as a separate feature with its own weight?
  * the full walk-forward record would have to be rebuilt and compared
    against the incumbent under the pre-stated rule, both directions.

## Pending, created 2026-08-04 by the ticker fix

**A FULL rebuild is owed.** `config/ticker_stoplist.csv` and
`config/ticker_allowlist.csv` both apply at INGESTION, so they only affect
newly-ingested posts until history is re-counted under them. Until that
rebuild runs on the machine holding `posts.parquet`:

* the jargon symbols (HYSA, DYOR, DRAM, BTC, REIT …) are still inflating
  historical mention counts — HYSA is still the largest "ticker" in the
  stored history;
* MU, AMD, IBM, QQQ, META, SOFI, HOOD, COIN and the rest are still
  under-counted in history — MU by roughly 44×, measured.

`python update_data.py --full` on the external machine, then re-run the
research pass and re-execute notebooks 00–04 and the presentation pack (11). This is a
RE-VALIDATION EVENT: every count moves, so both frozen thresholds must be
re-derived and compared against the incumbent.

## Four themes are drawn on a substitute instrument (2026-08-04)

`agriculture_food`, `consumer_retail`, `datacenters` and `europe_defense`
have anchors with **no price history in the store** (MOO, XRT, DTCR,
EUAD), so `resolve_anchor` falls through to the first priced line in the
chain and the theme is actually drawn on XLP, XLY, IYW and ITA
respectively. `europe_defense` is the one that matters: it is being
priced on a **US** aerospace line, which is exactly the substitution
that theme's own config note was written to prevent.

Root cause, now fixed in code: `pull_bloomberg_prices.py` built its
request from theme anchors and fallbacks only, so an approved
instrument that no theme pointed at was never asked for. Fifteen of the
sixty approved lines had no prices — the four anchors above, the factor
and style lines (MTUM, RSP, IVE, IVW, VTV, VUG), the CSI 1000 index,
BBH, KRE, and the two Chinese local listings.

**What is owed:** one price pull on the desk machine with the Terminal
open — `python pull_bloomberg_prices.py --dry-run` to preview, then
without the flag. Verify `MTUM TF Equity` first; `approved_instruments.csv`
records the exchange code as provided, and Cboe BZX is usually `UF`.
Until it runs, the EUPHORIA: Themes tab flags every substitution in
"the theme → instrument map" rather than leaving it silent.

## Full ticker-mapping audit, 2026-08-05

All 1,462 rows of `theme_tickers.csv` and 876 of `etf_constituents.csv`
checked against the Nasdaq/NYSE listing files (created 2026-07-02).

**Nine mappings were dead** and counting nothing, silently: `SQ` (Block
re-tickered to `XYZ`), `PARA` (`PSKY` after the Skydance merger), `MMC`
(`MRSH` since 14 Jan 2026), `VSCO` (`VSXY`), `ARMN` (`ARIS`), `SPLG`
(`SPYM`), `BITF` (`KEEL`), plus `CYBR` (merged into Palo Alto Networks
and delisted) and `DIDI` (NYSE delisting, OTC only). All corrected.
`KEEL` also changed THEME, not just ticker: Bitfarms stopped being a
bitcoin miner and became US AI infrastructure, so it moved crypto →
datacenters.

**Thirty symbols can never match as tickers by construction** — 26 OTC
ADRs, which the listed universe excludes, and four dotted class shares
(`BRK.B`, `HEI.A`, `MOG.A`, `UHAL.B`), which cannot pass
`_SYMBOL_OK = ^[A-Z]{1,5}$`. They now reach their themes by NAME
instead: 38 keyword rows added (`tencent`, `airbus`, `rolls-royce`,
`xiaomi`, `berkshire`, `moog`, `heico` …). A test fails if any
unmatchable symbol loses its name route.

**The 1–2 letter blind spot was measured and mostly left alone.** Eighty
mapped symbols are one or two letters and therefore invisible to
`WORD_BARE = [A-Z]{4,5}`. Measured over 176,126 comments, only three
cleared a volume-and-ratio bar, and reading the samples killed two of
them: `PM` is "send a PM" / "Canadian PM" (one hit in six was Philip
Morris) and `ES` is the E-mini S&P future, not Eversource. Only **KO**
survived — 48 CAPS, 12 cashtags, 4 lowercase, every sample Coca-Cola —
and it was added. `ES` went on the STOPLIST so it can never be
allowlisted later and poison `utilities_power`.

**Left as deliberate:** `AI`, `DD`, `NOW` and now `ES` are mapped to
themes AND stoplisted. The stoplist wins in `extract_tickers`, so C3.ai,
DuPont, ServiceNow and Eversource are documented as theme members but
never counted from prose. A test pins that exact set so a NEW clash is
noticed rather than absorbed.

## Run-log findings, 2026-08-05

Four things the run showed that were not the Bloomberg failure.

**~95 budgeted pages fetched nothing.** The comment crawl walks
newest-first, so once a subreddit stops yielding new comments every
further page is 100 rows we already hold — and nothing stopped it.
r/personalfinance burned 21 dead pages, r/Daytrading 14, r/Bogleheads
10, and those pages came out of the same ceiling that then deferred
r/wallstreetbets. The run also reported "page budget reached" for
subreddits that had simply run dry, and left their watermarks in place,
so the next run started in the same spot and bought the same dead pages
again — a dry subreddit could never make progress. Fixed:
`DRY_PAGES_STOP = 3` ends the crawl after three consecutive pages with
nothing new, and marks it completed so the watermark advances.

**r/Bitcoin stopped early on a rate limit disguised as a client error.**
HTTP 422 with `{"error": "Timeout. Maybe slow down a bit"}` was
classified as "our request is malformed, do not retry". The body now
decides: a 422 saying "slow down" backs off like a 429.

**The false-alarm budget was moving, and it was a race.** It was read
from `euphoria_report.json` while the euphoria stage rewrote that file
in PARALLEL, so the adoption bar depended on which stage finished first.
Two consecutive `--what phases` passes over identical data printed
"budget 0.23" then "budget 0.19" and disagreed: GET OUT captured 17 then
16, adjacency 4 then 5. Underneath the race it also ratcheted — each
run's realised FA rate became the next run's bar, drifting downward with
nothing on disk recording by how much. Now frozen at
`EUPHORIA_FA_BUDGET_PER_IY = 0.23` in `src/config.py`.
**This is a re-validation event:** re-run `--what phases --research` and
compare the stored record before trusting any threshold selected under
the old moving value.

**The universe says 36 themes; the config defines 37.**
`broad_market_passive` has no priced line anywhere in its chain (RSP,
VTV, VUG, IVE, IVW are all in the never-priced set), so it is counted in
the crowd data and then dropped before scoring. The dashboard now says
so instead of the count quietly disagreeing. One Bloomberg pull fixes
it; no config edit needed.

Both pandas `FutureWarning`s are also gone — `signals.py` and
`euphoria_phases.py` were filling NaN into boolean columns and relying
on a downcast pandas 2.x deprecated.

## The index / factor workstream, REMOVED 2026-08-05

Built and removed the same week on desk instruction ("remove all the
index stuff it doesnt seem to work"). Gone from the tree: the
`Index & factors` tab, the basket-breadth module
(`analytics/basket_breadth.py` was removed), the `sp500`,
`momentum_factor` and `growth_factor` themes, the MTUM/VUG constituent
rows, and `EUPHORIA_BOOM_MIN_INDEX` / `_CRASH_MIN_INDEX` /
`EUPHORIA_INDEX_SCALE_NAMES`. The tradeable universe is back to **34
themes**, exactly where it was before.

Kept deliberately, because they are correct on their own merits and
cost nothing: the **XLK and XLV constituent rows** (real S&P sector
holdings the file was missing), the **VOO/VTI/KO allowlist entries** and
the **ES stoplist row**, all of which came from measurement rather than
from the index idea.

`notebooks/_retired_2026-08-07_presentation_refactor/07_index_composite.py`
is kept as the RECORD of what was
tried, because the question will be asked again. Its verdict box now
carries the three measurements that killed it: the crowd never types a
factor fund (MTUM 0 mentions in 176k comments, "IVE" is "I've"); reading
a basket through its constituents worked but measured mega-cap tech
under three names (momentum overlapped `memory` 12/25); and the S&P
returns ZERO ground-truth episodes at the frozen bars — as does XLI —
with volatility scaling failing to rescue it, because an index's
drawdowns are shallower AND slower rather than merely smaller.

The durable conclusion, worth not re-learning: the detector earns its
keep on narratives the crowd argues about, and diversification is
precisely what stops an index staging the run-up-then-bust arc the
ground truth is built to find.

## The euphoria explainer now carries the full rule set

The two reference expanders on the Themes tab were removed on desk
instruction; only "what is euphoria?" remains, and it now contains a
table of every gate (A0–A4) with its GET IN and GET OUT value side by
side, the factor bank each rule scores, the cooldown, the FA budget and
the ground-truth definition. Every number is imported live from
`src/config.py`, so a constant and its explanation cannot drift apart.

The anchor-substitution CHECK the removed expanders performed is kept as
an inline warning that renders **only when something is wrong** — it is
what caught `europe_defense` being drawn on ITA, a US line standing in
for a European one.

## Project clean-up pass, 2026-08-05

**The one that mattered: the euphoria stage is ~4x faster.**
`_bullish_series` used `groupby("date").apply(lambda ...)`, which on the
306k-row sentiment store cost **482 ms per instrument** against **2 ms**
for the vectorised form - the same arithmetic done once per group in
Python instead of once in C. It ran once per instrument in TWO places
(`analytics/euphoria.py` and a copy-pasted twin in
`analytics/euphoria_phases.py`), so across 59 instruments that was ~84 s
of pure interpreter overhead in every full analytics run. Output verified
`.equals()`-identical before the swap. Measured end to end: the euphoria
stage went **19.0s -> 5.1s**.

**Dead code removed** (nothing imports, calls, tests or documents any of
it): `trade_desk` and `certainty_table` (overlays), `weekly_heatmap_frames`
and `snail_trail` (conviction), `load_prices` and `day_span` (loaders),
`author_label` (stocktwits_data). Plus eight unused imports across
`euphoria.py`, `euphoria_phases.py`, `overlays.py`, `fetch_stocktwits.py`,
`verify_deps.py` and `dashboard.py`.

**One duplicate implementation collapsed.** `resolve_anchor` existed
character-for-character in both `dashboard.py` and `analytics/euphoria.py`
- one more place for the fallback rule to drift away from the engine that
actually scores. The dashboard now imports it.

**Fourteen cited paths did not exist**, and the reason none of them were
caught is worth recording: `tools/verify_deps.py` only matches paths
inside QUOTED STRING LITERALS in `.py` files, so every reference living in
a `.md` file or a Python comment is structurally invisible to it. It
reports "0 findings. Nothing dangles" while the following were all broken:

* ~~**`helper/` does not exist in this repo**~~ — **THIS FINDING WAS
  WRONG, corrected 2026-08-05.** `helper/` DOES exist on the desk machine
  and contains `research_charts.py` and `find_emerging_terms.py`. The
  cloud working copy this audit ran in was an incomplete clone, and five
  citations were "corrected" to say the directory was missing before the
  error was caught. All five have been reverted. Same for
  `docs/panel_review_latest.md`, which also exists on the desk machine.
  **The lesson is about the method, not the folder:** an
  absence-of-evidence finding is only as good as the completeness of the
  tree it was run against, and `tools/verify_deps.py` cannot know it is
  looking at a partial checkout. Run it on the FULL repository before
  believing a "missing file" result. The desk machine also carries
  `docs/HANDOFF_PROMPT.md`, `docs/LIVE_INGESTION.md`,
  `docs/RESEARCH_REPORT.md` and `docs/DATA_FLOW.tex`, none of which were
  present in the audited copy either.
* **`RUNBOOK.md`'s notebook re-run command globbed `01/02/03_*.ipynb`**,
  none of which exist (01, 02, 03 and 05 are jupytext `.py` only), so the
  command failed on three unmatched patterns. Corrected.
* **`notebooks/05_influence_users_model.ipynb`** was cited by the RUNBOOK
  and printed to the user by the dashboard. Only the `.py` exists.
* `analytics/basket_breadth.py` (removed) in the handover (removed the same day),
  `ingestion/add_x_data.py` (removed) in `src/clean_data.py`,
  `docs/panel_review_latest.md` (absent) in `discover_subreddits.py`.

**Left deliberately, with reasons:**

* `docs/research/nb06_*.json`, `nb07_performance_battery.json` and
  `nb04_final_eval.json` are produced ONLY by notebooks now sitting in
  `notebooks/_to_delete_2026-07-31_merged_into_04/` - and `dashboard.py`
  actively reads `nb04_final_eval`. **Deleting that `_to_delete` folder
  orphans a live dashboard read.** Constraint on any future cleanup.
* `src/config.py` has three constants nothing imports: `DATA_DIR`,
  `COMMENT_EWMA_RUNS` and `DESK_EXIT_Z`. The first two are clerical.
  **`DESK_EXIT_Z` is not** - it sits in the desk-configuration block, so
  its absence may mean an exit rule was described in config and never
  wired into `euphoria_phases.py`. Check that before deleting it.
* `ingestion/merge_live.py` and `ingestion/append_live_abstracted.py`
  share four near-identical collector functions differing only in their
  column list. Both run on the same pass and write different stores, so
  they cannot be merged blind - but a normalisation fix applied to one and
  not the other is a silent divergence risk.
