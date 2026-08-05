# OPEN ITEMS — what is outstanding

*The live to-do list, and the only document in this project that is
expected to go out of date. Everything permanent lives elsewhere (see
the documentation map at the end of `docs/ARCHITECTURE.md`). Delete a
line when it is done; do not archive it here.*

Last reviewed: **2026-08-04**

## Needs the desk machine (Bloomberg / the VPN)

| Item | What to do | Why it is blocked |
|---|---|---|
| Prices for the new instruments | run a QUICK UPDATE with the Terminal open | EUAD, MOO, XRT, DTCR, BBH, KRE, 159915 CS, 588000 CH, MTUM have no price history yet; until then charts fall back down the ETF fallback chains |
| First real AI pulse and poll | run any update on the VPN | the committed `ai_pulse.json` is MOCK and `ai_poll.parquet` does not exist yet — the first live run writes both |
| Notebook 10 (AI sentiment) | re-execute on the desk machine with `AI_MAX_CALLS=80` | the FinBERT and LLM scoring cells were proxy-blocked in the cloud. Each caches its scores, so re-execution completes the verdict |
| Notebook 07 (index composite) | re-execute once MTUM is priced | MTUM judging auto-activates then; nothing to edit |
| `.env` | fill `APOLLO_AUTH_PASSWORD`, then `python -m src.ai --selftest` (expect Paris) | the gateway cannot authenticate without it |

## Needs a human decision

| Item | What to check |
|---|---|
| MTUM Bloomberg code | stored as `MTUM TF Equity` as provided; Cboe BZX is usually `UF`. One cell in `config/approved_instruments.csv` |
| EUAD approval | confirm it is actually on the firm's approved list — it was chosen as the `europe_defense` anchor (note in `config/theme_etfs.csv`) |
| CSIN0852 | the CSI 1000 INDEX, not a fund; priced for reference only. Flagged in the instruments CSV — decide whether to keep pulling it |

## Housekeeping (safe, unglamorous)

| Item | What to do |
|---|---|
| `data/processed/daily_ticker_conviction.parquet` (164MB) | written by the pipeline, read by nothing — the dashboard computes conviction live. Safe to delete; consider disabling the write in `analytics/conviction.py` |
| Git history (~85MB) | junk blobs committed inside old `_to_delete` folders. `git gc` locally, or `git filter-repo` on the `_to_delete*` paths before sharing the repo |
| Stale folders | `notebooks/_to_delete_2026-07-31_merged_into_04/` and `data/raw/RedditComments/_salvaged_originals/…tmp` — delete when convenient |

## Sealed until their gate opens — do not peek

*Both are pre-registered forward tests. Running a partial version of one
is how a pre-registration stops being one.*

| Test | Gate | Where |
|---|---|---|
| Does the AI poll LEAD our flags? | ≥60 distinct poll days | notebook 09 §2b — self-activating |

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
research pass and re-execute notebooks 00/04/06/07/08. This is a
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
`Index & factors` tab, `analytics/basket_breadth.py`, the `sp500`,
`momentum_factor` and `growth_factor` themes, the MTUM/VUG constituent
rows, and `EUPHORIA_BOOM_MIN_INDEX` / `_CRASH_MIN_INDEX` /
`EUPHORIA_INDEX_SCALE_NAMES`. The tradeable universe is back to **34
themes**, exactly where it was before.

Kept deliberately, because they are correct on their own merits and
cost nothing: the **XLK and XLV constituent rows** (real S&P sector
holdings the file was missing), the **VOO/VTI/KO allowlist entries** and
the **ES stoplist row**, all of which came from measurement rather than
from the index idea.

`notebooks/07_index_composite.py` is kept as the RECORD of what was
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
