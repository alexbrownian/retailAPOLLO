# CONTEXT PROMPT — RetailRadar / retailAPOLLO (continue from here)

You are picking up a mature quant project. Read this fully before acting.

## Who I am & how to work with me
I'm Alex, an intern (GIP 2026 project, MAARS Global Macro). Credit line used in the app: "Alex Brown - GIP 2026 Project - MAARS Global Macro".
- When answering questions, teach me HOW each step is derived, not just the answer.
- Code at a junior-programmer level (no clever tricks), heavily commented in a human style.
- ALWAYS verify code works before giving it to me (run tests, run the app).
- I present this project like a PhD defense: every rule needs evidence, every fitted number named, every limitation pre-answered.

## The project
`C:\Users\alexd\Desktop\GIC\RetailAPOLLO\retailAPOLLO` — a Python pipeline + Streamlit dashboard ("RetailRadar", no GIC branding anywhere) that measures retail attention/sentiment for themes, single names and retail commodities across 17 finance subreddits, X, and StockTwits.
`C:\Users\alexd\Desktop\GIC\RetailFlow1` — the predecessor project. STRICTLY READ-ONLY, never modify it.

**THE AIM (July 2026 re-aim): detect retail EUPHORIA and call price TOPS.** Success = an alert inside [peak−30d, peak+1d] of a genuine top. **HARD RULE: prediction is Reddit-only — price NEVER enters the euphoria level or alert; price exists only to define ground-truth peaks and score the detector** (enforced by a unit test: `compute_euphoria` accepts no price argument).

## Architecture (unchanged from RetailFlow1)
- Two machines, one repo: external holds raw text (`posts.parquet`, gitignored, can `--full` rebuild); internal holds only `ABSTRACTED_DATA/` (6 committed text-free daily aggregates). Text-free git contract enforced by FORBIDDEN-columns checks.
- One command does everything: `python update_data.py` (parallel fetch of 4 sources with 30s heartbeat → fold with first-seen-wins dedup → analytics → Bloomberg price pull). Window set in `src/config.py` (START_DATE/END_DATE; ""=live).
- Prices: `pull_bloomberg_prices.py` (blpapi, PX_LAST, incremental append-only). Currently 2021→today, ~287 symbols.
- Dashboard: `python -m streamlit run dashboard.py`. Tabs: EUPHORIA (headline), Influence tracker, Overlays, Top/Emerging trends, Conviction, AI Pulse, Historical checker. Sidebar runs pipelines as background processes with progress bar, plain-English stage checklist, cancel (kills process tree), time estimates on buttons. Default window 2026-01-01. Test harness: `streamlit.testing.v1.AppTest` must show 0 exceptions.
- Tests: `python -m pytest tests/ -v` — 33 pass, all invariants (merge maths, no look-ahead, text-free, Reddit-only rule, judgeable window, vol-scaled bar, shrinkage ordering, PageRank+bot filter, store refuses text columns).

## The euphoria detector (`analytics/euphoria.py`) — all rules documented in its docstring
Features = trailing percentile ranks vs the SAME name's own 365d (min 180d): E1 attention extremity (7d mention share), E2 sustained bullishness (28d net-bullish level × ≥75%-of-posting-days persistence gate), E3 crowd influx (28d share change), E5 super-exponential ATTENTION growth (LPPLS-lite: rolling 60d quadratic fit of log(1+mentions7d), positive convexity only — Sornette's signature applied to the crowd, not the chart). LEVEL = 100×mean(E1,E2,E3,E5). E4 fade = attention ≥90th pct while 14d bullish share falls → lowers the alert threshold by 10.
Alert gates: A0 coverage ≥100 posts/28d; A1 hype (7d share ≥ 2× own 120d median); A2 E1≥0.90; A2b E2>0; A3 level ≥ threshold (ONLY fitted number, walk-forward from past years only, conservative default when untrained); A4 21d cooldown. Ground truth (price, testing only): local 43d max, boom ≥25% ETF / ≥50% single off 120d low, bust ≥15%/≥30% within 90d; dual thresholds are a desk decision. Alerts judged only inside the "judgeable window" (price exists at alert and 45d after — else PENDING, not false).
**Validated numbers (walk-forward, real closes): 23% of coverage-detectable peaks captured, median lead 4d, 0.11 FAs/instrument-year.** An earlier price-assisted variant captured 46%/0.08 — the delta is the documented cost of the Reddit-only claim. 2023–25 have ~zero detectable peaks (thin archive coverage — the detector is blind there, not wrong; both denominators always reported). Universe: 59 instruments (34 theme anchor ETFs, rates_bonds+real_estate excluded; top-25 data-chosen single names needing ≥3000 posts).
**Ablation (rerun automatically each rebuild): hype gate = precision lever (+83 FAs removed), fade trigger = capture lever (−0.095), E1/E3 ρ=0.79 so single-drops understate them.** **ML challenger** (walk-forward logistic regression, same features) rejected under a PRE-STATED criterion (must win utility AND total capture AND recent-year capture; it was near-silent: 1 vs 4 captures). Its coefficients rank the same features top. **Key finding: attention features SELECT (E1 AUC 0.61 unconditional, 0.40 gated), sustained bullishness RANKS (E2 AUC 0.60 gated) — gates vs level roles match what each feature can do.**

## The influence tracker (`analytics/influence.py`) — method from Chan (Oxford M.Eng thesis 2026, "Informed Trading Decisions via Social Network Analysis"; PDF was inspiration only)
Calls = authored posts/comments with |VADER|≥0.20 mentioning a ticker. Judged vs 20d closes with volatility-scaled bar τ=max(3%, 0.5σ_90d) + abnormal-return z; enhanced = correct AND ≥1σ. Author usefulness = thesis §4.6 verbatim: three stance-weighted scores, Bayesian shrinkage α=10/5/10, min-max, composite 0.4/0.4/0.2, HIGH tier ≥0.66. Reply-graph (comment→comment + comment→post) → bot-filtered (edge cap 100, >1000C/>100P broadcast excluded, star filter only on ≥50-node graphs) → degree, weighted degree, PageRank (power iteration). "Loud but wrong" flag = top-quartile PageRank + below-median composite (thesis FP profile: hubs had 40% accuracy vs 79% for quiet users — never rank by size). Boom/bust record = bearish calls in [peak−30d, peak+5d] of confirmed tops.
**Operations: LIVE-ONLY, ZERO-TOUCH (latest desk decision).** Every `update_data.py` fetches comments (4th parallel fetcher, watermarked, 3d lookback, 3× timeout) and `run_analytics` calls `influence.update()`: per-file ingest ledger parses only NEW raw files, store self-creates on first pull, extends every pull, calls re-judge as 20d windows mature. NO backfill, NO --build needed (both exist as optional manual tools). Store (`data/reference/influence/`) is COMMITTED (text-free, pseudonymous — hard write-time check); only `ingest_ledger.json` stays local. Frame the board as a forward/out-of-sample record from inception.
Comment fetcher gotcha (fixed): Arctic Shift params must be epoch seconds consistently — mixed ISO/epoch bounds across pagination caused HTTP 422s; 4xx now fail fast, 429/5xx/network retry with backoff.

## Legacy decisions (evidence retained, see dashboard MODEL DECISIONS expander + docs)
BUY/SELL engine retired from dashboard (identical to RetailFlow1's — verified by file diff; full-history BUY = −1.42%/trade, 2022 regime failure; window-clipping created the "used to be good" illusion). Conviction display uses EWM baseline halflife 42 (validated: +1.36%/trade, 63% hit, plateau-robust); signal ENGINE keeps original rolling-84. Grey reversion-exit markers kept (0.080 vs 0.065 %/day). Shorts rejected. Parameters FROZEN in `src/config.py` with evidence quoted inline; changes require NEW out-of-sample evidence.

## Presentation assets (keep in sync when anything changes)
- `docs/DECISIONS.xlsx` — 13 defense-ordered sheets (storyline→aim→literature→data→design→validation→ablation→ML→influence→legacy→limitations→next steps→12 references incl. Barber/Odean 2022 JFE, Sornette LPPLS, Jame WSB, thesis).
- `docs/research/` — 8 figures + `research_stats.json` + README, ALL regenerated by `python helper/research_charts.py` from current data (walk-forward bars, ablation tornado, lead-time hist, Spearman matrix, two-population feature separation, calibration by level bucket, ML comparison + coefficients, 3 case studies incl. GLD flagged 1d before its Jan-2026 peak).
- Chart rules: one axis per panel, fixed validated categorical palette (#2a78d6/#008300/#e87ba4), direct labels, honest captions (calibration is NOT monotone — annotated on the figure).

## Outstanding / next steps
1. I still need to run (Bloomberg terminal open): `$env:PIPELINE_START_DATE="2017-01-01"; python pull_bloomberg_prices.py` then `python -m analytics.run_analytics --what euphoria` then `python helper/research_charts.py` — turns 2018–20 into real train/test years and refreshes all numbers/figures.
2. First live pull after the latest changes seeds the comment watermark + influence store automatically.
3. Named future work: comments folded into the ABSTRACTED aggregates (top capture-recovery lever), LLM call extraction, graph features into euphoria.

## Working conventions for you
Session sandbox had no external API access — external calls are tested with mocks/synthetic fixtures (then cleaned up; never leave synthetic data in stores you deliver); I run live tests on my machine and paste output. Always run pytest + AppTest before delivering. Deliver changed files to my machine via the device bridge to the exact repo paths. Never touch RetailFlow1. Update README/RUNBOOK/ARCHITECTURE/DECISIONS.xlsx when behaviour changes.
