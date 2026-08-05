# POST-INTERN HANDOVER — the files you edit, and when

**This file has exactly one job: it lists every file a maintainer is
expected to change, says when to change it, and says what to run
afterwards.** Nothing else lives here.

If you are looking for something else:

| You want | Read |
|---|---|
| how to run the pipeline, and what to do when it breaks | `RUNBOOK.md` |
| why the system is shaped this way, and the rules that must not be broken | `docs/ARCHITECTURE.md` |
| why a particular number has the value it has | `docs/PARAMETER_REGISTER.md` |
| what is still outstanding | `OPEN_ITEMS.md` |
| the evidence behind any claim | `notebooks/00`–`10` |

---

## 1. The configuration you are meant to edit

All plain CSV, all Excel-friendly, all in `config/`. The code loads them
at import and **fails loudly on a typo** — a bad edit shows up
immediately, never silently.

| File | What it controls | When to edit | Run afterwards |
|---|---|---|---|
| `theme_etfs.csv` | theme → anchor ETF + fallback chain. An EMPTY etf cell = tracked but untradeable | when the approved instrument list changes, or a proxy gets a proper line (the `note` column flags the known proxies: solar→LIT wants TAN, gaming→SOCL wants ESPO, ai/quantum→IYW want AIQ/QTUM) | QUICK UPDATE. The dashboard re-reads this file on its mtime, so a **rerun is enough — no restart** (fixed 2026-08-04; it used to need one and nothing on screen said so). The live map, its fallback chains and every `note` are shown under "the theme → instrument map" on the EUPHORIA: Themes tab, stamped with the file's last-edited time — look there first if an edit seems not to have landed |
| `approved_instruments.csv` | the firm-approved tradeable list: symbol + EXACT Bloomberg code. Every anchor and fallback must appear here | whenever an instrument is approved or removed. The Bloomberg puller derives its non-US securities from this file — a foreign line is one row here and nothing in code | QUICK UPDATE, Terminal open |
| `theme_keywords.csv` | the word/phrase → theme map (Signal 1). RULE: words and phrases only, **never bare ticker symbols** | via the keyword auditor (§2) or by hand any time | FULL rebuild — keywords apply at ingestion, so history only re-counts after one |
| `theme_tickers.csv` | ticker → theme map (Signal 2), including ETF-constituent rows (the source column says which ETF each came from). **A factor or index theme is built from its HOLDINGS, never from the fund symbol** — measured 2026-08-05 over 176k comments, MTUM appears 0 times, VTV 0, IVW 0, and "IVE" is people typing "I've" (4 caps vs 82 lowercase) | when constituents drift (annually is fine) or a new name matters. `config/etf_constituents.csv` holds the researched holdings behind it | FULL rebuild |
| `etf_constituents.csv` | the researched holdings behind `theme_tickers.csv`, and the baskets the dashboard's index/factor read is computed over (`analytics/basket_breadth.py`). Each row carries its rank and a `verified` flag | when a fund rebalances — MTUM is semiannual, and its holdings ARE the momentum theme, so a stale list quietly makes the theme wrong | FULL rebuild |
| `agentic_terms.csv` | the regex bank detecting "trading with an AI" posts (four categories) | when new AI products or phrasings appear | `python -m src.agentic_watch --rebuild` |
| `ticker_stoplist.csv` | symbols the crowd uses as WORDS, never counted as tickers (HYSA, DYOR, DRAM, BTC, CEO, YOLO …) | when a new finance abbreviation gets issued to a real ETF, or new slang appears. The test that earns a row: **a fund cannot be discussed before it is listed** — if mentions predate the listing, it is the word | FULL rebuild |
| `ticker_allowlist.csv` | real tickers the bare-word pass would otherwise miss: 1–3 letter symbols (MU, AMD, IBM, QQQ) that are too short for the `[A-Z]{4,5}` bare matcher, and English-word collisions (META, SOFI, HOOD, COIN, UBER) that the word-frequency screen suppressed | when a newly-prominent name is being under-counted. Allowlisted symbols match **only in CAPITALS**, so "coin" stays a word and "COIN" becomes Coinbase. Do NOT add single letters or names where the English word dominates — measure first | FULL rebuild |
| `ai_poll_prompts.csv` | the 30 retail questions the pipeline asks the LLM at every refresh: the plain questions retail types, plus the personas and agent scaffolds retail actually runs (the `family` column tags which) | when retail's typical questions drift, or a new AI-agent product changes how people prompt. **ADDING a prompt is always safe. REWORDING one is not** — the value is the time series, and an edit silently breaks that prompt_id's history, so add a new id instead. A unit test fails the build if p01–p12 are altered | nothing — it takes effect on the next update |
| `.env` (project root) | ALL credentials: `FETCHLAYER_KEY` plus the Apollo LLM auth (`ENVIRONMENT`, `APOLLO_AUTH_USERNAME`, `APOLLO_AUTH_PASSWORD`, `AI_MODEL`, `AI_DATA_CLASSIFICATION`, `AI_MAX_CALLS`; optional `AI_MOCK=1`). This row is the authoritative key list | on credential rotation | `python -m src.ai --selftest` (expect Paris) |

**`.env` is git-ignored and there is no template** — the old
`example.env` was retired on 2026-08-04 so there is only ever one env
file to confuse. Keep a private backup.

**Reddit forum coverage is NOT a config file.** The subreddit panel is
dynamic: `ingestion/subreddit_panel.json` is maintained by the discovery
pass (`ingestion/discover_subreddits.py`), seeded from
`ingestion/finance_subreddits.txt`. Edit the **seed list** to force a
forum in; the discovery pass handles the rest. Glance at the panel file
occasionally for dead or hijacked subreddits.

## 2. The keyword auditor — the one workflow with a human in it

The theme→keyword map rots quietly (new slang, renamed companies,
drifted meanings). `tools/ai_keyword_audit.py` keeps it fresh without
ever letting the model edit the config.

1. **Propose — automatic, weekly.** During any update, if the newest
   `config/keyword_suggestions_*.csv` is older than 7 days and the
   gateway is reachable, a fresh audit runs. The model sees the current
   map plus the top high-frequency UNMAPPED terms from the crowd's own
   text (trailing 60d) and writes
   `config/keyword_suggestions_<date>.csv`: one row per proposed
   add/move/remove, with a reason and an EMPTY `approved` column.
   (Manual run any time: `python tools/ai_keyword_audit.py`.)
2. **Review — human, minutes.** Open the CSV. Put `YES` in the
   `approved` column on rows you accept. Ignore the rest.
3. **Apply — human, one command.**
   `python tools/ai_keyword_audit.py --apply config/keyword_suggestions_<date>.csv`
   merges ONLY approved rows into `theme_keywords.csv` and prints the
   diff it made.
4. **Rebuild.** Keywords apply at ingestion, so run the FULL rebuild for
   history to re-count under the new map. Until then, new keywords
   affect only newly ingested posts.

The model is instructed never to propose bare ticker symbols
(case-insensitive matching would poison prose) and to propose nothing
when unsure — and it **cannot write to the config**. The apply step is
the only writer, and it only writes approved rows.

## 3. Cadence — what runs, how often

| Task | How | Cadence |
|---|---|---|
| QUICK UPDATE (prices + signals) | sidebar button, Terminal open | as needed |
| FULL UPDATE (live pull) | sidebar button | ~2×/week — the comment budget is sized for this, and the sidebar prints the measured cadence |
| Comment catch-up | sidebar EXTRA button | only after a long idle spell |
| Agentic scan | automatic at the end of every update; pure python, no gateway, always runs | automatic |
| AI pulse + AI poll | automatic at the end of every update; need the VPN and skip politely without it | automatic |
| Keyword audit (§2) | automatic weekly proposal; applying is manual | weekly propose, human apply |
| Bloomberg pull for NEW instruments | first QUICK UPDATE after editing the instrument list | on change |
| Notebook re-execution after a signal-code change | `python -m jupytext --to ipynb notebooks/<n>.py` then `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/<n>.ipynb` | on change |
| Test suite | `python -m pytest tests/ -q` — expect **150 passed, 2 skipped** | before any commit |

## 4. Before you change anything that scores

Editing `src/config.py` or any file under `analytics/` is a
**re-validation event**, not a code change. `docs/ARCHITECTURE.md` §6
has the protocol and the reasoning. In short: re-run the research pass,
compare the stored record, re-execute notebooks 00/04/06/07/08, and ship
nothing as a live flag unless it beats the incumbent under the
pre-stated rule.

The editable CSVs in §1 are outside that rule by design — that is
exactly why they are CSVs.
