# POST-INTERN HANDOVER — what to edit, when, and how

This file is the maintenance map for retailAPOLLO after the internship.
Everything a future owner needs to keep the system honest lives here:
which files are meant to be edited, on what cadence, what is pending,
and the rules that must not be broken. The deep documentation stays
where it always was (README, RUNBOOK, docs/ARCHITECTURE.md,
docs/PARAMETER_REGISTER.md, notebooks 00–10); this is the short list.

---

## 1. The editable configuration (all plain CSV, Excel-friendly)

Everything the desk is EXPECTED to edit lives in `config/`. The code
loads these at import and fails loudly on typos — a bad edit shows up
immediately, never silently.

| File | What it controls | When to edit |
|---|---|---|
| `config/theme_etfs.csv` | theme → anchor ETF + fallback chain + notes. An EMPTY etf cell = tracked but untradeable. | When the approved instrument list changes, or a proxy gets a proper line (notes column flags the known proxies: solar→LIT wants TAN, gaming→SOCL wants ESPO, ai/quantum→IYW want AIQ/QTUM). |
| `config/approved_instruments.csv` | The firm-approved tradeable list: symbol + EXACT Bloomberg code. Every anchor/fallback must appear here. | Whenever an instrument is approved/removed. The Bloomberg puller derives its non-US securities from this file — adding a foreign line is one row here, nothing in code. |
| `config/theme_keywords.csv` | The word/phrase → theme map (Signal 1). RULE: words and phrases only, never bare ticker symbols. | Via the AI keyword auditor (section 3) or by hand any time. Keywords apply at INGESTION — run a FULL rebuild after editing for history to reflect them. |
| `config/theme_tickers.csv` | ticker → theme map (Signal 2), incl. ETF-constituent rows (source column says which ETF each came from). | When constituents drift (annually is fine) or a new name matters. `config/etf_constituents.csv` holds the researched holdings behind it. |
| `config/agentic_terms.csv` | The regex bank detecting "trading with AI" posts (four categories). | When new AI products/phrasings appear (e.g. a new consumer agent product). One regex per row; the scanner fails loudly on a bad pattern. |
| `config/ai_poll_prompts.csv` | The retail questions the pipeline asks the LLM at every refresh. | When retail's typical questions drift. Keep prompts stable where possible — the VALUE is the time series, and editing a prompt breaks its history (add new prompt_ids instead of rewording old ones). |
| `.env` (project root - THE single env file, desk decision 2026-08-04; the old example.env template is retired) | ALL credentials: FETCHLAYER_KEY + Apollo LLM auth (ENVIRONMENT, APOLLO_AUTH_USERNAME, APOLLO_AUTH_PASSWORD, AI_MODEL, AI_DATA_CLASSIFICATION, AI_MAX_CALLS; optional AI_MOCK=1). | On credential rotation. GIT-IGNORED - a fresh clone has no template, so keep a private backup of this file; this row is the authoritative key list. |

Reddit forum coverage: the subreddit panel is DYNAMIC —
`ingestion/subreddit_panel.json` is maintained by the discovery pass
(`ingestion/discover_subreddits.py`, desk rules 2026-07-24) with
`ingestion/finance_subreddits.txt` as the seed list. Edit the seed list
to force a forum in; the panel discovery handles the rest. Check the
panel file occasionally for dead or hijacked subreddits.

## 2. The operating cadence

| Task | How | Cadence |
|---|---|---|
| QUICK UPDATE (prices + signals) | sidebar button, Terminal open | as needed |
| FULL UPDATE (live pull) | sidebar button | ~2×/week (the comment budget is sized for this — the sidebar prints the measured cadence) |
| Comment catch-up | sidebar EXTRA button | only after long idle spells |
| AI pulse + AI poll + agentic scan | automatic at the end of every update (VPN needed for the LLM parts; they skip politely otherwise) | automatic |
| Keyword audit (section 3) | automatic weekly during updates; applying is manual | weekly propose, human apply |
| Bloomberg price pull for NEW instruments | first QUICK UPDATE after editing the instrument list | on change |
| Notebook re-execution after a signal-code change | `python -m jupytext --to ipynb notebooks/<n>.py` then `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/<n>.ipynb` | on change |
| Test suite | `python -m pytest tests/ -q` (expect 106 passed) | before any commit |

## 3. THE KEYWORD AUDITOR — the workflow, in full

The theme→keyword map rots quietly (new slang, renamed companies,
drifted meanings). `tools/ai_keyword_audit.py` keeps it fresh WITHOUT
ever letting the model edit the config:

1. **Propose (automatic, weekly).** During any update, if the newest
   `config/keyword_suggestions_*.csv` is older than 7 days and the
   Apollo gateway is reachable, a fresh audit runs. The LLM sees the
   current map plus the top HIGH-FREQUENCY UNMAPPED terms from the
   crowd's own text (trailing 60d) and writes
   `config/keyword_suggestions_<date>.csv`: one row per proposed
   add/move/remove, with a reason, and an EMPTY `approved` column.
   (Manual run any time: `python tools/ai_keyword_audit.py`.)
2. **Review (human, minutes).** Open the suggestions CSV. Put `YES` in
   the `approved` column on rows you accept. Ignore the rest.
3. **Apply (human, one command).**
   `python tools/ai_keyword_audit.py --apply config/keyword_suggestions_<date>.csv`
   — merges ONLY approved rows into `theme_keywords.csv` and prints the
   diff it made.
4. **Rebuild.** Keywords apply at ingestion, so run the FULL rebuild
   (`python update_data.py --full`, on the machine holding
   posts.parquet) for history to re-count under the new map. Until
   then, new keywords affect only newly ingested posts.

Guardrails baked in: the model is instructed to never propose bare
ticker symbols (case-insensitive matching would poison prose), to
propose nothing when unsure, and it CANNOT write to the config — the
apply step is the only writer, and it only writes approved rows.

## 4. The AI layer — one connection, many consumers

* `src/ai.py` is the ONLY file that talks to the LLM (Apollo gateway
  via `dimsum_lite`; VPN + JFrog install required — desk machine only).
  Self-test: `python -m src.ai --selftest` (expect Paris). Model swap:
  `AI_MODEL` in `.env` ('model-not-found' → pick a deployment from
  `dimsum_lite.constants`). Every AI feature degrades politely off the
  VPN — banners, never crashes.
* **AI page on the dashboard** (one page, two sections, desk decision
  2026-08-04): A — the PULSE (the LLM's qualitative read of the posts;
  numbers from the stores, words from the model, evidence pack saved
  for audit); B — the POLL (what the AI recommends when asked like a
  retail trader). The CHATTER SCAN (posts about trading with AI) still
  runs at every update but is RESEARCH-ONLY — its record is notebook 09
  and `daily_agentic_counts.parquet`, deliberately not a page.
* **The poll series has no backfill** — protect its continuity. Its
  correlation test vs the flags is PRE-REGISTERED in notebook 09 §2b
  and self-activates at ≥60 poll days. Do not peek early, do not
  reword old prompts.
* Text-free boundary: raw crowd text goes TO the model; only
  model-written summaries/paraphrases are stored. The local-only text
  files (`data/reference/agentic_samples.jsonl`,
  `nb10_sample_texts.jsonl`) are git-ignored — keep them that way.

## 5. Pending items (state on 2026-08-04)

* **Prices missing for new instruments**: EUAD, MOO, XRT, DTCR, BBH,
  KRE, 159915 CS, 588000 CH, MTUM — run a QUICK UPDATE with the
  Terminal open. Until then charts fall back down the ETF chains.
* **MTUM Bloomberg code**: stored as `MTUM TF Equity` as provided —
  Cboe BZX is usually `UF`; verify on the Terminal
  (config/approved_instruments.csv, one cell).
* **CSIN0852** is the CSI 1000 INDEX, not a fund — priced for
  reference only (flagged in the instruments CSV).
* **EUAD approval**: confirm it is actually on the approved list
  (chosen as the europe_defense anchor; note in theme_etfs.csv).
* **Notebook 07 (index composite)**: MTUM judging auto-activates once
  MTUM is priced — just re-execute the notebook.
* **Notebook 10 (AI sentiment)**: FinBERT + LLM scoring cells are
  PENDING for the desk machine (proxy blocked them in the cloud);
  each caches its scores, re-execution completes the verdict. Set
  `AI_MAX_CALLS=80` for the LLM run.
* **First real AI pulse**: the committed `ai_pulse.json` is MOCK —
  the first update on the VPN replaces it.
* **Git history cleanup** (optional, ~85MB): junk blobs committed
  inside old `_to_delete` folders; `git gc` locally, or
  `git filter-repo` on the `_to_delete*` paths before sharing.
* **`data/processed/daily_ticker_conviction.parquet` (164MB)**: written
  by the pipeline, read by nothing (the dashboard computes conviction
  live). Safe to delete; consider disabling its write in
  `analytics/conviction.py`.
* **`notebooks/_to_delete_2026-07-31_merged_into_04/`** and
  `data/raw/RedditComments/_salvaged_originals/…tmp`: delete when
  convenient.

## 6. Rules that must not be broken

1. **Frozen means frozen.** Signal thresholds live in the walk-forward
   record (`euphoria_desk_report.json`, notebook 04 §1). Any change to
   scoring code or constants is a RE-VALIDATION EVENT: re-run the
   research pass (`python -m analytics.run_analytics --what phases
   --research`), compare the record, re-execute notebooks 00/04/06/07/08.
   Nothing ships as live flags unless it beats the incumbent under the
   pre-stated rule (see notebook 08 for the worked example of a change
   that WON the display and LOST the flags — that division is by design).
2. **Committed data is text-free.** `verify_abstracted` enforces it at
   every update; never weaken it.
3. **The dashboard shows conclusions; the notebooks are the record.**
   Performance claims belong in notebooks with CIs, not on screens.
4. **Every constant carries its provenance** (PARAMETER_REGISTER /
   notebook 04). No unexplained numbers, ever.
5. **AI writes words, never numbers, and never edits config.** The
   evidence pack pattern (pulse) and the approve-then-apply pattern
   (keyword auditor) are the templates for any future AI feature.
