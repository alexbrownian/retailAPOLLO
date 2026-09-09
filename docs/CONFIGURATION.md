# Configuration

Four layers, from the one most people touch to the one nobody should
touch without a research pass.

| Layer | Where | Format | Takes effect |
|---|---|---|---|
| Universe and vocabulary | `config/*.csv` | CSV, one file per concept | next pipeline run (the dashboard re-reads `theme_etfs.csv` on its own) |
| Settings | `config/settings.csv`, overridden per machine by `config/settings.local.csv` | `key,value,description` | next run or dashboard restart |
| Credentials and environment | `.env`, environment variables | `KEY=value` | next run |
| Frozen constants | `src/config.py` | Python, evidence quoted beside each value | only after a research pass; pinned by tests |

## 1. Universe and vocabulary (`config/`)

`config/README.md` documents every file and column. The short version:

| To change… | Edit |
|---|---|
| Which forums are crawled | `forums.csv` (`enabled` yes/no; add a row to add a forum) |
| Which themes exist and which ETF represents each | `theme_etfs.csv` |
| Whether a theme is shown on the dashboard (display only) | `theme_etfs.csv → show_on_dashboard` |
| The words that map a post to a theme | `theme_keywords.csv` |
| The tickers that belong to a theme | `theme_tickers.csv` |
| Which instruments can be priced, and their Bloomberg / Yahoo symbols | `approved_instruments.csv` |
| Which single names are always in or always out | `single_name_overrides.csv` |
| Words that must never count as tickers | `ticker_stoplist.csv` |
| Short tickers that count without a `$` | `ticker_allowlist.csv` |
| The AI poll questions and the "asked an AI" patterns | `ai_poll_prompts.csv`, `agentic_terms.csv` |

Validate after editing:

    python tools/validate_config.py

The validator checks columns, types and the references between files
(every theme in keywords and tickers exists; every ETF is an approved
instrument; the anchor leads its fallback chain; allow-list and
stop-list do not overlap; every regex compiles). `tools/preflight.py`
and the test suite run the same checks, so a broken config cannot reach
a run unnoticed.

Adding a theme touches four files: a row in `theme_etfs.csv`, its
keywords in `theme_keywords.csv`, its tickers in `theme_tickers.csv`,
and its ETF in `approved_instruments.csv`. Removing a theme from the
dashboard is one cell (`show_on_dashboard = no`); removing it from the
detector's universe is a research pass, because the frozen thresholds
were fitted on the universe as it stood.

## 2. Settings (`config/settings.csv`)

| Key | Default | Meaning |
|---|---|---|
| `app_title` | `RetailRadar` | Dashboard name. |
| `app_tagline` | — | One line under the title. |
| `app_credit` | empty | Optional credit line. |
| `show_pipeline_controls` | `false` | Fetch / price / rebuild buttons in the sidebar. Keep `false` in the committed file. |
| `price_provider` | `auto` | `auto`, `bloomberg` or `yfinance` (`RUNBOOK.md` §2.2). |
| `single_name_top_n` | 25 | Single names tracked, ranked by mentions. |
| `single_name_window_days` | 365 | Ranking window for the above. |
| `bot_screen_enabled` | `true` | Run the bot screen before aggregation. |
| `bot_screen_threshold` | 0.6 | Score at or above which a post is excluded. |
| `bot_screen_duplicate_jaccard` | 0.85 | Near-duplicate similarity cut. |
| `bot_screen_burst_posts_per_day` | 12 | Author posts per day that count as a burst. |

Missing keys fall back to the defaults in `src/settings.py`, so deleting
a row is safe. `config/settings.local.csv` (git-ignored) overrides any
key on one machine; `tools/publish_dashboard.py` creates it with
`show_pipeline_controls,true` on the machine that runs the pipeline.
`RETAILAPOLLO_CONTROLS=1` in the environment has the same effect for one
process.

Read settings from code through `src.settings` (`get`, `get_bool`,
`get_int`, `get_float`, `load_forums`, `single_name_overrides`); never
parse the CSV directly.

## 3. Credentials and environment

`.env` holds credentials (`docs/LIVE_INGESTION.md` §1 and
`example.env`). Environment variables that change behaviour:

| Variable | Effect |
|---|---|
| `PIPELINE_START_DATE`, `PIPELINE_END_DATE` | The analysis window; passed to child processes by `update_data.py`. |
| `RETAILAPOLLO_CONTROLS=1` | Show the sidebar pipeline buttons for this process. |
| `AI_PROVIDER`, `AI_MODEL`, `ANTHROPIC_MODEL`, `AI_MAX_CALLS` | AI layer selection and budget (`RUNBOOK.md` §5). |
| `DESK_MODEL_FAMILY` | Model family a research pass fits (`ens` by default; empty re-opens the tournament). |
| `EUPHORIA_STRICT_BETA` | F-beta weight for the strict operating point stored beside the production one. |
| `SIG_K`, `SIG_MIN_SCORE`, `SIG_MIN_SCORE_SELL`, `SIG_COOLDOWN` | The conviction-based signal layer's knobs; documented beside each in `src/config.py`. |

## 4. Frozen constants (`src/config.py`)

Everything that defines ground truth, features, gates and thresholds
lives in `src/config.py` with its evidence quoted beside it, and
`reference/KEY_PARAMETERS.md` has a row for each. These are not
settings: changing one changes what counts as an episode or how a score
is formed, which invalidates the stored walk-forward record.
`tests/test_production_hygiene.py` pins the ground-truth values, so an
accidental edit fails the suite. The procedure for a deliberate change
is in `RUNBOOK.md` §6.
