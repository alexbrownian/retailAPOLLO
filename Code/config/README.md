# Configuration

Everything a user is expected to change without touching Python lives in
this folder as CSV. Edit the files in any spreadsheet program (a UTF-8 BOM
from Excel is handled), then, from the project root, run:

    python Code/tools/validate_config.py

The validator checks every file's columns and the references between
files, and the test suite and `Code/tools/preflight.py` run the same checks.
Changes take effect on the next pipeline run or dashboard restart.

## Which file to edit

| I want to… | Edit |
|---|---|
| Change the dashboard title, tagline or credit line | `settings.csv` |
| Turn the pipeline buttons on or off in the sidebar | `settings.csv` → `show_pipeline_controls` |
| Add, pause or remove a forum (subreddit) | `forums.csv` |
| Add a tracked theme, or change which ETF represents it | `theme_etfs.csv` (+ `approved_instruments.csv`) |
| Hide a theme from the dashboard without changing any computed result | `theme_etfs.csv` → `show_on_dashboard` |
| Change which words map posts to a theme | `theme_keywords.csv` |
| Change which tickers belong to a theme | `theme_tickers.csv` |
| Add a tradeable instrument (ETF or index) | `approved_instruments.csv` |
| Force a single name into or out of the tracked set | `single_name_overrides.csv` |
| Change how many single names are tracked | `settings.csv` → `single_name_top_n` |
| Stop a word being read as a ticker (e.g. "DD", "YOLO") | `ticker_stoplist.csv` |
| Make a short ticker recognisable without a `$` (e.g. "MU") | `ticker_allowlist.csv` |
| Tune the bot screen | `settings.csv` → `bot_screen_*` |
| Rename the on-demand ETF lookup panel | `settings.csv` → `lookup_section_title` |
| Change the AI poll questions | `ai_poll_prompts.csv` |
| Change what counts as an "asked an AI" post | `agentic_terms.csv` |

## File reference

### `settings.csv` — `key,value,description`

One row per setting. Missing keys fall back to the defaults in
`src/settings.py`, so deleting a row is safe.

| Key | Type | Meaning |
|---|---|---|
| `app_title` | text | Name in the masthead and sidebar. |
| `app_tagline` | text | One line under the title. |
| `app_credit` | text | Credit line under the tagline. Empty shows nothing. |
| `bloomberg_max_failures` | int | Terminal failures tolerated in one run before the price pull gives up on it for that run. |
| `app_thanks` | text | Acknowledgement at the bottom of the sidebar. Empty shows nothing. |
| `show_pipeline_controls` | yes/no | Show fetch / price-pull / rebuild buttons in the sidebar. Keep **no** on a hosted copy: those buttons call paid APIs. `RETAILAPOLLO_CONTROLS=1` in the environment forces **yes**. |
| `single_name_top_n` | integer | How many single names the detector tracks. |
| `single_name_window_days` | integer | Trailing window used to rank single names by mentions. |
| `bot_screen_enabled` | yes/no | Run the bot screen before building aggregates. |
| `bot_screen_threshold` | 0–1 | Posts with `bot_score` at or above this are excluded from aggregates. |
| `bot_screen_duplicate_jaccard` | 0–1 | Near-duplicate similarity cut for the MinHash step. |
| `bot_screen_burst_posts_per_day` | integer | Author posts-per-day at or above which the burst flag fires. |
| `lookup_section_title` | text | Heading of the on-demand ETF lookup panel (beta) on the landing page. |

### `forums.csv` — `forum,source,tier,enabled,added,note`

The panel of forums the ingestion layer crawls, in file order.

- `forum` — the subreddit name without `r/`.
- `source` — `reddit` (the only forum-type source today).
- `tier` — `core` (founding panel) or `exploration` (added by the
  automatic panel review). Informational.
- `enabled` — `yes` to crawl, `no` to pause. Pausing keeps history.
- `added` — date or `founding`.
- `note` — free text. The panel review writes its reasoning here.

`ingestion/discover_subreddits.py` appends rows here when a new forum
qualifies; review them and set `enabled` as you see fit.

### `theme_etfs.csv` — `theme,etf,fallbacks,show_on_dashboard,note`

One row per theme. This is the **tradeable universe** for themes.

- `theme` — the slug used everywhere (`semiconductors`, `gold_metals`).
- `etf` — the anchor instrument whose price represents the theme. Must
  appear in `approved_instruments.csv`. Leave empty for a theme that is
  tracked for attention but has no tradeable line.
- `fallbacks` — `|`-separated instruments tried in order when the anchor
  has no price data. The anchor must lead the chain.
- `show_on_dashboard` — `no` hides the theme from every dashboard list.
  **Display only.** The theme is still fetched, scored and stored, so the
  frozen model thresholds are unaffected. To remove a theme from the
  detector's universe, delete the row and re-run a research pass.
- `note` — why this anchor; any known caveat.

To add a theme: add the row here, add its keywords to
`theme_keywords.csv`, its constituent tickers to `theme_tickers.csv`,
and make sure the ETF is in `approved_instruments.csv` with a price
source.

### `theme_keywords.csv` — `theme,keyword`

Words and phrases that map a post to a theme. One keyword per row; a
theme has many rows. Matching is case-insensitive on whole words for
single tokens and on the exact phrase otherwise. Every `theme` must
exist in `theme_etfs.csv`.

### `theme_tickers.csv` — `theme,ticker,source`

Tickers that belong to a theme, so a post mentioning `$NVDA` counts
toward `semiconductors`. `source` records where the mapping came from
(`curated`, `<ETF> constituent`). Every `theme` must exist in
`theme_etfs.csv`.

### `approved_instruments.csv` — `symbol,bloomberg,tiingo,name,note`

Every instrument the pipeline may price or trade. `symbol` is the
internal key; `bloomberg` is the price-source identifier used by
`pull_prices.py` (`--provider bloomberg`). The `tiingo` column is the
Tiingo ticker used by `--provider tiingo` for a line whose ticker
differs from the internal symbol; US tickers need no entry (`BRK.B`
maps to `BRK-B` automatically). Leave it empty for a line Tiingo does
not carry; it is then skipped with a message. Any other price provider must supply a
`prices.parquet` keyed on `symbol` (see `research.ipynb`).

### `single_name_overrides.csv` — `symbol,action,note`

The single-name universe is chosen from the data: the most-mentioned
priced tickers over the trailing `single_name_window_days`, up to
`single_name_top_n`, excluding ETFs and stop-listed words. This file
adjusts that list:

- `include` — always track the symbol (it must have price data).
- `exclude` — never track it, even if it ranks.

Leave the file with only its header for pure data-driven selection.

### `ticker_stoplist.csv` — `symbol,reason`

Upper-case words that look like tickers but are jargon (`DD`, `YOLO`).
Never counted as tickers.

### `ticker_allowlist.csv` — `symbol,reason`

Real tickers too short for the default pattern to catch without a `$`
prefix (`MU`, `AMD`). Counted when they appear as bare capitals.

### `etf_constituents.csv` — `etf,ticker,company,rank,verified`

Holdings of the anchor ETFs, used to explain which single names sit
inside a theme. Informational; refreshed by hand.

### `etf_catalogue.csv` — `symbol,name,keywords`

A convenience list of common ETFs for the dashboard's on-demand ETF
lookup, so a search by name or theme word works with no Terminal and
no network. `keywords` is `|`-separated. Add rows freely;
nothing else reads this file.

### `agentic_terms.csv` — `category,pattern,note`

Regular expressions that identify posts describing the use of an AI
assistant for investment decisions. Each pattern must compile.

### `ai_poll_prompts.csv` — `prompt_id,family,prompt,note`

The questions the AI poll asks each run. Keep the set small; every
prompt is one paid model call per run.

## Rules the validator enforces

- Every file exists and has its required columns.
- `settings.csv` values parse as the type each key expects.
- At least one forum is enabled; no duplicate forums.
- Every theme anchor and fallback is an approved instrument; the anchor
  leads its chain; no duplicate themes.
- Every theme named in keywords or tickers exists in `theme_etfs.csv`.
- The allow-list and stop-list do not overlap.
- Override actions are `include` or `exclude`.
- Every agentic pattern compiles.
