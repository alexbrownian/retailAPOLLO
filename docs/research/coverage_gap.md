# The 2023-2026 coverage gap

Measured 2026-08-11 from `ABSTRACTED_DATA/daily_ticker_counts.parquet`.
This is the reference the backfill runner (`tools/backfill_reddit.py`)
points at, and the reason its default window is 2023-04-01 -> 2026-01-01.

## What the stores show

Ticker-mention rows per quarter. A healthy quarter runs 10,000-16,000;
the drought quarters run ~1,000, a ~90% collapse, and 2023Q2 is missing
entirely.

| quarter | days with data | ticker-mention rows |
|---|---|---|
| 2017Q1 | 90 | 4,686 |
| 2017Q2 | 91 | 5,044 |
| 2017Q3 | 92 | 5,121 |
| 2017Q4 | 92 | 5,849 |
| 2018Q1 | 90 | 7,976 |
| 2018Q2 | 91 | 6,327 |
| 2018Q3 | 92 | 6,604 |
| 2018Q4 | 92 | 6,681 |
| 2019Q1 | 90 | 7,193 |
| 2019Q2 | 91 | 5,675 |
| 2019Q3 | 92 | 5,982 |
| 2019Q4 | 92 | 5,076 |
| 2020Q1 | 91 | 10,278 |
| 2020Q2 | 91 | 13,932 |
| 2020Q3 | 92 | 14,315 |
| 2020Q4 | 92 | 16,622 |
| 2021Q1 | 90 | 30,146 |
| 2021Q2 | 91 | 19,876 |
| 2021Q3 | 92 | 15,858 |
| 2021Q4 | 92 | 14,657 |
| 2022Q1 | 90 | 10,893 |
| 2022Q2 | 91 | 9,774 |
| 2022Q3 | 92 | 11,436 |
| 2022Q4 | 92 | 9,617 |
| 2023Q1 | 90 | 11,138 |
| 2023Q3 | 88 | 554 |
| 2023Q4 | 86 | 380 |
| 2024Q1 | 91 | 1,073 |
| 2024Q2 | 91 | 1,066 |
| 2024Q3 | 91 | 1,096 |
| 2024Q4 | 91 | 1,111 |
| 2025Q1 | 90 | 1,140 |
| 2025Q2 | 90 | 965 |
| 2025Q3 | 92 | 1,141 |
| 2025Q4 | 92 | 833 |
| 2026Q1 | 90 | 13,937 |
| 2026Q2 | 91 | 14,505 |
| 2026Q3 | 37 | 9,485 |

## What it costs

The gap is not cosmetic. 49 of the 55 tracked instruments have zero
scored days across 2024-25, and 118 of the 494 ground-truth episodes -
24% - fall inside the blind window with exactly one gradeable day. 2023
is degraded rather than blank: 63 episodes, 16 of them detectable. Every
performance number the pack quotes is therefore measured on 2021-22 and
2026 and is silently blind to the middle.

It also reaches the influence tracker: judged author calls exist only for
2021 and 2026 (see figure P12), so the long/short result there replicates
across a mania and a recovery but has nothing to say about the flat years
in between.

## Why it happened

Not a scoring bug. The posts were never pulled: the Arctic Shift
incremental window is `max(lookback, watermark)` and therefore cannot
walk backwards, so any span the pipeline was not running across stays
empty forever unless it is explicitly backfilled.

## Fixing it

**THE INSTRUCTIONS BELOW THIS LINE WERE WRONG AND COST 45 HOURS.**
Corrected 2026-08-18 after the desk ran the backfill and nothing moved.

`tools/backfill_reddit.py` pulls the raw posts correctly - 904k of them,
16 chunks, 45.7 h - but on THIS machine (the internal one, no
posts.parquet) they could never reach the aggregates. The only fold-in
path, `ingestion/append_live_abstracted.py`, keeps candidates dated
>= LIVE_START and drops everything older BY DESIGN, which is every
backfilled post. `update_data.py --full` cannot rescue it either: it
rebuilds from posts.parquet, which only exists on the external machine.
The posts sat unused in data/raw/RedditLive for a week.

The working sequence is:

    python tools/backfill_reddit.py            # pull raw (or download dumps)
    python tools/fold_historical.py --arctic   # THE MISSING STEP
    python -m analytics.run_analytics --what phases --research

`tools/fold_historical.py` is the deliberate second door: it aggregates
historical posts straight into ABSTRACTED_DATA (text-free, same
`aggregate_posts` the live path uses), with a per-(file, month) ledger
so re-running cannot double count, and a hard refusal to touch days at
or after LIVE_START.

MEASURED RESULT of the first fold (2026-08-18, 962,715 posts, 18 blocks
covering 2023-04 -> 2024-07):

| quarter | before | after |
|---|---|---|
| 2023Q2 | 0 | 10,253 |
| 2023Q3 | 554 | 10,557 |
| 2023Q4 | 380 | 10,027 |
| 2024Q1 | 1,073 | 14,231 |
| 2024Q2 | 1,066 | 13,024 |
| 2024Q3 | 1,096 | 5,194 (July only) |

Density matches the healthy quarters either side (2023Q1 11,138;
2026Q1 13,937), and the one block overlapping already-committed days
added +60 rows - i.e. the dedup held.

STILL OPEN: 2024-08 -> 2025-12 (17 months). Either continue the backfill
runner (~49 h at the measured pace) or download per-subreddit torrent
archives and fold them with `--dumps`. Ranked by how much coverage each
subreddit restores (measured on the healthy 2026 window, share of
covered name-days retained): wallstreetbets alone 24%, +valueinvesting
+stocks 59%, +dividends +bogleheads 76%, +personalfinance +pennystocks
+daytrading 90%.

The last step is not optional: a fold rewrites the history the
thresholds were chosen on, so scoring new history against thresholds
fitted on the old history would be a silent lookahead
(PARAMETER_REGISTER, "Two explicit ways to re-open research").
