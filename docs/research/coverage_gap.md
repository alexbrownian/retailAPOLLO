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

    python tools/backfill_reddit.py --estimate   # runtime, measured here
    python tools/backfill_reddit.py              # resumable, 33 chunks
    python update_data.py --skip-fetch
    python -m analytics.run_analytics --what phases --research

The last step is not optional: a backfill rewrites the history the
thresholds were chosen on, so scoring new history against thresholds
fitted on the old history would be a silent lookahead
(PARAMETER_REGISTER, "Two explicit ways to re-open research").
