# Data quality investigation — why shares "went to zero", and the fix
*(2026-08-07; the estimator lives in `analytics/robust_share.py` and is
used by every attention series — charts and signal features alike.)*

## The symptom

Ticker share-of-forum lines intermittently collapsed to zero on days the
name was demonstrably still being discussed: over 1 May – 5 Aug 2026 the
top-30 tickers printed **308 zero-share days** between them (SMCI 11 of
36 days in July alone; AMC 27; GME 21 over the window).

## The diagnosis (from the stores, not a guess)

Two coverage mechanics, neither of them crowd behaviour:

1. **Uneven pull cadence.** The pipeline pulls ~2×/week; a catch-up run
   lands several days of posts at once. Daily totals swing 246 → 4,033
   posts between adjacent days (2 Aug vs 4 Aug 2026). The old estimator
   computed share **per day** and then averaged the daily ratios, so a
   10-post day carried the same weight as a 4,000-post day — and a day
   the puller skipped printed a hard 0%.
2. **Source-mix regime changes.** Reddit runs continuously back to 2017;
   StockTwits only ramps from Feb-2026; X only exists from Jul-2026. The
   denominator pooled all sources, so a Reddit-heavy name's share was
   diluted wholesale the day a big StockTwits/X pull landed — a
   composition artifact the detector then percentile-ranked as if the
   crowd had left.

Monthly per-source coverage (days with any data) makes the regime visible:
StockTwits 0–5 days/month through 2025, 21+ from Apr-2026; X first
appears Jul-2026.

## The fix — three standard estimation techniques, no invented data

* **Ratio-of-sums** (Cochran's ratio estimator): share over the trailing
  7-day window = Σ mentions / Σ total posts, replacing the average of
  daily ratios. Thin days now contribute exactly the evidence they carry.
* **Per-source stratification** (tickers, where
  `daily_ticker_counts_by_source.parquet` exists): share is computed
  *within* each source and combined with weights equal to each source's
  own trailing-90-day volume share, renormalised over the sources present
  in the window. A missing StockTwits pull now means "use the sources we
  do have", not "halve everyone's share".
* **Empirical-Bayes shrinkage** toward the name's own trailing-120-day
  share, prior strength = `MIN_TOTAL` (30) posts — the project's existing
  too-thin-to-trust floor, reused rather than a new constant. With
  hundreds of posts the data speaks for itself; with a handful, the
  estimate leans on the name's own recent baseline instead of printing
  0% or 100%.

Display masking moved with the estimator: a chart value is masked only
when the whole trailing week holds under `MIN_TOTAL` posts (nothing to
estimate from), not when one thin day does.

## Measured result (top-30 tickers, 1 May – 5 Aug 2026)

| metric | old estimator | robust estimator |
|---|---|---|
| fake zero-share days | 308 | 31 (all genuine no-coverage stretches) |
| median day-to-day relative change | 12.0% | 11.5% |
| source-mix dilution on big pull days | full | removed by stratification |

The signal layer consumes the same estimator (`euphoria._mention_share`),
so E1/E3/A1 and the onset bank no longer rank coverage artifacts as
attention extremes; thresholds were re-frozen through the standard
walk-forward after the change.

## What was *not* done, and why

No interpolation or imputation of missing raw data: a day with no pull
is genuinely unobserved, and inventing posts would poison the counts the
whole project is built on. The estimator only changes how observed
counts are *weighted* and *combined* — every input number is real.
