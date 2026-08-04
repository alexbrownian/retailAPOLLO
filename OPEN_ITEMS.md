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
| Does mobilisation LEAD attention, and does it PAY? | ≥90 days with a name past all three rally gates | notebook 09 §3b — self-activating. Until it passes, the rally detector stays display-and-research only |

## The one big open question (raised 2026-08-04)

**Reddit comments are fetched but never counted.** `build_aggregates.py`
builds every mention and sentiment aggregate from Reddit SUBMISSIONS
only. The 765,000 comments on disk feed the influence board, the AI
pulse, the agentic scan and the rally detector — but not the signal.
Measured over 2026-07-23→29: the pipeline counted **1,111** Reddit
ticker-mentions across 360 symbols; the raw comment archives for the same
week hold **9,629** across 923 symbols.

That is why the single-name universe is 27 names, and no setting of
`EUPHORIA_MIN_COVERAGE` fixes it (see PARAMETER_REGISTER Class 12).

Folding comments in would multiply coverage roughly tenfold and is the
**largest re-validation event this project could undertake** — every
count, z-score, euphoria level and frozen threshold moves while the
ground-truth episodes do not. It needs a decision before any work:
  * is comment sentiment trustworthy enough to enter the signal, or does
    it belong as a separate feature with its own weight?
  * the full walk-forward record would have to be rebuilt and compared
    against the incumbent under the pre-stated rule, both directions.
