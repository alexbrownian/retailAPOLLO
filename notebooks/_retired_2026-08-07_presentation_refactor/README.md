# Retired 2026-08-07 — the presentation refactor

The live notebook pipeline was reshaped to mirror the final presentation
(desk instruction: "clean up all the notebooks that aren't needed and
the notebooks pipeline should replicate what the presentation will be").
The pipeline is now: **00 data & quality → 01 ground truth → 02 features
→ 03 model tournament → 04 evaluation → 05 influence → 11 the
presentation pack**.

Six notebooks moved here, files intact, nothing deleted. Every frozen
JSON they wrote **remains in `docs/research/` and is still read by the
dashboard** — retiring the notebook retires the *workbench*, never the
*record*. Why each left the live set:

| notebook | why it is here | its frozen record |
|---|---|---|
| `00_method_walkthrough` | a worked example of the July RULES mechanics (gates → thresholds → flag); the shipped desk signal is now the tournament-selected model, and the presentation-grade walkthrough role passed to `11_presentation_pack` | (none — it read nb04's) |
| `06_strictness_study` | answered "are the gates too strict?" about gates that no longer drive the desk signal | `nb06_strictness.json`, `nb06_desk_*.json`, `gauge_zones.json` (the dashboard gauge still reads this) |
| `07_index_composite` | SPY/MTUM composite study; verdict was descriptive-only (SPY has zero qualifying episodes) and MTUM judging is still blocked on a price pull | `nb07_index_composite.json`, `nb07_performance_battery.json` |
| `08_single_state` | the phase-clock display study; its verdict (phase clock as the DISPLAYED state) shipped and is frozen | `nb08_single_state.json` |
| `09_agentic_watch` | the agentic measurement ships via `src/agentic_watch.py` in the pipeline; every link test was null on ~4 months of data — recorded, nothing further to run until coverage grows | `nb09_agentic_watch.json` |
| `10_ai_sentiment` | the finVADER-vs-FinBERT-vs-LLM tournament; verdict PENDING a desk-machine run (`AI_MAX_CALLS=80`) — resurrect this notebook from here when that run happens | `nb10_ai_sentiment.json`, `nb10_sample.parquet` |

To resurrect one: move its `.py`/`.ipynb` pair back to `notebooks/` and
re-execute — they import the same live modules as everything else.
