# ARCHITECTURE — how the system is put together, and what may never change

*This file holds the SHAPE of the system and its invariants. It is the
home for "why is it built this way". It deliberately does not repeat
what lives elsewhere — see the documentation map at the end.*

> **Note, 2026-08-04.** Five shipped files (`README.md`, `dashboard.py`,
> `src/pipeline_budget.py`, `tools/verify_deps.py`,
> `POST_INTERN_HANDOVER.md`) cited `docs/ARCHITECTURE.md` and its
> section numbers while the file itself was missing from the repository.
> Rather than strip the references, the file was written and the cited
> sections (3.1b, 3.1b-i, 6.1) created to hold what those call sites
> promise. Same remedy the docs/RESEARCH_RECORD.md applied to its own broken
> pointer, and recorded for the same reason.

---

## 1. The shape of one run

`python update_data.py` is the whole system. Everything else — the
dashboard, the notebooks — reads what it leaves on disk.

```
 1. FETCH      three sources in parallel (Reddit, StockTwits, Reddit live),
               30-second heartbeat so a rate-limited pull never looks frozen
 2. STORE      new posts folded into posts.parquet (external machine only)
 3. REBUILD    raw text -> the small aggregate stores everything else reads
 4. COVERAGE   is the window even readable? reported, never silently patched
 5. ANALYSE    conviction -> signals -> euphoria + onset -> influence board
 5a. PRICES    Bloomberg pull for the approved instrument list
 5b. AI LAYER  the local scanner, then the gateway consumers (section 4)
 6. SAFETY     verify_abstracted: nothing with text reaches a commit
 7. SUMMARY    the run's key facts in one glance
```

**Where the speed came from** (the claim README makes). The old chain
re-derived nine years of signals by re-executing notebooks. The
re-engineering made three changes: every analysis became an importable
module operating on wide daily matrices instead of row loops; the
expensive text pass (keyword and ticker extraction) happens ONCE at
ingestion and is never repeated downstream; and the stores are small
typed parquet rather than re-parsed CSV. A full recompute is seconds
because nothing recomputes text.

## 2. Two machines, one contract

| | External machine | Desk machine |
|---|---|---|
| Holds | `posts.parquet`, the raw `.jsonl.zst` archives | the committed aggregates, prices |
| Can reach | the public internet, the fetchers | Bloomberg, the VPN, the Apollo LLM gateway |
| Runs | fetch, store, rebuild, the scanners | everything, plus the LLM consumers |

The contract between them is `ABSTRACTED_DATA/`: **counts only, never
text**. Section 5 states that rule; `verify_abstracted` enforces it at
every run.

## 3. The fetch budget

### 3.1b Why a budget exists at all

The desk's instruction is a cadence, not a size: run it about twice a
week, and a full refresh must stay under roughly ten minutes.
`PIPELINE_BUDGET_S = 600` is that sentence expressed as a number the
code can enforce. Everything in `src/pipeline_budget.py` exists to spend
that ceiling well rather than to make the run smaller.

### 3.1b-i Running short is a deferral, not data loss

The crawl walks **newest-first**, and a subreddit's watermark advances
only over ground a run FULLY covered. A subreddit that hits its cap
keeps its old watermark, so the next run resumes exactly where this one
stopped, and every deferral is printed. The pipeline never silently
collects less than it claims.

The two ledgers behind this (`pipeline_stage_times.json`,
`reddit_comments_cost.json`) are MACHINE-LOCAL and git-ignored on
purpose: they measure how long each stage takes *on this machine*.
Committing them would plan a laptop's run with a desktop's numbers.

**Nothing in this section touches a signal.** These numbers decide how
much data a run FETCHES, never how anything is scored.

## 4. The AI layer — one gateway, several consumers

`src/ai.py` is the ONLY file in the project that talks to a language
model. Everything else asks it. Self-test: `python -m src.ai --selftest`
(expect Paris).

```
                          src/ai.py
              (Apollo gateway via dimsum_lite; VPN + JFrog,
               so it only connects on the desk machine)
                              |
   +--------------+-----------+-----------+---------------+
   |              |                       |               |
 ai_pulse.py   ai_poll.py       tools/ai_keyword_audit  notebook 10 (retired 2026-08-07; record frozen)
 the words     the poll          the map auditor        sentiment test
```

One **scanner** feeds that layer and needs no gateway at all — it is
plain regex over the archives, so it runs everywhere and always:

* `src/agentic_watch.py` — posts about trading WITH an AI. Research
  only; its record is notebook 09 (retired 2026-08-07, JSON frozen) and `daily_agentic_counts.parquet`,
  deliberately not a dashboard page.

### 4.1 The rule that shapes every AI feature

**Numbers come from the stores; words come from the model.** The LLM is
never asked to produce a statistic. It is handed an evidence pack of
already-computed numbers, and that pack is saved beside the prose so any
sentence can be audited against its inputs.

Two patterns implement it, and any future AI feature should copy one:

* **Evidence pack** (the pulse) — compute first, narrate second, store
  both.
* **Approve-then-apply** (the keyword auditor) — the model proposes into
  a CSV with an empty `approved` column and CANNOT write to the config;
  a human types YES and runs the apply step, which is the only writer.

### 4.2 Degrade, never crash

Off the VPN, `available()` is False and every consumer returns a reason
instead of raising. The pipeline logs it and moves on; the dashboard
shows an honest banner and the last good output. An AI stage has never
been allowed to fail a run.

## 5. The text-free boundary

Raw crowd text goes TO the model and INTO the scanners. What comes back
and gets stored is counts and model-written paraphrases — no verbatim
crowd text, no usernames. The local-only text files
(`data/reference/agentic_samples.jsonl`,
`nb10_sample_texts.jsonl`) are git-ignored and must stay that way.

`verify_abstracted` enforces the boundary at every update. It is not a
lint; it is the reason the aggregates can be committed at all.

## 6. The signal contract

### 6.1 Parameters are FROZEN

Every knob lives in `src/config.py` with its evidence quoted beside it,
and every value belongs to a class recorded in
`docs/RESEARCH_RECORD.md`. The pipeline snapshots signals daily and
never revises them: that forward record is the only true out-of-sample
test this project has.

A live run therefore **never re-selects**. It does not choose a model,
re-fit a threshold, or re-run the walk-forward. Research decides once,
in the notebooks, and the answer is frozen into a stored record. The
reason is not speed — it is that re-fitting on every run makes the
number on screen untraceable: nothing on disk would describe how today's
threshold differs from yesterday's, and a threshold nobody can
reconstruct cannot be defended.

Two doors re-open research, both typed on purpose and never reached by
drift: `python -m analytics.run_analytics --what phases --research`, and
`python update_data.py --full` (a backfill rewrites the history the
thresholds were chosen on, so scoring new history against old thresholds
would be a silent lookahead).

### 6.2 The five invariants

1. **Frozen means frozen.** Any change to scoring code or constants is a
   RE-VALIDATION EVENT: re-run the research pass, compare the stored
   record, re-execute notebooks 00-04 and the presentation pack (11). Nothing ships as live
   flags unless it beats the incumbent under the pre-stated rule.
   The single-state study (notebook 08, retired 2026-08-07 with its
   record frozen) is the worked example of a change that WON the
   display and LOST the flags — that division is by design.
2. **Committed data is text-free.** Section 5. Never weaken it.
3. **The dashboard shows conclusions; the notebooks are the record.**
   Performance claims belong in notebooks with confidence intervals,
   not on screens.
4. **Every constant carries its provenance.** No unexplained numbers,
   ever — `docs/RESEARCH_RECORD.md` has a row for each.
5. **AI writes words, never numbers, and never edits config.** Section
   4.1.

---

*The map of which document holds what — one fact, one home — is in
`README.md` under "Where everything is documented". It is kept there
rather than here because the front door is where someone looks for it.*
