# Notebook house style — "an explanation paper you can defend"

The seven notebooks in `notebooks/` are the research record for retailAPOLLO.
They are read by three different people and have to work for all three:

- **the desk** — a PM who wants the conclusion and needs to trust it;
- **a reviewer** — someone checking whether a number is real or an artefact;
- **future us** — six months from now, deciding whether a change is safe.

A notebook that is only a script with headings fails all three. This file is
the pattern every notebook follows, and the reason for each part of it.

## The shape of a notebook

**1. The masthead.** Title, then the *one question* the notebook answers,
written as a question. Then a **VERDICT BOX**: a table of question → answer →
where the evidence is. A reader who stops after the verdict box must already
have the honest conclusion, including the parts that did not work.

**2. A DEFINITIONS block, before any result.** Every term the notebook uses,
in desk English, generated from `analytics/plain_english.py` so the wording is
identical in the notebook and on the dashboard. The rule: if a reviewer could
say "define that", it has an entry.

**3. Then sections, each with the same three blocks.**

```
### 4.2  <a heading phrased as the question this section settles>

**WHY THIS**
- what we did not know before this section
- what decision hangs on the answer

**HOW IT WORKS**            (only where a derivation is involved)
- the mechanism, in the order the code does it
- where each number came from: LEARNED / DERIVED / CONVENTION / GROUND TRUTH
  / DESK DECISION — never an unexplained constant

**SO WHAT**
- the finding, in one line, with the number in it
- what changes because of it — a rung adopted, a feature dropped, a limit
  written into the docs
- if the answer is "nothing changes", say that explicitly; a null result that
  is labelled is evidence, a null result that is silent looks like an omission
```

**4. An `IF ASKED` block wherever a reviewer would push.** The awkward
question, stated in their words, answered in ours. These are the ones that
make it defendable: *"why 0.66 and not 0.7?"*, *"isn't that circular?"*,
*"why is a null result worth a section?"*

**5. A closing section**: what ships, what does not, the limitations in plain
sentences, and one line for the PM.

## Rules that apply everywhere

- **Bullets for the WHY and the SO WHAT.** They are the parts people skim, and
  a paragraph does not survive skimming. Prose is right for a derivation, a
  caveat or an argument — anywhere the reasoning has to connect.
- **No unexplained number, anywhere.** Every constant carries its provenance
  class. "We used 0.66" is not admissible; "0.66 is Chan's HIGH-tier cut,
  adopted unchanged so our leaderboard is comparable to hers, and §4 measures
  what happens at 0.60 and 0.72" is.
- **Plain English in every label a human reads.** Import from
  `analytics.plain_english`. Stored column names stay as they are — the
  translation happens at the display layer only.
- **Every chart gets a caption that says what to look at.** A chart with no
  "read it like this" is decoration.
- **State the null results as loudly as the positive ones.** Half of what
  these notebooks establish is that something does *not* work; that is the
  half that stops us shipping it.
- **Delete dead code.** A superseded experiment that still runs is a trap: it
  produces numbers nobody uses and a reader cannot tell which are live. If it
  mattered, its verdict belongs in the prose and its code belongs deleted.

## Verifying a rewrite

Prose edits must not move numbers. Before editing:

```
python tools/nb_codehash.py snapshot notebooks/*.py
```

after editing:

```
python tools/nb_codehash.py check notebooks/*.py
```

`PROSE-ONLY` means the code cells are byte-identical and no result can have
changed. When a rewrite *does* touch code (deleting a dead experiment), the
check is expected to fail for that notebook — re-execute it and confirm the
recorded JSON artefact in `docs/research/` is unchanged apart from the
deliberate deletion.
