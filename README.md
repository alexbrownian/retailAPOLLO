# retailAPOLLO

**To refresh everything:** open a terminal in this folder and run

    python Code/update_data.py

That one command fetches the week's posts, screens bots, folds the new
days into the tables, re-scores every name, pulls prices, stages the
dashboard bundle and writes a log to `Reports/logs/`. It takes about ten
minutes. Then commit and push (`git add -A`, `git commit`, `git push`)
and the hosted dashboard updates itself.

**The light version:** `python Code/update_data.py --daily` does the same
run but prices only the theme anchor ETFs (about 27 symbols instead of
267), which is what to use with no Bloomberg Terminal open. Add
`--prices-only` to run just the price step.
[`RUNBOOK.md`](Reference%20Materials/RUNBOOK.md) section 2.4 has the
detail.

**To have it run itself:** `python Code/tools/setup_schedule.py --register`
creates a Windows task that refreshes daily and catches up a start missed
while the machine was off. Every run ends with an ALERT block naming
whatever it raised, and a daily GitHub Actions job checks from outside
the machine that the published data is still advancing and that the run
which published it raised nothing — GitHub notifies the repository owner
when that job fails, so there is nothing to configure and no secret to
hold. GitHub does disable a scheduled workflow on a repository with no
activity for 60 days; the pipeline's daily push resets that clock.
[`RUNBOOK.md`](Reference%20Materials/RUNBOOK.md) section 9.2 is the
reference.

To look at the dashboard locally: `python -m streamlit run dashboard.py`
(the root `dashboard.py` is the entry point the hosted app also uses; it
runs `Code/dashboard.py`).

## What it is

Retail-attention and sentiment monitoring for tradeable themes and single
names. Public finance forums are reduced to text-free daily counts and
sentiment, scored against a frozen, walk-forward-validated model, and
shown on a Streamlit dashboard as INCREASE / CUT EXPOSURE calls.

## Folders

| Folder | Contents |
|---|---|
| `Code/` | Everything runnable: the dashboard, the pipeline, its tests and config. |
| `Data/` | Committed aggregates, display bundle and research record; git-ignored runtime stores. |
| `Reports/` | Run logs and the small per-run reports (machine-local). |
| `Reference Materials/` | [`RUNBOOK.md`](Reference%20Materials/RUNBOOK.md) (how to run, publish and maintain it), [`research.ipynb`](Reference%20Materials/research.ipynb) (how it works and how it was validated) and `ice_source_review.ipynb` (a vendor data source, assessed). |
| `Presentations/` | Decks and demo material. |

## Configuration

Everything meant to be changed without touching Python is CSV in
[`Code/config/`](Code/config/) — the dashboard's title and credit line, which
forums are read, which ETF represents each theme, the keyword and ticker maps,
the bot-screen thresholds and the price provider.
[**`Code/config/README.md`**](Code/config/README.md) is the reference: it has a
"I want to… / edit this file" table and a per-column description of every file
in the folder.

| File | Holds |
|---|---|
| [`Code/config/settings.csv`](Code/config/settings.csv) | The single-value knobs: `key,value,description`. A missing key falls back to the default in `Code/src/settings.py`, so deleting a row is safe. |
| `Code/config/settings.local.csv` | Machine-local overrides of the same keys. Git-ignored, so a workstation can show the pipeline buttons while the hosted copy does not. |
| The other twelve CSVs | The universe and the language: forums, instruments, theme ETFs and constituents, keyword and ticker maps, allow/stop lists, agentic terms, AI poll prompts. |
| [`Code/example.env`](Code/example.env) | Credential template. Copy to `Code/.env` and fill in; `.env` is git-ignored and never leaves the machine. A copy with no credentials still runs — the fetch stages report themselves as off and everything recomputes from the tables on disk. |

After editing anything in `Code/config/`, check it:

    python Code/tools/validate_config.py

Changes take effect on the next pipeline run or dashboard restart.

## The research

[**`research.ipynb`**](Reference%20Materials/research.ipynb) is the read-first
document: what each of the eleven crowd measurements is, how they combine into
one score, where the cuts come from, and the walk-forward record behind them.
It runs top to bottom against the committed tables.

[**`RUNBOOK.md`**](Reference%20Materials/RUNBOOK.md) has first-time setup,
scheduling, publishing, recovery and every other command.
