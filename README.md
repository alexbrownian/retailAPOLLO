# retailAPOLLO

**To refresh everything:** open a terminal in this folder and run

    python Code/update_data.py

That one command fetches the week's posts, screens bots, folds the new
days into the tables, re-scores every name, pulls prices, stages the
dashboard bundle and writes a log to `Reports/logs/`. It takes about ten
minutes. Then commit and push (`git add -A`, `git commit`, `git push`)
and the hosted dashboard updates itself.

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
| `Reference Materials/` | `RUNBOOK.md` (how to run, publish and maintain it), `research.ipynb` (how it works and how it was validated) and `ice_source_review.ipynb` (a vendor data source, assessed). |
| `Presentations/` | Decks and demo material. |

First-time setup, scheduling, recovery and every other command are in
`Reference Materials/RUNBOOK.md`.
