# retailAPOLLO

Retail-attention and sentiment monitoring for tradeable themes and single
names. Public finance forums are reduced to text-free daily counts and
sentiment, scored against a frozen, walk-forward-validated model, and
shown on a Streamlit dashboard as INCREASE / CUT EXPOSURE calls.

    pip install -r requirements.txt
    python Code/update_data.py                 # fetch, fold, score, pull prices
    python -m streamlit run Code/dashboard.py

| Folder | Contents |
|---|---|
| `Code/` | Everything runnable: the dashboard, the pipeline, its tests and config. |
| `Data/` | Committed aggregates, display bundle and research record; git-ignored runtime stores. |
| `Reports/` | Run logs and the small per-run reports (machine-local). |
| `Reference Materials/` | `RUNBOOK.md` (how to run, publish and maintain it) and `research.ipynb` (how it works and how it was validated). |
| `Presentations/` | Decks and demo material. |
