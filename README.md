# Employee Attrition Risk — causal factors, fairness and actionability audit (BDS-28)

Group-level attrition analytics on two datasets, built as a pipeline + audit, **not** an individual-decision tool.

| Dataset | Rows | Attrition | Role |
|---|---|---|---|
| IBM HR (Kaggle) — **fictional** | 1,470 | 16% | development set; demonstrates the methods |
| Saudi employee survey (Mendeley) — self-reported, retrospective | 1,191 | 43% | second case study; the only human-sourced data |

## Run
```bash
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v   # 14 tests
PYTHONPATH=src python scripts/run_all.py                  # writes results/results.json + figures
uvicorn app.api:app --port 8000                           # group-level API
streamlit run app/dashboard.py                            # scenario dashboard
docker build -t attrition-risk .
```

## Layout
`src/attrition/` data audit (`data.py`), shared feature contract (`features.py`), models + calibration
(`models.py`), fairness (`fairness.py`), doubly-robust causal engine (`causal.py`), known-answer validation
(`synth.py`), governance guards (`governance.py`), serving logic (`service.py`). `app/` API + dashboard.
`docs/` DAG/assumptions, threat model, governance, team guide. `MODEL_CARD.md`. `results/` generated outputs.

## Verified vs not verified (read this before the viva)
* Verified by running here: data audit, all models/metrics, fairness tables, causal estimates, synthetic
  validation, transfer experiment, 14 unit tests (`unittest`).
* **Written but NOT executed** in the build sandbox (no network, libraries absent): `app/api.py` (FastAPI),
  `app/dashboard.py` (Streamlit), the Dockerfile build, and the CI workflow. Syntax-checked only. Run them yourselves.
* **Not implemented** (sandbox had no xgboost / lightgbm / shap / mlflow / econml / fairlearn): XGBoost/LightGBM
  (scikit-learn `HistGradientBoosting` used instead), SHAP (permutation importance used), MLflow tracking.
  Do not claim these unless you add them.
* Not something code can create: the git history, issues, PRs, code reviews and the 100-hour logs.
  Use `docs/hour_log_template.csv` and do the work in a real repo.
* `data/raw/` is committed only so this bundle runs. In your repo, commit a download script + `CHECKSUMS.txt` instead.
"# attrition-risk" 
Practice change
