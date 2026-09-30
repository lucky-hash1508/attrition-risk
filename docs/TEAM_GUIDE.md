# Team guide — who says and understands what (BDS-28)

Pick roles by strength. **Person A = pipeline and model. Person B = insight and product.** Both must be able to explain the shared parts.
Hard rule for the viva: only claim what is in `README.md` under "Verified" — and be ready to say what is *not* implemented.

## Both of you must be able to say (in your own words)
1. **Problem:** group-level attrition analytics with an audit of causal claims and fairness; *not* a tool to judge individuals.
2. **Data honesty:** IBM is fictional, so its results demonstrate methods only. The Saudi survey is real people but retrospective
   (almost every question is about "the last organization you left") and its Attrition label derivation must be confirmed from the paper.
3. **The three headline results:** (a) it does not transfer across datasets (AUC 0.59 / 0.55 vs 0.82 / 0.79 in-domain);
   (b) a naive comparison can have the *wrong sign* (synthetic: +0.077 vs true −0.091) while the doubly-robust estimate recovers it (−0.087);
   (c) removing protected attributes does not remove disparity (proxies).
4. **Limits:** observational data, unmeasured confounding, tiny test positives on IBM (~47), no live deployment.

## Person A — pipeline and model
**Say:** "I audited and cleaned the data myself. The pre-processed Saudi files have 17 fewer rows, all attrition=Yes, and the reason isn't documented.
I split before fitting and kept every transform inside the sklearn Pipeline, so nothing leaks from test data. On IBM the logistic baseline (PR-AUC 0.62)
matched or beat boosting (0.52) — the intervals overlap, so I won't claim boosting is better. Calibration barely changed anything because the models
were already close to calibrated."
**Understand:**
* Stratified split before scaling/encoding; why accuracy is useless at 16% positives → PR-AUC, calibration, Brier, ECE.
* Platt calibration = a logistic regression on the model's logit, fit on a separate split.
* Permutation importance vs SHAP (SHAP not implemented); stability = Spearman of importances across bootstrap refits (0.76 / 0.80).
* Shared feature contract: what was mapped to what (`features.py`), and why salary is a *relative* level.
* Saudi finding: tenure/experience carry the signal (0.86→0.67 AUC without them); plausible cause is retrospective measurement — say "consistent with", not "proves".
* The tree-file `JobTitle` encoding (0.746 vs honest 0.679) — why you didn't use the pre-processed files.
**Still to do (not done here):** MLflow tracking, XGBoost/LightGBM comparison, SHAP — only claim them if you add them.

## Person B — insight and product
**Say:** "I estimate lever effects with a doubly-robust (AIPW) estimator: a propensity model and an outcome model, so it's consistent if either is right.
On IBM, overtime raises attrition probability by about 20 points (CI 16–25) with good overlap. Stock options looked protective, but the overlap check
failed — 46% of propensities were extreme — so I report it as unusable rather than as a finding. I validated the method on simulated data where I knew the answer."
**Understand:**
* The DAG: what is adjusted for, and why satisfaction is *not* (mediator → over-control).
* Overlap/positivity, standardized mean difference before/after weighting, the E-value (how strong a hidden confounder must be to explain the effect away).
* Fairness metrics: per-group AUC, TPR/FPR, calibration gap; why gaps mix base-rate differences with model error; threshold trade-off
  (Saudi: TPR 0.60 vs 0.75 → 0.65 vs 0.67 at about the same overall recall).
* Governance features: cohort ≥ 10, suppression < 30 in fairness tables, notices, audit log.
* **Be upfront:** the API, dashboard, Docker build and CI were written but not executed in the build environment — run and fix them before the demo.

## Likely viva questions
* *Why not one merged dataset?* Schemas differ; merging would hide the transfer failure, which is a finding.
* *Is overtime "causing" attrition?* On IBM, an effect in a fictional generator under stated assumptions; on Saudi it was ≈0 with residual imbalance. Neither is workplace proof.
* *Why 92% CI coverage?* Only 12 simulation reps; it's consistent with 95% but not a precise coverage estimate.
* *Did removing gender/age help fairness?* Mixed: age/marital gaps narrowed, gender TPR gap widened, AUC dropped 0.028.
* *What would you do next?* Randomised pilot for one lever; local re-validation; drift monitoring; add SHAP/MLflow.
