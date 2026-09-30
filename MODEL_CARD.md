# Model card — attrition risk (group-level)

**Intended use:** aggregate, group-level risk views for HR analysts; teaching/demonstration of an audit workflow.
**Out of scope:** any decision about an individual (hiring, firing, promotion, discipline). The service refuses cohorts < 10.

## Data and provenance
* IBM HR: **fictional** (per its Kaggle description). Results demonstrate methods, not workplace facts.
* Saudi survey: self-reported. The questionnaire asks almost every item about *"the last organization you left"*,
  so features are recalled retrospectively (recall bias; features may be measured after the outcome).
  How the `Attrition` label is derived is **not stated in the questionnaire** — confirm from the dataset paper. Cite the Mendeley DOI.
* Cleaning: IBM — dropped 3 constant columns + ID. Saudi — own cleaning of the raw file (non-breaking spaces, stray
  spaces, label variants; 5 columns had variants). Pre-processed files have 17 fewer rows, **all Attrition=Yes** (515→498; No stays 676); cause not documented.
* The pre-processed Saudi files encode `JobTitle` as a numeric score; that score alone gives AUC 0.746 vs 0.679 for an honest
  cross-validated encoding of the raw titles — consistent with (not proof of) target-derived encoding. We do not use those files for modelling.

## Models and performance (held-out 20% test, stratified 60/20/20; Platt calibration on the middle 20%)
| Data / features | Model | ROC-AUC | PR-AUC (prevalence) |
|---|---|---|---|
| IBM, all features | Logistic regression | 0.804 | 0.624 (0.16), 95% CI 0.49–0.74 |
| IBM, all features | Gradient boosting | 0.784 | 0.516, CI 0.38–0.66 |
| IBM, shared contract | LR / GBM | 0.775 / 0.794 | 0.441 / 0.461 |
| Saudi, all features | LR / GBM | 0.884 / 0.904 | 0.876 / 0.900 (0.43) |
| Saudi, shared contract | LR / GBM | 0.859 / 0.859 | 0.872 / 0.875 |
IBM test set has ~47 positives, so intervals are wide: **boosting does not beat the logistic baseline on IBM**. Platt calibration barely
changed ECE (models were already close to calibrated: e.g. IBM GBM 0.045→0.042).

## Known limitations (measured)
* **Does not transfer.** IBM-trained contract model on Saudi: AUC 0.586 (in-domain 0.823), mean predicted risk 0.19 vs actual 0.43, ECE 0.25.
  Saudi→IBM: AUC 0.554 (in-domain 0.788), predicted 0.55 vs actual 0.16.
* **Saudi accuracy rests on experience/tenure.** Dropping `exp_band` and `tenure_band` takes the contract model from 0.86 to 0.67 AUC.
  Consistent with retrospective measurement; not proven. Health-issues item is *not* the driver (AUC 0.898 with vs 0.898 without).
* Label noise: IBM GBM PR-AUC 0.46 (clean) → 0.47 / 0.43 / 0.39 at 5 / 10 / 20% flipped training labels.
* Explanation stability (mean Spearman of permutation importance across bootstrap refits): IBM 0.76, Saudi 0.80.

## Fairness (out-of-fold predictions; gender, age band, marital status; groups < 30 suppressed)
See `results/results.json` → `fairness_*`. Highlights (LR, IBM, top-20% flagged): age-band TPR gap 0.64 (51+ group TPR 0.11, n=143) and AUC gap 0.17;
marital TPR gap 0.37; gender gaps small (AUC 0.02, TPR 0.07). Removing protected attributes lowered AUC 0.790→0.762 and narrowed age/marital
TPR gaps (0.64→0.43, 0.37→0.20) but **widened** the gender TPR gap (0.07→0.13): proxies remain, so "fairness through unawareness" is not a fix.
Saudi (top-45% flagged): female TPR 0.60 vs male 0.75 at one threshold; group thresholds matched to the same overall recall gave 0.65 vs 0.67 with
precision 0.746→0.768 and recall 0.674→0.659. On IBM the gender gap was already within noise and group thresholds only added noise.
Gaps mix real base-rate differences with model error; on IBM they demonstrate the pipeline, not a real employer's behaviour.

## Causal layer (AIPW, cross-fitted; observational; assumptions in `docs/assumptions_dag_threat_model.md`)
| Lever | Data | Effect on attrition prob. (95% CI) | Diagnostics |
|---|---|---|---|
| Overtime | IBM | +0.205 (0.159, 0.251); naive 0.201 | good overlap; E-value 5.4 |
| Stock options >0 | IBM | −0.145 (−0.201, −0.089) | **overlap fails** (46% extreme propensities; balance not achieved) → do not interpret |
| Training ≥3/yr | IBM | −0.039 (−0.076, −0.002) | E-value 1.9 — fragile |
| Overtime | Saudi | −0.007 (−0.079, 0.065) | max SMD after weighting 0.54 — residual imbalance |
| Recognition | Saudi | −0.151 (−0.235, −0.066) | SMD after 0.35; E-value 2.1; retrospective self-report |
Known-answer check (12 simulated datasets, true effect −0.091): naive estimate **+0.077** (wrong sign), AIPW −0.087, CI coverage 92%.
IBM effects are effects *in the data generator*, not evidence about employees.

## Ethical notes
Notices are shown in the API and dashboard; every scoring request and refusal is written to an audit log; cohorts < 10 are refused.
