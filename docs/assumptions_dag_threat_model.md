# Assumed DAG, assumptions, threat model, governance recommendations

## Assumed DAG (overtime example)
```
 Age, Gender, Marital, Education, Field, Dept, JobRole, JobLevel, Income, Experience, Distance, Travel  ──► Overtime ──► Attrition
                                        └──────────────────────────────────────────────────────────────────────────────►┘
 Satisfaction, WorkLifeBalance = MEDIATORS / consequences of overtime  →  deliberately NOT adjusted for
```
Adjusting for satisfaction would block part of the overtime effect (over-control). MonthlyIncome is treated as pre-treatment for overtime,
which is arguable (overtime pay). Stock options depend strongly on marital status/job level (overlap fails → estimate unusable).
Assumptions: (1) no unmeasured confounding given listed covariates; (2) positivity (checked via propensity range/share extreme); (3) consistency;
(4) Saudi covariates and outcome measured retrospectively — assumption (1) is weaker there. Sensitivity: E-value (how strong a hidden confounder must be).

## Threat / misuse model
| Threat | Mitigation in this repo | Residual risk |
|---|---|---|
| Score used to fire/deny promotion to an individual | API refuses cohorts < 10; notice on every response; audit log | HR can still misuse group output |
| Re-identification via tiny groups | Group metrics suppressed < 30 (fairness) and < 10 (service) | Differencing attacks across queries |
| Fictional data mistaken for evidence | Notices in API + dashboard + model card | Screenshots without notices |
| Proxy discrimination | Subgroup audit, ablation with/without protected attributes | Proxies remain (shown) |
| Distribution shift | Transfer experiment shows failure; no cross-org use | No live drift monitor implemented |
| Tampered data/model | Checksums for raw data | No signing/authN on API (not implemented) |

## Governance recommendations
1. Use only for group-level planning; require a documented purpose for each analysis.
2. Re-validate on the organisation's own data before use; do not reuse IBM- or Saudi-trained models elsewhere (AUC 0.55–0.59 when transferred).
3. Treat causal estimates as hypotheses to test with a pilot (e.g. reduce overtime for a randomly chosen team), not as proof.
4. Review subgroup gaps each release; report both with/without protected attributes.
5. Suggested go/no-go thresholds (for the team to agree, not validated): no-go if calibration error on local data > 0.10, any lever with failed overlap, or any use case naming individuals.
