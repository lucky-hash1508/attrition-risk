"""Framework-free serving logic (so it is testable without FastAPI). Group-level only, by design."""
from __future__ import annotations
import pandas as pd
from . import features as F, models as M, governance as G


class GroupRiskService:
    def __init__(self, X: pd.DataFrame, y: pd.Series, audit_path="results/audit_log.jsonl"):
        self.model = M.make_model("logreg", X).fit(X, y)
        self.audit_path = audit_path

    def score_group(self, rows: list[dict], who: str = "anonymous") -> dict:
        """Mean predicted risk for a cohort. Refuses cohorts smaller than K_MIN (no individual scores)."""
        if len(rows) < G.K_MIN:
            G.audit_log(self.audit_path, "refused_small_group", {"who": who, "n": len(rows)})
            raise ValueError(f"cohort of {len(rows)} is below the minimum group size {G.K_MIN}")
        X = pd.DataFrame(rows)
        F.check_contract(X)
        p = self.model.predict_proba(X[F.CONTRACT])[:, 1]
        G.audit_log(self.audit_path, "score_group", {"who": who, "n": len(rows)})
        return {"n": len(rows), "mean_risk": float(p.mean()), "share_high_risk": float((p >= 0.5).mean()),
                "notices": G.NOTICES}


def scenario(cohort_size: int, baseline_risk: float, treated_share: float, ate: float, ci: tuple[float, float]) -> dict:
    """What-if for a cohort: shift `treated_share` of people from lever=1 to lever=0 using the AIPW effect.
    ate is the effect of lever=1 vs 0 on attrition probability (positive = lever raises attrition)."""
    d = lambda e: -e * treated_share * cohort_size
    lo, hi = sorted((d(ci[0]), d(ci[1])))
    return {"expected_change_in_leavers": d(ate), "range_95": [lo, hi], "baseline_leavers": baseline_risk * cohort_size,
            "caveat": "Observational estimate; valid only if the assumed DAG holds. See sensitivity (E-value)."}
