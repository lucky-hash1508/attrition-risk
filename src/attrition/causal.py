"""Doubly-robust (AIPW) effect estimation with cross-fitting, overlap diagnostics and an E-value.
Observational data: read every number as 'effect under stated assumptions', not proof."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

EPS = 0.02


def _design(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, drop_first=True).astype(float)


def aipw(X: pd.DataFrame, t, y, folds=5, seed=0) -> dict:
    Z, t, y = _design(X), np.asarray(t).astype(int), np.asarray(y).astype(int)
    n = len(y)
    e, m1, m0 = np.zeros(n), np.zeros(n), np.zeros(n)
    strat = t * 2 + y
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Z, strat):
        ps = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Z.iloc[tr], t[tr])
        e[te] = ps.predict_proba(Z.iloc[te])[:, 1]
        for arm, store in ((1, m1), (0, m0)):
            idx = tr[t[tr] == arm]
            om = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                                l2_regularization=1.0, random_state=seed).fit(Z.iloc[idx], y[idx])
            store[te] = om.predict_proba(Z.iloc[te])[:, 1]
    e = np.clip(e, EPS, 1 - EPS)
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    ate, se = psi.mean(), psi.std(ddof=1) / np.sqrt(n)
    naive = y[t == 1].mean() - y[t == 0].mean()
    # balance: max |standardised mean difference| before / after inverse-propensity weighting
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    def smd(weights):
        d = []
        for c in Z.columns:
            a, b = Z[c].values[t == 1], Z[c].values[t == 0]
            wa, wb = weights[t == 1], weights[t == 0]
            ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
            sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
            d.append(abs(ma - mb) / sd)
        return float(max(d))
    p1 = float(np.clip(y[t == 1].mean(), 1e-6, 1)); p0 = float(np.clip(y[t == 0].mean(), 1e-6, 1))
    return {"n": int(n), "n_treated": int(t.sum()), "naive_diff": float(naive),
            "aipw_ate": float(ate), "ci95": [float(ate - 1.96 * se), float(ate + 1.96 * se)],
            "prop_min": float(e.min()), "prop_max": float(e.max()),
            "share_extreme_prop": float(((e < 0.05) | (e > 0.95)).mean()),
            "max_smd_before": smd(np.ones(n)), "max_smd_after_ipw": smd(w),
            "e_value_point": e_value(ate, y[t == 0].mean())}


def e_value(ate: float, base_risk: float) -> float:
    """E-value for the implied risk ratio: how strong an unmeasured confounder (RR with both T and Y)
    must be to fully explain away the estimate. Uses the harmful direction (RR>=1)."""
    r1 = max(base_risk + ate, 1e-6)
    rr = max(r1 / max(base_risk, 1e-6), max(base_risk, 1e-6) / r1)
    return float(rr + np.sqrt(rr * (rr - 1)))
