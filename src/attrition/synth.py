"""Known-answer validation: simulate confounded data with a KNOWN true effect, then check the estimator."""
from __future__ import annotations
import numpy as np, pandas as pd
from .causal import aipw


def simulate(n=3000, seed=0, effect=-0.10):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])
    e = 1 / (1 + np.exp(-(0.9 * X.x0 + 0.6 * X.x1)))            # confounded assignment
    t = rng.binomial(1, e)
    p0 = 1 / (1 + np.exp(-(-1.0 + 0.9 * X.x0 + 0.7 * X.x1 - 0.5 * X.x2)))
    p1 = np.clip(p0 + effect, 0.01, 0.99)                      # true individual effect = `effect` (clipped)
    y = rng.binomial(1, np.where(t == 1, p1, p0))
    return X, t, y, float((p1 - p0).mean())


def run(reps=12, n=3000):
    rows = []
    for r in range(reps):
        X, t, y, truth = simulate(n, seed=r)
        est = aipw(X, t, y, seed=r)
        rows.append({"truth": truth, "naive": est["naive_diff"], "aipw": est["aipw_ate"],
                     "covered": est["ci95"][0] <= truth <= est["ci95"][1]})
    d = pd.DataFrame(rows)
    return {"reps": reps, "n": n, "true_ate": float(d.truth.mean()),
            "naive_mean": float(d.naive.mean()), "naive_bias": float((d.naive - d.truth).mean()),
            "aipw_mean": float(d.aipw.mean()), "aipw_bias": float((d.aipw - d.truth).mean()),
            "aipw_ci_coverage": float(d.covered.mean())}
