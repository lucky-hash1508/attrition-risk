"""Baseline (logistic regression) vs boosted trees, Platt calibration, metrics, permutation importance."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42


def make_model(kind: str, X: pd.DataFrame) -> Pipeline:
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("str")]
    num = [c for c in X.columns if c not in cat]
    if kind == "logreg":
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                                 ("n", StandardScaler(), num)])
        est = LogisticRegression(C=0.5, max_iter=2000, class_weight=None)
    elif kind == "gbm":
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                                 ("n", "passthrough", num)])
        est = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             l2_regularization=1.0, random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", est)])  # preprocessing lives INSIDE the pipeline -> no leakage


def split3(X, y, seed=SEED):
    """Stratified 60/20/20 train / calibration / test. Split happens BEFORE any fitting."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
    return Xtr, Xca, Xte, ytr, yca, yte


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p))[:, 1]


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "prevalence": float(np.mean(y)), "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}


def boot_ci(y, p, fn=average_precision_score, n=300, seed=SEED):
    rng, y, p, vals = np.random.default_rng(seed), np.asarray(y), np.asarray(p), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def oof_predict(kind, X, y, seeds=(0, 1, 2), folds=5):
    """Out-of-fold probabilities (each row scored by a model that never saw it); averaged over seeds."""
    out = np.zeros(len(y))
    for s in seeds:
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=s).split(X, y):
            m = make_model(kind, X.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
            out[te] += m.predict_proba(X.iloc[te])[:, 1] / len(seeds)
    return out


def perm_importance(model, X, y, repeats=20, seed=SEED):
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=repeats, random_state=seed)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def importance_stability(kind, X, y, n_boot=8, seed=SEED):
    """Explanation stability: mean Spearman rank-correlation of permutation-importance across bootstrap refits."""
    rng, ranks = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        Xb, yb = X.iloc[i].reset_index(drop=True), y.iloc[i].reset_index(drop=True)
        m = make_model(kind, Xb).fit(Xb, yb)
        ranks.append(perm_importance(m, Xb, yb, repeats=5).reindex(X.columns))
    R = pd.concat(ranks, axis=1)
    c = R.corr(method="spearman").values
    return float(c[np.triu_indices_from(c, 1)].mean())
