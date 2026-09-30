"""Subgroup audit: per-group AUC, error rates, calibration-in-the-large, plus threshold trade-offs."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

MIN_GROUP = 30  # below this, group metrics are suppressed (too noisy / re-identification risk)


def flag_threshold(p, flag_rate):
    return float(np.quantile(p, 1 - flag_rate))


def group_table(df_attr: pd.Series, y, p, thr) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for g in sorted(df_attr.unique(), key=str):
        m = (df_attr == g).values
        n = int(m.sum())
        if n < MIN_GROUP:
            rows.append({"group": g, "n": n, "suppressed": True}); continue
        yy, pp = y[m], p[m]
        flag = pp >= thr
        pos, neg = yy == 1, yy == 0
        rows.append({
            "group": g, "n": n, "suppressed": False, "prevalence": yy.mean(),
            "auc": roc_auc_score(yy, pp) if 0 < yy.sum() < n else np.nan,
            "selection_rate": flag.mean(),
            "tpr": flag[pos].mean() if pos.any() else np.nan,
            "fpr": flag[neg].mean() if neg.any() else np.nan,
            "mean_pred": pp.mean(), "calib_gap": pp.mean() - yy.mean()})
    return pd.DataFrame(rows)


def gaps(tab: pd.DataFrame) -> dict:
    t = tab[~tab.suppressed]
    return {k: float(t[k].max() - t[k].min()) for k in ["auc", "tpr", "fpr", "selection_rate"] if k in t}


def group_thresholds(attr_cal: pd.Series, y_cal, p_cal, target_tpr):
    """Per-group thresholds chosen on the CALIBRATION set so each group reaches ~target TPR."""
    y_cal, p_cal, th = np.asarray(y_cal), np.asarray(p_cal), {}
    for g in attr_cal.unique():
        m = ((attr_cal == g).values) & (y_cal == 1)
        th[g] = float(np.quantile(p_cal[m], 1 - target_tpr)) if m.sum() >= 10 else np.nan
    return th


def apply_group_thresholds(attr, p, th, fallback):
    t = attr.map(lambda g: th.get(g, np.nan)).fillna(fallback).values
    return np.asarray(p) >= t


def tradeoff(y, flag_single, flag_group) -> dict:
    y = np.asarray(y)
    f = lambda fl: {"flag_rate": float(fl.mean()), "precision": float(y[fl].mean()) if fl.any() else np.nan,
                    "recall": float(fl[y == 1].mean())}
    return {"single_threshold": f(flag_single), "group_thresholds": f(flag_group)}
