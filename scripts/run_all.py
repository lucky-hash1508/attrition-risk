"""Run the whole pipeline end to end and write results/results.json + figures."""
import json, sys, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attrition import data, features as F, models as M, fairness as FA, causal as C, synth
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RAW, OUT = ROOT / "data/raw", ROOT / "results"
OUT.mkdir(exist_ok=True)
R = {}

# ---------- 1. data audit ----------
ibm = data.load_ibm(RAW / "ibm_hr.csv")
sau = data.load_saudi_original(RAW / "saudi_original.xlsx")
R["audit_saudi"] = data.audit_saudi(RAW / "saudi_original.xlsx", RAW / "saudi_tree.xlsx", RAW / "saudi_nontree.xlsx")
R["audit_ibm"] = {"shape": list(ibm.shape), "prevalence": float(ibm.y.mean()), "dropped_cols": data.IBM_DROP}
# leakage check on the pre-processed Saudi file: JobTitle is a target-style encoding (not a category)
tree = pd.read_excel(RAW / "saudi_tree.xlsx")
enc_auc = roc_auc_score(tree.Attrition, tree.JobTitle)
from sklearn.model_selection import StratifiedKFold
_te = np.zeros(len(sau))
for _tr, _va in StratifiedKFold(5, shuffle=True, random_state=0).split(sau, sau.y):
    _m = sau.iloc[_tr].groupby("JobTitle").y.agg(["sum", "count"])
    _enc = (_m["sum"] + 5 * sau.y.iloc[_tr].mean()) / (_m["count"] + 5)          # smoothed, computed on TRAIN folds only
    _te[_va] = sau.JobTitle.iloc[_va].map(_enc).fillna(sau.y.iloc[_tr].mean())
R["leakage_saudi_tree_jobtitle"] = {"distinct_values": int(tree.JobTitle.nunique()),
                                    "auc_of_encoded_value_alone": float(max(enc_auc, 1 - enc_auc)),
                                    "auc_of_honest_cv_title_encoding_on_raw": float(roc_auc_score(sau.y, _te))}

Xi, yi = F.ibm_contract(ibm), ibm.y
Xs, ys = F.saudi_contract(sau), sau.y
F.check_contract(Xi); F.check_contract(Xs)

# ---------- 2. models: baseline vs advanced, calibrated ----------
def evaluate(X, y, name, seed=42):
    Xtr, Xca, Xte, ytr, yca, yte = M.split3(X, y, seed)
    assert not (set(Xtr.index) & set(Xte.index)) and not (set(Xca.index) & set(Xte.index))  # leakage check
    res = {}
    for kind in ("logreg", "gbm"):
        m = M.make_model(kind, Xtr).fit(Xtr, ytr)
        raw = m.predict_proba(Xte)[:, 1]
        cal = M.PlattCalibrator().fit(m.predict_proba(Xca)[:, 1], yca)
        p = cal.predict(raw)
        res[kind] = {"raw": M.metrics(yte, raw), "calibrated": M.metrics(yte, p),
                     "pr_auc_ci95": M.boot_ci(yte, p)}
    res["n_train_cal_test"] = [len(ytr), len(yca), len(yte)]
    return res

full_ibm = ibm.drop(columns="y")
R["models"] = {
    "ibm_full_features": evaluate(full_ibm, yi, "ibm_full"),
    "ibm_contract": evaluate(Xi, yi, "ibm_contract"),
    "saudi_full_features": evaluate(sau.drop(columns=["y", "ID"]), ys, "saudi_full"),
    "saudi_contract": evaluate(Xs, ys, "saudi_contract"),
}

# explanation stability + permutation importance (contract features)
def importance(X, y):
    Xtr, Xca, Xte, ytr, yca, yte = M.split3(X, y)
    m = M.make_model("gbm", Xtr).fit(Xtr, ytr)
    return M.perm_importance(m, Xte, yte), M.importance_stability("gbm", X, y)
imp_i, stab_i = importance(Xi, yi); imp_s, stab_s = importance(Xs, ys)
R["importance"] = {"ibm": imp_i.round(4).to_dict(), "saudi": imp_s.round(4).to_dict(),
                   "stability_spearman_ibm": stab_i, "stability_spearman_saudi": stab_s}


# ---------- 2b. Saudi retrospective / outcome-defined feature check ----------
_full = sau.drop(columns=["y", "ID"])
_noh = _full.drop(columns=["Health_Issues"])
_ct = pd.crosstab(sau.Health_Issues, sau.y, normalize="index")
R["saudi_retrospective_check"] = {
    "attrition_rate_health_issues_yes": float(sau.y[sau.Health_Issues == "Yes"].mean()),
    "attrition_rate_health_issues_no": float(sau.y[sau.Health_Issues == "No"].mean()),
    "n_health_yes": int((sau.Health_Issues == "Yes").sum()),
    "gbm_full_with_health": M.metrics(ys, M.oof_predict("gbm", _full, ys, seeds=(0,)))["roc_auc"],
    "gbm_full_without_health": M.metrics(ys, M.oof_predict("gbm", _noh, ys, seeds=(0,)))["roc_auc"],
    "gbm_contract_no_tenure_no_exp": M.metrics(ys, M.oof_predict("gbm", Xs.drop(columns=["exp_band", "tenure_band"]), ys, seeds=(0,)))["roc_auc"],
}

# ---------- 3. fairness (out-of-fold predictions, contract features) ----------
def fairness_block(X, y, tag, flag_rate):
    out = {}
    for kind in ("logreg", "gbm"):
        p = M.oof_predict(kind, X, y)
        thr = FA.flag_threshold(p, flag_rate)
        blk = {"overall": M.metrics(y, p)}
        for attr in F.PROTECTED:
            tab = FA.group_table(X[attr], y, p, thr)
            blk[attr] = {"table": tab.round(3).to_dict("records"), "gaps": FA.gaps(tab)}
        out[kind] = blk
    # with vs without protected attributes
    keep = [c for c in F.CONTRACT if c not in F.PROTECTED]
    p_full = M.oof_predict("logreg", X, y); p_blind = M.oof_predict("logreg", X[keep], y)
    thr_f, thr_b = FA.flag_threshold(p_full, flag_rate), FA.flag_threshold(p_blind, flag_rate)
    R_ = {"with_protected": {"auc": float(roc_auc_score(y, p_full))},
          "without_protected": {"auc": float(roc_auc_score(y, p_blind))}}
    for attr in F.PROTECTED:
        R_["with_protected"][attr] = FA.gaps(FA.group_table(X[attr], y, p_full, thr_f))
        R_["without_protected"][attr] = FA.gaps(FA.group_table(X[attr], y, p_blind, thr_b))
    out["protected_attribute_ablation"] = R_
    # threshold trade-off: equalise TPR across gender using group thresholds fit on a calibration half
    idx = np.arange(len(y)); rng = np.random.default_rng(1); rng.shuffle(idx)
    cal, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    pp = pd.Series(p_full, index=X.index)
    thr_cal = FA.flag_threshold(pp.iloc[cal].values, flag_rate)
    ycal = y.iloc[cal].values; pcal = pp.iloc[cal].values
    target = float((pcal[ycal == 1] >= thr_cal).mean())          # SAME recall as the single-threshold policy on calibration data
    th = FA.group_thresholds(X.gender.iloc[cal], y.iloc[cal], pp.iloc[cal], target_tpr=target)
    single = pp.iloc[te].values >= thr_cal
    grp = FA.apply_group_thresholds(X.gender.iloc[te], pp.iloc[te], th, thr_cal)
    out["threshold_tradeoff_gender"] = FA.tradeoff(y.iloc[te].values, single, grp)
    out["threshold_tradeoff_gender"]["target_tpr_matched_to_single"] = target
    out["threshold_tradeoff_gender"]["group_thresholds"]["thresholds"] = {str(k): v for k, v in th.items()}
    for nm, fl in (("single_threshold", single), ("group_thresholds", grp)):
        gt = FA.group_table(X.gender.iloc[te], y.iloc[te], fl.astype(float), 0.5)
        out["threshold_tradeoff_gender"][nm]["tpr_by_gender"] = {str(r.group): float(r.tpr) for r in gt.itertuples()}
    out["flag_rate"] = flag_rate
    return out

R["fairness_ibm"] = fairness_block(Xi, yi, "ibm", 0.20)
R["fairness_saudi"] = fairness_block(Xs, ys, "saudi", 0.45)

# ---------- 4. causal levers ----------
ibm_conf = ["Age", "Gender", "MaritalStatus", "Education", "EducationField", "Department", "JobRole",
            "JobLevel", "DistanceFromHome", "NumCompaniesWorked", "TotalWorkingYears", "BusinessTravel", "MonthlyIncome"]
def lever(df, t, conf, y):
    return C.aipw(df[conf], t, y)
R["causal"] = {"ibm": {
    "overtime": lever(ibm, (ibm.OverTime == "Yes").astype(int), ibm_conf, ibm.y),
    "stock_options": lever(ibm, (ibm.StockOptionLevel > 0).astype(int), [c for c in ibm_conf if c != "MonthlyIncome"] + ["MonthlyIncome"], ibm.y),
    "training_3plus": lever(ibm, (ibm.TrainingTimesLastYear >= 3).astype(int), ibm_conf, ibm.y)}}
sau_conf = ["Gender", "Age", "Maritalstatus", "Academic_degree", "Years_Experience", "Years_experience_lastorganization",
            "Sector", "MonthlySalary", "Allowances", "Distance_to_work"]
R["causal"]["saudi"] = {"overtime": lever(sau, (sau.OverTime == "Yes").astype(int), sau_conf, sau.y),
                        "recognition": lever(sau, (sau.Recognition == "Yes").astype(int), sau_conf, sau.y)}
R["causal"]["synthetic_known_answer"] = synth.run()

# ---------- 5. cross-dataset transfer (failure-mode experiment) ----------
def transfer(Xa, ya, Xb, yb):
    m = M.make_model("logreg", Xa).fit(Xa, ya); p = m.predict_proba(Xb)[:, 1]
    inb = M.oof_predict("logreg", Xb, yb, seeds=(0,))
    return {"transfer_auc": float(roc_auc_score(yb, p)), "transfer_pr_auc": float(average_precision_score(yb, p)),
            "target_prevalence": float(yb.mean()), "mean_predicted_risk": float(p.mean()),
            "in_domain_auc": float(roc_auc_score(yb, inb)), "in_domain_pr_auc": float(average_precision_score(yb, inb)),
            "ece_transfer": M.ece(yb, p)}
R["transfer"] = {"ibm_to_saudi": transfer(Xi, yi, Xs, ys), "saudi_to_ibm": transfer(Xs, ys, Xi, yi)}

# ---------- 6. robustness: label noise ----------
rob = {}
Xtr, Xca, Xte, ytr, yca, yte = M.split3(Xi, yi)
for noise in (0.0, 0.05, 0.10, 0.20):
    rng = np.random.default_rng(0); flip = rng.random(len(ytr)) < noise
    yn = ytr.copy(); yn[flip] = 1 - yn[flip]
    m = M.make_model("gbm", Xtr).fit(Xtr, yn)
    rob[str(noise)] = M.metrics(yte, m.predict_proba(Xte)[:, 1])["pr_auc"]
R["robustness_label_noise_pr_auc_ibm"] = rob

json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)

# ---------- figures ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (X, y, nm) in zip(ax, ((Xi, yi, "IBM (fictional)"), (Xs, ys, "Saudi (survey)"))):
    for kind, ls in (("logreg", "-"), ("gbm", "--")):
        p = M.oof_predict(kind, X, y, seeds=(0,)); q = pd.qcut(p, 8, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y.values}).groupby(q, observed=True).mean()
        a.plot(g.p, g.y, marker="o", ls=ls, label=kind)
    a.plot([0, 1], [0, 1], "k:", lw=1); a.set_title(nm); a.set_xlabel("predicted"); a.set_ylabel("observed"); a.legend()
plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.5))
lv = R["causal"]["ibm"]; names = list(lv)
ate = [lv[k]["aipw_ate"] for k in names]; nv = [lv[k]["naive_diff"] for k in names]
err = [[lv[k]["aipw_ate"] - lv[k]["ci95"][0] for k in names], [lv[k]["ci95"][1] - lv[k]["aipw_ate"] for k in names]]
x = np.arange(len(names)); ax.bar(x - .2, nv, .4, label="naive difference"); ax.bar(x + .2, ate, .4, yerr=err, label="AIPW (95% CI)")
ax.set_xticks(x); ax.set_xticklabels(names); ax.axhline(0, c="k", lw=.5); ax.set_ylabel("effect on attrition probability"); ax.legend()
ax.set_title("IBM (fictional): naive vs doubly-robust"); plt.tight_layout(); plt.savefig(OUT / "causal_levers.png", dpi=120); plt.close()
print("done")
