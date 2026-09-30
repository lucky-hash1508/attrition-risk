import pathlib, sys, tempfile, unittest
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attrition import data, features as F, models as M, fairness as FA, causal as C, synth, governance as G, service

RAW = ROOT / "data/raw"
ibm, sau = data.load_ibm(RAW / "ibm_hr.csv"), data.load_saudi_original(RAW / "saudi_original.xlsx")


class TestData(unittest.TestCase):
    def test_ibm_constants_dropped(self):
        for c in data.IBM_DROP:
            self.assertNotIn(c, ibm.columns)
        self.assertEqual(ibm.shape[0], 1470)

    def test_saudi_labels_clean(self):
        self.assertEqual(sau.MonthlySalary.nunique(), 7)
        cols = [c for c in sau.columns if c not in ("ID", "y") and not pd.api.types.is_numeric_dtype(sau[c])]
        self.assertGreater(len(cols), 20)  # guard against a vacuous pass
        for c in cols:
            self.assertFalse(sau[c].str.contains("\xa0|^ | $", regex=True).any(), c)

    def test_row_gap_is_only_attrition_yes(self):
        a = data.audit_saudi(RAW / "saudi_original.xlsx", RAW / "saudi_tree.xlsx", RAW / "saudi_nontree.xlsx")
        self.assertEqual(a["row_gap"], 17); self.assertTrue(a["gap_is_entirely_attrition_yes"])


class TestContract(unittest.TestCase):
    def test_both_map_fully(self):
        F.check_contract(F.ibm_contract(ibm)); F.check_contract(F.saudi_contract(sau))

    def test_same_columns_and_ranges(self):
        a, b = F.ibm_contract(ibm), F.saudi_contract(sau)
        self.assertEqual(list(a.columns), list(b.columns))
        for c in ("job_sat", "env_sat", "wlb", "salary_level"):
            self.assertLessEqual(set(a[c]) | set(b[c]), {0, 1, 2})


class TestModels(unittest.TestCase):
    def test_split_disjoint_and_stratified(self):
        X, y = F.ibm_contract(ibm), ibm.y
        Xtr, Xca, Xte, ytr, yca, yte = M.split3(X, y)
        self.assertFalse(set(Xtr.index) & set(Xte.index)); self.assertFalse(set(Xca.index) & set(Xte.index))
        self.assertAlmostEqual(ytr.mean(), yte.mean(), delta=0.02)

    def test_calibrated_probs_valid_and_beat_prevalence(self):
        X, y = F.ibm_contract(ibm), ibm.y
        Xtr, Xca, Xte, ytr, yca, yte = M.split3(X, y)
        m = M.make_model("logreg", Xtr).fit(Xtr, ytr)
        p = M.PlattCalibrator().fit(m.predict_proba(Xca)[:, 1], yca).predict(m.predict_proba(Xte)[:, 1])
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(M.metrics(yte, p)["pr_auc"], yte.mean() + 0.1)

    def test_scaler_fit_inside_pipeline(self):
        m = M.make_model("logreg", F.ibm_contract(ibm))
        self.assertEqual(m.steps[0][0], "pre")  # preprocessing is part of the estimator -> refit per fold


class TestFairness(unittest.TestCase):
    def test_small_groups_suppressed(self):
        s = pd.Series([0] * 100 + [1] * 5); y = np.r_[np.zeros(95), np.ones(10)]; p = np.random.default_rng(0).random(105)
        t = FA.group_table(s, y, p, 0.5)
        self.assertTrue(t.loc[t.group == 1, "suppressed"].iloc[0]); self.assertFalse(t.loc[t.group == 0, "suppressed"].iloc[0])


class TestCausal(unittest.TestCase):
    def test_aipw_recovers_known_effect_where_naive_fails(self):
        X, t, y, truth = synth.simulate(n=4000, seed=3)
        est = C.aipw(X, t, y)
        self.assertLess(abs(est["aipw_ate"] - truth), 0.04)
        self.assertGreater(abs(est["naive_diff"] - truth), 0.10)

    def test_e_value_at_least_one(self):
        self.assertGreaterEqual(C.e_value(0.1, 0.2), 1.0)


class TestGovernance(unittest.TestCase):
    def setUp(self):
        X, y = F.ibm_contract(ibm), ibm.y
        self.tmp = tempfile.mkdtemp(); self.svc = service.GroupRiskService(X, y, audit_path=f"{self.tmp}/a.jsonl"); self.X = X

    def test_refuses_small_cohort_and_logs(self):
        rows = self.X.head(G.K_MIN - 1).to_dict("records")
        with self.assertRaises(ValueError):
            self.svc.score_group(rows)
        with open(f"{self.tmp}/a.jsonl") as f:
            self.assertIn("refused_small_group", f.read())

    def test_scores_cohort_with_notices(self):
        out = self.svc.score_group(self.X.head(50).to_dict("records"))
        self.assertTrue(0 <= out["mean_risk"] <= 1); self.assertTrue(any("NOT FOR INDIVIDUAL" in n for n in out["notices"]))

    def test_scenario_sign(self):
        s = service.scenario(200, 0.16, 0.5, 0.2, (0.15, 0.25))
        self.assertLess(s["expected_change_in_leavers"], 0)  # removing a harmful lever reduces leavers


if __name__ == "__main__":
    unittest.main()
