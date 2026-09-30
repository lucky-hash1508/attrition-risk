"""Shared feature contract: only concepts that exist in BOTH datasets, on one common scale.
Ordinal codes are RELATIVE (0=low ... 2=high) so IBM numeric scores and Saudi labels line up."""
from __future__ import annotations
import pandas as pd

CONTRACT = ["age_band", "gender", "marital", "overtime", "salary_level", "job_sat",
            "env_sat", "wlb", "exp_band", "tenure_band", "travel"]
PROTECTED = ["gender", "age_band", "marital"]
CATEGORICAL = ["marital"]


def _yrs_band(s: pd.Series) -> pd.Series:
    return pd.cut(s, [-1, 4, 10, 100], labels=[0, 1, 2]).astype(int)


def ibm_contract(df: pd.DataFrame) -> pd.DataFrame:
    sat = lambda s: s.map({1: 0, 2: 0, 3: 1, 4: 2})
    out = pd.DataFrame({
        "age_band": pd.cut(df.Age, [0, 30, 40, 50, 100], labels=[0, 1, 2, 3]).astype(int),
        "gender": (df.Gender == "Male").astype(int),
        "marital": df.MaritalStatus,
        "overtime": (df.OverTime == "Yes").astype(int),
        "salary_level": pd.qcut(df.MonthlyIncome, 3, labels=[0, 1, 2]).astype(int),
        "job_sat": sat(df.JobSatisfaction),
        "env_sat": sat(df.EnvironmentSatisfaction),
        "wlb": df.WorkLifeBalance.map({1: 0, 2: 1, 3: 1, 4: 2}),
        "exp_band": _yrs_band(df.TotalWorkingYears),
        "tenure_band": _yrs_band(df.YearsAtCompany),
        "travel": df.BusinessTravel.map({"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}),
    })
    return out


def saudi_contract(df: pd.DataFrame) -> pd.DataFrame:
    yrs = lambda s: s.map(lambda v: 0 if v.startswith("Less") else (1 if v.startswith("From 5") else 2))
    sal = df.MonthlySalary.map(lambda v: 0 if v.startswith("Less") else (1 if "5000 to 10000" in v else 2))
    out = pd.DataFrame({
        "age_band": df.Age.map({"21 to 30": 0, "31 to 40": 1, "41 to 50": 2, "51 to 60": 3}),
        "gender": (df.Gender == "Male").astype(int),
        "marital": df.Maritalstatus,
        "overtime": (df.OverTime == "Yes").astype(int),
        "salary_level": sal,
        "job_sat": df.Job_Satisfaction.map({"Not satisfied": 0, "Satisfied": 1, "Very satisfied": 2}),
        "env_sat": df.Environment_Satisfaction.map({"Low": 0, "Medium": 1, "High": 2}),
        "wlb": df.Work_Live_Balance.map({"Difficult": 0, "Medium": 1, "Easy": 2}),
        "exp_band": yrs(df.Years_Experience),
        "tenure_band": yrs(df.Years_experience_lastorganization),
        "travel": df.Business_Travel.map({"I do not travel for work": 0, "Travel rarely": 1, "Travel frequently": 2}),
    })
    return out


def check_contract(X: pd.DataFrame) -> None:
    missing = [c for c in CONTRACT if c not in X.columns]
    assert not missing, f"missing contract columns: {missing}"
    assert not X[CONTRACT].isna().any().any(), "NaN in contract features (unmapped label?)"
