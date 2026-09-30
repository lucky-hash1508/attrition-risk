"""Loading, cleaning and auditing both datasets. This module is the 'data audit' deliverable."""
from __future__ import annotations
import re
import pandas as pd
from pandas.api.types import is_string_dtype

IBM_DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]  # constants + ID


def _clean_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\xa0", " ", regex=False)
    return s.str.strip().str.replace(r"\s+", " ", regex=True)


def load_ibm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in IBM_DROP if c in df.columns])
    df["y"] = (df["Attrition"] == "Yes").astype(int)
    return df.drop(columns=["Attrition"])


def _fix_col(c: str) -> str:
    c = c.replace("\xa0", " ").strip()
    return re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_")


def load_saudi_original(path: str) -> pd.DataFrame:
    """Own cleaning of the raw survey file: whitespace, non-breaking spaces, label variants."""
    df = pd.read_excel(path)
    df.columns = [_fix_col(c) for c in df.columns]
    for c in df.columns:
        if is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = _clean_text(df[c])
    df["Sector"] = df["Sector"].str.replace("education sector", "Education sector", regex=False)
    df["y"] = (df["Attrition"] == "Yes").astype(int)
    return df.drop(columns=["Attrition"])


def audit_saudi(raw_path: str, tree_path: str, nontree_path: str) -> dict:
    """Machine-readable audit: label variants fixed, duplicates, and the row gap vs pre-processed files."""
    raw = pd.read_excel(raw_path)
    raw.columns = [_fix_col(c) for c in raw.columns]
    dirty = {}
    for c in raw.columns:
        if is_string_dtype(raw[c]) or raw[c].dtype == object:
            before, after = raw[c].nunique(), _clean_text(raw[c]).nunique()
            if before != after:
                dirty[c] = {"distinct_before": int(before), "distinct_after": int(after)}
    clean = load_saudi_original(raw_path)
    tree, nontree = pd.read_excel(tree_path), pd.read_excel(nontree_path)
    gap = len(clean) - len(tree)
    return {
        "raw_shape": list(raw.shape),
        "missing_cells": int(raw.isna().sum().sum()),
        "columns_with_label_variants": dirty,
        "exact_duplicate_rows_excluding_ID": int(clean.drop(columns="ID").duplicated().sum()),
        "attrition_yes_raw": int(clean.y.sum()),
        "attrition_yes_preprocessed": int(tree["Attrition"].sum()),
        "rows_raw": int(len(clean)),
        "rows_preprocessed_tree": int(len(tree)),
        "rows_preprocessed_nontree": int(len(nontree)),
        "row_gap": int(gap),
        "gap_is_entirely_attrition_yes": bool(int(clean.y.sum() - tree["Attrition"].sum()) == gap),
    }
