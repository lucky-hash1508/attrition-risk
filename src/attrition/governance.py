"""Governance guards used by the API and dashboard: small-group suppression, notices, audit log."""
from __future__ import annotations
import json, time, pathlib
import pandas as pd

K_MIN = 10
NOTICES = [
    "NOT FOR INDIVIDUAL DECISIONS: scores describe group-level risk and must not be used to hire, fire, promote or discipline a person.",
    "IBM HR data is fictional; Saudi data is a self-reported survey. Findings demonstrate a method and are not evidence about any real employer.",
]


def suppress_small(df: pd.DataFrame, count_col="n", k=K_MIN) -> pd.DataFrame:
    out = df.copy()
    small = out[count_col] < k
    for c in out.columns:
        if c not in (count_col, "group") and out[c].dtype != bool:
            out.loc[small, c] = None
    out["suppressed"] = small
    return out


def audit_log(path, action: str, detail: dict):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": action, **detail}) + "\n")
