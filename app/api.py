"""FastAPI wrapper. NOTE: written for the Docker image; not executed in the build sandbox (fastapi not installed)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from attrition import data, features as F, service, governance as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
ibm = data.load_ibm(ROOT / "data/raw/ibm_hr.csv")
svc = service.GroupRiskService(F.ibm_contract(ibm), ibm.y, audit_path=str(ROOT / "results/audit_log.jsonl"))
app = FastAPI(title="Attrition risk (group-level)", version="0.1.0")


class Cohort(BaseModel):
    rows: list[dict]
    who: str = "anonymous"


@app.get("/health")
def health():
    return {"status": "ok", "notices": G.NOTICES}


@app.post("/score_group")
def score_group(c: Cohort):
    try:
        return svc.score_group(c.rows, c.who)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
