"""Streamlit scenario dashboard. NOTE: not executed in the build sandbox (streamlit not installed)."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd, streamlit as st
from attrition import service, governance as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "results/results.json"))
st.set_page_config(page_title="Attrition risk audit", layout="wide")
for n in G.NOTICES:
    st.warning(n)
tab1, tab2, tab3, tab4 = st.tabs(["Models", "Fairness", "Causal levers", "Scenario"])
with tab1:
    rows = [{"dataset": d, "model": m, **{k: round(v, 3) for k, v in r[m]["calibrated"].items()}}
            for d, r in R["models"].items() for m in ("logreg", "gbm")]
    st.dataframe(pd.DataFrame(rows)); st.image(str(ROOT / "results/calibration.png"))
with tab2:
    ds = st.selectbox("Dataset", ["fairness_ibm", "fairness_saudi"]); attr = st.selectbox("Attribute", ["gender", "age_band", "marital"])
    st.dataframe(G.suppress_small(pd.DataFrame(R[ds]["logreg"][attr]["table"])))
with tab3:
    st.image(str(ROOT / "results/causal_levers.png"))
    st.json({k: {"ate": v["aipw_ate"], "ci95": v["ci95"], "share_extreme_propensity": v["share_extreme_prop"]}
             for k, v in R["causal"]["ibm"].items()})
with tab4:
    n = st.slider("Cohort size", 10, 1000, 200); share = st.slider("Share moved off overtime", 0.0, 1.0, 0.3)
    c = R["causal"]["ibm"]["overtime"]
    st.json(service.scenario(n, R["models"]["ibm_contract"]["logreg"]["calibrated"]["prevalence"], share, c["aipw_ate"], tuple(c["ci95"])))
