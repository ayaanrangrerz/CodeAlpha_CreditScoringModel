"""Streamlit interface for the educational Credit Scoring Model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
METADATA_PATH = ROOT / "reports" / "feature_metadata.json"

st.set_page_config(page_title="CreditLens | Credit Risk Scoring", page_icon="📊", layout="wide")

st.markdown("""
<style>
.main {background: #0b1220;}
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stMetric"] {background: #121d30; border: 1px solid #25334a; padding: 16px; border-radius: 14px;}
div[data-testid="stForm"] {background: #111c2e; padding: 20px; border-radius: 16px; border: 1px solid #26364d;}
</style>
""", unsafe_allow_html=True)

st.title("📊 CreditLens")
st.caption("Machine Learning Credit Risk Assessment · CodeAlpha Internship Project")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("Model files are missing. First run `python train.py` in the project folder.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
metadata = json.loads(METADATA_PATH.read_text())

overview, assessment = st.tabs(["Model Overview", "Applicant Assessment"])

with overview:
    st.subheader("Project overview")
    st.write(
        "This educational classification system estimates whether an applicant resembles "
        "the good-risk or bad-risk examples in the historical German Credit dataset."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Training dataset", "1,000 rows")
    c2.metric("Input features", str(len(artifact["feature_columns"])))
    c3.metric("Selected model", artifact["selected_model"])
    st.info(
        "Important: This is an internship demonstration using a small historical dataset. "
        "It is not a real credit bureau score and must not be used to approve or reject real applicants."
    )
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.subheader("Test-set performance")
        chosen = metrics["model_comparison"][artifact["selected_model"]]
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{chosen['accuracy']:.1%}")
        b.metric("Bad-risk recall", f"{chosen['bad_recall']:.1%}")
        c.metric("Bad-risk precision", f"{chosen['bad_precision']:.1%}")
        d.metric("Bad-risk F1", f"{chosen['bad_f1']:.1%}")
        e.metric("ROC-AUC", f"{chosen['roc_auc']:.3f}")
        st.subheader("Model comparison")
        comparison = pd.DataFrame(metrics["model_comparison"]).T
        st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

with assessment:
    st.subheader("Applicant profile")
    st.write("Enter sample applicant information and select **Assess credit risk**.")
    with st.form("credit_assessment_form"):
        values = {}
        cols = st.columns(2)
        for index, feature in enumerate(artifact["feature_columns"]):
            meta = metadata[feature]
            with cols[index % 2]:
                if meta["type"] == "categorical":
                    options = meta["options"]
                    values[feature] = st.selectbox(feature.replace("_", " ").title(), options)
                else:
                    minimum = float(meta["min"])
                    maximum = float(meta["max"])
                    default = min(max(float(meta["median"]), minimum), maximum)
                    step = 1.0 if meta.get("integer") else max((maximum - minimum) / 100, 0.01)
                    values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum, max_value=maximum, value=default, step=step
                    )
        submitted = st.form_submit_button("Assess credit risk", type="primary", use_container_width=True)

    if submitted:
        applicant = pd.DataFrame([values], columns=artifact["feature_columns"])
        predicted = str(pipeline.predict(applicant)[0])
        classes = list(pipeline.named_steps["classifier"].classes_)
        bad_index = classes.index("bad")
        bad_probability = float(pipeline.predict_proba(applicant)[0][bad_index])
        if bad_probability >= 0.65:
            tier, message = "Higher estimated risk", "The model assigns a higher probability to the bad-risk class."
        elif bad_probability >= 0.35:
            tier, message = "Moderate estimated risk", "The model's estimated risk is in the middle band."
        else:
            tier, message = "Lower estimated risk", "The model assigns a lower probability to the bad-risk class."

        st.divider()
        st.subheader("Assessment result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted dataset class", predicted.upper())
        r2.metric("Estimated bad-risk probability", f"{bad_probability:.1%}")
        r3.metric("Illustrative risk band", tier)
        st.progress(bad_probability, text="Estimated probability of bad-risk class")
        st.caption(message)
        st.warning("This prediction is for learning and demonstration only, not a lending decision.")
