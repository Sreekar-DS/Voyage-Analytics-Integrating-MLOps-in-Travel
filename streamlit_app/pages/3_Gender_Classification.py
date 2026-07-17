"""Profile-label classification demonstration page."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import build_gender_frame, load_artifact, load_csv  # noqa: E402

st.title("👤 Profile Classification Demonstration")
st.write(
    "This page demonstrates the saved text-and-tabular classification pipeline using "
    "name, company, and age inputs."
)
st.warning(
    "This output is only a statistical demonstration based on historical labels. It is "
    "not verified identity information and must not be used for high-stakes decisions."
)

try:
    model = load_artifact("gender_classifier.joblib")
except Exception as exc:
    st.error(str(exc))
    st.stop()

users = load_csv("users.csv")
company_suggestions: list[str] = []
if not users.empty and "company" in users.columns:
    company_suggestions = sorted(users["company"].dropna().astype(str).unique())

left, right = st.columns(2)
with left:
    name = st.text_input("Name", value="Ana Silva")
    age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
with right:
    if company_suggestions:
        company = st.selectbox("Company", company_suggestions)
    else:
        company = st.text_input("Company", value="Acme")

if st.button("Run classification demo", type="primary", use_container_width=True):
    try:
        features = build_gender_frame(name=name, company=company, age=age)
        prediction = str(model.predict(features)[0])
    except Exception as exc:
        st.error(f"Classification failed: {exc}")
    else:
        st.success("Classification completed")
        st.metric("Predicted historical label", prediction.title())

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            probability_frame = pd.DataFrame(
                {
                    "Label": [str(value).title() for value in model.classes_],
                    "Probability": [float(value) for value in probabilities],
                }
            ).set_index("Label")
            st.subheader("Model confidence")
            st.bar_chart(probability_frame)
            st.dataframe(
                probability_frame.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
            )

        with st.expander("View model input"):
            st.dataframe(features, hide_index=True, use_container_width=True)

st.divider()
st.markdown("### Model context")
st.write(
    "The training target contains male, female, and none values. The none entries are "
    "treated as unavailable labels, leaving 900 known-label records for supervised "
    "training. The final SGD classifier combines character TF-IDF name features with "
    "company and age information."
)
st.caption("Validated weighted F1 score: approximately 0.833.")
