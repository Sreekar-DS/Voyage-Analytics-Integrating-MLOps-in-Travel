"""Voyage Analytics portfolio application entrypoint."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈️ Voyage Analytics")
st.subheader("End-to-end travel machine learning and MLOps portfolio")
st.write(
    "Explore three production-style machine-learning experiences built from linked "
    "flight, hotel, and user datasets. The application loads the same saved Joblib "
    "artifacts used by the Flask API and Docker deployment."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 🛫 Flight Price Prediction")
        st.write(
            "Estimate a flight price from a validated route, flight type, agency, "
            "duration, distance, and travel date."
        )
        st.metric("Final model", "Random Forest")
        st.metric("Validation RMSE", "0.1562")
        st.page_link(
            "pages/1_Flight_Price_Prediction.py",
            label="Open flight predictor",
            icon="🛫",
        )

with col2:
    with st.container(border=True):
        st.markdown("### 🏨 Hotel Recommendation")
        st.write(
            "Generate personalized hotel rankings from booking history, including "
            "a popularity fallback for unseen users."
        )
        st.metric("Hit Rate@3", "0.391")
        st.metric("MRR@3", "0.236")
        st.page_link(
            "pages/2_Hotel_Recommendation.py",
            label="Open hotel recommender",
            icon="🏨",
        )

with col3:
    with st.container(border=True):
        st.markdown("### 👤 Profile Classification")
        st.write(
            "Demonstrate the text-and-tabular classification pipeline using name, "
            "company, and age inputs."
        )
        st.metric("Weighted F1", "0.833")
        st.metric("Final model", "SGD Classifier")
        st.page_link(
            "pages/3_Gender_Classification.py",
            label="Open classifier",
            icon="👤",
        )

st.divider()

left, right = st.columns([1.25, 1])

with left:
    st.markdown("## MLOps architecture")
    st.code(
        """Data validation and feature engineering
                    ↓
       Model training and evaluation
                    ↓
          Saved Joblib artifacts
             ↙              ↘
      Flask REST API      Streamlit UI
             ↓
           Docker
             ↓
         Kubernetes
       ↙      ↓       ↘
   Airflow  Jenkins  MLflow""",
        language="text",
    )

with right:
    st.markdown("## Project capabilities")
    st.markdown(
        """
- Leakage-aware grouped flight regression
- Text and tabular classification pipeline
- Personalized ranking with cold-start handling
- Flask REST endpoints for all three models
- MLflow experiment and artifact tracking
- Docker, Kubernetes, Airflow, and Jenkins configuration
- GitHub Actions continuous integration
- Public Streamlit portfolio interface
"""
    )

st.info(
    "The exceptionally high tree-based flight-regression score reflects the supplied "
    "dataset's near-deterministic fare combinations. It should not be interpreted as "
    "guaranteed performance on real-time airline pricing markets."
)

st.caption(
    "Built by Tarun Sreekar Parasa · MSc Data Analytics · "
    "Berlin School of Business and Innovation"
)
