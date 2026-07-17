"""Interactive hotel recommendation page."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import load_artifact, load_csv, recommend_hotels  # noqa: E402

st.title("🏨 Personalized Hotel Recommendation")
st.write("Rank hotels from booking history, with a popularity fallback for new users.")

try:
    state = load_artifact("hotel_recommender.joblib")
except Exception as exc:
    st.error(str(exc))
    st.stop()

bookings = load_csv("hotels.csv", date_columns=("date",))
available_users = sorted(int(value) for value in state["user_to_idx"].keys())

with st.sidebar:
    st.header("Recommendation controls")
    mode = st.radio("Profile", ["Existing user", "New user"])
    if mode == "Existing user":
        selected_user = st.selectbox("User code", available_users)
    else:
        selected_user = st.number_input(
            "New user code",
            min_value=0,
            value=max(available_users) + 1,
            step=1,
        )
    top_k = st.slider("Recommendations", 1, 5, 3)

recommendations = recommend_hotels(state, int(selected_user), top_k)
if not bookings.empty and "userCode" in bookings.columns:
    history = bookings[bookings["userCode"] == int(selected_user)].copy()
else:
    history = pd.DataFrame()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Historical bookings", f"{len(history):,}")
k2.metric("Hotels visited", history["name"].nunique() if not history.empty else 0)
k3.metric("Average stay", f"{history['days'].mean():.1f} days" if not history.empty else "N/A")
k4.metric("Average booking value", f"{history['total'].mean():.2f}" if not history.empty else "N/A")

st.subheader("Recommended hotels")
st.info(f"Recommendation mode: **{recommendations.attrs['mode']}**")
for rank, row in recommendations.reset_index(drop=True).iterrows():
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f"### {rank + 1}. {row['name']}")
        c1.write(f"📍 {row.get('place', 'Unknown destination')}")
        c1.caption(row["reason"])
        c2.metric("Median daily price", f"{float(row.get('price', 0.0)):.2f}")
        c3.metric("Score", f"{row['recommendation_score']:.3f}")

st.divider()
if history.empty:
    st.subheader("Cold-start explanation")
    st.write("No booking history is available, so hotels are ranked by overall popularity.")
else:
    left, right = st.columns(2)
    with left:
        counts = history["name"].value_counts().rename_axis("Hotel").reset_index(name="Bookings")
        chart = px.bar(counts, x="Bookings", y="Hotel", orientation="h", title="Hotels booked most often")
        st.plotly_chart(chart, use_container_width=True)
    with right:
        places = history["place"].value_counts().rename_axis("Destination").reset_index(name="Bookings")
        chart = px.pie(places, values="Bookings", names="Destination", title="Destination share")
        st.plotly_chart(chart, use_container_width=True)

    columns = [name for name in ["date", "name", "place", "days", "price", "total"] if name in history.columns]
    st.subheader("Recent bookings")
    st.dataframe(history.sort_values("date", ascending=False)[columns].head(10), hide_index=True, use_container_width=True)

st.caption("Evaluation uses a temporal holdout and reports Hit Rate@3 and MRR@3.")
