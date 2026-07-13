"""Interactive Voyage Analytics hotel recommendation dashboard."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("VOYAGE_MODEL_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("VOYAGE_DATA_DIR", BASE_DIR / "data" / "raw"))

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def load_recommender() -> dict:
    path = MODEL_DIR / "hotel_recommender.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/recommendation/train.py before starting Streamlit."
        )
    return joblib.load(path)


@st.cache_data
def load_booking_data() -> pd.DataFrame:
    path = DATA_DIR / "hotels.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


def recommend(state: dict, user_code: int, top_k: int) -> pd.DataFrame:
    items = state["items"]
    popularity = np.asarray(state["popularity"], dtype=float)
    matrix = np.asarray(state["train_matrix"], dtype=float)
    user_to_idx = state["user_to_idx"]
    personal_weight = float(state.get("personal_weight", 0.8))

    if user_code in user_to_idx:
        history = matrix[user_to_idx[user_code]]
        normalized = history / history.max() if history.max() > 0 else history
        scores = personal_weight * normalized + (1.0 - personal_weight) * popularity
        mode = "Personalized from booking history"
    else:
        history = np.zeros(len(items), dtype=float)
        scores = popularity.copy()
        mode = "Cold-start popularity fallback"

    top_indices = np.argsort(-scores)[:top_k]
    recommendations = pd.DataFrame(
        {
            "name": [items[i] for i in top_indices],
            "recommendation_score": [float(scores[i]) for i in top_indices],
            "your_past_bookings": [int(history[i]) for i in top_indices],
        }
    )
    recommendations = recommendations.merge(
        state["hotel_metadata"],
        on="name",
        how="left",
    )
    recommendations["reason"] = np.where(
        recommendations["your_past_bookings"] > 0,
        "Matches your repeated booking history",
        "Popular among travellers",
    )
    recommendations.attrs["mode"] = mode
    return recommendations


st.title("✈️ Voyage Analytics Travel Recommendation Dashboard")
st.caption(
    "Personalized hotel recommendations based on historical booking behavior, "
    "with a popularity fallback for new users."
)

try:
    state = load_recommender()
except Exception as exc:
    st.error(str(exc))
    st.stop()

bookings = load_booking_data()
available_users = sorted(int(user) for user in state["user_to_idx"].keys())

with st.sidebar:
    st.header("Recommendation Controls")
    user_mode = st.radio("User type", ["Existing user", "New / cold-start user"])
    if user_mode == "Existing user":
        selected_user = st.selectbox("Select user code", available_users)
    else:
        selected_user = st.number_input(
            "Enter a new user code",
            min_value=0,
            value=(max(available_users) + 1 if available_users else 9999),
            step=1,
        )
    top_k = st.slider("Number of recommendations", min_value=1, max_value=5, value=3)

recommendations = recommend(state, int(selected_user), top_k)

if not bookings.empty:
    user_history = bookings[bookings["userCode"] == int(selected_user)].copy()
else:
    user_history = pd.DataFrame()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Historical bookings", f"{len(user_history):,}")
kpi2.metric(
    "Hotels visited",
    f"{user_history['name'].nunique() if not user_history.empty else 0}",
)
kpi3.metric(
    "Average stay",
    f"{user_history['days'].mean():.1f} days" if not user_history.empty else "N/A",
)
kpi4.metric(
    "Average booking value",
    f"{user_history['total'].mean():.2f}" if not user_history.empty else "N/A",
)

st.subheader("Recommended Hotels")
st.info(f"Recommendation mode: **{recommendations.attrs['mode']}**")

for rank, row in recommendations.reset_index(drop=True).iterrows():
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.markdown(f"### {rank + 1}. {row['name']}")
        col1.write(f"📍 {row['place']}")
        col1.caption(row["reason"])
        col2.metric("Median daily price", f"{row['price']:.2f}")
        col3.metric("Recommendation score", f"{row['recommendation_score']:.3f}")

st.divider()

if user_history.empty:
    st.subheader("Cold-Start Explanation")
    st.write(
        "This user has no booking history in the training data, so recommendations are ranked "
        "using overall hotel popularity. As new bookings arrive, the recommender can be retrained "
        "to produce personalized results."
    )
else:
    left, right = st.columns(2)

    with left:
        st.subheader("Your Booking History")
        history_counts = (
            user_history["name"].value_counts().rename_axis("Hotel").reset_index(name="Bookings")
        )
        fig = px.bar(
            history_counts,
            x="Bookings",
            y="Hotel",
            orientation="h",
            title="Hotels Booked Most Often",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Destination Preferences")
        destination_counts = (
            user_history["place"].value_counts().rename_axis("Destination").reset_index(name="Bookings")
        )
        fig = px.pie(
            destination_counts,
            values="Bookings",
            names="Destination",
            title="Historical Destination Share",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Bookings")
    display_columns = ["date", "name", "place", "days", "price", "total"]
    st.dataframe(
        user_history.sort_values("date", ascending=False)[display_columns].head(10),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Recommendation evaluation uses a temporal holdout. The selected production strategy "
    "combines personalized booking frequency with a small popularity prior."
)
