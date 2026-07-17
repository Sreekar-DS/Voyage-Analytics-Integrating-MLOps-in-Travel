"""Shared data, model, and inference helpers for the Voyage Analytics app."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("VOYAGE_MODEL_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("VOYAGE_DATA_DIR", BASE_DIR / "data" / "raw"))


@st.cache_resource(show_spinner=False)
def load_artifact(filename: str) -> Any:
    """Load and cache a required Joblib artifact."""
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact was not found: {path}. "
            "Add the trained artifact to the models directory."
        )
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_csv(filename: str, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load and cache a project CSV, parsing requested date columns."""
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)
    for column in date_columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


def build_flight_frame(
    origin: str,
    destination: str,
    flight_type: str,
    agency: str,
    duration: float,
    distance: float,
    travel_date: Any,
) -> pd.DataFrame:
    """Create the exact feature frame expected by the regression pipeline."""
    parsed_date = pd.to_datetime(travel_date, errors="raise")
    return pd.DataFrame(
        [
            {
                "from": str(origin),
                "to": str(destination),
                "flightType": str(flight_type),
                "agency": str(agency),
                "time": float(duration),
                "distance": float(distance),
                "year": int(parsed_date.year),
                "month": int(parsed_date.month),
                "dayofweek": int(parsed_date.dayofweek),
            }
        ]
    )


def build_gender_frame(name: str, company: str, age: float) -> pd.DataFrame:
    """Create the exact feature frame expected by the classification pipeline."""
    cleaned_name = str(name).strip()
    cleaned_company = str(company).strip()
    if not cleaned_name:
        raise ValueError("Name must not be empty.")
    if not cleaned_company:
        raise ValueError("Company must not be empty.")
    return pd.DataFrame(
        [{"name": cleaned_name, "company": cleaned_company, "age": float(age)}]
    )


def recommend_hotels(state: dict[str, Any], user_code: int, top_k: int) -> pd.DataFrame:
    """Generate personalized hotel recommendations with a cold-start fallback."""
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

    safe_top_k = max(1, min(int(top_k), len(items)))
    top_indices = np.argsort(-scores)[:safe_top_k]
    recommendations = pd.DataFrame(
        {
            "name": [items[index] for index in top_indices],
            "recommendation_score": [float(scores[index]) for index in top_indices],
            "your_past_bookings": [int(history[index]) for index in top_indices],
        }
    )
    recommendations = recommendations.merge(
        state["hotel_metadata"], on="name", how="left"
    )
    recommendations["reason"] = np.where(
        recommendations["your_past_bookings"] > 0,
        "Matches your repeated booking history",
        "Popular among travellers",
    )
    recommendations.attrs["mode"] = mode
    return recommendations
