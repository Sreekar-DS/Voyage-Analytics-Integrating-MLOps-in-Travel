"""Train and persist the Voyage Analytics hotel recommendation state."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PERSONAL_WEIGHT = 0.8


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate hotel booking data."""
    df = pd.read_csv(path)
    required = {"travelCode", "userCode", "name", "place", "days", "price", "total", "date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The hotel dataset is empty.")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="raise")
    return df


def build_state(df: pd.DataFrame) -> dict:
    """Create the user-item matrix, popularity prior, and hotel metadata."""
    items = sorted(df["name"].unique())
    users = sorted(df["userCode"].unique())
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    user_to_idx = {user: idx for idx, user in enumerate(users)}

    matrix = np.zeros((len(users), len(items)), dtype=float)
    for row in df.itertuples():
        matrix[user_to_idx[row.userCode], item_to_idx[row.name]] += 1.0

    popularity = matrix.sum(axis=0)
    popularity = popularity / popularity.max()
    metadata = (
        df.groupby("name")
        .agg(place=("place", "first"), price=("price", "median"), bookings=("name", "size"))
        .reset_index()
    )

    return {
        "items": items,
        "item_to_idx": item_to_idx,
        "user_to_idx": user_to_idx,
        "train_matrix": matrix,
        "popularity": popularity,
        "hotel_metadata": metadata,
        "personal_weight": PERSONAL_WEIGHT,
    }


def recommend(state: dict, user_code: int, top_k: int = 3) -> pd.DataFrame:
    """Return top-k recommendations for a known or cold-start user."""
    items = state["items"]
    popularity = state["popularity"]
    user_to_idx = state["user_to_idx"]
    matrix = state["train_matrix"]
    personal_weight = float(state.get("personal_weight", PERSONAL_WEIGHT))

    if user_code in user_to_idx:
        history = matrix[user_to_idx[user_code]]
        normalized = history / history.max() if history.max() > 0 else history
        scores = personal_weight * normalized + (1.0 - personal_weight) * popularity
    else:
        history = np.zeros(len(items), dtype=float)
        scores = popularity.copy()

    top_indices = np.argsort(-scores)[:top_k]
    result = pd.DataFrame({
        "name": [items[i] for i in top_indices],
        "score": [float(scores[i]) for i in top_indices],
        "past_bookings": [int(history[i]) for i in top_indices],
    })
    return result.merge(state["hotel_metadata"], on="name", how="left")


def train(data_path: Path, model_path: Path) -> dict:
    """Build and persist recommendation state."""
    df = load_data(data_path)
    state = build_state(df)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, model_path, compress=3)
    return {
        "users": len(state["user_to_idx"]),
        "hotels": len(state["items"]),
        "bookings": len(df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/hotels.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/hotel_recommender.joblib"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = train(args.data, args.model_out)
    print("Training complete")
    for name, value in summary.items():
        print(f"{name}: {value}")
