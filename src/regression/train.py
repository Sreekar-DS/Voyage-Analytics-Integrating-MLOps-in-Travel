"""Train and persist the Voyage Analytics flight-price regression pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42
FEATURES = [
    "from", "to", "flightType", "agency", "time", "distance",
    "year", "month", "dayofweek",
]
CATEGORICAL_FEATURES = ["from", "to", "flightType", "agency"]
NUMERICAL_FEATURES = ["time", "distance", "year", "month", "dayofweek"]


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the raw flights CSV."""
    df = pd.read_csv(path)
    required = {
        "travelCode", "userCode", "from", "to", "flightType", "price",
        "time", "distance", "agency", "date",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The flights dataset is empty.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features used by the deployed model."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="raise")
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek
    return data


def build_pipeline(n_estimators: int = 20) -> Pipeline:
    """Build the preprocessing and Random Forest regression pipeline."""
    preprocessor = ColumnTransformer([
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            CATEGORICAL_FEATURES,
        ),
        ("num", "passthrough", NUMERICAL_FEATURES),
    ])
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train(data_path: Path, model_path: Path, n_estimators: int = 20) -> dict[str, float]:
    """Train with a travelCode-grouped split, evaluate, and persist the pipeline."""
    df = engineer_features(load_data(data_path))
    X = df[FEATURES]
    y = df["price"]
    groups = df["travelCode"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    pipeline = build_pipeline(n_estimators=n_estimators)
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = pipeline.predict(X.iloc[test_idx])

    metrics = {
        "mae": float(mean_absolute_error(y.iloc[test_idx], predictions)),
        "rmse": float(mean_squared_error(y.iloc[test_idx], predictions) ** 0.5),
        "r2": float(r2_score(y.iloc[test_idx], predictions)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/flights.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/flight_price_model.joblib"))
    parser.add_argument("--n-estimators", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train(args.data, args.model_out, args.n_estimators)
    print("Training complete")
    for name, value in results.items():
        print(f"{name.upper()}: {value:.6f}")
