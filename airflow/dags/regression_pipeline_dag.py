"""Airflow DAG for the Voyage Analytics regression MLOps workflow."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

try:  # Airflow 3.x public Task SDK
    from airflow.sdk import dag, task
except ImportError:  # Backward-compatible fallback for Airflow 2.x
    from airflow.decorators import dag, task

PROJECT_ROOT = Path(os.getenv("VOYAGE_PROJECT_ROOT", "/opt/airflow/voyage-project"))
DATA_PATH = Path(os.getenv("VOYAGE_FLIGHTS_CSV", PROJECT_ROOT / "data" / "raw" / "flights.csv"))
MODEL_PATH = Path(
    os.getenv(
        "VOYAGE_FLIGHT_MODEL",
        PROJECT_ROOT / "models" / "flight_price_model.joblib",
    )
)


@dag(
    dag_id="voyage_flight_price_regression_pipeline",
    description="Validate data, train the regression model, validate the artifact, and log MLflow experiments.",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    tags=["voyage-analytics", "regression", "mlops"],
)
def voyage_regression_pipeline():
    @task
    def validate_data() -> dict:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Flights dataset not found: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        required = {
            "travelCode", "userCode", "from", "to", "flightType", "price",
            "time", "distance", "agency", "date",
        }
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            raise ValueError("Flights dataset is empty.")
        if df["price"].isna().any():
            raise ValueError("Target column price contains missing values.")

        return {
            "rows": int(len(df)),
            "unique_travel_codes": int(df["travelCode"].nunique()),
            "unique_users": int(df["userCode"].nunique()),
        }

    @task
    def train_regression_model(validation_summary: dict) -> dict:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "regression" / "train.py"),
            "--data",
            str(DATA_PATH),
            "--model-out",
            str(MODEL_PATH),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "validation": validation_summary,
            "model_path": str(MODEL_PATH),
            "training_output": completed.stdout,
        }

    @task
    def validate_model_artifact(training_summary: dict) -> dict:
        model_path = Path(training_summary["model_path"])
        if not model_path.exists() or model_path.stat().st_size == 0:
            raise FileNotFoundError(f"Model artifact was not created correctly: {model_path}")

        model = joblib.load(model_path)
        raw = pd.read_csv(DATA_PATH, nrows=1)
        raw["date"] = pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="raise")
        sample = pd.DataFrame(
            [
                {
                    "from": raw.iloc[0]["from"],
                    "to": raw.iloc[0]["to"],
                    "flightType": raw.iloc[0]["flightType"],
                    "agency": raw.iloc[0]["agency"],
                    "time": raw.iloc[0]["time"],
                    "distance": raw.iloc[0]["distance"],
                    "year": int(raw.iloc[0]["date"].year),
                    "month": int(raw.iloc[0]["date"].month),
                    "dayofweek": int(raw.iloc[0]["date"].dayofweek),
                }
            ]
        )
        prediction = float(model.predict(sample)[0])
        return {
            **training_summary,
            "artifact_size_bytes": int(model_path.stat().st_size),
            "sanity_prediction": prediction,
        }

    @task
    def track_experiments(model_validation: dict) -> dict:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "mlflow_tracking" / "train_with_mlflow.py"),
            "--data",
            str(DATA_PATH),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            **model_validation,
            "mlflow_output": completed.stdout,
        }

    validated = validate_data()
    trained = train_regression_model(validated)
    model_checked = validate_model_artifact(trained)
    track_experiments(model_checked)


voyage_regression_pipeline()
