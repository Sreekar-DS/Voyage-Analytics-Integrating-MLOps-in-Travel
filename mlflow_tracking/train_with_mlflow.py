"""Track flight-price regression experiments with MLflow 3."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
FEATURES = [
    "from", "to", "flightType", "agency", "time", "distance",
    "year", "month", "dayofweek",
]
CATEGORICAL_FEATURES = ["from", "to", "flightType", "agency"]
NUMERICAL_FEATURES = ["time", "distance", "year", "month", "dayofweek"]


def load_features(path: Path):
    df = pd.read_csv(path)
    required = {
        "travelCode", "from", "to", "flightType", "price",
        "time", "distance", "agency", "date",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="raise")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    X = df[FEATURES]
    y = df["price"]
    groups = df["travelCode"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], df


def metric_dict(y_true, prediction):
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
    }


def build_experiments():
    linear_preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERICAL_FEATURES),
    ])
    tree_preprocessor = ColumnTransformer([
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            CATEGORICAL_FEATURES,
        ),
        ("num", "passthrough", NUMERICAL_FEATURES),
    ])

    return [
        (
            "linear_regression",
            Pipeline([
                ("preprocessor", linear_preprocessor),
                ("model", LinearRegression()),
            ]),
            {"model_type": "LinearRegression"},
        ),
        (
            "decision_tree",
            Pipeline([
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(
                        random_state=RANDOM_STATE,
                        min_samples_leaf=2,
                    ),
                ),
            ]),
            {
                "model_type": "DecisionTreeRegressor",
                "min_samples_leaf": 2,
            },
        ),
        (
            "random_forest",
            Pipeline([
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=20,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]),
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 20,
            },
        ),
    ]


def run_experiments(data_path: Path, experiment_name: str):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    X_train, X_test, y_train, y_test, raw_df = load_features(data_path)

    dataset_summary = {
        "rows": int(len(raw_df)),
        "columns": int(raw_df.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "unique_travel_codes": int(raw_df["travelCode"].nunique()),
        "unique_routes": int(raw_df.groupby(["from", "to"]).ngroups),
    }

    results = []
    for run_name, pipeline, parameters in build_experiments():
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(parameters)
            mlflow.log_dict(dataset_summary, "dataset_summary.json")

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            metrics = metric_dict(y_test, predictions)
            mlflow.log_metrics(metrics)

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=f"{run_name}_model",
                input_example=X_train.head(5),
            )

            results.append(
                {
                    "run_name": run_name,
                    "run_id": run.info.run_id,
                    "model_id": getattr(model_info, "model_id", None),
                    **metrics,
                }
            )
            print(run_name, metrics)

    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\nExperiment comparison:")
    print(results_df.to_string(index=False))
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/flights.csv"))
    parser.add_argument(
        "--experiment-name",
        default="voyage-flight-price-regression",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args.data, args.experiment_name)
