"""Train and persist the Voyage Analytics gender-classification pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
FEATURES = ["name", "company", "age"]
KNOWN_LABELS = {"male", "female"}


def load_labeled_users(path: Path) -> pd.DataFrame:
    """Load users and retain records with known binary target labels."""
    df = pd.read_csv(path)
    required = {"code", "company", "name", "gender", "age"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    labeled = df[df["gender"].isin(KNOWN_LABELS)].copy()
    if labeled.empty:
        raise ValueError("No known male/female labels were found.")
    labeled["name"] = labeled["name"].astype(str).str.strip()
    labeled["company"] = labeled["company"].astype(str).str.strip()
    return labeled


def build_pipeline() -> Pipeline:
    """Build text and tabular preprocessing plus an SGD logistic classifier."""
    preprocessor = ColumnTransformer([
        (
            "name_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=2,
                sublinear_tf=True,
            ),
            "name",
        ),
        ("company", OneHotEncoder(handle_unknown="ignore"), ["company"]),
        ("age", StandardScaler(), ["age"]),
    ])
    classifier = SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def train(data_path: Path, model_path: Path) -> dict[str, float]:
    """Train, evaluate, and persist the classification pipeline."""
    df = load_labeled_users(data_path)
    X = df[FEATURES]
    y = df["gender"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/users.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/gender_classifier.joblib"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train(args.data, args.model_out)
    print("Training complete")
    for name, value in results.items():
        print(f"{name}: {value:.6f}")
