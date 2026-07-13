"""Flask REST API for Voyage Analytics models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("VOYAGE_MODEL_DIR", BASE_DIR / "models"))


def _load_artifact(filename: str) -> Any | None:
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


flight_model = _load_artifact("flight_price_model.joblib")
gender_model = _load_artifact("gender_classifier.joblib")
hotel_recommender = _load_artifact("hotel_recommender.joblib")


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _require_json() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _require_fields(payload: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def _flight_frame(payload: dict[str, Any]) -> pd.DataFrame:
    required = ["from", "to", "flightType", "agency", "time", "distance", "date"]
    _require_fields(payload, required)
    try:
        travel_date = pd.to_datetime(payload["date"], errors="raise")
        row = {
            "from": str(payload["from"]),
            "to": str(payload["to"]),
            "flightType": str(payload["flightType"]),
            "agency": str(payload["agency"]),
            "time": float(payload["time"]),
            "distance": float(payload["distance"]),
            "year": int(travel_date.year),
            "month": int(travel_date.month),
            "dayofweek": int(travel_date.dayofweek),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid flight input: {exc}") from exc
    return pd.DataFrame([row])


def _gender_frame(payload: dict[str, Any]) -> pd.DataFrame:
    required = ["name", "company", "age"]
    _require_fields(payload, required)
    try:
        row = {
            "name": str(payload["name"]).strip(),
            "company": str(payload["company"]).strip(),
            "age": float(payload["age"]),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid classification input: {exc}") from exc
    if not row["name"]:
        raise ValueError("name must not be empty.")
    return pd.DataFrame([row])


def _recommend_hotels(state: dict[str, Any], user_code: int, top_k: int) -> list[dict[str, Any]]:
    items = state["items"]
    popularity = np.asarray(state["popularity"], dtype=float)
    user_to_idx = state["user_to_idx"]
    matrix = np.asarray(state["train_matrix"], dtype=float)
    personal_weight = float(state.get("personal_weight", 0.8))

    if user_code in user_to_idx:
        history = matrix[user_to_idx[user_code]]
        normalized = history / history.max() if history.max() > 0 else history
        scores = personal_weight * normalized + (1.0 - personal_weight) * popularity
        cold_start = False
    else:
        history = np.zeros(len(items), dtype=float)
        scores = popularity.copy()
        cold_start = True

    top_k = max(1, min(int(top_k), len(items)))
    top_indices = np.argsort(-scores)[:top_k]
    metadata = state["hotel_metadata"].copy()
    metadata_lookup = metadata.set_index("name").to_dict(orient="index")

    recommendations: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        hotel_name = items[idx]
        details = metadata_lookup.get(hotel_name, {})
        past_bookings = int(history[idx])
        reason = (
            "Frequently booked by this user"
            if past_bookings > 0
            else "Popular among travellers"
        )
        recommendations.append(
            {
                "rank": rank,
                "hotel": hotel_name,
                "place": details.get("place"),
                "median_daily_price": float(details.get("price", 0.0)),
                "historical_bookings": int(details.get("bookings", 0)),
                "user_past_bookings": past_bookings,
                "score": float(scores[idx]),
                "reason": reason,
                "cold_start": cold_start,
            }
        )
    return recommendations


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "voyage-analytics-api",
                "models": {
                    "flight_price": flight_model is not None,
                    "gender_classifier": gender_model is not None,
                    "hotel_recommender": hotel_recommender is not None,
                },
            }
        )

    @app.post("/predict/flight-price")
    def predict_flight_price():
        if flight_model is None:
            return _json_error("Flight-price model artifact is not available.", 503)
        try:
            payload = _require_json()
            features = _flight_frame(payload)
            prediction = float(flight_model.predict(features)[0])
            return jsonify({"predicted_price": round(prediction, 2), "currency": "dataset units"})
        except ValueError as exc:
            return _json_error(str(exc), 400)
        except Exception as exc:
            return _json_error(f"Prediction failed: {exc}", 500)

    @app.post("/predict/gender")
    def predict_gender():
        if gender_model is None:
            return _json_error("Gender-classification model artifact is not available.", 503)
        try:
            payload = _require_json()
            features = _gender_frame(payload)
            prediction = str(gender_model.predict(features)[0])
            response: dict[str, Any] = {"predicted_gender": prediction}
            if hasattr(gender_model, "predict_proba"):
                probabilities = gender_model.predict_proba(features)[0]
                classes = list(gender_model.classes_)
                response["probabilities"] = {
                    str(label): round(float(probability), 4)
                    for label, probability in zip(classes, probabilities)
                }
            response["disclaimer"] = (
                "This is a statistical prediction from historical labels and must not be treated as verified identity."
            )
            return jsonify(response)
        except ValueError as exc:
            return _json_error(str(exc), 400)
        except Exception as exc:
            return _json_error(f"Classification failed: {exc}", 500)

    @app.post("/recommend/hotels")
    def recommend_hotels():
        if hotel_recommender is None:
            return _json_error("Hotel recommender artifact is not available.", 503)
        try:
            payload = _require_json()
            _require_fields(payload, ["userCode"])
            user_code = int(payload["userCode"])
            top_k = int(payload.get("top_k", 3))
            recommendations = _recommend_hotels(hotel_recommender, user_code, top_k)
            return jsonify({"userCode": user_code, "recommendations": recommendations})
        except ValueError as exc:
            return _json_error(str(exc), 400)
        except Exception as exc:
            return _json_error(f"Recommendation failed: {exc}", 500)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
