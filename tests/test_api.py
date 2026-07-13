"""Unit tests for the Voyage Analytics Flask API."""
from __future__ import annotations

import importlib

import numpy as np

api_module = importlib.import_module("api.app")


class DummyFlightModel:
    def predict(self, frame):
        assert set(frame.columns) == {
            "from", "to", "flightType", "agency", "time", "distance",
            "year", "month", "dayofweek",
        }
        return np.array([321.45])


class DummyGenderModel:
    classes_ = np.array(["female", "male"])

    def predict(self, frame):
        return np.array(["female"])

    def predict_proba(self, frame):
        return np.array([[0.8, 0.2]])


def test_health_endpoint():
    client = api_module.create_app().test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["service"] == "voyage-analytics-api"


def test_flight_prediction(monkeypatch):
    monkeypatch.setattr(api_module, "flight_model", DummyFlightModel())
    client = api_module.create_app().test_client()
    response = client.post(
        "/predict/flight-price",
        json={
            "from": "Recife (PE)",
            "to": "Brasilia (DF)",
            "flightType": "economic",
            "agency": "CloudFy",
            "time": 1.76,
            "distance": 1658,
            "date": "2026-07-13",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["predicted_price"] == 321.45


def test_flight_prediction_rejects_missing_fields(monkeypatch):
    monkeypatch.setattr(api_module, "flight_model", DummyFlightModel())
    client = api_module.create_app().test_client()
    response = client.post("/predict/flight-price", json={"from": "A"})
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_gender_prediction(monkeypatch):
    monkeypatch.setattr(api_module, "gender_model", DummyGenderModel())
    client = api_module.create_app().test_client()
    response = client.post(
        "/predict/gender",
        json={"name": "Ana Silva", "company": "Acme", "age": 30},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["predicted_gender"] == "female"
    assert payload["probabilities"]["female"] == 0.8


def test_hotel_recommendation(monkeypatch):
    state = {
        "items": ["Hotel A", "Hotel B", "Hotel C"],
        "user_to_idx": {1: 0},
        "train_matrix": np.array([[3.0, 1.0, 0.0]]),
        "popularity": np.array([1.0, 0.7, 0.4]),
        "personal_weight": 0.8,
        "hotel_metadata": __import__("pandas").DataFrame(
            {
                "name": ["Hotel A", "Hotel B", "Hotel C"],
                "place": ["City A", "City B", "City C"],
                "price": [100.0, 120.0, 90.0],
                "bookings": [10, 8, 4],
            }
        ),
    }
    monkeypatch.setattr(api_module, "hotel_recommender", state)
    client = api_module.create_app().test_client()
    response = client.post("/recommend/hotels", json={"userCode": 1, "top_k": 2})
    assert response.status_code == 200
    recommendations = response.get_json()["recommendations"]
    assert len(recommendations) == 2
    assert recommendations[0]["hotel"] == "Hotel A"
