# Voyage Analytics: Integrating MLOps in Travel

[![Voyage Analytics CI](https://github.com/Sreekar-DS/Voyage-Analytics-Integrating-MLOps-in-Travel/actions/workflows/ci.yml/badge.svg)](https://github.com/Sreekar-DS/Voyage-Analytics-Integrating-MLOps-in-Travel/actions/workflows/ci.yml)
[![Open Live App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://voyage-analytics-tarun.streamlit.app/)

An end-to-end travel analytics and MLOps capstone built from linked user, flight, and hotel datasets.

## Live Application

**Public app:** https://voyage-analytics-tarun.streamlit.app/

The deployed multipage Streamlit portfolio app includes:

- **Flight Price Prediction**
- **Hotel Recommendation**
- **Profile Classification Demonstration**
- **Project and MLOps Overview**

**Deployment coordinates**

| Setting | Value |
|---|---|
| Repository | `Sreekar-DS/Voyage-Analytics-Integrating-MLOps-in-Travel` |
| Branch | `main` |
| Entrypoint | `streamlit_app/app.py` |
| Python | `3.12` |
| Subdomain | `voyage-analytics-tarun` |

See [`docs/STREAMLIT_DEPLOYMENT.md`](docs/STREAMLIT_DEPLOYMENT.md) for the deployment configuration.

## Project Overview

The project solves three travel-related machine-learning problems and connects the primary regression model to a full MLOps lifecycle:

1. **Flight Price Regression** — predicts flight prices from route, flight type, agency, distance, duration, and engineered calendar features.
2. **Profile Classification** — demonstrates text-and-tabular classification from names and supporting profile features while treating raw `none` targets as unavailable labels.
3. **Hotel Recommendation** — ranks hotels from historical booking frequency with a popularity fallback for cold-start users.

The flight-price regression model is the main model used to demonstrate serving, containerization, scaling, scheduling, CI/CD, and experiment tracking.

## Dataset Summary

| Dataset | Rows | Purpose |
|---|---:|---|
| `users.csv` | 1,340 | User profile and classification data |
| `flights.csv` | 271,888 | Flight price regression and travel history |
| `hotels.csv` | 40,552 | Hotel booking history for recommendations |

Initial validation found no missing values or exact duplicate rows in the raw datasets.

## Validated Model Results

### Flight Price Regression

A leakage-aware `GroupShuffleSplit` is performed by `travelCode` so paired records from the same journey do not appear on both sides of the train/test split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ~80.63 | ~102.10 | ~0.921 |
| Decision Tree | ~0.0044 | ~0.5298 | ~0.999998 |
| Random Forest | ~0.0047 | ~0.1562 | ~0.9999998 |

The high tree-based scores reflect the supplied dataset's nearly fixed fare for each exact origin, destination, flight type, and agency combination. The result does not imply guaranteed generalization to unseen dynamic airline pricing.

### Profile Classification

The raw target contains `male`, `female`, and `none`. The `none` value is treated as unavailable target information. Supervised modeling uses 900 known-label records.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Logistic Regression | ~0.811 | ~0.811 |
| Linear SVM | ~0.828 | ~0.827 |
| SGD Classifier | ~0.833 | ~0.833 |

This component is a technical demonstration based on historical labels. Its output is not verified identity information and must not be used for high-stakes decisions.

### Hotel Recommendation

Evaluation uses a temporal holdout: each eligible user's latest booking is held out and ranked from earlier booking history.

| Recommender | Hit Rate@3 | MRR@3 |
|---|---:|---:|
| Global Popularity | ~0.353 | ~0.222 |
| Personalized Frequency | ~0.391 | ~0.236 |
| Tested Hybrid | ~0.366 | ~0.227 |

The personalized-frequency strategy is selected because the catalogue contains nine hotels and user histories are dense.

## MLOps Architecture

```text
Users / Flights / Hotels CSV
            |
            v
     Data Validation
            |
   +--------+--------+
   |        |        |
   v        v        v
Regression  Profile  Hotel
 Model       Model   Recommender
   |          |         |
   +----------+---------+
              |
         Saved Artifacts
        /              \
 Flask REST API     Streamlit App
        |
      Docker
        |
    Kubernetes
   /     |      \
Airflow Jenkins  MLflow
```

## Implemented Components

- Reusable regression, classification, and recommendation training modules
- Flask REST API with health and inference endpoints
- Multipage Streamlit portfolio application
- MLflow experiment tracking with persistent SQLite storage
- Dockerfile for the Flask API
- Kubernetes Deployment, Service, and Horizontal Pod Autoscaler
- Apache Airflow regression workflow
- Jenkins declarative CI/CD pipeline
- GitHub Actions Python-test and Docker-build workflow
- Pytest Flask API tests
- Three AlmaBetter-format modeling notebooks

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── notebooks/
├── data/raw/
├── models/
├── src/
│   ├── regression/train.py
│   ├── classification/train.py
│   └── recommendation/train.py
├── api/app.py
├── streamlit_app/
│   ├── app.py
│   ├── common.py
│   ├── requirements.txt
│   └── pages/
│       ├── 1_Flight_Price_Prediction.py
│       ├── 2_Hotel_Recommendation.py
│       └── 3_Gender_Classification.py
├── mlflow_tracking/train_with_mlflow.py
├── airflow/dags/regression_pipeline_dag.py
├── kubernetes/
├── tests/test_api.py
├── Dockerfile
├── Jenkinsfile
├── requirements.txt
└── README.md
```

## Run the Streamlit App

```bash
python -m streamlit run streamlit_app/app.py
```

The public Streamlit app loads the same three Joblib artifacts used by the Flask API. Flight inputs are constrained to route and service combinations observed in `flights.csv`.

## Flask REST API

```bash
python api/app.py
```

Endpoints:

```text
GET  /health
POST /predict/flight-price
POST /predict/gender
POST /recommend/hotels
```

Example flight request:

```json
{
  "from": "Recife (PE)",
  "to": "Brasilia (DF)",
  "flightType": "economic",
  "agency": "CloudFy",
  "time": 1.76,
  "distance": 1658,
  "date": "2026-07-13"
}
```

## MLflow Tracking

```bash
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python mlflow_tracking/train_with_mlflow.py --data data/raw/flights.csv
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001 --workers 1
```

The experiment compares Linear Regression, Decision Tree, and Random Forest and logs parameters, MAE, RMSE, R², dataset metadata, and model artifacts.

## Docker

```bash
docker build -t voyage-analytics-api .
docker run --rm -p 5000:5000 voyage-analytics-api
```

## Kubernetes

Replace `YOUR_DOCKERHUB_USERNAME` in `kubernetes/deployment.yml`, then apply:

```bash
kubectl apply -f kubernetes/deployment.yml
kubectl apply -f kubernetes/service.yml
kubectl apply -f kubernetes/hpa.yml
```

## Apache Airflow

The DAG `voyage_flight_price_regression_pipeline` performs:

```text
validate_data
    -> train_regression_model
    -> validate_model_artifact
    -> track_experiments
```

## Jenkins CI/CD

The `Jenkinsfile` performs:

```text
Checkout
  -> Python Validation
  -> Install Test Dependencies
  -> Run Tests
  -> Build Docker Image
  -> Container Smoke Test
  -> Optional Docker Push
  -> Optional Kubernetes Deployment
```

## GitHub Actions

The custom workflow runs automatically on pushes and pull requests:

```text
Python dependency installation
  -> Python compilation
  -> Model artifact validation
  -> Pytest API tests
  -> Docker image build validation
```

For this public repository, the workflow uses standard GitHub-hosted runners.

## Development Status

- [x] Three modeling notebooks
- [x] Reusable training modules
- [x] Flask API and API tests
- [x] Multipage Streamlit portfolio app
- [x] Public Streamlit Community Cloud deployment
- [x] MLflow tracking
- [x] Docker configuration
- [x] Kubernetes manifests
- [x] Airflow DAG
- [x] Jenkins pipeline
- [x] GitHub Actions CI
- [ ] Execute Docker, Kubernetes, Airflow, and Jenkins and capture evidence
- [ ] Complete final documentation and presentation

## Author

**Tarun Sreekar Parasa**  
MSc Data Analytics — Berlin School of Business and Innovation