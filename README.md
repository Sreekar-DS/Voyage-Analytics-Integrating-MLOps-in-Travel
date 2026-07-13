# Voyage Analytics: Integrating MLOps in Travel

An end-to-end travel analytics and MLOps capstone built from three linked datasets covering users, flights, and hotels.

## Project Overview

The project solves three travel-related machine-learning problems and connects the primary regression model to a full MLOps lifecycle:

1. **Flight Price Regression** — predicts flight prices from route, flight type, agency, distance, duration, and engineered calendar features.
2. **Gender Classification** — predicts known binary labels from user names and supporting profile features while treating the raw `none` target values as unavailable labels.
3. **Hotel Recommendation** — ranks hotels from historical booking frequency with a popularity fallback for cold-start users.

The **flight price regression model** is the main model used to demonstrate serving, containerization, scaling, scheduling, CI/CD, and experiment tracking.

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

The very high tree-based scores must be interpreted in the context of this dataset: each exact origin, destination, flight type, and agency combination effectively has a fixed fare. The result demonstrates strong performance for the supplied pricing structure, not guaranteed generalization to unseen dynamic-market pricing regimes.

### Gender Classification

The raw target contains `male`, `female`, and `none`. The `none` value is treated as unavailable target information. Supervised modeling uses the 900 known male/female records.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Logistic Regression | ~0.811 | ~0.811 |
| Linear SVM | ~0.828 | ~0.827 |
| SGD Classifier | ~0.833 | ~0.833 |

The classification component is a technical project requirement and should not be used to override self-described identity or for high-stakes decisions.

### Hotel Recommendation

Evaluation uses a temporal holdout: each eligible user's latest booking is held out and ranked from earlier booking history.

| Recommender | Hit Rate@3 | MRR@3 |
|---|---:|---:|
| Global Popularity | ~0.353 | ~0.222 |
| Personalized Frequency | ~0.391 | ~0.236 |
| Tested Hybrid | ~0.366 | ~0.227 |

The simple personalized-frequency recommender is selected because the catalogue contains only nine hotels and user histories are dense.

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
Regression  Gender   Hotel
 Model      Model    Recommender
   |          |         |
   +----------+---------+
              |
         Flask REST API
              |
            Docker
              |
          Kubernetes
        /      |       \
   Airflow   Jenkins   MLflow
              |
        Streamlit App
```

## Implemented Components

- **Reusable training code** for regression, classification, and recommendation
- **Flask REST API** with health, regression, classification, and recommendation endpoints
- **Streamlit dashboard** for personalized hotel recommendations and travel-history insights
- **Dockerfile** for the Flask API
- **Kubernetes Deployment** with three initial replicas
- **Kubernetes Service** for API exposure
- **Horizontal Pod Autoscaler** for CPU-based scaling
- **Apache Airflow DAG** for data validation, model training, artifact validation, and MLflow tracking
- **Jenkins declarative CI/CD pipeline** for validation, testing, Docker build, smoke test, optional registry push, and optional Kubernetes deployment
- **MLflow experiment script** comparing the three regression models
- **Pytest API tests**

## Repository Structure

```text
.
├── notebooks/                 # Three AlmaBetter-format modeling notebooks
├── data/
│   └── README.md              # Dataset guidance
├── src/
│   ├── regression/
│   │   └── train.py
│   ├── classification/
│   │   └── train.py
│   └── recommendation/
│       └── train.py
├── models/                    # Saved Joblib artifacts
├── api/
│   └── app.py
├── streamlit_app/
│   └── app.py
├── kubernetes/
│   ├── deployment.yml
│   ├── service.yml
│   └── hpa.yml
├── airflow/
│   └── dags/
│       └── regression_pipeline_dag.py
├── mlflow_tracking/
│   └── train_with_mlflow.py
├── tests/
│   └── test_api.py
├── docs/
│   └── PROJECT_PLAN.md
├── Dockerfile
├── Jenkinsfile
├── requirements.txt
└── README.md
```

## Model Training

Place the raw datasets under `data/raw/` and run:

```bash
python src/regression/train.py \
  --data data/raw/flights.csv \
  --model-out models/flight_price_model.joblib

python src/classification/train.py \
  --data data/raw/users.csv \
  --model-out models/gender_classifier.joblib

python src/recommendation/train.py \
  --data data/raw/hotels.csv \
  --model-out models/hotel_recommender.joblib
```

## Flask REST API

Run the API after the model artifacts exist:

```bash
python api/app.py
```

### Health

```http
GET /health
```

### Flight Price Prediction

```http
POST /predict/flight-price
Content-Type: application/json
```

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

### Gender Classification

```http
POST /predict/gender
Content-Type: application/json
```

```json
{
  "name": "Ana Silva",
  "company": "Acme",
  "age": 30
}
```

### Hotel Recommendations

```http
POST /recommend/hotels
Content-Type: application/json
```

```json
{
  "userCode": 1,
  "top_k": 3
}
```

## Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

The dashboard displays recommendation cards, booking-history KPIs, hotel frequency, destination preferences, and a cold-start explanation when the selected user has no training history.

## Docker

```bash
docker build -t voyage-analytics-api .
docker run --rm -p 5000:5000 voyage-analytics-api
```

The container exposes the Flask API on port `5000` and includes a `/health` health check.

## Kubernetes

Before deployment, replace `YOUR_DOCKERHUB_USERNAME` in `kubernetes/deployment.yml` with the registry account used to publish the Docker image.

```bash
kubectl apply -f kubernetes/deployment.yml
kubectl apply -f kubernetes/service.yml
kubectl apply -f kubernetes/hpa.yml
```

The deployment starts with three replicas. The HPA is configured to scale between two and eight replicas using CPU utilization.

## Apache Airflow

The DAG `voyage_flight_price_regression_pipeline` performs:

```text
validate_data
    -> train_regression_model
    -> validate_model_artifact
    -> track_experiments
```

The project root and flights CSV path can be configured through:

- `VOYAGE_PROJECT_ROOT`
- `VOYAGE_FLIGHTS_CSV`
- `VOYAGE_FLIGHT_MODEL`
- `MLFLOW_TRACKING_URI`

## Jenkins CI/CD

The `Jenkinsfile` contains the following stages:

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

Docker registry publishing uses a Jenkins credential with the ID `dockerhub-credentials`. Registry push and Kubernetes deployment are controlled by build parameters.

## MLflow Tracking

Run:

```bash
python mlflow_tracking/train_with_mlflow.py --data data/raw/flights.csv
```

The script creates separate runs for Linear Regression, Decision Tree, and Random Forest, logging model parameters, MAE, RMSE, R², dataset metadata, and model artifacts.

## Development Status

- [x] Repository initialized
- [x] Dataset and submission-template audit
- [x] Flight price regression notebook prepared
- [x] Gender classification notebook prepared
- [x] Hotel recommendation notebook prepared
- [x] Reusable model training code
- [x] Flask API
- [x] Streamlit recommendation app
- [x] MLflow tracking code
- [x] Docker configuration
- [x] Kubernetes deployment and autoscaling manifests
- [x] Airflow orchestration DAG
- [x] Jenkins CI/CD pipeline
- [x] API tests
- [ ] Upload final notebook and Joblib binary artifacts to this repository
- [ ] Run each deployment stage in the target environment and capture screenshots
- [ ] Complete final Google Doc workflow documentation
- [ ] Record the regression MLOps presentation video

## Author

**Tarun Sreekar Parasa**  
MSc Data Analytics — Berlin School of Business and Innovation
