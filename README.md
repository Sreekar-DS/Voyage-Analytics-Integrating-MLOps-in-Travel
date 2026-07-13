# Voyage Analytics: Integrating MLOps in Travel

An end-to-end travel analytics and MLOps capstone built with three linked datasets covering users, flights, and hotels.

## Project Overview

The project combines machine learning and MLOps to solve three travel-related problems:

1. **Flight Price Regression** — predict flight prices from route, flight type, agency, distance, duration, and engineered travel features.
2. **Gender Classification** — build a supervised classification model from labelled user records while treating unavailable target values separately.
3. **Hotel Recommendation** — recommend hotels using historical booking behaviour and user preferences.

The flight price regression model is the primary model used to demonstrate the full MLOps lifecycle.

## MLOps Scope

The project will include:

- Flask REST API for real-time model inference
- Docker containerization
- Kubernetes deployment and scaling
- Apache Airflow workflow orchestration
- Jenkins CI/CD pipeline
- MLflow experiment tracking and model management
- Streamlit application for travel recommendations and insights

## Dataset Summary

| Dataset | Rows | Purpose |
|---|---:|---|
| `users.csv` | 1,340 | User profile and classification data |
| `flights.csv` | 271,888 | Flight price prediction and travel history |
| `hotels.csv` | 40,552 | Hotel booking history for recommendations |

Initial validation found no missing values or duplicate rows in the raw datasets.

## Planned Repository Structure

```text
.
├── notebooks/                 # Three AlmaBetter-format modelling notebooks
├── data/                      # Dataset notes and optional raw/processed data
├── src/                       # Reusable modelling and inference modules
├── models/                    # Saved model artifacts
├── api/                       # Flask REST API
├── streamlit_app/             # Streamlit recommendation application
├── kubernetes/                # Kubernetes manifests
├── airflow/dags/              # Airflow DAGs
├── mlflow_tracking/           # MLflow experiment scripts
├── tests/                     # Model and API tests
├── docs/                      # Architecture notes and screenshot references
├── Dockerfile
├── Jenkinsfile
├── requirements.txt
└── README.md
```

## Development Status

The repository is being developed incrementally so that the commit history reflects the actual project workflow:

- [x] Repository initialized
- [x] Dataset and submission-template audit
- [ ] Flight price regression notebook
- [ ] Gender classification notebook
- [ ] Hotel recommendation notebook
- [ ] Reusable model training code
- [ ] Flask API
- [ ] Streamlit app
- [ ] MLflow tracking
- [ ] Docker deployment
- [ ] Kubernetes deployment
- [ ] Airflow orchestration
- [ ] Jenkins CI/CD
- [ ] Final documentation and screenshots

## Key Modelling Considerations

### Flight price regression

The data contains paired outbound and return flight records for each `travelCode`. Model validation will therefore avoid placing records from the same trip on both sides of a train/test split. This reduces leakage and gives a more defensible performance estimate.

### Gender classification

The target contains `male`, `female`, and `none`. The `none` value is treated as unavailable or unlabeled target information rather than automatically forcing it into a third semantic gender class. Supervised evaluation will focus on the labelled records.

### Hotel recommendation

The hotel catalogue is compact and user histories are relatively dense. The recommender will therefore prioritise an explainable implicit-feedback or hybrid approach rather than an unnecessarily complex large-scale recommender architecture.

## Author

**Tarun Sreekar Parasa**

MSc Data Analytics — Berlin School of Business and Innovation
