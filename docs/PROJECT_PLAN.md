# Project Plan — Voyage Analytics: Integrating MLOps in Travel

## Objective

Build an end-to-end travel machine-learning project containing three modelling tasks and a complete MLOps workflow centred on the flight price regression model.

## Datasets

### Users
- 1,340 rows
- user identifier, company, name, gender, age

### Flights
- 271,888 rows
- 135,944 unique `travelCode` values
- paired outbound and return records per trip
- target for regression: `price`

### Hotels
- 40,552 rows
- 1,310 users with hotel-booking history
- 9 hotels across 9 cities

The initial raw-data audit found no missing values or duplicate rows.

## Workstream 1 — Flight Price Regression

### Goal
Predict flight price and use the selected model as the primary production model for the MLOps lifecycle.

### Validation principle
Records sharing a `travelCode` must not be split across training and testing because outbound and return records belong to the same trip. Group-aware splitting will be used to reduce leakage.

### Planned modelling workflow
1. Data understanding and exploratory analysis
2. Feature engineering
3. Preprocessing pipeline
4. Baseline model
5. Tree-based ensemble models
6. Cross-validation and tuning
7. MAE, RMSE and R² evaluation
8. Error analysis and explainability
9. Save the final pipeline as a model artifact

## Workstream 2 — Gender Classification

The raw target contains `male`, `female`, and `none`. The `none` label is treated as unavailable target information for supervised evaluation rather than automatically assigning it a third semantic class.

### Planned workflow
1. Isolate labelled records
2. Build name-based character TF-IDF features
3. Add suitable structured user features where justified
4. Compare classification models
5. Evaluate with accuracy, precision, recall, F1-score and confusion matrix
6. Save the final classification pipeline
7. Optionally generate inferred predictions for previously unlabeled records

## Workstream 3 — Hotel Recommendation

The catalogue is small and user histories are relatively dense, so the project will use an explainable recommendation design rather than an unnecessarily complex deep-learning recommender.

### Planned approach
A hybrid implicit-feedback method using some combination of:
- historical booking frequency
- user similarity
- hotel popularity
- price preference
- cold-start fallback logic

Evaluation will use a holdout strategy and ranking metrics such as Hit Rate@K and Recall@K where appropriate.

## MLOps Workflow

The regression model will demonstrate:

1. **Flask REST API** — real-time predictions
2. **MLflow** — experiment tracking and model logging
3. **Docker** — reproducible container image
4. **Kubernetes** — deployment and replica-based scaling
5. **Apache Airflow** — automated model workflow
6. **Jenkins** — CI/CD pipeline
7. **Tests** — model and API validation

The hotel recommendation system will power a Streamlit application for recommendations and travel insights.

## Planned Repository Structure

```text
notebooks/
  01_flight_price_regression.ipynb
  02_gender_classification.ipynb
  03_hotel_recommendation.ipynb

data/
  README.md
  raw/

src/
  regression/
  classification/
  recommendation/

models/
api/
streamlit_app/
kubernetes/
airflow/dags/
mlflow_tracking/
tests/
docs/
```

## Submission Deliverables

- Three completed AlmaBetter-format notebooks
- One GitHub repository containing all MLOps project files
- Saved model artifacts
- Flask application
- Streamlit application
- Dockerfile
- Kubernetes deployment files
- Airflow DAG
- Jenkinsfile
- MLflow tracking code
- Comprehensive repository README
- Separate workflow documentation with execution screenshots
- Final video presentation focused on the regression MLOps lifecycle
