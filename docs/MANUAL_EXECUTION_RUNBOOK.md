# Manual Execution Runbook

This runbook is for executing the completed Voyage Analytics project components and collecting evidence for the final submission. It intentionally focuses on **running the project**, not installing the tools.

## 1. Required Repository Artifacts

Before deployment, confirm these files exist in the repository:

```text
notebooks/
  Voyage_Analytics_Flight_Price_Regression.ipynb
  Voyage_Analytics_Gender_Classification.ipynb
  Voyage_Analytics_Hotel_Recommendation.ipynb

models/
  flight_price_model.joblib
  gender_classifier.joblib
  hotel_recommender.joblib
```

Also place the raw datasets under `data/raw/` when running training, Streamlit, Airflow, or MLflow locally:

```text
data/raw/flights.csv
data/raw/users.csv
data/raw/hotels.csv
```

Do not commit passwords, tokens, or registry credentials.

---

## 2. Execute and Save the Three Colab Notebooks

Open each `.ipynb` file in Google Colab and run all cells from top to bottom:

1. Flight Price Regression
2. Gender Classification
3. Hotel Recommendation

After execution:

- confirm charts and metrics render correctly;
- confirm the GitHub repository link appears in each notebook;
- save the executed notebook;
- upload the executed version to the `notebooks/` folder in GitHub.

Recommended evidence:

- regression model comparison and final evaluation output;
- classification report/confusion matrix;
- recommendation evaluation metrics and sample recommendations.

---

## 3. Regenerate Model Artifacts When Needed

From the repository root:

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

Expected regression benchmark from the validated dataset and grouped split is approximately:

```text
MAE:  0.0047
RMSE: 0.1562
R2:   0.9999998
```

The unusually high performance reflects the deterministic fare structure of the supplied dataset and should be explained in the report and video.

---

## 4. Flask REST API

Start the API from the repository root:

```bash
python api/app.py
```

### Health check

```bash
curl http://127.0.0.1:5000/health
```

The response should show `status: ok` and `true` for the available model artifacts.

### Flight price prediction

```bash
curl -X POST http://127.0.0.1:5000/predict/flight-price \
  -H "Content-Type: application/json" \
  -d '{
    "from": "Recife (PE)",
    "to": "Brasilia (DF)",
    "flightType": "economic",
    "agency": "CloudFy",
    "time": 1.76,
    "distance": 1658,
    "date": "2023-07-13"
  }'
```

### Gender prediction

```bash
curl -X POST http://127.0.0.1:5000/predict/gender \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ana Silva",
    "company": "Acme",
    "age": 30
  }'
```

### Hotel recommendation

```bash
curl -X POST http://127.0.0.1:5000/recommend/hotels \
  -H "Content-Type: application/json" \
  -d '{
    "userCode": 1,
    "top_k": 3
  }'
```

Capture the health response and at least one successful regression prediction.

---

## 5. Streamlit Recommendation Application

Run:

```bash
streamlit run streamlit_app/app.py
```

Demonstrate both:

- an existing user with personalized recommendations;
- a new user code showing the cold-start popularity fallback.

Capture:

- sidebar user selection;
- recommendation cards;
- booking-history KPIs;
- at least one analytical chart.

---

## 6. Docker Deployment

Build the image:

```bash
docker build -t voyage-analytics-api:latest .
```

Run the container:

```bash
docker run --rm -d \
  --name voyage-analytics-api \
  -p 5000:5000 \
  voyage-analytics-api:latest
```

Check the container:

```bash
docker ps
curl http://127.0.0.1:5000/health
```

Run the same regression prediction request used for the local Flask API.

Stop the container after the screenshots are captured:

```bash
docker stop voyage-analytics-api
```

Capture:

- successful image build;
- running container;
- containerized health response;
- containerized flight prediction.

---

## 7. Publish the Docker Image

Use your own private registry credentials locally. The repository must not contain those credentials.

Tag the image using the registry account that will be referenced by Kubernetes:

```bash
docker tag voyage-analytics-api:latest \
  <DOCKERHUB_USERNAME>/voyage-analytics-api:latest
```

Push:

```bash
docker push <DOCKERHUB_USERNAME>/voyage-analytics-api:latest
```

Then replace `YOUR_DOCKERHUB_USERNAME` in `kubernetes/deployment.yml` with the real registry username.

---

## 8. Kubernetes Deployment

Apply the manifests:

```bash
kubectl apply -f kubernetes/deployment.yml
kubectl apply -f kubernetes/service.yml
kubectl apply -f kubernetes/hpa.yml
```

Verify:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get hpa
kubectl rollout status deployment/voyage-analytics-api
```

For a local cluster where a LoadBalancer address is not directly available, use the cluster-specific service exposure method or port forwarding, for example:

```bash
kubectl port-forward service/voyage-analytics-api-service 8080:80
```

Then test:

```bash
curl http://127.0.0.1:8080/health
```

Capture:

- multiple running pods;
- service status;
- HPA status;
- rollout success;
- successful API response through the Kubernetes service.

---

## 9. MLflow Experiment Tracking

From the repository root:

```bash
python mlflow_tracking/train_with_mlflow.py \
  --data data/raw/flights.csv
```

The script creates separate runs for:

- Linear Regression;
- Decision Tree;
- Random Forest.

The runs log:

- model parameters;
- MAE;
- RMSE;
- R-squared;
- dataset summary information;
- model artifacts.

Open the MLflow tracking UI for the configured tracking location and capture:

- the experiment table;
- the comparison of the three runs;
- the selected Random Forest run;
- metrics and artifact/model information.

---

## 10. Apache Airflow Workflow

The DAG file is:

```text
airflow/dags/regression_pipeline_dag.py
```

Default project path expected by the DAG:

```text
/opt/airflow/voyage-project
```

The paths can instead be supplied through these environment variables:

```text
VOYAGE_PROJECT_ROOT
VOYAGE_FLIGHTS_CSV
VOYAGE_FLIGHT_MODEL
MLFLOW_TRACKING_URI
```

The DAG dependency chain is:

```text
validate_data
  -> train_regression_model
  -> validate_model_artifact
  -> track_experiments
```

Trigger the DAG and capture:

- DAG listed in Airflow;
- graph view;
- successful run;
- all four tasks in the success state.

---

## 11. Jenkins CI/CD Pipeline

Create a Jenkins pipeline job pointing to this GitHub repository. The pipeline is defined in the root `Jenkinsfile`.

The stages are:

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

For registry publishing, configure a Jenkins username/password credential with this ID:

```text
dockerhub-credentials
```

Build parameters:

```text
DOCKERHUB_USERNAME
PUSH_IMAGE
DEPLOY_K8S
```

Recommended sequence:

1. Run with `PUSH_IMAGE=false` and `DEPLOY_K8S=false` to validate tests, Docker build, and smoke test.
2. Run with `PUSH_IMAGE=true` after registry credentials are configured.
3. Run with `DEPLOY_K8S=true` only when the Jenkins agent has valid cluster access.

Capture the successful stage view and relevant console output.

---

## 12. Final Evidence Order

Capture and insert evidence in this order:

1. Regression model comparison
2. Repository or architecture overview
3. Flask health endpoint
4. Flask regression prediction
5. Streamlit recommendations
6. Streamlit analytics/cold-start behavior
7. Docker build
8. Docker container health
9. Docker prediction
10. Kubernetes pods
11. Kubernetes service and HPA
12. Kubernetes rollout status
13. Airflow DAG list
14. Airflow graph
15. Airflow successful run
16. Jenkins stage view
17. Jenkins console output
18. MLflow experiment comparison
19. MLflow Random Forest metrics
20. MLflow model/artifact view

These numbers match the placeholders in the Google Doc workflow documentation.
