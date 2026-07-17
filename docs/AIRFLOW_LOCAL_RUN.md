# Apache Airflow Local Execution

This project uses a lightweight single-container Airflow development environment. It is intended for local academic demonstration, not production deployment.

## Workflow

The DAG `voyage_flight_price_regression_pipeline` runs these tasks in order:

```text
validate_data
  -> train_regression_model
  -> validate_model_artifact
  -> track_experiments
```

The repository is mounted at `/opt/airflow/voyage-project`, allowing the DAG to use the real CSV, training script, model directory, and MLflow script.

## Start Airflow

Run these commands from the repository root:

```powershell
git pull origin main
docker compose -f airflow/docker-compose.yml build
docker compose -f airflow/docker-compose.yml up -d
```

Check the service:

```powershell
docker compose -f airflow/docker-compose.yml ps
docker compose -f airflow/docker-compose.yml logs -f airflow
```

The first startup can take several minutes while Airflow initializes its metadata database and services.

## Retrieve Login Credentials

The username is normally `admin`. Retrieve the generated password with:

```powershell
docker compose -f airflow/docker-compose.yml exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Open:

```text
http://localhost:8080
```

## Run the Voyage DAG

1. Open **Dags** in the Airflow UI.
2. Find `voyage_flight_price_regression_pipeline`.
3. Unpause it if necessary.
4. Select **Trigger Dag**.
5. Open the Dag run and monitor the graph until every task is green.

You can confirm discovery from the terminal:

```powershell
docker compose -f airflow/docker-compose.yml exec airflow airflow dags list
```

## Expected Outputs

A successful run validates `data/raw/flights.csv`, retrains `models/flight_price_model.joblib`, validates a sample prediction, and logs the model comparison to the project MLflow database.

## Stop Airflow

Keep the metadata volume for later use:

```powershell
docker compose -f airflow/docker-compose.yml down
```

Remove the Airflow containers and reset the local Airflow metadata volume:

```powershell
docker compose -f airflow/docker-compose.yml down --volumes
```

Use the volume-removal command only when a clean reset is needed.
