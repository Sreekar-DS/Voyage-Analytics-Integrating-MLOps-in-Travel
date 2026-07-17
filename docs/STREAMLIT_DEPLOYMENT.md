# Streamlit Community Cloud Deployment

The repository is ready for a public Streamlit Community Cloud deployment. The final authorization must be completed by the GitHub account owner because Streamlit requires GitHub OAuth approval.

## One-time deployment

1. Open `https://share.streamlit.io/`.
2. Select **Continue with GitHub** and authorize Streamlit to access the public repository.
3. Click **Create app**.
4. Choose **Yup, I have an app**.
5. Enter the following values:

| Field | Value |
|---|---|
| Repository | `Sreekar-DS/Voyage-Analytics-Integrating-MLOps-in-Travel` |
| Branch | `main` |
| Main file path | `streamlit_app/app.py` |
| App URL | `voyage-analytics-tarun` if available |

6. Open **Advanced settings** and select **Python 3.12**.
7. No secrets are required for this application.
8. Click **Deploy**.

Community Cloud will read `streamlit_app/requirements.txt`, load the tracked Joblib artifacts from `models/`, and use the CSV files in `data/raw/`.

## Expected pages

- Voyage Analytics overview
- Flight Price Prediction
- Hotel Recommendation
- Profile Classification Demonstration

## After deployment

Copy the final `https://...streamlit.app` URL and replace the pending deployment note in `README.md` with:

```markdown
[![Open Live App](https://img.shields.io/badge/Live_App-Open-FF4B4B?logo=streamlit&logoColor=white)](FINAL_STREAMLIT_URL)
```

Also add the URL to the GitHub repository **About** section under **Website**.

## Updating the live app

The GitHub repository is the source of truth. New commits to `main` are detected by Community Cloud and reflected in the deployed application. Dependency changes in `streamlit_app/requirements.txt` trigger a rebuild.

## Troubleshooting

### Model artifact not found

Confirm these tracked files exist:

```text
models/flight_price_model.joblib
models/gender_classifier.joblib
models/hotel_recommender.joblib
```

### Dataset not found

Confirm these files exist:

```text
data/raw/flights.csv
data/raw/hotels.csv
data/raw/users.csv
```

### scikit-learn version warning

The application pins `scikit-learn==1.8.0` because the saved Joblib artifacts were created with that version.

### App build is slow

The Streamlit-specific requirements file intentionally excludes Flask, MLflow, Airflow, and Jenkins dependencies to keep the public app deployment lightweight.
