# Saved Model Artifacts

The deployment components expect these Joblib files in this folder:

- `flight_price_model.joblib`
- `gender_classifier.joblib`
- `hotel_recommender.joblib`

They can be regenerated from the raw datasets with:

```bash
python src/regression/train.py --data data/raw/flights.csv --model-out models/flight_price_model.joblib
python src/classification/train.py --data data/raw/users.csv --model-out models/gender_classifier.joblib
python src/recommendation/train.py --data data/raw/hotels.csv --model-out models/hotel_recommender.joblib
```

The Flask API and Streamlit application load these files at runtime. The regression and classification artifacts contain the complete preprocessing-plus-model pipeline so inference uses the same transformations as training.
