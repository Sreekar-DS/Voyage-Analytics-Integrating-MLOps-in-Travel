"""Interactive flight-price prediction page."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from common import build_flight_frame, load_artifact, load_csv  # noqa: E402

st.title("🛫 Flight Price Prediction")
st.write(
    "Select an observed route and service combination, review the derived travel "
    "characteristics, and generate a prediction from the saved Random Forest pipeline."
)

try:
    model = load_artifact("flight_price_model.joblib")
except Exception as exc:
    st.error(str(exc))
    st.stop()

flights = load_csv("flights.csv", date_columns=("date",))
if flights.empty:
    st.error(
        "The route catalogue is unavailable. Add data/raw/flights.csv to enable the "
        "interactive flight predictor."
    )
    st.stop()

required_columns = {
    "from",
    "to",
    "flightType",
    "agency",
    "time",
    "distance",
}
missing_columns = required_columns.difference(flights.columns)
if missing_columns:
    st.error(f"flights.csv is missing required columns: {sorted(missing_columns)}")
    st.stop()

left, right = st.columns(2)

with left:
    origin_options = sorted(flights["from"].dropna().astype(str).unique())
    origin = st.selectbox("Origin", origin_options)

    destination_options = sorted(
        flights.loc[flights["from"].astype(str) == origin, "to"]
        .dropna()
        .astype(str)
        .unique()
    )
    destination = st.selectbox("Destination", destination_options)

    route_rows = flights[
        (flights["from"].astype(str) == origin)
        & (flights["to"].astype(str) == destination)
    ]

    flight_type_options = sorted(route_rows["flightType"].dropna().astype(str).unique())
    flight_type = st.selectbox("Flight type", flight_type_options)

    type_rows = route_rows[route_rows["flightType"].astype(str) == flight_type]
    agency_options = sorted(type_rows["agency"].dropna().astype(str).unique())
    agency = st.selectbox("Agency", agency_options)

with right:
    matching_rows = type_rows[type_rows["agency"].astype(str) == agency]
    default_duration = float(matching_rows["time"].median())
    default_distance = float(matching_rows["distance"].median())

    travel_date = st.date_input("Travel date", value=date.today())
    edit_route_values = st.checkbox(
        "Manually adjust duration and distance",
        help="The defaults are derived from the selected route in the supplied dataset.",
    )

    duration = st.number_input(
        "Duration (hours)",
        min_value=0.01,
        value=round(default_duration, 2),
        step=0.01,
        disabled=not edit_route_values,
    )
    distance = st.number_input(
        "Distance",
        min_value=1.0,
        value=round(default_distance, 2),
        step=1.0,
        disabled=not edit_route_values,
    )

st.caption(
    "Inputs are constrained to route and service combinations observed in the training "
    "dataset, reducing unsupported extrapolation."
)

if st.button("Predict flight price", type="primary", use_container_width=True):
    try:
        features = build_flight_frame(
            origin=origin,
            destination=destination,
            flight_type=flight_type,
            agency=agency,
            duration=duration,
            distance=distance,
            travel_date=travel_date,
        )
        prediction = float(model.predict(features)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        st.success("Prediction completed")
        metric_col, route_col = st.columns([1, 2])
        with metric_col:
            st.metric("Predicted price", f"{prediction:,.2f}")
        with route_col:
            st.markdown("#### Selected itinerary")
            st.write(
                f"**{origin} → {destination}** · {flight_type} · {agency} · "
                f"{duration:.2f} hours · {distance:,.0f} distance units"
            )

        with st.expander("View model input features"):
            st.dataframe(features, hide_index=True, use_container_width=True)

st.divider()
st.markdown("### Model context")
st.write(
    "The final model is a Random Forest trained with a group-aware split by travelCode. "
    "Its unusually low validation error is caused by the supplied dataset's largely fixed "
    "pricing for exact route, flight-type, and agency combinations."
)
st.caption("Predicted values are expressed in the original dataset's price units.")
