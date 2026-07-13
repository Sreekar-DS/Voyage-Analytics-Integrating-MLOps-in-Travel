# Data Directory

This project uses three linked travel datasets.

## `users.csv`

| Column | Description |
|---|---|
| `code` | User identifier |
| `company` | Associated company |
| `name` | User name |
| `gender` | Recorded gender label; includes `none` for unavailable labels |
| `age` | User age |

## `flights.csv`

| Column | Description |
|---|---|
| `travelCode` | Travel identifier |
| `userCode` | User identifier linked to `users.csv` |
| `from` | Origin city |
| `to` | Destination city |
| `flightType` | Flight class/type |
| `price` | Flight price and regression target |
| `time` | Flight duration |
| `distance` | Flight distance |
| `agency` | Flight agency |
| `date` | Flight date |

## `hotels.csv`

| Column | Description |
|---|---|
| `travelCode` | Travel identifier linked to trip history |
| `userCode` | User identifier linked to `users.csv` |
| `name` | Hotel name |
| `place` | Hotel location |
| `days` | Number of stay days |
| `price` | Price per day |
| `total` | Total stay cost |
| `date` | Booking date |

## Initial Audit

| Dataset | Rows | Missing values | Duplicate rows |
|---|---:|---:|---:|
| Users | 1,340 | 0 | 0 |
| Flights | 271,888 | 0 | 0 |
| Hotels | 40,552 | 0 | 0 |

## Data Handling

The raw CSV files are the source of truth. Modelling notebooks will perform explicit loading, validation, feature engineering and preprocessing so the workflow remains reproducible.

For the flight regression task, records sharing the same `travelCode` will be kept together during holdout splitting to reduce trip-level leakage.
