"""Reproducible baseline for the California housing portfolio project."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["median_income", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]
CATEGORICAL = ["ocean_proximity"]
TARGET = "median_house_value"

def build_model():
    """Create an unfitted pipeline shared by both evaluation protocols."""
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ]), CATEGORICAL)
    ])
    return Pipeline([("preprocessing", preprocessing), ("regression", LinearRegression())])


def run(csv_path):
    data = pd.read_csv(csv_path)
    required = NUMERIC + CATEGORICAL + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    try:
        y = pd.to_numeric(data[TARGET], errors="raise")
    except (TypeError, ValueError):
        raise ValueError("Target median_house_value must be numeric and finite") from None
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Target median_house_value must be numeric and finite")
    # With a 20% holdout, at least six rows are needed for two test samples.
    if len(data) < 6:
        raise ValueError("At least 6 rows are required for an R2 score with the 20% holdout")
    X = data[NUMERIC + CATEGORICAL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "train_rows": len(X_train), "test_rows": len(X_test),
        "r2": float(r2_score(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions))
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "housing.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.data), indent=2))
