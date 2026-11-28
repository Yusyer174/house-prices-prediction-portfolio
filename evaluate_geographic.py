"""Fixed, exploratory geographic holdout; not a buffered or temporal split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from train import NUMERIC, CATEGORICAL, TARGET, build_model

COORDINATES = ["latitude", "longitude"]
SEED = 42
TEST_GROUP_FRACTION = 0.2


def geographic_groups(data):
    """A group is a one-degree cell; floor works for negative longitude."""
    try:
        coordinates = data[COORDINATES].to_numpy(dtype=float)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Latitude and longitude must be numeric and finite") from exc
    if (not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:, 0]) > 90).any()
            or (np.abs(coordinates[:, 1]) > 180).any()):
        raise ValueError("Latitude and longitude must be finite valid coordinates")
    cells = np.floor(coordinates).astype(int)
    return np.array([f"{lat}:{lon}" for lat, lon in cells])


def geographic_split(data):
    groups = geographic_groups(data)
    if len(set(groups)) < 2:
        raise ValueError("At least two geographic cells are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRACTION,
                                  random_state=SEED)
    train_idx, test_idx = next(splitter.split(data, groups=groups))
    if min(len(train_idx), len(test_idx)) < 2:
        raise ValueError("Both geographic partitions need at least two rows")
    return train_idx, test_idx, groups


def metrics(y, predictions):
    mse = float(mean_squared_error(y, predictions))
    return {"r2": float(r2_score(y, predictions)), "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, predictions))}


def evaluate(data):
    required = NUMERIC + CATEGORICAL + [TARGET] + COORDINATES
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    try:
        y = pd.to_numeric(data[TARGET], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target must be numeric and finite") from exc
    if not np.isfinite(y).all():
        raise ValueError("Target must be numeric and finite")
    tr, te, groups = geographic_split(data)
    if len(np.unique(y[te])) < 2:
        raise ValueError("Held-out target must vary for meaningful R2")
    train_groups, test_groups = set(groups[tr]), set(groups[te])
    if train_groups & test_groups:
        raise RuntimeError("Geographic group overlap")
    x = data[NUMERIC + CATEGORICAL]
    model = build_model()
    model.fit(x.iloc[tr], y[tr])
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x.iloc[tr], y[tr])
    return {
        "protocol": "one_degree_group_holdout_v1",
        "seed": SEED,
        "requested_test_group_fraction": TEST_GROUP_FRACTION,
        "train_rows": len(tr), "test_rows": len(te),
        "test_row_fraction": len(te) / len(data),
        "train_groups": len(train_groups), "test_groups": len(test_groups),
        "group_overlap": len(train_groups & test_groups),
        "linear_regression": metrics(y[te], model.predict(x.iloc[te])),
        "training_mean_dummy": metrics(y[te], dummy.predict(x.iloc[te])),
    }


def run(path):
    path = Path(path)
    # Read once so the hash describes the exact bytes supplied to the parser.
    from io import BytesIO
    contents = path.read_bytes()
    report = evaluate(pd.read_csv(BytesIO(contents)))
    report["dataset_sha256"] = hashlib.sha256(contents).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "housing.csv")
    print(json.dumps(run(parser.parse_args().data), indent=2, allow_nan=False))
