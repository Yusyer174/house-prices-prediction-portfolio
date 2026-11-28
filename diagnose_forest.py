"""Development-only diagnostic analysis of the fixed Random Forest."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_models import SEED, candidates, development_folds
from evaluate_geographic import COORDINATES, metrics
from train import NUMERIC, CATEGORICAL, TARGET

FEATURES = NUMERIC + CATEGORICAL
REPEATS = 3
BANDS = (
    ("below_150k", -np.inf, 150000),
    ("150k_to_300k", 150000, 300000),
    ("300k_to_500k", 300000, 500000),
    ("500k_or_more", 500000, np.inf),
)


def error_summary(actual, predicted):
    if len(actual) == 0:
        return {"rows": 0, "mae": None, "rmse": None, "signed_error": None}
    residual = np.asarray(predicted) - np.asarray(actual)
    return {"rows": len(residual), "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "signed_error": float(np.mean(residual))}


def price_errors(actual, predicted):
    return {name: error_summary(actual[mask], predicted[mask])
            for name, low, high in BANDS
            for mask in [(actual >= low) & (actual < high)]}


def permutation_deltas(model, x, actual, baseline, rng):
    """Shuffle raw held-out columns without fitting or changing the input."""
    deltas = {}
    for feature in FEATURES:
        scores = []
        for _ in range(REPEATS):
            shuffled = x.copy()
            shuffled[feature] = x[feature].to_numpy()[rng.permutation(len(x))]
            score = error_summary(actual, model.predict(shuffled))["mae"]
            scores.append(score - baseline)
        deltas[feature] = scores
    return deltas


def aggregate_importance(folds):
    report = {}
    weights = np.asarray([f["rows"] for f in folds], dtype=float)
    for feature in FEATURES:
        deltas = np.asarray([f["permutation_deltas"][feature] for f in folds])
        # Equal repeats within each fold; weight folds by validation rows.
        per_repeat = np.average(deltas, axis=0, weights=weights)
        fold_means = deltas.mean(axis=1)
        report[feature] = {
            "mean_mae_increase": float(per_repeat.mean()),
            "repeat_weighted_std": float(per_repeat.std()),
            "fold_mean_min": float(fold_means.min()),
            "fold_mean_max": float(fold_means.max()),
        }
    return report


def diagnose(data):
    required = FEATURES + [TARGET] + COORDINATES
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    dev, reserved, groups, folds = development_folds(data)
    y = np.full(len(data), np.nan)
    try:
        y[dev] = pd.to_numeric(data[TARGET].iloc[dev], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Development targets must be numeric and finite") from exc
    if not np.isfinite(y[dev]).all():
        raise ValueError("Development targets must be numeric and finite")
    x = data[FEATURES]
    oof = np.full(len(data), np.nan)
    results = []
    rng = np.random.default_rng(SEED)
    for fold_id, (tr, va) in enumerate(folds, start=1):
        model = candidates()["random_forest"]
        model.fit(x.iloc[tr], y[tr])
        validation = x.iloc[va]
        prediction = model.predict(validation)
        oof[va] = prediction
        errors = error_summary(y[va], prediction)
        results.append({
            "fold": fold_id, "validation_groups": len(set(groups[va])),
            **errors, "permutation_deltas": permutation_deltas(
                model, validation, y[va], errors["mae"], rng),
        })
    if not np.isfinite(oof[dev]).all() or not np.isnan(oof[reserved]).all():
        raise RuntimeError("Invalid predictions or reserved-row scoring")
    return {
        "protocol": "forest_development_diagnostics_v1",
        "seed": SEED, "permutation_repeats_per_fold": REPEATS,
        "development_rows": len(dev), "reserved_rows": len(reserved),
        "reserved_rows_scored": 0,
        "overall": {**error_summary(y[dev], oof[dev]), **metrics(y[dev], oof[dev])},
        "price_bands": price_errors(y[dev], oof[dev]),
        "importance": aggregate_importance(results),
        "folds": results,
    }


def run(path):
    contents = Path(path).read_bytes()
    report = diagnose(pd.read_csv(BytesIO(contents)))
    report["dataset_sha256"] = hashlib.sha256(contents).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parent / "housing.csv")
    print(json.dumps(run(parser.parse_args().data), indent=2, allow_nan=False))
