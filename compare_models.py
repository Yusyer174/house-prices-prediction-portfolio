"""Compare fixed models by grouped CV on the geographic development partition only."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from evaluate_geographic import COORDINATES, geographic_split, metrics
from train import CATEGORICAL, NUMERIC, TARGET, build_model

FOLDS = 5
SEED = 42


def candidates():
    """Fresh pipelines with fixed hyperparameters; no search or test-set tuning."""
    return {
        "linear": build_model(),
        "random_forest": build_model().set_params(regression=RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=5,
            random_state=SEED, n_jobs=1)),
        "gradient_boosting": build_model().set_params(regression=GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=5, random_state=SEED)),
        "training_mean_dummy": DummyRegressor(strategy="mean"),
    }


def development_folds(data):
    """Return original row indices; reserved cells never enter any CV fold."""
    dev, reserved, groups = geographic_split(data)
    if len(set(groups[dev])) < FOLDS:
        raise ValueError("At least five development cells are required")
    reserved_groups = set(groups[reserved])
    counts = np.zeros(len(data), dtype=int)
    folds = []
    for tr_local, va_local in GroupKFold(n_splits=FOLDS).split(
            dev, groups=groups[dev]):
        tr, va = dev[tr_local], dev[va_local]
        tr_groups, va_groups = set(groups[tr]), set(groups[va])
        if tr_groups & va_groups or (tr_groups | va_groups) & reserved_groups:
            raise RuntimeError("Geographic group overlap")
        if min(len(tr), len(va)) < 2:
            raise ValueError("Each fold partition needs at least two rows")
        counts[va] += 1
        folds.append((tr, va))
    if not np.all(counts[dev] == 1) or not np.all(counts[reserved] == 0):
        raise RuntimeError("Invalid out-of-fold coverage")
    return dev, reserved, groups, folds


def compare(data):
    required = NUMERIC + CATEGORICAL + [TARGET] + COORDINATES
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    dev, reserved, groups, folds = development_folds(data)
    # Only the development targets enter validation, fitting and scoring.
    try:
        y_dev = pd.to_numeric(data[TARGET].iloc[dev], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Development targets must be numeric and finite") from exc
    if not np.isfinite(y_dev).all():
        raise ValueError("Development targets must be numeric and finite")
    y = np.full(len(data), np.nan)
    y[dev] = y_dev
    x = data[NUMERIC + CATEGORICAL]
    oof = {name: np.full(len(data), np.nan) for name in candidates()}
    fold_results = []
    for fold_id, (tr, va) in enumerate(folds, start=1):
        if len(np.unique(y[va])) < 2:
            raise ValueError("Validation target must vary for meaningful R2")
        row = {
            "fold": fold_id, "train_rows": len(tr), "validation_rows": len(va),
            "train_groups": len(set(groups[tr])),
            "validation_groups": len(set(groups[va])), "models": {},
        }
        for name, estimator in candidates().items():
            estimator.fit(x.iloc[tr], y[tr])
            prediction = estimator.predict(x.iloc[va])
            oof[name][va] = prediction
            row["models"][name] = metrics(y[va], prediction)
        fold_results.append(row)

    summary = {}
    for name, predictions in oof.items():
        if not np.isfinite(predictions[dev]).all() or not np.isnan(predictions[reserved]).all():
            raise RuntimeError("Invalid predictions or reserved-row scoring")
        scores = [fold["models"][name]["mae"] for fold in fold_results]
        summary[name] = {
            **metrics(y[dev], predictions[dev]),
            "fold_mae_std": float(np.std(scores, ddof=0)),
            "fold_mae_min": float(min(scores)),
            "fold_mae_max": float(max(scores)),
        }
    # The dummy is a reference, not one of the three model candidates.
    ranking = sorted((n for n in summary if n != "training_mean_dummy"),
                     key=lambda n: (summary[n]["mae"], n))
    return {
        "protocol": "geographic_development_group_cv_v1",
        "seed": SEED, "fold_count": FOLDS,
        "development_rows": len(dev), "development_groups": len(set(groups[dev])),
        "reserved_rows": len(reserved), "reserved_groups": len(set(groups[reserved])),
        "reserved_rows_scored": 0,
        "primary_metric": "pooled_out_of_fold_mae",
        "candidate_ranking": ranking, "models": summary, "folds": fold_results,
    }


def run(path):
    contents = Path(path).read_bytes()
    report = compare(pd.read_csv(BytesIO(contents)))
    report["dataset_sha256"] = hashlib.sha256(contents).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parent / "housing.csv")
    print(json.dumps(run(parser.parse_args().data), indent=2, allow_nan=False))
