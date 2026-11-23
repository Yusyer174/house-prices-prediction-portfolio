import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from train import run

ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def test_reproducible_baseline(self):
        result = run(ROOT / "housing.csv")
        self.assertEqual(result["train_rows"], 16512)
        self.assertEqual(result["test_rows"], 4128)
        self.assertAlmostEqual(result["r2"], 0.6161192328109408, places=5)
        for key in ("r2", "mse", "rmse", "mae"):
            self.assertTrue(math.isfinite(result[key]))
        self.assertAlmostEqual(result["rmse"] ** 2, result["mse"], delta=1e-4)
        self.assertGreaterEqual(result["rmse"], result["mae"])

    def test_missing_required_column_has_clear_error(self):
        data = pd.read_csv(ROOT / "housing.csv", nrows=30).drop(columns=["median_income"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_column.csv"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing required CSV columns: median_income"):
                run(path)

    def test_non_numeric_target_rejected(self):
        data = pd.read_csv(ROOT / "housing.csv", nrows=30)
        data["median_house_value"] = "not-a-number"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_target.csv"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "must be numeric and finite"):
                run(path)

    def test_missing_target_rejected(self):
        data = pd.read_csv(ROOT / "housing.csv", nrows=30)
        data.loc[0, "median_house_value"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_target.csv"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "must be numeric and finite"):
                run(path)

    def test_missing_features_are_imputed(self):
        rng = np.random.default_rng(42)
        n = 100
        data = pd.DataFrame({
            "median_income": rng.uniform(1, 8, n),
            "housing_median_age": rng.integers(1, 50, n),
            "total_rooms": rng.integers(100, 5000, n),
            "total_bedrooms": rng.uniform(10, 1000, n),
            "population": rng.integers(100, 5000, n),
            "households": rng.integers(10, 1000, n),
            "ocean_proximity": ["INLAND", "NEAR BAY"] * (n // 2),
        })
        data["median_house_value"] = 50000 * data["median_income"] + rng.normal(0, 1000, n)
        data.loc[0:9, "total_bedrooms"] = np.nan
        data.loc[10:14, "ocean_proximity"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            data.to_csv(path, index=False)
            result = run(path)
        self.assertEqual(result["test_rows"], 20)
        self.assertTrue(math.isfinite(result["r2"]))


if __name__ == "__main__":
    unittest.main()
