"""Synthetic checks of diagnostic semantics and partition isolation."""
import unittest

import numpy as np
import pandas as pd

from compare_models import development_folds
from diagnose_forest import (FEATURES, REPEATS, aggregate_importance, diagnose,
                             error_summary, permutation_deltas, price_errors)
from train import CATEGORICAL, NUMERIC, TARGET


class IncomePredictor:
    def predict(self, frame):
        return frame["median_income"].to_numpy()


def synthetic():
    rows = np.arange(60, dtype=float)
    data = pd.DataFrame({name: rows + i + 1 for i, name in enumerate(NUMERIC)})
    data[CATEGORICAL[0]] = ["INLAND", "NEAR BAY"] * 30
    data[TARGET] = 1000 * rows + 2000
    data["latitude"] = 30 + (rows // 5) + 0.1
    data["longitude"] = -120.2
    return data


class DiagnosticTests(unittest.TestCase):
    def test_signed_error_and_empty_band(self):
        actual = np.array([100., 200.])
        predicted = np.array([150., 170.])
        self.assertEqual(error_summary(actual, predicted)["signed_error"], 10)
        self.assertEqual(error_summary(actual, predicted)["mae"], 40)
        self.assertIsNone(error_summary(actual[:0], predicted[:0])["mae"])

    def test_price_band_boundaries(self):
        actual = np.array([149999., 150000., 299999., 300000., 500000.])
        result = price_errors(actual, actual + 1)
        self.assertEqual([v["rows"] for v in result.values()], [1, 2, 1, 1])
        self.assertTrue(all(v["signed_error"] == 1 for v in result.values()))

    def test_permutation_detects_used_feature_and_preserves_frame(self):
        x = synthetic()[FEATURES]
        original = x.copy(deep=True)
        y = x["median_income"].to_numpy()
        deltas = permutation_deltas(IncomePredictor(), x, y, 0, np.random.default_rng(42))
        self.assertTrue(all(v > 0 for v in deltas["median_income"]))
        for feature in FEATURES:
            if feature != "median_income":
                self.assertEqual(deltas[feature], [0] * REPEATS)
        pd.testing.assert_frame_equal(x, original)

    def test_weighted_importance_preserves_negative_values(self):
        folds = [{"rows": 1, "permutation_deltas": {f: [10] * REPEATS for f in FEATURES}},
                 {"rows": 3, "permutation_deltas": {f: [-6] * REPEATS for f in FEATURES}}]
        report = aggregate_importance(folds)
        self.assertEqual(report[FEATURES[0]]["mean_mae_increase"], -2)
        self.assertEqual(report[FEATURES[0]]["fold_mean_min"], -6)

    def test_reserved_rows_do_not_change_diagnostics(self):
        data = synthetic()
        dev, reserved, _, _ = development_folds(data)
        before = diagnose(data)
        data.loc[reserved, NUMERIC + [TARGET]] = np.nan
        after = diagnose(data)
        self.assertEqual(before, after)
        self.assertEqual(after["reserved_rows_scored"], 0)
        self.assertEqual(sum(b["rows"] for b in after["price_bands"].values()), len(dev))


if __name__ == "__main__":
    unittest.main()
