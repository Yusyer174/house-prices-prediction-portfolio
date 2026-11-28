"""Tests of shared folds, reserved-cell isolation and comparison semantics."""
import unittest

import numpy as np
import pandas as pd

from compare_models import candidates, compare, development_folds
from train import CATEGORICAL, NUMERIC, TARGET


def synthetic():
    rows = np.arange(60, dtype=float)
    data = pd.DataFrame({name: rows + i + 1 for i, name in enumerate(NUMERIC)})
    data[CATEGORICAL[0]] = ["INLAND", "NEAR BAY"] * 30
    data[TARGET] = 1000 * rows + 2000
    data["latitude"] = 30 + (rows // 5) + 0.1
    data["longitude"] = -120.2
    return data


class GroupCVTests(unittest.TestCase):
    def test_each_development_row_validated_once_and_reserved_never_used(self):
        data = synthetic()
        dev, reserved, groups, folds = development_folds(data)
        counts = np.zeros(len(data), dtype=int)
        for tr, va in folds:
            self.assertFalse(set(groups[tr]) & set(groups[va]))
            self.assertFalse((set(groups[tr]) | set(groups[va])) & set(groups[reserved]))
            counts[va] += 1
        np.testing.assert_array_equal(counts[dev], 1)
        np.testing.assert_array_equal(counts[reserved], 0)

    def test_comparison_ignores_reserved_targets_and_features(self):
        data = synthetic()
        _, reserved, _, _ = development_folds(data)
        expected = compare(data)
        data.loc[reserved, TARGET] = np.nan
        data.loc[reserved, NUMERIC] = np.nan
        actual = compare(data)
        self.assertEqual(expected, actual)
        self.assertEqual(actual["reserved_rows_scored"], 0)
        self.assertEqual(len(actual["folds"]), 5)
        self.assertEqual(set(actual["candidate_ranking"]),
                         {"linear", "random_forest", "gradient_boosting"})

    def test_pooled_dummy_mae_weights_rows_not_folds(self):
        data = synthetic()
        dev, _, _, folds = development_folds(data)
        absolute_errors = []
        for tr, va in folds:
            absolute_errors.extend(abs(data[TARGET].iloc[va] - data[TARGET].iloc[tr].mean()))
        self.assertEqual(len(absolute_errors), len(dev))
        report = compare(data)
        self.assertAlmostEqual(report["models"]["training_mean_dummy"]["mae"],
                               np.mean(absolute_errors))

    def test_candidate_pipeline_instances_are_fresh(self):
        one, two = candidates(), candidates()
        for name in one:
            self.assertIsNot(one[name], two[name])
        self.assertEqual(one["random_forest"].get_params()["regression__n_jobs"], 1)

    def test_invalid_development_target_rejected(self):
        data = synthetic()
        dev, _, _, _ = development_folds(data)
        data.loc[dev[0], TARGET] = np.nan
        with self.assertRaisesRegex(ValueError, "Development targets"):
            compare(data)


if __name__ == "__main__":
    unittest.main()
