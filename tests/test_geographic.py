"""Synthetic checks of the frozen geographic evaluation protocol."""
import unittest

import numpy as np
import pandas as pd

from evaluate_geographic import evaluate, geographic_groups, geographic_split
from train import NUMERIC, CATEGORICAL, TARGET


def synthetic():
    rows = np.arange(30, dtype=float)
    data = pd.DataFrame({name: rows + i + 1 for i, name in enumerate(NUMERIC)})
    data[CATEGORICAL[0]] = ["INLAND", "NEAR BAY"] * 15
    data[TARGET] = 1000 * rows + 2000
    data["latitude"] = 32 + (rows // 5) + 0.1
    data["longitude"] = -120.2
    return data


class GeographicTests(unittest.TestCase):
    def test_negative_longitude_floors(self):
        groups = geographic_groups(pd.DataFrame({"latitude": [34.1, 34.9],
                                                  "longitude": [-118.1, -118.9]}))
        self.assertEqual(groups.tolist(), ["34:-119", "34:-119"])

    def test_split_is_disjoint_complete_and_deterministic(self):
        data = synthetic()
        tr, te, groups = geographic_split(data)
        tr2, te2, _ = geographic_split(data)
        np.testing.assert_array_equal(tr, tr2)
        np.testing.assert_array_equal(te, te2)
        self.assertFalse(set(groups[tr]) & set(groups[te]))
        self.assertFalse(set(tr) & set(te))
        self.assertEqual(set(tr) | set(te), set(range(len(data))))

    def test_dummy_uses_training_mean(self):
        data = synthetic()
        tr, te, _ = geographic_split(data)
        result = evaluate(data)
        expected = np.abs(data[TARGET].iloc[te] - data[TARGET].iloc[tr].mean()).mean()
        self.assertAlmostEqual(result["training_mean_dummy"]["mae"], expected)
        self.assertEqual(result["group_overlap"], 0)
        self.assertEqual(result["train_rows"] + result["test_rows"], len(data))

    def test_invalid_coordinates_rejected(self):
        for bad in (np.nan, np.inf, 91, "unknown"):
            with self.subTest(value=bad):
                data = synthetic()
                data["latitude"] = data["latitude"].astype(object)
                data.loc[0, "latitude"] = bad
                with self.assertRaisesRegex(ValueError, "Latitude and longitude"):
                    geographic_split(data)

    def test_one_group_rejected(self):
        data = synthetic()
        data["latitude"] = 34.1
        with self.assertRaisesRegex(ValueError, "At least two geographic cells"):
            geographic_split(data)


if __name__ == "__main__":
    unittest.main()
