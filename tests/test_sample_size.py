"""Regression checks for the minimum sample size of the fixed holdout."""
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from train import CATEGORICAL, NUMERIC, TARGET, run


class SampleSizeTests(unittest.TestCase):
    def test_small_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            for n in (0, 1, 2, 3, 4, 5):
                with self.subTest(rows=n):
                    self.write_data(path, n)
                    with self.assertRaisesRegex(ValueError, "At least 6 rows"):
                        run(path)

    def test_six_rows_have_defined_r2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minimum.csv"
            self.write_data(path, 6)
            result = run(path)
        self.assertEqual(result["train_rows"], 4)
        self.assertEqual(result["test_rows"], 2)
        self.assertTrue(math.isfinite(result["r2"]))

    @staticmethod
    def write_data(path, n):
        data = {name: list(range(1, n + 1)) for name in NUMERIC}
        data[CATEGORICAL[0]] = ["INLAND"] * n
        data[TARGET] = [1000 * i for i in range(1, n + 1)]
        pd.DataFrame(data).to_csv(path, index=False)
