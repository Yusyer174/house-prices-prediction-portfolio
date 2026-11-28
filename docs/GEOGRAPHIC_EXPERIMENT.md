# Geographic holdout V1 — experiment record

Protocol specified in [issue #12](https://github.com/Yusyer174/house-prices-prediction-portfolio/issues/12) before execution. Source: [commit 195f01b](https://github.com/Yusyer174/house-prices-prediction-portfolio/commit/195f01b7e6c5d4297ae033bd15b8c419257b94a7). Evidence: [GitHub Actions run 46148453511](https://github.com/Yusyer174/house-prices-prediction-portfolio/actions/runs/46148453511), Python 3.11.18, 2026-11-28.

## Reproduce

```sh
python -m pip install -r requirements.txt
python train.py
python evaluate_geographic.py
python -m unittest discover -s tests -v
```

Recorded dataset SHA-256: `8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e`.

## Recorded output

```json
{
  "protocol": "one_degree_group_holdout_v1",
  "seed": 42,
  "requested_test_group_fraction": 0.2,
  "train_rows": 14410,
  "test_rows": 6230,
  "test_row_fraction": 0.3018410852713178,
  "train_groups": 44,
  "test_groups": 11,
  "group_overlap": 0,
  "linear_regression": {
    "r2": 0.6181720657438811,
    "rmse": 74424.65567290096,
    "mae": 50958.565766989
  },
  "training_mean_dummy": {
    "r2": -0.02199575829596956,
    "rmse": 121760.81090441476,
    "mae": 99167.69574561984
  },
  "dataset_sha256": "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e"
}
```

## Interpretation

- Group splitting assigns all rows in a cell to one partition; 20% of cells need not mean 20% of rows.
- Coordinates are used for grouping, not as input features. The pipeline fits imputers, encoder and regression only on training rows. The dummy fits only the training mean.
- R² uses the held-out target mean as its reference, whereas the dummy predicts the training mean; hence a negative dummy R² is possible.
- A higher R² but also higher RMSE across two different splits is possible: target variance and the distribution of residuals differ. Only the linear-versus-dummy comparison uses the identical held-out rows.
- This is one exploratory split with previously explored historical data; no untouched final test set, temporal holdout, spatial buffer, confidence interval or downstream fairness evidence is claimed. The fixed seed and cell size must not be changed to select a favorable outcome.

## Review questions

1. Why can adjacent observations on either side of a cell boundary still be similar?
2. Why must imputation and encoding be fitted after the split?
3. Why does a dummy predicting the training mean sometimes have negative R²?
4. What extra evaluation would be needed before claiming robust geographic generalization?

All 12 tests passed in the recorded hosted run. The original baseline was reproduced with R² 0.6161192328109408. This document records the evidence; it is not independent review or approval.
