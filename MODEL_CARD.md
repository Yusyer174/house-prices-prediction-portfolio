# Model card: California housing linear-regression baseline

## Overview

- **Task:** Predict historical California median house value from tabular housing and demographic features.
- **Type:** Educational supervised regression baseline using scikit-learn `LinearRegression`.
- **Status:** Reproducible research exercise, **not** a deployed service.
- **Code:** [train.py](train.py) · [Regression tests](tests/test_train.py)
- **Dataset:** Local `housing.csv`, originally sourced from the [Hands-On Machine Learning California housing example](https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv). Dataset ownership and license belong to its upstream rights holders; the repository MIT license applies to project code only.

## Intended use

Learning about train/test splitting, numeric and categorical preprocessing, baseline metrics, and reproducibility. Suitable for comparing independently evaluated educational models on the **same holdout protocol**.

**Do not use** for pricing properties, mortgages, lending, eligibility, insurance, investment decisions, or decisions affecting individuals. The data is historical and may not represent current conditions.

## Inputs and preprocessing

- Numeric features: `median_income`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`.
- Categorical feature: `ocean_proximity`.
- Target: `median_house_value`.

The script rejects missing required CSV columns and non-finite or nonnumeric targets. Custom CSVs with fewer than six rows are rejected because the 20% holdout would not contain two test observations; this is a mathematical minimum, not sufficient sample size for meaningful validation. Missing numeric features are imputed with the **training-set median**; categorical missing values use the **training-set most frequent** value, followed by one-hot encoding. The full preprocessing pipeline is fit only on the training split.

## Evaluation protocol

- Single random 80/20 split, `random_state=42`.
- 16,512 training examples; 4,128 held-out examples.
- This historical random-split protocol is distinct from the geographic experiment below. No temporal validation has been performed.

| Test-set metric | Recorded baseline |
| --- | ---: |
| R² | 0.616119 |
| MSE | 5,030,408,039 USD² |
| RMSE | 70,925.37 USD |
| MAE | 51,215.08 USD |

These values are from a previously recorded [CI execution](https://github.com/Yusyer174/house-prices-prediction-portfolio/actions/runs/43220651075) on Python 3.11. They are **not** a prediction guarantee and should not be described as accuracy percentage.

## Reproduction

```bash
python -m pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -v
```

Use Python 3.11 for the recorded baseline; see `requirements.txt` for package versions. The command reads the checked-in local dataset by default. A different CSV can be passed via `python train.py --data path/to/file.csv`, provided it follows the same schema.

## Limitations and risks

- Temporal generalization was **not** evaluated. The exploratory geographic split does not exclude adjacent cells or establish robust transfer to arbitrary regions.
- Historical census-like aggregates are proxies, not a current home-level appraisal.
- A single split gives no confidence interval or robustness guarantee.
- Baseline linear regression may underfit nonlinear interactions.
- Demographic or geographic features may reflect socioeconomic disparities. No group fairness or downstream-impact study has been performed.
- The dataset's provenance and limitations should be reviewed before any reuse.

## Separate geographic experiment

The new script `evaluate_geographic.py` defines a fixed one-degree group holdout (20% of cells, seed 42) with no shared cells between train and test. Coordinates define groups only and are not added as features. The same preprocessing and regression pipeline is fitted on the training partition, alongside a training-mean dummy comparator. The recorded results below are separate from the unchanged historical metrics above. This tests transfer to held-out cells, not arbitrary new regions or future prices. Neighboring cells may be spatially correlated, and a single group split is not a robustness estimate.

Recorded [CI results](https://github.com/Yusyer174/house-prices-prediction-portfolio/actions/runs/46148453511) (Python 3.11.18, 2026-11-28, commit `195f01b`): 44 training cells / 14,410 rows; 11 held-out cells / 6,230 rows (30.18% of rows); zero overlapping cells.

| Model | R² | RMSE (USD) | MAE (USD) |
| --- | ---: | ---: | ---: |
| Linear regression | 0.618172 | 74,424.66 | 50,958.57 |
| Training-mean dummy | -0.021996 | 121,760.81 | 99,167.70 |

The linear model outperforms the dummy on the same held-out observations. The random-split and geographic metrics are not a controlled head-to-head model comparison because the evaluated rows and training sizes differ. These are exploratory historical results, not proof of geographic robustness. All 12 unit tests passed in the cited run.

## Development-only nonlinear comparison

The separate [comparison protocol](docs/MODEL_COMPARISON.md) compares three fixed candidates on five geographic folds within the original development partition. It does not alter this model card's historical baseline or score new candidates on the previously held-out cells. The resulting ranking is exploratory, not evidence of independent final-test performance.

## Next experiments (not yet claimed)

- Buffered geographic validation and additional predefined group splits.
- Multiple random seeds or cross-validation, without tuning on a reserved final test set.
- Error analysis by region and price segment.
- Independently reserved evaluation data for a frozen selected model.
