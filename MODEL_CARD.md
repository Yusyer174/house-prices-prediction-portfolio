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

The script rejects missing required CSV columns and non-finite or nonnumeric targets. Missing numeric features are imputed with the **training-set median**; categorical missing values use the **training-set most frequent** value, followed by one-hot encoding. The full preprocessing pipeline is fit only on the training split.

## Evaluation protocol

- Single random 80/20 split, `random_state=42`.
- 16,512 training examples; 4,128 held-out examples.
- No temporal validation or geographic holdout has been performed.

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

- Geographic and temporal generalization were **not** evaluated; nearby observations may be correlated.
- Historical census-like aggregates are proxies, not a current home-level appraisal.
- A single split gives no confidence interval or robustness guarantee.
- Baseline linear regression may underfit nonlinear interactions.
- Demographic or geographic features may reflect socioeconomic disparities. No group fairness or downstream-impact study has been performed.
- The dataset's provenance and limitations should be reviewed before any reuse.

## Next experiments (not yet claimed)

- Geographic holdout instead of random splitting.
- Multiple random seeds or cross-validation, without tuning on a reserved final test set.
- Error analysis by region and price segment.
- Nonlinear baselines evaluated under a pre-registered comparison protocol.
