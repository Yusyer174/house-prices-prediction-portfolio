# California Housing Price Prediction

[![Baseline validation](https://github.com/Yusyer174/house-prices-prediction-portfolio/actions/workflows/baseline.yml/badge.svg)](https://github.com/Yusyer174/house-prices-prediction-portfolio/actions/workflows/baseline.yml)

An educational scikit-learn regression baseline for historical California median housing values. The code, dataset and fixed split are included so the evaluation can be reproduced.

## Overview

The baseline uses linear regression, numeric median imputation and categorical one-hot encoding. Preprocessing is fitted on the training split only, using a scikit-learn pipeline. A fixed 80/20 train/test split (random state 42) makes the experiment reproducible.

## Run locally

Use Python 3.11 to match the recorded CI environment.

```bash
git clone https://github.com/Yusyer174/house-prices-prediction-portfolio.git
cd house-prices-prediction-portfolio
python -m pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -v
```

To explore the data interactively, run `jupyter notebook notebook.ipynb` from the repository root.

## Results

The earlier four-feature exploratory model reported test R² = 0.5013. The improved pipeline uses additional housing features and handles missing values using preprocessing fitted on training data only.

**Historical holdout results** from the [recorded GitHub Actions run](https://github.com/Yusyer174/house-prices-prediction-portfolio/actions/runs/43220651075) on Python 3.11.

| Metric | Value |
| --- | ---: |
| Training rows | 16,512 |
| Test rows | 4,128 |
| R² | 0.616119 |
| MSE (USD²) | 5,030,408,039 |
| RMSE (USD) | 70,925.37 |
| MAE (USD) | 51,215.08 |

R² measures explained variance relative to a mean-prediction baseline; it is not classification accuracy. The results are from a single random split and do not establish geographic or temporal generalization.

## Explore the data

![Scatter plot of median income against median California housing value in US dollars](house_income_vs_value.png.png)

The plot shows a relationship in the input data. The holdout metrics above measure model performance.

## Repository layout

- `train.py` — reproducible training and evaluation
- `notebook.ipynb` — interactive exploration and baseline evaluation
- `housing.csv` — local historical dataset
- `requirements.txt` — Python dependencies
- [`house_income_vs_value.png.png`](house_income_vs_value.png.png) — exploratory income versus housing value plot

## Data and limitations

The original notebook loaded the California housing CSV from the [Hands-On Machine Learning example dataset](https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv). The local dataset is retained for reproducibility. The repository's MIT license covers the project code, not any third-party dataset rights; consult the upstream source for dataset terms.

This is an educational historical-data experiment, not a current property valuation service. See the [model card](MODEL_CARD.md) for the dataset provenance, intended use, evaluation protocol, fairness considerations and limitations.

The training script checks required input columns and rejects missing, nonnumeric, or non-finite target values with a clear error before fitting. Future improvements include geographic holdout validation, nonlinear baselines and error analysis across regions.

## Contributing

Focused contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and include tests and reproducible evidence when proposing changes to features or metrics.

## License

Project code: [MIT](LICENSE).
