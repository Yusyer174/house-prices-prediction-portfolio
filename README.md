# California Housing Price Prediction

A machine-learning portfolio project exploring how housing and demographic features relate to median home values in California.

## Overview

The baseline uses linear regression, numeric median imputation and categorical one-hot encoding. Preprocessing is fitted on the training split only, using a scikit-learn pipeline. A fixed 80/20 train/test split (random state 42) makes the experiment reproducible.

![Income versus housing value](house_income_vs_value.png.png)

## Run locally

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python train.py
```

To explore the data interactively, run `jupyter notebook notebook.ipynb` from the repository root.

## Results

The earlier four-feature exploratory model reported test R² = 0.5013. The improved pipeline uses additional housing features, handles missing values without fitting preprocessing on the test set, and reports R², MSE, RMSE and MAE. Run the command above to obtain the current baseline metrics; no unverified performance figure is claimed.

## Repository layout

- `train.py` — reproducible training and evaluation
- `notebook.ipynb` — interactive exploration and baseline evaluation
- `housing.csv` — local historical dataset
- `requirements.txt` — Python dependencies
- `house_income_vs_value.png.png` — existing exploratory visualization

## Data and limitations

The original notebook loaded the California housing CSV from the [Hands-On Machine Learning example dataset](https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.csv). The local dataset is retained for reproducibility. The repository's MIT license covers the project code, not any third-party dataset rights; consult the upstream source for dataset terms.

This is an educational historical-data experiment, not a current property valuation service. Future improvements include geographic holdout validation, nonlinear baselines and error analysis across regions.

## License

Project code: [MIT](LICENSE).
