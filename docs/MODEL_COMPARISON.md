# Grouped model comparison V1

Protocol recorded before scoring in [issue #14](https://github.com/Yusyer174/house-prices-prediction-portfolio/issues/14).

## Goal

Compare three fixed regression candidates with the same inputs, geographic partitions and metrics. This is development-only exploratory selection, not an independent final-test claim.

## Fixed design

- Reuse the seed-42 one-degree geographic partition. Exclude its reserved cells from all new model fitting and scoring.
- Use five deterministic GroupKFold folds within the development partition. Each development row is scored exactly once out of fold.
- Refit the original preprocessing pipeline within each fold; coordinates are only used for grouping.
- Primary metric: pooled out-of-fold MAE, weighted implicitly by rows. Report pooled RMSE/R² and descriptive fold MAE spread. Rank the three candidates by MAE (alphabetical ties); the dummy is a reference.
- No hyperparameter search, no post-result selection of the split, no reserved-cell scoring for the new candidates.

| Candidate | Fixed settings |
| --- | --- |
| Linear regression | Original pipeline and LinearRegression defaults |
| Random Forest | 100 trees, max_depth=12, min_samples_leaf=5, seed=42, n_jobs=1 |
| Gradient Boosting | 100 stages, max_depth=3, learning_rate=0.05, min_samples_leaf=5, seed=42 |
| Dummy | Mean of fold training targets |

## Reproduce

```sh
python -m pip install -r requirements.txt
python compare_models.py
python -m unittest discover -s tests -v
```

The script emits JSON, including the dataset SHA-256, all fold scores, the candidate ranking and reserved-row count. CI also runs the earlier baseline and geographic experiment.

## Interpret carefully

Fold scores are correlated and their standard deviation is descriptive, not a confidence interval. Pooled out-of-fold R² uses the pooled development target mean; the dummy is trained separately per fold. Results rank only these configurations, features and folds. Adjacent geographic cells may be correlated. The historical reserved cells have already been explored by the earlier linear experiment and cannot serve as a truly untouched final test. Do not tune on them or claim independent generalization. No real-world valuation use is intended.

## Learn from the experiment

1. Explain why a fresh pipeline is constructed inside every fold.
2. Explain how Random Forest and Gradient Boosting differ.
3. Explain why pooled MAE weights observations rather than averaging fold MAEs equally.
4. Compare the weakest fold with the pooled result before choosing a candidate.
