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


## Recorded results

Recorded [CI results](https://github.com/Yusyer174/house-prices-prediction-portfolio/actions/runs/46148760695) (2026-11-28, source commit `4c1a1b9`): 14,410 development rows across 44 cells; 6,230 reserved rows across 11 cells; zero reserved rows scored by this comparison. All 17 tests passed.

| Model | Pooled R² | RMSE (USD) | MAE (USD) |
| --- | ---: | ---: | ---: |
| Random Forest | 0.648948 | 66,782.07 | 47,578.25 |
| Gradient Boosting | 0.601424 | 71,158.99 | 50,763.68 |
| Linear regression | 0.557791 | 74,952.79 | 53,081.09 |
| Training-mean dummy | -0.067184 | 116,437.75 | 92,548.50 |

Random Forest ranks first by the predefined pooled MAE criterion (10.37% lower than the linear candidate), and has the lowest MAE among the three candidates in each of the five folds. This is a development-CV result, not independent final-test evidence or a claim of real-world prediction accuracy. The historical random and geographic baseline metrics remain separate.

### Per-fold MAE (USD)

| Fold | Validation rows | Validation cells | Linear | Random Forest | Gradient Boosting |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2882 | 7 | 74,507.88 | 61,868.88 | 69,843.49 |
| 2 | 2882 | 9 | 48,170.26 | 42,483.19 | 46,422.26 |
| 3 | 2883 | 9 | 48,183.48 | 44,727.44 | 47,093.76 |
| 4 | 2882 | 10 | 47,193.49 | 45,796.84 | 46,455.33 |
| 5 | 2881 | 9 | 47,350.04 | 43,014.33 | 44,002.48 |

Fold 1 is the hardest for all three candidates by MAE. Random Forest's fold MAE ranges from 42,483.19 to 61,868.88 USD. This variation remains material, and no confidence interval or independent generalization claim follows. The selected configuration is a candidate for a future independently reserved evaluation; no further tuning or scoring of historical reserved cells was performed by this comparison.

[Exact recorded JSON](MODEL_COMPARISON_RESULTS.json) preserves all model and fold scores and the dataset hash. The recorded source and hosted run above identify the evidence. The original baseline commands ran in the same workflow; they are not part of the candidate ranking.
