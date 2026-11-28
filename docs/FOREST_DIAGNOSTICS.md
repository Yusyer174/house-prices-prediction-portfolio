# Random Forest diagnostics V1

Protocol fixed before results in [issue #16](https://github.com/Yusyer174/house-prices-prediction-portfolio/issues/16). Reuses the selected Random Forest configuration and the exact five geographic development folds from the [comparison](MODEL_COMPARISON.md). No new model selection or tuning.

## Run

```sh
python diagnose_forest.py
python -m unittest discover -s tests -v
```

## Measures

- Out-of-fold MAE, RMSE and signed error (prediction minus actual). Positive signed error means overprediction, negative means underprediction.
- Counts and errors by actual target bands: below 150,000; 150,000–under 300,000; 300,000–under 500,000; at least 500,000 USD. Empty bands report null metrics. These bands are descriptive, not decision thresholds.
- Raw-feature permutation importance: shuffle one feature within each validation fold, after training, and measure the increase in MAE over unpermuted predictions. Three repeats per feature with a fixed RNG seed. Average fold scores weighted by validation-row count, retaining negative values. Report repeat spread and minimum/maximum fold means.
- No individual prediction records, no scoring on reserved cells, no changes to the original model or training procedure.

## Interpretation

Permutation importance measures dependence on a feature for this fitted model and distribution, not a causal effect on prices. Correlated features can substitute for each other; shuffling can generate unrealistic combinations. Repeat spread is descriptive and not a confidence interval. Error differences by observed price band are not evidence of fairness or discrimination, nor estimates for current individual properties. The top price band may include capped historical labels. All results are exploratory development diagnostics after candidate selection, not independent final-test evidence.

## Questions for learning

1. Why does the signed mean error reveal something that MAE cannot?
2. Why do we permute validation features only after fitting?
3. Why can a feature have negative importance?
4. How could correlated variables make the ranking misleading?
