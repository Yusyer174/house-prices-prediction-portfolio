# Contributing

This repository is an educational machine-learning baseline, not a real-world valuation product. Focused bug reports, documentation corrections and reproducible experiments are welcome.

## Before a pull request

1. Open an issue for changes that could alter the dataset, target, split, metrics, or interpretation.
2. Work in a branch and use synthetic data for new failure-path tests where feasible.
3. Run `python -m unittest discover -s tests -v` and `python train.py` in a clean Python 3.11 environment.
4. Explain any metric change with the input dataset, dependencies, split strategy and exact commands.
5. Update [MODEL_CARD.md](MODEL_CARD.md) and [README.md](README.md) if limitations or reported results change.

Do **not** claim an improvement by choosing a better test split or repeatedly tuning against the held-out test set. Keep held-out evaluations independent.

## Reports

Include Python and dependency versions, expected and actual results, logs with private paths removed, and a minimal reproducible example. Never upload private datasets, access tokens, personal data, or credentials.

The dataset is third-party material. Check upstream usage rights before redistributing it elsewhere.
