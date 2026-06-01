# Validation Strategy

Follows the de-facto standard for Predictive Process Monitoring established by
**Teinemaa et al. (2019)** *"Outcome-oriented predictive process monitoring:
Review and benchmark"* (ACM TKDD, 13(2), Article 17).

## Temporal Split

Cases sorted by start timestamp:

- **80% earliest cases** → train pool
- **20% latest cases** → test set
- Of train pool: **latest 20%** → validation set
- Effective ratio: **64% train / 16% val / 20% test**

## Debiased Split

Cases whose `[start, end]` interval crosses the train/test boundary are
**dropped** to prevent leakage via shared temporal context.
(Teinemaa 2019, §4 "Common Pitfalls")

## Why No k-Fold CV?

Temporal data invalidates random folds. Optional: expanding-window CV via
`sklearn.model_selection.TimeSeriesSplit` for robustness checks only.

## Metrics

### Outcome (classification)
- **AUC-ROC** (primary) — robust to class imbalance
- **F1** (positive class) — operational relevance
- **Brier Score** — probability calibration

### Remaining Time (regression)
- **MAE in days** (primary) — interpretable for stakeholders
- **RMSE** (secondary) — penalizes outliers

## Implementation Notes

- Implemented in `t6_train.py` via a shared `split_temporal()` helper
- Split indices saved to `data/features/split_indices.json` for reproducibility
- Validation set used for early stopping (LSTM) and hyperparameter selection (XGBoost)