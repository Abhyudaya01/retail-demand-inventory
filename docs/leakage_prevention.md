# Leakage prevention

Phase 7 uses walk-forward validation only. Each fold trains on earlier dates and validates on a later contiguous horizon.

Leakage vectors considered:

- Train/validation date overlap: guarded by a strict `train.date.max() < val.date.min()` assertion before every fit call.
- Feature-level future leakage: guarded by the Phase 5 Gold leakage audit and by a pre-fit check that flags large train/validation feature NaN-rate shifts.
- Random-shuffled CV: never used; all splits come from `walk_forward_splits`.
- Global target encoding of categoricals: not used; LightGBM and XGBoost receive native categorical columns.
- Hyperparameter tuning on test folds: not used; Phase 7 model defaults are fixed before cross-validation.
- Leakage through imputation: Silver price imputation uses last observation carried forward within prior weeks only.

If a leakage assertion fails, the experiment raises immediately and does not fit that fold.
