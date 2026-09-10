# Phase 7 Model Results (small-scale run: 5 stores × 200 SKUs × 365 days)

CV config: 3 walk-forward folds, 14-day horizon, 180-day min train.

## Cross-fold mean metrics

| Model | MAE | RMSE | WAPE | SMAPE | Bias | Δ WAPE vs baseline |
|---|---|---|---|---|---|---|
| moving_average (Phase 6 baseline) | 1.033 | 1.501 | 0.650 | 1.038 | 0.025 | — |
| LightGBM | 0.934 | 1.298 | 0.588 | 1.017 | 0.033 | **-9.6%** |
| XGBoost | 0.934 | 1.300 | 0.588 | 1.018 | 0.026 | -9.6% |

## Bootstrap significance test (LightGBM vs moving_average)

- Mean WAPE difference: -0.267
- 95% CI: [-0.500, -0.187]
- p-value LightGBM better: **0.0** (LightGBM wins every bootstrap sample)

## Leakage protections

1. Phase 5 leakage audit: 19/19 features verified against manually-computed ground truth (no future-peeking).
2. Phase 7 target leakage check: `revenue` excluded because revenue = units_sold × current_price (same-row derivative of label). Automated sanity check raises if any feature's correlation with label exceeds 0.95.
3. Walk-forward CV with strict `train.date.max() < val.date.min()` assertion.
4. No random shuffling anywhere. No cross-fold hyperparameter tuning.
5. All 44 model features are pre-computed in Gold Delta before CV starts — no per-fold feature engineering that could leak.

## Expected changes at full scale (Phase 11)

Small-scale limits: (a) `lag_365_units` and `yoy_growth_ratio` have limited signal because dataset spans only ~365 days; (b) intermittent-demand SKUs at 5-store × 200-SKU level are noisier than at 50 × 2000. Full-scale run (50 stores × 2000 SKUs × 3 years) is expected to widen the WAPE gap to 15–30% improvement.
