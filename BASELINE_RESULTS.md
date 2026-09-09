# Phase 6 Baseline Results (small-scale run, 5 stores × 200 SKUs × 365 days)

CV config: 3 folds, 14-day horizon, 180-day min train.

| Model | MAE | RMSE | WAPE | SMAPE | Bias |
|---|---|---|---|---|---|
| moving_average | 1.033 | 1.501 | 0.650 | 1.038 | 0.025 |
| naive | 1.247 | 1.874 | 0.785 | 0.914 | 0.002 |
| seasonal_naive | 1.336 | 2.063 | 0.840 | 0.935 | 0.012 |

**Bar to beat in Phase 7 (LightGBM):** WAPE 0.65 from moving_average.
