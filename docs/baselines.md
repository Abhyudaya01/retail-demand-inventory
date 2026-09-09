# Forecasting baselines

Phase 6 establishes simple, explainable forecasting baselines and logs each run to MLflow.
These models are the bar that later LightGBM and inventory-aware models must beat.

## Models

| Model | Formula | Why it is useful |
| --- | --- | --- |
| Naive | `yhat[t] = y[t-1]` | Hard to beat for stable daily demand; catches recency quickly. |
| Seasonal naive | `yhat[t] = y[t-7]` | Captures weekly retail seasonality and day-of-week behavior. |
| Moving average | `yhat[t] = mean(y[t-28:t-1])` | Smooths noisy demand while staying strictly lagged. |

All baselines use Spark DataFrame projections from the time-aware Gold feature table. The
prediction interface returns pandas only for the validation forecast rows that are logged as
MLflow artifacts.

## Metrics

- WAPE: primary metric. It measures total absolute error over total actual demand and remains
  stable across sparse store/SKU series.
- MAE: average absolute unit error, easy to explain to operators.
- RMSE: penalizes larger misses more heavily than MAE.
- SMAPE: percentage-style error guarded for zero-demand periods.
- Bias: signed error ratio. Positive values mean over-forecasting; negative values mean
  under-forecasting.

Every baseline fold logs overall metrics, parameters, a forecast CSV, and an
actual-vs-predicted scatter plot. Segment metrics use top-volume categories when `category`
is present in `features_master`; the current Gold table falls back to top-volume SKUs.

## Validation

The default experiment uses three expanding-window walk-forward folds, each with a 28-day
validation horizon and at least 365 days of training history. Validation windows are
contiguous and non-overlapping, so the comparison table reports a fair mean across folds.

## Local Run

Build Gold first, then run:

```bash
make baselines-local
```

Local MLflow tracking writes under `/tmp/retail-demand-mlruns` by default from the Makefile
target. On Databricks, the notebook uses the hosted MLflow tracking server and the configured
experiment path.
