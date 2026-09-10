## Phase 7 — feature_importances.csv contains stale data

The `feature_importances.csv` artifact in MLflow runs shows importances from an older leaky model (revenue at 5.13M). The actual trained model uses the correct 44-feature set (no revenue), verified by inspecting `get_feature_columns` output post-fit. The `feature_importance_gain.json` and `.png` artifacts appear correct — only the CSV is stale.

Root cause: likely a logging bug in run_gbm.py where the CSV is written before or independent of the actual model fit. Fix in Phase 11 cleanup.
