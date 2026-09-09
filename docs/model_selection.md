# Model selection

Phase 7 adds LightGBM and XGBoost because retail demand forecasting is mostly a tabular learning problem: demand lags, rolling statistics, price context, calendar features, store attributes, and product attributes are already assembled in Gold `features_master`.

LightGBM is the primary candidate because it trains quickly on wide tabular data and handles high-cardinality categorical columns without global target encoding. XGBoost is the challenger because its histogram tree method is strong on sparse and nonlinear feature interactions, and it gives a useful second opinion when feature effects are uneven across stores and SKUs.

Both models use count-aware objectives. LightGBM uses `poisson`, and XGBoost uses `count:poisson`, matching the non-negative integer nature of unit demand better than squared-error regression. The hyperparameters are fixed before validation: moderate learning rate, many boosting rounds with early stopping, constrained leaves/depth, row and feature subsampling, and L2 regularization. Those defaults are intentionally conservative so Phase 7 measures a credible first GBM benchmark rather than a tuned test-fold result.

WAPE is the primary metric because it reports total absolute error relative to total demand. It behaves well when some store/SKU/day rows have zero units and it aligns with inventory impact: missing 100 units across high-volume SKUs matters more than the same percentage error on tiny demand.
