# Architecture

This project will use a medallion architecture for retail demand and inventory analytics.

```text
Synthetic sales generator
        |
        v
AWS S3 raw parquet
        |
        v
Databricks + Delta Lake
  Bronze -> Silver -> Gold
        |
        v
LightGBM demand forecasts + reorder decisions
        |
        v
Power BI inventory and service-level dashboards
```

Phase 1 creates the importable Python package, reproducible synthetic retail data generator,
and local quality checks. Later phases will add the S3 ingestion, Databricks notebooks/jobs,
Delta Lake transformations, forecasting, inventory optimization, and dashboard assets.
