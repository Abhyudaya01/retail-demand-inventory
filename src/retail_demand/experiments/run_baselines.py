"""Run Phase 6 baseline forecasts against Gold features_master."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from retail_demand.eval.metrics import metric_dict, per_segment_metrics
from retail_demand.eval.splits import walk_forward_splits
from retail_demand.mlflow_setup import configure_mlflow, log_forecast_run
from retail_demand.models.baselines import (
    BaselineForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from retail_demand.spark.session import get_spark_session

DEFAULT_EXPERIMENT_PATH = "/Users/alohani@umd.edu/retail-demand-forecasting"


def _spark_to_pandas(df: DataFrame) -> pd.DataFrame:
    """Collect a Spark DataFrame as pandas through Arrow."""
    return df.toArrow().to_pandas()


def _timestamp_to_ns(value: pd.Timestamp) -> int:
    return int(value.value)


def _filter_between(df: DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> DataFrame:
    return df.filter(
        (F.col("date") >= F.lit(_timestamp_to_ns(start)))
        & (F.col("date") <= F.lit(_timestamp_to_ns(end)))
    )


def _with_actuals(val_df: DataFrame, forecast_df: DataFrame) -> pd.DataFrame:
    actual_cols = ["store_id", "sku_id", "date", F.col("label_units_sold").alias("y_true")]
    if "category" in val_df.columns:
        actual_cols.append("category")
    actual_pd = _spark_to_pandas(val_df.select(*actual_cols))
    forecast_pd = _spark_to_pandas(forecast_df)
    return actual_pd.merge(forecast_pd, on=["store_id", "sku_id", "date"], how="inner")


def _top_segment_metrics(joined_pd: pd.DataFrame) -> dict[str, float]:
    segment_col = "category" if "category" in joined_pd.columns else "sku_id"
    segment_prefix = "category" if segment_col == "category" else "sku"
    top_segments = (
        joined_pd.groupby(segment_col)["y_true"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    segment_df = per_segment_metrics(
        joined_pd[joined_pd[segment_col].isin(top_segments)], [segment_col]
    )
    metrics: dict[str, float] = {}
    for _, row in segment_df.iterrows():
        segment_value = str(row[segment_col])
        for metric in ("mae", "rmse", "wape", "smape", "bias"):
            metrics[f"top5_{segment_prefix}_{segment_value}_{metric}"] = float(row[metric])
    return metrics


def _baseline_suite() -> list[BaselineForecaster]:
    return [NaiveForecaster(), SeasonalNaiveForecaster(), MovingAverageForecaster(window=28)]


def run_baselines(
    spark: SparkSession,
    gold_root: str,
    experiment_path: str = DEFAULT_EXPERIMENT_PATH,
    cv_folds: int = 3,
    horizon_days: int = 28,
    min_train_days: int = 365,
) -> pd.DataFrame:
    """Run all baselines over walk-forward folds, log MLflow runs and return comparison."""
    import mlflow

    configure_mlflow(experiment_path)
    features = spark.read.format("delta").load(f"{gold_root.rstrip('/')}/features_master")
    folds = walk_forward_splits(
        features.select("date"), cv_folds, horizon_days, min_train_days
    )
    rows: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="phase-6-baseline-comparison"):
        mlflow.set_tag("phase", "6-baselines")
        for baseline in _baseline_suite():
            for fold in folds:
                train_df = _filter_between(
                    features, fold["train_start"], fold["train_end"]
                )
                val_df = _filter_between(features, fold["val_start"], fold["val_end"])
                model = baseline.fit(train_df)
                forecast_df = model.predict(val_df)
                joined_pd = _with_actuals(val_df, forecast_df)
                metrics = metric_dict(
                    joined_pd["y_true"].to_numpy(), joined_pd["y_pred"].to_numpy()
                )
                metrics.update(_top_segment_metrics(joined_pd))
                params = {
                    "baseline_name": baseline.name,
                    "fold_id": fold["fold_id"],
                    "horizon": horizon_days,
                    "train_start": fold["train_start"].isoformat(),
                    "train_end": fold["train_end"].isoformat(),
                    "val_start": fold["val_start"].isoformat(),
                    "val_end": fold["val_end"].isoformat(),
                    "n_train_rows": train_df.count(),
                    "n_val_rows": val_df.count(),
                }
                log_forecast_run(
                    f"{baseline.name}-fold-{fold['fold_id']}",
                    baseline.name,
                    params,
                    metrics,
                    joined_pd,
                    int(fold["fold_id"]),
                )
                rows.append({"model": baseline.name, "fold_id": fold["fold_id"], **metrics})

    results = pd.DataFrame(rows)
    comparison = results.groupby("model")[["mae", "rmse", "wape", "smape", "bias"]].mean()
    comparison = comparison.reset_index().sort_values("wape")
    print(comparison.to_string(index=False))
    return comparison


def main() -> None:
    """Parse local CLI options and run baseline comparison."""
    parser = argparse.ArgumentParser(description="Run baseline forecast experiments.")
    parser.add_argument("--gold-root", default="data/gold")
    parser.add_argument("--experiment-path", default="retail-demand-baselines-local")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=28)
    parser.add_argument("--min-train-days", type=int, default=365)
    args = parser.parse_args()
    spark = get_spark_session(app_name="retail-demand-baselines-local")
    try:
        run_baselines(
            spark,
            args.gold_root,
            args.experiment_path,
            args.cv_folds,
            args.horizon_days,
            args.min_train_days,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
