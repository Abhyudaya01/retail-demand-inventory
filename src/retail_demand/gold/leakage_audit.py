"""Leakage checks for the Gold features_master table."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

DAY_NS = 86_400_000_000_000
LAG_FEATURES = {
    "lag_1_units": 1,
    "lag_7_units": 7,
    "lag_14_units": 14,
    "lag_28_units": 28,
    "lag_365_units": 365,
}
ROLLING_WINDOWS = [7, 14, 28, 91]


def _sample_master(master: DataFrame, sample_size: int) -> DataFrame:
    total_rows = master.count()
    if total_rows <= sample_size:
        return master
    return master.orderBy(F.rand(seed=42)).limit(sample_size)


def _compare_feature(
    sample: DataFrame,
    expected: DataFrame,
    feature_col: str,
    expected_col: str,
) -> dict[str, int | bool]:
    joined = sample.select("store_id", "sku_id", "date", feature_col).join(
        expected.select("store_id", "sku_id", "date", expected_col),
        ["store_id", "sku_id", "date"],
        "left",
    )
    actual = F.col(feature_col).cast("double")
    expected_value = F.col(expected_col).cast("double")
    mismatches = joined.filter(
        (F.col(feature_col).isNull() != F.col(expected_col).isNull())
        | (
            F.col(feature_col).isNotNull()
            & F.col(expected_col).isNotNull()
            & (F.abs(actual - expected_value) > F.lit(1e-9))
        )
    )
    mismatch_count = mismatches.count()
    if mismatch_count:
        mismatches.show(20, truncate=False)
    return {
        "passes": mismatch_count == 0,
        "sample_size_checked": joined.count(),
        "mismatches": mismatch_count,
    }


def _expected_lags(stg: DataFrame) -> DataFrame:
    window = Window.partitionBy("store_id", "sku_id").orderBy("date")
    result = stg.select("store_id", "sku_id", "date", "units_sold")
    for feature, offset in LAG_FEATURES.items():
        result = result.withColumn(f"expected_{feature}", F.lag("units_sold", offset).over(window))
    return result.drop("units_sold")


def _expected_rolling(stg: DataFrame) -> DataFrame:
    result = stg.select("store_id", "sku_id", "date", "units_sold")
    for days in ROLLING_WINDOWS:
        window = (
            Window.partitionBy("store_id", "sku_id")
            .orderBy("date")
            .rowsBetween(-days, -1)
        )
        result = result.withColumn(
            f"expected_rolling_mean_{days}", F.avg("units_sold").over(window)
        )
        result = result.withColumn(
            f"expected_rolling_std_{days}", F.stddev_samp("units_sold").over(window)
        )
        result = result.withColumn(
            f"expected_rolling_zero_rate_{days}",
            F.avg(F.when(F.col("units_sold") == 0, F.lit(1.0)).otherwise(F.lit(0.0))).over(window),
        )
    return result.drop("units_sold")


def _expected_prices(gold_root: str) -> str:
    return gold_root.rstrip("/")


def audit_features_master(
    spark: SparkSession, gold_root: str, sample_size: int = 1000
) -> dict[str, dict[str, Any]]:
    """Audit sampled Gold feature rows for time-aware lag, rolling and price consistency."""
    root = _expected_prices(gold_root)
    master = spark.read.format("delta").load(f"{root}/features_master")
    stg = spark.read.format("delta").load(f"{root}/stg_sales_daily")
    price = spark.read.format("delta").load(f"{root}/features_price")
    sample = _sample_master(master, sample_size)
    results: dict[str, dict[str, Any]] = {}

    expected_lags = _expected_lags(stg)
    for feature in LAG_FEATURES:
        results[feature] = _compare_feature(
            sample, expected_lags, feature, f"expected_{feature}"
        )

    expected_rolling = _expected_rolling(stg)
    for days in ROLLING_WINDOWS:
        for metric in ("mean", "std", "zero_rate"):
            feature = f"rolling_{metric}_{days}"
            results[feature] = _compare_feature(
                sample, expected_rolling, feature, f"expected_{feature}"
            )

    expected_price = price.select(
        "store_id",
        "sku_id",
        "date",
        F.col("current_price").alias("expected_current_price"),
        F.col("price_week_start").alias("expected_price_week_start"),
    )
    results["current_price"] = _compare_feature(
        sample, expected_price, "current_price", "expected_current_price"
    )
    results["price_week_start"] = _compare_feature(
        sample, expected_price, "price_week_start", "expected_price_week_start"
    )

    failed = [name for name, result in results.items() if not result["passes"]]
    if failed:
        raise ValueError(f"Gold leakage audit failed: {', '.join(failed)}")
    return results
