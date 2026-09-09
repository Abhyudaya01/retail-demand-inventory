"""Tests for Gold leakage auditing."""

from __future__ import annotations

from collections.abc import Generator
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from retail_demand.gold.build import GoldBuilder, default_sql_dir
from retail_demand.gold.leakage_audit import audit_features_master
from retail_demand.spark.session import get_spark_session


def _ns(day: date) -> int:
    return int(datetime(day.year, day.month, day.day).timestamp() * 1_000_000_000)


def _write_single_series_silver(spark: SparkSession, silver_root: Path, days: int = 90) -> None:
    start = date(2026, 1, 1)
    calendar_rows = []
    sales_rows = []
    price_rows = []
    for offset in range(days):
        current = start + timedelta(days=offset)
        current_ns = _ns(current)
        calendar_rows.append(
            (
                current_ns,
                current.weekday(),
                int(current.strftime("%V")),
                current.month,
                (current.month - 1) // 3 + 1,
                current.year,
                current.weekday() >= 5,
                False,
                datetime(2026, 1, 1),
                "silver-run",
                "bronze-run",
            )
        )
        sales_rows.append(
            (
                "store_0001",
                "sku_00001",
                current_ns,
                offset,
                float(offset * 10),
                current.year,
                current.month,
                datetime(2026, 1, 1),
                "silver-run",
                "bronze-run",
            )
        )
    for offset in range(0, days, 7):
        current = start + timedelta(days=offset)
        price_rows.append(
            (
                "store_0001",
                "sku_00001",
                _ns(current),
                10.0 + offset,
                offset % 14 == 0,
                current.year,
                datetime(2026, 1, 1),
                "silver-run",
                "bronze-run",
            )
        )

    spark.createDataFrame(
        [
            (
                "store_0001",
                "West",
                "CA",
                "A",
                date(2020, 1, 1),
                1000,
                datetime(2026, 1, 1),
                "silver-run",
                "bronze-run",
            )
        ],
        "store_id string, region string, state string, store_type string, open_date date, "
        "sqft long, _silvered_at timestamp, _silver_run_id string, _bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "stores"))
    spark.createDataFrame(
        [
            (
                "sku_00001",
                "Grocery",
                "Pantry",
                10.0,
                False,
                1,
                datetime(2026, 1, 1),
                "silver-run",
                "bronze-run",
            )
        ],
        "sku_id string, category string, subcategory string, base_price double, "
        "is_perishable boolean, pack_size long, _silvered_at timestamp, _silver_run_id string, "
        "_bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "products"))
    spark.createDataFrame(
        calendar_rows,
        "date long, dow int, week long, month int, quarter int, year int, is_weekend boolean, "
        "is_us_holiday boolean, _silvered_at timestamp, _silver_run_id string, "
        "_bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "calendar"))
    spark.createDataFrame(
        price_rows,
        "store_id string, sku_id string, week_start long, price double, is_promo boolean, "
        "year int, _silvered_at timestamp, _silver_run_id string, _bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "prices"))
    spark.createDataFrame(
        sales_rows,
        "store_id string, sku_id string, date long, units_sold long, revenue double, year int, "
        "month int, _silvered_at timestamp, _silver_run_id string, _bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "sales"))


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a Spark session for Gold leakage tests."""
    session = get_spark_session(app_name="test-gold-leakage")
    yield session
    session.stop()


def test_leakage_audit_passes_clean_features_and_catches_corruption(
    spark: SparkSession, tmp_path: Path
) -> None:
    """The audit passes clean SQL output and catches an intentionally leaky lag feature."""
    silver_root = tmp_path / "silver"
    gold_root = tmp_path / "gold"
    _write_single_series_silver(spark, silver_root)
    GoldBuilder(spark, str(silver_root), str(gold_root), default_sql_dir()).build_all()

    clean_results = audit_features_master(spark, str(gold_root), sample_size=1000)
    print(
        "clean audit:",
        {
            "lag_7_units": clean_results["lag_7_units"],
            "rolling_mean_7": clean_results["rolling_mean_7"],
        },
    )
    assert clean_results["lag_7_units"]["passes"]
    assert clean_results["rolling_mean_7"]["passes"]

    master_path = str(gold_root / "features_master")
    leaky = spark.read.format("delta").load(master_path).withColumn(
        "lag_7_units",
        F.when(F.col("date").isNotNull(), F.col("label_units_sold")).otherwise(
            F.col("lag_7_units")
        ),
    )
    leaky.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).save(master_path)

    with pytest.raises(ValueError, match="lag_7_units") as exc_info:
        audit_features_master(spark, str(gold_root), sample_size=1000)
    print("intentional leak failure:", str(exc_info.value))
