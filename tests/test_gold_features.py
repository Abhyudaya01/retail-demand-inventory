"""Feature-value tests for Gold SQL."""

from __future__ import annotations

from collections.abc import Generator
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from retail_demand.gold.build import GoldBuilder, default_sql_dir
from retail_demand.spark.session import get_spark_session

DAY_NS = 86_400_000_000_000


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a Spark session for Gold feature tests."""
    session = get_spark_session(app_name="test-gold-features")
    yield session


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


def test_gold_lag_and_rolling_features_match_manual_values(
    spark: SparkSession, tmp_path: Path
) -> None:
    """A 90-day single series produces manually verifiable lag and rolling features."""
    silver_root = tmp_path / "silver"
    gold_root = tmp_path / "gold"
    _write_single_series_silver(spark, silver_root)

    GoldBuilder(spark, str(silver_root), str(gold_root), default_sql_dir()).build_all()
    target_date = _ns(date(2026, 1, 15))
    row = (
        spark.read.format("delta")
        .load(str(gold_root / "features_master"))
        .filter(f"date = {target_date}")
        .first()
    )

    assert row["label_units_sold"] == 14
    assert row["lag_7_units"] == 7
    assert row["rolling_mean_7"] == pytest.approx(sum(range(7, 14)) / 7)
    assert row["rolling_zero_rate_7"] == 0.0
