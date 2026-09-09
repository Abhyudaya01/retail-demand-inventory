"""Tests for Silver cleaning transforms."""

from __future__ import annotations

from collections.abc import Generator
from datetime import date, datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from retail_demand.silver.cleaning import clean_calendar, clean_prices, clean_sales, clean_stores
from retail_demand.spark.session import get_spark_session


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a Spark session for Silver cleaning tests."""
    session = get_spark_session(app_name="test-silver-cleaning")
    yield session


def test_clean_sales_normalizes_dedupes_and_drops_negative(spark: SparkSession) -> None:
    """Dirty sales rows are normalized, deduped and filtered."""
    schema = T.StructType(
        [
            T.StructField("store_id", T.StringType(), True),
            T.StructField("sku_id", T.StringType(), True),
            T.StructField("date", T.LongType(), True),
            T.StructField("units_sold", T.LongType(), True),
            T.StructField("revenue", T.DoubleType(), True),
            T.StructField("year", T.IntegerType(), True),
            T.StructField("month", T.IntegerType(), True),
            T.StructField("_ingested_at", T.TimestampType(), True),
        ]
    )
    df = spark.createDataFrame(
        [
            ("STORE_0001", "sku_00001", 1, 2, 10.0, 2026, 1, datetime(2026, 1, 1, 1)),
            ("store_0001", "sku_00001", 1, 3, 15.0, 2026, 1, datetime(2026, 1, 1, 2)),
            ("store_0001", "sku_00002", 1, -1, -5.0, 2026, 1, datetime(2026, 1, 1, 3)),
        ],
        schema=schema,
    )

    cleaned, metrics = clean_sales(df)

    assert metrics["rows_in"] == 3
    assert metrics["rows_out"] == 1
    assert metrics["dupes_dropped"] == 1
    assert metrics["invalid_dropped"] == 1
    row = cleaned.first()
    assert row["store_id"] == "store_0001"
    assert row["units_sold"] == 2


def test_clean_prices_imputes_prior_price_and_drops_first_null(spark: SparkSession) -> None:
    """Price nulls are carried forward only when a prior price exists."""
    df = spark.createDataFrame(
        [
            ("STORE_0001", "sku_00001", 1, None, False, 2026),
            ("store_0001", "sku_00001", 2, 9.99, False, 2026),
            ("store_0001", "sku_00001", 3, None, True, 2026),
        ],
        "store_id string, sku_id string, week_start long, price double, is_promo boolean, year int",
    )

    cleaned, metrics = clean_prices(df)

    assert metrics["rows_in"] == 3
    assert metrics["rows_out"] == 2
    assert metrics["nulls_dropped"] == 1
    assert metrics["imputations_applied"] == 1
    rows = {row["week_start"]: row["price"] for row in cleaned.collect()}
    assert rows == {2: 9.99, 3: 9.99}
    assert {row["store_id"] for row in cleaned.collect()} == {"store_0001"}


def test_clean_light_tables_enforce_required_columns(spark: SparkSession, tmp_path: Path) -> None:
    """Dimension cleaners cast, drop required nulls and deduplicate natural keys."""
    stores = spark.createDataFrame(
        [
            ("store_0001", "West", "CA", "A", date(2020, 1, 1), 1000),
            ("store_0001", "West", "CA", "A", date(2020, 1, 1), 1000),
            (None, "West", "CA", "A", date(2020, 1, 1), 1000),
        ],
        "store_id string, region string, state string, store_type string, "
        "open_date date, sqft long",
    )
    calendar = spark.createDataFrame(
        [(1, 0, 1, 1, 1, 2026, False, False), (1, 0, 1, 1, 1, 2026, False, False)],
        "date long, dow int, week long, month int, quarter int, year int, "
        "is_weekend boolean, is_us_holiday boolean",
    )

    cleaned_stores, store_metrics = clean_stores(stores)
    cleaned_calendar, calendar_metrics = clean_calendar(calendar)

    assert cleaned_stores.count() == 1
    assert store_metrics["nulls_dropped"] == 1
    assert store_metrics["dupes_dropped"] == 1
    assert cleaned_calendar.count() == 1
    assert calendar_metrics["dupes_dropped"] == 1
