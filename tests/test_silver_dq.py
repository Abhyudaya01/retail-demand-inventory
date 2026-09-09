"""Tests for Silver DQ checks."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from retail_demand.silver.dq import SilverDQ
from retail_demand.spark.session import get_spark_session


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a Spark session for Silver DQ tests."""
    session = get_spark_session(app_name="test-silver-dq")
    yield session


def test_basic_dq_checks_pass_and_fail(spark: SparkSession) -> None:
    """No-null, uniqueness, RI and row-count checks expose good and bad input."""
    good = spark.createDataFrame([(1, "a"), (2, "b")], "id int, value string")
    bad = spark.createDataFrame([(1, "a"), (1, None)], "id int, value string")
    parent = spark.createDataFrame([(1,), (2,)], "id int")
    child = spark.createDataFrame([(1,), (3,)], "id int")

    assert SilverDQ.check_no_nulls(good, ["id", "value"])
    assert not SilverDQ.check_no_nulls(bad, ["id", "value"])
    assert SilverDQ.check_unique(good, ["id"])
    assert not SilverDQ.check_unique(bad, ["id"])
    assert SilverDQ.check_referential_integrity(child, parent, "id") == {
        "orphan_count": 1,
        "pass": False,
    }
    assert SilverDQ.check_row_count(good, 2)
    assert not SilverDQ.check_row_count(good, 3)


def test_run_all_checks_reads_silver_paths(spark: SparkSession, tmp_path: Path) -> None:
    """The fixed suite passes when all path-backed Silver tables are clean."""
    silver_root = tmp_path / "silver"
    spark.createDataFrame(
        [("store_0001", "West", "CA", "A", "2020-01-01", 1000, "ts", "run", "bronze")],
        "store_id string, region string, state string, store_type string, open_date string, "
        "sqft long, _silvered_at string, _silver_run_id string, _bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "stores"))
    spark.createDataFrame(
        [("sku_00001", "Grocery", "Pantry", 9.99, False, 1, "ts", "run", "bronze")],
        "sku_id string, category string, subcategory string, base_price double, "
        "is_perishable boolean, pack_size long, _silvered_at string, _silver_run_id string, "
        "_bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "products"))
    spark.createDataFrame(
        [(1, 0, 1, 1, 1, 2026, False, False, "ts", "run", "bronze")],
        "date long, dow int, week long, month int, quarter int, year int, is_weekend boolean, "
        "is_us_holiday boolean, _silvered_at string, _silver_run_id string, _bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "calendar"))
    spark.createDataFrame(
        [("store_0001", "sku_00001", 1, 9.99, False, 2026, "ts", "run", "bronze")],
        "store_id string, sku_id string, week_start long, price double, is_promo boolean, "
        "year int, _silvered_at string, _silver_run_id string, _bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "prices"))
    spark.createDataFrame(
        [("store_0001", "sku_00001", 1, 2, 19.98, 2026, 1, "ts", "run", "bronze")],
        "store_id string, sku_id string, date long, units_sold long, revenue double, year int, "
        "month int, _silvered_at string, _silver_run_id string, _bronze_run_id string",
    ).write.format("delta").save(str(silver_root / "sales"))

    checks = SilverDQ.run_all_checks(spark, str(silver_root))

    assert checks
    assert all(check["passed"] for check in checks.values())

