"""End-to-end tests for the Silver builder."""

from __future__ import annotations

from collections.abc import Generator
from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from retail_demand.bronze.ingest import BronzeIngestor
from retail_demand.config import Settings
from retail_demand.data_generation.generator import SyntheticDataConfig, write_dataset
from retail_demand.silver.build import SilverBuilder
from retail_demand.silver.dq import SilverDQ
from retail_demand.spark.session import get_spark_session


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a Spark session for end-to-end Silver tests."""
    session = get_spark_session(app_name="test-silver-build")
    yield session


def test_silver_build_end_to_end_from_bronze_paths(
    spark: SparkSession, tmp_path: Path
) -> None:
    """Generator to Bronze to Silver produces path-backed Delta tables that pass DQ."""
    raw_root = tmp_path / "raw"
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    config = SyntheticDataConfig(
        num_stores=2,
        num_skus=8,
        num_days=21,
        end_date=date(2026, 1, 31),
        duplicate_sales_rate=0.02,
        negative_units_rate=0.02,
        wrong_store_case_rate=0.02,
        missing_price_rate=0.0,
    )
    write_dataset(config, target="local", output_path=str(raw_root), settings=Settings())
    BronzeIngestor(spark, str(raw_root), str(bronze_root), "default").ingest_all()

    results = SilverBuilder(spark, str(bronze_root), "local", str(silver_root)).build_all()

    assert list(results) == ["stores", "products", "calendar", "prices", "sales"]
    assert len({stats["silver_run_id"] for stats in results.values()}) == 1
    for table, stats in results.items():
        assert (silver_root / table / "_delta_log").exists()
        assert stats["silver_location"] == str(silver_root / table)
        assert stats["registered"] is False
        actual_rows = spark.read.format("delta").load(stats["silver_location"]).count()
        assert stats["rows_out"] == actual_rows

    checks = SilverDQ.run_all_checks(spark, str(silver_root))
    assert all(check["passed"] for check in checks.values())
