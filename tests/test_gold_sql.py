"""Tests for Gold SQL feature files."""

from __future__ import annotations

from collections.abc import Generator
from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from retail_demand.bronze.ingest import BronzeIngestor
from retail_demand.config import Settings
from retail_demand.data_generation.generator import SyntheticDataConfig, write_dataset
from retail_demand.gold.build import BUILD_ORDER, GoldBuilder, default_sql_dir
from retail_demand.silver.build import SilverBuilder
from retail_demand.spark.session import get_spark_session


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a Spark session for Gold SQL tests."""
    session = get_spark_session(app_name="test-gold-sql")
    yield session


@pytest.fixture(scope="module")
def built_gold(
    spark: SparkSession, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, dict]]:
    """Build a tiny generated Silver and Gold fixture."""
    root = tmp_path_factory.mktemp("gold-sql")
    raw_root = root / "raw"
    bronze_root = root / "bronze"
    silver_root = root / "silver"
    gold_root = root / "gold"
    config = SyntheticDataConfig(
        num_stores=1,
        num_skus=3,
        num_days=35,
        end_date=date(2026, 1, 31),
        missing_price_rate=0.0,
    )
    write_dataset(config, target="local", output_path=str(raw_root), settings=Settings())
    BronzeIngestor(spark, str(raw_root), str(bronze_root), "default").ingest_all()
    SilverBuilder(spark, str(bronze_root), "local", str(silver_root)).build_all()
    results = GoldBuilder(spark, str(silver_root), str(gold_root), default_sql_dir()).build_all()
    return gold_root, results


@pytest.mark.parametrize("table_name", BUILD_ORDER)
def test_gold_sql_outputs_rows_and_expected_columns(
    spark: SparkSession, built_gold: tuple[Path, dict[str, dict]], table_name: str
) -> None:
    """Every Gold SQL file writes a path-backed Delta table with expected columns."""
    gold_root, results = built_gold
    frame = spark.read.format("delta").load(str(gold_root / table_name))

    assert results[table_name]["rows_written"] == frame.count()
    assert "_gold_run_id" in frame.columns
    if table_name != "features_calendar":
        assert {"store_id", "sku_id", "date"}.issubset(frame.columns)
    else:
        assert {"date", "dow_sin", "days_to_next_holiday"}.issubset(frame.columns)
    assert frame.count() > 0

