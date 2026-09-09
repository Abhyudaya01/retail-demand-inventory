"""Tests for Bronze data ingestion."""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from retail_demand.bronze.ingest import BronzeIngestor
from retail_demand.bronze.schemas import TABLE_SCHEMAS
from retail_demand.bronze.validation import validate_bronze_table
from retail_demand.config import Settings
from retail_demand.data_generation.generator import SyntheticDataConfig, write_dataset
from retail_demand.spark.session import get_spark_session


class _FakeColumn:
    def __init__(self, value: Any) -> None:
        self.value = value

    def alias(self, name: str) -> _FakeColumn:
        return self


class _FakeWriter:
    def __init__(self, spark: _FakeSpark) -> None:
        self.spark = spark

    def format(self, name: str) -> _FakeWriter:
        return self

    def mode(self, name: str) -> _FakeWriter:
        return self

    def option(self, name: str, value: str) -> _FakeWriter:
        return self

    def partitionBy(self, *columns: str) -> _FakeWriter:
        return self

    def save(self, location: str) -> None:
        self.spark.saved_locations.append(location)


class _FakeSourceFrame:
    def __init__(self, spark: _FakeSpark) -> None:
        self.spark = spark

    @property
    def write(self) -> _FakeWriter:
        return _FakeWriter(self.spark)

    def withColumn(self, name: str, value: Any) -> _FakeSourceFrame:
        return self


class _FakeStatsFrame:
    _ingest_run_id = "run-id"

    def filter(self, condition: Any) -> _FakeStatsFrame:
        return self

    def agg(self, *columns: Any) -> _FakeStatsFrame:
        return self

    def first(self) -> dict[str, int]:
        return {"rows": 2, "files": 1}


class _FakeReader:
    def __init__(self, spark: _FakeSpark) -> None:
        self.spark = spark

    def schema(self, schema: Any) -> _FakeReader:
        return self

    def parquet(self, path: str) -> _FakeSourceFrame:
        return _FakeSourceFrame(self.spark)

    def format(self, name: str) -> _FakeReader:
        return self

    def load(self, location: str) -> _FakeStatsFrame:
        return _FakeStatsFrame()


class _FakeSpark:
    def __init__(self) -> None:
        self.read = _FakeReader(self)
        self.saved_locations: list[str] = []
        self.sql_calls: list[str] = []

    def sql(self, statement: str) -> None:
        self.sql_calls.append(statement)


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a module-level Spark session."""
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        del os.environ["DATABRICKS_RUNTIME_VERSION"]
    session = get_spark_session(app_name="test-bronze-ingest")
    yield session


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a tiny dataset once per test module."""
    out_dir = tmp_path_factory.mktemp("raw")
    config = SyntheticDataConfig(num_stores=2, num_skus=20, num_days=60, end_date=date(2026, 1, 31))
    settings = Settings(s3_bucket="not-used")
    write_dataset(config, target="local", output_path=str(out_dir), settings=settings)
    return out_dir


def test_accepts_volume_bronze_root() -> None:
    """Volume-backed Bronze roots are valid Delta file destinations."""
    ingestor = BronzeIngestor(
        spark=_FakeSpark(),  # type: ignore[arg-type]
        source_root="/Volumes/workspace/retail_demand/raw",
        bronze_root="/Volumes/workspace/retail_demand/bronze",
        schema_name="workspace.retail_demand_bronze",
    )

    assert ingestor.bronze_root == "/Volumes/workspace/retail_demand/bronze"


def test_volume_bronze_root_skips_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Volume-backed Bronze writes by path and skips Unity Catalog registration."""
    monkeypatch.setattr("retail_demand.bronze.ingest.current_timestamp", lambda: "timestamp")
    monkeypatch.setattr("retail_demand.bronze.ingest.input_file_name", lambda: "source")
    monkeypatch.setattr("retail_demand.bronze.ingest.lit", lambda value: value)
    monkeypatch.setattr("retail_demand.bronze.ingest.count", lambda value: _FakeColumn(value))
    monkeypatch.setattr(
        "retail_demand.bronze.ingest.countDistinct", lambda value: _FakeColumn(value)
    )
    fake_spark = _FakeSpark()
    ingestor = BronzeIngestor(
        spark=fake_spark,  # type: ignore[arg-type]
        source_root="/Volumes/workspace/retail_demand/raw",
        bronze_root="/Volumes/workspace/retail_demand/bronze",
        schema_name="workspace.retail_demand_bronze",
    )

    stats = ingestor.ingest_table("stores")

    assert fake_spark.saved_locations == ["/Volumes/workspace/retail_demand/bronze/stores"]
    assert fake_spark.sql_calls == []
    assert stats["bronze_location"] == "/Volumes/workspace/retail_demand/bronze/stores"
    assert stats["registered"] is False


def test_ingest_table_adds_metadata_and_matches_row_count(
    spark: SparkSession,
    synthetic_data: Path,
    tmp_path: Path,
) -> None:
    """Inputs: raw dataset; outputs: assertions; side effects: writes Delta tables."""
    bronze_root = tmp_path / "bronze"
    ingestor = BronzeIngestor(
        spark=spark,
        source_root=str(synthetic_data),
        bronze_root=str(bronze_root),
        schema_name="default",
    )

    results = ingestor.ingest_all()
    assert list(results) == ["stores", "products", "calendar", "prices", "sales"]
    assert len({stats["ingest_run_id"] for stats in results.values()}) == 1
    for table, stats in results.items():
        assert len(stats["ingest_run_id"]) == 32

        # Verify location and Delta log exists
        delta_path = bronze_root / table
        assert (delta_path / "_delta_log").exists()

        # Verify metadata
        df = spark.read.format("delta").load(str(delta_path))
        files = list((synthetic_data / table).rglob("*.parquet"))
        assert (
            df.count()
            == stats["rows_written"]
            == sum(pq.read_metadata(file).num_rows for file in files)
        )
        assert stats["source_files"] == len(files)
        timestamp_field = {"calendar": "date", "prices": "week_start", "sales": "date"}.get(table)
        if timestamp_field:
            raw_values = [
                value
                for file in files
                for value in pq.ParquetFile(file)
                .read(columns=[timestamp_field])
                .column(timestamp_field)
                .cast(pa.int64())
                .to_pylist()
            ]
            assert sorted(raw_values) == sorted(
                row[0] for row in df.select(timestamp_field).collect()
            )
        source = spark.read.schema(TABLE_SCHEMAS[table]).parquet(str(synthetic_data / table))
        actual = df.select(*source.columns)
        assert actual.exceptAll(source).count() == 0
        assert source.exceptAll(actual).count() == 0
        assert validate_bronze_table(spark, str(delta_path))["checks_passed"]
        assert spark.table(f"default.{table}").count() == stats["rows_written"]
        columns = df.columns
        assert "_ingested_at" in columns
        assert "_source_path" in columns
        assert "_ingest_run_id" in columns


def test_ingest_overwrite_replaces_table(
    spark: SparkSession,
    synthetic_data: Path,
    tmp_path: Path,
) -> None:
    """Inputs: raw dataset; outputs: assertions; side effects: writes Delta table twice."""
    bronze_root = tmp_path / "bronze"
    ingestor = BronzeIngestor(
        spark=spark,
        source_root=str(synthetic_data),
        bronze_root=str(bronze_root),
        schema_name="default",
    )

    # First write
    stats1 = ingestor.ingest_table("stores", mode="overwrite")
    count1 = stats1["rows_written"]

    # Simulate a smaller dataset by filtering and rewriting raw source
    # We can just write again with overwrite. Since source is identical, count is identical
    # But if we want to assert it replaces, we can append first, then overwrite

    ingestor.ingest_table("stores", mode="append")
    df_appended = spark.read.format("delta").load(str(bronze_root / "stores"))
    assert df_appended.count() == count1 * 2

    # Now overwrite
    stats3 = ingestor.ingest_table("stores", mode="overwrite")
    df_overwritten = spark.read.format("delta").load(str(bronze_root / "stores"))
    assert df_overwritten.count() == count1
    assert stats3["rows_written"] == count1


def test_validate_bronze_table_passes_on_fresh_ingest(
    spark: SparkSession,
    synthetic_data: Path,
    tmp_path: Path,
) -> None:
    """Inputs: raw dataset; outputs: assertions; side effects: writes Delta table."""
    bronze_root = tmp_path / "bronze"
    ingestor = BronzeIngestor(
        spark=spark,
        source_root=str(synthetic_data),
        bronze_root=str(bronze_root),
        schema_name="default",
    )

    ingestor.ingest_table("products", mode="overwrite")

    validation = validate_bronze_table(spark, str(bronze_root / "products"), expected_min_rows=20)

    assert validation["checks_passed"] is True
    assert validation["rows"] == 20
    assert validation["distinct_ingest_run_ids"] == 1
    assert validation["min_ingested_at"] is not None
    assert validation["max_ingested_at"] is not None


@pytest.mark.parametrize("missing", [True, False])
def test_validation_rejects_missing_or_null_metadata(
    spark: SparkSession,
    tmp_path: Path,
    missing: bool,
) -> None:
    """Reject incomplete metadata without modifying any business rows."""
    from pyspark.sql.functions import current_timestamp, lit

    frame = spark.range(1)
    if not missing:
        frame = (
            frame.withColumn("_ingested_at", current_timestamp())
            .withColumn("_source_path", lit(None).cast("string"))
            .withColumn("_ingest_run_id", lit("test"))
        )
    location = str(tmp_path / "invalid")
    frame.write.format("delta").save(location)
    assert not validate_bronze_table(spark, location)["checks_passed"]


def test_partitioned_overwrite_removes_old_partitions(
    spark: SparkSession,
    synthetic_data: Path,
    tmp_path: Path,
) -> None:
    """A full overwrite removes partitions absent from the next source batch."""
    from pyspark.sql.functions import col

    raw = tmp_path / "raw"
    frame = spark.read.schema(TABLE_SCHEMAS["sales"]).parquet(str(synthetic_data / "sales"))
    frame.write.partitionBy("year", "month").parquet(str(raw / "sales"))
    ingestor = BronzeIngestor(spark, str(raw), str(tmp_path / "bronze"), "default")
    ingestor.ingest_table("sales")
    smaller = frame.filter(col("year") == 2026)
    smaller.write.mode("overwrite").partitionBy("year", "month").parquet(str(raw / "sales"))
    original = spark.conf.get("spark.sql.sources.partitionOverwriteMode")
    try:
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        result = ingestor.ingest_table("sales")
    finally:
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", original)
    actual = spark.read.format("delta").load(result["bronze_location"])
    assert actual.count() == smaller.count() < frame.count()
    assert actual.filter(col("year") != 2026).count() == 0
