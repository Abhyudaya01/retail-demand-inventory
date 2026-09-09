"""Smoke checks for Bronze Delta tables; business data quality belongs in Silver."""

from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min

from retail_demand.utils.logging import get_logger

logger = get_logger(__name__)
METADATA_COLS = ("_ingested_at", "_source_path", "_ingest_run_id")


def validate_bronze_table(
    spark: SparkSession,
    bronze_location: str,
    expected_min_rows: int = 1,
) -> dict[str, Any]:
    """Read Delta and check minimum rows plus present, non-null ingestion metadata.

    Return counts, ISO timestamp bounds and checks_passed. Missing columns return
    a failed check; unreadable tables raise their underlying Spark exception.
    No business columns are validated or modified.
    """
    if expected_min_rows < 0:
        raise ValueError("expected_min_rows must be nonnegative")
    df = spark.read.format("delta").load(bronze_location)
    missing = [name for name in METADATA_COLS if name not in df.columns]
    if missing:
        result = {
            "rows": df.count(),
            "distinct_ingest_run_ids": 0,
            "min_ingested_at": None,
            "max_ingested_at": None,
            "checks_passed": False,
        }
    else:
        stats = df.agg(
            count("*").alias("rows"),
            countDistinct("_ingest_run_id").alias("runs"),
            spark_min("_ingested_at").alias("min_at"),
            spark_max("_ingested_at").alias("max_at"),
            *[count(col(name)).alias(name) for name in METADATA_COLS],
        ).first()
        result = {
            "rows": stats["rows"],
            "distinct_ingest_run_ids": stats["runs"],
            "min_ingested_at": stats["min_at"].isoformat() if stats["min_at"] else None,
            "max_ingested_at": stats["max_at"].isoformat() if stats["max_at"] else None,
            "checks_passed": stats["rows"] >= expected_min_rows
            and all(stats[name] == stats["rows"] for name in METADATA_COLS),
        }
    logger.info("bronze_validation_complete", location=bronze_location, **result)
    return result
