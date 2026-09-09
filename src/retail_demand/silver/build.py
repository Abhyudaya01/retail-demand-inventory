"""Build path-backed Silver Delta tables from Bronze Delta tables."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from retail_demand.silver.cleaning import CLEANERS
from retail_demand.silver.schemas import PARTITION_COLS
from retail_demand.utils.logging import get_logger

logger = get_logger(__name__)


class SilverBuilder:
    """Build Silver Delta tables from Bronze Delta paths."""

    def __init__(
        self,
        spark: SparkSession,
        bronze_root: str,
        silver_schema: str,
        silver_root: str,
    ) -> None:
        """Store path roots; silver_schema is retained as a run label, not a catalog source."""
        self.spark = spark
        self.bronze_root = self._normalize_root(bronze_root)
        self.silver_schema = silver_schema
        self.silver_root = self._normalize_root(silver_root)
        self._run_id: str | None = None

    @staticmethod
    def _normalize_root(root: str) -> str:
        return (
            root.rstrip("/")
            if ":" in root or root.startswith("/")
            else str(Path(root).resolve())
        )

    def build_table(
        self,
        table_name: str,
        mode: Literal["overwrite", "append"] = "overwrite",
    ) -> dict[str, Any]:
        """Read Bronze path, clean rows, write Silver path and return metrics."""
        if table_name not in CLEANERS:
            raise ValueError(f"Unknown Silver table: {table_name}")
        if mode not in ("overwrite", "append"):
            raise ValueError("mode must be overwrite or append")

        start_time = time.monotonic()
        silver_run_id = self._run_id or uuid.uuid4().hex
        bronze_location = f"{self.bronze_root}/{table_name}"
        silver_location = f"{self.silver_root}/{table_name}"

        bronze_df = self.spark.read.format("delta").load(bronze_location)
        cleaned_df, metrics = CLEANERS[table_name](bronze_df)
        bronze_run_id = (
            bronze_df.agg(F.first(F.col("_ingest_run_id"), ignorenulls=True).alias("run_id"))
            .first()["run_id"]
            if "_ingest_run_id" in bronze_df.columns
            else "unknown"
        )
        silver_df = (
            cleaned_df.withColumn("_silvered_at", F.current_timestamp())
            .withColumn("_silver_run_id", F.lit(silver_run_id))
            .withColumn("_bronze_run_id", F.lit(bronze_run_id or "unknown"))
        )

        writer = silver_df.write.format("delta").mode(mode)
        partition_cols = PARTITION_COLS[table_name]
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        if mode == "overwrite":
            writer = writer.option("partitionOverwriteMode", "static")
        writer.save(silver_location)

        result = {
            **metrics,
            "bronze_location": bronze_location,
            "silver_location": silver_location,
            "silver_run_id": silver_run_id,
            "duration_seconds": round(time.monotonic() - start_time, 3),
            "registered": False,
        }
        logger.info("silver_build_table_complete", table=table_name, **result)
        return result

    def build_all(
        self, mode: Literal["overwrite", "append"] = "overwrite"
    ) -> dict[str, dict[str, Any]]:
        """Build all Silver tables in dependency order and share one Silver run ID."""
        results: dict[str, dict[str, Any]] = {}
        self._run_id = uuid.uuid4().hex
        try:
            for table in ["stores", "products", "calendar", "prices", "sales"]:
                results[table] = self.build_table(table, mode=mode)
            return results
        finally:
            self._run_id = None
