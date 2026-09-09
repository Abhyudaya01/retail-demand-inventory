"""Bronze ingestion logic for the retail-demand project."""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, current_timestamp, input_file_name, lit

from retail_demand.bronze.schemas import PARTITION_COLS, TABLE_SCHEMAS
from retail_demand.utils.logging import get_logger

logger = get_logger(__name__)


def _is_volume_path(path: str) -> bool:
    """Return whether a path points at a Databricks Volume."""
    return path.removeprefix("dbfs:").startswith("/Volumes/")


class BronzeIngestor:
    """Ingests raw Parquet files into Bronze Delta tables."""

    def __init__(
        self,
        spark: SparkSession,
        source_root: str,
        bronze_root: str,
        schema_name: str,
    ) -> None:
        """Inputs: spark session and paths; outputs: none; side effects: stores config."""
        self.spark = spark
        self.source_root = source_root.rstrip("/")
        self.bronze_root = (
            bronze_root.rstrip("/")
            if ":" in bronze_root or bronze_root.startswith("/")
            else str(Path(bronze_root).resolve())
        )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", schema_name):
            raise ValueError("schema_name must be a schema or catalog.schema identifier")
        if "REPLACE_WITH" in self.bronze_root:
            raise ValueError("Configure bronze_root with your Unity Catalog external location.")
        self._run_id: str | None = None
        self.schema_name = schema_name

    def ingest_table(
        self,
        table_name: str,
        mode: Literal["overwrite", "append"] = "overwrite",
    ) -> dict[str, Any]:
        """Inputs: table name and mode; outputs: stats; side effects: writes Delta table."""
        start_time = time.monotonic()
        ingest_run_id = self._run_id or uuid.uuid4().hex
        if table_name not in TABLE_SCHEMAS:
            raise ValueError(f"Unknown source table: {table_name}")
        if mode not in ("overwrite", "append"):
            raise ValueError("mode must be overwrite or append")

        source_path = f"{self.source_root}/{table_name}"
        bronze_location = f"{self.bronze_root}/{table_name}"

        schema = TABLE_SCHEMAS[table_name]
        partition_cols = PARTITION_COLS[table_name]

        # Read source data with explicit schema
        df = self.spark.read.schema(schema).parquet(source_path)

        # Add ingestion metadata
        df_enriched = (
            df.withColumn("_ingested_at", current_timestamp())
            .withColumn(
                "_source_path",
                col("_metadata.file_path")
                if "DATABRICKS_RUNTIME_VERSION" in os.environ
                else input_file_name(),
            )
            .withColumn("_ingest_run_id", lit(ingest_run_id))
        )

        # Write to Delta
        writer = df_enriched.write.format("delta").mode(mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)

        if mode == "overwrite":
            writer = writer.option("partitionOverwriteMode", "static")
        writer.save(bronze_location)

        registered = False
        if _is_volume_path(bronze_location):
            logger.info(
                "Skipping Unity Catalog registration for Volume-backed Bronze table %s; "
                "downstream reads will use path %s.",
                table_name,
                bronze_location,
            )
        else:
            # Register in Unity Catalog (or local metastore)
            escaped_location = bronze_location.replace("\\", "\\\\").replace("'", "\\'")
            self.spark.sql(
                f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} "
                f"USING DELTA LOCATION '{escaped_location}'"
            )
            registered = True

        written_df = self.spark.read.format("delta").load(bronze_location)
        if mode == "append":
            # only count the rows for this run id
            stats_df = written_df.filter(written_df._ingest_run_id == ingest_run_id)
        else:
            stats_df = written_df

        counts = stats_df.agg(
            count("*").alias("rows"), countDistinct("_source_path").alias("files")
        ).first()
        rows_written = counts["rows"]
        source_files = counts["files"]

        duration = round(time.monotonic() - start_time, 3)

        stats = {
            "rows_written": rows_written,
            "source_files": source_files,
            "bronze_location": bronze_location,
            "ingest_run_id": ingest_run_id,
            "duration_seconds": duration,
            "registered": registered,
        }

        logger.info(
            "bronze_ingest_table_complete",
            table=table_name,
            rows=rows_written,
            location=bronze_location,
            mode=mode,
        )

        return stats

    def ingest_all(
        self, mode: Literal["overwrite", "append"] = "overwrite"
    ) -> dict[str, dict[str, Any]]:
        """Inputs: write mode; outputs: per-table stats; side effects: writes all Delta tables."""
        tables = ["stores", "products", "calendar", "prices", "sales"]
        results = {}
        self._run_id = uuid.uuid4().hex
        try:
            for table in tables:
                results[table] = self.ingest_table(table, mode=mode)
            return results
        finally:
            self._run_id = None
