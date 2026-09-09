"""Build path-backed Gold Delta feature tables from Silver Delta tables."""

from __future__ import annotations

import time
import uuid
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from retail_demand.utils.logging import get_logger

logger = get_logger(__name__)

BUILD_ORDER = [
    "stg_sales_daily",
    "features_lag",
    "features_rolling",
    "features_price",
    "features_promo",
    "features_calendar",
    "features_seasonal",
    "features_master",
]
SILVER_TABLES = ["stores", "products", "calendar", "prices", "sales"]


def default_sql_dir() -> str:
    """Return the package SQL directory path for local files and Databricks notebooks."""
    return str(resources.files("retail_demand.gold").joinpath("sql"))


class GoldBuilder:
    """Build Gold feature tables with SQL over path-backed temp views."""

    def __init__(
        self,
        spark: SparkSession,
        silver_root: str,
        gold_root: str,
        sql_dir: str | Path | None = None,
    ) -> None:
        """Store path roots and SQL directory; no catalog tables are read or registered."""
        self.spark = spark
        self.silver_root = self._normalize_root(silver_root)
        self.gold_root = self._normalize_root(gold_root)
        self.sql_dir = Path(sql_dir or default_sql_dir())
        self._run_id: str | None = None

    @staticmethod
    def _normalize_root(root: str) -> str:
        return (
            root.rstrip("/")
            if ":" in root or root.startswith("/")
            else str(Path(root).resolve())
        )

    def _create_path_views(self) -> None:
        for table in SILVER_TABLES:
            self.spark.read.format("delta").load(
                f"{self.silver_root}/{table}"
            ).createOrReplaceTempView(f"silver_{table}")
        for table in BUILD_ORDER:
            location = f"{self.gold_root}/{table}"
            try:
                self.spark.read.format("delta").load(location).createOrReplaceTempView(
                    f"gold_{table}"
                )
            except Exception:
                logger.debug("gold_dependency_view_missing", table=table, location=location)

    def _read_sql(self, table_name: str) -> str:
        sql_path = self.sql_dir / f"{table_name}.sql"
        return sql_path.read_text(encoding="utf-8").format(
            SILVER_ROOT=self.silver_root,
            GOLD_ROOT=self.gold_root,
        )

    def _with_partitions(self, df: DataFrame) -> DataFrame:
        if "date" not in df.columns:
            return df
        timestamp_col = F.to_timestamp(
            F.from_unixtime((F.col("date") / F.lit(1_000_000_000)).cast("long"))
        )
        result = df
        if "year" not in result.columns:
            result = result.withColumn("year", F.year(timestamp_col))
        if "month" not in result.columns:
            result = result.withColumn("month", F.month(timestamp_col))
        return result

    def build_table(
        self,
        table_name: str,
        mode: Literal["overwrite", "append"] = "overwrite",
    ) -> dict[str, Any]:
        """Run a Gold SQL file, write its Delta path and return build metrics."""
        if table_name not in BUILD_ORDER:
            raise ValueError(f"Unknown Gold table: {table_name}")
        if mode not in ("overwrite", "append"):
            raise ValueError("mode must be overwrite or append")

        start_time = time.monotonic()
        gold_run_id = self._run_id or uuid.uuid4().hex
        self._create_path_views()
        sql = self._read_sql(table_name)
        gold_df = self.spark.sql(sql)
        output_df = self._with_partitions(gold_df).withColumn("_gold_run_id", F.lit(gold_run_id))
        gold_location = f"{self.gold_root}/{table_name}"

        writer = output_df.write.format("delta").mode(mode)
        if "year" in output_df.columns and "month" in output_df.columns:
            writer = writer.partitionBy("year", "month")
        if mode == "overwrite":
            writer = writer.option("partitionOverwriteMode", "static")
        writer.save(gold_location)
        self.spark.read.format("delta").load(gold_location).createOrReplaceTempView(
            f"gold_{table_name}"
        )

        rows_written = self.spark.read.format("delta").load(gold_location).count()
        result = {
            "rows_written": rows_written,
            "gold_location": gold_location,
            "gold_run_id": gold_run_id,
            "duration_seconds": round(time.monotonic() - start_time, 3),
            "registered": False,
        }
        logger.info("gold_build_table_complete", table=table_name, **result)
        return result

    def build_all(
        self, mode: Literal["overwrite", "append"] = "overwrite"
    ) -> dict[str, dict[str, Any]]:
        """Build all Gold tables in dependency order with one shared run ID."""
        self._run_id = uuid.uuid4().hex
        results: dict[str, dict[str, Any]] = {}
        try:
            for table in BUILD_ORDER:
                results[table] = self.build_table(table, mode=mode)
            return results
        finally:
            self._run_id = None
