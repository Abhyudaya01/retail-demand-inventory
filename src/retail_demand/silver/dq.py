"""Data quality checks for path-backed Silver Delta tables."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from retail_demand.silver.schemas import NATURAL_KEYS

TABLES = ["stores", "products", "calendar", "prices", "sales"]


class SilverDQ:
    """Reusable Silver data quality checks."""

    @staticmethod
    def check_no_nulls(df: DataFrame, cols: list[str]) -> bool:
        """Return True when all listed columns are present and contain no null values."""
        if any(col not in df.columns for col in cols):
            return False
        condition = None
        for name in cols:
            next_condition = F.col(name).isNull()
            condition = next_condition if condition is None else condition | next_condition
        return df.filter(condition).count() == 0 if condition is not None else True

    @staticmethod
    def check_unique(df: DataFrame, cols: list[str]) -> bool:
        """Return True when the listed columns uniquely identify each row."""
        if any(col not in df.columns for col in cols):
            return False
        return df.count() == df.select(*cols).distinct().count()

    @staticmethod
    def check_referential_integrity(
        child_df: DataFrame, parent_df: DataFrame, join_col: str
    ) -> dict[str, int | bool]:
        """Return orphan count and pass flag for a single-column child-to-parent relation."""
        orphan_count = child_df.select(join_col).distinct().join(
            parent_df.select(join_col).distinct(), join_col, "left_anti"
        ).count()
        return {"orphan_count": orphan_count, "pass": orphan_count == 0}

    @staticmethod
    def check_row_count(df: DataFrame, min_rows: int) -> bool:
        """Return True when the DataFrame contains at least min_rows rows."""
        if min_rows < 0:
            raise ValueError("min_rows must be nonnegative")
        return df.count() >= min_rows

    @staticmethod
    def run_all_checks(spark: SparkSession, silver_schema_name: str) -> dict[str, dict[str, Any]]:
        """Run fixed Silver DQ checks against Delta tables loaded from a Silver root path."""
        silver_root = silver_schema_name.rstrip("/")
        frames = {
            table: spark.read.format("delta").load(f"{silver_root}/{table}") for table in TABLES
        }
        results: dict[str, dict[str, Any]] = {}

        def record(name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
            results[name] = {
                "check_name": name,
                "passed": passed,
                "details": details or {},
                "timestamp": datetime.now(UTC).isoformat(),
            }

        for table, keys in NATURAL_KEYS.items():
            record(
                f"{table}_natural_key_no_nulls",
                SilverDQ.check_no_nulls(frames[table], keys),
                {"columns": keys},
            )
            record(
                f"{table}_natural_key_unique",
                SilverDQ.check_unique(frames[table], keys),
                {"columns": keys},
            )

        relationships = [
            ("sales_store_id_in_stores", "sales", "stores", "store_id"),
            ("sales_sku_id_in_products", "sales", "products", "sku_id"),
            ("sales_date_in_calendar", "sales", "calendar", "date"),
            ("prices_store_id_in_stores", "prices", "stores", "store_id"),
            ("prices_sku_id_in_products", "prices", "products", "sku_id"),
        ]
        for name, child, parent, join_col in relationships:
            details = SilverDQ.check_referential_integrity(
                frames[child], frames[parent], join_col
            )
            record(name, bool(details["pass"]), details)

        return results

