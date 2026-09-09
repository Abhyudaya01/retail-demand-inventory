"""Command-line runner for local Bronze ingestion."""

from __future__ import annotations

import argparse
from typing import Literal

from retail_demand.bronze.ingest import BronzeIngestor
from retail_demand.spark.session import get_spark_session
from retail_demand.utils.logging import get_logger


def ingest(
    source_root: str = "data/synthetic",
    bronze_root: str = "data/bronze",
    mode: Literal["overwrite", "append"] = "overwrite",
) -> None:
    """Run the local pipeline at explicit paths, log statistics and stop Spark."""
    logger = get_logger(__name__)
    spark = get_spark_session(app_name="retail-demand-bronze-local")
    try:
        ingestor = BronzeIngestor(spark, source_root, bronze_root, "default")
        logger.info("starting_local_bronze_ingestion", source=source_root, dest=bronze_root)
        results = ingestor.ingest_all(mode=mode)
        for table, stats in results.items():
            logger.info("bronze_table_synced", table=table, **stats)
        logger.info("local_bronze_ingestion_complete")
    finally:
        spark.stop()


def main() -> None:
    """Parse local paths and write mode, then execute Bronze ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest raw Parquet into local Bronze Delta tables."
    )
    parser.add_argument("--source-root", default="data/synthetic")
    parser.add_argument("--bronze-root", default="data/bronze")
    parser.add_argument("--mode", choices=("overwrite", "append"), default="overwrite")
    args = parser.parse_args()
    ingest(args.source_root, args.bronze_root, args.mode)


if __name__ == "__main__":
    main()
