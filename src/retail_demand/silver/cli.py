"""Command-line runner for local Silver builds."""

from __future__ import annotations

import argparse
from typing import Literal

from retail_demand.silver.build import SilverBuilder
from retail_demand.silver.dq import SilverDQ
from retail_demand.spark.session import get_spark_session
from retail_demand.utils.logging import get_logger


def build(
    bronze_root: str = "data/bronze",
    silver_root: str = "data/silver",
    mode: Literal["overwrite", "append"] = "overwrite",
) -> None:
    """Build local Silver tables from Bronze Delta paths, run DQ and stop Spark."""
    logger = get_logger(__name__)
    spark = get_spark_session(app_name="retail-demand-silver-local")
    try:
        builder = SilverBuilder(spark, bronze_root, "local", silver_root)
        results = builder.build_all(mode=mode)
        for table, stats in results.items():
            logger.info("silver_table_built", table=table, **stats)
        checks = SilverDQ.run_all_checks(spark, silver_root)
        failed = [name for name, check in checks.items() if not check["passed"]]
        if failed:
            raise RuntimeError(f"Silver DQ checks failed: {', '.join(failed)}")
        logger.info("local_silver_build_complete", checks=len(checks))
    finally:
        spark.stop()


def main() -> None:
    """Parse local paths and write mode, then build Silver."""
    parser = argparse.ArgumentParser(description="Build local Silver Delta tables.")
    parser.add_argument("--bronze-root", default="data/bronze")
    parser.add_argument("--silver-root", default="data/silver")
    parser.add_argument("--mode", choices=("overwrite", "append"), default="overwrite")
    args = parser.parse_args()
    build(args.bronze_root, args.silver_root, args.mode)


if __name__ == "__main__":
    main()
