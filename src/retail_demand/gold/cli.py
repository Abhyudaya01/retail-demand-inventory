"""Command-line runner for local Gold builds."""

from __future__ import annotations

import argparse
from typing import Literal

from retail_demand.gold.build import GoldBuilder, default_sql_dir
from retail_demand.gold.leakage_audit import audit_features_master
from retail_demand.spark.session import get_spark_session
from retail_demand.utils.logging import get_logger


def build(
    silver_root: str = "data/silver",
    gold_root: str = "data/gold",
    mode: Literal["overwrite", "append"] = "overwrite",
) -> None:
    """Build local Gold tables from Silver Delta paths, run leakage audit and stop Spark."""
    logger = get_logger(__name__)
    spark = get_spark_session(app_name="retail-demand-gold-local")
    try:
        builder = GoldBuilder(spark, silver_root, gold_root, default_sql_dir())
        results = builder.build_all(mode=mode)
        for table, stats in results.items():
            logger.info("gold_table_built", table=table, **stats)
        audit_features_master(spark, gold_root)
        logger.info("local_gold_build_complete", tables=len(results))
    finally:
        spark.stop()


def main() -> None:
    """Parse local paths and write mode, then build Gold."""
    parser = argparse.ArgumentParser(description="Build local Gold Delta feature tables.")
    parser.add_argument("--silver-root", default="data/silver")
    parser.add_argument("--gold-root", default="data/gold")
    parser.add_argument("--mode", choices=("overwrite", "append"), default="overwrite")
    args = parser.parse_args()
    build(args.silver_root, args.gold_root, args.mode)


if __name__ == "__main__":
    main()
