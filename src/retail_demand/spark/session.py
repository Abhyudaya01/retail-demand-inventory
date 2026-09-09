"""Spark session configuration for retail-demand project."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "retail-demand-bronze",
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """Get or create a SparkSession, configuring Delta Lake if running locally.

    If DATABRICKS_RUNTIME_VERSION is set, this function returns the existing
    Databricks-managed SparkSession (which has Delta pre-configured) and avoids
    re-configuring or calling builder.getOrCreate().

    If running locally, it uses `delta.configure_spark_with_delta_pip` to initialize
    a Spark session capable of reading and writing Delta tables.

    Inputs
    ------
    app_name : Name of the Spark application.
    extra_conf : Additional Spark configuration properties to set.

    Outputs
    -------
    SparkSession instance.

    Side effects
    ------------
    May start a local Spark JVM if one does not already exist.
    """
    if os.environ.get("DATABRICKS_RUNTIME_VERSION") is not None:
        # On Databricks, the session is managed by the platform.
        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError(
                "No active Databricks SparkSession; pass the notebook spark session."
            )
        return session

    # Keep Python workers on the same interpreter as the local driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    # Local mode
    import delta

    builder = (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.databricks.delta.snapshotPartitions", "2")
    )

    if extra_conf:
        for key, value in extra_conf.items():
            builder = builder.config(key, value)

    return delta.configure_spark_with_delta_pip(builder).getOrCreate()
