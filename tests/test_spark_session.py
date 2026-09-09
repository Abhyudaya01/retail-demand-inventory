"""Tests for the Spark session configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from retail_demand.spark.session import get_spark_session


def test_get_spark_session_returns_functional_delta_session(tmp_path: Path) -> None:
    """Inputs: local environment; outputs: assertions; side effects: runs Spark locally."""
    # Ensure DATABRICKS_RUNTIME_VERSION is not set so we get a local Delta session
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        del os.environ["DATABRICKS_RUNTIME_VERSION"]

    spark = get_spark_session(app_name="test-session")

    # Create a trivial DataFrame
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])

    # Write and read a Delta table in the temp dir
    delta_path = str(tmp_path / "test_delta")
    df.write.format("delta").save(delta_path)

    read_df = spark.read.format("delta").load(delta_path)
    assert read_df.count() == 2
    assert set(read_df.columns) == {"id", "val"}


def test_databricks_returns_active_session_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing Databricks session is returned without configuring Delta again."""
    from unittest.mock import patch

    from pyspark.sql import SparkSession

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
    sentinel = object()
    with patch.object(SparkSession, "getActiveSession", return_value=sentinel):
        assert get_spark_session(extra_conf={"ignored": "true"}) is sentinel
