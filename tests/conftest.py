"""Shared environment defaults for local tests."""

from collections.abc import Generator

import pytest
from pyspark.sql import SparkSession

from retail_demand.spark.session import get_spark_session


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set test defaults and restore the environment after each test."""
    monkeypatch.setenv("S3_BUCKET", "retail-demand-test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("DATABRICKS_VOLUME_ROOT", "/tmp/fake-volume")
    yield


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Provide a shared Spark session for tests that do not define a module fixture."""
    session = get_spark_session(app_name="retail-demand-tests")
    if session.sparkContext._jsc is None:
        session = get_spark_session(app_name="retail-demand-tests")
    yield session
    if session.sparkContext._jsc is not None:
        session.stop()
