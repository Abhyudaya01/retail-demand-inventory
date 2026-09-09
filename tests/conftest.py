"""Shared environment defaults for local tests."""

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set test defaults and restore the environment after each test."""
    monkeypatch.setenv("S3_BUCKET", "retail-demand-test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("DATABRICKS_VOLUME_ROOT", "/tmp/fake-volume")
    yield
