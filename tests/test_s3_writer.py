"""Tests for S3 parquet writer utilities."""

from __future__ import annotations

import logging

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from retail_demand.config import Settings
from retail_demand.io.s3_writer import S3ParquetWriter


def _settings(bucket: str = "retail-demand-test") -> Settings:
    """Inputs: optional bucket; outputs: test settings; side effects: none."""
    return Settings(s3_bucket=bucket, aws_region="us-east-1")


def _create_bucket(bucket: str = "retail-demand-test") -> object:
    """Inputs: bucket name; outputs: boto3 client; side effects: creates moto bucket."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=bucket)
    return client


def _keys(client: object, bucket: str = "retail-demand-test") -> list[str]:
    """Inputs: S3 client/bucket; outputs: sorted keys; side effects: lists moto bucket."""
    response = client.list_objects_v2(Bucket=bucket)
    return sorted(item["Key"] for item in response.get("Contents", []))


@mock_aws
def test_write_partitioned_parquet_uses_hive_style_keys() -> None:
    """Inputs: partitioned frame; outputs: assertions; side effects: writes moto S3."""
    client = _create_bucket()
    writer = S3ParquetWriter(_settings())
    frame = pd.DataFrame(
        {
            "sku_id": ["sku_1", "sku_2"],
            "price": [10.0, 8.0],
            "year": [2026, 2026],
        }
    )

    stats = writer.write_table(frame, "prices", partition_cols=["year"])

    assert stats["rows"] == 2
    assert stats["uri"] == "s3://retail-demand-test/raw/prices/"
    assert any(key.startswith("raw/prices/year=2026/part-00000") for key in _keys(client))


@mock_aws
def test_overwrite_safety_refuses_protected_prefix() -> None:
    """Inputs: bad raw prefix; outputs: assertions; side effects: creates moto bucket."""
    _create_bucket()
    settings = Settings(s3_bucket="retail-demand-test", s3_raw_prefix="bronze")
    writer = S3ParquetWriter(settings)

    with pytest.raises(ValueError, match="raw prefix is unsafe"):
        writer.write_table(pd.DataFrame({"x": [1]}), "sales")


@mock_aws
def test_empty_dataframe_logs_warning_and_does_not_fail(caplog: pytest.LogCaptureFixture) -> None:
    """Inputs: empty frame; outputs: assertions; side effects: writes no S3 data."""
    client = _create_bucket()
    writer = S3ParquetWriter(_settings())

    with caplog.at_level(logging.WARNING):
        stats = writer.write_table(pd.DataFrame({"x": []}), "empty_table")

    assert stats["rows"] == 0
    assert "empty_table_skipped" in caplog.text
    assert _keys(client) == []
