"""Tests for the Databricks-side S3 → Volume sync utility."""

from __future__ import annotations

from pathlib import Path

import boto3
from moto import mock_aws

from retail_demand.io.databricks_sync import sync_s3_to_volume

BUCKET = "retail-demand-sync-test"
REGION = "us-east-1"
# Fake key/secret — accepted by moto without authentication
_FAKE_KEY = "testing"
_FAKE_SECRET = "testing"


def _create_bucket() -> object:
    """Inputs: none; outputs: boto3 client with a moto bucket; side effects: creates bucket."""
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    return client


def _upload_objects(client: object, keys_and_contents: dict[str, bytes]) -> None:
    """Inputs: dict of {key: content}; outputs: none; side effects: uploads to moto S3."""
    for key, content in keys_and_contents.items():
        client.put_object(Bucket=BUCKET, Key=key, Body=content)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

S3_OBJECTS: dict[str, bytes] = {
    "raw/sales/year=2026/month=1/part-00000-0.parquet": b"fake-parquet-1",
    "raw/sales/year=2026/month=1/part-00001-0.parquet": b"fake-parquet-2",
    "raw/sales/year=2026/month=2/part-00000-0.parquet": b"fake-parquet-3",
    "raw/stores/part-00000-0.parquet": b"fake-stores",
    "raw/products/part-00000-0.parquet": b"fake-products",
}


@mock_aws
def test_sync_copies_all_objects_preserving_structure(tmp_path: Path) -> None:
    """Inputs: moto S3 with objects; outputs: assertions; side effects: writes tmpdir."""
    client = _create_bucket()
    _upload_objects(client, S3_OBJECTS)

    result = sync_s3_to_volume(
        s3_uri=f"s3://{BUCKET}/raw/sales/",
        volume_path=str(tmp_path / "sales"),
        aws_access_key=_FAKE_KEY,
        aws_secret_key=_FAKE_SECRET,
        aws_region=REGION,
    )

    # Three parquet files are under raw/sales/
    assert result["files_synced"] == 3
    assert result["files_skipped"] == 0
    assert result["bytes_transferred"] > 0

    # Directory structure mirrors S3 partition layout
    assert (tmp_path / "sales" / "year=2026" / "month=1" / "part-00000-0.parquet").exists()
    assert (tmp_path / "sales" / "year=2026" / "month=1" / "part-00001-0.parquet").exists()
    assert (tmp_path / "sales" / "year=2026" / "month=2" / "part-00000-0.parquet").exists()


@mock_aws
def test_sync_is_idempotent_on_rerun(tmp_path: Path) -> None:
    """Inputs: moto S3 + already-synced files; outputs: assertions; side effects: writes tmpdir."""
    client = _create_bucket()
    _upload_objects(client, S3_OBJECTS)
    dest = str(tmp_path / "sales")

    # First run: all files should be synced
    first = sync_s3_to_volume(
        s3_uri=f"s3://{BUCKET}/raw/sales/",
        volume_path=dest,
        aws_access_key=_FAKE_KEY,
        aws_secret_key=_FAKE_SECRET,
        aws_region=REGION,
    )
    assert first["files_synced"] == 3
    assert first["files_skipped"] == 0

    # Second run: same size on disk → all files should be skipped
    second = sync_s3_to_volume(
        s3_uri=f"s3://{BUCKET}/raw/sales/",
        volume_path=dest,
        aws_access_key=_FAKE_KEY,
        aws_secret_key=_FAKE_SECRET,
        aws_region=REGION,
    )
    assert second["files_skipped"] == first["files_synced"]
    assert second["files_synced"] == 0
    assert second["bytes_transferred"] == 0


@mock_aws
def test_sync_directory_tree_matches_s3_layout(tmp_path: Path) -> None:
    """Inputs: moto S3; outputs: assertions; side effects: writes tmpdir."""
    client = _create_bucket()
    _upload_objects(client, S3_OBJECTS)
    dest = tmp_path / "stores"

    sync_s3_to_volume(
        s3_uri=f"s3://{BUCKET}/raw/stores/",
        volume_path=str(dest),
        aws_access_key=_FAKE_KEY,
        aws_secret_key=_FAKE_SECRET,
        aws_region=REGION,
    )

    # The top-level volume dir should contain only the expected file
    written = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert written == ["part-00000-0.parquet"]


@mock_aws
def test_sync_returns_correct_summary_stats(tmp_path: Path) -> None:
    """Inputs: moto S3; outputs: stat assertions; side effects: writes tmpdir."""
    client = _create_bucket()
    _upload_objects(client, S3_OBJECTS)

    result = sync_s3_to_volume(
        s3_uri=f"s3://{BUCKET}/raw/sales/",
        volume_path=str(tmp_path / "sales"),
        aws_access_key=_FAKE_KEY,
        aws_secret_key=_FAKE_SECRET,
        aws_region=REGION,
    )

    assert "files_synced" in result
    assert "files_skipped" in result
    assert "bytes_transferred" in result
    assert "duration_seconds" in result
    assert isinstance(result["duration_seconds"], float)
