"""Databricks-side sync utility: copies raw S3 objects into a Databricks Volume.

This module intentionally avoids Spark and Delta so it can run on Databricks
Free Edition serverless compute, which does not allow setting fs.s3a credentials
at runtime.  All S3 access goes through boto3, which accepts credentials from
environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger(__name__)


def sync_s3_to_volume(
    s3_uri: str,
    volume_path: str,
    aws_access_key: str,
    aws_secret_key: str,
    aws_region: str = "us-east-1",
) -> dict[str, Any]:
    """Download every object under an S3 prefix into a local Volume path.

    Inputs
    ------
    s3_uri       : S3 URI of the prefix to sync, e.g. ``s3://bucket/raw/sales/``.
    volume_path  : Absolute destination path (Databricks Volume or local tempdir).
    aws_access_key : AWS access key ID — never commit this value.
    aws_secret_key : AWS secret access key — never commit this value.
    aws_region   : AWS region of the bucket (default ``us-east-1``).

    Outputs
    -------
    dict with keys: files_synced, files_skipped, bytes_transferred, duration_seconds.

    Side effects
    ------------
    Creates directories under *volume_path* as needed; downloads objects from S3.
    """
    bucket, prefix = _parse_s3_uri(s3_uri)
    client = boto3.client(
        "s3",
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )

    objects = _list_all_objects(client, bucket, prefix)
    dest_root = Path(volume_path)

    files_synced = 0
    files_skipped = 0
    bytes_transferred = 0
    start = time.monotonic()

    for idx, obj in enumerate(objects, start=1):
        key: str = obj["Key"]
        size: int = obj["Size"]

        # Build the destination path, preserving partition directory structure
        # e.g. raw/sales/year=2026/month=1/part-00000-0.parquet
        # -> <volume_path>/sales/year=2026/month=1/part-00000-0.parquet
        # Strip the common prefix so we only keep the table-relative path.
        relative_key = key[len(prefix) :].lstrip("/")
        dest_file = dest_root / relative_key

        if dest_file.exists() and dest_file.stat().st_size == size:
            files_skipped += 1
        else:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest_file))
            files_synced += 1
            bytes_transferred += size

        if idx % 100 == 0:
            elapsed = time.monotonic() - start
            logger.info(
                "sync_progress",
                extra={
                    "files_processed": idx,
                    "files_synced": files_synced,
                    "files_skipped": files_skipped,
                    "bytes_transferred": bytes_transferred,
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

    duration = round(time.monotonic() - start, 3)
    logger.info(
        "sync_complete",
        extra={
            "files_synced": files_synced,
            "files_skipped": files_skipped,
            "bytes_transferred": bytes_transferred,
            "duration_seconds": duration,
        },
    )
    return {
        "files_synced": files_synced,
        "files_skipped": files_skipped,
        "bytes_transferred": bytes_transferred,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Inputs: S3 URI string; outputs: (bucket, prefix); side effects: raises on bad shape."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    without_scheme = uri[len("s3://") :]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket, prefix


def _list_all_objects(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Inputs: boto3 client/bucket/prefix; outputs: all objects; side effects: paginates S3."""
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects
