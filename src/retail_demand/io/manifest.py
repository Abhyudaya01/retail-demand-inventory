"""Manifest writer for successful raw S3 generation runs."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import boto3
import s3fs

from retail_demand.config import Settings
from retail_demand.io.s3_writer import _key_from_uri, _moto_is_active

if TYPE_CHECKING:
    from retail_demand.data_generation.generator import SyntheticDataConfig


def write_manifest(
    settings: Settings,
    generation_config: SyntheticDataConfig,
    table_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Inputs: settings/config/stats; outputs: manifest; side effects: writes JSON to S3."""
    run_id = uuid.uuid4().hex
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": generation_config.seed,
        "config": {
            "settings": _redacted_snapshot(settings),
            "generation": _redacted_snapshot(generation_config),
        },
        "tables": _manifest_table_stats(table_stats),
    }
    manifest_uri = settings.s3_uri("raw", f"_manifests/{run_id}.json").removesuffix("/")
    manifest_body = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    if _moto_is_active():
        boto3.client("s3", region_name=settings.aws_region).put_object(
            Bucket=settings.s3_bucket,
            Key=_key_from_uri(manifest_uri),
            Body=manifest_body.encode("utf-8"),
        )
    else:
        fs = _s3_filesystem(settings)
        with fs.open(manifest_uri, "w", encoding="utf-8") as handle:
            handle.write(manifest_body)
    manifest["manifest_uri"] = manifest_uri
    return manifest


def _s3_filesystem(settings: Settings) -> s3fs.S3FileSystem:
    """Inputs: settings; outputs: S3 filesystem; side effects: none until filesystem use."""
    client_kwargs = {"region_name": settings.aws_region}
    if settings.aws_profile:
        return s3fs.S3FileSystem(profile=settings.aws_profile, client_kwargs=client_kwargs)
    return s3fs.S3FileSystem(client_kwargs=client_kwargs)


def _redacted_snapshot(value: Any) -> dict[str, Any]:
    """Inputs: model/dataclass; outputs: redacted dict; side effects: none."""
    if hasattr(value, "model_dump"):
        snapshot = value.model_dump(mode="json")
    elif is_dataclass(value):
        snapshot = asdict(value)
    else:
        snapshot = dict(value)
    redacted: dict[str, Any] = {}
    secret_terms = ("secret", "password", "token", "key")
    for key, item in snapshot.items():
        if any(term in key.lower() for term in secret_terms):
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = item
    return redacted


def _manifest_table_stats(table_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Inputs: writer stats; outputs: manifest stats; side effects: none."""
    return {
        table: {
            "row_count": int(stats["rows"]),
            "byte_size": int(stats["bytes"]),
            "s3_uri": stats["uri"],
            "partition_cols": stats["partition_cols"],
        }
        for table, stats in table_stats.items()
    }
