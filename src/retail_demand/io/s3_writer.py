"""S3 parquet writer and reader utilities for the raw landing zone."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import s3fs
import typer

from retail_demand.config import Settings, get_settings
from retail_demand.utils.logging import get_logger

WriteMode = Literal["overwrite", "append"]

app = typer.Typer(help="Inspect raw parquet tables in S3.")


@dataclass
class S3ParquetWriter:
    """Writer for raw-zone parquet tables in S3."""

    settings: Settings
    bucket: str | None = None
    profile: str | None = None
    _write_counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Inputs: settings/bucket/profile; outputs: none; side effects: creates S3 filesystems."""
        self.bucket = self.bucket or self.settings.s3_bucket
        self.profile = self.profile if self.profile is not None else self.settings.aws_profile
        if not self.bucket:
            raise ValueError("S3 bucket is required for S3ParquetWriter")
        client_kwargs = {"region_name": self.settings.aws_region}
        if self.profile:
            self.s3 = s3fs.S3FileSystem(profile=self.profile, client_kwargs=client_kwargs)
        else:
            self.s3 = s3fs.S3FileSystem(client_kwargs=client_kwargs)
        self.pyarrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(self.s3))
        self.boto3_s3 = boto3.client("s3", region_name=self.settings.aws_region)
        self._use_moto_fallback = _moto_is_active()
        self.logger = get_logger(__name__)

    def write_table(
        self,
        df: pd.DataFrame,
        table: str,
        partition_cols: list[str] | None = None,
        mode: WriteMode = "overwrite",
    ) -> dict[str, int | str | list[str]]:
        """Inputs: DataFrame/table/partitions/mode; outputs: stats; side effects: writes S3."""
        uri = self._raw_uri(table)
        self._validate_raw_prefix(uri)
        partition_cols = partition_cols or []
        if df.empty:
            self.logger.warning("empty_table_skipped", uri=uri, rows=0, table=table)
            return {"rows": 0, "bytes": 0, "uri": uri, "partition_cols": partition_cols}

        if self._use_moto_fallback:
            return self._write_table_with_boto3_fallback(df, table, partition_cols, mode, uri)

        if mode == "overwrite":
            self._delete_prefix(uri)
            self._write_counts[table] = 0
        elif table not in self._write_counts:
            self._write_counts[table] = self._next_part_number(uri)

        before_bytes = self._prefix_size(uri)
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
        ds.write_dataset(
            arrow_table,
            base_dir=uri,
            basename_template=f"part-{self._write_counts[table]:05d}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            format="parquet",
            filesystem=self.pyarrow_fs,
            partitioning=partition_cols if partition_cols else None,
            partitioning_flavor="hive" if partition_cols else None,
        )
        self._write_counts[table] += 1
        after_bytes = self._prefix_size(uri)
        stats = {
            "rows": int(len(df)),
            "bytes": int(max(0, after_bytes - before_bytes)),
            "uri": uri,
            "partition_cols": partition_cols,
        }
        self.logger.info("s3_write_complete", uri=uri, rows=len(df), partition_cols=partition_cols)
        return stats

    def _write_table_with_boto3_fallback(
        self,
        df: pd.DataFrame,
        table: str,
        partition_cols: list[str],
        mode: WriteMode,
        uri: str,
    ) -> dict[str, int | str | list[str]]:
        """Inputs: table write request; outputs: stats; side effects: writes to moto S3."""
        if mode == "overwrite":
            self._delete_prefix(uri)
            self._write_counts[table] = 0
        elif table not in self._write_counts:
            self._write_counts[table] = self._next_part_number(uri)

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / table
            arrow_table = pa.Table.from_pandas(df, preserve_index=False)
            ds.write_dataset(
                arrow_table,
                base_dir=str(base_dir),
                basename_template=f"part-{self._write_counts[table]:05d}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                format="parquet",
                partitioning=partition_cols if partition_cols else None,
                partitioning_flavor="hive" if partition_cols else None,
            )
            written_bytes = self._upload_local_dataset(base_dir, uri)

        self._write_counts[table] += 1
        stats = {
            "rows": int(len(df)),
            "bytes": int(written_bytes),
            "uri": uri,
            "partition_cols": partition_cols,
        }
        self.logger.info("s3_write_complete", uri=uri, rows=len(df), partition_cols=partition_cols)
        return stats

    def _upload_local_dataset(self, base_dir: Path, uri: str) -> int:
        """Inputs: local dataset and target URI; outputs: bytes; side effects: uploads S3."""
        total_bytes = 0
        key_prefix = _key_from_uri(uri).rstrip("/")
        for file_path in base_dir.rglob("*.parquet"):
            relative_key = file_path.relative_to(base_dir).as_posix()
            object_key = f"{key_prefix}/{relative_key}"
            total_bytes += file_path.stat().st_size
            self.boto3_s3.upload_file(str(file_path), str(self.bucket), object_key)
        return total_bytes

    def _next_part_number(self, uri: str) -> int:
        """Inputs: S3 table URI; outputs: next deterministic part number; side effects: lists S3."""
        key = _key_from_uri(uri)
        if self._use_moto_fallback:
            return _next_part_number_from_keys(
                _list_boto3_keys(self.boto3_s3, str(self.bucket), key)
            )
        path = f"{self.bucket}/{key}"
        if not self.s3.exists(path):
            return 0
        object_keys = list(self.s3.find(path))
        return _next_part_number_from_keys(object_keys)

    def _delete_prefix(self, uri: str) -> None:
        """Inputs: safe raw S3 URI; outputs: none; side effects: deletes objects under prefix."""
        self._validate_raw_prefix(uri)
        key = _key_from_uri(uri)
        if self._use_moto_fallback:
            _delete_boto3_prefix(self.boto3_s3, str(self.bucket), key)
            return
        if self.s3.exists(f"{self.bucket}/{key}"):
            self.s3.rm(f"{self.bucket}/{key}", recursive=True)

    def _prefix_size(self, uri: str) -> int:
        """Inputs: S3 URI; outputs: total bytes below URI; side effects: lists objects."""
        key = _key_from_uri(uri)
        if self._use_moto_fallback:
            return _boto3_prefix_size(self.boto3_s3, str(self.bucket), key)
        path = f"{self.bucket}/{key}"
        if not self.s3.exists(path):
            return 0
        return int(self.s3.du(path, total=True))

    def _validate_raw_prefix(self, uri: str) -> None:
        """Inputs: S3 URI; outputs: none; side effects: raises on unsafe delete prefixes."""
        key = _key_from_uri(uri)
        raw_prefix = self.settings.s3_raw_prefix.strip("/")
        protected_prefixes = {
            self.settings.s3_bronze_prefix.strip("/"),
            self.settings.s3_silver_prefix.strip("/"),
            self.settings.s3_gold_prefix.strip("/"),
        }
        if not raw_prefix or raw_prefix in protected_prefixes:
            raise ValueError("Refusing to write/delete because raw prefix is unsafe")
        if not key.startswith(f"{raw_prefix}/"):
            raise ValueError(f"Refusing to write/delete outside raw prefix: {uri}")

    def _raw_uri(self, table: str) -> str:
        """Inputs: table name; outputs: raw S3 URI; side effects: none."""
        raw_prefix = self.settings.s3_raw_prefix.strip("/")
        clean_table = table.strip("/")
        return f"s3://{self.bucket}/{raw_prefix}/{clean_table}/"


@dataclass
class S3Reader:
    """Reader for raw-zone parquet datasets in S3."""

    settings: Settings

    def __post_init__(self) -> None:
        """Inputs: settings; outputs: none; side effects: creates S3 filesystem."""
        if not self.settings.s3_bucket:
            raise ValueError("S3_BUCKET is required for S3Reader")
        client_kwargs = {"region_name": self.settings.aws_region}
        if self.settings.aws_profile:
            self.s3 = s3fs.S3FileSystem(
                profile=self.settings.aws_profile,
                client_kwargs=client_kwargs,
            )
        else:
            self.s3 = s3fs.S3FileSystem(client_kwargs=client_kwargs)
        self.pyarrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(self.s3))
        self.boto3_s3 = boto3.client("s3", region_name=self.settings.aws_region)
        self._use_moto_fallback = _moto_is_active()

    def read_table(self, table: str) -> ds.Dataset:
        """Inputs: raw table name; outputs: Arrow dataset; side effects: reads S3 metadata."""
        uri = self.settings.s3_uri("raw", table)
        if self._use_moto_fallback:
            return self._read_table_with_boto3_fallback(table, uri)
        return ds.dataset(uri, filesystem=self.pyarrow_fs, format="parquet", partitioning="hive")

    def _read_table_with_boto3_fallback(self, table: str, uri: str) -> ds.Dataset:
        """Inputs: table/URI; outputs: Arrow dataset; side effects: downloads moto objects."""
        prefix = _key_from_uri(uri)
        object_keys = _list_boto3_keys(self.boto3_s3, self.settings.s3_bucket, prefix)
        base_dir = Path(tempfile.mkdtemp(prefix="retail-demand-s3reader-")) / table
        for object_key in object_keys:
            relative_path = Path(object_key).relative_to(prefix)
            destination = base_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.boto3_s3.download_file(self.settings.s3_bucket, object_key, str(destination))
        return ds.dataset(str(base_dir), format="parquet", partitioning="hive")


def _key_from_uri(uri: str) -> str:
    """Inputs: S3 URI; outputs: object key prefix; side effects: validates URI shape."""
    prefix = "s3://"
    if not uri.startswith(prefix):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    bucket_and_key = uri[len(prefix) :]
    parts = bucket_and_key.split("/", 1)
    if len(parts) != 2:
        return ""
    return parts[1].strip("/")


def _list_boto3_keys(client: object, bucket: str, prefix: str) -> list[str]:
    """Inputs: boto3 client/bucket/prefix; outputs: matching keys; side effects: lists S3."""
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return keys


def _delete_boto3_prefix(client: object, bucket: str, prefix: str) -> None:
    """Inputs: boto3 client/bucket/prefix; outputs: none; side effects: deletes S3 objects."""
    keys = _list_boto3_keys(client, bucket, prefix)
    if not keys:
        return
    client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": key} for key in keys]})


def _boto3_prefix_size(client: object, bucket: str, prefix: str) -> int:
    """Inputs: boto3 client/bucket/prefix; outputs: byte size; side effects: lists S3."""
    total = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        total += sum(int(item["Size"]) for item in page.get("Contents", []))
    return total


def _part_number_from_key(object_key: str) -> int:
    """Inputs: object key; outputs: parsed part number or -1; side effects: none."""
    filename = object_key.rsplit("/", 1)[-1]
    if filename.startswith("part-") and len(filename) >= 15:
        candidate = filename[5:10]
        if candidate.isdigit():
            return int(candidate)
    return -1


def _next_part_number_from_keys(object_keys: list[str]) -> int:
    """Inputs: object keys; outputs: next part number; side effects: none."""
    return max((_part_number_from_key(object_key) for object_key in object_keys), default=-1) + 1


def _moto_is_active() -> bool:
    """Inputs: none; outputs: whether moto mock_aws is active; side effects: imports if present."""
    try:
        from moto.core.models import MockAWS
    except ImportError:
        return False
    return bool(getattr(MockAWS, "_nested_count", 0))


@app.command()
def verify() -> None:
    """Inputs: environment settings; outputs: row counts; side effects: reads S3 metadata."""
    settings = get_settings()
    reader = S3Reader(settings)
    for table in ["stores", "products", "calendar", "prices", "sales"]:
        row_count = reader.read_table(table).count_rows()
        typer.echo(f"{table}: {row_count:,} rows")


if __name__ == "__main__":
    app()
