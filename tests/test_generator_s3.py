"""Tests for S3-backed generator flows."""

from __future__ import annotations

import json
from datetime import date

import boto3
import pandas as pd
from moto import mock_aws
from typer.testing import CliRunner

from retail_demand.config import Settings
from retail_demand.data_generation.generator import SyntheticDataConfig, app, write_dataset
from retail_demand.io.s3_writer import S3Reader

runner = CliRunner()


def _create_bucket(bucket: str = "retail-demand-test") -> object:
    """Inputs: bucket name; outputs: boto3 client; side effects: creates moto bucket."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=bucket)
    return client


def _settings(bucket: str = "retail-demand-test") -> Settings:
    """Inputs: optional bucket; outputs: S3 settings; side effects: none."""
    return Settings(s3_bucket=bucket, aws_region="us-east-1", random_seed=42)


def _keys(client: object, bucket: str = "retail-demand-test") -> list[str]:
    """Inputs: S3 client/bucket; outputs: sorted object keys; side effects: lists moto S3."""
    response = client.list_objects_v2(Bucket=bucket)
    return sorted(item["Key"] for item in response.get("Contents", []))


@mock_aws
def test_s3_dry_run_prints_uris_and_writes_nothing(monkeypatch: object) -> None:
    """Inputs: CLI dry run; outputs: assertions; side effects: creates moto bucket."""
    client = _create_bucket()
    monkeypatch.setenv("S3_BUCKET", "retail-demand-test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    result = runner.invoke(
        app,
        [
            "--target",
            "s3",
            "--dry-run",
            "--n-stores",
            "2",
            "--n-skus",
            "20",
            "--days",
            "60",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "s3://retail-demand-test/raw/sales/" in result.output
    assert "s3://retail-demand-test/raw/prices/" in result.output
    assert _keys(client) == []


@mock_aws
def test_s3_generation_round_trip_and_manifest() -> None:
    """Inputs: tiny S3 generation; outputs: assertions; side effects: writes moto S3."""
    client = _create_bucket()
    settings = _settings()
    config = SyntheticDataConfig(num_stores=2, num_skus=20, num_days=60, end_date=date(2026, 1, 31))

    write_dataset(config, target="s3", output_path="unused", settings=settings, mode="overwrite")

    reader = S3Reader(settings)
    assert reader.read_table("stores").count_rows() == 2
    assert reader.read_table("products").count_rows() == 20
    assert reader.read_table("calendar").count_rows() == 60
    assert reader.read_table("sales").count_rows() >= config.expected_sales_rows

    sales = reader.read_table("sales").to_table().to_pandas()
    prices = reader.read_table("prices").to_table().to_pandas()
    calendar = reader.read_table("calendar").to_table().to_pandas()
    assert _promo_lift_ratio(sales, prices, calendar) > 1.05

    manifest_keys = [key for key in _keys(client) if key.startswith("raw/_manifests/")]
    assert len(manifest_keys) == 1
    manifest = json.loads(
        client.get_object(Bucket="retail-demand-test", Key=manifest_keys[0])["Body"].read()
    )
    assert set(manifest["tables"]) == {"stores", "products", "calendar", "prices", "sales"}
    assert manifest["tables"]["stores"]["row_count"] == 2
    assert manifest["tables"]["products"]["row_count"] == 20
    assert manifest["tables"]["calendar"]["row_count"] == 60
    assert manifest["tables"]["sales"]["row_count"] == reader.read_table("sales").count_rows()
    assert manifest["tables"]["sales"]["s3_uri"] == "s3://retail-demand-test/raw/sales/"
    assert manifest["tables"]["sales"]["partition_cols"] == ["year", "month"]


def _promo_lift_ratio(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
) -> float:
    """Inputs: generated tables; outputs: aggregate promo lift ratio; side effects: none."""
    calendar_with_week = calendar[["date"]].copy()
    calendar_dates = pd.to_datetime(calendar_with_week["date"])
    calendar_with_week["week_start"] = calendar_dates - pd.to_timedelta(
        calendar_dates.dt.dayofweek,
        unit="D",
    )
    clean_sales = sales[sales["units_sold"] >= 0]
    joined = clean_sales.merge(calendar_with_week, on="date", how="left").merge(
        prices,
        on=["store_id", "sku_id", "week_start"],
        how="inner",
    )
    sku_promo_means = (
        joined.groupby(["store_id", "sku_id", "is_promo"])["units_sold"]
        .mean()
        .unstack("is_promo")
        .dropna()
    )
    return float((sku_promo_means[True] / sku_promo_means[False]).mean())
