"""Tests for project configuration helpers."""

from __future__ import annotations

from retail_demand.config import Settings


def test_s3_uri_for_raw_sales() -> None:
    """Inputs: settings/table; outputs: assertions; side effects: none."""
    settings = Settings(s3_bucket="retail-demand-test")

    assert settings.s3_uri("raw", "sales") == "s3://retail-demand-test/raw/sales/"


def test_s3_uri_all_layers() -> None:
    """Inputs: settings/layers; outputs: assertions; side effects: none."""
    settings = Settings(
        s3_bucket="retail-demand-test",
        s3_raw_prefix="raw",
        s3_bronze_prefix="bronze",
        s3_silver_prefix="silver",
        s3_gold_prefix="gold",
    )

    assert settings.s3_uri("raw", "stores") == "s3://retail-demand-test/raw/stores/"
    assert settings.s3_uri("bronze", "stores") == "s3://retail-demand-test/bronze/stores/"
    assert settings.s3_uri("silver", "stores") == "s3://retail-demand-test/silver/stores/"
    assert settings.s3_uri("gold", "stores") == "s3://retail-demand-test/gold/stores/"
