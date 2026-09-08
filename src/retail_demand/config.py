"""Configuration for synthetic retail demand data generation."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    aws_profile: str | None = Field(default=None, validation_alias="AWS_PROFILE")
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    s3_bucket: str = Field(default="", validation_alias="S3_BUCKET")
    s3_raw_prefix: str = Field(default="raw", validation_alias="S3_RAW_PREFIX")
    s3_bronze_prefix: str = Field(default="bronze", validation_alias="S3_BRONZE_PREFIX")
    s3_silver_prefix: str = Field(default="silver", validation_alias="S3_SILVER_PREFIX")
    s3_gold_prefix: str = Field(default="gold", validation_alias="S3_GOLD_PREFIX")
    databricks_volume_root: str = Field(
        default="/Volumes/workspace/retail_demand/raw",
        validation_alias="DATABRICKS_VOLUME_ROOT",
    )
    random_seed: int = Field(default=42, validation_alias="RANDOM_SEED")
    data_start_date: date | None = Field(default=None, validation_alias="DATA_START_DATE")
    data_end_date: date = Field(default_factory=date.today, validation_alias="DATA_END_DATE")
    default_num_stores: int = Field(default=50, validation_alias="DEFAULT_NUM_STORES")
    default_num_skus: int = Field(default=2_000, validation_alias="DEFAULT_NUM_SKUS")
    default_num_days: int = Field(default=365 * 3, validation_alias="DEFAULT_NUM_DAYS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def effective_start_date(self) -> date:
        """Inputs: settings fields; outputs: generation start date; side effects: none."""
        if self.data_start_date is not None:
            return self.data_start_date
        return self.data_end_date - timedelta(days=self.default_num_days - 1)

    def s3_uri(self, layer: Literal["raw", "bronze", "silver", "gold"], table: str) -> str:
        """Inputs: medallion layer/table; outputs: S3 URI; side effects: validates bucket."""
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET is required when building S3 URIs")
        prefixes = {
            "raw": self.s3_raw_prefix,
            "bronze": self.s3_bronze_prefix,
            "silver": self.s3_silver_prefix,
            "gold": self.s3_gold_prefix,
        }
        prefix = prefixes[layer].strip("/")
        clean_table = table.strip("/")
        return f"s3://{self.s3_bucket}/{prefix}/{clean_table}/"


def get_settings() -> Settings:
    """Inputs: environment/.env; outputs: Settings instance; side effects: reads environment."""
    return Settings()
