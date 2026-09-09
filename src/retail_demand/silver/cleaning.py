"""Cleaning transforms for Silver Delta tables."""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from retail_demand.silver.schemas import BUSINESS_SCHEMAS, NATURAL_KEYS
from retail_demand.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_KEYS = (
    "rows_in",
    "rows_out",
    "dupes_dropped",
    "nulls_dropped",
    "invalid_dropped",
    "imputations_applied",
)


def _empty_metrics(rows_in: int) -> dict[str, int]:
    return {
        "rows_in": rows_in,
        "rows_out": rows_in,
        "dupes_dropped": 0,
        "nulls_dropped": 0,
        "invalid_dropped": 0,
        "imputations_applied": 0,
    }


def _select_cast(df: DataFrame, table_name: str) -> DataFrame:
    schema = BUSINESS_SCHEMAS[table_name]
    return df.select(
        *[F.col(field.name).cast(field.dataType).alias(field.name) for field in schema.fields]
    )


def _drop_required_nulls(df: DataFrame, required_cols: list[str]) -> tuple[DataFrame, int]:
    before = df.count()
    cleaned = df.dropna(subset=required_cols)
    return cleaned, before - cleaned.count()


def _drop_duplicate_keys(df: DataFrame, table_name: str) -> tuple[DataFrame, int]:
    before = df.count()
    cleaned = df.dropDuplicates(NATURAL_KEYS[table_name])
    return cleaned, before - cleaned.count()


def _coerce_business_schema(df: DataFrame, table_name: str) -> DataFrame:
    schema = BUSINESS_SCHEMAS[table_name]
    return df.select(
        *[F.col(field.name).cast(field.dataType).alias(field.name) for field in schema.fields]
    )


def _finalize(
    table_name: str,
    rows_in: int,
    cleaned: DataFrame,
    dupes_dropped: int = 0,
    nulls_dropped: int = 0,
    invalid_dropped: int = 0,
    imputations_applied: int = 0,
) -> tuple[DataFrame, dict[str, int]]:
    cleaned = _coerce_business_schema(cleaned, table_name)
    metrics = {
        "rows_in": rows_in,
        "rows_out": cleaned.count(),
        "dupes_dropped": dupes_dropped,
        "nulls_dropped": nulls_dropped,
        "invalid_dropped": invalid_dropped,
        "imputations_applied": imputations_applied,
    }
    logger.info("silver_cleaning_complete", table=table_name, **metrics)
    return cleaned, metrics


def _clean_light_table(df: DataFrame, table_name: str) -> tuple[DataFrame, dict[str, int]]:
    rows_in = df.count()
    casted = _select_cast(df, table_name)
    nonnull, nulls_dropped = _drop_required_nulls(
        casted, [field.name for field in BUSINESS_SCHEMAS[table_name].fields]
    )
    deduped, dupes_dropped = _drop_duplicate_keys(nonnull, table_name)
    return _finalize(
        table_name,
        rows_in,
        deduped,
        dupes_dropped=dupes_dropped,
        nulls_dropped=nulls_dropped,
    )


def clean_stores(df: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Pre: Bronze stores columns are present. Post: typed, non-null, unique store rows."""
    return _clean_light_table(df, "stores")


def clean_products(df: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Pre: Bronze product columns are present. Post: typed, non-null, unique SKU rows."""
    return _clean_light_table(df, "products")


def clean_calendar(df: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Pre: Bronze calendar columns are present. Post: typed, non-null, unique date rows."""
    return _clean_light_table(df, "calendar")


def clean_prices(df: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Pre: Bronze prices columns are present. Post: price nulls are LOCF-imputed or dropped."""
    rows_in = df.count()
    casted = _select_cast(df, "prices").withColumn("store_id", F.lower(F.col("store_id")))
    required_except_price = [
        field.name for field in BUSINESS_SCHEMAS["prices"].fields if field.name != "price"
    ]
    nonnull_keys, nulls_without_price = _drop_required_nulls(casted, required_except_price)

    window = (
        Window.partitionBy("store_id", "sku_id")
        .orderBy("week_start")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    with_imputed = nonnull_keys.withColumn(
        "_prior_price", F.last("price", ignorenulls=True).over(window)
    ).withColumn(
        "price",
        F.when(F.col("price").isNull(), F.col("_prior_price")).otherwise(F.col("price")),
    )
    imputations_applied = with_imputed.filter(
        F.col("_prior_price").isNotNull() & F.col("price").isNotNull()
    ).join(
        nonnull_keys.filter(F.col("price").isNull()).select("store_id", "sku_id", "week_start"),
        ["store_id", "sku_id", "week_start"],
        "inner",
    ).count()
    before_price_drop = with_imputed.count()
    priced = with_imputed.drop("_prior_price").dropna(subset=["price"])
    impossible_imputations = before_price_drop - priced.count()
    deduped, dupes_dropped = _drop_duplicate_keys(priced, "prices")

    return _finalize(
        "prices",
        rows_in,
        deduped,
        dupes_dropped=dupes_dropped,
        nulls_dropped=nulls_without_price + impossible_imputations,
        imputations_applied=imputations_applied,
    )


def clean_sales(df: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    """Pre: Bronze sales columns and _ingested_at are present. Post: valid unique sales rows."""
    rows_in = df.count()
    schema = BUSINESS_SCHEMAS["sales"]
    casted = df.select(
        *[
            (
                F.lower(F.col(field.name)).cast(field.dataType).alias(field.name)
                if field.name == "store_id"
                else F.col(field.name).cast(field.dataType).alias(field.name)
            )
            for field in schema.fields
        ],
        F.col("_ingested_at").cast(T.TimestampType()).alias("_ingested_at"),
    )
    nonnull, nulls_dropped = _drop_required_nulls(casted, [field.name for field in schema.fields])
    valid = nonnull.filter(F.col("units_sold") >= 0)
    invalid_dropped = nonnull.count() - valid.count()

    dedupe_window = Window.partitionBy(*NATURAL_KEYS["sales"]).orderBy(
        F.col("_ingested_at").asc_nulls_last()
    )
    with_rank = valid.withColumn("_row_number", F.row_number().over(dedupe_window))
    deduped_with_order = with_rank.filter(F.col("_row_number") == 1).drop(
        "_row_number", "_ingested_at"
    )
    dupes_dropped = valid.count() - deduped_with_order.count()

    return _finalize(
        "sales",
        rows_in,
        deduped_with_order,
        dupes_dropped=dupes_dropped,
        nulls_dropped=nulls_dropped,
        invalid_dropped=invalid_dropped,
    )


CLEANERS: dict[str, Any] = {
    "stores": clean_stores,
    "products": clean_products,
    "calendar": clean_calendar,
    "prices": clean_prices,
    "sales": clean_sales,
}
