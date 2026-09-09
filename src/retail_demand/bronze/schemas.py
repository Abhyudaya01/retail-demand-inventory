"""PySpark schemas for the Bronze ingestion layer."""

from __future__ import annotations

from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Dimension Tables
# Phase 1 writes timestamp[ns] for calendar.date, prices.week_start and sales.date.
# Spark 3.5 cannot represent nanosecond timestamps. The approved Bronze mapping
# uses LONG epoch nanoseconds without rounding or changing the stored values.
# Partition year/month fields are reconstructed from the raw Hive directories.
# ---------------------------------------------------------------------------

STORES_SCHEMA = StructType(
    [
        StructField("store_id", StringType(), True),
        StructField("region", StringType(), True),
        StructField("state", StringType(), True),
        StructField("store_type", StringType(), True),
        StructField("open_date", DateType(), True),
        StructField("sqft", LongType(), True),
    ]
)

PRODUCTS_SCHEMA = StructType(
    [
        StructField("sku_id", StringType(), True),
        StructField("category", StringType(), True),
        StructField("subcategory", StringType(), True),
        StructField("base_price", DoubleType(), True),
        StructField("is_perishable", BooleanType(), True),
        StructField("pack_size", LongType(), True),
    ]
)

CALENDAR_SCHEMA = StructType(
    [
        StructField("date", LongType(), True),
        StructField("dow", IntegerType(), True),
        StructField("week", LongType(), True),
        StructField("month", IntegerType(), True),
        StructField("quarter", IntegerType(), True),
        StructField("year", IntegerType(), True),
        StructField("is_weekend", BooleanType(), True),
        StructField("is_us_holiday", BooleanType(), True),
    ]
)

# ---------------------------------------------------------------------------
# Fact Tables
# ---------------------------------------------------------------------------

PRICES_SCHEMA = StructType(
    [
        StructField("store_id", StringType(), True),
        StructField("sku_id", StringType(), True),
        StructField("week_start", LongType(), True),
        StructField("price", DoubleType(), True),
        StructField("is_promo", BooleanType(), True),
        # Partition columns included in the schema
        StructField("year", IntegerType(), True),
    ]
)

SALES_SCHEMA = StructType(
    [
        StructField("store_id", StringType(), True),
        StructField("sku_id", StringType(), True),
        StructField("date", LongType(), True),
        StructField("units_sold", LongType(), True),
        StructField("revenue", DoubleType(), True),
        # Partition columns included in the schema
        StructField("year", IntegerType(), True),
        StructField("month", IntegerType(), True),
    ]
)

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

TABLE_SCHEMAS: dict[str, StructType] = {
    "stores": STORES_SCHEMA,
    "products": PRODUCTS_SCHEMA,
    "calendar": CALENDAR_SCHEMA,
    "prices": PRICES_SCHEMA,
    "sales": SALES_SCHEMA,
}

PARTITION_COLS: dict[str, list[str]] = {
    "stores": [],
    "products": [],
    "calendar": [],
    "prices": ["year"],
    "sales": ["year", "month"],
}
