"""Strict PySpark schemas and natural keys for the Silver layer."""

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
    TimestampType,
)

NATURAL_KEYS: dict[str, list[str]] = {
    "stores": ["store_id"],
    "products": ["sku_id"],
    "calendar": ["date"],
    "prices": ["store_id", "sku_id", "week_start"],
    "sales": ["store_id", "sku_id", "date"],
}

STORES_BUSINESS_SCHEMA = StructType(
    [
        StructField("store_id", StringType(), False),
        StructField("region", StringType(), False),
        StructField("state", StringType(), False),
        StructField("store_type", StringType(), False),
        StructField("open_date", DateType(), False),
        StructField("sqft", LongType(), False),
    ]
)

PRODUCTS_BUSINESS_SCHEMA = StructType(
    [
        StructField("sku_id", StringType(), False),
        StructField("category", StringType(), False),
        StructField("subcategory", StringType(), False),
        StructField("base_price", DoubleType(), False),
        StructField("is_perishable", BooleanType(), False),
        StructField("pack_size", LongType(), False),
    ]
)

CALENDAR_BUSINESS_SCHEMA = StructType(
    [
        StructField("date", LongType(), False),
        StructField("dow", IntegerType(), False),
        StructField("week", LongType(), False),
        StructField("month", IntegerType(), False),
        StructField("quarter", IntegerType(), False),
        StructField("year", IntegerType(), False),
        StructField("is_weekend", BooleanType(), False),
        StructField("is_us_holiday", BooleanType(), False),
    ]
)

PRICES_BUSINESS_SCHEMA = StructType(
    [
        StructField("store_id", StringType(), False),
        StructField("sku_id", StringType(), False),
        StructField("week_start", LongType(), False),
        StructField("price", DoubleType(), False),
        StructField("is_promo", BooleanType(), False),
        StructField("year", IntegerType(), False),
    ]
)

SALES_BUSINESS_SCHEMA = StructType(
    [
        StructField("store_id", StringType(), False),
        StructField("sku_id", StringType(), False),
        StructField("date", LongType(), False),
        StructField("units_sold", LongType(), False),
        StructField("revenue", DoubleType(), False),
        StructField("year", IntegerType(), False),
        StructField("month", IntegerType(), False),
    ]
)

SILVER_METADATA_FIELDS = [
    StructField("_silvered_at", TimestampType(), False),
    StructField("_silver_run_id", StringType(), False),
    StructField("_bronze_run_id", StringType(), False),
]

BUSINESS_SCHEMAS: dict[str, StructType] = {
    "stores": STORES_BUSINESS_SCHEMA,
    "products": PRODUCTS_BUSINESS_SCHEMA,
    "calendar": CALENDAR_BUSINESS_SCHEMA,
    "prices": PRICES_BUSINESS_SCHEMA,
    "sales": SALES_BUSINESS_SCHEMA,
}

SILVER_SCHEMAS: dict[str, StructType] = {
    name: StructType([*schema.fields, *SILVER_METADATA_FIELDS])
    for name, schema in BUSINESS_SCHEMAS.items()
}

PARTITION_COLS: dict[str, list[str]] = {
    "stores": [],
    "products": [],
    "calendar": [],
    "prices": ["year"],
    "sales": ["year", "month"],
}
