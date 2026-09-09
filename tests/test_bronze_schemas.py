"""Tests for the explicit PySpark schemas used in the Bronze layer."""

from __future__ import annotations

from pyspark.sql.types import StructType

from retail_demand.bronze.schemas import (
    CALENDAR_SCHEMA,
    PARTITION_COLS,
    PRICES_SCHEMA,
    PRODUCTS_SCHEMA,
    SALES_SCHEMA,
    STORES_SCHEMA,
    TABLE_SCHEMAS,
)


def test_schemas_are_valid_struct_types() -> None:
    """Inputs: schemas; outputs: assertions; side effects: none."""
    assert isinstance(STORES_SCHEMA, StructType)
    assert isinstance(PRODUCTS_SCHEMA, StructType)
    assert isinstance(CALENDAR_SCHEMA, StructType)
    assert isinstance(PRICES_SCHEMA, StructType)
    assert isinstance(SALES_SCHEMA, StructType)

    for _table, schema in TABLE_SCHEMAS.items():
        assert isinstance(schema, StructType)
        assert len(schema.fields) > 0


def test_schema_field_names_match_expected() -> None:
    """Inputs: schemas; outputs: assertions; side effects: none."""
    assert set(STORES_SCHEMA.fieldNames()) == {
        "store_id", "region", "state", "store_type", "open_date", "sqft"
    }
    assert set(PRODUCTS_SCHEMA.fieldNames()) == {
        "sku_id", "category", "subcategory", "base_price", "is_perishable", "pack_size"
    }
    assert set(CALENDAR_SCHEMA.fieldNames()) == {
        "date", "dow", "week", "month", "quarter", "year", "is_weekend", "is_us_holiday"
    }
    assert set(PRICES_SCHEMA.fieldNames()) == {
        "store_id", "sku_id", "week_start", "price", "is_promo", "year"
    }
    assert set(SALES_SCHEMA.fieldNames()) == {
        "store_id", "sku_id", "date", "units_sold", "revenue", "year", "month"
    }


def test_partition_cols_configured() -> None:
    """Inputs: partition cols; outputs: assertions; side effects: none."""
    assert PARTITION_COLS["sales"] == ["year", "month"]
    assert PARTITION_COLS["prices"] == ["year"]
    assert PARTITION_COLS["stores"] == []
