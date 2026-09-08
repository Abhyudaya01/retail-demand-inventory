"""Schemas for generated retail entities."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class StoreRecord(BaseModel):
    """Schema for one generated retail store."""

    store_id: str
    region: str
    state: str
    store_type: str = Field(pattern="^[ABC]$")
    open_date: date
    sqft: int

    model_config = ConfigDict(extra="forbid")


class ProductRecord(BaseModel):
    """Schema for one generated product SKU."""

    sku_id: str
    category: str
    subcategory: str
    base_price: float
    is_perishable: bool
    pack_size: int

    model_config = ConfigDict(extra="forbid")


class CalendarRecord(BaseModel):
    """Schema for one generated calendar date."""

    date: date
    dow: int
    week: int
    month: int
    quarter: int
    year: int
    is_weekend: bool
    is_us_holiday: bool

    model_config = ConfigDict(extra="forbid")


class PriceRecord(BaseModel):
    """Schema for one weekly store/SKU price."""

    store_id: str
    sku_id: str
    week_start: date
    price: float
    is_promo: bool

    model_config = ConfigDict(extra="forbid")


class SalesRecord(BaseModel):
    """Schema for one daily store/SKU sales observation."""

    store_id: str
    sku_id: str
    date: date
    units_sold: int
    revenue: float

    model_config = ConfigDict(extra="forbid")
