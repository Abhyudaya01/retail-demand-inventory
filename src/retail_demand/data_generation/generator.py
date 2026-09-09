"""Synthetic retail data generator for the demand and inventory project."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import typer

from retail_demand.config import Settings, get_settings
from retail_demand.io.manifest import write_manifest
from retail_demand.io.s3_writer import S3ParquetWriter, WriteMode
from retail_demand.utils.logging import get_logger

app = typer.Typer(help="Generate synthetic retail demand parquet datasets.")


STORE_COLUMNS = ["store_id", "region", "state", "store_type", "open_date", "sqft"]
PRODUCT_COLUMNS = [
    "sku_id",
    "category",
    "subcategory",
    "base_price",
    "is_perishable",
    "pack_size",
]
CALENDAR_COLUMNS = [
    "date",
    "dow",
    "week",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_us_holiday",
]
PRICE_COLUMNS = ["store_id", "sku_id", "week_start", "price", "is_promo"]
SALES_COLUMNS = ["store_id", "sku_id", "date", "units_sold", "revenue"]


class OutputTarget(StrEnum):
    """Supported parquet output targets."""

    local = "local"
    s3 = "s3"


class OutputMode(StrEnum):
    """Supported parquet write modes."""

    overwrite = "overwrite"
    append = "append"


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration values that control synthetic data generation."""

    num_stores: int = 50
    num_skus: int = 2_000
    num_days: int = 365 * 3
    end_date: date = field(default_factory=date.today)
    seed: int = 42
    duplicate_sales_rate: float = 0.005
    negative_units_rate: float = 0.002
    missing_price_rate: float = 0.002
    wrong_store_case_rate: float = 0.001
    price_chunk_skus: int = 50

    @property
    def start_date(self) -> date:
        """Inputs: config dates; outputs: inclusive start date; side effects: none."""
        return self.end_date - timedelta(days=self.num_days - 1)

    @property
    def expected_sales_rows(self) -> int:
        """Inputs: config dimensions; outputs: base sales row count; side effects: none."""
        return self.num_stores * self.num_skus * self.num_days


@dataclass(frozen=True)
class GenerationContext:
    """Internal demand parameters shared by price and sales generation."""

    base_demand: pd.DataFrame
    product_profiles: pd.DataFrame
    holiday_effects: pd.DataFrame


@dataclass(frozen=True)
class SyntheticDataset:
    """In-memory generated tables for small samples and tests."""

    stores: pd.DataFrame
    products: pd.DataFrame
    calendar: pd.DataFrame
    prices: pd.DataFrame
    sales: pd.DataFrame


def _category_catalog() -> dict[str, list[str]]:
    """Inputs: none; outputs: category/subcategory catalog; side effects: none."""
    return {
        "Grocery": ["Pantry", "Breakfast", "Baking", "Snacks"],
        "Dairy": ["Milk", "Cheese", "Yogurt", "Cream"],
        "Produce": ["Fruit", "Vegetables", "Fresh Herbs", "Salads"],
        "Meat": ["Beef", "Chicken", "Pork", "Seafood"],
        "Frozen": ["Meals", "Desserts", "Vegetables", "Breakfast"],
        "Beverages": ["Coffee", "Tea", "Soda", "Juice"],
        "Health": ["Vitamins", "First Aid", "Wellness", "Personal Care"],
        "Beauty": ["Skin Care", "Hair Care", "Cosmetics", "Fragrance"],
        "Household": ["Cleaning", "Paper Goods", "Laundry", "Storage"],
        "Pet": ["Dog", "Cat", "Treats", "Supplies"],
        "Baby": ["Diapers", "Feeding", "Care", "Toys"],
        "Electronics": ["Audio", "Charging", "Accessories", "Smart Home"],
        "Apparel": ["Basics", "Seasonal", "Active", "Accessories"],
        "Home": ["Kitchen", "Bedding", "Bath", "Decor"],
        "Seasonal": ["Holiday", "Outdoor", "School", "Celebration"],
        "Office": ["Paper", "Writing", "Organization", "Shipping"],
        "Toys": ["Games", "Learning", "Outdoor", "Collectibles"],
        "Automotive": ["Fluids", "Accessories", "Cleaning", "Emergency"],
        "Garden": ["Plants", "Soil", "Tools", "Pest Control"],
        "Bakery": ["Bread", "Pastry", "Dessert", "Prepared"],
    }


def _region_states() -> dict[str, list[str]]:
    """Inputs: none; outputs: region/state mapping; side effects: none."""
    return {
        "Northeast": ["NY", "NJ", "MA", "PA", "CT"],
        "Southeast": ["FL", "GA", "NC", "TN", "VA"],
        "Midwest": ["IL", "OH", "MI", "MN", "WI"],
        "Southwest": ["TX", "AZ", "NM", "OK", "CO"],
        "West": ["CA", "WA", "OR", "NV", "UT"],
    }


def _rate_count(row_count: int, rate: float) -> int:
    """Inputs: row count and rate; outputs: deterministic affected count; side effects: none."""
    if row_count == 0 or rate <= 0:
        return 0
    return max(1, int(round(row_count * rate)))


def _week_start_from_dates(dates: pd.Series) -> pd.Series:
    """Inputs: date-like series; outputs: Monday week-start timestamps; side effects: none."""
    datetime_values = pd.to_datetime(dates)
    return datetime_values - pd.to_timedelta(datetime_values.dt.dayofweek, unit="D")


def generate_stores(config: SyntheticDataConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Inputs: config and RNG; outputs: stores DataFrame; side effects: advances RNG state."""
    states_by_region = _region_states()
    regions = np.array(list(states_by_region))
    chosen_regions = rng.choice(regions, size=config.num_stores, replace=True)
    store_types = rng.choice(
        np.array(["A", "B", "C"]),
        size=config.num_stores,
        p=[0.45, 0.35, 0.20],
    )
    sqft_baseline = {"A": 62_000, "B": 42_000, "C": 26_000}
    sqft = [
        max(12_000, int(rng.normal(sqft_baseline[store_type], sqft_baseline[store_type] * 0.12)))
        for store_type in store_types
    ]
    open_offsets = rng.integers(365, 365 * 18, size=config.num_stores)
    stores = pd.DataFrame(
        {
            "store_id": [f"store_{idx:04d}" for idx in range(1, config.num_stores + 1)],
            "region": chosen_regions,
            "state": [rng.choice(states_by_region[region]) for region in chosen_regions],
            "store_type": store_types,
            "open_date": [
                config.start_date - timedelta(days=int(offset))
                for offset in open_offsets
            ],
            "sqft": sqft,
        }
    )
    return stores[STORE_COLUMNS]


def generate_products(config: SyntheticDataConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Inputs: config and RNG; outputs: products DataFrame; side effects: advances RNG state."""
    catalog = _category_catalog()
    categories = np.array(list(catalog))
    category_choices = rng.choice(categories, size=config.num_skus, replace=True)
    perishable_categories = {"Dairy", "Produce", "Meat", "Bakery"}
    category_price_centers = {
        category: rng.uniform(2.5, 28.0) * (1.4 if category == "Electronics" else 1.0)
        for category in categories
    }
    products = pd.DataFrame(
        {
            "sku_id": [f"sku_{idx:05d}" for idx in range(1, config.num_skus + 1)],
            "category": category_choices,
            "subcategory": [
                rng.choice(catalog[category])
                for category in category_choices
            ],
            "base_price": [
                round(max(0.99, rng.lognormal(np.log(category_price_centers[category]), 0.28)), 2)
                for category in category_choices
            ],
            "is_perishable": [category in perishable_categories for category in category_choices],
            "pack_size": rng.choice(np.array([1, 2, 4, 6, 12, 24]), size=config.num_skus),
        }
    )
    return products[PRODUCT_COLUMNS]


def generate_calendar(config: SyntheticDataConfig) -> pd.DataFrame:
    """Inputs: config date range; outputs: calendar DataFrame; side effects: none."""
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    years = sorted({int(value) for value in dates.year})
    us_holidays = holidays.country_holidays("US", years=years)
    calendar = pd.DataFrame({"date": dates})
    iso_calendar = calendar["date"].dt.isocalendar()
    calendar["dow"] = calendar["date"].dt.dayofweek
    calendar["week"] = iso_calendar.week.astype(int)
    calendar["month"] = calendar["date"].dt.month
    calendar["quarter"] = calendar["date"].dt.quarter
    calendar["year"] = calendar["date"].dt.year
    calendar["is_weekend"] = calendar["dow"].isin([5, 6])
    calendar["is_us_holiday"] = calendar["date"].dt.date.isin(us_holidays)
    return calendar[CALENDAR_COLUMNS]


def build_generation_context(
    config: SyntheticDataConfig,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    rng: np.random.Generator,
) -> GenerationContext:
    """Inputs: dimensions and RNG; outputs: demand context; side effects: advances RNG state."""
    categories = sorted(products["category"].unique())
    store_types = sorted(stores["store_type"].unique())
    category_baselines = {
        category: rng.gamma(shape=1.9, scale=0.75) + 0.15
        for category in categories
    }
    type_effects = {"A": 1.18, "B": 1.0, "C": 0.78}
    base_demand = pd.DataFrame(
        [
            {
                "store_type": store_type,
                "category": category,
                "base_lambda": category_baselines[category]
                * type_effects.get(store_type, 1.0)
                * rng.lognormal(mean=0.0, sigma=0.12),
            }
            for store_type in store_types
            for category in categories
        ]
    )

    sku_count = len(products)
    intermittent_count = min(sku_count, _rate_count(sku_count, 0.05))
    intermittent_positions = set(rng.choice(sku_count, size=intermittent_count, replace=False))
    high_volume_pool = [idx for idx in range(sku_count) if idx not in intermittent_positions]
    high_volume_count = min(len(high_volume_pool), _rate_count(sku_count, 0.05))
    high_volume_positions = set(rng.choice(high_volume_pool, size=high_volume_count, replace=False))
    category_elasticities = {category: rng.uniform(-2.5, -1.5) for category in categories}
    product_profiles = products[["sku_id", "category"]].copy()
    product_profiles["sku_multiplier"] = rng.lognormal(mean=0.0, sigma=0.32, size=sku_count)
    product_profiles["elasticity"] = (
        product_profiles["category"].map(category_elasticities).astype(float)
    )
    product_profiles["is_intermittent"] = [
        position in intermittent_positions for position in range(sku_count)
    ]
    product_profiles["is_high_volume"] = [
        position in high_volume_positions for position in range(sku_count)
    ]
    holiday_effects = pd.DataFrame(
        {
            "category": categories,
            "holiday_factor": [
                rng.uniform(0.82, 0.96)
                if category in {"Office", "Automotive"}
                else rng.uniform(1.08, 1.35)
                for category in categories
            ],
        }
    )
    return GenerationContext(
        base_demand=base_demand,
        product_profiles=product_profiles,
        holiday_effects=holiday_effects,
    )


def generate_prices(
    config: SyntheticDataConfig,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inputs: dimensions and RNG; outputs: weekly prices; side effects: advances RNG state."""
    weeks = pd.DataFrame({"week_start": _week_start_from_dates(calendar["date"]).drop_duplicates()})
    prices = (
        stores[["store_id"]]
        .merge(products[["sku_id", "base_price"]], how="cross")
        .merge(weeks, how="cross")
    )
    row_count = len(prices)
    promo_mask = rng.random(row_count) < 0.15
    if row_count > 0 and not promo_mask.any():
        promo_mask[rng.integers(0, row_count)] = True
    discounts = rng.uniform(0.10, 0.40, size=row_count)
    prices["is_promo"] = promo_mask
    prices["price"] = np.where(
        prices["is_promo"],
        prices["base_price"] * (1.0 - discounts),
        prices["base_price"],
    ).round(2)

    missing_count = min(row_count - 1, _rate_count(row_count, config.missing_price_rate))
    if missing_count > 0:
        missing_indices = rng.choice(prices.index.to_numpy(), size=missing_count, replace=False)
        prices.loc[missing_indices, "price"] = np.nan

    if len(prices) > 0 and not prices["is_promo"].any():
        first_index = prices.index[0]
        prices.loc[first_index, "is_promo"] = True
        prices.loc[first_index, "price"] = round(
            float(prices.loc[first_index, "base_price"]) * 0.75,
            2,
        )

    return prices[PRICE_COLUMNS].reset_index(drop=True)


def _sales_base_frame(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Inputs: dimension tables; outputs: store/SKU/date frame; side effects: none."""
    calendar_with_week = calendar.copy()
    calendar_with_week["week_start"] = _week_start_from_dates(calendar_with_week["date"])
    return (
        stores[["store_id", "store_type", "sqft"]]
        .merge(products[["sku_id", "category", "base_price"]], how="cross")
        .merge(
            calendar_with_week[
                [
                    "date",
                    "week_start",
                    "dow",
                    "month",
                    "is_us_holiday",
                ]
            ],
            how="cross",
        )
    )


def generate_sales(
    config: SyntheticDataConfig,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    context: GenerationContext,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inputs: dimensions, prices, context, RNG; outputs: sales; side effects: advances RNG."""
    sales = _sales_base_frame(stores, products, calendar)
    sales = sales.merge(prices, on=["store_id", "sku_id", "week_start"], how="left")
    sales["price"] = sales["price"].fillna(sales["base_price"])
    sales["is_promo"] = sales["is_promo"].eq(True)
    sales = sales.merge(context.base_demand, on=["store_type", "category"], how="left")
    sales = sales.merge(context.product_profiles, on=["sku_id", "category"], how="left")
    sales = sales.merge(context.holiday_effects, on="category", how="left")

    dow_effects = pd.Series({0: 0.86, 1: 0.90, 2: 0.96, 3: 1.00, 4: 1.12, 5: 1.28, 6: 1.16})
    month_effects = pd.Series(
        {
            1: 0.90,
            2: 0.88,
            3: 0.96,
            4: 1.00,
            5: 1.04,
            6: 1.10,
            7: 1.08,
            8: 1.02,
            9: 1.00,
            10: 1.05,
            11: 1.22,
            12: 1.34,
        }
    )
    median_sqft = max(float(stores["sqft"].median()), 1.0)
    store_size_effect = np.clip((sales["sqft"] / median_sqft) ** 0.25, 0.72, 1.35)
    price_ratio = np.clip(sales["price"] / sales["base_price"], 0.4, 1.6)
    promo_effect = np.power(price_ratio, sales["elasticity"])
    holiday_effect = np.where(sales["is_us_holiday"], sales["holiday_factor"], 1.0)
    intermittent_effect = np.where(sales["is_intermittent"], 0.32, 1.0)
    high_volume_effect = np.where(sales["is_high_volume"], 3.8, 1.0)
    demand_lambda = (
        sales["base_lambda"]
        * sales["sku_multiplier"]
        * store_size_effect
        * sales["dow"].map(dow_effects).astype(float)
        * sales["month"].map(month_effects).astype(float)
        * holiday_effect
        * promo_effect
        * intermittent_effect
        * high_volume_effect
    )
    demand_lambda = np.clip(demand_lambda, 0.01, 500.0)
    units = rng.poisson(demand_lambda)
    intermittent_zero_mask = sales["is_intermittent"].to_numpy() & (rng.random(len(sales)) < 0.72)
    units[intermittent_zero_mask] = 0
    sales["units_sold"] = units.astype(int)
    sales["revenue"] = (sales["units_sold"] * sales["price"]).round(2)
    sales = _inject_sales_dirtiness(sales, config, rng)
    return sales[SALES_COLUMNS].reset_index(drop=True)


def _inject_sales_dirtiness(
    sales: pd.DataFrame,
    config: SyntheticDataConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inputs: clean sales, config, RNG; outputs: dirty sales; side effects: advances RNG state."""
    dirty_sales = sales.copy()
    row_count = len(dirty_sales)

    negative_count = min(row_count, _rate_count(row_count, config.negative_units_rate))
    if negative_count > 0:
        negative_indices = rng.choice(
            dirty_sales.index.to_numpy(),
            size=negative_count,
            replace=False,
        )
        dirty_sales.loc[negative_indices, "units_sold"] = -(
            dirty_sales.loc[negative_indices, "units_sold"].abs() + 1
        )
        dirty_sales.loc[negative_indices, "revenue"] = (
            dirty_sales.loc[negative_indices, "units_sold"]
            * dirty_sales.loc[negative_indices, "price"]
        ).round(2)

    wrong_case_count = min(row_count, _rate_count(row_count, config.wrong_store_case_rate))
    if wrong_case_count > 0:
        wrong_case_indices = rng.choice(
            dirty_sales.index.to_numpy(),
            size=wrong_case_count,
            replace=False,
        )
        dirty_sales.loc[wrong_case_indices, "store_id"] = dirty_sales.loc[
            wrong_case_indices,
            "store_id",
        ].str.upper()

    duplicate_count = min(row_count, _rate_count(row_count, config.duplicate_sales_rate))
    if duplicate_count > 0:
        duplicate_indices = rng.choice(
            dirty_sales.index.to_numpy(),
            size=duplicate_count,
            replace=False,
        )
        dirty_sales = pd.concat(
            [dirty_sales, dirty_sales.loc[duplicate_indices].copy()],
            ignore_index=True,
        )

    return dirty_sales


def generate_dataset(config: SyntheticDataConfig) -> SyntheticDataset:
    """Inputs: config; outputs: all generated tables; side effects: allocates DataFrames."""
    rng = np.random.default_rng(config.seed)
    stores = generate_stores(config, rng)
    products = generate_products(config, rng)
    calendar = generate_calendar(config)
    context = build_generation_context(config, stores, products, rng)
    prices = generate_prices(config, stores, products, calendar, rng)
    sales = generate_sales(config, stores, products, calendar, prices, context, rng)
    return SyntheticDataset(
        stores=stores,
        products=products,
        calendar=calendar,
        prices=prices,
        sales=sales,
    )


def _iter_product_chunks(products: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """Inputs: products and chunk size; outputs: product chunks; side effects: none."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [
        products.iloc[start : start + chunk_size].copy()
        for start in range(0, len(products), chunk_size)
    ]


def _resolve_output_root(target: OutputTarget, output_path: str, settings: Settings) -> str:
    """Inputs: target/path/settings; outputs: local or S3 root URI; side effects: none."""
    if target == OutputTarget.local:
        return str(Path(output_path).resolve())
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET is required for --target s3")
    return f"s3://{settings.s3_bucket}/{settings.s3_raw_prefix.strip('/')}"


def _parse_end_date(value: str | None, default: date) -> date:
    """Inputs: optional ISO date and default; outputs: parsed date; side effects: none."""
    if value is None:
        return default
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter("--end-date must use YYYY-MM-DD format") from exc


def _filesystem_and_path(root: str) -> tuple[pafs.FileSystem, str]:
    """Inputs: output root; outputs: pyarrow filesystem/path; side effects: creates local root."""
    if root.startswith("s3://"):
        return pafs.FileSystem.from_uri(root)
    local_root = Path(root)
    local_root.mkdir(parents=True, exist_ok=True)
    return pafs.LocalFileSystem(), str(local_root)


def _write_single_parquet(
    df: pd.DataFrame,
    table_name: str,
    filesystem: pafs.FileSystem,
    root_path: str,
) -> dict[str, int | str | list[str]]:
    """Inputs: table and destination; outputs: write stats; side effects: writes parquet."""
    table_path = f"{root_path.rstrip('/')}/{table_name}"
    filesystem.create_dir(table_path, recursive=True)
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    file_path = f"{table_path}/{table_name}.parquet"
    pq.write_table(arrow_table, file_path, filesystem=filesystem)
    return {
        "rows": int(len(df)),
        "bytes": _local_path_size(table_path),
        "uri": table_path,
        "partition_cols": [],
    }


def _write_partitioned_parquet(
    df: pd.DataFrame,
    table_name: str,
    partition_cols: list[str],
    filesystem: pafs.FileSystem,
    root_path: str,
    part_number: int,
) -> dict[str, int | str | list[str]]:
    """Inputs: table and partition columns; outputs: write stats; side effects: writes parquet."""
    table_path = f"{root_path.rstrip('/')}/{table_name}"
    filesystem.create_dir(table_path, recursive=True)
    before_bytes = _local_path_size(table_path)
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        arrow_table,
        root_path=table_path,
        partition_cols=partition_cols,
        filesystem=filesystem,
        basename_template=f"part-{part_number:05d}-{{i}}.parquet",
    )
    after_bytes = _local_path_size(table_path)
    return {
        "rows": int(len(df)),
        "bytes": max(0, after_bytes - before_bytes),
        "uri": table_path,
        "partition_cols": partition_cols,
    }


def _local_path_size(path: str) -> int:
    """Inputs: local path; outputs: byte size below path; side effects: reads file metadata."""
    local_path = Path(path)
    if not local_path.exists():
        return 0
    if local_path.is_file():
        return local_path.stat().st_size
    return sum(
        file_path.stat().st_size for file_path in local_path.rglob("*") if file_path.is_file()
    )


def _delete_local_table(root_path: str, table_name: str) -> None:
    """Inputs: local root/table; outputs: none; side effects: deletes that local table path."""
    table_path = Path(root_path) / table_name
    if table_path.exists():
        shutil.rmtree(table_path)


def _with_price_partitions(prices: pd.DataFrame) -> pd.DataFrame:
    """Inputs: prices DataFrame; outputs: prices with year partition; side effects: none."""
    price_output = prices.copy()
    price_output["year"] = pd.to_datetime(price_output["week_start"]).dt.year
    return price_output


def _with_sales_partitions(sales: pd.DataFrame) -> pd.DataFrame:
    """Inputs: sales DataFrame; outputs: sales with year/month partitions; side effects: none."""
    sales_output = sales.copy()
    sales_output["year"] = pd.to_datetime(sales_output["date"]).dt.year
    sales_output["month"] = pd.to_datetime(sales_output["date"]).dt.month
    return sales_output


def _calendar_month_chunks(calendar: pd.DataFrame) -> list[pd.DataFrame]:
    """Inputs: calendar table; outputs: per-month calendar chunks; side effects: none."""
    calendar_with_keys = calendar.copy()
    calendar_with_keys["chunk_year"] = pd.to_datetime(calendar_with_keys["date"]).dt.year
    calendar_with_keys["chunk_month"] = pd.to_datetime(calendar_with_keys["date"]).dt.month
    chunks = []
    for _, chunk in calendar_with_keys.groupby(["chunk_year", "chunk_month"], sort=True):
        chunks.append(chunk.drop(columns=["chunk_year", "chunk_month"]).reset_index(drop=True))
    return chunks


def _empty_stats(settings: Settings, target: OutputTarget, output_root: str) -> dict[str, dict]:
    """Inputs: settings/target/root; outputs: stats shell; side effects: none."""
    tables = {
        "stores": [],
        "products": [],
        "calendar": [],
        "prices": ["year"],
        "sales": ["year", "month"],
    }
    stats = {}
    for table, partition_cols in tables.items():
        uri = (
            settings.s3_uri("raw", table)
            if target == OutputTarget.s3
            else f"{output_root}/{table}"
        )
        stats[table] = {"rows": 0, "bytes": 0, "uri": uri, "partition_cols": partition_cols}
    return stats


def _merge_table_stats(
    table_stats: dict[str, dict],
    table_name: str,
    write_stats: dict[str, int | str | list[str]],
) -> None:
    """Inputs: accumulated and latest stats; outputs: none; side effects: mutates stats dict."""
    table_stats[table_name]["rows"] += int(write_stats["rows"])
    table_stats[table_name]["bytes"] += int(write_stats["bytes"])
    table_stats[table_name]["uri"] = str(write_stats["uri"])
    table_stats[table_name]["partition_cols"] = list(write_stats["partition_cols"])


def _estimated_plan(
    config: SyntheticDataConfig,
    target: OutputTarget,
    output_path: str,
    settings: Settings,
) -> dict[str, dict[str, int | str | list[str]]]:
    """Inputs: config and destination; outputs: planned write estimates; side effects: none."""
    output_root = _resolve_output_root(target, output_path, settings)
    week_count = int(
        pd.Series(pd.date_range(config.start_date, config.end_date, freq="D"))
        .pipe(_week_start_from_dates)
        .nunique()
    )
    estimates = {
        "stores": config.num_stores,
        "products": config.num_skus,
        "calendar": config.num_days,
        "prices": config.num_stores * config.num_skus * week_count,
        "sales": config.expected_sales_rows
        + _rate_count(config.expected_sales_rows, config.duplicate_sales_rate),
    }
    partition_cols = {"prices": ["year"], "sales": ["year", "month"]}
    return {
        table: {
            "rows": rows,
            "bytes": 0,
            "uri": settings.s3_uri("raw", table)
            if target == OutputTarget.s3
            else f"{output_root}/{table}",
            "partition_cols": partition_cols.get(table, []),
        }
        for table, rows in estimates.items()
    }


def _print_dry_run_plan(
    config: SyntheticDataConfig,
    target: OutputTarget,
    output_path: str,
    settings: Settings,
) -> None:
    """Inputs: config and destination; outputs: none; side effects: prints planned writes."""
    typer.echo("Dry run: no files will be written.")
    for table, stats in _estimated_plan(config, target, output_path, settings).items():
        partition_cols = ",".join(stats["partition_cols"]) or "none"
        typer.echo(
            f"{table}: uri={stats['uri']} rows~={stats['rows']:,} partitions={partition_cols}"
        )


def _write_dimension_tables(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    target: OutputTarget,
    mode: WriteMode,
    writer: S3ParquetWriter | None,
    filesystem: pafs.FileSystem | None,
    root_path: str,
    table_stats: dict[str, dict],
) -> None:
    """Inputs: dimensions and writer; outputs: none; side effects: writes dimension parquet."""
    dimension_tables = {"stores": stores, "products": products, "calendar": calendar}
    for table_name, table_df in dimension_tables.items():
        if target == OutputTarget.s3:
            if writer is None:
                raise ValueError("S3 writer is required for S3 targets")
            write_stats = writer.write_table(table_df, table_name, mode=mode)
        else:
            if filesystem is None:
                raise ValueError("Local filesystem is required for local targets")
            if mode == "overwrite":
                _delete_local_table(root_path, table_name)
            write_stats = _write_single_parquet(table_df, table_name, filesystem, root_path)
        _merge_table_stats(table_stats, table_name, write_stats)


def write_dataset(
    config: SyntheticDataConfig,
    target: OutputTarget,
    output_path: str,
    settings: Settings,
    mode: WriteMode = "overwrite",
    dry_run: bool = False,
) -> str:
    """Inputs: config, target, output path; outputs: resolved root; side effects: writes parquet."""
    if settings.aws_profile:
        os.environ["AWS_PROFILE"] = settings.aws_profile

    if dry_run:
        _print_dry_run_plan(config, target, output_path, settings)
        return _resolve_output_root(target, output_path, settings)

    logger = get_logger(__name__)
    rng = np.random.default_rng(config.seed)
    stores = generate_stores(config, rng)
    products = generate_products(config, rng)
    calendar = generate_calendar(config)
    context = build_generation_context(config, stores, products, rng)
    output_root = _resolve_output_root(target, output_path, settings)
    table_stats = _empty_stats(settings, target, output_root)
    filesystem: pafs.FileSystem | None = None
    root_path = output_root
    writer: S3ParquetWriter | None = None
    if target == OutputTarget.s3:
        writer = S3ParquetWriter(settings)
    else:
        filesystem, root_path = _filesystem_and_path(output_root)

    _write_dimension_tables(
        stores,
        products,
        calendar,
        target,
        mode,
        writer,
        filesystem,
        root_path,
        table_stats,
    )

    price_part_number = 0
    sales_part_number = 0
    for product_chunk_number, product_chunk in enumerate(
        _iter_product_chunks(products, config.price_chunk_skus),
        start=1,
    ):
        prices = generate_prices(config, stores, product_chunk, calendar, rng)
        price_output = _with_price_partitions(prices)
        price_mode = mode if product_chunk_number == 1 else "append"
        if target == OutputTarget.s3:
            if writer is None:
                raise ValueError("S3 writer is required for S3 targets")
            price_stats = writer.write_table(
                price_output,
                "prices",
                partition_cols=["year"],
                mode=price_mode,
            )
        else:
            if filesystem is None:
                raise ValueError("Local filesystem is required for local targets")
            if price_mode == "overwrite":
                _delete_local_table(root_path, "prices")
            price_stats = _write_partitioned_parquet(
                price_output,
                "prices",
                ["year"],
                filesystem,
                root_path,
                price_part_number,
            )
            price_part_number += 1
        _merge_table_stats(table_stats, "prices", price_stats)

        for month_number, month_calendar in enumerate(_calendar_month_chunks(calendar), start=1):
            sales = generate_sales(
                config,
                stores,
                product_chunk,
                month_calendar,
                prices,
                context,
                rng,
            )
            sales_output = _with_sales_partitions(sales)
            sales_mode = mode if product_chunk_number == 1 and month_number == 1 else "append"
            if target == OutputTarget.s3:
                if writer is None:
                    raise ValueError("S3 writer is required for S3 targets")
                sales_stats = writer.write_table(
                    sales_output,
                    "sales",
                    partition_cols=["year", "month"],
                    mode=sales_mode,
                )
            else:
                if filesystem is None:
                    raise ValueError("Local filesystem is required for local targets")
                if sales_mode == "overwrite":
                    _delete_local_table(root_path, "sales")
                sales_stats = _write_partitioned_parquet(
                    sales_output,
                    "sales",
                    ["year", "month"],
                    filesystem,
                    root_path,
                    sales_part_number,
                )
                sales_part_number += 1
            _merge_table_stats(table_stats, "sales", sales_stats)
            logger.info(
                "wrote_sales_chunk",
                product_chunk_number=product_chunk_number,
                month_number=month_number,
                sku_count=len(product_chunk),
                sales_rows=len(sales),
            )

    if target == OutputTarget.s3:
        manifest = write_manifest(settings, config, table_stats)
        logger.info("manifest_written", manifest_uri=manifest["manifest_uri"])
    logger.info("dataset_written", target=str(target), output_root=output_root)
    return output_root


@app.command()
def generate(
    target: Annotated[OutputTarget, typer.Option(help="Write destination: local or s3.")] = (
        OutputTarget.local
    ),
    output_path: Annotated[
        str,
        typer.Option(help="Local directory or S3 key/URI for generated parquet."),
    ] = "data/synthetic",
    n_stores: Annotated[int | None, typer.Option(help="Override number of stores.")] = None,
    n_skus: Annotated[int | None, typer.Option(help="Override number of SKUs.")] = None,
    days: Annotated[
        int | None,
        typer.Option(help="Override number of daily calendar rows."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(help="Inclusive generation end date as YYYY-MM-DD."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Override random seed.")] = None,
    chunk_skus: Annotated[int, typer.Option(help="Number of SKUs per parquet write chunk.")] = 50,
    dry_run: Annotated[
        bool,
        typer.Option(help="Print planned writes without writing data."),
    ] = False,
    mode: Annotated[OutputMode, typer.Option(help="Write mode: overwrite or append.")] = (
        OutputMode.overwrite
    ),
) -> None:
    """Inputs: CLI options; outputs: none; side effects: writes generated parquet datasets."""
    settings = get_settings()
    config = SyntheticDataConfig(
        num_stores=n_stores or settings.default_num_stores,
        num_skus=n_skus or settings.default_num_skus,
        num_days=days or settings.default_num_days,
        end_date=_parse_end_date(end_date, settings.data_end_date),
        seed=seed if seed is not None else settings.random_seed,
        price_chunk_skus=chunk_skus,
    )
    output_root = write_dataset(
        config,
        target,
        output_path,
        settings,
        mode=mode.value,
        dry_run=dry_run,
    )
    if dry_run:
        typer.echo(f"Planned synthetic retail dataset at {output_root}")
    else:
        typer.echo(f"Wrote synthetic retail dataset to {output_root}")


if __name__ == "__main__":
    app()
