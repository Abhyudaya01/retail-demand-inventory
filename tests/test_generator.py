"""Tests for the synthetic retail data generator."""

from __future__ import annotations

from datetime import date

import pandas as pd

from retail_demand.data_generation.generator import (
    CALENDAR_COLUMNS,
    PRICE_COLUMNS,
    PRODUCT_COLUMNS,
    SALES_COLUMNS,
    STORE_COLUMNS,
    SyntheticDataConfig,
    generate_dataset,
)


def _small_dataset_config() -> SyntheticDataConfig:
    """Inputs: none; outputs: deterministic small config; side effects: none."""
    return SyntheticDataConfig(num_stores=1, num_skus=10, num_days=90, end_date=date(2026, 1, 31))


def test_small_sample_row_counts_and_schemas() -> None:
    """Inputs: generated sample; outputs: assertions; side effects: none."""
    config = _small_dataset_config()
    dataset = generate_dataset(config)

    assert list(dataset.stores.columns) == STORE_COLUMNS
    assert list(dataset.products.columns) == PRODUCT_COLUMNS
    assert list(dataset.calendar.columns) == CALENDAR_COLUMNS
    assert list(dataset.prices.columns) == PRICE_COLUMNS
    assert list(dataset.sales.columns) == SALES_COLUMNS

    assert len(dataset.stores) == 1
    assert len(dataset.products) == 10
    assert len(dataset.calendar) == 90

    week_count = _week_count(dataset.calendar)
    expected_price_rows = config.num_stores * config.num_skus * week_count
    assert len(dataset.prices) == expected_price_rows

    expected_duplicates = max(
        1,
        int(round(config.expected_sales_rows * config.duplicate_sales_rate)),
    )
    assert len(dataset.sales) == config.expected_sales_rows + expected_duplicates


def test_required_columns_have_no_nans() -> None:
    """Inputs: generated sample; outputs: assertions; side effects: none."""
    dataset = generate_dataset(_small_dataset_config())

    tables = [dataset.stores, dataset.products, dataset.calendar, dataset.sales]
    columns = [STORE_COLUMNS, PRODUCT_COLUMNS, CALENDAR_COLUMNS, SALES_COLUMNS]
    for table, required_columns in zip(tables, columns, strict=True):
        assert not table[required_columns].isna().any().any()

    price_required_except_price = [column for column in PRICE_COLUMNS if column != "price"]
    assert not dataset.prices[price_required_except_price].isna().any().any()


def test_missing_prices_rate() -> None:
    """Inputs: generated prices; outputs: null-price rate assertion; side effects: none."""
    config = SyntheticDataConfig(
        num_stores=5,
        num_skus=200,
        num_days=90,
        end_date=date(2026, 1, 31),
    )
    dataset = generate_dataset(config)

    missing_rate = float(dataset.prices["price"].isna().mean())

    assert 0.001 <= missing_rate <= 0.010


def test_promo_lift_is_visible_in_aggregate() -> None:
    """Inputs: generated sample; outputs: assertions; side effects: none."""
    dataset = generate_dataset(_small_dataset_config())
    calendar = dataset.calendar[["date"]].copy()
    calendar["week_start"] = pd.to_datetime(calendar["date"]) - pd.to_timedelta(
        pd.to_datetime(calendar["date"]).dt.dayofweek,
        unit="D",
    )
    clean_sales = dataset.sales[
        dataset.sales["store_id"].isin(dataset.stores["store_id"])
        & (dataset.sales["units_sold"] >= 0)
    ]
    joined = clean_sales.merge(calendar, on="date", how="left").merge(
        dataset.prices,
        on=["store_id", "sku_id", "week_start"],
        how="inner",
    )
    sku_promo_means = (
        joined.groupby(["store_id", "sku_id", "is_promo"])["units_sold"]
        .mean()
        .unstack("is_promo")
        .dropna()
    )
    assert not sku_promo_means.empty
    assert (sku_promo_means[True] / sku_promo_means[False]).mean() > 1.05


def test_dq_dirtiness_is_present_at_expected_rate() -> None:
    """Inputs: generated sample; outputs: assertions; side effects: none."""
    config = _small_dataset_config()
    dataset = generate_dataset(config)
    duplicate_rows = len(dataset.sales) - len(dataset.sales.drop_duplicates())
    negative_rows = int((dataset.sales["units_sold"] < 0).sum())
    wrong_case_rows = int(dataset.sales["store_id"].str.contains("[A-Z]", regex=True).sum())

    duplicate_rate = duplicate_rows / config.expected_sales_rows
    negative_rate = negative_rows / config.expected_sales_rows
    wrong_case_rate = wrong_case_rows / config.expected_sales_rows

    assert 0.002 <= duplicate_rate <= 0.010
    assert 0.001 <= negative_rate <= 0.005
    assert 0.0005 <= wrong_case_rate <= 0.004


def _week_count(calendar: pd.DataFrame) -> int:
    """Inputs: calendar DataFrame; outputs: number of weeks represented; side effects: none."""
    week_start = pd.to_datetime(calendar["date"]) - pd.to_timedelta(
        pd.to_datetime(calendar["date"]).dt.dayofweek,
        unit="D",
    )
    return int(week_start.nunique())
