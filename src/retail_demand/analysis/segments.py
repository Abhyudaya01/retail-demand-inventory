"""Segment assignment helpers for Phase 8 error analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_GROUP_COLS = ["store_id", "sku_id"]
NANOSECONDS_PER_DAY = 86_400_000_000_000


def _to_datetime(values: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(values):
        return pd.to_datetime(values, unit="ns")
    return pd.to_datetime(values)


def _week_start(values: pd.Series) -> pd.Series:
    dates = _to_datetime(values)
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def assign_volume_tier(
    sales_history_df: pd.DataFrame, group_cols: list[str] | None = None
) -> pd.DataFrame:
    """Inputs: sales history; outputs: group-level Pareto volume tiers; side effects: none."""
    group_cols = group_cols or DEFAULT_GROUP_COLS
    grouped = (
        sales_history_df.groupby(group_cols, dropna=False, observed=True)["units_sold"]
        .sum()
        .reset_index(name="total_units_sold")
        .sort_values("total_units_sold", ascending=False)
        .reset_index(drop=True)
    )
    total_volume = float(grouped["total_units_sold"].sum())
    if total_volume <= 0.0:
        grouped["volume_share"] = 0.0
        grouped["cumulative_volume_share"] = 0.0
        grouped["volume_tier"] = "C_bottom50pct"
        return grouped

    grouped["volume_share"] = grouped["total_units_sold"] / total_volume
    previous_share = grouped["volume_share"].cumsum().shift(fill_value=0.0)
    grouped["cumulative_volume_share"] = grouped["volume_share"].cumsum()
    grouped["volume_tier"] = np.select(
        [previous_share < 0.20, previous_share < 0.50],
        ["A_top20pct", "B_middle30pct"],
        default="C_bottom50pct",
    )
    return grouped


def assign_intermittency_class(
    sales_history_df: pd.DataFrame, group_cols: list[str] | None = None
) -> pd.DataFrame:
    """Inputs: sales history; outputs: group-level ADI/CV2 demand classes; side effects: none."""
    group_cols = group_cols or DEFAULT_GROUP_COLS
    rows: list[dict[str, Any]] = []
    for keys, group in sales_history_df.groupby(group_cols, dropna=False, observed=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        units = group["units_sold"].astype(float)
        non_zero = units[units > 0.0]
        total_periods = int(len(units))
        non_zero_periods = int(len(non_zero))
        adi = float("inf") if non_zero_periods == 0 else total_periods / non_zero_periods
        if non_zero_periods == 0:
            cv_squared = float("inf")
        else:
            mean_demand = float(non_zero.mean())
            variance = float(non_zero.var(ddof=0))
            cv_squared = float("inf") if mean_demand == 0.0 else variance / (mean_demand**2)

        if adi < 1.32 and cv_squared < 0.49:
            demand_class = "smooth"
        elif adi < 1.32:
            demand_class = "erratic"
        elif cv_squared < 0.49:
            demand_class = "intermittent"
        else:
            demand_class = "lumpy"

        rows.append(
            {
                **dict(zip(group_cols, key_values, strict=True)),
                "total_periods": total_periods,
                "non_zero_periods": non_zero_periods,
                "adi": adi,
                "cv_squared": cv_squared,
                "intermittency_class": demand_class,
            }
        )
    return pd.DataFrame(rows)


def assign_promo_flag(preds_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Inputs: predictions and weekly prices.

    Outputs: predictions with promo flag. Side effects: none.
    """
    result = preds_df.copy()
    prices = prices_df.copy()
    result["date"] = _to_datetime(result["date"])
    if "week_start" in prices.columns:
        prices["week_start"] = _to_datetime(prices["week_start"])
    else:
        prices["week_start"] = _week_start(prices["date"])
    result["week_start"] = _week_start(result["date"])
    promo = prices[["store_id", "sku_id", "week_start", "is_promo"]].drop_duplicates()
    result = result.merge(promo, on=["store_id", "sku_id", "week_start"], how="left")
    result["is_promo_day"] = result["is_promo"].fillna(False).astype(bool)
    return result.drop(columns=["is_promo"])
