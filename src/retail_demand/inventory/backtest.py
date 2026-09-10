"""Portfolio inventory backtests driven by Phase 7 point forecasts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from retail_demand.inventory.policies import (
    policy_ml_95_service,
    policy_ml_99_service,
    policy_naive_rolling_mean,
    policy_seasonal_naive,
)
from retail_demand.inventory.simulation import simulate_sku

DEFAULT_PARAMS: dict[str, float | int] = {
    "lead_time_days": 7,
    "service_level": 0.95,
    "holding_cost_rate": 0.25,
    "stockout_cost_multiplier": 3.0,
    "initial_stock_days": 14,
    "unit_margin_rate": 0.30,
}


def compute_forecast_error_std(
    preds_df: pd.DataFrame, groupby_cols: list[str] | None = None
) -> pd.DataFrame:
    """Compute validation forecast error std per group.

    Inputs are Phase 7 predictions with y_true/y_pred. Output is one row per group with
    forecast_error_std for the safety-stock formula. Side effects: none.
    """
    groupby_cols = groupby_cols or ["store_id", "sku_id"]
    frame = preds_df.copy()
    frame["forecast_error"] = frame["y_true"].astype(float) - frame["y_pred"].astype(float)
    return (
        frame.groupby(groupby_cols, dropna=False, observed=True)["forecast_error"]
        .std(ddof=0)
        .fillna(0.0)
        .reset_index(name="forecast_error_std")
    )


def _date_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column])
    return result


def _price_lookup(prices_df: pd.DataFrame, products_df: pd.DataFrame | None) -> pd.DataFrame:
    prices = prices_df.copy()
    price_table = (
        prices.groupby(["store_id", "sku_id"], dropna=False, observed=True)["price"]
        .median()
        .reset_index(name="unit_price")
    )
    if products_df is not None:
        products = products_df[["sku_id", "category", "base_price"]].drop_duplicates("sku_id")
        price_table = price_table.merge(products, on="sku_id", how="left")
        price_table["unit_price"] = price_table["unit_price"].fillna(price_table["base_price"])
    else:
        price_table["category"] = pd.NA
        price_table["base_price"] = price_table["unit_price"]
    price_table["base_price"] = price_table["base_price"].fillna(price_table["unit_price"])
    return price_table


def _default_policies(lead_time_days: int) -> dict[str, Callable[..., int]]:
    return {
        "baseline_rolling_mean": policy_naive_rolling_mean(window=7),
        "baseline_seasonal_naive": policy_seasonal_naive(),
        "ML_95": policy_ml_95_service(lead_time=lead_time_days),
        "ML_99": policy_ml_99_service(lead_time=lead_time_days),
    }


def backtest_inventory(
    preds_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    policies: dict[str, Callable[..., int]] | None,
    params: dict[str, Any] | None,
) -> dict[str, pd.DataFrame]:
    """Backtest inventory policies across all validation SKUs.

    Defaults reflect common retail planning assumptions: 7-day DC-to-store lead time,
    95% target service for the standard ML policy, 25% annual holding cost covering capital,
    warehousing and obsolescence, stockout penalty of 3 x unit margin, and two weeks of
    opening stock. Inputs are pandas DataFrames; outputs are per-SKU and portfolio summary
    DataFrames. Side effects: none.
    """
    config = {**DEFAULT_PARAMS, **(params or {})}
    lead_time_days = int(config["lead_time_days"])
    preds = _date_frame(preds_df, "date")
    actuals = _date_frame(actuals_df, "date")
    products_df = config.get("products_df")
    products = products_df.copy() if isinstance(products_df, pd.DataFrame) else None
    errors = compute_forecast_error_std(preds)
    price_table = _price_lookup(prices_df, products)
    policies = policies or _default_policies(lead_time_days)
    val_start = preds["date"].min()
    sku_keys = ["store_id", "sku_id"]
    per_sku_rows: list[dict[str, Any]] = []

    for keys, sku_preds in preds.groupby(sku_keys, dropna=False, observed=True):
        store_id, sku_id = keys
        sku_preds = sku_preds.sort_values("date").merge(errors, on=sku_keys, how="left")
        history = actuals[
            (actuals["store_id"] == store_id)
            & (actuals["sku_id"] == sku_id)
            & (actuals["date"] < val_start)
        ].sort_values("date")
        history_units = history["units_sold"].astype(float).tolist()
        mean_daily_demand = float(np.mean(history_units[-28:])) if history_units else float(
            sku_preds["y_true"].mean()
        )
        economics = price_table[
            (price_table["store_id"] == store_id) & (price_table["sku_id"] == sku_id)
        ]
        if economics.empty:
            unit_price = 1.0
            category = pd.NA
        else:
            unit_price = float(economics.iloc[0]["base_price"])
            category = economics.iloc[0].get("category", pd.NA)
        h_cost = unit_price * float(config["holding_cost_rate"]) / 365.0
        s_cost = unit_price * float(config["unit_margin_rate"]) * float(
            config["stockout_cost_multiplier"]
        )
        initial_stock = int(np.ceil(mean_daily_demand * int(config["initial_stock_days"])))

        for policy_name, policy_fn in policies.items():
            sim_input = sku_preds.copy()
            sim_input.attrs["initial_demand_history"] = history_units
            sim = simulate_sku(
                sim_input,
                policy_fn,
                initial_stock=initial_stock,
                lead_time=lead_time_days,
                h_cost=h_cost,
                s_cost=s_cost,
            )
            total_holding_cost = float(sim["holding_cost"].sum())
            total_stockout_cost = float(sim["stockout_cost"].sum())
            total_demand = float(sim["y_true"].sum())
            stockout_units = float(sim["stockout_units"].sum())
            avg_inventory = float(sim["on_hand"].mean())
            service_level = 1.0 if total_demand == 0.0 else 1.0 - stockout_units / total_demand
            per_sku_rows.append(
                {
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "category": category,
                    "policy": policy_name,
                    "total_holding_cost": total_holding_cost,
                    "total_stockout_cost": total_stockout_cost,
                    "total_cost": total_holding_cost + total_stockout_cost,
                    "service_level": service_level,
                    "inventory_turns": total_demand / max(avg_inventory, 1e-9),
                    "avg_days_of_stock": avg_inventory / max(mean_daily_demand, 1e-9),
                    "avg_inventory": avg_inventory,
                    "stockout_units": stockout_units,
                    "num_stockout_events": int((sim["stockout_units"] > 0).sum()),
                }
            )

    per_sku = pd.DataFrame(per_sku_rows)
    summary = (
        per_sku.groupby("policy", dropna=False, observed=True)
        .agg(
            total_holding_cost=("total_holding_cost", "sum"),
            total_stockout_cost=("total_stockout_cost", "sum"),
            total_cost=("total_cost", "sum"),
            mean_service_level=("service_level", "mean"),
            total_stockout_units=("stockout_units", "sum"),
            mean_avg_inventory=("avg_inventory", "mean"),
            sku_count=("sku_id", "count"),
        )
        .reset_index()
        .sort_values("total_cost")
    )
    return {"per_sku": per_sku, "summary": summary}

