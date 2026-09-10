"""Tests for Phase 9 inventory backtesting."""

from __future__ import annotations

import pandas as pd

from retail_demand.inventory.backtest import backtest_inventory, compute_forecast_error_std
from retail_demand.inventory.policies import policy_ml_95_service, policy_ml_99_service


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2026-01-01", periods=14)
    preds = pd.DataFrame(
        {
            "store_id": ["store_0"] * 14,
            "sku_id": ["sku_0"] * 14,
            "date": dates,
            "y_true": [10] * 14,
            "y_pred": [10.0] * 14,
        }
    )
    actuals = pd.DataFrame(
        {
            "store_id": ["store_0"] * 35,
            "sku_id": ["sku_0"] * 35,
            "date": pd.date_range("2025-12-01", periods=35),
            "units_sold": [10] * 35,
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["store_0"],
            "sku_id": ["sku_0"],
            "week_start": [pd.Timestamp("2025-12-29")],
            "price": [10.0],
            "is_promo": [False],
        }
    )
    products = pd.DataFrame({"sku_id": ["sku_0"], "category": ["Grocery"], "base_price": [10.0]})
    return preds, actuals, prices, products


def test_compute_forecast_error_std_per_sku() -> None:
    """Inputs: prediction errors; outputs: per-SKU error std; side effects: none."""
    preds, _actuals, _prices, _products = _frames()
    preds.loc[0, "y_pred"] = 8.0

    result = compute_forecast_error_std(preds)

    assert result.loc[0, "forecast_error_std"] > 0.0


def test_backtest_two_threshold_policies_differ_on_same_path() -> None:
    """Inputs: identical demand path and two policies; outputs: cost/service contrast."""
    preds, actuals, prices, products = _frames()

    def never_order(
        day: int, mean_forecast: float, std_forecast: float, on_hand: int, on_order: int
    ) -> int:
        del day, mean_forecast, std_forecast, on_hand, on_order
        return 0

    def always_order(
        day: int, mean_forecast: float, std_forecast: float, on_hand: int, on_order: int
    ) -> int:
        del day, mean_forecast, std_forecast, on_hand, on_order
        return 30

    result = backtest_inventory(
        preds,
        actuals,
        prices,
        {"always": always_order, "never": never_order},
        {"products_df": products, "initial_stock_days": 2, "lead_time_days": 2},
    )["summary"].set_index("policy")

    assert (
        result.loc["always", "total_stockout_units"]
        < result.loc["never", "total_stockout_units"]
    )
    assert result.loc["always", "mean_service_level"] > result.loc["never", "mean_service_level"]
    assert result.loc["always", "total_cost"] != result.loc["never", "total_cost"]


def test_ml_99_service_costs_more_and_stocks_out_less_than_95() -> None:
    """Inputs: volatile path and ML service policies; outputs: 99% has more cushion."""
    preds, actuals, prices, products = _frames()
    preds["y_true"] = [18, 2] * 7
    preds["y_pred"] = [10.0] * 14

    result = backtest_inventory(
        preds,
        actuals,
        prices,
        {"ML_95": policy_ml_95_service(lead_time=2), "ML_99": policy_ml_99_service(lead_time=2)},
        {"products_df": products, "initial_stock_days": 0, "lead_time_days": 2},
    )["summary"].set_index("policy")

    assert result.loc["ML_99", "total_holding_cost"] >= result.loc["ML_95", "total_holding_cost"]
    assert (
        result.loc["ML_99", "total_stockout_units"]
        <= result.loc["ML_95", "total_stockout_units"]
    )
