"""Tests for daily lost-sales inventory simulation."""

from __future__ import annotations

import pandas as pd

from retail_demand.inventory.policies import policy_ml_95_service
from retail_demand.inventory.simulation import InventorySimulator, simulate_sku


def test_no_reorders_stockouts_start_after_inventory_depletes() -> None:
    """Inputs: 30 days of demand and no orders; outputs: day-11 stockout; side effects: none."""
    simulator = InventorySimulator(
        initial_stock=100,
        lead_time_days=7,
        holding_cost_per_unit_day=0.1,
        stockout_cost_per_unit=2.0,
    )
    rows = [simulator.step(day, 10, {}, 0) for day in range(30)]
    frame = pd.DataFrame(rows)

    assert frame.loc[9, "stockout_units"] == 0
    assert frame.loc[10, "stockout_units"] == 10
    assert frame["stockout_units"].sum() == 200


def test_reorders_at_rop_prevent_stockouts() -> None:
    """Inputs: reorder rule at ROP 50; outputs: no stockouts; side effects: none."""
    simulator = InventorySimulator(
        initial_stock=100,
        lead_time_days=3,
        holding_cost_per_unit_day=0.1,
        stockout_cost_per_unit=2.0,
    )
    rows = []
    for day in range(30):
        reorder_qty = 80 if simulator.on_hand + sum(simulator.pipeline.values()) <= 50 else 0
        rows.append(simulator.step(day, 10, {}, reorder_qty))
    frame = pd.DataFrame(rows)

    assert frame["stockout_units"].sum() == 0


def test_cost_accounting_matches_daily_holding_and_stockout_units() -> None:
    """Inputs: simulated costs; outputs: accounting identities; side effects: none."""
    simulator = InventorySimulator(
        initial_stock=20,
        lead_time_days=7,
        holding_cost_per_unit_day=0.5,
        stockout_cost_per_unit=3.0,
    )
    frame = pd.DataFrame([simulator.step(day, 10, {}, 0) for day in range(3)])

    assert frame["holding_cost"].sum() == 0.5 * frame["on_hand"].sum()
    assert frame["stockout_cost"].sum() == 3.0 * frame["stockout_units"].sum()


def test_simulate_sku_with_ml_reorders_has_no_stockouts() -> None:
    """Inputs: simple SKU path and ML policy; outputs: simulation rows; side effects: none."""
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "y_true": [10] * 10,
            "y_pred": [10.0] * 10,
            "forecast_error_std": [0.0] * 10,
        }
    )
    result = simulate_sku(
        history,
        policy_ml_95_service(lead_time=2),
        initial_stock=100,
        lead_time=2,
        h_cost=0.1,
        s_cost=5.0,
    )

    assert len(result) == 10
    assert result["stockout_units"].sum() == 0


def test_ml_policy_uses_forward_lead_time_forecast_mean() -> None:
    """Inputs: rising predictions; outputs: ML sees the forward lead-time mean."""
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5),
            "y_true": [1, 1, 1, 1, 1],
            "y_pred": [1.0, 10.0, 10.0, 10.0, 10.0],
            "forecast_error_std": [0.0] * 5,
        }
    )
    result = simulate_sku(
        history,
        policy_ml_95_service(lead_time=3),
        initial_stock=0,
        lead_time=3,
        h_cost=0.0,
        s_cost=1.0,
    )

    assert result.loc[0, "mean_forecast"] == 7.0
