"""Daily lost-sales inventory simulation for Phase 9 policy backtests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


class InventorySimulator:
    """Lost-sales daily inventory simulator with fixed lead-time replenishment."""

    def __init__(
        self,
        initial_stock: int,
        lead_time_days: int,
        holding_cost_per_unit_day: float,
        stockout_cost_per_unit: float,
    ) -> None:
        """Inputs: starting inventory and USD cost rates; outputs: simulator; side effects: none."""
        if lead_time_days <= 0:
            raise ValueError("lead_time_days must be positive")
        self.on_hand = int(initial_stock)
        self.lead_time_days = lead_time_days
        self.holding_cost_per_unit_day = float(holding_cost_per_unit_day)
        self.stockout_cost_per_unit = float(stockout_cost_per_unit)
        self.pipeline: dict[int, int] = {}

    def step(
        self,
        day: int,
        demand_actual: int,
        order_placed_days_ago: dict[int, int],
        reorder_qty: int,
    ) -> dict[str, float | int]:
        """Advance one day, receiving elapsed orders, serving demand, and placing a new order."""
        for placed_day, qty in order_placed_days_ago.items():
            arrival_day = placed_day + self.lead_time_days
            self.pipeline[arrival_day] = self.pipeline.get(arrival_day, 0) + int(qty)

        received_qty = self.pipeline.pop(day, 0)
        self.on_hand += received_qty
        demand = max(0, int(demand_actual))
        served_units = min(self.on_hand, demand)
        stockout_units = max(0, demand - self.on_hand)
        self.on_hand -= served_units

        placed_qty = max(0, int(reorder_qty))
        if placed_qty > 0:
            arrival_day = day + self.lead_time_days
            self.pipeline[arrival_day] = self.pipeline.get(arrival_day, 0) + placed_qty

        holding_cost = self.on_hand * self.holding_cost_per_unit_day
        stockout_cost = stockout_units * self.stockout_cost_per_unit
        return {
            "day": day,
            "received_qty": received_qty,
            "on_hand": self.on_hand,
            "on_order": sum(self.pipeline.values()),
            "served_units": served_units,
            "stockout_units": stockout_units,
            "holding_cost": float(holding_cost),
            "stockout_cost": float(stockout_cost),
            "reorder_qty": placed_qty,
        }


def _rolling_mean(history: list[float], window: int) -> float:
    if not history:
        return 0.0
    return float(np.mean(history[-window:]))


def _mean_for_policy(
    policy_fn: Callable[..., int],
    frame: pd.DataFrame,
    idx: int,
    history: list[float],
    lead_time: int,
) -> float:
    policy_kind = getattr(policy_fn, "policy_kind", "ml")
    window = int(getattr(policy_fn, "window", 7))
    if policy_kind == "rolling_mean":
        return _rolling_mean(history, window)
    if policy_kind == "seasonal_naive":
        return float(history[-7]) if len(history) >= 7 else _rolling_mean(history, window)

    forecast_col = "y_pred" if "y_pred" in frame.columns else "mean_forecast"
    horizon = frame.iloc[idx : idx + lead_time]
    assert (
        horizon["date"] >= frame.loc[idx, "date"]
    ).all(), "ML policy cannot use predictions before the current day"
    return float(horizon[forecast_col].astype(float).mean())


def simulate_sku(
    sku_history: pd.DataFrame,
    policy_fn: Callable[[int, float, float, int, int], int],
    initial_stock: int,
    lead_time: int,
    h_cost: float,
    s_cost: float,
) -> pd.DataFrame:
    """Run one store/SKU through a no-backorder daily inventory simulation.

    Inputs are one validation-period history with y_true/y_pred columns plus optional
    forecast_error_std. Outputs a daily simulation DataFrame. Side effects: none.
    """
    frame = sku_history.sort_values("date").reset_index(drop=True).copy()
    assert frame["date"].is_monotonic_increasing, "sku_history must be sorted by date"
    simulator = InventorySimulator(initial_stock, lead_time, h_cost, s_cost)
    rows: list[dict[str, Any]] = []
    demand_history: list[float] = list(frame.attrs.get("initial_demand_history", []))
    for idx, row in frame.iterrows():
        assert len(demand_history) >= idx, "policy cannot see current or future actual demand"
        mean_forecast = _mean_for_policy(policy_fn, frame, int(idx), demand_history, lead_time)
        std_forecast = float(row.get("forecast_error_std", 0.0))
        on_order = sum(simulator.pipeline.values())
        reorder_qty = policy_fn(
            int(idx), mean_forecast, std_forecast, simulator.on_hand, on_order
        )
        result = simulator.step(
            day=int(idx),
            demand_actual=int(round(float(row["y_true"]))),
            order_placed_days_ago={},
            reorder_qty=reorder_qty,
        )
        rows.append(
            {
                "date": row["date"],
                "y_true": float(row["y_true"]),
                "y_pred": float(row.get("y_pred", mean_forecast)),
                "mean_forecast": mean_forecast,
                "forecast_error_std": std_forecast,
                **result,
            }
        )
        demand_history.append(float(row["y_true"]))
    return pd.DataFrame(rows)


def compare_policies(
    sku_history: pd.DataFrame, policies: dict[str, Callable[..., int]], **sim_kwargs: Any
) -> pd.DataFrame:
    """Run several policies on the same demand path and return per-policy totals."""
    rows = []
    for policy_name, policy_fn in policies.items():
        sim = simulate_sku(sku_history, policy_fn, **sim_kwargs)
        total_demand = float(sim["y_true"].sum())
        served_units = total_demand - float(sim["stockout_units"].sum())
        rows.append(
            {
                "policy": policy_name,
                "total_cost": float(sim["holding_cost"].sum() + sim["stockout_cost"].sum()),
                "service_level": 1.0 if total_demand == 0.0 else served_units / total_demand,
                "avg_inventory": float(sim["on_hand"].mean()),
                "num_stockout_events": int((sim["stockout_units"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)
