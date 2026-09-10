"""Inventory control formulas for reorder decisions."""

from __future__ import annotations

import math

SERVICE_LEVELS = {"90%": 1.282, "95%": 1.645, "97.5%": 1.960, "99%": 2.326}


def safety_stock(forecast_error_std: float, z_score: float, lead_time_days: int) -> float:
    """Return safety stock, SS = z * forecast_error_std * sqrt(L).

    Assumes forecast_error_std is the horizon-level daily forecast error standard deviation,
    demand errors are independent enough to scale by sqrt(lead time), and lead time is fixed.
    """
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be positive")
    return float(z_score * forecast_error_std * math.sqrt(lead_time_days))


def reorder_point(
    mean_daily_demand: float, lead_time_days: int, safety_stock_units: float
) -> float:
    """Return reorder point, ROP = mean_daily_demand * L + safety_stock.

    Assumes mean demand and safety stock are expressed in units for the same store/SKU and that
    incoming supply arrives after a fixed lead time.
    """
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be positive")
    return float(mean_daily_demand * lead_time_days + safety_stock_units)


def order_quantity_eoq(
    annual_demand: float, order_cost: float, holding_cost_per_unit_year: float
) -> float:
    """Return economic order quantity, EOQ = sqrt(2DS / H).

    Assumes annual_demand is units/year, order_cost is USD/order, and holding_cost_per_unit_year
    is USD/unit/year. Returns 0 when annual demand is zero or negative.
    """
    if annual_demand <= 0.0:
        return 0.0
    if order_cost < 0.0:
        raise ValueError("order_cost must be non-negative")
    if holding_cost_per_unit_year <= 0.0:
        raise ValueError("holding_cost_per_unit_year must be positive")
    return float(math.sqrt(2.0 * annual_demand * order_cost / holding_cost_per_unit_year))


def order_up_to_level(
    rop: float, target_days_of_stock: int, mean_daily_demand: float
) -> float:
    """Return periodic-review order-up-to level, ROP + target_days_of_stock * demand.

    Assumes target_days_of_stock is the cycle stock cushion beyond the reorder point and mean
    demand is stable over that target-stock window.
    """
    if target_days_of_stock < 0:
        raise ValueError("target_days_of_stock must be non-negative")
    return float(rop + target_days_of_stock * mean_daily_demand)

