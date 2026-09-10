"""Policy factories that convert forecasts and inventory position into reorder quantities."""

from __future__ import annotations

import math
from collections.abc import Callable

from retail_demand.inventory.policy import reorder_point, safety_stock

PolicyFn = Callable[[int, float, float, int, int], int]


def _clip_order(order_up_to: float, on_hand: int, on_order: int) -> int:
    return max(0, int(math.ceil(order_up_to - (on_hand + on_order))))


def policy_naive_rolling_mean(window: int = 7, target_days: int = 14) -> PolicyFn:
    """Return a policy using prior rolling-mean demand as its mean forecast.

    The simulator supplies the rolling mean computed from history available before day t.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    def policy(
        day: int, mean_forecast: float, std_forecast: float, on_hand: int, on_order: int
    ) -> int:
        del day, std_forecast
        return _clip_order(target_days * mean_forecast, on_hand, on_order)

    policy.policy_kind = "rolling_mean"  # type: ignore[attr-defined]
    policy.window = window  # type: ignore[attr-defined]
    return policy


def policy_seasonal_naive(target_days: int = 14) -> PolicyFn:
    """Return a policy using last week's same-day demand as its mean forecast."""

    def policy(
        day: int, mean_forecast: float, std_forecast: float, on_hand: int, on_order: int
    ) -> int:
        del day, std_forecast
        return _clip_order(target_days * mean_forecast, on_hand, on_order)

    policy.policy_kind = "seasonal_naive"  # type: ignore[attr-defined]
    policy.window = 7  # type: ignore[attr-defined]
    return policy


def policy_ml_95_service(
    lead_time: int, z: float = 1.645, target_days: int = 14
) -> PolicyFn:
    """Return an ML reorder-up-to policy with safety stock for the chosen service z-score."""
    if lead_time <= 0:
        raise ValueError("lead_time must be positive")

    def policy(
        day: int, mean_forecast: float, std_forecast: float, on_hand: int, on_order: int
    ) -> int:
        del day
        ss = safety_stock(std_forecast, z, lead_time)
        rop = reorder_point(mean_forecast, lead_time, ss)
        inventory_position = on_hand + on_order
        if inventory_position < rop:
            return _clip_order(rop + target_days * mean_forecast, on_hand, on_order)
        return 0

    policy.policy_kind = "ml"  # type: ignore[attr-defined]
    policy.z = z  # type: ignore[attr-defined]
    return policy


def policy_ml_99_service(lead_time: int, target_days: int = 14) -> PolicyFn:
    """Return an ML reorder-up-to policy with 99% service-level safety stock."""
    return policy_ml_95_service(lead_time=lead_time, z=2.326, target_days=target_days)
