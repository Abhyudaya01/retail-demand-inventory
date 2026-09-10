"""Tests for inventory reorder policy factories."""

from __future__ import annotations

from retail_demand.inventory.policies import policy_ml_95_service, policy_naive_rolling_mean


def test_policy_naive_rolling_mean_orders_to_target_days() -> None:
    """Inputs: flat rolling mean and inventory; outputs: reorder quantity; side effects: none."""
    policy = policy_naive_rolling_mean(window=7, target_days=14)

    assert policy(day=8, mean_forecast=10.0, std_forecast=0.0, on_hand=40, on_order=20) == 80


def test_policy_ml_95_service_orders_more_with_high_error_std() -> None:
    """Inputs: same forecast with different error std; outputs: higher safety-stock order."""
    policy = policy_ml_95_service(lead_time=7, target_days=14)

    low_std = policy(day=0, mean_forecast=10.0, std_forecast=1.0, on_hand=20, on_order=0)
    high_std = policy(day=0, mean_forecast=10.0, std_forecast=8.0, on_hand=20, on_order=0)

    assert high_std > low_std

