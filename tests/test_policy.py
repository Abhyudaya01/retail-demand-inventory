"""Tests for textbook inventory policy formulas."""

from __future__ import annotations

import math

import pytest

from retail_demand.inventory.policy import (
    SERVICE_LEVELS,
    order_quantity_eoq,
    order_up_to_level,
    reorder_point,
    safety_stock,
)


def test_service_level_z_scores() -> None:
    """Inputs: service-level table; outputs: z-score assertions; side effects: none."""
    assert SERVICE_LEVELS["90%"] == pytest.approx(1.282, abs=0.001)
    assert SERVICE_LEVELS["95%"] == pytest.approx(1.645, abs=0.001)
    assert SERVICE_LEVELS["97.5%"] == pytest.approx(1.960, abs=0.001)
    assert SERVICE_LEVELS["99%"] == pytest.approx(2.326, abs=0.001)


def test_safety_stock_and_reorder_point_textbook_examples() -> None:
    """Inputs: textbook values; outputs: formula results; side effects: none."""
    ss = safety_stock(forecast_error_std=4.0, z_score=1.645, lead_time_days=9)

    assert ss == pytest.approx(19.74)
    assert reorder_point(10.0, 7, ss) == pytest.approx(89.74)


def test_eoq_and_order_up_to_examples() -> None:
    """Inputs: textbook EOQ values; outputs: order quantity assertions; side effects: none."""
    result = order_quantity_eoq(annual_demand=10_000, order_cost=50, holding_cost_per_unit_year=2)
    assert result == pytest.approx(math.sqrt(500_000))
    assert order_quantity_eoq(annual_demand=0, order_cost=50, holding_cost_per_unit_year=2) == 0.0
    assert order_up_to_level(rop=80.0, target_days_of_stock=14, mean_daily_demand=10.0) == 220.0
