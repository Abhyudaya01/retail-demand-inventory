"""Tests for Phase 8 segment assignment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_demand.analysis.segments import (
    assign_intermittency_class,
    assign_promo_flag,
    assign_volume_tier,
)


def test_assign_volume_tier_uses_pareto_volume_share() -> None:
    """Inputs: tiny history; outputs: deterministic A/B/C volume tiers; side effects: none."""
    totals = [20, 15, 15, 10, 10, 10, 5, 5, 5, 5]
    rows = []
    for idx, total in enumerate(totals):
        for day in range(30):
            rows.append(
                {
                    "store_id": f"store_{idx // 5}",
                    "sku_id": f"sku_{idx % 5}",
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "units_sold": total / 30,
                }
            )

    tiers = assign_volume_tier(pd.DataFrame(rows))

    assert tiers["volume_tier"].tolist() == [
        "A_top20pct",
        "B_middle30pct",
        "B_middle30pct",
        "C_bottom50pct",
        "C_bottom50pct",
        "C_bottom50pct",
        "C_bottom50pct",
        "C_bottom50pct",
        "C_bottom50pct",
        "C_bottom50pct",
    ]


def test_assign_intermittency_class_sbc_examples() -> None:
    """Inputs: hand-crafted demand patterns; outputs: SBC classes; side effects: none."""
    rows = []
    patterns = {
        "smooth": [9, 11, 10, 10, 9, 11],
        "intermittent": [0, 0, 10, 0, 0, 10],
        "lumpy": [0, 0, 2, 0, 0, 20],
    }
    for sku_id, units in patterns.items():
        for idx, value in enumerate(units):
            rows.append(
                {
                    "store_id": "store_0",
                    "sku_id": sku_id,
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx),
                    "units_sold": value,
                }
            )

    classes = assign_intermittency_class(pd.DataFrame(rows)).set_index("sku_id")

    assert classes.loc["smooth", "intermittency_class"] == "smooth"
    assert classes.loc["intermittent", "intermittency_class"] == "intermittent"
    assert classes.loc["lumpy", "intermittency_class"] == "lumpy"
    assert np.isclose(classes.loc["smooth", "adi"], 1.0)
    assert np.isclose(classes.loc["intermittent", "adi"], 3.0)
    assert classes.loc["lumpy", "cv_squared"] >= 0.49


def test_assign_promo_flag_joins_on_week_start_boundaries() -> None:
    """Inputs: predictions and weekly prices; outputs: day-level promo flags; side effects: none."""
    preds = pd.DataFrame(
        {
            "store_id": ["store_0", "store_0", "store_0"],
            "sku_id": ["sku_0", "sku_0", "sku_0"],
            "date": pd.to_datetime(["2026-01-04", "2026-01-05", "2026-01-11"]),
            "y_true": [1, 2, 3],
            "y_pred": [1, 2, 3],
        }
    )
    prices = pd.DataFrame(
        {
            "store_id": ["store_0", "store_0"],
            "sku_id": ["sku_0", "sku_0"],
            "week_start": pd.to_datetime(["2025-12-29", "2026-01-05"]),
            "is_promo": [False, True],
        }
    )

    flagged = assign_promo_flag(preds, prices)

    assert flagged["is_promo_day"].tolist() == [False, True, True]
