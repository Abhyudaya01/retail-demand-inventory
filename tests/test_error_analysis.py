"""Tests for Phase 8 error-analysis metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retail_demand.analysis.error_analysis import (
    best_and_worst_skus,
    bias_flags,
    per_segment_metrics,
)


def test_per_segment_metrics_matches_hand_computed_values() -> None:
    """Inputs: 20-row toy predictions; outputs: hand-computed metrics; side effects: none."""
    frame = pd.DataFrame(
        {
            "segment": ["A"] * 10 + ["B"] * 10,
            "y_true": [10.0] * 10 + [20.0] * 10,
            "y_pred": [12.0] * 10 + [15.0] * 10,
        }
    )

    result = per_segment_metrics(frame, ["segment"]).set_index("segment")

    assert result.loc["A", "n_rows"] == 10
    assert result.loc["A", "mae"] == 2.0
    assert result.loc["A", "rmse"] == 2.0
    assert result.loc["A", "wape"] == 0.2
    assert np.isclose(result.loc["A", "smape"], 2.0 / 11.0)
    assert result.loc["A", "bias"] == 2.0
    assert result.loc["B", "bias"] == -5.0


def test_bias_flags_identifies_systemic_over_forecast() -> None:
    """Inputs: segment metrics; outputs: over-forecast bias flag; side effects: none."""
    segments = pd.DataFrame(
        {
            "segment": ["over", "clean", "zero"],
            "mean_actual": [10.0, 10.0, 0.0],
            "bias": [3.0, 0.5, 2.0],
            "wape": [0.3, 0.05, 1.0],
        }
    )

    flagged = bias_flags(segments, threshold=0.10)

    assert flagged["segment"].tolist() == ["over"]
    assert flagged.iloc[0]["bias_direction"] == "over"
    assert flagged.iloc[0]["bias_ratio"] == 0.3


def test_best_and_worst_skus_returns_disjoint_sets_of_correct_size() -> None:
    """Inputs: SKU-level predictions; outputs: disjoint best/worst tables; side effects: none."""
    rows = []
    for sku_idx in range(6):
        for day in range(5):
            rows.append(
                {
                    "store_id": "store_0",
                    "sku_id": f"sku_{sku_idx}",
                    "category": "cat",
                    "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "y_true": 10.0,
                    "y_pred": 10.0 + sku_idx,
                }
            )
    frame = pd.DataFrame(rows)

    best, worst = best_and_worst_skus(frame, n=2)

    assert len(best) == 2
    assert len(worst) == 2
    assert set(best["sku_id"]).isdisjoint(set(worst["sku_id"]))
    assert best["sku_id"].tolist() == ["sku_0", "sku_1"]
    assert worst["sku_id"].tolist() == ["sku_5", "sku_4"]


def test_best_and_worst_skus_excludes_low_actual_wape_artifacts() -> None:
    """Inputs: low-demand and normal-demand SKUs; outputs: sane WAPE ranking; side effects: none."""
    frame = pd.DataFrame(
        {
            "store_id": ["store_0"] * 12,
            "sku_id": ["SKU_A"] * 6 + ["SKU_B"] * 6,
            "category": ["intermittent"] * 6 + ["steady"] * 6,
            "date": pd.date_range("2026-01-01", periods=6).tolist() * 2,
            "y_true": [0, 0, 0, 0, 0, 1] + [10, 10, 10, 10, 10, 10],
            "y_pred": [5, 5, 5, 5, 5, 5] + [15, 15, 15, 15, 15, 15],
        }
    )

    _best, worst = best_and_worst_skus(frame, n=1, min_total_actual=10)

    assert "SKU_A" not in worst["sku_id"].tolist()
    assert worst.iloc[0]["sku_id"] == "SKU_B"
    assert worst.iloc[0]["wape"] == 0.5
