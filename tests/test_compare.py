"""Tests for model comparison statistics."""

from __future__ import annotations

import numpy as np

from retail_demand.eval.compare import bootstrap_significance


def test_bootstrap_identical_predictions_contains_zero() -> None:
    """Identical predictions have a zero-centered WAPE difference interval."""
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([11.0, 19.0, 31.0, 39.0])

    result = bootstrap_significance(actual, pred, pred, n_bootstrap=200)

    assert result["ci_lower_95"] <= 0.0 <= result["ci_upper_95"]
    assert result["mean_diff"] == 0.0


def test_bootstrap_detects_model_a_better_than_b() -> None:
    """A consistently better model has a small probability of being worse or tied."""
    actual = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    pred_a = actual.copy()
    pred_b = actual + 10.0

    result = bootstrap_significance(actual, pred_a, pred_b, n_bootstrap=200)

    assert result["mean_diff"] < 0.0
    assert result["p_value_a_better_than_b"] < 0.05
