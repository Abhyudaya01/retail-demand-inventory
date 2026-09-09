"""Known-answer tests for forecast metrics."""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.sql import SparkSession

from retail_demand.eval.metrics import bias, mae, per_segment_metrics, rmse, smape, wape


def test_metrics_known_answers() -> None:
    """Tiny arrays return hand-computable metric values."""
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])

    assert mae(actual, predicted) == pytest.approx(7.0 / 3.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(17.0 / 3.0))
    assert wape(actual, predicted) == pytest.approx(7.0 / 60.0)
    assert bias(actual, predicted) == pytest.approx(3.0 / 60.0)
    expected_smape = np.mean([2 / 11, 2 / 19, 3 / 31.5])
    assert smape(actual, predicted) == pytest.approx(expected_smape)


def test_metrics_zero_actuals_are_guarded() -> None:
    """Zero-demand periods do not divide by zero."""
    actual = np.array([0.0, 0.0])
    predicted = np.array([0.0, 5.0])

    assert np.isfinite(wape(actual, predicted))
    assert np.isfinite(smape(actual, predicted))
    assert np.isfinite(bias(actual, predicted))
    assert mae(actual, predicted) == pytest.approx(2.5)


def test_spark_dataframe_metrics_known_answers(spark: SparkSession) -> None:
    """Spark DataFrame inputs produce the same metric values as array inputs."""
    frame = spark.createDataFrame(
        [(10.0, 12.0), (20.0, 18.0), (30.0, 33.0)],
        "y_true double, y_pred double",
    )

    assert mae(frame) == pytest.approx(7.0 / 3.0)
    assert rmse(frame) == pytest.approx(np.sqrt(17.0 / 3.0))
    assert wape(frame) == pytest.approx(7.0 / 60.0)
    assert bias(frame) == pytest.approx(3.0 / 60.0)


def test_per_segment_metrics_returns_one_row_per_segment() -> None:
    """Segment metrics aggregate forecasts by the requested dimensions."""
    frame = [
        {"sku_id": "sku_a", "y_true": 10.0, "y_pred": 12.0},
        {"sku_id": "sku_a", "y_true": 20.0, "y_pred": 18.0},
        {"sku_id": "sku_b", "y_true": 5.0, "y_pred": 7.0},
    ]

    result = per_segment_metrics(frame, ["sku_id"])

    assert set(result["sku_id"]) == {"sku_a", "sku_b"}
    assert result.loc[result["sku_id"] == "sku_a", "mae"].iloc[0] == pytest.approx(2.0)
