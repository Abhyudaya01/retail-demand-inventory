"""Forecast accuracy metrics for numpy arrays, pandas objects and Spark DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

EPSILON = 1e-9


def _as_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return actual, predicted


def _spark_metric(
    df: DataFrame,
    expr: Any,
) -> float:
    value = df.agg(expr.alias("metric")).first()["metric"]
    return float(value or 0.0)


def mae(y_true: Any, y_pred: Any = "y_pred", actual_col: str = "y_true") -> float:
    """Return mean absolute error, accepting arrays or a Spark DataFrame."""
    if isinstance(y_true, DataFrame):
        return _spark_metric(y_true, F.avg(F.abs(F.col(actual_col) - F.col(str(y_pred)))))
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: Any, y_pred: Any = "y_pred", actual_col: str = "y_true") -> float:
    """Return root mean squared error, accepting arrays or a Spark DataFrame."""
    if isinstance(y_true, DataFrame):
        return _spark_metric(y_true, F.sqrt(F.avg((F.col(actual_col) - F.col(str(y_pred))) ** 2)))
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(y_true: Any, y_pred: Any = "y_pred", actual_col: str = "y_true") -> float:
    """Return weighted absolute percentage error with zero-demand protection."""
    if isinstance(y_true, DataFrame):
        num = F.sum(F.abs(F.col(actual_col) - F.col(str(y_pred))))
        den = F.greatest(F.sum(F.abs(F.col(actual_col))), F.lit(EPSILON))
        return _spark_metric(y_true, num / den)
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.sum(np.abs(actual - predicted)) / max(np.sum(np.abs(actual)), EPSILON))


def smape(y_true: Any, y_pred: Any = "y_pred", actual_col: str = "y_true") -> float:
    """Return symmetric MAPE with guarded denominator."""
    if isinstance(y_true, DataFrame):
        den = F.greatest(
            (F.abs(F.col(actual_col)) + F.abs(F.col(str(y_pred)))) / F.lit(2.0),
            F.lit(EPSILON),
        )
        return _spark_metric(y_true, F.avg(F.abs(F.col(actual_col) - F.col(str(y_pred))) / den))
    actual, predicted = _as_arrays(y_true, y_pred)
    den = np.maximum((np.abs(actual) + np.abs(predicted)) / 2.0, EPSILON)
    return float(np.mean(np.abs(actual - predicted) / den))


def bias(y_true: Any, y_pred: Any = "y_pred", actual_col: str = "y_true") -> float:
    """Return signed forecast bias as sum(predicted - actual) over guarded actual sum."""
    if isinstance(y_true, DataFrame):
        num = F.sum(F.col(str(y_pred)) - F.col(actual_col))
        den = F.greatest(F.sum(F.abs(F.col(actual_col))), F.lit(EPSILON))
        return _spark_metric(y_true, num / den)
    actual, predicted = _as_arrays(y_true, y_pred)
    return float(np.sum(predicted - actual) / max(np.sum(np.abs(actual)), EPSILON))


def metric_dict(
    y_true: Any, y_pred: Any = "y_pred", actual_col: str = "y_true"
) -> dict[str, float]:
    """Return all baseline metrics with consistent names."""
    return {
        "mae": mae(y_true, y_pred, actual_col),
        "rmse": rmse(y_true, y_pred, actual_col),
        "wape": wape(y_true, y_pred, actual_col),
        "smape": smape(y_true, y_pred, actual_col),
        "bias": bias(y_true, y_pred, actual_col),
    }


def per_segment_metrics(preds_df: Any, segment_cols: list[str]) -> pd.DataFrame:
    """Return metrics per segment from columns y_true and y_pred."""
    if isinstance(preds_df, DataFrame):
        rows = []
        for row in preds_df.select(*segment_cols).distinct().collect():
            filters = None
            details = row.asDict()
            for col_name, value in details.items():
                condition = F.col(col_name) == value
                filters = condition if filters is None else filters & condition
            segment_df = preds_df.filter(filters)
            rows.append({**details, **metric_dict(segment_df)})
        return pd.DataFrame(rows)

    frame = pd.DataFrame(preds_df)
    rows = []
    for keys, group in frame.groupby(segment_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(segment_cols, key_values, strict=True))
        row.update(metric_dict(group["y_true"].to_numpy(), group["y_pred"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)
