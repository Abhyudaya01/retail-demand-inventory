"""Time-based validation split helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pyspark.sql import DataFrame


def _to_timestamp_index(dates: Any) -> pd.DatetimeIndex:
    if isinstance(dates, DataFrame):
        values = [row[0] for row in dates.select("date").distinct().collect()]
        return pd.DatetimeIndex(pd.to_datetime(values, unit="ns")).sort_values()
    if isinstance(dates, pd.Series):
        return pd.DatetimeIndex(pd.to_datetime(dates.drop_duplicates())).sort_values()
    return pd.DatetimeIndex(pd.to_datetime(pd.Index(dates).drop_duplicates())).sort_values()


def walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex | DataFrame,
    n_folds: int,
    horizon_days: int,
    min_train_days: int,
) -> list[dict[str, int | pd.Timestamp]]:
    """Return expanding-window, contiguous validation folds as pandas Timestamps."""
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if min_train_days <= 0:
        raise ValueError("min_train_days must be positive")

    unique_dates = _to_timestamp_index(dates)
    required = min_train_days + n_folds * horizon_days
    if len(unique_dates) < required:
        raise ValueError("not enough dates for requested walk-forward split")

    first_val_pos = len(unique_dates) - n_folds * horizon_days
    if first_val_pos < min_train_days:
        raise ValueError("not enough training history before first validation fold")

    folds = []
    for fold_id in range(n_folds):
        val_start_pos = first_val_pos + fold_id * horizon_days
        val_end_pos = val_start_pos + horizon_days - 1
        folds.append(
            {
                "fold_id": fold_id + 1,
                "train_start": pd.Timestamp(unique_dates[0]),
                "train_end": pd.Timestamp(unique_dates[val_start_pos - 1]),
                "val_start": pd.Timestamp(unique_dates[val_start_pos]),
                "val_end": pd.Timestamp(unique_dates[val_end_pos]),
            }
        )
    return folds


def last_n_days_holdout(
    dates: pd.Series | pd.DatetimeIndex | DataFrame, holdout_days: int
) -> dict[str, pd.Timestamp]:
    """Return a single expanding train and final holdout split."""
    if holdout_days <= 0:
        raise ValueError("holdout_days must be positive")
    unique_dates = _to_timestamp_index(dates)
    if len(unique_dates) <= holdout_days:
        raise ValueError("holdout_days leaves no training rows")
    val_start_pos = len(unique_dates) - holdout_days
    return {
        "train_start": pd.Timestamp(unique_dates[0]),
        "train_end": pd.Timestamp(unique_dates[val_start_pos - 1]),
        "val_start": pd.Timestamp(unique_dates[val_start_pos]),
        "val_end": pd.Timestamp(unique_dates[-1]),
    }

