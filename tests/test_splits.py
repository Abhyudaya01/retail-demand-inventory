"""Tests for time-based validation splits."""

from __future__ import annotations

import pandas as pd
from pyspark.sql import SparkSession

from retail_demand.eval.splits import last_n_days_holdout, walk_forward_splits


def test_walk_forward_splits_are_expanding_and_non_overlapping() -> None:
    """A 500-day range supports 3 contiguous 28-day validation folds."""
    dates = pd.date_range("2025-01-01", periods=500, freq="D")

    folds = walk_forward_splits(dates, n_folds=3, horizon_days=28, min_train_days=365)

    assert len(folds) == 3
    train_ends = [fold["train_end"] for fold in folds]
    assert train_ends == sorted(train_ends)
    for fold in folds:
        assert (fold["val_end"] - fold["val_start"]).days + 1 == 28
        assert fold["train_end"] < fold["val_start"]
    assert folds[0]["val_end"] < folds[1]["val_start"] < folds[1]["val_end"]
    assert folds[1]["val_end"] < folds[2]["val_start"] < folds[2]["val_end"]


def test_last_n_days_holdout() -> None:
    """The holdout split uses the final n dates for validation."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")

    split = last_n_days_holdout(dates, holdout_days=7)

    assert split["train_end"] == pd.Timestamp("2026-02-02")
    assert split["val_start"] == pd.Timestamp("2026-02-03")
    assert split["val_end"] == pd.Timestamp("2026-02-09")


def test_walk_forward_splits_accept_spark_dataframe(spark: SparkSession) -> None:
    """Spark date columns are collected as distinct validation dates only."""
    rows = [(int(day.value),) for day in pd.date_range("2025-01-01", periods=500, freq="D")]
    frame = spark.createDataFrame(rows, "date long")

    folds = walk_forward_splits(frame, n_folds=2, horizon_days=14, min_train_days=365)

    assert len(folds) == 2
    assert folds[0]["train_end"] < folds[0]["val_start"]
    assert folds[0]["val_end"] < folds[1]["val_start"]
