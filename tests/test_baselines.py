"""Tests for simple baseline forecasters."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from retail_demand.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def _feature_frame(spark: SparkSession):
    rows = []
    start = date(2026, 1, 1)
    for store in range(1, 3):
        for sku in range(1, 4):
            for offset in range(90):
                day = start + timedelta(days=offset)
                rows.append(
                    (
                        f"store_{store:04d}",
                        f"sku_{sku:05d}",
                        int(datetime(day.year, day.month, day.day).timestamp() * 1_000_000_000),
                        float(offset + store + sku),
                    )
                )
    base = spark.createDataFrame(
        rows,
        "store_id string, sku_id string, date long, label_units_sold double",
    )
    window = Window.partitionBy("store_id", "sku_id").orderBy("date")
    return (
        base.withColumn("lag_1_units", F.lag("label_units_sold", 1).over(window))
        .withColumn("lag_7_units", F.lag("label_units_sold", 7).over(window))
        .withColumn("rolling_mean_28", F.avg("label_units_sold").over(window.rowsBetween(-28, -1)))
    )


def test_naive_forecaster_predicts_t_minus_1(spark: SparkSession) -> None:
    """Naive predictions equal the precomputed t-1 feature."""
    features = _feature_frame(spark)
    train = features.filter("date < 1769835600000000000")
    future = features.filter("date >= 1769835600000000000")

    preds = NaiveForecaster().fit(train).predict(future)
    row = preds.collect()[0]
    actual_lag = future.filter(
        (F.col("store_id") == row["store_id"])
        & (F.col("sku_id") == row["sku_id"])
        & (F.col("date") == int(row["date"]))
    ).first()["lag_1_units"]

    assert preds.count() == future.count()
    assert row["y_pred"] == pytest.approx(actual_lag)


def test_seasonal_and_moving_average_forecasters_shape(spark: SparkSession) -> None:
    """Seasonal naive and moving-average baselines return one prediction per future row."""
    features = _feature_frame(spark)
    train = features.filter("date < 1769835600000000000")
    future = features.filter("date >= 1769835600000000000")

    seasonal = SeasonalNaiveForecaster().fit(train).predict(future)
    moving = MovingAverageForecaster(window=28).fit(train).predict(future)

    assert seasonal.count() == future.count()
    assert moving.count() == future.count()
    assert set(seasonal.columns) == {"store_id", "sku_id", "date", "y_pred"}
    assert set(moving.columns) == {"store_id", "sku_id", "date", "y_pred"}


def test_moving_average_forecaster_uses_lagged_rolling_mean(spark: SparkSession) -> None:
    """Moving-average predictions equal the precomputed lagged rolling mean feature."""
    features = _feature_frame(spark)
    train = features.filter("date < 1769835600000000000")
    future = features.filter("date >= 1769835600000000000")

    preds = MovingAverageForecaster(window=28).fit(train).predict(future)
    row = preds.collect()[0]
    expected = future.filter(
        (F.col("store_id") == row["store_id"])
        & (F.col("sku_id") == row["sku_id"])
        & (F.col("date") == int(row["date"]))
    ).first()["rolling_mean_28"]

    assert row["y_pred"] == pytest.approx(expected)
