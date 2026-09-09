"""Spark-native baseline forecasters over Gold features_master rows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class BaselineForecaster(ABC):
    """Base interface for simple Spark-native forecasting baselines."""

    name = "baseline"

    def __init__(self) -> None:
        self.global_mean_: float = 0.0

    def fit(self, train_df: DataFrame) -> BaselineForecaster:
        """Store fallback training statistics without collecting training rows."""
        value = train_df.agg(F.avg("label_units_sold").alias("mean_units")).first()["mean_units"]
        self.global_mean_ = float(value or 0.0)
        return self

    @abstractmethod
    def _prediction_col(self, future_df: DataFrame) -> Any:
        """Return a Spark Column containing the baseline prediction."""

    def predict(self, future_df: DataFrame) -> DataFrame:
        """Return a Spark DataFrame with store_id, sku_id, date and y_pred."""
        return (
            future_df.select(
                "store_id",
                "sku_id",
                "date",
                F.coalesce(self._prediction_col(future_df), F.lit(self.global_mean_))
                .cast("double")
                .alias("y_pred"),
            )
            .orderBy("store_id", "sku_id", "date")
        )


class NaiveForecaster(BaselineForecaster):
    """Predict demand at t with observed demand at t-1."""

    name = "naive"

    def _prediction_col(self, future_df: DataFrame) -> Any:
        return F.col("lag_1_units")


class SeasonalNaiveForecaster(BaselineForecaster):
    """Predict demand at t with observed demand at t-7."""

    name = "seasonal_naive"

    def _prediction_col(self, future_df: DataFrame) -> Any:
        return F.col("lag_7_units")


class MovingAverageForecaster(BaselineForecaster):
    """Predict demand at t with the lagged rolling mean over the prior window."""

    name = "moving_average"

    def __init__(self, window: int = 28) -> None:
        super().__init__()
        self.window = window

    def _prediction_col(self, future_df: DataFrame) -> Any:
        return F.col(f"rolling_mean_{self.window}")
