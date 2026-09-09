"""Tests for Phase 7 leakage assertions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from retail_demand.experiments import run_gbm


class _NoopRun:
    def __enter__(self) -> _NoopRun:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class _FakeMlflow:
    lightgbm = SimpleNamespace(autolog=lambda **_kwargs: None)
    xgboost = SimpleNamespace(autolog=lambda **_kwargs: None)

    @staticmethod
    def start_run(**_kwargs: Any) -> _NoopRun:
        return _NoopRun()

    @staticmethod
    def set_tag(*_args: Any, **_kwargs: Any) -> None:
        return None

    @staticmethod
    def log_params(*_args: Any, **_kwargs: Any) -> None:
        return None

    @staticmethod
    def log_metrics(*_args: Any, **_kwargs: Any) -> None:
        return None

    @staticmethod
    def log_artifact(*_args: Any, **_kwargs: Any) -> None:
        return None


class _DummyModel:
    name = "dummy"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.params = params or {}
        self.categorical_features = categorical_features or []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> _DummyModel:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=float)

    @property
    def feature_importances_(self) -> pd.Series:
        return pd.Series({"lag_1_units": 1.0})


def _features() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    labels = 20.0 + np.sin(np.arange(40, dtype=float)) * 5.0
    return pd.DataFrame(
        {
            "store_id": pd.Series(["store_0001"] * 40, dtype="category"),
            "sku_id": pd.Series(["sku_00001"] * 40, dtype="category"),
            "date": dates,
            "label_units_sold": labels,
            "lag_1_units": np.arange(40, dtype=float),
            "store_type": pd.Series(["A"] * 40, dtype="category"),
        }
    )


def _patch_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "mlflow", _FakeMlflow)
    monkeypatch.setattr(run_gbm, "_log_importance_artifacts", lambda *_args, **_kwargs: None)


def test_run_cv_experiment_rejects_overlapping_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fold with train max date at or after validation min date raises before fit."""
    _patch_mlflow(monkeypatch)
    fold = {
        "fold_id": 1,
        "train_start": pd.Timestamp("2026-01-01"),
        "train_end": pd.Timestamp("2026-01-20"),
        "val_start": pd.Timestamp("2026-01-20"),
        "val_end": pd.Timestamp("2026-01-25"),
    }
    monkeypatch.setattr(run_gbm, "walk_forward_splits", lambda *_args, **_kwargs: [fold])

    with pytest.raises(AssertionError):
        run_gbm.run_cv_experiment(_features(), _DummyModel, {}, {}, {})


def test_run_cv_experiment_accepts_clean_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean fold fits and returns one metrics row."""
    _patch_mlflow(monkeypatch)
    fold = {
        "fold_id": 1,
        "train_start": pd.Timestamp("2026-01-01"),
        "train_end": pd.Timestamp("2026-01-25"),
        "val_start": pd.Timestamp("2026-01-26"),
        "val_end": pd.Timestamp("2026-02-09"),
    }
    monkeypatch.setattr(run_gbm, "walk_forward_splits", lambda *_args, **_kwargs: [fold])

    result = run_gbm.run_cv_experiment(
        _features(), _DummyModel, {}, {}, {"print_comparison": False}
    )

    assert len(result) == 1
    assert result.iloc[0]["model"] == "dummy"


def test_run_cv_experiment_rejects_feature_nan_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A validation feature that is much more complete than train is rejected."""
    _patch_mlflow(monkeypatch)
    frame = _features()
    frame["future_join_signal"] = 1.0
    frame.loc[
        (frame["date"] >= pd.Timestamp("2026-01-14"))
        & (frame["date"] <= pd.Timestamp("2026-01-25")),
        "future_join_signal",
    ] = np.nan
    fold = {
        "fold_id": 1,
        "train_start": pd.Timestamp("2026-01-01"),
        "train_end": pd.Timestamp("2026-01-25"),
        "val_start": pd.Timestamp("2026-01-26"),
        "val_end": pd.Timestamp("2026-02-09"),
    }
    monkeypatch.setattr(run_gbm, "walk_forward_splits", lambda *_args, **_kwargs: [fold])

    with pytest.raises(AssertionError):
        run_gbm.run_cv_experiment(frame, _DummyModel, {}, {}, {})
