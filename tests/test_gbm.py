"""Tests for LightGBM and XGBoost forecaster wrappers."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from retail_demand.models.gbm import LightGBMForecaster, XGBoostForecaster, _xgb_wape_eval


def _install_fake_lightgbm(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        __import__("lightgbm")
        return
    except ModuleNotFoundError:
        pass

    class Dataset:
        def __init__(self, data: pd.DataFrame, label: pd.Series, **_kwargs: Any) -> None:
            self.data = data
            self.label = label

        def get_label(self) -> pd.Series:
            return self.label

    class Model:
        best_iteration = 1

        def __init__(self, train: Dataset) -> None:
            self.train = train

        def predict(self, X: pd.DataFrame, **_kwargs: Any) -> np.ndarray:
            return np.full(len(X), float(np.mean(self.train.label)))

        def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
            return np.ones(len(self.train.data.columns))

    fake = type(
        "FakeLightGBM",
        (),
        {
            "Dataset": Dataset,
            "train": staticmethod(lambda _params, train, **_kwargs: Model(train)),
            "early_stopping": staticmethod(lambda *_args, **_kwargs: None),
        },
    )
    monkeypatch.setitem(sys.modules, "lightgbm", fake)


def _install_fake_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        __import__("xgboost")
        return
    except ModuleNotFoundError:
        pass

    class Booster:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns

        def get_score(self, importance_type: str = "gain") -> dict[str, float]:
            return {col: 1.0 for col in self.columns}

    class XGBRegressor:
        def __init__(self, **_kwargs: Any) -> None:
            self.columns: list[str] = []
            self.mean = 0.0

        def fit(self, X: pd.DataFrame, y: pd.Series, **_kwargs: Any) -> XGBRegressor:
            self.columns = list(X.columns)
            self.mean = float(np.mean(y))
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), self.mean)

        def get_booster(self) -> Booster:
            return Booster(self.columns)

    fake = type("FakeXGBoost", (), {"XGBRegressor": XGBRegressor})
    monkeypatch.setitem(sys.modules, "xgboost", fake)


def _panel() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rows = []
    for store in range(2):
        for sku in range(5):
            for day in range(60):
                rows.append(
                    {
                        "store_id": f"store_{store:04d}",
                        "sku_id": f"sku_{sku:05d}",
                        "date_idx": day,
                        "lag_1_units": float(day % 7 + sku),
                        "rolling_mean_28": float(day % 11 + store),
                        "is_weekend": day % 7 in (5, 6),
                        "label_units_sold": float(max(0, day % 7 + sku + store)),
                    }
                )
    frame = pd.DataFrame(rows)
    for col in ("store_id", "sku_id"):
        frame[col] = frame[col].astype("category")
    train = frame[frame["date_idx"] < 45]
    val = frame[frame["date_idx"] >= 45]
    features = ["store_id", "sku_id", "date_idx", "lag_1_units", "rolling_mean_28", "is_weekend"]
    return train[features], train["label_units_sold"], val[features], val["label_units_sold"]


def test_xgb_wape_eval_accepts_dmatrix_or_label_array() -> None:
    """XGBoost 1.x passes DMatrix-like data; 2.1+ may pass the label array directly."""

    class Labels:
        def get_label(self) -> np.ndarray:
            return np.array([2.0, 4.0, 8.0])

    preds = np.array([1.0, 5.0, 7.0])

    dmatrix_result = _xgb_wape_eval(preds, Labels())
    array_result = _xgb_wape_eval(preds, np.array([2.0, 4.0, 8.0]))

    assert dmatrix_result[0] == "wape"
    assert array_result[0] == "wape"
    assert isinstance(dmatrix_result[1], float)
    assert isinstance(array_result[1], float)
    assert dmatrix_result == array_result


def test_lightgbm_trains_and_predicts_tiny_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """LightGBM handles categorical columns and returns finite predictions."""
    _install_fake_lightgbm(monkeypatch)
    X_train, y_train, X_val, y_val = _panel()

    model = LightGBMForecaster(
        params={"n_estimators": 20, "min_child_samples": 1, "num_leaves": 7},
        categorical_features=["store_id", "sku_id"],
        early_stopping_rounds=5,
    ).fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)

    assert preds.shape == (len(X_val),)
    assert np.isfinite(preds).all()
    assert not model.feature_importances_.empty


def test_xgboost_trains_and_predicts_tiny_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """XGBoost handles categorical columns and returns finite predictions."""
    _install_fake_xgboost(monkeypatch)
    X_train, y_train, X_val, y_val = _panel()

    model = XGBoostForecaster(
        params={"n_estimators": 20, "min_child_weight": 1, "max_depth": 3},
        categorical_features=["store_id", "sku_id"],
        early_stopping_rounds=5,
    ).fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)

    assert preds.shape == (len(X_val),)
    assert np.isfinite(preds).all()
    assert not model.feature_importances_.empty
