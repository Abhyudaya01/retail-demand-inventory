"""Gradient-boosted forecasters for Phase 7 demand modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "poisson",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "metric": "None",
    "verbose": -1,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "count:poisson",
    "n_estimators": 2000,
    "eta": 0.05,
    "max_depth": 8,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.1,
    "verbosity": 0,
    "enable_categorical": True,
    "tree_method": "hist",
}


def _wape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = max(float(np.sum(np.abs(y_true))), 1e-9)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _lgb_wape(y_pred: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
    return "wape", _wape_np(dataset.get_label(), y_pred), False


def _xgb_wape(y_pred: np.ndarray, data: Any) -> tuple[str, float]:
    return "wape", _wape_np(data.get_label(), y_pred)


class LightGBMForecaster:
    """LightGBM Poisson forecaster with explicit categorical feature handling."""

    name = "lightgbm"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
        self.categorical_features = categorical_features or []
        self.early_stopping_rounds = early_stopping_rounds
        self.model_: Any | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> LightGBMForecaster:
        """Fit using validation WAPE for early stopping."""
        import lightgbm as lgb

        self.feature_names_ = list(X_train.columns)
        train = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.categorical_features,
            free_raw_data=False,
        )
        valid = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=self.categorical_features,
            reference=train,
            free_raw_data=False,
        )
        params = dict(self.params)
        num_boost_round = int(params.pop("n_estimators", 2000))
        self.model_ = lgb.train(
            params,
            train,
            num_boost_round=num_boost_round,
            valid_sets=[valid],
            valid_names=["validation"],
            feval=_lgb_wape,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds)],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict non-negative demand for a pandas feature frame."""
        if self.model_ is None:
            raise RuntimeError("LightGBMForecaster must be fit before predict")
        preds = self.model_.predict(X, num_iteration=self.model_.best_iteration)
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    @property
    def feature_importances_(self) -> pd.Series:
        """Return gain-based feature importances indexed by feature name."""
        if self.model_ is None:
            return pd.Series(dtype=float)
        values = self.model_.feature_importance(importance_type="gain")
        return pd.Series(values, index=self.feature_names_, dtype=float).sort_values(
            ascending=False
        )


class XGBoostForecaster:
    """XGBoost Poisson forecaster with the same public interface as LightGBM."""

    name = "xgboost"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.params = {**DEFAULT_XGB_PARAMS, **(params or {})}
        self.categorical_features = categorical_features or []
        self.early_stopping_rounds = early_stopping_rounds
        self.model_: Any | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> XGBoostForecaster:
        """Fit using validation WAPE for early stopping."""
        import xgboost as xgb

        self.feature_names_ = list(X_train.columns)
        params = {**self.params, "early_stopping_rounds": self.early_stopping_rounds}
        self.model_ = xgb.XGBRegressor(**params, eval_metric=_xgb_wape)
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict non-negative demand for a pandas feature frame."""
        if self.model_ is None:
            raise RuntimeError("XGBoostForecaster must be fit before predict")
        preds = self.model_.predict(X)
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    @property
    def feature_importances_(self) -> pd.Series:
        """Return model feature importances indexed by feature name."""
        if self.model_ is None:
            return pd.Series(dtype=float)
        booster = self.model_.get_booster()
        scores = booster.get_score(importance_type="gain")
        values = [float(scores.get(name, 0.0)) for name in self.feature_names_]
        return pd.Series(values, index=self.feature_names_, dtype=float).sort_values(
            ascending=False
        )
