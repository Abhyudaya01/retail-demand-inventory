"""End-to-end tests for the Phase 7 GBM runner."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from retail_demand.bronze.ingest import BronzeIngestor
from retail_demand.config import Settings
from retail_demand.data_generation.generator import SyntheticDataConfig, write_dataset
from retail_demand.experiments.run_gbm import load_features_master_pandas, run_cv_experiment
from retail_demand.gold.build import GoldBuilder, default_sql_dir
from retail_demand.models.gbm import LightGBMForecaster
from retail_demand.silver.build import SilverBuilder


class _NoopRun:
    def __enter__(self) -> _NoopRun:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


class _FakeMlflow:
    logged_metrics: list[dict[str, float]] = []
    lightgbm = SimpleNamespace(autolog=lambda **_kwargs: None)
    xgboost = SimpleNamespace(autolog=lambda **_kwargs: None)

    @staticmethod
    def set_experiment(_name: str) -> None:
        return None

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
    def log_metrics(metrics: dict[str, float]) -> None:
        _FakeMlflow.logged_metrics.append(metrics)

    @staticmethod
    def log_artifact(*_args: Any, **_kwargs: Any) -> None:
        return None

    @staticmethod
    def search_runs(**_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"metrics.wape": [0.5]})


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        __import__("mlflow")
        return
    except ModuleNotFoundError:
        pass
    _FakeMlflow.logged_metrics = []
    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow)


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

        def predict(self, X: pd.DataFrame, **_kwargs: Any) -> Any:
            return [float(self.train.label.mean())] * len(X)

        def feature_importance(self, importance_type: str = "gain") -> list[float]:
            return [1.0] * len(self.train.data.columns)

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


def test_tiny_lightgbm_run_logs_expected_mlflow_outputs(
    spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny medallion build can run one LightGBM fold through MLflow."""
    _install_fake_lightgbm(monkeypatch)
    _install_fake_mlflow(monkeypatch)
    monkeypatch.setattr(
        "retail_demand.experiments.run_gbm._log_importance_artifacts",
        lambda *_args, **_kwargs: None,
    )
    raw_root = tmp_path / "raw"
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    gold_root = tmp_path / "gold"
    config = SyntheticDataConfig(
        num_stores=2,
        num_skus=5,
        num_days=200,
        end_date=date(2026, 1, 31),
        duplicate_sales_rate=0.0,
        negative_units_rate=0.0,
        wrong_store_case_rate=0.0,
        missing_price_rate=0.0,
    )
    write_dataset(config, target="local", output_path=str(raw_root), settings=Settings())
    BronzeIngestor(spark, str(raw_root), str(bronze_root), "default").ingest_all()
    SilverBuilder(spark, str(bronze_root), "unused", str(silver_root)).build_all()
    GoldBuilder(spark, str(silver_root), str(gold_root), default_sql_dir()).build_all()

    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    experiment_name = "test-gbm-local"
    mlflow.set_experiment(experiment_name)
    features = load_features_master_pandas(spark, str(gold_root), sample_rows=None)
    assert pd.api.types.is_datetime64_any_dtype(features["date"])

    with mlflow.start_run(run_name="parent"):
        result = run_cv_experiment(
            features,
            LightGBMForecaster,
            {"n_estimators": 10, "min_child_samples": 1, "num_leaves": 7},
            {"cv_folds": 1, "horizon_days": 14, "min_train_days": 120},
            {"print_comparison": False},
        )

    assert len(result) == 1
    runs = mlflow.search_runs(filter_string="tags.phase = '7-gbm'")
    assert not runs.empty
    assert "metrics.wape" in runs.columns
