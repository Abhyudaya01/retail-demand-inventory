"""Run Phase 7 GBM forecast experiments with strict leakage guards."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as pd_types
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from retail_demand.eval.metrics import metric_dict, per_segment_metrics
from retail_demand.eval.splits import walk_forward_splits
from retail_demand.mlflow_setup import configure_mlflow
from retail_demand.models.gbm import LightGBMForecaster, XGBoostForecaster
from retail_demand.spark.session import get_spark_session

DEFAULT_EXPERIMENT_PATH = "/Users/alohani@umd.edu/retail-demand-forecasting"
BASELINE_WAPE = 0.650
BOOL_COLUMNS = ["is_on_promo", "is_weekend", "is_us_holiday"]
CATEGORICAL_COLUMNS = [
    "store_id",
    "sku_id",
    "category",
    "subcategory",
    "region",
    "state",
    "store_type",
]
EXCLUDE_COLUMNS = {"label_units_sold", "date", "_gold_run_id", "year", "month", "revenue"}


def _spark_to_pandas_arrow(df: Any) -> pd.DataFrame:
    if hasattr(df, "toArrow"):
        return df.toArrow().to_pandas()
    import pyarrow as pa

    return pa.Table.from_batches(df._collect_as_arrow()).to_pandas()


def load_features_master_pandas(
    spark: SparkSession, gold_root: str, sample_rows: int | None = None
) -> pd.DataFrame:
    """Read Gold features_master by path and collect once through Arrow."""
    frame = spark.read.format("delta").load(f"{gold_root.rstrip('/')}/features_master")
    if sample_rows is not None:
        order_cols = [col for col in ("store_id", "sku_id", "date") if col in frame.columns]
        frame = frame.orderBy(F.xxhash64(*[F.col(col) for col in order_cols])).limit(sample_rows)
    result = _spark_to_pandas_arrow(frame)
    result["date"] = pd.to_datetime(result["date"], unit="ns")
    for col in BOOL_COLUMNS:
        if col in result.columns:
            result[col] = result[col].astype("bool")
    for col in CATEGORICAL_COLUMNS:
        if col in result.columns:
            result[col] = result[col].astype("category")
    object_features = [
        col for col in result.columns
        if col not in EXCLUDE_COLUMNS and result[col].dtype == "object"
    ]
    assert not object_features, f"Object dtype model features: {object_features}"
    return result


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return model feature columns and the categorical subset.

    Exclude revenue because revenue = units_sold × current_price at the same date;
    including it leaks the label.
    """
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in feature_cols]
    return feature_cols, categorical_cols


def sanity_check_no_target_leakage(
    features_df: pd.DataFrame, feature_cols: list[str], label_col: str = "label_units_sold"
) -> None:
    """Compute correlation between numeric features and the label.

    Warn if any single feature has |corr| > 0.95, which indicates target leakage.
    """
    for col in feature_cols:
        if pd_types.is_float_dtype(features_df[col]) or pd_types.is_integer_dtype(
            features_df[col]
        ):
            valid = features_df[[col, label_col]].dropna()
            if valid[col].nunique() < 2 or valid[label_col].nunique() < 2:
                continue
            corr = valid[col].corr(valid[label_col])
            if pd.notna(corr) and abs(corr) > 0.95:
                raise ValueError(
                    f"Feature '{col}' has correlation {corr:.3f} with label. "
                    "This indicates target leakage. Exclude this feature or investigate."
                )


def assert_no_fold_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    nan_rate_tolerance: float = 0.05,
) -> None:
    """Raise before fitting when temporal or feature-completeness leakage is detected."""
    assert train_df["date"].max() < val_df["date"].min(), "train/validation dates overlap"
    val_span = val_df["date"].max() - val_df["date"].min()
    recent_train_start = train_df["date"].max() - val_span
    recent_train = train_df[train_df["date"] >= recent_train_start]
    train_nan = recent_train[feature_cols].isna().mean()
    val_nan = val_df[feature_cols].isna().mean()
    suspicious = (train_nan - val_nan) > nan_rate_tolerance
    if bool(suspicious.any()):
        offenders = suspicious[suspicious].index.tolist()
        raise AssertionError(f"validation feature NaN rates differ from train by >5%: {offenders}")


def _split_fold(df: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df["date"] >= fold["train_start"]) & (df["date"] <= fold["train_end"])]
    val = df[(df["date"] >= fold["val_start"]) & (df["date"] <= fold["val_end"])]
    return train.copy(), val.copy()


def _add_volume_tier(train_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["store_id", "sku_id"]
    volume = train_df.groupby(keys, observed=True)["label_units_sold"].sum().reset_index()
    try:
        volume["volume_tier"] = pd.qcut(
            volume["label_units_sold"], q=3, labels=["low", "medium", "high"], duplicates="drop"
        )
    except ValueError:
        volume["volume_tier"] = "medium"
    volume["volume_tier"] = volume["volume_tier"].astype(str)
    return scored_df.merge(volume[[*keys, "volume_tier"]], on=keys, how="left")


def _segment_metric_log(scored_df: pd.DataFrame, train_df: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    enriched = _add_volume_tier(train_df, scored_df)
    for segment_col, top_n in (("category", 5), ("store_type", 3), ("volume_tier", None)):
        if segment_col not in enriched.columns:
            continue
        if top_n is None:
            segment_values = enriched[segment_col].dropna().unique().tolist()
        else:
            segment_values = (
                enriched.groupby(segment_col, observed=True)["y_true"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
        segment_frame = enriched[enriched[segment_col].isin(segment_values)]
        for _, row in per_segment_metrics(segment_frame, [segment_col]).iterrows():
            value = str(row[segment_col])
            for metric in ("mae", "rmse", "wape", "smape", "bias"):
                result[f"{segment_col}_{value}_{metric}"] = float(row[metric])
    return result


def _log_importance_artifacts(model: Any, X_val: pd.DataFrame, run_name: str) -> None:
    import matplotlib.pyplot as plt
    import mlflow
    import shap

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        importances = model.feature_importances_.rename("importance").reset_index()
        importances = importances.rename(columns={"index": "feature"})
        importance_csv = tmp_path / "feature_importances.csv"
        importances.to_csv(importance_csv, index=False)
        mlflow.log_artifact(str(importance_csv))

        plot_path = tmp_path / "feature_importances_plot.png"
        top = importances.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["importance"])
        ax.set_title(f"{run_name} feature importances")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(plot_path))

        shap_path = tmp_path / "shap_summary_top15.png"
        shap_sample = X_val.head(5000)
        if getattr(model, "model_", None) is not None and not shap_sample.empty:
            explainer = shap.TreeExplainer(model.model_)
            shap_values = explainer.shap_values(shap_sample)
            shap.summary_plot(shap_values, shap_sample, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(shap_path))


def _run_one_fold(
    model: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    fold: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    assert_no_fold_leakage(train_df, val_df, feature_cols)
    X_train = train_df[feature_cols]
    y_train = train_df["label_units_sold"]
    X_val = val_df[feature_cols]
    y_val = val_df["label_units_sold"]
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_val)
    scored = val_df[["store_id", "sku_id", "date", "label_units_sold"]].copy()
    for col in ("category", "store_type"):
        if col in val_df.columns:
            scored[col] = val_df[col]
    scored = scored.rename(columns={"label_units_sold": "y_true"})
    scored["y_pred"] = y_pred
    metrics = metric_dict(scored["y_true"].to_numpy(), scored["y_pred"].to_numpy())
    metrics.update(_segment_metric_log(scored, train_df))
    metrics["delta_wape_vs_moving_average"] = (BASELINE_WAPE - metrics["wape"]) / BASELINE_WAPE
    return scored, metrics


def run_cv_experiment(
    features_df: pd.DataFrame,
    model_class: type,
    model_params: dict[str, Any] | None,
    cv_config: dict[str, Any],
    mlflow_config: dict[str, Any],
) -> pd.DataFrame:
    """Run walk-forward CV for one GBM class with leakage assertions before each fit."""
    import mlflow

    feature_cols, categorical_cols = get_feature_columns(features_df)
    sanity_check_no_target_leakage(features_df, feature_cols)
    folds = walk_forward_splits(
        features_df["date"],
        n_folds=int(cv_config.get("cv_folds", 3)),
        horizon_days=int(cv_config.get("horizon_days", 14)),
        min_train_days=int(cv_config.get("min_train_days", 180)),
    )
    rows: list[dict[str, Any]] = []
    model_name = getattr(model_class, "name", model_class.__name__)
    if model_name == "lightgbm":
        mlflow.lightgbm.autolog(log_models=False)
    elif model_name == "xgboost":
        mlflow.xgboost.autolog(log_models=False)

    for fold in folds:
        train_df, val_df = _split_fold(features_df, fold)
        model = model_class(params=model_params or {}, categorical_features=categorical_cols)
        with mlflow.start_run(run_name=f"{model_name}-fold-{fold['fold_id']}", nested=True):
            mlflow.set_tag("phase", "7-gbm")
            mlflow.set_tag("model", model_name)
            mlflow.set_tag("fold_id", str(fold["fold_id"]))
            mlflow.log_params(
                {
                    "model": model_name,
                    "fold_id": fold["fold_id"],
                    "horizon_days": cv_config.get("horizon_days", 14),
                    "min_train_days": cv_config.get("min_train_days", 180),
                    "feature_count": len(feature_cols),
                    "categorical_features": ",".join(categorical_cols),
                    **(model_params or {}),
                }
            )
            scored, metrics = _run_one_fold(model, train_df, val_df, feature_cols, fold)
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            with tempfile.TemporaryDirectory() as tmpdir:
                forecast_path = Path(tmpdir) / "forecast_vs_actual.csv"
                scored.to_csv(forecast_path, index=False)
                mlflow.log_artifact(str(forecast_path))
            _log_importance_artifacts(
                model, val_df[feature_cols], f"{model_name}-fold-{fold['fold_id']}"
            )
            rows.append({"model": model_name, "fold_id": fold["fold_id"], **metrics})

    result = pd.DataFrame(rows)
    if not result.empty and mlflow_config.get("print_comparison", True):
        comparison = result.groupby("model")[["mae", "rmse", "wape", "smape", "bias"]].mean()
        print(comparison.reset_index().sort_values("wape").to_string(index=False))
    return result


def run_gbm_experiments(
    spark: SparkSession,
    gold_root: str,
    experiment_path: str = DEFAULT_EXPERIMENT_PATH,
    cv_folds: int = 3,
    horizon_days: int = 14,
    min_train_days: int = 180,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """Run LightGBM and XGBoost under separate MLflow parent runs."""
    import mlflow

    experiment_id = configure_mlflow(experiment_path)
    features_df = load_features_master_pandas(spark, gold_root, sample_rows=sample_rows)
    cv_config = {
        "cv_folds": cv_folds,
        "horizon_days": horizon_days,
        "min_train_days": min_train_days,
    }
    all_results = []
    for model_class, parent_name in (
        (LightGBMForecaster, "phase-7-lightgbm"),
        (XGBoostForecaster, "phase-7-xgboost"),
    ):
        with mlflow.start_run(run_name=parent_name):
            mlflow.set_tag("phase", "7-gbm")
            mlflow.set_tag("model", model_class.name)
            mlflow.log_param("baseline_wape_bar", BASELINE_WAPE)
            result = run_cv_experiment(
                features_df,
                model_class,
                model_params={},
                cv_config=cv_config,
                mlflow_config={"experiment_id": experiment_id},
            )
            all_results.append(result)
    comparison = pd.concat(all_results, ignore_index=True)
    print(comparison.groupby("model")[["mae", "rmse", "wape", "smape", "bias"]].mean())
    return comparison


def main() -> None:
    """Run Phase 7 GBM experiments from the command line."""
    parser = argparse.ArgumentParser(description="Run GBM forecast experiments.")
    parser.add_argument("--gold-root", default="data/gold")
    parser.add_argument("--experiment-path", default="retail-demand-gbm-local")
    parser.add_argument("--cv-folds", type=int, default=1)
    parser.add_argument("--horizon-days", type=int, default=14)
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--sample-rows", type=int, default=None)
    args = parser.parse_args()
    spark = get_spark_session(app_name="retail-demand-gbm-local")
    try:
        run_gbm_experiments(
            spark,
            args.gold_root,
            args.experiment_path,
            args.cv_folds,
            args.horizon_days,
            args.min_train_days,
            args.sample_rows,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
