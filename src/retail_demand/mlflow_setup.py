"""MLflow setup and logging helpers for forecasting experiments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def configure_mlflow(experiment_path: str) -> str:
    """Configure Databricks or local MLflow tracking and return the experiment ID."""
    import mlflow

    if "DATABRICKS_RUNTIME_VERSION" not in os.environ and "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment(experiment_path)
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_path)
    else:
        experiment_id = experiment.experiment_id
    return str(experiment_id)


def log_forecast_run(
    run_name: str,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    forecast_df: pd.DataFrame,
    fold_id: int,
) -> None:
    """Log params, metrics, forecast CSV and scatter plot to the active MLflow run."""
    import matplotlib.pyplot as plt
    import mlflow

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("phase", "6-baselines")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("fold_id", str(fold_id))
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            forecast_path = tmp_path / "forecast_vs_actual.csv"
            forecast_df.to_csv(forecast_path, index=False)
            mlflow.log_artifact(str(forecast_path))

            plot_path = tmp_path / "actual_vs_predicted.png"
            plot_df = forecast_df[["y_true", "y_pred"]].dropna().head(10_000)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(plot_df["y_true"], plot_df["y_pred"], alpha=0.25, s=8)
            ax.set_xlabel("Actual units")
            ax.set_ylabel("Predicted units")
            ax.set_title(f"{model_name} fold {fold_id}")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path))
