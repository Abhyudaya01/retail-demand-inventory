"""Model comparison utilities for MLflow-tracked forecast runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retail_demand.eval.metrics import wape

BASELINE_WAPE = 0.650


def compare_models_across_folds(
    mlflow_experiment_id: str, tags_filter: dict[str, str] | None = None
) -> pd.DataFrame:
    """Load MLflow runs, aggregate fold metrics and rank models by mean WAPE."""
    import mlflow

    clauses = [f"tags.`{key}` = '{value}'" for key, value in (tags_filter or {}).items()]
    query = " and ".join(clauses) if clauses else ""
    runs = mlflow.search_runs(experiment_ids=[mlflow_experiment_id], filter_string=query)
    if runs.empty:
        return pd.DataFrame()

    metric_cols = [col for col in runs.columns if col.startswith("metrics.")]
    base_cols = ["tags.model", "tags.fold_id", *metric_cols]
    frame = runs[[col for col in base_cols if col in runs.columns]].copy()
    frame = frame.rename(columns=lambda col: col.replace("metrics.", "").replace("tags.", ""))
    if "model" not in frame.columns:
        frame["model"] = runs.get("tags.model_name", "unknown")
    grouped = (
        frame.groupby(["model", "fold_id"], dropna=False).mean(numeric_only=True).reset_index()
    )
    summary = grouped.groupby("model").mean(numeric_only=True).reset_index()
    summary["delta_wape_vs_baseline"] = (BASELINE_WAPE - summary["wape"]) / BASELINE_WAPE
    summary["rank"] = summary["wape"].rank(method="dense").astype(int)
    return summary.sort_values(["rank", "wape"]).reset_index(drop=True)


def bootstrap_significance(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap WAPE difference a-b with a fixed seed."""
    actual = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(y_pred_a, dtype=float)
    pred_b = np.asarray(y_pred_b, dtype=float)
    if not (len(actual) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, y_pred_a and y_pred_b must have equal length")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, len(actual), size=len(actual))
        diffs[idx] = wape(actual[sample_idx], pred_a[sample_idx]) - wape(
            actual[sample_idx], pred_b[sample_idx]
        )
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower_95": float(np.quantile(diffs, 0.025)),
        "ci_upper_95": float(np.quantile(diffs, 0.975)),
        "p_value_a_better_than_b": float(np.mean(diffs >= 0.0)),
    }
