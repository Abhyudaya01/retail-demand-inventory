"""Metric tables for Phase 8 segment-level error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-9


def _metrics(group: pd.DataFrame, label_col: str, pred_col: str) -> dict[str, float | int]:
    actual = group[label_col].to_numpy(dtype=float)
    predicted = group[pred_col].to_numpy(dtype=float)
    error = predicted - actual
    abs_actual_sum = max(float(np.sum(np.abs(actual))), EPSILON)
    smape_den = np.maximum((np.abs(actual) + np.abs(predicted)) / 2.0, EPSILON)
    return {
        "n_rows": int(len(group)),
        "mean_actual": float(np.mean(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.sum(np.abs(error)) / abs_actual_sum),
        "smape": float(np.mean(np.abs(error) / smape_den)),
        "bias": float(np.mean(error)),
    }


def per_segment_metrics(
    preds_df: pd.DataFrame,
    segment_cols: list[str],
    label_col: str = "y_true",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Inputs: predictions and segment columns.

    Outputs: tidy segment metrics. Side effects: none.
    """
    rows = []
    for keys, group in preds_df.groupby(segment_cols, dropna=False, observed=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(segment_cols, key_values, strict=True)),
                **_metrics(group, label_col, pred_col),
            }
        )
    return pd.DataFrame(rows).sort_values(segment_cols).reset_index(drop=True)


def bias_flags(per_segment_df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Inputs: segment metrics; outputs: over/under-forecast flags; side effects: none."""
    frame = per_segment_df.copy()
    denom = frame["mean_actual"].abs()
    frame["bias_ratio"] = np.where(denom > EPSILON, frame["bias"] / denom, np.nan)
    flagged = frame[frame["bias_ratio"].abs() > threshold].copy()
    flagged["bias_direction"] = np.where(flagged["bias_ratio"] > 0.0, "over", "under")
    return flagged.sort_values("bias_ratio", key=lambda col: col.abs(), ascending=False)


def best_and_worst_skus(
    preds_df: pd.DataFrame, n: int = 20, min_total_actual: float = 10.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs: predictions; outputs: non-intermittent best/worst SKU WAPE tables.

    WAPE is unreliable on intermittent or near-zero-demand series because tiny actual
    denominators can produce huge ratios. This function ranks only store/SKU pairs with
    total actual demand of at least min_total_actual in the validation window.
    Side effects: none.
    """
    group_cols = ["store_id", "sku_id"]
    metrics = per_segment_metrics(preds_df, group_cols)
    metrics = metrics.rename(columns={"n_rows": "n_days"})
    total_actual = (
        preds_df.groupby(group_cols, dropna=False, observed=True)["y_true"]
        .sum()
        .reset_index(name="total_actual")
    )
    metrics = metrics.merge(total_actual, on=group_cols, how="left")
    metrics = metrics[metrics["total_actual"] >= min_total_actual]
    if "category" in preds_df.columns:
        categories = preds_df.groupby(group_cols, dropna=False, observed=True)["category"].first()
        metrics = metrics.merge(categories.reset_index(), on=group_cols, how="left")
    else:
        metrics["category"] = pd.NA
    cols = ["store_id", "sku_id", "category", "wape", "n_days"]
    ranked = metrics.sort_values("wape", ascending=True).reset_index(drop=True)
    best = ranked.head(n)
    worst = ranked.sort_values("wape", ascending=False).head(n)
    return best[cols].reset_index(drop=True), worst[cols].reset_index(drop=True)
