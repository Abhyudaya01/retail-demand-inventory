"""Matplotlib plots for Phase 8 error analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _prepare_output(output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def error_heatmap(
    per_segment_df: pd.DataFrame, x: str, y: str, output_path: str, metric: str = "wape"
) -> None:
    """Inputs: segment metrics; outputs: PNG heatmap; side effects: writes output_path."""
    import matplotlib.pyplot as plt

    path = _prepare_output(output_path)
    pivot = per_segment_df.pivot(index=y, columns=x, values=metric).sort_index()
    fig_width = max(6.0, 1.1 * len(pivot.columns))
    fig_height = max(4.0, 0.6 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{metric.upper()} by {y} and {x}")
    for row_idx, row_name in enumerate(pivot.index):
        for col_idx, col_name in enumerate(pivot.columns):
            value = pivot.loc[row_name, col_name]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def actual_vs_predicted_scatter(
    preds_df: pd.DataFrame, sample_n: int = 10000, output_path: str = "reports/figures/scatter.png"
) -> None:
    """Inputs: predictions; outputs: PNG log-log scatter; side effects: writes output_path."""
    import matplotlib.pyplot as plt

    path = _prepare_output(output_path)
    sample = preds_df.sample(n=min(sample_n, len(preds_df)), random_state=42)
    actual = sample["y_true"].to_numpy(dtype=float)
    predicted = sample["y_pred"].to_numpy(dtype=float)
    upper = max(float(np.max(actual, initial=0.0)), float(np.max(predicted, initial=0.0)), 1.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual + 1.0, predicted + 1.0, alpha=0.35, s=12)
    ax.plot([1.0, upper + 1.0], [1.0, upper + 1.0], color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual units sold + 1")
    ax.set_ylabel("Predicted units sold + 1")
    ax.set_title("Actual vs Predicted Demand")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def worst_sku_time_series(
    preds_df: pd.DataFrame, store_id: str, sku_id: str, output_path: str
) -> None:
    """Inputs: predictions and store/SKU key.

    Outputs: PNG time series. Side effects: writes output_path.
    """
    import matplotlib.pyplot as plt

    path = _prepare_output(output_path)
    frame = preds_df[(preds_df["store_id"] == store_id) & (preds_df["sku_id"] == sku_id)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(frame["date"], frame["y_true"], marker="o", linewidth=1.5, label="Actual")
    ax.plot(frame["date"], frame["y_pred"], marker="o", linewidth=1.5, label="Predicted")
    ax.set_title(f"Worst SKU Time Series: {store_id} / {sku_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
