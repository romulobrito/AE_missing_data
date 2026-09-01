"""Diagnostic figures for baseline experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

CB_BLUE = "#4477AA"
CB_GREEN = "#228833"
CB_ORANGE = "#EE7733"
CB_RED = "#CC3311"
CB_GRAY = "#BBBBBB"


def save_scatter_true_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save masked-index scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color=CB_BLUE, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color=CB_RED, linestyle="--", linewidth=1.2, label="Ideal")
    ax.set_xlabel("True target (EU)")
    ax.set_ylabel("Predicted target (EU)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save residual distribution for masked indices."""
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=50, color=CB_ORANGE, alpha=0.8, edgecolor=CB_GRAY)
    ax.axvline(0.0, color=CB_RED, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (pred - true, EU)")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_comparison_bars(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    methods: Sequence[str],
    mechanisms: Sequence[str],
    output_path: Path,
    title: str,
    method_labels: Dict[str, str] | None = None,
    ci_metric: bool = False,
) -> None:
    """Grouped bar chart across methods and mechanisms."""
    x = np.arange(len(mechanisms), dtype=float)
    n_methods = len(methods)
    width = min(0.16, 0.82 / max(n_methods, 1))
    colors = [CB_BLUE, CB_GREEN, CB_ORANGE, CB_RED, "#AA3377"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, method in enumerate(methods):
        values: List[float] = []
        yerr_low: List[float] = []
        yerr_high: List[float] = []
        for mech in mechanisms:
            row = summary[method][mech]
            point = float(row[metric])
            values.append(point)
            if ci_metric and metric == "MAE":
                low_key = "MAE_CI_low"
                high_key = "MAE_CI_high"
                if low_key in row and high_key in row:
                    yerr_low.append(max(0.0, point - float(row[low_key])))
                    yerr_high.append(max(0.0, float(row[high_key]) - point))
                else:
                    yerr_low.append(0.0)
                    yerr_high.append(0.0)
        offset = (idx - (n_methods - 1) / 2.0) * width
        label = method_labels.get(method, method.upper()) if method_labels else method.upper()
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=colors[idx % len(colors)],
            alpha=0.9,
        )
        if ci_metric and metric == "MAE" and yerr_low and any(yerr_low + yerr_high):
            ax.errorbar(
                [bar.get_x() + bar.get_width() / 2.0 for bar in bars],
                values,
                yerr=[yerr_low, yerr_high],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms)
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
