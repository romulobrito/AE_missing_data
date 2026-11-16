#!/usr/bin/env python3
"""Create readable publication comparison figures from archived aggregate JSON.

This is a presentational replot of existing statistics, not re-estimation.
Produces high-resolution raster PNG and vector PDF files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_NAMES = {
    "knn": "KNN-Imputer",
    "ridge": "Ridge",
    "hgb": "HGB",
    "cdae_l1": r"TCDR ($\lambda=1$)",
    "cdae_l0": r"TCDR ($\lambda=0$)",
}
MECHANISMS = ("MCAR", "MAR", "MNAR")


def plot_metric(data: dict, key: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    names = list(MODEL_NAMES)
    x = np.arange(len(MECHANISMS))
    bar_width = 0.15
    for i, model in enumerate(names):
        values = [float(data[model][mech][key]) for mech in MECHANISMS]
        pos = x + (i - (len(names) - 1) / 2) * bar_width
        if key == "MAE":
            low = [float(data[model][mech]["MAE_CI_low"]) for mech in MECHANISMS]
            high = [float(data[model][mech]["MAE_CI_high"]) for mech in MECHANISMS]
            yerr = [np.maximum(np.array(values) - low, 0), np.maximum(np.array(high) - values, 0)]
            ax.bar(pos, values, width=bar_width, yerr=yerr, capsize=2, label=MODEL_NAMES[model])
        else:
            ax.bar(pos, values, width=bar_width, label=MODEL_NAMES[model])
    ax.set_xticks(x, MECHANISMS)
    ax.set_xlabel("Controlled missingness mechanism", fontsize=12)
    ax.set_ylabel("MAE (anonymized engineering units)" if key == "MAE" else r"$R^2$", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=10)
    if key == "R2":
        ax.set_ylim(bottom=0.0, top=1.03)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    for model in MODEL_NAMES:
        for mech in MECHANISMS:
            if model not in data or mech not in data[model]:
                raise ValueError(f"Missing summary: {model}/{mech}")
    plot_metric(data, "MAE", args.output / "comparison_mae_by_mechanism")
    plot_metric(data, "R2", args.output / "comparison_r2_by_mechanism")
    print(f"Created publication-ready charts in {args.output}")


if __name__ == "__main__":
    main()
