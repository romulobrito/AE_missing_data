#!/usr/bin/env python3
"""
compute_band_metrics.py
Post-hoc computation of band-restricted metrics for the low-flow regime
[0, 20] EU from Phase 2 runs.

Outputs
-------
- band_metrics_summary.json  (inside --results_dir)
- Console table with Bias, MAE (with 95% CI), and ShareTrain per mechanism

Author: Romulo Brito da Silva - UFRJ
Date: 2026-03-25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    point = float(stat_fn(values))
    boots: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(stat_fn(values[idx])))
    arr = np.array(boots)
    ci_low = float(np.percentile(arr, 100 * alpha / 2))
    ci_high = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return point, ci_low, ci_high


# ---------------------------------------------------------------------------
# Reconstruction from saved artifacts (fallback when .npy are absent)
# ---------------------------------------------------------------------------

def reconstruct_predictions(
    run_dir: Path,
    data_path: Path,
    mechanism: str,
    missing_rate: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild y_true_missing, y_pred_missing, y_train_raw from saved
    model + scalers + split indices + original HDF5 dataset.

    Returns
    -------
    y_true_missing, y_pred_missing, y_train_raw  (all 1-D float arrays)
    """
    import joblib
    import pandas as pd
    import torch

    from teste_controlado_dados_faltantes_improved import inject_missing_data

    results_json = run_dir / "controlled_test_results.json"
    with open(results_json, "r") as fh:
        meta = json.load(fh)

    features: List[str] = meta["config"]["features"]
    target_var: str = meta["config"]["target_variable"]

    splits = np.load(str(run_dir / "splits_indices.npz"), allow_pickle=True)
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    scaler_X = joblib.load(str(run_dir / "scaler_X.joblib"))
    scaler_y = joblib.load(str(run_dir / "scaler_y.joblib"))

    means_json = run_dir / "train_feature_means.json"
    with open(means_json, "r") as fh:
        train_means_dict = json.load(fh)
    train_feature_means = pd.Series(train_means_dict)

    df = pd.read_hdf(str(data_path))
    valid_mask = df[target_var].notna()
    df_valid = df.loc[valid_mask]

    X_all = df_valid[features]
    y_all = df_valid[[target_var]]

    y_train_raw = y_all.loc[train_idx].values.astype(np.float32).flatten()

    df_test = pd.DataFrame(
        X_all.loc[test_idx].fillna(train_feature_means).values,
        columns=features,
        index=test_idx,
    )
    df_test[target_var] = y_all.loc[test_idx].values.flatten()

    aux_col: Optional[str] = None
    if mechanism == "MAR":
        aux_col = features[0]

    df_missing = inject_missing_data(
        df_test, mechanism, missing_rate, target_var, aux_col, seed
    )

    X_test_infer = df_missing[features].fillna(train_feature_means)
    X_test_scaled = scaler_X.transform(X_test_infer)

    from teste_controlado_cdae_v2 import ConditionalDenoisingAE

    latent_dim = meta["config"].get("latent_dim", 16)
    dropout = meta["config"].get("dropout", 0.2)
    model = ConditionalDenoisingAE(
        in_x=len(features), latent=latent_dim, dropout=dropout
    )
    state = torch.load(str(run_dir / "best_model.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.as_tensor(X_test_scaled, dtype=torch.float32)
        t0 = torch.zeros((len(X_test_scaled), 1), dtype=torch.float32)
        m1 = torch.ones((len(X_test_scaled), 1), dtype=torch.float32)
        y_pred_scaled = model(X_tensor, t0, m1).cpu().numpy().flatten()

    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    miss_mask = df_missing[target_var].isna().values
    y_true_missing = df_test[target_var].values[miss_mask]
    y_pred_missing = y_pred[miss_mask]

    return y_true_missing, y_pred_missing, y_train_raw


# ---------------------------------------------------------------------------
# Band-restricted metric computation
# ---------------------------------------------------------------------------

def compute_band_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    lo: float = 0.0,
    hi: float = 20.0,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, object]:
    """Compute Bias, MAE and ShareTrain restricted to band [lo, hi].

    Parameters
    ----------
    y_true : 1-D array -- true values on masked test points
    y_pred : 1-D array -- predicted values on masked test points
    y_train : 1-D array -- training target values (original scale)
    lo, hi : band boundaries
    n_boot : bootstrap iterations
    seed : RNG seed for bootstrap

    Returns
    -------
    dict with keys:
        bias, bias_ci_low, bias_ci_high,
        mae, mae_ci_low, mae_ci_high,
        share_train, n_band, n_total
    """
    band_mask = (y_true >= lo) & (y_true <= hi)
    n_band = int(band_mask.sum())
    n_total = len(y_true)

    if n_band == 0:
        return {
            "bias": float("nan"),
            "bias_ci_low": float("nan"),
            "bias_ci_high": float("nan"),
            "mae": float("nan"),
            "mae_ci_low": float("nan"),
            "mae_ci_high": float("nan"),
            "share_train": 0.0,
            "n_band": 0,
            "n_total": n_total,
        }

    residuals_band = y_pred[band_mask] - y_true[band_mask]
    abs_errors_band = np.abs(residuals_band)

    bias, bias_lo, bias_hi = bootstrap_ci(
        residuals_band, np.mean, n_boot=n_boot, seed=seed
    )
    mae, mae_lo, mae_hi = bootstrap_ci(
        abs_errors_band, np.mean, n_boot=n_boot, seed=seed
    )

    train_in_band = int(((y_train >= lo) & (y_train <= hi)).sum())
    share_train = 100.0 * train_in_band / max(1, len(y_train))

    return {
        "bias": bias,
        "bias_ci_low": bias_lo,
        "bias_ci_high": bias_hi,
        "mae": mae,
        "mae_ci_low": mae_lo,
        "mae_ci_high": mae_hi,
        "share_train": share_train,
        "n_band": n_band,
        "n_total": n_total,
    }


# ---------------------------------------------------------------------------
# Run directory discovery
# ---------------------------------------------------------------------------

def find_run_dir(base: Path, mechanism: str) -> Optional[Path]:
    """Find the most recent controlled_cdae_v2_* folder for a given
    mechanism inside a results tree.

    Supports multiple layouts:
        base/phase2/{MECH}_80pct/controlled_cdae_v2_*/
        base/{MECH}_80pct/controlled_cdae_v2_*/
        base/{MECH}/controlled_cdae_v2_*/          (ablation folders)
    """
    candidates = [
        base / "phase2" / f"{mechanism}_80pct",
        base / f"{mechanism}_80pct",
        base / mechanism,
    ]
    for mech_dir in candidates:
        if not mech_dir.is_dir():
            continue
        subdirs = sorted(mech_dir.glob("controlled_cdae_v2_*"))
        if subdirs:
            return subdirs[-1]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute band-restricted [0,20] metrics from Phase 2 runs "
            "and save to JSON."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(SCRIPT_DIR / "testes_blocked_20260211_092413"),
        help="Root directory containing phase2/{MCAR,MAR,MNAR}_80pct/",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(
            SCRIPT_DIR.parent.parent
            / "sulfatos_dados_concatenados_formatados_2021.h5"
        ),
        help="Path to the HDF5 dataset (needed only if .npy files are absent)",
    )
    parser.add_argument(
        "--band_lo", type=float, default=0.0, help="Lower band limit"
    )
    parser.add_argument(
        "--band_hi", type=float, default=20.0, help="Upper band limit"
    )
    parser.add_argument(
        "--n_boot", type=int, default=2000, help="Bootstrap iterations"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--missing_rate",
        type=float,
        default=0.8,
        help="Missing rate used in Phase 2",
    )
    args = parser.parse_args()

    results_base = Path(args.results_dir)
    data_path = Path(args.data)
    mechanisms = ["MCAR", "MAR", "MNAR"]

    all_stats: Dict[str, Dict[str, object]] = {}

    for mech in mechanisms:
        print(f"\n{'='*60}")
        print(f"  {mech}")
        print(f"{'='*60}")

        run_dir = find_run_dir(results_base, mech)
        if run_dir is None:
            print(f"[ERROR] Run directory not found for {mech}")
            continue
        print(f"  Run dir: {run_dir}")

        npy_true = run_dir / "y_true_missing.npy"
        npy_pred = run_dir / "y_pred_missing.npy"
        npy_train = run_dir / "y_train_raw.npy"

        if npy_true.exists() and npy_pred.exists() and npy_train.exists():
            print("  Loading from saved .npy files")
            y_true = np.load(str(npy_true))
            y_pred = np.load(str(npy_pred))
            y_train = np.load(str(npy_train))
        else:
            print("  .npy files not found -- reconstructing from model")
            if not data_path.exists():
                print(f"[ERROR] HDF5 dataset not found: {data_path}")
                continue
            y_true, y_pred, y_train = reconstruct_predictions(
                run_dir, data_path, mech, args.missing_rate, args.seed
            )
            np.save(str(npy_true), y_true)
            np.save(str(npy_pred), y_pred)
            np.save(str(npy_train), y_train)
            print("  Saved .npy files for future use")

        stats = compute_band_stats(
            y_true,
            y_pred,
            y_train,
            lo=args.band_lo,
            hi=args.band_hi,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        all_stats[mech] = stats

        print(
            f"  Bias  = {stats['bias']:+.2f}  "
            f"[{stats['bias_ci_low']:+.2f}, {stats['bias_ci_high']:+.2f}]"
        )
        print(
            f"  MAE   = {stats['mae']:.2f}  "
            f"[{stats['mae_ci_low']:.2f}, {stats['mae_ci_high']:.2f}]"
        )
        print(f"  ShareTrain = {stats['share_train']:.1f}%")
        print(
            f"  Band samples: {stats['n_band']} / {stats['n_total']} "
            f"({100 * stats['n_band'] / max(1, stats['n_total']):.1f}%)"
        )

    # -----------------------------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------------------------
    summary_path = results_base / "band_metrics_summary.json"
    with open(str(summary_path), "w") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"\n[OK] Summary saved to {summary_path}")

    # -----------------------------------------------------------------------
    # Print console table for easy copy-paste
    # -----------------------------------------------------------------------
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  {'Mechanism':<8} {'Bias [95% CI]':<28} {'MAE [95% CI]':<28} {'ShareTrain'}")
    print(sep)
    for mech in mechanisms:
        if mech not in all_stats:
            continue
        s = all_stats[mech]
        bias_col = f"{s['bias']:+.2f} [{s['bias_ci_low']:+.2f}, {s['bias_ci_high']:+.2f}]"
        mae_col = f"{s['mae']:.2f} [{s['mae_ci_low']:.2f}, {s['mae_ci_high']:.2f}]"
        share_col = f"{s['share_train']:.1f}%"
        print(f"  {mech:<8} {bias_col:<28} {mae_col:<28} {share_col}")
    print(sep)


if __name__ == "__main__":
    main()
