"""Run one baseline model under one missingness mechanism."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from teste_controlado_cdae_v2 import bootstrap_regression_metrics
from teste_controlado_dados_faltantes_improved import inject_missing_data, set_seed

from baselines.config import BaselineProtocolConfig, SUPPORTED_MECHANISMS, SUPPORTED_MODELS
from baselines.models import (
    fit_hgb,
    fit_knn_imputer,
    fit_ridge,
    predict_knn_masked,
    predict_supervised,
)
from baselines.plots import save_residual_histogram, save_scatter_true_vs_pred
from baselines.protocol import load_protocol_artifacts


def run_single_baseline(
    cfg: BaselineProtocolConfig,
    artifacts: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Execute one model x mechanism baseline and persist artifacts."""
    if cfg.model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {cfg.model}")
    if cfg.mechanism not in SUPPORTED_MECHANISMS:
        raise ValueError(f"Unsupported mechanism: {cfg.mechanism}")

    set_seed(cfg.seed)
    if artifacts is None:
        artifacts = load_protocol_artifacts(cfg)
    run_dir = cfg.run_dir(cfg.model, cfg.mechanism)
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    target_var = str(artifacts["target_var"])
    features: List[str] = list(artifacts["features"])
    test_idx_model = list(artifacts["test_idx_model"])
    train_feature_means = artifacts["train_feature_means"]
    scaler_x = artifacts["scaler_x"]
    scaler_y = artifacts["scaler_y"]

    x_train_scaled = artifacts["x_train_scaled"]
    x_val_scaled = artifacts["x_val_scaled"]
    x_test_scaled = artifacts["x_test_scaled"]
    y_train_scaled = artifacts["y_train_scaled"]
    y_val_scaled = artifacts["y_val_scaled"]
    y_test_scaled = artifacts["y_test_scaled"]
    x_test_raw = artifacts["x_test_raw"]
    y_test_raw = artifacts["y_test_raw"]

    df_test = pd.DataFrame(x_test_raw, columns=features, index=test_idx_model)
    df_test[target_var] = y_test_raw

    aux_col = cfg.aux_col.strip()
    if cfg.mechanism == "MAR" and not aux_col:
        aux_col = features[0]

    df_missing = inject_missing_data(
        df_test.copy(),
        cfg.mechanism,
        cfg.missing_rate,
        target_var,
        aux_col if aux_col else None,
        cfg.seed,
    )
    miss_mask = df_missing[target_var].isna().values
    n_missing = int(miss_mask.sum())
    if n_missing == 0:
        raise ValueError("No masked target values after injection.")

    x_test_for_pred = df_missing[features].fillna(train_feature_means)
    x_test_scaled_for_pred = scaler_x.transform(x_test_for_pred.values.astype(np.float32))

    if cfg.model == "knn":
        model, model_config = fit_knn_imputer(
            x_train_scaled, y_train_scaled, cfg.k_neighbors
        )
        y_pred = predict_knn_masked(
            model,
            x_test_scaled_for_pred,
            y_test_scaled,
            miss_mask,
            scaler_y,
        )
    elif cfg.model == "ridge":
        model, model_config = fit_ridge(
            x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled
        )
        y_pred = predict_supervised(model, x_test_scaled_for_pred, scaler_y)
    elif cfg.model == "hgb":
        model, model_config = fit_hgb(
            x_train_scaled,
            y_train_scaled,
            x_val_scaled,
            y_val_scaled,
            cfg.seed,
        )
        y_pred = predict_supervised(model, x_test_scaled_for_pred, scaler_y)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    y_true_missing = y_test_raw[miss_mask]
    y_pred_missing = y_pred[miss_mask]

    mae = float(mean_absolute_error(y_true_missing, y_pred_missing))
    rmse = float(np.sqrt(mean_squared_error(y_true_missing, y_pred_missing)))
    r2 = float(r2_score(y_true_missing, y_pred_missing))
    boot = bootstrap_regression_metrics(
        y_true_missing, y_pred_missing, n_boot=cfg.bootstrap_n, seed=cfg.seed
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_samples": n_missing,
        "mechanism": cfg.mechanism,
        "missing_rate": float(cfg.missing_rate),
        "model": cfg.model,
    }
    metrics.update(boot)

    np.save(run_dir / "y_true_missing.npy", y_true_missing)
    np.save(run_dir / "y_pred_missing.npy", y_pred_missing)
    np.save(run_dir / "miss_mask.npy", miss_mask)
    joblib.dump(model, run_dir / "model.joblib")

    with open(run_dir / "model_config.json", "w", encoding="utf-8") as fh:
        json.dump(model_config, fh, indent=2, ensure_ascii=True)

    results = {
        "metrics": metrics,
        "protocol": cfg.to_dict(),
        "model_config": model_config,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "baseline_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=True)

    title_prefix = f"{cfg.model.upper()} - {cfg.mechanism} (masked indices)"
    save_scatter_true_vs_pred(
        y_true_missing,
        y_pred_missing,
        f"{title_prefix}: True vs Predicted",
        fig_dir / "scatter_true_vs_pred.png",
    )
    save_residual_histogram(
        y_true_missing,
        y_pred_missing,
        f"{title_prefix}: Residuals",
        fig_dir / "residual_histogram.png",
    )

    print(f"[OK] {cfg.model}/{cfg.mechanism}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    print(f"     Artifacts: {run_dir}")
    return results
