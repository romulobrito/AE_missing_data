"""Evaluate a frozen aux_mlp checkpoint under frozen MCAR/MAR/MNAR masks."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from teste_controlado_cdae_v2 import bootstrap_regression_metrics
from teste_controlado_dados_faltantes_improved import set_seed

from ablations.config import AblationConfig, SUPPORTED_MECHANISMS
from ablations.models import AuxiliaryOnlyMLP
from ablations.prepare import _baseline_cfg_from_ablation
from baselines.protocol import load_protocol_artifacts


def _load_mask(cfg: AblationConfig, mechanism: str) -> np.ndarray:
    path = cfg.masks_dir() / f"{mechanism}_rate080_seed{cfg.seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen mask: {path}")
    payload = np.load(path, allow_pickle=True)
    return np.asarray(payload["miss_mask"], dtype=bool)


def evaluate_aux_mlp(
    cfg: AblationConfig,
    mechanisms: List[str] | None = None,
) -> Dict[str, Dict[str, object]]:
    """Score one checkpoint on each mechanism using frozen masks."""
    set_seed(cfg.seed)
    mechs = list(mechanisms) if mechanisms else list(SUPPORTED_MECHANISMS)
    baseline_cfg = _baseline_cfg_from_ablation(cfg)
    artifacts = load_protocol_artifacts(baseline_cfg)

    model_dir = cfg.model_dir()
    ckpt_path = model_dir / "best_model.pth"
    config_path = model_dir / "model_config.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        model_config = json.load(fh)

    n_features = int(len(list(artifacts["features"])))
    model = AuxiliaryOnlyMLP(
        n_features=n_features,
        latent_dim=int(model_config.get("latent_dim", cfg.latent_dim)),
        dropout=float(model_config.get("dropout", cfg.dropout)),
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scaler_y = artifacts["scaler_y"]
    x_test_scaled = artifacts["x_test_scaled"]
    y_test_raw = artifacts["y_test_raw"]

    with torch.no_grad():
        y_pred_scaled = model(torch.as_tensor(x_test_scaled, dtype=torch.float32))
        y_pred_scaled_np = y_pred_scaled.cpu().numpy().reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled_np).reshape(-1)

    summary: Dict[str, Dict[str, object]] = {}
    for mechanism in mechs:
        miss_mask = _load_mask(cfg, mechanism)
        y_true_missing = y_test_raw[miss_mask]
        y_pred_missing = y_pred[miss_mask]
        n_missing = int(miss_mask.sum())
        if n_missing == 0:
            raise ValueError(f"Empty mask for {mechanism}")

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
            "mechanism": mechanism,
            "missing_rate": float(cfg.missing_rate),
            "model": "aux_mlp",
            "bootstrap_B": int(cfg.bootstrap_n),
        }
        metrics.update(boot)

        eval_dir = cfg.eval_dir(mechanism)
        eval_dir.mkdir(parents=True, exist_ok=True)
        np.save(eval_dir / "y_true_missing.npy", y_true_missing)
        np.save(eval_dir / "y_pred_missing.npy", y_pred_missing)
        np.save(eval_dir / "miss_mask.npy", miss_mask)
        payload = {
            "metrics": metrics,
            "model_config": model_config,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(eval_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)

        summary[mechanism] = metrics
        print(
            f"[OK] aux_mlp/{mechanism}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}"
        )

    return summary
