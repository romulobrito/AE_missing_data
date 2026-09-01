"""Prepare shared protocol artifacts and frozen masks for aux_mlp."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from teste_controlado_dados_faltantes_improved import inject_missing_data, set_seed

from ablations.config import AblationConfig, SUPPORTED_MECHANISMS
from baselines.config import BaselineProtocolConfig
from baselines.protocol import load_protocol_artifacts


def _baseline_cfg_from_ablation(cfg: AblationConfig) -> BaselineProtocolConfig:
    """Map ablation settings onto the baselines protocol loader."""
    return BaselineProtocolConfig(
        data_path=cfg.data_path,
        feature_config_path=cfg.feature_config_path,
        output_root=cfg.output_root,
        seed=cfg.seed,
        bootstrap_n=cfg.bootstrap_n,
        missing_rate=cfg.missing_rate,
        aux_col=cfg.aux_col,
        blocked_by_index=True,
        auto_skip_until_coverage=True,
    )


def prepare_shared_protocol(cfg: AblationConfig) -> Dict[str, object]:
    """
    Copy frozen baseline shared artifacts into the ablation output root,
    then reload them through the baselines protocol loader.
    """
    set_seed(cfg.seed)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    shared_dst = cfg.shared_dir()
    shared_src = cfg.protocol_shared_source
    if not shared_src.is_dir():
        raise FileNotFoundError(
            f"Protocol shared source not found: {shared_src}. "
            "Run baselines prepare first."
        )

    if shared_dst.exists():
        shutil.rmtree(shared_dst)
    shutil.copytree(shared_src, shared_dst)

    baseline_cfg = _baseline_cfg_from_ablation(cfg)
    artifacts = load_protocol_artifacts(baseline_cfg)
    return artifacts


def create_and_save_masks(
    cfg: AblationConfig,
    artifacts: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    """Generate and freeze MCAR/MAR/MNAR masks on the test fold."""
    set_seed(cfg.seed)
    masks_dir = cfg.masks_dir()
    masks_dir.mkdir(parents=True, exist_ok=True)

    target_var = str(artifacts["target_var"])
    features: List[str] = list(artifacts["features"])
    test_idx_model = list(artifacts["test_idx_model"])
    x_test_raw = artifacts["x_test_raw"]
    y_test_raw = artifacts["y_test_raw"]

    df_test = pd.DataFrame(x_test_raw, columns=features, index=test_idx_model)
    df_test[target_var] = y_test_raw

    audit: Dict[str, Dict[str, object]] = {}
    for mechanism in SUPPORTED_MECHANISMS:
        aux_col = cfg.aux_col.strip()
        if mechanism == "MAR" and not aux_col:
            aux_col = features[0]

        df_missing = inject_missing_data(
            df_test.copy(),
            mechanism,
            cfg.missing_rate,
            target_var,
            aux_col if aux_col else None,
            cfg.seed,
        )
        miss_mask = df_missing[target_var].isna().values.astype(bool)
        masked_pos = np.where(miss_mask)[0].astype(np.int64)
        eligible = np.arange(len(miss_mask), dtype=np.int64)

        out_path = masks_dir / f"{mechanism}_rate080_seed{cfg.seed}.npz"
        np.savez(
            out_path,
            miss_mask=miss_mask,
            masked_indices=masked_pos,
            eligible_indices=eligible,
            mechanism=np.array(mechanism),
            rate=np.array(cfg.missing_rate),
            seed=np.array(cfg.seed),
        )

        y_masked = y_test_raw[miss_mask]
        y_unmasked = y_test_raw[~miss_mask]
        audit[mechanism] = {
            "n_eligible": int(len(eligible)),
            "n_masked": int(miss_mask.sum()),
            "rate_realized": float(miss_mask.mean()),
            "target_mean_masked": float(np.mean(y_masked)) if len(y_masked) else None,
            "target_mean_unmasked": float(np.mean(y_unmasked)) if len(y_unmasked) else None,
            "low_flow_masked_count": int(np.sum(y_masked <= 20.0)) if len(y_masked) else 0,
            "mask_file": str(out_path),
        }

    with open(masks_dir / "mask_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=True)
    return audit


def write_run_manifest(
    cfg: AblationConfig,
    artifacts: Dict[str, object],
    model_config: Dict[str, object] | None = None,
) -> Path:
    """Persist run-level provenance for LaTeX/reproducibility."""
    coverage = artifacts.get("coverage_report", {})
    skip = 0.0
    if isinstance(coverage, dict):
        skip = float(coverage.get("skip_head_frac_used", 0.0))

    payload: Dict[str, object] = {
        "variant": "aux_mlp",
        "seed": cfg.seed,
        "protocol_source": str(cfg.protocol_shared_source),
        "split_mode": "blocked_by_index",
        "head_skip": skip,
        "n_features": len(list(artifacts["features"])),
        "missing_rate": cfg.missing_rate,
        "mechanisms": list(SUPPORTED_MECHANISMS),
        "training": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "patience": cfg.patience,
            "dropout": cfg.dropout,
            "latent_dim": cfg.latent_dim,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
    }
    if model_config is not None:
        payload["model_config"] = model_config

    path = cfg.output_root / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
    return path
