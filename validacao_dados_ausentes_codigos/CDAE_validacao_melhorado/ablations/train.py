"""Train the capacity-matched auxiliary-only MLP once."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from teste_controlado_cdae_v2 import EarlyStopping, ProcessDataset
from teste_controlado_dados_faltantes_improved import set_seed

from ablations.config import AblationConfig
from ablations.models import AuxiliaryOnlyMLP
from ablations.prepare import (
    _baseline_cfg_from_ablation,
    write_run_manifest,
)
from baselines.protocol import load_protocol_artifacts


def train_aux_mlp(cfg: AblationConfig) -> Dict[str, object]:
    """Fit aux_mlp on shared protocol folds and persist best checkpoint."""
    set_seed(cfg.seed)
    baseline_cfg = _baseline_cfg_from_ablation(cfg)
    artifacts = load_protocol_artifacts(baseline_cfg)

    n_features = int(len(list(artifacts["features"])))
    model = AuxiliaryOnlyMLP(
        n_features=n_features,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    )
    model_config = model.config_dict()
    model_config["dropout"] = cfg.dropout
    model_config["seed"] = cfg.seed

    x_train = artifacts["x_train_scaled"]
    x_val = artifacts["x_val_scaled"]
    y_train = artifacts["y_train_scaled"].reshape(-1, 1)
    y_val = artifacts["y_val_scaled"].reshape(-1, 1)

    loaders = {
        "train": DataLoader(
            ProcessDataset(x_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "val": DataLoader(
            ProcessDataset(x_val, y_val),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    device = torch.device(cfg.device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    early = EarlyStopping(patience=cfg.patience, min_delta=1e-6)

    model_dir = cfg.model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loaders["train"]:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch in loaders["val"]:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_hat = model(x_batch)
                loss = criterion(y_hat, y_batch)
                val_loss += float(loss.item())
                n_val += 1
        val_loss /= max(1, n_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        early(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_dir / "best_model.pth")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )
        if early.stop:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with open(model_dir / "model_config.json", "w", encoding="utf-8") as fh:
        json.dump(model_config, fh, indent=2, ensure_ascii=True)
    with open(model_dir / "train_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=True)

    write_run_manifest(cfg, artifacts, model_config=model_config)
    print(
        f"[OK] aux_mlp trained: params={model_config['parameter_count']}, "
        f"best_val={best_val:.6f}"
    )
    print(f"     Checkpoint: {model_dir / 'best_model.pth'}")
    return {
        "model_config": model_config,
        "best_val_loss": float(best_val),
        "history": history,
        "model_dir": str(model_dir),
    }
