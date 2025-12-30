#!/usr/bin/env python3
"""
teste_controlado_cdae_v2.py
Controlled validation for CDAE with ALL methodological improvements from CDAE.py

Melhorias portadas:
1. Split temporal leakage-safe (temporal_split_indices, blocked_split_indices)
2. Auto-skip para garantir cobertura minima (temporal_split_indices_with_skip)
3. Imputacao leakage-safe (medias do train only)
4. Selecao de features leakage-safe
5. Bootstrap com intervalos de confianca
6. Experimento downstream (simulacao de alertas)
7. Stress test com multiplos p_mask
8. Coverage report
9. Artefatos reprodutiveis
10. Plot temporal comparativo

Author: Romulo Brito da Silva - UFRJ
Date: 2025-01-18
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ks_2samp, shapiro
import matplotlib.pyplot as plt
import joblib

# Local import helpers
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Reuse utilities from base script
from teste_controlado_dados_faltantes_improved import (
    load_feature_config,
    set_seed,
    timestamp,
    inject_missing_data,
)


# ============================================================================
# DATASET HELPERS
# ============================================================================

class ProcessDataset(Dataset):
    """Custom dataset for industrial process data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping implementation."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.stop = False
    
    def __call__(self, val_loss: float):
        if self.mode == 'min':
            improved = val_loss < self.best - self.min_delta
        else:
            improved = val_loss > self.best + self.min_delta
            
        if improved:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================================
# CDAE MODEL
# ============================================================================

class ConditionalDenoisingAE(nn.Module):
    """Conditional denoising autoencoder for target t.
    Input: [x; t_tilde; m_t] -> Output: t_hat
    """
    
    def __init__(self, in_x: int, latent: int = 16, dropout: float = 0.2):
        super().__init__()
        in_dim = in_x + 2  # x + t_tilde + m_t
        h1, h2 = max(64, 8 * in_dim), max(32, 4 * in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.LeakyReLU(0.1), nn.Dropout(dropout / 2),
            nn.Linear(h2, latent), nn.BatchNorm1d(latent), nn.LeakyReLU(0.1),
            nn.Linear(latent, h2), nn.LeakyReLU(0.1),
            nn.Linear(h2, 1),
        )
    
    def forward(self, x: torch.Tensor, t_tilde: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t_tilde, m_t], dim=1)
        return self.net(inp)


# ============================================================================
# TRAINING
# ============================================================================

def train_cdae(model: nn.Module, loaders: Dict[str, DataLoader], epochs: int, lr: float, patience: int,
               device: str, output_dir: str, p_mask: float = 0.5, masked_weight: float = 1.0) -> Dict[str, list]:
    """Train conditional denoising AE: random mask on t, reconstruct true t.
    Loss: weighted MSE, weighting masked samples by (1 + masked_weight).
    """
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                           verbose=True, min_lr=1e-6)
    early = EarlyStopping(patience=patience, min_delta=1e-6)

    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, n_batches = 0.0, 0
        for X_batch, y_batch in loaders['train']:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # build random mask and noised target
            m_t = (torch.rand_like(y_batch) < p_mask).float()
            t_tilde = (1.0 - m_t) * y_batch
            optimizer.zero_grad()
            y_hat = model(X_batch, t_tilde, m_t)
            per_elem = criterion(y_hat, y_batch)
            weights = 1.0 + masked_weight * m_t
            loss = (per_elem * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        # Val (use same corruption pattern for stability)
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in loaders['val']:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                m_t = (torch.rand_like(y_batch) < p_mask).float()
                t_tilde = (1.0 - m_t) * y_batch
                y_hat = model(X_batch, t_tilde, m_t)
                per_elem = criterion(y_hat, y_batch)
                weights = 1.0 + masked_weight * m_t
                loss = (per_elem * weights).mean()
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(1, n_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        early(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(best_state, Path(output_dir) / 'best_model.pth')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        if early.stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_cdae(model: nn.Module, loader: DataLoader, scaler_y, device: str) -> np.ndarray:
    """Evaluate by reconstructing t with m_t=1 and t_tilde=0 (in scaled space)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X = X_batch.to(device)
            # zeros (scaled space) and ones mask
            t_zero = torch.zeros_like(y_batch).to(device)
            m_one = torch.ones_like(y_batch).to(device)
            y_hat_scaled = model(X, t_zero, m_one).cpu().numpy()
            preds.append(y_hat_scaled)
            trues.append(y_batch.numpy())
    y_pred_scaled = np.concatenate(preds)
    y_true_scaled = np.concatenate(trues)
    # inverse transform to original
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)
    return y_true.ravel(), y_pred.ravel()


# ============================================================================
# SPLIT FUNCTIONS (PORTADAS DO CDAE.PY)
# ============================================================================

def temporal_split_indices(df: pd.DataFrame, time_column: str, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create contiguous train/val/test splits ordered by time."""
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in dataset")
    sorted_idx = df[time_column].sort_values(kind='mergesort').index.to_numpy()
    n = len(sorted_idx)
    n_test = max(1, int(np.ceil(n * float(test_size))))
    n_rem = max(1, n - n_test)
    n_val = max(1, int(np.ceil(n_rem * float(val_size))))
    if (n_test + n_val) >= n:
        raise ValueError("Temporal split sizes too large for dataset length")
    test_idx = sorted_idx[-n_test:]
    val_idx = sorted_idx[-(n_test + n_val):-n_test]
    train_idx = sorted_idx[:(n - n_test - n_val)]
    return train_idx, val_idx, test_idx


def blocked_split_indices(df: pd.DataFrame, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create contiguous splits using the current row order as acquisition time."""
    idx = df.index.to_numpy()
    n = len(idx)
    n_test = max(1, int(np.ceil(n * float(test_size))))
    n_rem = max(1, n - n_test)
    n_val = max(1, int(np.ceil(n_rem * float(val_size))))
    if (n_test + n_val) >= n:
        raise ValueError("Blocked split sizes too large for dataset length")
    test_idx = idx[-n_test:]
    val_idx = idx[-(n_test + n_val):-n_test]
    train_idx = idx[:(n - n_test - n_val)]
    return train_idx, val_idx, test_idx


def compute_target_coverage(df: pd.DataFrame, idx: np.ndarray, target: str) -> Tuple[float, int, int]:
    """Compute coverage of target present within indices."""
    if len(idx) == 0:
        return 0.0, 0, 0
    present_mask = df[target].notna()
    total = int(len(idx))
    present = int(present_mask.loc[idx].sum())
    coverage = float(present) / float(total) if total > 0 else 0.0
    return coverage, present, total


def temporal_split_indices_with_skip(
    df: pd.DataFrame,
    time_column: str,
    test_size: float,
    val_size: float,
    target: str,
    min_target_coverage: float,
    min_target_count: int,
    skip_head_frac: float = 0.0,
    auto_skip_until_coverage: bool = False,
    max_skip_head_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create temporal splits after optionally skipping an initial head fraction."""
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in dataset")

    sorted_idx = df[time_column].sort_values(kind="mergesort").index.to_numpy()
    n_total = int(len(sorted_idx))
    if n_total < 3:
        raise ValueError("Dataset too small for temporal split")

    def compute_from(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(len(arr))
        n_test = max(1, int(np.ceil(n * float(test_size))))
        n_rem = max(1, n - n_test)
        n_val = max(1, int(np.ceil(n_rem * float(val_size))))
        if (n_test + n_val) >= n:
            raise ValueError("Temporal split sizes too large for dataset length")
        test_idx = arr[-n_test:]
        val_idx = arr[-(n_test + n_val):-n_test]
        train_idx = arr[: (n - n_test - n_val)]
        return train_idx, val_idx, test_idx

    present_mask = df[target].notna()
    start_frac = max(0.0, float(skip_head_frac))
    start_frac = min(start_frac, 0.95)

    if auto_skip_until_coverage:
        upper = float(max_skip_head_frac)
        upper = float(min(max(upper, start_frac), 0.95))
        candidate_fracs = np.linspace(start_frac, upper, num=51)
        for frac in candidate_fracs:
            start_idx = int(np.floor(n_total * frac))
            arr = sorted_idx[start_idx:]
            if len(arr) < 10:
                continue
            try:
                train_idx, val_idx, test_idx = compute_from(arr)
            except ValueError:
                continue
            total_train = int(len(train_idx))
            if total_train <= 0:
                continue
            present_train = int(present_mask.loc[train_idx].sum())
            coverage_train = float(present_train) / float(total_train)
            if (coverage_train >= float(min_target_coverage)) and (present_train >= int(min_target_count)):
                return train_idx, val_idx, test_idx, float(frac)
        start_idx = int(np.floor(n_total * start_frac))
        arr = sorted_idx[start_idx:]
        train_idx, val_idx, test_idx = compute_from(arr)
        return train_idx, val_idx, test_idx, float(start_frac)

    start_idx = int(np.floor(n_total * start_frac))
    arr = sorted_idx[start_idx:]
    train_idx, val_idx, test_idx = compute_from(arr)
    return train_idx, val_idx, test_idx, float(start_frac)


def blocked_split_indices_with_skip(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    target: str,
    min_target_coverage: float,
    min_target_count: int,
    skip_head_frac: float = 0.0,
    auto_skip_until_coverage: bool = False,
    max_skip_head_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create blocked-by-index splits after optionally skipping an initial head fraction."""
    idx = df.index.to_numpy()
    n_total = int(len(idx))
    if n_total < 3:
        raise ValueError("Dataset too small for blocked split")

    def compute_from(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(len(arr))
        n_test = max(1, int(np.ceil(n * float(test_size))))
        n_rem = max(1, n - n_test)
        n_val = max(1, int(np.ceil(n_rem * float(val_size))))
        if (n_test + n_val) >= n:
            raise ValueError("Blocked split sizes too large for dataset length")
        test_idx = arr[-n_test:]
        val_idx = arr[-(n_test + n_val):-n_test]
        train_idx = arr[: (n - n_test - n_val)]
        return train_idx, val_idx, test_idx

    present_mask = df[target].notna()
    start_frac = max(0.0, float(skip_head_frac))
    start_frac = min(start_frac, 0.95)

    if auto_skip_until_coverage:
        upper = float(max_skip_head_frac)
        upper = float(min(max(upper, start_frac), 0.95))
        candidate_fracs = np.linspace(start_frac, upper, num=51)
        for frac in candidate_fracs:
            start_idx = int(np.floor(n_total * frac))
            arr = idx[start_idx:]
            if len(arr) < 10:
                continue
            try:
                train_idx, val_idx, test_idx = compute_from(arr)
            except ValueError:
                continue
            total_train = int(len(train_idx))
            if total_train <= 0:
                continue
            present_train = int(present_mask.loc[train_idx].sum())
            coverage_train = float(present_train) / float(total_train)
            if (coverage_train >= float(min_target_coverage)) and (present_train >= int(min_target_count)):
                return train_idx, val_idx, test_idx, float(frac)
        start_idx = int(np.floor(n_total * start_frac))
        arr = idx[start_idx:]
        train_idx, val_idx, test_idx = compute_from(arr)
        return train_idx, val_idx, test_idx, float(start_frac)

    start_idx = int(np.floor(n_total * start_frac))
    arr = idx[start_idx:]
    train_idx, val_idx, test_idx = compute_from(arr)
    return train_idx, val_idx, test_idx, float(start_frac)


# ============================================================================
# BOOTSTRAP E DOWNSTREAM (PORTADOS DO CDAE.PY)
# ============================================================================

def bootstrap_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    """Bootstrap MAE and RMSE to compute CIs."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes: List[float] = []
    rmses: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        mae_i = float(mean_absolute_error(y_true[idx], y_pred[idx]))
        rmse_i = float(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))
        maes.append(mae_i)
        rmses.append(rmse_i)
    mae_arr = np.array(maes)
    rmse_arr = np.array(rmses)
    def q(a: np.ndarray, p: float) -> float:
        return float(np.quantile(a, p))
    return {
        'MAE_bootstrap': float(mae_arr.mean()),
        'MAE_CI_low': q(mae_arr, 0.025),
        'MAE_CI_high': q(mae_arr, 0.975),
        'RMSE_bootstrap': float(rmse_arr.mean()),
        'RMSE_CI_low': q(rmse_arr, 0.025),
        'RMSE_CI_high': q(rmse_arr, 0.975),
    }


def downstream_experiment(y_true: np.ndarray, y_pred: np.ndarray, missing_pct: float, recall_target: float = 0.8, label_percentile: float = 80.0, seed: int = 42, threshold: Optional[float] = None) -> Dict[str, float]:
    """Simulate missingness on the target and compare alert pipeline using t vs t_hat.
    - If threshold is provided, uses it directly (deploy-like: threshold computed on training data).
    - If threshold is None, computes it from y_true (legacy behavior, may cause leakage).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}
    missing_pct = float(missing_pct)
    if missing_pct < 0.0 or missing_pct > 1.0:
        raise ValueError("missing_pct must be within [0, 1]")
    m = rng.random(n) < missing_pct
    # Use provided threshold (from training) or compute from y_true (legacy)
    if threshold is None:
        threshold = float(np.percentile(y_true, label_percentile))
    else:
        threshold = float(threshold)
    y_label = (y_true >= threshold).astype(int)
    scores_baseline = y_true.copy()
    scores_vs = y_true.copy()
    scores_baseline[m] = 0.0
    scores_vs[m] = y_pred[m]
    pr_auc_base = float(average_precision_score(y_label, scores_baseline))
    pr_auc_vs = float(average_precision_score(y_label, scores_vs))
    def precision_at_recall(scores: np.ndarray, target_recall: float) -> float:
        p, r, th = precision_recall_curve(y_label, scores)
        mask = r >= target_recall
        if not np.any(mask):
            return float(p[-1])
        return float(np.max(p[mask]))
    prec_base = precision_at_recall(scores_baseline, recall_target)
    prec_vs = precision_at_recall(scores_vs, recall_target)
    return {
        'missing_pct': missing_pct,
        'label_percentile': float(label_percentile),
        'PR_AUC_baseline': pr_auc_base,
        'PR_AUC_virtual_sensor': pr_auc_vs,
        'precision_at_recall_baseline': prec_base,
        'precision_at_recall_virtual_sensor': prec_vs,
        'recall_target': float(recall_target),
    }


# ============================================================================
# PLOTS
# ============================================================================

def plot_temporal_missing_focus(y_true_missing: np.ndarray, y_pred_missing: np.ndarray,
                                missing_indices: np.ndarray, mechanism: str, rate: float,
                                target_name: str, output_dir: str):
    """Plot focado apenas nos pontos com missing."""
    plt.figure(figsize=(16, 9))
    plt.scatter(missing_indices, y_true_missing, color='blue', s=18, label='Real Value (hole)', alpha=0.8)
    plt.scatter(missing_indices, y_pred_missing, color='red', s=18, label='Reconstructed', alpha=0.8)
    plt.title(f'{mechanism} - {rate*100:.0f}% missing data (Missing Values Focus)')
    plt.xlabel('Sample Index')
    plt.ylabel(f'Value of {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    mae = mean_absolute_error(y_true_missing, y_pred_missing)
    rmse = float(np.sqrt(mean_squared_error(y_true_missing, y_pred_missing)))
    r2 = r2_score(y_true_missing, y_pred_missing)
    plt.text(0.01, 0.98, f'Missing: {len(y_true_missing)} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}',
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    path = f"{output_dir}/{mechanism}_{int(rate*100)}_temporal_missing_focus.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Temporal missing focus plot saved to: {path}")


def plot_controlled_results(history: Dict[str, list], y_true_missing: np.ndarray, 
                            y_pred_missing: np.ndarray, mechanism: str, rate: float,
                            output_dir: str):
    """Plot multiplo com 6 paineis (training, distributions, residuals, summary)."""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history
    axes[0, 0].plot(history['train_loss'], label='Training', alpha=0.8, linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.8, linewidth=2)
    axes[0, 0].set_title('Training History', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution comparison
    y_true = y_true_missing.ravel()
    y_pred = y_pred_missing.ravel()
    axes[0, 1].hist(y_true, bins=50, alpha=0.7, label='Original', density=True, color='blue')
    axes[0, 1].hist(y_pred, bins=50, alpha=0.7, label='Reconstructed', density=True, color='red')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution Comparison', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter: Original vs Reconstructed
    axes[0, 2].scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors='k', linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0, 2].plot(lims, lims, 'r--', linewidth=2, alpha=0.8)
    axes[0, 2].set_xlabel('Original Values')
    axes[0, 2].set_ylabel('Reconstructed Values')
    axes[0, 2].set_title('Original vs Reconstructed', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_pred - y_true
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Reconstructed Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Analysis', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Residuals Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary
    axes[1, 2].text(0.1, 0.8, 'CONTROLLED TEST SUMMARY:', fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    
    summary_text = f"""
Mechanism: {mechanism}
Missing Rate: {rate*100:.1f}%
Missing Values Evaluated: {len(y_true):,}

ORIGINAL STATISTICS:
Mean: {y_true.mean():.4f}
Std: {y_true.std():.4f}
Min: {y_true.min():.4f}
Max: {y_true.max():.4f}

RECONSTRUCTED STATISTICS:
Mean: {y_pred.mean():.4f}
Std: {y_pred.std():.4f}
Min: {y_pred.min():.4f}
Max: {y_pred.max():.4f}

METRICS:
MAE: {mae:.4f}
RMSE: {rmse:.4f}
R2: {r2:.4f}
    """
    axes[1, 2].text(0.1, 0.7, summary_text.strip(), fontsize=10, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Test Summary', fontweight='bold')
    
    plt.suptitle(f'Controlled Missing Data Test: {mechanism} - {rate*100:.1f}% Missing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = f"{output_dir}/controlled_test_results.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Results visualization saved to: {path}")


def generate_temporal_comparison_plot_cdae(model: nn.Module, df: pd.DataFrame, selected_features: List[str],
                                           target_var: str, scaler_X, scaler_y, device: str, output_dir: str,
                                           train_feature_means: pd.Series):
    """Temporal plot for CDAE using inference inputs [X; 0; 1]."""
    os.makedirs(output_dir, exist_ok=True)

    all_features = df[selected_features].copy()
    all_features_imputed = all_features.fillna(train_feature_means).fillna(0)
    X_scaled = scaler_X.transform(all_features_imputed)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        t_zero = torch.zeros((len(X_scaled), 1), dtype=torch.float32).to(device)
        m_one = torch.ones((len(X_scaled), 1), dtype=torch.float32).to(device)
        y_hat_scaled = model(X_tensor, t_zero, m_one).cpu().numpy().flatten()
    y_hat = scaler_y.inverse_transform(y_hat_scaled.reshape(-1, 1)).flatten()
    y_hat = np.clip(y_hat, 0.001, 100.0)

    dados_originais = df[target_var].copy()
    present = ~dados_originais.isna()
    idx = np.arange(len(df))

    plt.figure(figsize=(18, 9))
    plt.scatter(idx[present], dados_originais[present], label='Real Values', alpha=0.6, color='blue', s=40)
    plt.scatter(idx[present], y_hat[present], label='Predictions (Present Data)', alpha=0.4, color='green', s=40)
    plt.scatter(idx[~present], y_hat[~present], label='Predictions (Missing Data)', alpha=0.8, color='red', s=40)
    plt.xlabel('Sample Index')
    plt.ylabel(f'Value of {target_var}')
    plt.title('Comparison between Real and Predicted Values (CDAE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    path = f"{output_dir}/temporal_comparison_plot_cdae.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Temporal comparison plot saved to: {path}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(metrics: Dict, config: Dict, eda_info: Dict, output_dir: str):
    """Save complete results."""
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'metrics': metrics,
        'config': config,
        'eda_info': eda_info,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(f"{output_dir}/controlled_test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f" Complete results saved to: {output_dir}/controlled_test_results.json")


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    print(" CONTROLLED MISSING DATA EXPERIMENT (CDAE V2 - MELHORADO)")
    print("=" * 70)

    set_seed(args.seed)

    print(f" Loading data from: {args.data}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    if data_path.suffix in {".h5", ".hdf5"}:
        df = pd.read_hdf(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f" Dataset loaded: {df.shape[0]} samples, {df.shape[1]} variables")

    # Load features
    target_var, features = load_feature_config(args.feature_config)
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataset!")

    vars_needed = [target_var] + features
    missing_vars = [v for v in vars_needed if v not in df.columns]
    if missing_vars:
        print(f" Missing variables: {missing_vars}")
        return

    # EDA info
    eda_info = {
        'dataset_shape': df.shape,
        'target_variable': target_var,
        'features': features,
        'n_features': len(features)
    }

    # Pre-split indices (leakage-safe)
    if args.time_column:
        print(f" Using temporal split by column: {args.time_column}")
        if (args.time_skip_head_frac > 0.0) or args.auto_skip_until_coverage:
            train_idx, val_idx, test_idx, skip_used = temporal_split_indices_with_skip(
                df,
                args.time_column,
                args.time_test_size,
                args.time_val_size,
                target_var,
                args.min_target_coverage,
                args.min_target_count,
                skip_head_frac=args.time_skip_head_frac,
                auto_skip_until_coverage=bool(args.auto_skip_until_coverage),
                max_skip_head_frac=args.max_skip_head_frac,
            )
            print(f"  Applied head skip fraction: {skip_used:.3f}")
        else:
            train_idx, val_idx, test_idx = temporal_split_indices(
                df, args.time_column, args.time_test_size, args.time_val_size
            )
    elif args.blocked_by_index:
        print(" Using blocked-by-index split (row order treated as time)")
        if (args.time_skip_head_frac > 0.0) or args.auto_skip_until_coverage:
            train_idx, val_idx, test_idx, skip_used = blocked_split_indices_with_skip(
                df,
                args.time_test_size,
                args.time_val_size,
                target_var,
                args.min_target_coverage,
                args.min_target_count,
                skip_head_frac=args.time_skip_head_frac,
                auto_skip_until_coverage=bool(args.auto_skip_until_coverage),
                max_skip_head_frac=args.max_skip_head_frac,
            )
            print(f"  Applied head skip fraction: {skip_used:.3f}")
        else:
            train_idx, val_idx, test_idx = blocked_split_indices(
                df, args.time_test_size, args.time_val_size
            )
    else:
        all_indices = df.index.to_numpy()
        train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, random_state=args.seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)

    # Coverage checks
    cov_train, cnt_train, tot_train = compute_target_coverage(df, np.array(train_idx), target_var)
    cov_val, cnt_val, tot_val = compute_target_coverage(df, np.array(val_idx), target_var)
    cov_test, cnt_test, tot_test = compute_target_coverage(df, np.array(test_idx), target_var)

    coverage_report = {
        'train': {'coverage': cov_train, 'present': cnt_train, 'total': tot_train},
        'val':   {'coverage': cov_val,   'present': cnt_val,   'total': tot_val},
        'test':  {'coverage': cov_test,  'present': cnt_test,  'total': tot_test},
        'thresholds': {
            'min_target_coverage': float(args.min_target_coverage),
            'min_target_count': int(args.min_target_count),
        },
        'split_mode': 'time_column' if args.time_column else ('blocked_by_index' if args.blocked_by_index else 'random')
    }

    # Enforce thresholds if strict
    if (args.time_column or args.blocked_by_index) and args.strict_temporal:
        if (cov_train < float(args.min_target_coverage)) or (cnt_train < int(args.min_target_count)):
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / 'coverage_report.json').open('w') as f:
                json.dump(coverage_report, f, indent=2)
            raise ValueError(
                "Insufficient target coverage in TRAIN block under strict temporal policy. "
                f"Coverage={cov_train:.3f}, present={cnt_train}, total={tot_train}. "
                "Adjust time_val_size/time_test_size or disable --strict_temporal, or reduce thresholds via "
                "--min_target_coverage/--min_target_count."
            )

    # Save coverage report
    out_dir = Path(args.output) / f"controlled_cdae_v2_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'coverage_report.json').open('w') as f:
        json.dump(coverage_report, f, indent=2)

    # Keep only rows with target present
    df_with_target = df[df[target_var].notna()].copy()

    # Intersect folds
    train_idx_model = [i for i in train_idx if i in df_with_target.index]
    val_idx_model = [i for i in val_idx if i in df_with_target.index]
    test_idx_model = [i for i in test_idx if i in df_with_target.index]

    if (len(train_idx_model) == 0) or (len(val_idx_model) == 0) or (len(test_idx_model) == 0):
        print(" WARNING: Empty split after filtering by target. Resplitting within available rows.")
        available_idx = list(df_with_target.index)
        train_idx_model, temp_idx_model = train_test_split(available_idx, test_size=0.3, random_state=args.seed)
        val_idx_model, test_idx_model = train_test_split(temp_idx_model, test_size=0.5, random_state=args.seed)

    # Build matrices with train-only imputation
    X_all = df_with_target[features].apply(pd.to_numeric, errors="coerce")
    y_all = df_with_target[[target_var]]

    train_feature_means = X_all.loc[train_idx_model].mean()
    if train_feature_means.isna().any():
        bad = train_feature_means[train_feature_means.isna()].index.tolist()
        raise ValueError(f"No training data to compute mean for: {bad}")

    def impute_with_train_means(df_part: pd.DataFrame) -> pd.DataFrame:
        return df_part.fillna(train_feature_means)

    X_train_raw = impute_with_train_means(X_all.loc[train_idx_model]).values.astype(np.float32)
    X_val_raw = impute_with_train_means(X_all.loc[val_idx_model]).values.astype(np.float32)
    X_test_raw = impute_with_train_means(X_all.loc[test_idx_model]).values.astype(np.float32)

    y_train_raw = y_all.loc[train_idx_model].values.astype(np.float32)
    y_val_raw = y_all.loc[val_idx_model].values.astype(np.float32)
    y_test_raw = y_all.loc[test_idx_model].values.astype(np.float32)

    for name, arr in {"X_train_raw": X_train_raw, "X_val_raw": X_val_raw, "X_test_raw": X_test_raw}.items():
        if np.isnan(arr).any():
            raise ValueError(f"NaN remains in {name} after train-mean imputation.")

    print(
        f"Final dimensions (after target filter and feature imputation): "
        f"X_train={X_train_raw.shape}, X_val={X_val_raw.shape}, X_test={X_test_raw.shape}"
    )

    # Scaling (fit on train only)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler_X.transform(X_val_raw).astype(np.float32)
    X_test = scaler_X.transform(X_test_raw).astype(np.float32)

    scaler_y = RobustScaler() if args.robust_target else StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw).astype(np.float32)
    y_val = scaler_y.transform(y_val_raw).astype(np.float32)
    y_test = scaler_y.transform(y_test_raw).astype(np.float32)

    loaders = {
        "train": DataLoader(ProcessDataset(X_train, y_train), batch_size=args.batch, shuffle=True, num_workers=2),
        "val": DataLoader(ProcessDataset(X_val, y_val), batch_size=args.batch, shuffle=False, num_workers=2),
        "test": DataLoader(ProcessDataset(X_test, y_test), batch_size=args.batch, shuffle=False, num_workers=2),
    }

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Decide p_mask list (stress or single)
    p_masks: List[float] = [float(args.p_mask)]
    if args.stress_mask_probs and len(args.stress_mask_probs) > 0:
        p_masks = [float(p) for p in args.stress_mask_probs]

    all_run_metrics: List[Dict[str, float]] = []

    for p_mask_value in p_masks:
        # Create output subdir if running stress
        sub_output = str(out_dir)
        if len(p_masks) > 1:
            sub_output = str(out_dir / f"p_mask_{p_mask_value:.2f}")
        Path(sub_output).mkdir(parents=True, exist_ok=True)

        model = ConditionalDenoisingAE(in_x=len(features), latent=args.latent, dropout=args.dropout).to(device)

        print(" Model created:")
        print(f"  Input: {len(features)} features")
        print(f"  Latent: {args.latent} dimensions")
        print(f"  Device: {device.upper()}")
        print(f"  Training with p_mask={p_mask_value:.2f}, masked_weight={args.masked_weight:.2f}")

        # Train CDAE
        history = train_cdae(
            model, loaders, epochs=args.epochs, lr=args.lr, patience=args.patience, device=device,
            output_dir=sub_output, p_mask=p_mask_value, masked_weight=args.masked_weight
        )

        # Restore best checkpoint
        best_path = Path(sub_output) / "best_model.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))

        # PHASE 2: Inject missing on TEST and evaluate
        df_test = pd.DataFrame(X_all.loc[test_idx_model].fillna(train_feature_means).values, 
                              columns=features, index=test_idx_model)
        df_test[target_var] = y_all.loc[test_idx_model].values.flatten()

        aux_col = args.aux_col
        if args.mechanism == 'MAR' and aux_col is None:
            aux_col = features[0]
            print(f"Auto-selected auxiliary column for MAR: {aux_col}")

        df_missing = inject_missing_data(df_test, args.mechanism, args.missing_rate, target_var, aux_col, args.seed)

        # Reconstruct
        X_test_infer = df_missing[features].fillna(train_feature_means)
        X_test_scaled = scaler_X.transform(X_test_infer)
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.as_tensor(X_test_scaled, dtype=torch.float32).to(device)
            t0 = torch.zeros((len(X_test_scaled), 1), dtype=torch.float32).to(device)
            m1 = torch.ones((len(X_test_scaled), 1), dtype=torch.float32).to(device)
            y_pred_scaled = model(X_tensor, t0, m1).cpu().numpy().flatten()
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        miss_mask = df_missing[target_var].isna().values
        if miss_mask.sum() == 0:
            print(" No missing target values found for evaluation!")
            continue

        y_true_missing = df_test[target_var].values[miss_mask]
        y_pred_missing = y_pred[miss_mask]

        # Metrics
        mae = float(mean_absolute_error(y_true_missing, y_pred_missing))
        rmse = float(np.sqrt(mean_squared_error(y_true_missing, y_pred_missing)))
        r2 = float(r2_score(y_true_missing, y_pred_missing))
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'n_samples': int(len(y_true_missing)),
            'p_mask': float(p_mask_value),
            'masked_weight': float(args.masked_weight),
            'mechanism': args.mechanism,
            'missing_rate': float(args.missing_rate),
        }

        # Bootstrap CIs
        boot = bootstrap_regression_metrics(y_true_missing, y_pred_missing, n_boot=args.bootstrap_n, seed=args.seed)
        metrics.update(boot)

        # KS test (distributional alignment)
        ks_stat, ks_p = ks_2samp(y_true_missing, y_pred_missing)
        metrics['KS_statistic'] = float(ks_stat)
        metrics['KS_pvalue'] = float(ks_p)

        # Shapiro-Wilk test (residual normality)
        residuals = y_pred_missing - y_true_missing
        # Shapiro-Wilk works best with sample sizes between 3 and 5000
        # For larger samples, we can use a subset or report limitation
        if len(residuals) > 5000:
            # Use a random sample of 5000 for Shapiro-Wilk
            np.random.seed(args.seed)
            sample_idx = np.random.choice(len(residuals), size=5000, replace=False)
            residuals_sample = residuals[sample_idx]
            shapiro_stat, shapiro_p = shapiro(residuals_sample)
            metrics['Shapiro_statistic'] = float(shapiro_stat)
            metrics['Shapiro_pvalue'] = float(shapiro_p)
            metrics['Shapiro_n_sample'] = 5000
            metrics['Shapiro_n_total'] = len(residuals)
        else:
            shapiro_stat, shapiro_p = shapiro(residuals)
            metrics['Shapiro_statistic'] = float(shapiro_stat)
            metrics['Shapiro_pvalue'] = float(shapiro_p)
            metrics['Shapiro_n_sample'] = len(residuals)
            metrics['Shapiro_n_total'] = len(residuals)

        # Optional downstream experiment
        if args.downstream_experiment:
            # Calculate threshold from training data (deploy-like, leakage-safe)
            threshold_train = float(np.percentile(y_train_raw.flatten(), args.downstream_label_percentile))
            ds = downstream_experiment(
                y_true_missing, y_pred_missing, missing_pct=args.downstream_missing_pct,
                recall_target=args.downstream_recall_target, label_percentile=args.downstream_label_percentile,
                threshold=threshold_train, seed=args.seed
            )
            metrics['downstream'] = ds

        # Temporal comparison plot
        generate_temporal_comparison_plot_cdae(
            model, df_missing, features, target_var, scaler_X, scaler_y, device, sub_output, train_feature_means
        )
        
        # Plot focused on missing values
        missing_indices = df_missing.index.values[miss_mask]
        plot_temporal_missing_focus(
            y_true_missing, y_pred_missing, missing_indices, args.mechanism, args.missing_rate, target_var, sub_output
        )
        
        # Multi-panel results plot
        plot_controlled_results(
            history, y_true_missing, y_pred_missing, args.mechanism, args.missing_rate, sub_output
        )

        # Save artifacts
        np.savez(f"{sub_output}/splits_indices.npz",
                 train_idx=np.array(train_idx_model), val_idx=np.array(val_idx_model), test_idx=np.array(test_idx_model))
        train_feature_means.to_json(f"{sub_output}/train_feature_means.json")
        joblib.dump(scaler_X, f"{sub_output}/scaler_X.joblib")
        joblib.dump(scaler_y, f"{sub_output}/scaler_y.joblib")

        # Save results
        config = {
            'mechanism': args.mechanism,
            'missing_rate': float(args.missing_rate),
            'target_variable': target_var,
            'features': features,
            'batch_size': args.batch,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'patience': args.patience,
            'latent_dim': args.latent,
            'dropout': args.dropout,
            'p_mask': float(p_mask_value),
            'masked_weight': float(args.masked_weight),
            'seed': args.seed,
            'device': device,
            'split_mode': coverage_report['split_mode'],
        }
        save_results(metrics, config, eda_info, sub_output)

        all_run_metrics.append(metrics)

        print("\nCONTROLLED CDAE V2 RESULTS")
        print("=" * 45)
        print(f"Mechanism: {args.mechanism}")
        print(f"Missing Rate: {args.missing_rate*100:.1f}%")
        print(f"Missing Target Values: {metrics['n_samples']}")
        print(f"MAE:  {metrics['MAE']:.4f} [{metrics['MAE_CI_low']:.4f}, {metrics['MAE_CI_high']:.4f}]")
        print(f"RMSE: {metrics['RMSE']:.4f} [{metrics['RMSE_CI_low']:.4f}, {metrics['RMSE_CI_high']:.4f}]")
        print(f"R2:   {metrics['R2']:.4f}")
        print(f"KS:   {metrics['KS_statistic']:.4f} (p={metrics['KS_pvalue']:.6f})")
        if 'Shapiro_statistic' in metrics:
            print(f"Shapiro-Wilk: {metrics['Shapiro_statistic']:.4f} (p={metrics['Shapiro_pvalue']:.6f}, n={metrics['Shapiro_n_sample']})")

    # If stress mode, persist aggregated metrics
    if len(all_run_metrics) > 1:
        out_json = out_dir / "metrics_stress.json"
        with out_json.open("w") as f:
            json.dump(all_run_metrics, f, indent=2)
        print(f" Aggregated stress metrics saved to: {out_json}")

    print("\nCONTROLLED CDAE V2 TEST COMPLETED!")
    print(f" All results saved in: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled CDAE V2 validation (MCAR/MAR/MNAR) with all improvements from CDAE.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data", required=True, help="Path to .h5/.csv file with industrial data")
    parser.add_argument("--feature_config", help="Path to feature_selection_config.json")
    parser.add_argument("--mechanism", choices=['MCAR', 'MAR', 'MNAR'], default='MCAR', help='Missing data mechanism')
    parser.add_argument("--missing_rate", type=float, default=0.4, help="Missing data rate (0.0 to 1.0)")
    parser.add_argument("--aux_col", help="Auxiliary column for MAR mechanism")

    parser.add_argument("--batch", type=int, default=64, help="Batch size for training")
    parser.add_argument("--latent", type=int, default=16, help="Latent space dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--robust_target", action="store_true", help="Use RobustScaler for target variable")

    # Temporal split and coverage options (PORTADO DO CDAE.PY)
    parser.add_argument("--time_column", type=str, default="", help="Column name for temporal split")
    parser.add_argument("--time_test_size", type=float, default=0.3, help="Fraction for test at end of timeline")
    parser.add_argument("--time_val_size", type=float, default=0.5, help="Fraction for validation")
    parser.add_argument("--blocked_by_index", action="store_true", help="Use row order as time for splits")
    parser.add_argument("--min_target_coverage", type=float, default=0.2, help="Min fraction of present target in TRAIN")
    parser.add_argument("--min_target_count", type=int, default=5000, help="Min number of present target samples in TRAIN")
    parser.add_argument("--strict_temporal", action="store_true", help="Abort if TRAIN coverage below thresholds")
    parser.add_argument("--time_skip_head_frac", type=float, default=0.0, help="Fraction to skip at beginning")
    parser.add_argument("--auto_skip_until_coverage", action="store_true", help="Auto-adjust skip to meet coverage")
    parser.add_argument("--max_skip_head_frac", type=float, default=0.5, help="Max fraction to skip for auto-skipping")

    # CDAE corruption and weighting (PORTADO DO CDAE.PY)
    parser.add_argument("--p_mask", type=float, default=0.5, help="Probability of masking target during training")
    parser.add_argument("--masked_weight", type=float, default=1.0, help="Extra weight for masked samples in loss")
    parser.add_argument("--stress_mask_probs", type=float, nargs='*', help="Run multiple trainings for these p_mask values")

    # Bootstrap and downstream (PORTADO DO CDAE.PY)
    parser.add_argument("--bootstrap_n", type=int, default=1000, help="Number of bootstrap resamples for CIs")
    parser.add_argument("--downstream_experiment", action="store_true", help="Run downstream alert experiment")
    parser.add_argument("--downstream_missing_pct", type=float, default=0.3, help="Fraction of targets set as missing in downstream")
    parser.add_argument("--downstream_recall_target", type=float, default=0.8, help="Recall target for precision@recall")
    parser.add_argument("--downstream_label_percentile", type=float, default=80.0, help="Percentile to define positive class")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--cpu", action="store_true", help="Force use of CPU")
    parser.add_argument("--output", default="outputs", help="Directory to save results")

    args = parser.parse_args()
    main(args)

