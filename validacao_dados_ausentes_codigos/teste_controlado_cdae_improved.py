#!/usr/bin/env python3
"""
Teste_controlado_cdae_improved.py
Controlled validation for Conditional Denoising Autoencoder (CDAE) to reconstruct
missing target values under MCAR/MAR/MNAR with leakage-safe protocol.

This script mirrors the structure of teste_controlado_dados_faltantes_improved.py,
but the architecture is a CDAE compatible with cs_regressor_with_eda.py:
- Input to the network: [X; t_tilde; m_t]
- Output: t_hat (reconstructed target)

Author: Romulo Brito da Silva - UFRJ
Date: 2025-08-09
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ks_2samp, shapiro
import matplotlib.pyplot as plt

# Local import helpers
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse utilities and injection from the base script
from teste_controlado_dados_faltantes_improved import (
    load_feature_config,
    set_seed,
    timestamp,
    inject_missing_data,
    bootstrap_ci,
    save_results,
)

# ---------------------------- Dataset helpers ---------------------------- #

class ProcessDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------- CDAE model -------------------------------- #

class ConditionalDenoisingAE(nn.Module):
    """Conditional denoising AE for target t. Input [X; t_tilde; m_t] -> t_hat."""
    def __init__(self, in_x: int, latent: int = 32, dropout: float = 0.2):
        super().__init__()
        in_dim = in_x + 2
        h1, h2 = max(64, 8 * in_dim), max(32, 4 * in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.LeakyReLU(0.1), nn.Dropout(dropout/2),
            nn.Linear(h2, latent), nn.BatchNorm1d(latent), nn.LeakyReLU(0.1),
            nn.Linear(latent, h2), nn.LeakyReLU(0.1),
            nn.Linear(h2, 1)
        )
    def forward(self, x: torch.Tensor, t_tilde: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t_tilde, m_t], dim=1))


def train_cdae(model: nn.Module, loaders: Dict[str, DataLoader], epochs: int, lr: float,
               patience: int, device: str, output_dir: Path, p_mask: float = 0.5,
               masked_weight: float = 1.0) -> Dict[str, List[float]]:
    """Train CDAE with random masking on t and weighted MSE."""
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=10, verbose=True)
    history = {'train_loss': [], 'val_loss': []}
    # Improved early stopping state
    no_improve = 0
    best_val = float('inf')
    best_state = None

    # Fixed generator for validation masking (stabilizes val loss)
    g_val = torch.Generator(device=device).manual_seed(12345)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        for Xb, yb in loaders['train']:
            Xb, yb = Xb.to(device), yb.to(device)
            m_t = (torch.rand_like(yb) < p_mask).float()
            t_tilde = (1.0 - m_t) * yb
            optimizer.zero_grad()
            y_hat = model(Xb, t_tilde, m_t)
            per_elem = criterion(y_hat, yb)
            weights = 1.0 + masked_weight * m_t
            loss = (per_elem * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item(); n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for Xb, yb in loaders['val']:
                Xb, yb = Xb.to(device), yb.to(device)
                # stable validation masking
                m_t = (torch.rand(yb.shape, device=device, generator=g_val) < p_mask).float()
                t_tilde = (1.0 - m_t) * yb
                y_hat = model(Xb, t_tilde, m_t)
                per_elem = criterion(y_hat, yb)
                weights = 1.0 + masked_weight * m_t
                loss = (per_elem * weights).mean()
                val_loss += loss.item(); n_val += 1
        val_loss /= max(1, n_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        # Improved early stopping logic
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, output_dir / 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        if no_improve >= patience:
            print(f" Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def cdae_infer(model: nn.Module, X_scaled: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X = torch.as_tensor(X_scaled, dtype=torch.float32).to(device)
        t0 = torch.zeros((len(X_scaled), 1), dtype=torch.float32).to(device)
        m1 = torch.ones((len(X_scaled), 1), dtype=torch.float32).to(device)
        y_hat = model(X, t0, m1).cpu().numpy().flatten()
    return y_hat


def temporal_plot_cdae(df: pd.DataFrame, y_pred: np.ndarray, target_var: str, output_dir: Path):
    plt.figure(figsize=(16, 9))
    idx = np.arange(len(df))
    present = ~df[target_var].isna()
    plt.scatter(idx[present], df[target_var].values[present], s=8, c='blue', label='Real Values', alpha=0.7)
    plt.scatter(idx[present], y_pred[present], s=8, c='green', label='Predictions (Present Data)', alpha=0.5)
    plt.scatter(idx[~present], y_pred[~present], s=16, c='red', label='Predictions (Missing Data)', alpha=0.8)
    plt.xlabel('Sample Index'); plt.ylabel(f'Value of {target_var}')
    plt.title('Comparison between Real and Predicted Values (CDAE)')
    plt.legend(); plt.grid(True, alpha=0.3)
    path = output_dir / 'temporal_comparison_cdae.png'
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f" Temporal plot saved to: {path}")


def plot_temporal_missing_focus(y_true_missing: np.ndarray, y_pred_missing: np.ndarray,
                                missing_indices: np.ndarray, mechanism: str, rate: float,
                                target_name: str, output_dir: Path):
    plt.figure(figsize=(16, 9))
    plt.scatter(missing_indices, y_true_missing, color='blue', s=18, label='Real Value (hole)', alpha=0.8)
    plt.scatter(missing_indices, y_pred_missing, color='red', s=18, label='Reconstructed', alpha=0.8)
    plt.title(f'{mechanism} - {rate*100:.0f}% missing data (Missing Values Focus)')
    plt.xlabel('Sample Index'); plt.ylabel(f'Value of {target_name}')
    plt.legend(); plt.grid(True, alpha=0.3)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true_missing, y_pred_missing)
    rmse = float(np.sqrt(mean_squared_error(y_true_missing, y_pred_missing)))
    r2 = r2_score(y_true_missing, y_pred_missing)
    plt.text(0.01, 0.98, f'Missing: {len(y_true_missing)} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}',
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    path = output_dir / f'{mechanism}_{int(rate*100)}_temporal_missing_focus.png'
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f" Temporal missing focus plot saved to: {path}")


def plot_controlled_results(history: Dict[str, list], df_original: pd.DataFrame,
                            df_imputed: pd.DataFrame, mechanism: str, rate: float,
                            output_dir: Path):
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Training history
    axes[0, 0].plot(history['train_loss'], label='Training', alpha=0.8, linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.8, linewidth=2)
    axes[0, 0].set_title('Training History', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    # Distribution comparison
    y_true = df_original.values.ravel(); y_pred = df_imputed.values.ravel()
    axes[0, 1].hist(y_true, bins=50, alpha=0.7, label='Original', density=True, color='blue')
    axes[0, 1].hist(y_pred, bins=50, alpha=0.7, label='Reconstructed', density=True, color='red')
    axes[0, 1].set_xlabel('Values'); axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution Comparison', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    # Scatter: Original vs Reconstructed
    axes[0, 2].scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors='k', linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0, 2].plot(lims, lims, 'r--', linewidth=2, alpha=0.8)
    axes[0, 2].set_xlabel('Original Values'); axes[0, 2].set_ylabel('Reconstructed Values')
    axes[0, 2].set_title('Original vs Reconstructed', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    # Residuals
    residuals = y_pred - y_true
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Reconstructed Values'); axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Analysis', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    # Residual histogram
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals'); axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Residuals Distribution', fontweight='bold'); axes[1, 1].grid(True, alpha=0.3)
    # Summary
    missing_mask = df_original.isna()
    missing_pct = (missing_mask.sum().sum() / missing_mask.size) * 100
    axes[1, 2].text(0.1, 0.8, 'CONTROLLED TEST SUMMARY:', fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
    summary_text = f"""
Mechanism: {mechanism}
Missing Rate: {rate*100:.1f}%
Total Variables: {df_original.shape[1]}
Total Samples: {df_original.shape[0]}
Missing Values: {missing_mask.sum().sum():,}
Missing Percentage: {missing_pct:.2f}%

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
    """
    axes[1, 2].text(0.1, 0.7, summary_text.strip(), fontsize=10, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_xlim(0, 1); axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Test Summary', fontweight='bold')
    plt.suptitle(f'Controlled Missing Data Test: {mechanism} - {rate*100:.1f}% Missing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = output_dir / 'controlled_test_results.png'
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f" Results visualization saved to: {path}")


# ---------------------------- Main -------------------------------------- #

def main(args):
    print(" CONTROLLED MISSING DATA EXPERIMENT (CDAE)")
    print("=" * 55)

    set_seed(args.seed)
    out_dir = Path(args.output) / f"controlled_cdae_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.data}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    if data_path.suffix in {'.h5', '.hdf5'}:
        df = pd.read_hdf(data_path)
    else:
        df = pd.read_csv(data_path)

    # Load features (from exported JSON) or default
    target_var, features = load_feature_config(args.feature_config)
    vars_needed = [target_var] + features
    missing_vars = [v for v in vars_needed if v not in df.columns]
    if missing_vars:
        print(f" Missing variables: {missing_vars}"); return

    # Keep only rows with target present for model training
    df_target = df[vars_needed].dropna(subset=[target_var]).copy()

    # Split indices (leakage-safe) on rows with target present
    idx_all = df_target.index.values
    train_idx, temp_idx = train_test_split(idx_all, test_size=0.3, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)

    # Prepare feature matrices with train-only imputation
    X_all = df_target[features].apply(pd.to_numeric, errors='coerce')
    y_all = df_target[[target_var]]

    train_means = X_all.loc[train_idx].mean()
    if train_means.isna().any():
        bad = train_means[train_means.isna()].index.tolist()
        raise ValueError(f"No training data to compute mean for: {bad}")

    X_train_raw = X_all.loc[train_idx].fillna(train_means).values.astype(np.float32)
    X_val_raw   = X_all.loc[val_idx].fillna(train_means).values.astype(np.float32)
    X_test_raw  = X_all.loc[test_idx].fillna(train_means).values.astype(np.float32)

    scaler_X = StandardScaler(); scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw).astype(np.float32)
    X_val   = scaler_X.transform(X_val_raw).astype(np.float32)
    X_test  = scaler_X.transform(X_test_raw).astype(np.float32)

    y_train = scaler_y.fit_transform(y_all.loc[train_idx].values).astype(np.float32)
    y_val   = scaler_y.transform(y_all.loc[val_idx].values).astype(np.float32)
    y_test  = scaler_y.transform(y_all.loc[test_idx].values).astype(np.float32)

    train_loader = DataLoader(ProcessDataset(X_train, y_train), batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader   = DataLoader(ProcessDataset(X_val, y_val), batch_size=args.batch, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = ConditionalDenoisingAE(in_x=len(features), latent=args.latent_dim, dropout=args.dropout).to(device)

    print(f"Model: CDAE | features={len(features)} | latent={args.latent_dim} | device={device}")

    # Train CDAE (Phase 1 equivalent)
    history = train_cdae(model, {'train': train_loader, 'val': val_loader},
                         epochs=args.epochs, lr=args.lr, patience=args.patience,
                         device=device, output_dir=out_dir, p_mask=args.p_mask,
                         masked_weight=args.masked_weight)

    # Save model
    torch.save(model.state_dict(), out_dir / 'best_model.pth')

    # Phase 2: Controlled validation on injected-missing TEST target
    # Build test frame for injection using original scale of y
    df_test = pd.DataFrame(X_all.loc[test_idx].fillna(train_means).values, columns=features, index=test_idx)
    df_test[target_var] = y_all.loc[test_idx].values.flatten()

    aux_col = args.aux_col
    if args.mechanism == 'MAR' and aux_col is None:
        aux_col = features[0]
        print(f"Auto-selected auxiliary column for MAR: {aux_col}")

    df_missing = inject_missing_data(df_test, args.mechanism, args.rate, target_var, aux_col, args.seed)

    # Reconstruct only where target is missing
    X_test_infer = df_missing[features].fillna(train_means)
    X_test_scaled = scaler_X.transform(X_test_infer)
    y_pred_scaled = cdae_infer(model, X_test_scaled, device)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    miss_mask = df_missing[target_var].isna().values
    if miss_mask.sum() == 0:
        print(" No missing target values found for evaluation!"); return

    y_true_missing = df_test[target_var].values[miss_mask]
    y_pred_missing = y_pred[miss_mask]

    # Metrics on missing points
    metrics = {
        'MAE': mean_absolute_error(y_true_missing, y_pred_missing),
        'RMSE': float(np.sqrt(mean_squared_error(y_true_missing, y_pred_missing))),
        'R2': r2_score(y_true_missing, y_pred_missing),
        'n_samples': int(miss_mask.sum())
    }
    ks_stat, ks_p = ks_2samp(y_true_missing, y_pred_missing)
    res = y_pred_missing - y_true_missing
    sh_stat, sh_p = shapiro(res[:min(5000, len(res))])

    mae_b, mae_lo, mae_hi = bootstrap_ci(mean_absolute_error, y_true_missing, y_pred_missing, n_boot=1000, ci=0.95)
    rmse_b, rmse_lo, rmse_hi = bootstrap_ci(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
                                        y_true_missing, y_pred_missing, n_boot=1000, ci=0.95)
    metrics.update({
        'KS_statistic': ks_stat, 'KS_pvalue': ks_p,
        'Shapiro_statistic': sh_stat, 'Shapiro_pvalue': sh_p,
        'MAE_bootstrap': mae_b, 'MAE_CI_low': mae_lo, 'MAE_CI_high': mae_hi,
        'RMSE_bootstrap': rmse_b, 'RMSE_CI_low': rmse_lo, 'RMSE_CI_high': rmse_hi
    })

    print("\nCONTROLLED CDAE RESULTS")
    print("=" * 35)
    print(f"Mechanism: {args.mechanism}")
    print(f"Missing Rate: {args.rate*100:.1f}%")
    print(f"Missing Target Values: {metrics['n_samples']}")
    print(f"MAE:  {metrics['MAE']:.4f} [{metrics['MAE_CI_low']:.4f}, {metrics['MAE_CI_high']:.4f}]")
    print(f"RMSE: {metrics['RMSE']:.4f} [{metrics['RMSE_CI_low']:.4f}, {metrics['RMSE_CI_high']:.4f}]")
    print(f"R2:   {metrics['R2']:.4f}")
    print(f"KS:   {metrics['KS_statistic']:.4f} (p={metrics['KS_pvalue']:.6f})")
    print(f"Shapiro: {metrics['Shapiro_statistic']:.4f} (p={metrics['Shapiro_pvalue']:.6f})")

    # Save results JSON
    cfg = {
        'mechanism': args.mechanism,
        'missing_rate': args.rate,
        'target_variable': target_var,
        'features': features,
        'auxiliary_column': aux_col,
        'batch_size': args.batch,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'patience': args.patience,
        'latent_dim': args.latent_dim,
        'dropout': args.dropout,
        'p_mask': args.p_mask,
        'masked_weight': args.masked_weight,
        'seed': args.seed,
        'device': device,
        'n_features': len(features),
        'n_samples_train': int(len(train_idx)),
        'n_samples_test': int(len(test_idx)),
        'total_missing_injected': int(df_missing[target_var].isna().sum())
    }
    save_results(metrics, cfg, str(out_dir))

    # Persist splits and scalers for reproducibility
    import joblib
    np.savez(out_dir / 'splits_indices.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    joblib.dump(scaler_X, out_dir / 'scaler_X.joblib')
    joblib.dump(scaler_y, out_dir / 'scaler_y.joblib')

    # Multi-panel results (like original script)
    plot_controlled_results(
        history,
        pd.DataFrame({'target': y_true_missing}),
        pd.DataFrame({'target': y_pred_missing}),
        args.mechanism, args.rate, out_dir
    )

    # Temporal plots
    temporal_plot_cdae(df_missing, y_pred, target_var, out_dir)
    missing_indices = np.where(miss_mask)[0]
    plot_temporal_missing_focus(y_true_missing, y_pred_missing, missing_indices, args.mechanism, args.rate, target_var, out_dir)

    print("\n CONTROLLED CDAE TEST COMPLETED!")
    print(f"All results saved in: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled CDAE validation (MCAR/MAR/MNAR) with leakage-safe protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', required=True, help='Path to .h5/.csv file with industrial data')
    parser.add_argument('--feature_config', help='Path to feature_selection_config.json')
    parser.add_argument('--mechanism', choices=['MCAR', 'MAR', 'MNAR'], default='MCAR', help='Missing data mechanism')
    parser.add_argument('--rate', type=float, default=0.4, help='Missing data rate (0.0 to 1.0)')
    parser.add_argument('--aux_col', help='Auxiliary column for MAR mechanism (optional)')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent size for CDAE')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--p_mask', type=float, default=0.5, help='Mask probability for CDAE training')
    parser.add_argument('--masked_weight', type=float, default=1.0, help='Extra weight for masked samples in loss')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--output', default='outputs', help='Output dir')

    args = parser.parse_args()
    main(args) 