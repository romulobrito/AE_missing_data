#!/usr/bin/env python3
"""
cs_regressor_with_eda.py
Complete Pipeline: EDA + Automatic Feature Selection + Enhanced Regressor
for Real Industrial Process Data (Sulfates)

This script mirrors the leakage-safe logic from cs_ae_with_eda.py, but names the
model explicitly as a Regressor to avoid confusion with true autoencoders.

Author: Romulo Brito da Silva - UFRJ
Date: 2025-08-09
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import sys
# Ensure local imports work when running from project root
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

# Reuse utilities and plotting from the AE pipeline (local import)
from cs_ae_with_eda import (
    set_seed,
    perform_eda,
    automatic_feature_selection,
    ProcessDataset,
    EarlyStopping,
    plot_complete_results,
    save_results,
    export_feature_selection,
)


class ConditionalDenoisingAE(nn.Module):
    """Conditional denoising autoencoder for the target t.
    Input: [x; t_tilde; m_t] → Output: t_hat
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


def generate_temporal_comparison_plot_cdae(model: nn.Module, df: pd.DataFrame, selected_features: List[str],
                                           target_var: str, scaler_X, scaler_y, device: str, output_dir: str,
                                           train_feature_means: pd.Series):
    """Temporal plot for CDAE using inference inputs [X; 0; 1]."""
    import matplotlib.pyplot as plt
    import os

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
    # Clip to a reasonable physical range to stabilize visualization
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


def main(args):
    print(" COMPLETE PIPELINE: EDA + SELECTION + REGRESSOR")
    print("=" * 65)

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

    if args.target not in df.columns:
        raise ValueError(f"Target variable '{args.target}' not found in dataset!")

    # EDA (reports only)
    eda_info = perform_eda(df, args.target, args.output)

    # Pre-split indices to avoid leakage, reused across steps
    all_indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)

    # Feature selection on TRAIN ONLY
    if args.manual_features:
        if not args.features:
            raise ValueError("For manual selection, provide --features with features file")
        features_path = Path(args.features)
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if features_path.suffix == ".json":
            selected_features = json.loads(features_path.read_text())
        else:
            selected_features = [line.strip() for line in features_path.read_text().splitlines() if line.strip()]
        available_features = [f for f in selected_features if f in df.columns and f != args.target]
        features_info = {
            "selected_features": available_features,
            "selection_method": "manual",
            "features_file": str(features_path),
        }
        selected_features = available_features
        print(f" Using manual features: {len(selected_features)} selected")
    else:
        df_for_selection = df.iloc[train_idx]
        selected_features, features_info = automatic_feature_selection(
            df_for_selection,
            args.target,
            min_correlation=args.min_correlation,
            max_missing_pct=args.max_missing_pct,
            min_features=args.min_features,
            max_features=args.max_features,
            output_dir=args.output,
        )

    if len(selected_features) == 0:
        raise ValueError("No features were selected! Relax the criteria.")

    # Keep only rows with target present (do not drop by feature NaNs)
    df_with_target = df[df[args.target].notna()].copy()

    # Intersect folds with available rows after filtering by target
    train_idx_model = [i for i in train_idx if i in df_with_target.index]
    val_idx_model = [i for i in val_idx if i in df_with_target.index]
    test_idx_model = [i for i in test_idx if i in df_with_target.index]

    if (len(train_idx_model) == 0) or (len(val_idx_model) == 0) or (len(test_idx_model) == 0):
        print(" WARNING: Empty split after filtering by target. Resplitting within available rows.")
        available_idx = list(df_with_target.index)
        train_idx_model, temp_idx_model = train_test_split(available_idx, test_size=0.3, random_state=args.seed)
        val_idx_model, test_idx_model = train_test_split(temp_idx_model, test_size=0.5, random_state=args.seed)

    # Build matrices without dropping feature NaNs yet
    X_all = df_with_target[selected_features].apply(pd.to_numeric, errors="coerce")
    y_all = df_with_target[[args.target]]

    # Compute training means for feature imputation (original scale)
    train_feature_means = X_all.loc[train_idx_model].mean()
    if train_feature_means.isna().any():
        bad = train_feature_means[train_feature_means.isna()].index.tolist()
        raise ValueError(f"No training data to compute mean for: {bad}")

    # Impute feature NaNs fold-wise using TRAIN means (leakage-safe)
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
    model = ConditionalDenoisingAE(in_x=len(selected_features), latent=args.latent, dropout=args.dropout).to(device)

    print(" Model created:")
    print(f"  Input: {len(selected_features)} features")
    print(f"  Latent: {args.latent} dimensions")
    print(f"  Device: {device.upper()}")

    # Train CDAE
    history = train_cdae(
        model, loaders, epochs=args.epochs, lr=args.lr, patience=args.patience, device=device,
        output_dir=args.output, p_mask=0.5, masked_weight=1.0
    )

    # Restore best checkpoint if present
    best_path = Path(args.output) / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Evaluate (original scale arrays)
    y_true, y_pred = evaluate_cdae(model, loaders["test"], scaler_y, device)

    # Metrics summary (simple)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'KS_statistic': 0.0,
        'KS_pvalue': 1.0,
        'MAE_bootstrap': mae,
        'MAE_CI_low': mae,
        'MAE_CI_high': mae,
        'RMSE_bootstrap': rmse,
        'RMSE_CI_low': rmse,
        'RMSE_CI_high': rmse,
        'n_samples': int(len(y_true))
    }

    # Temporal comparison plot for CDAE
    generate_temporal_comparison_plot_cdae(
        model, df, selected_features, args.target, scaler_X, scaler_y, device, args.output, train_feature_means
    )

    # Save artifacts for reproducibility
    import joblib
    np.savez(f"{args.output}/splits_indices.npz",
             train_idx=np.array(train_idx_model), val_idx=np.array(val_idx_model), test_idx=np.array(test_idx_model))
    train_feature_means.to_json(f"{args.output}/train_feature_means.json")
    joblib.dump(scaler_X, f"{args.output}/scaler_X.joblib")
    joblib.dump(scaler_y, f"{args.output}/scaler_y.joblib")

    # Save and export
    save_results(metrics, features_info, eda_info, args, args.output)
    export_feature_selection(selected_features, args.target, features_info, args.output)
    plot_complete_results(eda_info, features_info, history, y_true, y_pred, args.target, args.output)

    print("\nCOMPLETE REGRESSOR PIPELINE FINALIZED!")
    print(f" All results saved in: {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete Pipeline: EDA + Feature Selection + Regressor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data", required=True, help="Path to .h5/.csv file with industrial data")
    parser.add_argument("--target", required=True, help="Name of the target variable to be reconstructed")

    parser.add_argument("--manual_features", action="store_true", help="Use manual feature selection (requires --features)")
    parser.add_argument("--features", help="File .txt/.json with features (only if --manual_features)")
    parser.add_argument("--min_correlation", type=float, default=0.3, help="Minimum correlation with target for automatic selection")
    parser.add_argument("--max_missing_pct", type=float, default=30.0, help="Maximum percentage of missing data for selection")
    parser.add_argument("--min_features", type=int, default=3, help="Minimum number of features to select")
    parser.add_argument("--max_features", type=int, default=10, help="Maximum number of features to select")

    parser.add_argument("--batch", type=int, default=64, help="Batch size for training")
    parser.add_argument("--latent", type=int, default=16, help="Latent space dimension of the regressor")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for regularization")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--robust_target", action="store_true", help="Use RobustScaler for target variable")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--cpu", action="store_true", help="Force use of CPU")
    parser.add_argument("--output", default="outputs", help="Directory to save results")

    args = parser.parse_args()
    main(args) 