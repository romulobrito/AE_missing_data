#!/usr/bin/env python3
"""
cs_ae_enhanced.py ───────────────────────────────────────────────────────────────
Pipeline Enhanced to Imputation of Missing Data using Autoencoder
in Real Industrial Process Data (Sulfates)

Based on the analysis of the notebook, this script implements significant improvements:
• Complete control of reproducibility
• More robust autoencoder architecture with BatchNorm and Dropout
• Early-Stopping and ReduceLROnPlateau for stable training
• Advanced metrics: MAE/RMSE des-escalonadas + KS + Bootstrap CI 95%
• Flexible CLI interface for experimentation
• Integration with industrial process chemical data

Author: Romulo Brito da Silva - UFRJ
Data: 2025-08-07
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Sequence, Dict, Tuple, Optional
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ╭───────────────────────────── Utilities ─────────────────────────────╮

def set_seed(seed: int = 42) -> None:
    """Define seeds completas para reprodutibilidade total."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Seed defined: {seed}")


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot: int = 1000, ci: float = 0.95):
    """Non-parametric bootstrap estimate for confidence interval."""
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.quantile(stats, (1-ci)/2)
    upper = np.quantile(stats, 1 - (1-ci)/2)
    return np.mean(stats), (lower, upper)


def save_results(metrics: Dict, args, output_dir: str = "outputs"):
    """Save results and experiment configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar métricas
    results = {
        'metrics': metrics,
        'config': vars(args),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(f"{output_dir}/experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_dir}/experiment_results.json")

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Dataset & DataLoader ───────────────────────╮

class ProcessDataset(Dataset):
    """Custom dataset for industrial process data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Modelo Autoencoder ───────────────────────╮

class EnhancedAutoencoder(nn.Module):
    """
    Autoencoder for reconstruction of industrial process variables.
    """
    
    def __init__(self, in_features: int, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        
        # Intermediate dimensions based on input size
        hidden1 = max(64, in_features * 8)
        hidden2 = max(32, in_features * 4)
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            
            nn.Linear(hidden2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout/2),  # Less dropout in decoder
            
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout/2),
            
            nn.Linear(hidden1, 1),  # Output for a target variable
        )
        
        # Xavier/Glorot initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Improved weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Early Stopping ─────────────────────────────╮

class EarlyStopping:
    """Implementation of Early Stopping with configurable patience."""
    
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

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Treinamento & Avaliação ────────────────────╮

def train_model(model: nn.Module, loaders: Dict[str, DataLoader], 
                epochs: int, lr: float, patience: int, device: str,
                output_dir: str = "outputs") -> Dict[str, list]:
    """
    Trains the autoencoder model with early stopping and scheduler.
    
    Returns:
        Dict with training and validation loss history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    print(f" Starting training on {device.upper()}")
    print(f"Parâmetros: epochs={epochs}, lr={lr}, patience={patience}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for X_batch, y_batch in tqdm(loaders['train'], desc=f"Epoch {epoch+1}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loaders['val']:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.stop:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
    print(f" Model saved to: {output_dir}/best_model.pth")
    
    return history


def evaluate_model(model: nn.Module, loader: DataLoader, scaler_y, 
                  device: str, target_name: str = "target") -> Dict:
    """
    Evaluates the model with advanced metrics.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    predictions, targets = [], []
    
    print(" Evaluating model...")
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Avaliação"):
            X_batch = X_batch.to(device)
            pred_batch = model(X_batch).cpu().numpy()
            predictions.append(pred_batch)
            targets.append(y_batch.numpy())
    
    # Concatenate results
    y_pred_scaled = np.concatenate(predictions)
    y_true_scaled = np.concatenate(targets)
    
    # Des-escalate to original values
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    y_true_original = scaler_y.inverse_transform(y_true_scaled)
    
    # Flatten for metrics
    y_pred_flat = y_pred_original.ravel()
    y_true_flat = y_true_original.ravel()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Kolmogorov-Smirnov test (distribution fit)
    ks_stat, ks_pvalue = ks_2samp(y_true_flat, y_pred_flat)
    
    # Bootstrap CI for MAE
    mae_bootstrap, (mae_ci_low, mae_ci_high) = bootstrap_ci(
        mean_absolute_error, y_true_flat, y_pred_flat, n_boot=1000
    )
    
    # Bootstrap CI for RMSE
    rmse_bootstrap, (rmse_ci_low, rmse_ci_high) = bootstrap_ci(
        lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
        y_true_flat, y_pred_flat, n_boot=1000
    )
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'KS_statistic': ks_stat,
        'KS_pvalue': ks_pvalue,
        'MAE_bootstrap': mae_bootstrap,
        'MAE_CI_low': mae_ci_low,
        'MAE_CI_high': mae_ci_high,
        'RMSE_bootstrap': rmse_bootstrap,
        'RMSE_CI_low': rmse_ci_low,
        'RMSE_CI_high': rmse_ci_high,
        'n_samples': len(y_true_flat)
    }
    
    return metrics, y_true_flat, y_pred_flat


def plot_results(history: Dict, y_true: np.ndarray, y_pred: np.ndarray, 
                target_name: str, output_dir: str = "outputs"):
    """Generates visualizations of results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure style
    plt.style.use('seaborn-v0_8')
    
    # Plot 1: Training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss history
    axes[0, 0].plot(history['train_loss'], label='Training', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.8)
    axes[0, 0].set_title('Training Loss History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot: Actual vs Predicted
    axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0, 1].plot(lims, lims, 'r--', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title(f'Actual vs Predicted - {target_name}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Analysis')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Residuals Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Plots saved to: {output_dir}/training_evaluation_results.png")

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────────── Main ─────────────────────────────────╮

def load_industrial_data(data_path: str, target: str, features_file: Optional[str] = None) -> pd.DataFrame:
    """Loads industrial data with specific configurations for sulfates."""
    print(f" Loading data from: {data_path}")
    
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    if path.suffix in {'.h5', '.hdf5'}:
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path)
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} variables")
    
    # Load specific features if provided
    if features_file:
        features_path = Path(features_file)
        if features_path.exists():
            if features_path.suffix == '.json':
                with open(features_path, 'r') as f:
                    features = json.load(f)
            else:
                features = [line.strip() for line in features_path.read_text().splitlines() 
                           if line.strip()]
            
            available_features = [f for f in features if f in df.columns]
            df = df[available_features + [target]]
            print(f" Selected features: {len(available_features)}")
        else:
            print(f"  Features file not found: {features_path}")
    
    # Default configuration for sulfates data 
    if target == '1251_FIT_801C_2' and features_file is None:
        default_features = [
            '1251_FIC_801C',
            '1251_PIT_806C', 
            '1251_PIT_808C',
            '1251_FIT_802C',
            '1251_PDI_807C'
        ]
        available_features = [f for f in default_features if f in df.columns]
        df = df[available_features + [target]]
        print(f" Using default configuration for {target}: {len(available_features)} features")
    
    # Missing data report
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("\ Missing data by variable:")
        for var, missing in missing_info[missing_info > 0].items():
            pct = (missing / len(df)) * 100
            print(f"  {var}: {missing} ({pct:.2f}%)")
    
    return df


def main(args):
    """Main function of the pipeline."""
    print(" ENHANCED AUTOENCODER PIPELINE FOR INDUSTRIAL DATA")
    print("=" * 65)
    
    # Configure reproducibility
    set_seed(args.seed)
    
    # Load data
    df = load_industrial_data(args.data, args.target, args.features)
    
    # Remove NAs (strategy can be adapted)
    df_clean = df.dropna()
    print(f"Data after cleaning: {df_clean.shape[0]} samples")
    
    if len(df_clean) < 100:
        raise ValueError("Insufficient data after cleaning. Consider a different strategy for NAs.")
    
    # Prepare data
    X = df_clean.drop(columns=[args.target]).values.astype(np.float32)
    y = df_clean[[args.target]].values.astype(np.float32)
    
    print(f"Final dimensions: X={X.shape}, y={y.shape}")
    
    # Normalization
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = RobustScaler() if args.robust_target else StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    print(f" Normalization: {'RobustScaler' if args.robust_target else 'StandardScaler'} for target")
    
    # Stratified split by percentiles (for continuous data)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed
    )
    
    print(f" Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples") 
    print(f"  Test: {len(X_test)} samples")
    
    # DataLoaders
    loaders = {
        'train': DataLoader(ProcessDataset(X_train, y_train), 
                           batch_size=args.batch, shuffle=True, num_workers=2),
        'val': DataLoader(ProcessDataset(X_val, y_val), 
                         batch_size=args.batch, shuffle=False, num_workers=2),
        'test': DataLoader(ProcessDataset(X_test, y_test), 
                          batch_size=args.batch, shuffle=False, num_workers=2),
    }
    
    # Model
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = EnhancedAutoencoder(
        in_features=X.shape[1], 
        latent_dim=args.latent, 
        dropout=args.dropout
    )
    
    print(f" Model created:")
    print(f"  Input: {X.shape[1]} features")
    print(f"  Latent: {args.latent} dimensions")
    print(f"  Dropout: {args.dropout}")
    print(f"  Device: {device.upper()}")
    
    # Training
    history = train_model(
        model, loaders, 
        epochs=args.epochs, 
        lr=args.lr, 
        patience=args.patience, 
        device=device,
        output_dir=args.output
    )
    
    # Evaluation
    metrics, y_true, y_pred = evaluate_model(
        model, loaders['test'], scaler_y, device, args.target
    )
    
    # Show results
    print("\nEVALUATION METRICS (Test Set)")
    print("=" * 50)
    print(f"MAE:              {metrics['MAE']:.4f}")
    print(f"RMSE:             {metrics['RMSE']:.4f}") 
    print(f"R²:               {metrics['R2']:.4f}")
    print(f"KS Statistic:     {metrics['KS_statistic']:.4f}")
    print(f"KS p-value:       {metrics['KS_pvalue']:.4f}")
    print("\ CONFIDENCE INTERVALS (Bootstrap 95%)")
    print(f"MAE:  {metrics['MAE_bootstrap']:.4f} [{metrics['MAE_CI_low']:.4f}, {metrics['MAE_CI_high']:.4f}]")
    print(f"RMSE: {metrics['RMSE_bootstrap']:.4f} [{metrics['RMSE_CI_low']:.4f}, {metrics['RMSE_CI_high']:.4f}]")
    
    # Save results
    save_results(metrics, args, args.output)
    
    # Generate visualizations
    plot_results(history, y_true, y_pred, args.target, args.output)
    
    print(f"\ Pipeline completed successfully!")
    print(f" Results saved in: {args.output}/")

# ╰───────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Autoencoder Pipeline for Industrial Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data', required=True, 
                       help='Path to .h5/.csv file with industrial data')
    parser.add_argument('--target', required=True,
                       help='Name of the target variable to be reconstructed')
    
    # Optional arguments - Data
    parser.add_argument('--features', 
                       help='File .txt/.json with selected features')
    parser.add_argument('--robust_target', action='store_true',
                       help='Use RobustScaler for target variable (recommended for outliers)')
    
    # Optional arguments - Model
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--latent', type=int, default=16,
                       help='Latent space dimension of the autoencoder')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for regularization')
    
    # Optional arguments - Training
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate inicial')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    
    # Optional arguments - System
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for reproducibility')
    parser.add_argument('--cpu', action='store_true',
                       help='Force use of CPU even with CUDA available')
    parser.add_argument('--output', default='outputs',
                       help='Directory to save results')
    
    args = parser.parse_args()
    main(args) 