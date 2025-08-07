#!/usr/bin/env python3
"""
teste_controlado_dados_faltantes_improved.py ───────────────────────────────────
Controlled Experiment Script for Evaluating Autoencoder's Ability to Reconstruct
Artificially Generated Missing Data (MCAR/MAR/MNAR) in Real Industrial Datasets

This script implements a comprehensive controlled experiment to validate the
autoencoder's reconstruction capabilities by:
1. Using complete real data as ground truth
2. Injecting controlled missing data patterns (MCAR/MAR/MNAR)
3. Training autoencoder on complete data
4. Testing reconstruction on artificially missing data
5. Comparing original vs reconstructed values

Key Improvements:
• Complete seed control for reproducibility
• CLI interface with comprehensive parameters
• Advanced missing data injection mechanisms
• Robust evaluation metrics with bootstrap CI
• Professional visualization and reporting
• Organized output structure with timestamps

Author: Romulo Brito da Silva - UFRJ
Date: 2025-08-07
"""
from __future__ import annotations

import argparse
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ks_2samp, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ╭───────────────────────────── Utilities ─────────────────────────────╮

def load_feature_config(config_path: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Load feature configuration from JSON file exported by cs_ae_with_eda.py.
    
    Args:
        config_path: Path to feature_selection_config.json file
        
    Returns:
        Tuple of (target_variable, selected_features)
    """
    # Default configuration (fallback)
    default_target = '1251_FIT_801C_2'
    default_features = [
        '1251_FIC_801C',
        '1251_PIT_806C', 
        '1251_PIT_808C',
        '1251_FIT_802C',
        '1251_PDI_807C'
    ]
    
    if config_path is None:
        print(" Using default feature configuration (hardcoded)")
        return default_target, default_features
    
    config_file = Path(config_path)
    if not config_file.exists():
        print(f" Feature config file not found: {config_path}")
        print(" Falling back to default configuration")
        return default_target, default_features
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        target_var = config.get('target_variable', default_target)
        features = config.get('selected_features', default_features)
        selection_method = config.get('selection_method', 'unknown')
        feature_count = config.get('feature_count', len(features))
        
        print(f" Loaded feature configuration from: {config_path}")
        print(f"   Target: {target_var}")
        print(f"   Features: {feature_count} variables")
        print(f"   Selection method: {selection_method}")
        
        # Validate configuration
        if not target_var or not features:
            print(" Invalid configuration detected, using defaults")
            return default_target, default_features
        
        return target_var, features
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f" Error loading feature config: {e}")
        print(" Falling back to default configuration")
        return default_target, default_features


def set_seed(seed: int = 42) -> None:
    """Define complete seeds for total reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Seed defined: {seed}")


def timestamp() -> str:
    """Generate timestamp for output organization."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Non-parametric bootstrap estimate for confidence interval."""
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.quantile(stats, (1-ci)/2)
    upper = np.quantile(stats, 1 - (1-ci)/2)
    return float(np.mean(stats)), float(lower), float(upper)


def save_results(metrics: Dict, config: Dict, output_dir: str = "outputs"):
    """Save experiment results and configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/controlled_test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f" Results saved to: {output_dir}/controlled_test_results.json")

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Missing Data Injection ────────────────────╮

def inject_mcar(df: pd.DataFrame, rate: float, target_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject Missing Completely At Random (MCAR) data.
    
    Args:
        df: Complete DataFrame
        rate: Missing data rate (0.0 to 1.0)
        target_col: Target column to inject missing data
        rng: Random number generator
        
    Returns:
        DataFrame with MCAR missing data
    """
    data_missing = df.copy()
    n_samples = len(df)
    n_missing = int(n_samples * rate)
    
    # Missing Completely At Random
    missing_indices = rng.choice(n_samples, n_missing, replace=False)
    mask = np.zeros(n_samples, dtype=bool)
    mask[missing_indices] = True
    
    # Apply mask only to target column 
    data_missing.loc[mask, target_col] = np.nan
    
    return data_missing


def inject_mar(df: pd.DataFrame, rate: float, target_col: str, aux_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject Missing At Random (MAR) data based on auxiliary variable.
    
    Args:
        df: Complete DataFrame
        rate: Missing data rate (0.0 to 1.0)
        target_col: Target column to inject missing data
        aux_col: Auxiliary column for MAR mechanism
        rng: Random number generator
        
    Returns:
        DataFrame with MAR missing data
    """
    if aux_col not in df.columns:
        raise ValueError(f"Auxiliary column '{aux_col}' not found in dataset")
    
    data_missing = df.copy()
    n_samples = len(df)
    n_missing = int(n_samples * rate)
    
    print(f"MAR based on auxiliary variable: {aux_col}")
    
    # Calculate quartiles of auxiliary variable (not target!)
    aux_values = df[aux_col].dropna()
    quartiles = aux_values.quantile([0.25, 0.75])
    
    # Higher probability of missing in extremes of auxiliary variable
    extreme_mask = (df[aux_col] <= quartiles.iloc[0]) | (df[aux_col] >= quartiles.iloc[1])
    extreme_indices = np.where(extreme_mask & ~df[aux_col].isnull())[0]
    
    # 70% of missing when auxiliary variable is in extremes
    n_extreme = min(int(n_missing * 0.7), len(extreme_indices))
    n_random = n_missing - n_extreme
    
    # Available indices (without NaN in auxiliary variable)
    available_indices = np.where(~df[aux_col].isnull())[0]
    non_extreme_indices = [i for i in available_indices if i not in extreme_indices]
    
    missing_indices = []
    
    # Select from extremes
    if n_extreme > 0 and len(extreme_indices) > 0:
        selected_extreme = rng.choice(extreme_indices, min(n_extreme, len(extreme_indices)), replace=False)
        missing_indices.extend(selected_extreme)
    
    # Select randomly from the rest
    remaining_needed = n_missing - len(missing_indices)
    if remaining_needed > 0 and len(non_extreme_indices) > 0:
        selected_random = rng.choice(non_extreme_indices, min(remaining_needed, len(non_extreme_indices)), replace=False)
        missing_indices.extend(selected_random)
    
    # If we still need more, take from anywhere
    if len(missing_indices) < n_missing:
        all_available = [i for i in range(n_samples) if i not in missing_indices and not pd.isnull(df.iloc[i][aux_col])]
        remaining = n_missing - len(missing_indices)
        if len(all_available) >= remaining:
            additional = rng.choice(all_available, remaining, replace=False)
            missing_indices.extend(additional)
    
    missing_indices = np.array(missing_indices)
    mask = np.zeros(n_samples, dtype=bool)
    mask[missing_indices] = True
    
    # Apply mask only to target column 
    data_missing.loc[mask, target_col] = np.nan
    
    return data_missing


def inject_mnar(df: pd.DataFrame, rate: float, target_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject Missing Not At Random (MNAR) data based on target variable values.
    
    Args:
        df: Complete DataFrame
        rate: Missing data rate (0.0 to 1.0)
        target_col: Target column to inject missing data
        rng: Random number generator
        
    Returns:
        DataFrame with MNAR missing data
    """
    data_missing = df.copy()
    n_samples = len(df)
    n_missing = int(n_samples * rate)
    
    # Probability proportional to value (after normalization)
    values = df[target_col]
    values_norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    probabilities = values_norm ** 2  # Higher values have higher probability
    probabilities = probabilities / probabilities.sum()
    
    missing_indices = rng.choice(
        n_samples, 
        n_missing, 
        replace=False, 
        p=probabilities
    )
    
    mask = np.zeros(n_samples, dtype=bool)
    mask[missing_indices] = True
    
    # Apply mask only to target column 
    data_missing.loc[mask, target_col] = np.nan
    
    return data_missing


def inject_missing_data(df: pd.DataFrame, mechanism: str, rate: float, 
                       target_col: str, aux_col: Optional[str] = None, seed: int = 42) -> pd.DataFrame:
    """
    Inject missing data using specified mechanism.
    
    Args:
        df: Complete DataFrame
        mechanism: 'MCAR', 'MAR', or 'MNAR'
        rate: Missing data rate (0.0 to 1.0)
        target_col: Target column to inject missing data
        aux_col: Auxiliary column for MAR mechanism
        seed: Random seed
        
    Returns:
        DataFrame with injected missing data
    """
    rng = np.random.default_rng(seed)
    
    if mechanism == 'MCAR':
        return inject_mcar(df, rate, target_col, rng)
    elif mechanism == 'MAR':
        if aux_col is None:
            raise ValueError("Auxiliary column required for MAR mechanism")
        return inject_mar(df, rate, target_col, aux_col, rng)
    elif mechanism == 'MNAR':
        return inject_mnar(df, rate, target_col, rng)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Dataset & Model ───────────────────────────╮

class ProcessDataset(Dataset):
    """Dataset para dados de processo industrial """
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class ControlledAutoencoder(nn.Module):
    """
    Autoencoder for controlled missing data reconstruction.
    
    """
    
    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features)
        )
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Training ──────────────────────────────────╮

class EarlyStopping:
    """Early stopping implementation with configurable patience."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
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


def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int, lr: float, patience: int, device: str) -> Dict[str, list]:
    """
    Train the autoencoder model with early stopping and scheduler.
    
    Returns:
        Dict with training and validation loss history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True  
    )
    
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f" Starting training on {device.upper()}")
    print(f"Parameters: epochs={epochs}, lr={lr}, patience=15")
    
    for epoch in range(epochs):
        # Training phase 
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Combine features and targets for autoencoder input 
            combined_input = torch.cat([batch_features, batch_targets.unsqueeze(1)], dim=1)
            
            optimizer.zero_grad()
            outputs = model(combined_input)
            loss = criterion(outputs, combined_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase 
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                # Combine features and targets for autoencoder input 
                combined_input = torch.cat([batch_features, batch_targets.unsqueeze(1)], dim=1)
                
                outputs = model(combined_input)
                loss = criterion(outputs, combined_input)
                val_loss += loss.item()
        
    
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress 
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping check 
        if patience_counter >= 15:  # Early stopping
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model 
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Evaluation ────────────────────────────────╮

def reconstruct_data(model: nn.Module, data_loader: DataLoader, device: str) -> np.ndarray:
    """Reconstruct data using trained autoencoder ."""
    model.eval()
    reconstructions = []
    
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Combine features and targets for autoencoder input (like original)
            combined_input = torch.cat([batch_features, batch_targets.unsqueeze(1)], dim=1)
            
            reconstructed = model(combined_input)
            reconstructions.append(reconstructed.cpu().numpy())
    
    return np.concatenate(reconstructions, axis=0)


def reconstruct_data_original_style(model: nn.Module, X_missing: pd.DataFrame, 
                                   scaler_X: StandardScaler, scaler_y: StandardScaler,
                                   missing_mask: np.ndarray, device: str, X_train_stats: dict) -> np.ndarray:
    """
    Reconstruct data using trained autoencoder  script.
    
    Args:
        model: Trained autoencoder model
        X_missing: Features with potentially missing data
        scaler_X: Feature scaler (fitted on training set only)
        scaler_y: Target scaler (fitted on training set only)
        missing_mask: Mask of missing data
        device: Device for computation (CPU/GPU)
        X_train_stats: Statistics calculated on training set for imputation
    """
    model.eval()
    
    # Impute NaN in features using TRAINING statistics (avoid data leakage)
    X_imputed = X_missing.copy()
    for col in X_imputed.columns:
        if X_imputed[col].isnull().any():
            if X_train_stats and col in X_train_stats:
                #  Use mean calculated on training set
                fill_value = X_train_stats[col]['mean']
                print(f"Imputing {X_imputed[col].isnull().sum()} missing values in '{col}' with training mean: {fill_value:.4f}")
            else:
                # Fallback: use global median (not ideal, but better than test mean)
                fill_value = X_imputed[col].median()
                print(f"⚠️  Fallback: Imputing '{col}' with global median: {fill_value:.4f}")
            
            X_imputed[col].fillna(fill_value, inplace=True)
    
    # Scale features
    X_scaled = scaler_X.transform(X_imputed)
    
    # For data with missing target, use 0 (will be ignored in reconstruction)
    y_dummy = np.zeros(len(X_missing))
    
    # Create combined input
    input_data = np.column_stack([X_scaled, y_dummy])
    
    # Inference 
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_data).to(device)
        reconstructed = model(input_tensor).cpu().numpy()
    
    # Extract reconstructed target (last column)
    target_reconstructed_scaled = reconstructed[:, -1]
    
    # Inverse transform only the target
    target_reconstructed = scaler_y.inverse_transform(
        target_reconstructed_scaled.reshape(-1, 1)
    ).flatten()
    
    return target_reconstructed


def impute_missing_data(df_missing: pd.DataFrame, reconstructed: np.ndarray, 
                       scaler: StandardScaler) -> pd.DataFrame:
    """
    Impute missing data using reconstructed values.
    
    Args:
        df_missing: DataFrame with missing data
        reconstructed: Reconstructed values from autoencoder
        scaler: Fitted scaler for inverse transformation
        
    Returns:
        DataFrame with imputed values
    """
    # Inverse transform reconstructed values
    reconstructed_original = scaler.inverse_transform(reconstructed)
    
    # Create output DataFrame
    df_imputed = df_missing.copy()
    
    # Convert reconstructed to DataFrame for easier indexing
    reconstructed_df = pd.DataFrame(reconstructed_original, columns=df_missing.columns, index=df_missing.index)
    
    # Impute missing values
    df_imputed = df_imputed.fillna(reconstructed_df)
    
    return df_imputed


def calculate_metrics(df_original: pd.DataFrame, df_imputed: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        df_original: Original complete DataFrame
        df_imputed: DataFrame with imputed values
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Flatten arrays for metrics calculation
    y_true = df_original.values.ravel()
    y_pred = df_imputed.values.ravel()
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Statistical tests
    ks_stat, ks_pvalue = ks_2samp(y_true, y_pred)
    
    # Shapiro test for residuals (normality)
    residuals = y_pred - y_true
    shapiro_stat, shapiro_p = shapiro(residuals[:min(5000, len(residuals))])  # Limit for Shapiro
    
    # Bootstrap confidence intervals
    mae_bootstrap, mae_ci_low, mae_ci_high = bootstrap_ci(
        mean_absolute_error, y_true, y_pred, n_boot=1000
    )
    
    rmse_bootstrap, rmse_ci_low, rmse_ci_high = bootstrap_ci(
        lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
        y_true, y_pred, n_boot=1000
    )
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'KS_statistic': ks_stat,
        'KS_pvalue': ks_pvalue,
        'Shapiro_statistic': shapiro_stat,
        'Shapiro_pvalue': shapiro_p,
        'MAE_bootstrap': mae_bootstrap,
        'MAE_CI_low': mae_ci_low,
        'MAE_CI_high': mae_ci_high,
        'RMSE_bootstrap': rmse_bootstrap,
        'RMSE_CI_low': rmse_ci_low,
        'RMSE_CI_high': rmse_ci_high,
        'n_samples': len(y_true)
    }
    
    return metrics

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Visualization ─────────────────────────────╮

def plot_temporal_comparison(y_true_all: np.ndarray, y_pred_all: np.ndarray, 
                           missing_mask: np.ndarray, mechanism: str, rate: float,
                           target_name: str, output_dir: str = "outputs"):
    """
    Plot temporal comparison showing original vs reconstructed values,
    highlighting missing data regions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 9))
    idx = np.arange(len(y_true_all))
    
    # Real values (present data)
    present_mask = ~missing_mask
    plt.scatter(idx[present_mask], y_true_all[present_mask], 
               color='blue', s=8, label='Real Values', alpha=0.7)
    
    # Predictions for present data (should match real values)
    plt.scatter(idx[present_mask], y_pred_all[present_mask], 
               color='green', s=8, alpha=0.5, label='Predictions (Present Data)')
    
    # Predictions for missing data (reconstructed values)
    plt.scatter(idx[missing_mask], y_pred_all[missing_mask], 
               color='red', s=16, label='Predictions (Missing Data)', alpha=0.8)
    
    plt.title(f'{mechanism} - {rate*100:.0f}% missing data (Comparative)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel(f'Value of {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    total = len(y_true_all)
    present = present_mask.sum()
    missing = missing_mask.sum()
    
    plt.text(0.01, 0.98, 
             f'Total: {total:,} | Present: {present:,} | Missing: {missing:,}',
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f"{output_dir}/{mechanism}_{int(rate*100)}_temporal_comparative.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Temporal comparative plot saved to: {filename}")


def plot_temporal_missing_focus(y_true_missing: np.ndarray, y_pred_missing: np.ndarray,
                               missing_indices: np.ndarray, mechanism: str, rate: float,
                               target_name: str, output_dir: str = "outputs"):
    """
    Plot focusing only on missing data points (original hole values vs reconstructed).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 9))
    
    # Plot only the missing data points
    plt.scatter(missing_indices, y_true_missing, 
               color='blue', s=18, label='Real Value (hole)', alpha=0.8)
    plt.scatter(missing_indices, y_pred_missing, 
               color='red', s=18, label='Reconstructed', alpha=0.8)
    
    plt.title(f'{mechanism} - {rate*100:.0f}% missing data (Missing Values Focus)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel(f'Value of {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mae = mean_absolute_error(y_true_missing, y_pred_missing)
    rmse = np.sqrt(mean_squared_error(y_true_missing, y_pred_missing))
    r2 = r2_score(y_true_missing, y_pred_missing)
    
    plt.text(0.01, 0.98, 
             f'Missing Values: {len(y_true_missing):,} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}',
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    filename = f"{output_dir}/{mechanism}_{int(rate*100)}_temporal_missing_focus.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Temporal missing focus plot saved to: {filename}")


def plot_controlled_results(history: Dict[str, list], df_original: pd.DataFrame, 
                           df_imputed: pd.DataFrame, mechanism: str, rate: float,
                           output_dir: str = "outputs"):
    """Generate comprehensive visualizations of controlled test results."""
    os.makedirs(output_dir, exist_ok=True)
    
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
    y_true = df_original.values.ravel()
    y_pred = df_imputed.values.ravel()
    
    axes[0, 1].hist(y_true, bins=50, alpha=0.7, label='Original', density=True, color='blue')
    axes[0, 1].hist(y_pred, bins=50, alpha=0.7, label='Reconstructed', density=True, color='red')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution Comparison', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: Original vs Reconstructed
    axes[0, 2].scatter(y_true, y_pred, alpha=0.6, s=20, edgecolors='k', linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0, 2].plot(lims, lims, 'r--', linewidth=2, alpha=0.8)
    axes[0, 2].set_xlabel('Original Values')
    axes[0, 2].set_ylabel('Reconstructed Values')
    axes[0, 2].set_title('Original vs Reconstructed', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Residuals analysis
    residuals = y_pred - y_true
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('Reconstructed Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Analysis', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1, 1].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Residuals Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Missing data pattern visualization
    missing_mask = df_original.isna()
    missing_pct = (missing_mask.sum().sum() / missing_mask.size) * 100
    
    axes[1, 2].text(0.1, 0.8, f'CONTROLLED TEST SUMMARY:', fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
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
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Test Summary', fontweight='bold')
    
    plt.suptitle(f'Controlled Missing Data Test: {mechanism} - {rate*100:.1f}% Missing', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/controlled_test_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Results visualization saved to: {output_dir}/controlled_test_results.png")

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────────── Main ─────────────────────────────────╮

def main(args):
    """Main function for controlled missing data experiment."""
    print(" CONTROLLED MISSING DATA EXPERIMENT")
    print("=" * 50)
    
    # Configure reproducibility
    set_seed(args.seed)
    
    # Create output directory with timestamp
    timestamp_str = timestamp()
    output_dir = Path(args.output) / f"controlled_test_{timestamp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    print(f"Loading data from: {args.data}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    if data_path.suffix in {'.h5', '.hdf5'}:
        df = pd.read_hdf(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} variables")
    
    # Load feature configuration (from cs_ae_with_eda.py export or default)
    target_var, features = load_feature_config(args.feature_config)
    
    # Check if target and features exist
    vars_interesse = [target_var] + features
    missing_vars = [var for var in vars_interesse if var not in df.columns]
    if missing_vars:
        print(f" Missing variables in dataset: {missing_vars}")
        return
    
    # Filter complete records for specific variables
    df_complete = df[vars_interesse].dropna()
    print(f"Complete records: {len(df_complete)} ({len(df_complete)/len(df)*100:.2f}%)")
    
    if len(df_complete) < 100:
        print(" Too few complete records for reliable testing!")
        return
    
    # Apply physical filter (remove extreme outliers)
    target_values = df_complete[target_var]
    q1, q3 = target_values.quantile([0.01, 0.99])  # Use 1% and 99% percentiles
    mask_fisico = (target_values >= q1) & (target_values <= q3)
    df_complete = df_complete[mask_fisico]
    
    print(f"After physical filter: {len(df_complete)} records")
    
    # Separate features and target 
    X = df_complete[features]
    y = df_complete[target_var]
    
    print(f"Features: {len(features)} variables")
    print(f"Target: {target_var}")
    
    # Split data 
    test_size = 0.2
    val_size = 0.1
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=args.seed
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=test_size/(test_size + val_size),
        random_state=args.seed
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scaling (FIT only on training)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler_X.transform(X_val).astype(np.float32)
    X_test_scaled = scaler_X.transform(X_test).astype(np.float32)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten().astype(np.float32)
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten().astype(np.float32)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten().astype(np.float32)
    
    # Calculate training statistics for imputation (avoid data leakage)
    X_train_stats = {}
    for col in X_train.columns:
        X_train_stats[col] = {
            'mean': X_train[col].mean(),
            'median': X_train[col].median(),
            'mode': X_train[col].mode().iloc[0] if len(X_train[col].mode()) > 0 else X_train[col].median()
        }
    
    # Inject missing data in TEST set only
    print(f"\nInjecting {args.mechanism} missing data at {args.rate*100:.1f}% rate in TEST set...")
    
    # Create test dataframe for missing injection
    df_test_complete = pd.DataFrame(X_test)
    df_test_complete[target_var] = y_test.values
    
    if args.mechanism == 'MAR' and args.aux_col is None:
        # Auto-select auxiliary column for MAR
        aux_col = features[0]  # Use first feature as auxiliary
        print(f"Auto-selected auxiliary column for MAR: {aux_col}")
    else:
        aux_col = args.aux_col
    
    df_test_missing = inject_missing_data(
        df_test_complete, args.mechanism, args.rate, target_var, aux_col, args.seed
    )
    
    # Report missing data statistics
    missing_stats = df_test_missing.isnull().sum()
    total_missing = missing_stats.sum()
    print(f"Total missing values injected: {total_missing:,}")
    print(f"Missing percentage: {(total_missing/df_test_missing.size)*100:.2f}%")
    
    # Prepare data for autoencoder training (features + target concatenated)
    print("\nPreparing data for autoencoder training...")
    
    # Create combined datasets 
    train_data = np.column_stack([X_train_scaled, y_train_scaled])
    val_data = np.column_stack([X_val_scaled, y_val_scaled])
    
    train_dataset = ProcessDataset(train_data[:, :-1], train_data[:, -1])
    val_dataset = ProcessDataset(val_data[:, :-1], val_data[:, -1])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2)
    
    # Model setup (input size = features + target)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    input_size = len(features) + 1  # features + target
    
    model = ControlledAutoencoder(
        n_features=input_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout
    )
    
    print(f"Model created:")
    print(f"  Input features: {len(features)} + 1 target = {input_size}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Device: {device.upper()}")
    
    # Training
    history = train_autoencoder(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), output_dir / "best_model.pth")
    print(f"Model saved to: {output_dir}/best_model.pth")
    
    # Reconstruction and evaluation on TEST data with missing values 
    print("\nEvaluating reconstruction on test data with injected missing values...")
    
    # Prepare test data for reconstruction 
    X_test_missing = df_test_missing[features]
    y_test_missing = df_test_missing[target_var]
    
    # Use original style reconstruction
    target_reconstructed = reconstruct_data_original_style(
        model, X_test_missing, scaler_X, scaler_y, 
        df_test_missing[target_var].isnull().values, device, X_train_stats
    )
    
    # Create final reconstructed dataframe
    df_test_reconstructed = df_test_missing.copy()
    
    # Only replace missing target values with reconstructions
    target_missing_mask = df_test_missing[target_var].isnull()
    df_test_reconstructed.loc[target_missing_mask, target_var] = target_reconstructed[target_missing_mask]
    
    # Calculate metrics only on artificially missing values
    missing_mask = df_test_missing[target_var].isnull()
    if missing_mask.sum() == 0:
        print(" No missing target values found for evaluation!")
        return
    
    y_true_missing = y_test.values[missing_mask]
    y_pred_missing = target_reconstructed[missing_mask]
    
    print(f"Evaluating {len(y_true_missing)} reconstructed target values...")
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_true_missing, y_pred_missing),
        'RMSE': np.sqrt(mean_squared_error(y_true_missing, y_pred_missing)),
        'R2': r2_score(y_true_missing, y_pred_missing),
        'n_samples': len(y_true_missing)
    }
    
    # Statistical tests
    ks_stat, ks_pvalue = ks_2samp(y_true_missing, y_pred_missing)
    residuals = y_pred_missing - y_true_missing
    shapiro_stat, shapiro_p = shapiro(residuals[:min(5000, len(residuals))])
    
    # Bootstrap confidence intervals
    mae_bootstrap, mae_ci_low, mae_ci_high = bootstrap_ci(
        mean_absolute_error, y_true_missing, y_pred_missing, n_boot=1000
    )
    rmse_bootstrap, rmse_ci_low, rmse_ci_high = bootstrap_ci(
        lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
        y_true_missing, y_pred_missing, n_boot=1000
    )
    
    metrics.update({
        'KS_statistic': ks_stat,
        'KS_pvalue': ks_pvalue,
        'Shapiro_statistic': shapiro_stat,
        'Shapiro_pvalue': shapiro_p,
        'MAE_bootstrap': mae_bootstrap,
        'MAE_CI_low': mae_ci_low,
        'MAE_CI_high': mae_ci_high,
        'RMSE_bootstrap': rmse_bootstrap,
        'RMSE_CI_low': rmse_ci_low,
        'RMSE_CI_high': rmse_ci_high
    })
    
    # Display results
    print("\nCONTROLLED TEST RESULTS")
    print("=" * 40)
    print(f"Mechanism: {args.mechanism}")
    print(f"Missing Rate: {args.rate*100:.1f}%")
    print(f"Missing Target Values: {metrics['n_samples']:,}")
    print(f"\nEVALUATION METRICS:")
    print(f"MAE:  {metrics['MAE']:.4f} [{metrics['MAE_CI_low']:.4f}, {metrics['MAE_CI_high']:.4f}]")
    print(f"RMSE: {metrics['RMSE']:.4f} [{metrics['RMSE_CI_low']:.4f}, {metrics['RMSE_CI_high']:.4f}]")
    print(f"R²:   {metrics['R2']:.4f}")
    print(f"KS:   {metrics['KS_statistic']:.4f} (p={metrics['KS_pvalue']:.6f})")
    print(f"Shapiro: {metrics['Shapiro_statistic']:.4f} (p={metrics['Shapiro_pvalue']:.6f})")
    
    # Save results
    config = {
        'mechanism': args.mechanism,
        'missing_rate': args.rate,
        'target_variable': target_var,
        'features': features,
        'auxiliary_column': aux_col,
        'batch_size': args.batch,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'patience': args.patience,
        'hidden_dim': args.hidden_dim,
        'latent_dim': args.latent_dim,
        'dropout': args.dropout,
        'seed': args.seed,
        'device': device,
        'n_features': len(features),
        'n_samples_complete': len(df_complete),
        'n_samples_test': len(X_test),
        'total_missing_injected': int(total_missing)
    }
    
    save_results(metrics, config, str(output_dir))
    
    # Generate visualizations (use only missing values for comparison)
    plot_controlled_results(
        history, 
        pd.DataFrame({'target': y_true_missing}), 
        pd.DataFrame({'target': y_pred_missing}), 
        args.mechanism, args.rate, str(output_dir)
    )
    
    # Generate temporal plots showing original vs reconstructed
    # Create full arrays for temporal plots
    y_true_full = y_test.values.copy()
    y_pred_full = target_reconstructed.copy()
    target_missing_mask_array = df_test_missing[target_var].isnull().values
    missing_indices = np.where(target_missing_mask_array)[0]
    
    # Temporal comparative plot (like original script)
    plot_temporal_comparison(
        y_true_full, y_pred_full, target_missing_mask_array,
        args.mechanism, args.rate, target_var, str(output_dir)
    )
    
    # Temporal focus on missing values only
    plot_temporal_missing_focus(
        y_true_missing, y_pred_missing, missing_indices,
        args.mechanism, args.rate, target_var, str(output_dir)
    )
    
    print(f"\n CONTROLLED TEST COMPLETED!")
    print(f"All results saved in: {output_dir}/")

# ╰───────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controlled Missing Data Experiment with Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data', required=True,
                       help='Path to .h5/.csv file with industrial data')
    
    # Feature configuration arguments
    parser.add_argument('--feature_config',
                       help='Path to feature_selection_config.json from cs_ae_with_eda.py (optional, uses defaults if not provided)')
    
    # Missing data injection arguments
    parser.add_argument('--mechanism', choices=['MCAR', 'MAR', 'MNAR'], default='MCAR',
                       help='Missing data mechanism')
    parser.add_argument('--rate', type=float, default=0.4,
                       help='Missing data rate (0.0 to 1.0)')
    parser.add_argument('--aux_col',
                       help='Auxiliary column for MAR mechanism (auto-selected if not provided)')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent space dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for regularization')
    
    # Training arguments
    parser.add_argument('--batch', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for reproducibility')
    parser.add_argument('--cpu', action='store_true',
                       help='Force use of CPU even with CUDA available')
    parser.add_argument('--output', default='outputs',
                       help='Directory to save results')
    
    args = parser.parse_args()
    main(args) 