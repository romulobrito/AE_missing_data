#!/usr/bin/env python3
"""
cs_ae_with_eda.py ───────────────────────────────────────────────────────────────
Complete Pipeline: EDA + Automatic Feature Selection + Enhanced Autoencoder
for Real Industrial Process Data (Sulfates)

This script integrates:
1. Automatic Data Exploration (EDA)
2. Automatic feature selection based on correlation and completeness
3. Enhanced autoencoder training
4. Advanced evaluation metrics

Pipeline flow:
EDA → Feature Selection → AE Training → Evaluation → Reports

Author: Romulo Brito da Silva - UFRJ
Date: 2025-08-07
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Sequence, Dict, Tuple, Optional, List
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
import missingno as msno

warnings.filterwarnings('ignore')

# ╭───────────────────────────── Utilitários ─────────────────────────────╮

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


def save_results(metrics: Dict, features_info: Dict, eda_info: Dict, args, output_dir: str = "outputs"):
    """Save complete pipeline results."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'eda_analysis': eda_info,
        'feature_selection': features_info,
        'model_metrics': metrics,
        'config': vars(args),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(f"{output_dir}/complete_pipeline_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f" Complete results saved to: {output_dir}/complete_pipeline_results.json")


def export_feature_selection(selected_features: List[str], target_var: str, 
                            features_info: Dict, output_dir: str = "outputs"):
    """
    Export feature selection configuration for use by other scripts.
    
    This creates a standardized JSON file that can be consumed by controlled
    testing scripts to maintain consistency in feature usage.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create exportable configuration
    feature_config = {
        'target_variable': target_var,
        'selected_features': selected_features,
        'feature_count': len(selected_features),
        'selection_method': features_info.get('selection_method', 'automatic'),
        'timestamp': pd.Timestamp.now().isoformat(),
        'selection_criteria': features_info.get('selection_criteria', {}),
        'analysis_summary': features_info.get('analysis_summary', {}),
        # Include feature details if available (for automatic selection)
        'feature_details': features_info.get('selected_features_details', [])
    }
    
    # Save feature configuration
    config_path = f"{output_dir}/feature_selection_config.json"
    with open(config_path, 'w') as f:
        json.dump(feature_config, f, indent=2, default=str)
    
    print(f" Feature selection config exported to: {config_path}")
    
    return config_path

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Análise Exploratória (AED) ─────────────────╮

def perform_eda(df: pd.DataFrame, target_var: str, output_dir: str = "outputs") -> Dict:
    """
    Update complete data exploration analysis.
    
    Returns:
        Dict with EDA information
    """
    print("\n STARTING DATA EXPLORATION (EDA)")
    print("=" * 55)
    
    os.makedirs(output_dir, exist_ok=True)
    eda_info = {}
    
    # Basic dataset information
    print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} variables")
    print(f" Target variable: {target_var}")
    
    eda_info['dataset_shape'] = df.shape
    eda_info['target_variable'] = target_var
    
    # Missing data analysis
    print("\n MISSING DATA ANALYSIS")
    print("-" * 35)
    
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df)) * 100
    
    # Missing data statistics
    total_missing = missing_summary.sum()
    total_cells = df.size
    overall_missing_pct = (total_missing / total_cells) * 100
    
    print(f"Total missing values: {total_missing:,}")
    print(f"Total percentage: {overall_missing_pct:.2f}%")
    
    eda_info['missing_analysis'] = {
        'total_missing': int(total_missing),
        'overall_percentage': float(overall_missing_pct),
        'variables_with_missing': int((missing_summary > 0).sum()),
        'complete_variables': int((missing_summary == 0).sum())
    }
    
    # Variables with missing data
    missing_vars = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if len(missing_vars) > 0:
        print(f"\n Variables with missing data ({len(missing_vars)}):")
        for var, count in missing_vars.head(10).items():
            pct = (count / len(df)) * 100
            print(f"  {var}: {count:,} ({pct:.2f}%)")
        
        eda_info['top_missing_variables'] = {
            var: {'count': int(count), 'percentage': float((count/len(df))*100)}
            for var, count in missing_vars.head(10).items()
        }
    
    # Visualizations of missing data
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Missing data matrix plot
    if len(missing_vars) > 0:
        msno.matrix(df, ax=axes[0,0], fontsize=8)
        axes[0,0].set_title('Missing Data Matrix', fontsize=12, fontweight='bold')
        
        # Completeness bar plot
        completeness = (1 - missing_pct/100) * 100
        top_incomplete = completeness[completeness < 100].sort_values().head(15)
        if len(top_incomplete) > 0:
            top_incomplete.plot(kind='barh', ax=axes[0,1])
            axes[0,1].set_title('Data Completeness by Variable (%)', fontsize=12, fontweight='bold')
            axes[0,1].set_xlabel('Completeness (%)')
    else:
        axes[0,0].text(0.5, 0.5, 'No missing data!', ha='center', va='center', fontsize=16)
        axes[0,0].set_title('Missing Data Matrix', fontsize=12, fontweight='bold')
        axes[0,1].text(0.5, 0.5, 'Complete dataset', ha='center', va='center', fontsize=16)
        axes[0,1].set_title('Data Completeness by Variable', fontsize=12, fontweight='bold')
    
    # Target variable descriptive statistics
    target_stats = df[target_var].describe()
    axes[1,0].text(0.1, 0.9, f'Descriptive Statistics - {target_var}:', fontsize=12, fontweight='bold', transform=axes[1,0].transAxes)
    stats_text = f"""
Count: {target_stats['count']:.0f}
Mean: {target_stats['mean']:.4f}
Std: {target_stats['std']:.4f}
Min: {target_stats['min']:.4f}
25%: {target_stats['25%']:.4f}
50%: {target_stats['50%']:.4f}
75%: {target_stats['75%']:.4f}
Max: {target_stats['max']:.4f}
Missing: {df[target_var].isnull().sum()} ({(df[target_var].isnull().sum()/len(df))*100:.2f}%)
    """
    axes[1,0].text(0.1, 0.8, stats_text.strip(), fontsize=10, transform=axes[1,0].transAxes, verticalalignment='top')
    axes[1,0].set_xlim(0, 1)
    axes[1,0].set_ylim(0, 1)
    axes[1,0].set_xticks([])
    axes[1,0].set_yticks([])
    axes[1,0].set_title(f'Target Variable Statistics', fontsize=12, fontweight='bold')
    
    # Target variable histogram
    df[target_var].dropna().hist(bins=50, ax=axes[1,1], edgecolor='black', alpha=0.7)
    axes[1,1].set_title(f'Distribution - {target_var}', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Value')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_missing_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    eda_info['target_statistics'] = target_stats.to_dict()
    eda_info['target_missing_count'] = int(df[target_var].isnull().sum())
    
    print(f" EDA plots saved to: {output_dir}/eda_missing_analysis.png")
    
    return eda_info

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Seleção de Features ────────────────────────╮

def automatic_feature_selection(df: pd.DataFrame, target_var: str, 
                               min_correlation: float = 0.3,
                               max_missing_pct: float = 30.0,
                               min_features: int = 3,
                               max_features: int = 10,
                               output_dir: str = "outputs") -> Tuple[List[str], Dict]:
    """
    Automatic feature selection based on correlation and completeness.
    
    Args:
        df: DataFrame with data
        target_var: Name of the target variable
        min_correlation: Minimum correlation with target (abs)
        max_missing_pct: Maximum percentage of missing data
        min_features: Minimum number of features
        max_features: Maximum number of features
        
    Returns:
        List of selected features and dict with information
    """
    print("\n AUTOMATIC FEATURE SELECTION")
    print("=" * 40)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    target_correlations = correlation_matrix[target_var].abs().sort_values(ascending=False)
    
    # Analyze completeness
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    # Selection criteria
    print(f" Selection criteria:")
    print(f"  • Minimum correlation with {target_var}: {min_correlation:.2f}")
    print(f"  • Maximum missing data: {max_missing_pct:.1f}%")
    print(f"  • Number of features: {min_features}-{max_features}")
    
    # Analyze each candidate feature
    feature_analysis = []
    
    for feature in df.columns:
        if feature == target_var:
            continue
            
        corr_value = target_correlations.get(feature, 0)
        missing_percentage = missing_pct.get(feature, 0)
        
        # Selection criteria
        meets_correlation = abs(corr_value) >= min_correlation
        meets_completeness = missing_percentage <= max_missing_pct
        
        # Selection score: |correlation| × (completeness/100)
        completeness_factor = (100 - missing_percentage) / 100
        selection_score = abs(corr_value) * completeness_factor
        
        feature_analysis.append({
            'feature': feature,
            'correlation': corr_value,
            'missing_pct': missing_percentage,
            'completeness': 100 - missing_percentage,
            'selection_score': selection_score,
            'meets_correlation': meets_correlation,
            'meets_completeness': meets_completeness,
            'selected': meets_correlation and meets_completeness
        })
    
    # Convert to DataFrame for easier analysis
    df_analysis = pd.DataFrame(feature_analysis)
    
    # Select features
    eligible_features = df_analysis[
        (df_analysis['meets_correlation']) & 
        (df_analysis['meets_completeness'])
    ].sort_values('selection_score', ascending=False)
    
    # Apply limits on number of features
    if len(eligible_features) < min_features:
        print(f"  Only {len(eligible_features)} features meet the criteria!")
        print(" Relaxing criteria to reach minimum number...")
        
        # Relax criteria gradually
        relaxed_features = df_analysis[
            df_analysis['missing_pct'] <= max_missing_pct
        ].sort_values('selection_score', ascending=False)
        
        selected_features = relaxed_features.head(min_features)['feature'].tolist()
        
    else:
        selected_features = eligible_features.head(max_features)['feature'].tolist()
    
    print(f"\nSelected features ({len(selected_features)}):")
    for feature in selected_features:
        row = df_analysis[df_analysis['feature'] == feature].iloc[0]
        print(f"  • {feature}:")
        print(f"    - Correlation: {row['correlation']:.3f}")
        print(f"    - Completeness: {row['completeness']:.1f}%")
        print(f"    - Score: {row['selection_score']:.3f}")
    
    # Visualization of selection
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    #  Scatter plot: Correlation vs Completeness
    colors = ['green' if f in selected_features else 'red' for f in df_analysis['feature']]
    scatter = axes[0,0].scatter(df_analysis['correlation'].abs(), df_analysis['completeness'], 
                               c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    axes[0,0].axvline(x=min_correlation, color='blue', linestyle='--', alpha=0.7, label=f'Corr mín: {min_correlation}')
    axes[0,0].axhline(y=100-max_missing_pct, color='orange', linestyle='--', alpha=0.7, label=f'Compl mín: {100-max_missing_pct:.1f}%')
    axes[0,0].set_xlabel('|Correlation| with Target')
    axes[0,0].set_ylabel('Completeness (%)')
    axes[0,0].set_title('Feature Selection: Correlation vs Completeness')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    #  Bar plot: Selection scores
    top_features = df_analysis.nlargest(15, 'selection_score')
    colors_bar = ['green' if f in selected_features else 'lightcoral' for f in top_features['feature']]
    axes[0,1].barh(range(len(top_features)), top_features['selection_score'], color=colors_bar)
    axes[0,1].set_yticks(range(len(top_features)))
    axes[0,1].set_yticklabels(top_features['feature'], fontsize=8)
    axes[0,1].set_xlabel('Selection Score')
    axes[0,1].set_title('Top Features by Selection Score')
    axes[0,1].grid(True, alpha=0.3)
    
    #  Correlation matrix of selected features
    if len(selected_features) > 1:
        selected_corr = correlation_matrix.loc[selected_features + [target_var], selected_features + [target_var]]
        im = axes[1,0].imshow(selected_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1,0].set_xticks(range(len(selected_corr.columns)))
        axes[1,0].set_yticks(range(len(selected_corr.columns)))
        axes[1,0].set_xticklabels(selected_corr.columns, rotation=45, ha='right', fontsize=8)
        axes[1,0].set_yticklabels(selected_corr.columns, fontsize=8)
        
        # Add values to matrix
        for i in range(len(selected_corr)):
            for j in range(len(selected_corr.columns)):
                axes[1,0].text(j, i, f'{selected_corr.iloc[i,j]:.2f}', 
                              ha='center', va='center', fontsize=8,
                              color='white' if abs(selected_corr.iloc[i,j]) > 0.5 else 'black')
        
        axes[1,0].set_title('Correlation Matrix - Selected Features')
        plt.colorbar(im, ax=axes[1,0], shrink=0.8)
    else:
        axes[1,0].text(0.5, 0.5, 'Only 1 feature\nselected', ha='center', va='center', fontsize=14)
        axes[1,0].set_title('Correlation Matrix - Selected Features')
    
    #  Selection statistics
    stats_text = f"""
SELECTION STATISTICS:

Total variables: {len(df.columns)-1}
Features analyzed: {len(df_analysis)}
Meet correlation: {(df_analysis['meets_correlation']).sum()}
Meet completeness: {(df_analysis['meets_completeness']).sum()}
Meet both criteria: {(df_analysis['selected']).sum()}
Selected features: {len(selected_features)}

APPLIED CRITERIA:
• Minimum correlation: {min_correlation:.2f}
• Minimum completeness: {100-max_missing_pct:.1f}%
• Range of features: {min_features}-{max_features}

AVERAGE SELECTION SCORE:
{df_analysis[df_analysis['feature'].isin(selected_features)]['selection_score'].mean():.3f}
    """
    
    axes[1,1].text(0.1, 0.9, stats_text.strip(), fontsize=10, transform=axes[1,1].transAxes, 
                   verticalalignment='top', fontfamily='monospace')
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])
    axes[1,1].set_title('Selection Statistics')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_selection_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save selected features to file
    with open(f"{output_dir}/selected_features.txt", 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    # Information for return
    selection_info = {
        'selected_features': selected_features,
        'selection_criteria': {
            'min_correlation': min_correlation,
            'max_missing_pct': max_missing_pct,
            'min_features': min_features,
            'max_features': max_features
        },
        'analysis_summary': {
            'total_candidates': len(df_analysis),
            'meets_correlation': int((df_analysis['meets_correlation']).sum()),
            'meets_completeness': int((df_analysis['meets_completeness']).sum()),
            'meets_both': int((df_analysis['selected']).sum()),
            'final_selected': len(selected_features)
        },
        'selected_features_details': df_analysis[df_analysis['feature'].isin(selected_features)].to_dict('records')
    }
    
    print(f" Selection analysis saved to: {output_dir}/feature_selection_analysis.png")
    print(f" Selected features saved to: {output_dir}/selected_features.txt")
    
    return selected_features, selection_info

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Dataset & Modelo AE ────────────────────────╮

class ProcessDataset(Dataset):
    """Custom dataset for industrial process data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EnhancedAutoencoder(nn.Module):
    """Enhanced autoencoder for industrial process variable reconstruction."""
    
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
            nn.Dropout(dropout/2),
            
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout/2),
            
            nn.Linear(hidden1, 1),  # Output for a target variable
        )
        
        # Inicialização Xavier/Glorot
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

# ╭────────────────────────── Training & Evaluation ────────────────────╮
# 

def generate_temporal_comparison_plot(model: nn.Module, df: pd.DataFrame, selected_features: List[str], 
                                     target_var: str, scaler_X, scaler_y, device: str, 
                                     output_dir: str = "outputs", train_feature_means: Optional[pd.Series] = None):
    """
    Generate temporal comparison plot showing real vs predicted values with missing data regions.
    
    This function:
    1. Imputes missing values in features using training means
    2. Generates predictions for the entire dataset
    3. Creates a scatter plot comparing real vs predicted values
    4. Highlights regions with missing data
    """
    print("\n GENERATING TEMPORAL COMPARISON PLOT")
    print("-" * 45)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare all features data
    all_features = df[selected_features].copy()
    
    # Impute missing values in features using TRAINING means (passed in)
    if train_feature_means is None:
        print("  Warning: train_feature_means not provided; using global means (may leak).")
        train_means = all_features.mean()
    else:
        train_means = train_feature_means
    all_features_imputed = all_features.fillna(train_means)
    
    if all_features_imputed.isna().sum().sum() > 0:
        print("  Warning: Still have NaNs, filling with 0")
        all_features_imputed = all_features_imputed.fillna(0)
    
    # Scale features
    all_features_scaled = scaler_X.transform(all_features_imputed)
    all_features_tensor = torch.FloatTensor(all_features_scaled).to(device)
    
    # Generate predictions for entire dataset
    model.eval()
    with torch.no_grad():
        all_predictions_scaled = model(all_features_tensor).cpu().numpy().flatten()
    
    # Inverse transform predictions
    all_predictions_original = scaler_y.inverse_transform(
        all_predictions_scaled.reshape(-1, 1)
    ).flatten()
    
    # Clip predictions to reasonable range
    all_predictions_original = np.clip(all_predictions_original, 0.001, 100)
    
    # Get original target data
    dados_originais = df[target_var].copy()
    indices_presentes = ~dados_originais.isna()
    valores_reais = dados_originais[indices_presentes]
    indices_reais = np.where(indices_presentes)[0]
    indices_ausentes = np.where(indices_presentes == False)[0]
    
    print(f"  Real values: {len(valores_reais)}")
    print(f"  Missing values: {len(indices_ausentes)}")
    print(f"  Total predictions: {len(all_predictions_original)}")
    
    # Create temporal comparison plot
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(18, 9))
    
    # Plot real values
    plt.scatter(indices_reais, valores_reais, 
               label='Real Values', alpha=0.6, color='blue', s=40)
    
    # Plot predictions for missing data
    plt.scatter(indices_ausentes, all_predictions_original[indices_ausentes],
               label='Predictions (Missing Data)', alpha=0.8, color='red', s=40)
    
    # Plot predictions for present data
    plt.scatter(indices_reais, all_predictions_original[indices_reais],
               label='Predictions (Present Data)', alpha=0.4, color='green', s=40)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel(f'Value of {target_var}', fontsize=12)
    plt.title('Comparison between Real and Predicted Values\n(Highlighting Regions with Missing Data)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save plot
    plot_path = f"{output_dir}/temporal_comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Temporal comparison plot saved to: {plot_path}")
    
    # Calculate additional metrics for missing data regions
    if len(indices_ausentes) > 0:
        missing_predictions = all_predictions_original[indices_ausentes]
        print(f"  Missing data predictions statistics:")
        print(f"    Mean: {missing_predictions.mean():.4f}")
        print(f"    Std: {missing_predictions.std():.4f}")
        print(f"    Min: {missing_predictions.min():.4f}")
        print(f"    Max: {missing_predictions.max():.4f}")
    
    return {
        'real_values_count': len(valores_reais),
        'missing_values_count': len(indices_ausentes),
        'total_predictions': len(all_predictions_original),
        'missing_predictions_stats': {
            'mean': float(missing_predictions.mean()) if len(indices_ausentes) > 0 else 0,
            'std': float(missing_predictions.std()) if len(indices_ausentes) > 0 else 0,
            'min': float(missing_predictions.min()) if len(indices_ausentes) > 0 else 0,
            'max': float(missing_predictions.max()) if len(indices_ausentes) > 0 else 0
        } if len(indices_ausentes) > 0 else {}
    }


def train_model(model: nn.Module, loaders: Dict[str, DataLoader], 
                epochs: int, lr: float, patience: int, device: str,
                output_dir: str = "outputs") -> Dict[str, list]:
    """Train the autoencoder model with early stopping and scheduler."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    print(f" Starting training on {device.upper()}")
    print(f"Parameters: epochs={epochs}, lr={lr}, patience={patience}")
    
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
                  device: str, target_name: str = "target") -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate the model with advanced metrics."""
    model.eval()
    predictions, targets = [], []
    
    print(" Evaluating model...")
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc="Evaluation"):
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
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = ks_2samp(y_true_flat, y_pred_flat)
    
    # Bootstrap CI
    mae_bootstrap, (mae_ci_low, mae_ci_high) = bootstrap_ci(
        mean_absolute_error, y_true_flat, y_pred_flat, n_boot=1000
    )
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


def plot_complete_results(eda_info: Dict, features_info: Dict, history: Dict, 
                         y_true: np.ndarray, y_pred: np.ndarray, target_name: str, 
                         output_dir: str = "outputs"):
    """Generate complete visualization of pipeline results."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Row 1: EDA Summary
    # Missing data overview
    missing_vars = eda_info.get('top_missing_variables', {})
    if missing_vars:
        vars_names = list(missing_vars.keys())[:10]
        percentages = [missing_vars[var]['percentage'] for var in vars_names]
        axes[0,0].barh(range(len(vars_names)), percentages, color='lightcoral')
        axes[0,0].set_yticks(range(len(vars_names)))
        axes[0,0].set_yticklabels(vars_names, fontsize=8)
        axes[0,0].set_xlabel('Percentage of Missing Data (%)')
        axes[0,0].set_title('Top 10 Variables with Missing Data', fontweight='bold')
    else:
        axes[0,0].text(0.5, 0.5, 'Dataset Completo\n(Sem dados ausentes)', 
                      ha='center', va='center', fontsize=14)
        axes[0,0].set_title('Missing Data Analysis', fontweight='bold')
    
    # Feature selection summary
    selected_features = features_info['selected_features']
    if len(selected_features) > 0:
        # Check if we have selection scores (automatic selection) or just feature names (manual)
        if 'selected_features_details' in features_info:
            scores = [detail['selection_score'] for detail in features_info['selected_features_details']]
            axes[0,1].barh(range(len(selected_features)), scores, color='lightgreen')
            axes[0,1].set_xlabel('Selection Score')
        else:
            # Manual features - just show feature names
            axes[0,1].barh(range(len(selected_features)), [1]*len(selected_features), color='lightgreen')
            axes[0,1].set_xlabel('Manual Selection')
        
        axes[0,1].set_yticks(range(len(selected_features)))
        axes[0,1].set_yticklabels(selected_features, fontsize=8)
        axes[0,1].set_title(f'Selected Features ({len(selected_features)})', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
    
    # Row 2: Training Results
    # Loss history
    axes[1,0].plot(history['train_loss'], label='Training', alpha=0.8, linewidth=2)
    axes[1,0].plot(history['val_loss'], label='Validation', alpha=0.8, linewidth=2)
    axes[1,0].set_title('Training History', fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('MSE Loss')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Actual vs Predicted
    axes[1,1].scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1,1].plot(lims, lims, 'r--', linewidth=2, alpha=0.8)
    axes[1,1].set_xlabel('Actual Values')
    axes[1,1].set_ylabel('Predicted Values')
    axes[1,1].set_title(f'Actual vs Predicted - {target_name}', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    # Row 3: Detailed Analysis
    # Residuals
    residuals = y_true - y_pred
    axes[2,0].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[2,0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[2,0].set_xlabel('Predicted Values')
    axes[2,0].set_ylabel('Residuals')
    axes[2,0].set_title('Residual Analysis', fontweight='bold')
    axes[2,0].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[2,1].hist(y_true, bins=30, alpha=0.7, label='Actual', density=True, color='blue')
    axes[2,1].hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True, color='red')
    axes[2,1].set_xlabel('Value')
    axes[2,1].set_ylabel('Density')
    axes[2,1].set_title('Distribution Comparison', fontweight='bold')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/complete_pipeline_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Complete results saved to: {output_dir}/complete_pipeline_results.png")

# ╰───────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────────── Main ─────────────────────────────────╮

def main(args):
    """Complete pipeline: AED → Selection → Training → Evaluation."""
    
    print(" COMPLETE PIPELINE: AED + SELECTION + AUTOENCODER")
    print("=" * 65)
    
    # Configure reproducibility
    set_seed(args.seed)
    
    # Load data
    print(f" Loading data from: {args.data}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    if data_path.suffix in {'.h5', '.hdf5'}:
        df = pd.read_hdf(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f" Dataset loaded: {df.shape[0]} samples, {df.shape[1]} variables")
    
    # Check if target exists
    if args.target not in df.columns:
        raise ValueError(f"Target variable '{args.target}' not found in dataset!")
    
    # EXPLORATORY DATA ANALYSIS (AED)
    eda_info = perform_eda(df, args.target, args.output)
    
    # Create a single split of indices FIRST to avoid leakage and to reuse
    # the same folds for feature selection and model training.
    all_indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)
    
    #  AUTOMATIC FEATURE SELECTION
    if args.manual_features:
        # Use manually provided features
        if args.features:
            features_path = Path(args.features)
            if features_path.exists():
                if features_path.suffix == '.json':
                    with open(features_path, 'r') as f:
                        selected_features = json.load(f)
                else:
                    selected_features = [line.strip() for line in features_path.read_text().splitlines() 
                                       if line.strip()]
                
                available_features = [f for f in selected_features if f in df.columns]
                features_info = {
                    'selected_features': available_features,
                    'selection_method': 'manual',
                    'features_file': str(features_path)
                }
                print(f" Using manual features: {len(available_features)} selected")
            else:
                raise FileNotFoundError(f"Features file not found: {features_path}")
        else:
            raise ValueError("For manual selection, provide --features with features file")
    else:
        # Automatic selection ON TRAINING ONLY to avoid leakage
        df_for_selection = df.iloc[train_idx]
        selected_features, features_info = automatic_feature_selection(
            df_for_selection, args.target, 
            min_correlation=args.min_correlation,
            max_missing_pct=args.max_missing_pct,
            min_features=args.min_features,
            max_features=args.max_features,
            output_dir=args.output
        )
    
    if len(selected_features) == 0:
        raise ValueError("No features were selected! Relax the criteria.")
    
    # Prepare data for training
    df_model = df[selected_features + [args.target]].dropna()
    print(f" Data after cleaning: {df_model.shape[0]} samples")
    
    if len(df_model) < 100:
        raise ValueError("Insufficient data after cleaning!")
    
    # Prepare features and target in original scale
    X_all = df_model[selected_features]
    y_all = df_model[[args.target]]
    
    # Reuse the SAME split indices, intersecting with rows available after dropna
    train_idx_model = [i for i in train_idx if i in df_model.index]
    val_idx_model = [i for i in val_idx if i in df_model.index]
    test_idx_model = [i for i in test_idx if i in df_model.index]
    
    # Fallback: if any split is empty after dropna, resplit within df_model indices
    if (len(train_idx_model) == 0) or (len(val_idx_model) == 0) or (len(test_idx_model) == 0):
        print(" WARNING: Original fold intersection produced empty split after dropna. Resplitting within available rows.")
        available_idx = list(df_model.index)
        if len(available_idx) < 10:
            raise ValueError("Too few samples available after dropna to create splits.")
        train_idx_model, temp_idx_model = train_test_split(available_idx, test_size=0.3, random_state=args.seed)
        val_idx_model, test_idx_model = train_test_split(temp_idx_model, test_size=0.5, random_state=args.seed)
    
    X_train_raw = X_all.loc[train_idx_model].values.astype(np.float32)
    X_val_raw   = X_all.loc[val_idx_model].values.astype(np.float32)
    X_test_raw  = X_all.loc[test_idx_model].values.astype(np.float32)
    y_train_raw = y_all.loc[train_idx_model].values.astype(np.float32)
    y_val_raw   = y_all.loc[val_idx_model].values.astype(np.float32)
    y_test_raw  = y_all.loc[test_idx_model].values.astype(np.float32)
    
    print(f"Final dimensions: X_train={X_train_raw.shape}, X_val={X_val_raw.shape}, X_test={X_test_raw.shape}")
    
    # Compute training feature means (original scale) for later imputations
    train_feature_means = pd.Series(X_train_raw.mean(axis=0), index=selected_features)
    
    # Scaling: fit on training only, transform val/test
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler_X.transform(X_val_raw).astype(np.float32)
    X_test = scaler_X.transform(X_test_raw).astype(np.float32)
    
    scaler_y = RobustScaler() if args.robust_target else StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw).astype(np.float32)
    y_val = scaler_y.transform(y_val_raw).astype(np.float32)
    y_test = scaler_y.transform(y_test_raw).astype(np.float32)
    
    # DataLoaders
    loaders = {
        'train': DataLoader(ProcessDataset(X_train, y_train), 
                           batch_size=args.batch, shuffle=True, num_workers=2),
        'val': DataLoader(ProcessDataset(X_val, y_val), 
                         batch_size=args.batch, shuffle=False, num_workers=2),
        'test': DataLoader(ProcessDataset(X_test, y_test), 
                          batch_size=args.batch, shuffle=False, num_workers=2),
    }
    
    # TRAINING THE AUTOENCODER
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = EnhancedAutoencoder(
        in_features=len(selected_features), 
        latent_dim=args.latent, 
        dropout=args.dropout
    )
    
    print(f" Model created:")
    print(f"  Input: {len(selected_features)} features")
    print(f"  Latent: {args.latent} dimensions")
    print(f"  Device: {device.upper()}")
    
    history = train_model(
        model, loaders, 
        epochs=args.epochs, 
        lr=args.lr, 
        patience=args.patience, 
        device=device,
        output_dir=args.output
    )
    
    # EVALUATION
    metrics, y_true, y_pred = evaluate_model(
        model, loaders['test'], scaler_y, device, args.target
    )
    
    # Show results
    print("\nFINAL PIPELINE RESULTS")
    print("=" * 45)
    print(f"Selected features: {len(selected_features)}")
    print(f"Samples used: {len(df_model)}")
    print("\n EVALUATION METRICS:")
    print(f"MAE:  {metrics['MAE']:.4f} [{metrics['MAE_CI_low']:.4f}, {metrics['MAE_CI_high']:.4f}]")
    print(f"RMSE: {metrics['RMSE']:.4f} [{metrics['RMSE_CI_low']:.4f}, {metrics['RMSE_CI_high']:.4f}]")
    print(f"R²:   {metrics['R2']:.4f}")
    print(f"KS:   {metrics['KS_statistic']:.4f} (p={metrics['KS_pvalue']:.4f})")
    
    # GENERATE TEMPORAL COMPARISON PLOT (impute with training-only means)
    temporal_info = generate_temporal_comparison_plot(
        model, df, selected_features, args.target, scaler_X, scaler_y, device, args.output, train_feature_means
    )
    
    # SAVE COMPLETE RESULTS
    save_results(metrics, features_info, eda_info, args, args.output)
    
    # EXPORT FEATURE SELECTION CONFIG
    export_feature_selection(selected_features, args.target, features_info, args.output)
    
    # GENERATE COMPLETE VISUALIZATIONS
    plot_complete_results(eda_info, features_info, history, y_true, y_pred, args.target, args.output)
    
    print(f"\nCOMPLETE PIPELINE FINALIZED!")
    print(f" All results saved in: {args.output}/")

# ╰───────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete Pipeline: AED + Feature Selection + Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data', required=True, 
                       help='Path to .h5/.csv file with industrial data')
    parser.add_argument('--target', required=True,
                       help='Name of the target variable to be reconstructed')
    
    # Optional arguments - Feature selection
    parser.add_argument('--manual_features', action='store_true',
                       help='Use manual feature selection (requires --features)')
    parser.add_argument('--features', 
                       help='File .txt/.json with features (only if --manual_features)')
    parser.add_argument('--min_correlation', type=float, default=0.3,
                       help='Minimum correlation with target for automatic selection')
    parser.add_argument('--max_missing_pct', type=float, default=30.0,
                       help='Maximum percentage of missing data for selection')
    parser.add_argument('--min_features', type=int, default=3,
                       help='Minimum number of features to select')
    parser.add_argument('--max_features', type=int, default=10,
                       help='Maximum number of features to select')
    
    # Optional arguments - Model
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--latent', type=int, default=16,
                       help='Latent space dimension of the autoencoder')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for regularization')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    parser.add_argument('--robust_target', action='store_true',
                       help='Use RobustScaler for target variable')
    
    # Optional arguments - System
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for reproducibility')
    parser.add_argument('--cpu', action='store_true',
                       help='Force use of CPU')
    parser.add_argument('--output', default='outputs',
                       help='Directory to save results')
    
    args = parser.parse_args()
    main(args) 