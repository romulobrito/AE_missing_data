#!/usr/bin/env python3
"""
plot_exploratorio.py
Gera plots explicativos da série temporal das features e target
para o pipeline CDAE de Virtual Sensor.

Author: Romulo Brito da Silva - UFRJ
Date: 2025-09-18
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data_and_config():
    """Carrega dados e configuração do run final."""
    # Paths
    base_dir = Path(__file__).parent
    outputs_dir = base_dir / 'outputs' / 'cdae_blocked_auto_final'
    
    # Carregar configuração
    with open(outputs_dir / 'complete_pipeline_results.json', 'r') as f:
        results = json.load(f)
    
    config = results['config']
    features = results['feature_selection']['selected_features']
    target = config['target']
    
    # Carregar dados
    data_path = config['data']
    if data_path.endswith('.h5'):
        df = pd.read_hdf(data_path)
    else:
        df = pd.read_csv(data_path)
    
    return df, features, target, results

def plot_target_overview(df, target, output_dir):
    """Plot overview do target: distribuição e série temporal."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Série temporal
    axes[0,0].plot(df.index, df[target], alpha=0.7, linewidth=0.5)
    axes[0,0].set_title(f'Série Temporal: {target}')
    axes[0,0].set_xlabel('Índice (proxy temporal)')
    axes[0,0].set_ylabel('Valor')
    axes[0,0].grid(True, alpha=0.3)
    
    # Histograma
    df[target].dropna().hist(bins=50, ax=axes[0,1], alpha=0.7)
    axes[0,1].set_title(f'Distribuição: {target}')
    axes[0,1].set_xlabel('Valor')
    axes[0,1].set_ylabel('Frequência')
    axes[0,1].grid(True, alpha=0.3)
    
    # Disponibilidade ao longo do tempo
    available = df[target].notna().astype(int)
    window = max(1000, len(df) // 100)  # janela móvel
    availability_rolling = available.rolling(window, center=True).mean()
    
    axes[1,0].plot(df.index, availability_rolling, color='green', linewidth=1)
    axes[1,0].set_title(f'Disponibilidade do Target (janela {window})')
    axes[1,0].set_xlabel('Índice (proxy temporal)')
    axes[1,0].set_ylabel('Fração Presente')
    axes[1,0].set_ylim(0, 1.1)
    axes[1,0].grid(True, alpha=0.3)
    
    # Estatísticas
    stats = df[target].describe()
    missing_pct = df[target].isna().mean() * 100
    
    stats_text = f"""Estatísticas do Target:
    Count: {stats['count']:.0f}
    Missing: {missing_pct:.1f}%
    Mean: {stats['mean']:.2f}
    Std: {stats['std']:.2f}
    Min: {stats['min']:.2f}
    Q25: {stats['25%']:.2f}
    Q50: {stats['50%']:.2f}
    Q75: {stats['75%']:.2f}
    Max: {stats['max']:.2f}"""
    
    axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'target_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Target overview salvo em: {output_dir}/target_overview.png")

def plot_features_overview(df, features, target, output_dir):
    """Plot overview das features: correlações e séries temporais."""
    # Matriz de correlação
    features_and_target = features + [target]
    corr_data = df[features_and_target].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlação: Features vs Target')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Matriz de correlação salva em: {output_dir}/correlation_matrix.png")
    
    # Séries temporais das features (subplot)
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Subsample para visualização se muito grande
        if len(df) > 10000:
            step = len(df) // 5000
            idx_plot = df.index[::step]
            data_plot = df[feature].iloc[::step]
        else:
            idx_plot = df.index
            data_plot = df[feature]
        
        ax.plot(idx_plot, data_plot, alpha=0.7, linewidth=0.5)
        ax.set_title(f'{feature} (r={corr_data.loc[feature, target]:.3f})')
        ax.set_xlabel('Índice')
        ax.grid(True, alpha=0.3)
        
        # Marca missing data
        missing_pct = df[feature].isna().mean() * 100
        if missing_pct > 0:
            ax.text(0.02, 0.98, f'Missing: {missing_pct:.1f}%', 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Remove subplots vazios
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'features_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Séries temporais das features salvas em: {output_dir}/features_timeseries.png")

def plot_missing_patterns(df, features, target, output_dir):
    """Plot padrões de dados faltantes."""
    features_and_target = features + [target]
    missing_df = df[features_and_target].isna()
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Heatmap de missing data
    # Subsample se muito grande
    if len(df) > 5000:
        step = len(df) // 2000
        missing_plot = missing_df.iloc[::step]
        indices_plot = df.index[::step]
    else:
        missing_plot = missing_df
        indices_plot = df.index
    
    sns.heatmap(missing_plot.T, cbar=True, yticklabels=True, xticklabels=False, 
                cmap='RdYlBu_r', ax=axes[0])
    axes[0].set_title('Padrão de Dados Faltantes ao Longo do Tempo')
    axes[0].set_xlabel('Índice (subsampled)')
    
    # Porcentagem de missing por variável
    missing_pct = missing_df.mean().sort_values(ascending=False) * 100
    missing_pct.plot(kind='bar', ax=axes[1], color='steelblue', alpha=0.7)
    axes[1].set_title('Porcentagem de Dados Faltantes por Variável')
    axes[1].set_ylabel('% Missing')
    axes[1].set_xlabel('Variáveis')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Destacar target
    target_idx = missing_pct.index.get_loc(target)
    axes[1].patches[target_idx].set_color('red')
    axes[1].patches[target_idx].set_alpha(0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'missing_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Padrões de dados faltantes salvos em: {output_dir}/missing_patterns.png")

def plot_target_vs_top_features(df, features, target, output_dir, top_n=4):
    """Scatter plots do target vs top features por correlação."""
    # Selecionar top features por correlação absoluta
    correlations = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n).index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        # Usar apenas dados onde ambos estão presentes
        mask = df[feature].notna() & df[target].notna()
        x_data = df.loc[mask, feature]
        y_data = df.loc[mask, target]
        
        # Subsample se muito grande
        if len(x_data) > 5000:
            idx_sample = np.random.choice(len(x_data), 5000, replace=False)
            x_data = x_data.iloc[idx_sample]
            y_data = y_data.iloc[idx_sample]
        
        axes[i].scatter(x_data, y_data, alpha=0.5, s=1)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'{feature} vs {target} (r={correlations[feature]:.3f})')
        axes[i].grid(True, alpha=0.3)
        
        # Linha de tendência
        if len(x_data) > 10:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_data.min(), x_data.max(), 50)
            axes[i].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'target_vs_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Scatter plots target vs features salvos em: {output_dir}/target_vs_features.png")

def plot_top_features_with_target(df, features, target, output_dir, top_n=4):
    """Plot das top features + target em séries temporais numa única imagem."""
    # Selecionar top features por correlação absoluta
    correlations = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n).index.tolist()
    
    # Preparar dados (subsample se muito grande)
    if len(df) > 10000:
        step = len(df) // 5000
        df_plot = df.iloc[::step]
        indices_plot = df.index[::step]
    else:
        df_plot = df
        indices_plot = df.index
    
    # Normalizar para mesma escala visual (z-score)
    features_to_plot = top_features + [target]
    df_norm = pd.DataFrame(index=df_plot.index)
    
    for col in features_to_plot:
        data = df_plot[col].dropna()
        if len(data) > 0:
            mean_val = data.mean()
            std_val = data.std()
            if std_val > 0:
                df_norm[col] = (df_plot[col] - mean_val) / std_val
            else:
                df_norm[col] = df_plot[col] - mean_val
        else:
            df_norm[col] = df_plot[col]
    
    # Plot
    plt.figure(figsize=(18, 10))
    
    # Cores diferentes para cada série
    colors = ['steelblue', 'forestgreen', 'darkorange', 'purple', 'red']
    
    for i, feature in enumerate(top_features):
        corr_val = correlations[feature]
        plt.plot(indices_plot, df_norm[feature], 
                label=f'{feature} (r={corr_val:.3f})', 
                alpha=0.7, linewidth=1.0, color=colors[i])
    
    # Target em destaque
    plt.plot(indices_plot, df_norm[target], 
            label=f'{target} (TARGET)', 
            alpha=0.9, linewidth=1.5, color=colors[top_n], linestyle='-')
    
    plt.title('Séries Temporais: Top Features vs Target (normalizadas)')
    plt.xlabel('Índice (proxy temporal)')
    plt.ylabel('Valores normalizados (z-score)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adicionar info de missing data
    missing_info = []
    for feature in features_to_plot:
        missing_pct = df[feature].isna().mean() * 100
        missing_info.append(f'{feature}: {missing_pct:.1f}% missing')
    
    info_text = 'Missing data:\n' + '\n'.join(missing_info)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_features_with_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Séries temporais top features + target salvas em: {output_dir}/top_features_with_target.png")

def main():
    """Pipeline principal de geração de plots."""
    print("Gerando plots exploratórios da série temporal...")
    
    # Carregar dados
    df, features, target, results = load_data_and_config()
    
    # Diretório de saída
    output_dir = Path(__file__).parent / 'plots_exploratorios'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Dataset: {df.shape[0]} amostras, {df.shape[1]} variáveis")
    print(f"Target: {target}")
    print(f"Features selecionadas: {len(features)}")
    print(f"Salvando plots em: {output_dir}")
    
    # Gerar plots
    plot_target_overview(df, target, output_dir)
    plot_features_overview(df, features, target, output_dir)
    plot_missing_patterns(df, features, target, output_dir)
    plot_target_vs_top_features(df, features, target, output_dir)
    plot_top_features_with_target(df, features, target, output_dir)
    
    # Sumário final
    summary = f"""
Sumário dos Plots Gerados:
- target_overview.png: Distribuição e série temporal do target
- correlation_matrix.png: Matriz de correlação features vs target  
- features_timeseries.png: Séries temporais de todas as features
- missing_patterns.png: Padrões de dados faltantes
- target_vs_features.png: Scatter plots target vs top features
- top_features_with_target.png: Top 4 features + target em séries temporais

Dataset: {df.shape[0]} amostras, {target} (missing {df[target].isna().mean()*100:.1f}%)
Features: {len(features)} selecionadas (correlação mín: 0.3)
"""
    
    with open(output_dir / 'README.txt', 'w') as f:
        f.write(summary)
    
    print("\nPlots exploratórios gerados com sucesso!")
    print(f"Confira os arquivos em: {output_dir}")

if __name__ == "__main__":
    main() 