# Reconstructing Highly Missing Industrial Data: EDA-Driven Autoencoder Success and Systematic Validation

This study presents a comprehensive two-phase approach for autoencoder-based missing data imputation in sulfate production processes

##  Overview

The project provides two main approaches for handling missing data in industrial datasets:

1. **Enhanced Autoencoder Pipeline** (`cs_ae_enhanced.py`) - Uses pre-selected features with robust training
2. **Complete Pipeline with EDA** (`cs_ae_with_eda.py`) - Integrates exploratory data analysis and automatic feature selection

##  Repository Structure

```
├── validacao_dados_ausentes_codigos/
│   ├── cs_ae_enhanced.py              # Enhanced autoencoder with pre-selected features
│   ├── cs_ae_with_eda.py              # Complete pipeline with EDA integration
│   ├── plot_variable_selection.py     # Variable selection visualization
│   ├── features_1251_FIT_801C_2.txt   # Pre-selected features for target variable
│   ├── config_sulfatos.json           # Configuration file for different scenarios
│   ├── run_example.py                 # Example execution script
│   ├── run_examples_both.py           # Comparison script for both approaches
│   └── outputs/                       # Generated results and plots
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/romulobrito/AE_missing_data.git
cd AE_missing_data

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Enhanced Autoencoder (Pre-selected Features)
```bash
python validacao_dados_ausentes_codigos/cs_ae_enhanced.py \
    --data "path/to/your/data.h5" \
    --target "1251_FIT_801C_2" \
    --epochs 100 \
    --output "outputs/enhanced_results"
```

#### Complete Pipeline with EDA
```bash
python validacao_dados_ausentes_codigos/cs_ae_with_eda.py \
    --data "path/to/your/data.h5" \
    --target "1251_FIT_801C_2" \
    --epochs 100 \
    --output "outputs/complete_results"
```

##  Key Features

### Enhanced Autoencoder (`cs_ae_enhanced.py`)
- **Robust Architecture**: Batch normalization, dropout, and configurable latent dimensions
- **Advanced Training**: Early stopping, learning rate scheduling, and comprehensive metrics
- **Statistical Validation**: KS test, bootstrap confidence intervals, and residual analysis
- **CLI Interface**: Flexible command-line interface for different configurations

### Complete Pipeline (`cs_ae_with_eda.py`)
- **Automatic EDA**: Missing data analysis, correlation analysis, and data quality assessment
- **Feature Selection**: Automatic selection based on correlation and completeness criteria
- **End-to-End Processing**: From raw data to final predictions
- **Comprehensive Visualization**: EDA plots, feature selection analysis, and model evaluation

##  Performance Results

### Enhanced Autoencoder Results
- **MAE**: 1.4680 ± 0.0520 (95% CI)
- **RMSE**: 2.0679 ± 0.2270 (95% CI)
- **R²**: 0.9952
- **Features Used**: 5 pre-selected variables

### Complete Pipeline Results
- **MAE**: 1.0548 ± 0.0368 (95% CI)
- **RMSE**: 1.4596 ± 0.0655 (95% CI)
- **R²**: 0.9973
- **Features Used**: 10 automatically selected variables

##  Configuration

The `config_sulfatos.json` file contains pre-defined configurations for different scenarios:

- **Basic**: Simple configuration for quick testing
- **Robust**: Balanced configuration for production use
- **High Capacity**: Maximum model capacity for complex patterns
- **Conservative**: Conservative settings for stability

##  Output Files

Both scripts generate comprehensive outputs:

- **Model Files**: `.pth` files containing trained autoencoder models
- **Results JSON**: Detailed metrics and configuration information
- **Visualization Plots**: Training history, predictions vs actual, residual analysis
- **Feature Information**: Selected features and their characteristics (EDA pipeline)

##  Technical Details

### Model Architecture
- **Encoder**: Multiple fully connected layers with ReLU activation
- **Latent Space**: Configurable dimensionality (default: 16)
- **Decoder**: Symmetric architecture to encoder
- **Regularization**: Batch normalization and dropout for stability

### Training Features
- **Optimizer**: Adam with configurable learning rate
- **Loss Function**: MSE with optional Huber loss
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Learning Rate Scheduling**: Reduces learning rate on plateau

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **KS Test**: Kolmogorov-Smirnov test for distribution comparison
- **Bootstrap CI**: Non-parametric confidence intervals

##  Documentation

Additional documentation files:
- `README_enhanced_autoencoder.md`: Detailed guide for enhanced autoencoder
- `README_pipeline_comparison.md`: Comparison between both approaches
- `COMO_EXECUTAR.md`: Step-by-step execution guide

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Author

**Romulo Brito** - UFRJ (Federal University of Rio de Janeiro)

##  Acknowledgments

- Chemical Engineering Department, UFRJ
- PyTorch and scikit-learn communities

---

**Note**: This repository is designed for industrial process data analysis and missing data imputation. The implementations are optimized for chemical process variables and may require adaptation for other domains. 