#!/usr/bin/env python3
"""
run_example.py ──────────────────────────────────────────────────────────────
Example of using the enhanced autoencoder pipeline for industrial data.

This script demonstrates how to run cs_ae_enhanced.py with different configurations
for the sulfates data analyzed in the notebook.
"""

import subprocess
import sys
from pathlib import Path

def run_autoencoder_experiment(config_name: str, **kwargs):
    """Executes an experiment with specific configuration."""
    
    print(f"\n EXECUTING EXPERIMENT: {config_name}")
    print("=" * 60)
    
    # Base parameters
    base_cmd = [
        sys.executable, "cs_ae_enhanced.py",
        "--data", "/home/romulo/Downloads/sulfatos_dados_concatenados_formatados_2021.h5",
        "--target", "1251_FIT_801C_2"
    ]
    
    # Add specific parameters
    for key, value in kwargs.items():
        if value is True:
            base_cmd.append(f"--{key}")
        elif value is not False and value is not None:
            base_cmd.extend([f"--{key}", str(value)])
    
    # Execute command
    try:
        result = subprocess.run(base_cmd, capture_output=True, text=True, check=True)
        print(" Experiment completed successfully!")
        print("\nProgram output:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f" Error during execution: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
    except FileNotFoundError:
        print(" cs_ae_enhanced.py not found in current directory")

def main():
    """Executes different experiment configurations."""
    
    print(" EXPERIMENTS WITH ENHANCED AUTOENCODER")
    print(" Data: Sulfates - Variable: 1251_FIT_801C_2")
    print("=" * 70)
    
    # Check if data file exists
    data_path = Path("/home/romulo/Downloads/sulfatos_dados_concatenados_formatados_2021.h5")
    if not data_path.exists():
        print(f" Data file not found: {data_path}")
        print("Please adjust the path in the script or download the data.")
        return
    
    # Experiment 1: Basic configuration (based on notebook)
    run_autoencoder_experiment(
        "Basic Configuration",
        features="features_1251_FIT_801C_2.txt",
        batch=64,
        latent=16,
        dropout=0.2,
        epochs=100,
        lr=1e-3,
        patience=15,
        output="outputs/exp1_basic"
    )
    
    # Experiment 2: Robust configuration (for data with outliers)
    run_autoencoder_experiment(
        "Robust Configuration",
        features="features_1251_FIT_801C_2.txt",
        robust_target=True,
        batch=32,
        latent=8,
        dropout=0.3,
        epochs=150,
        lr=5e-4,
        patience=20,
        output="outputs/exp2_robust"
    )
    
    # Experiment 3: High capacity configuration
    run_autoencoder_experiment(
        "High Capacity",
        features="features_1251_FIT_801C_2.txt",
        batch=128,
        latent=32,
        dropout=0.1,
        epochs=200,
        lr=1e-3,
        patience=25,
        output="outputs/exp3_high_capacity"
    )
    
    # Experiment 4: Conservative training (overfitting prevention)
    run_autoencoder_experiment(
        "Conservative Training",
        features="features_1251_FIT_801C_2.txt",
        batch=16,
        latent=4,
        dropout=0.4,
        epochs=100,
        lr=1e-4,
        patience=10,
        output="outputs/exp4_conservative"
    )
    
    print("\n ALL EXPERIMENTS COMPLETED!")
    print("\n To view results:")
    print("  - Check the outputs/exp* directories")
    print("  - Analyze the experiment_results.json file")
    print("  - Visualize the training_evaluation_results.png file")

if __name__ == "__main__":
    main() 