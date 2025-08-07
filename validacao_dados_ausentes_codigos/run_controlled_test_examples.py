#!/usr/bin/env python3
"""
run_controlled_test_examples.py ───────────────────────────────────────────────
Example script demonstrating how to run controlled missing data experiments
with different configurations and mechanisms.

This script shows how to use teste_controlado_dados_faltantes_improved.py
with various parameter combinations for comprehensive testing.

Author: Romulo Brito da Silva - UFRJ
Date: 2025-08-07
"""

import subprocess
import sys
from pathlib import Path

def run_controlled_test(data_path: str, mechanism: str, rate: float, 
                       output_dir: str = "outputs", **kwargs):
    """
    Run controlled test with specified parameters.
    
    Args:
        data_path: Path to data file
        mechanism: MCAR, MAR, or MNAR
        rate: Missing data rate (0.0 to 1.0)
        output_dir: Output directory
        **kwargs: Additional arguments for the script
    """
    cmd = [
        sys.executable, "teste_controlado_dados_faltantes_improved.py",
        "--data", data_path,
        "--mechanism", mechanism,
        "--rate", str(rate),
        "--output", output_dir
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if key.startswith('--'):
            cmd.extend([key, str(value)])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"\n{'='*60}")
    print(f"Running controlled test:")
    print(f"  Mechanism: {mechanism}")
    print(f"  Missing Rate: {rate*100:.1f}%")
    print(f"  Additional args: {kwargs}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Test completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run various controlled test configurations."""
    
    # Data path - adjust as needed
    data_path = "/home/romulo/Downloads/sulfatos_dados_concatenados_formatados_2021.h5"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable in this script.")
        return
    
    print("🎯 CONTROLLED MISSING DATA EXPERIMENTS")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        # Basic MCAR tests
        {
            'mechanism': 'MCAR',
            'rate': 0.2,
            'description': 'Low missing rate MCAR'
        },
        {
            'mechanism': 'MCAR',
            'rate': 0.5,
            'description': 'Medium missing rate MCAR'
        },
        {
            'mechanism': 'MCAR',
            'rate': 0.8,
            'description': 'High missing rate MCAR'
        },
        
        # MAR tests
        {
            'mechanism': 'MAR',
            'rate': 0.3,
            'description': 'MAR with medium missing rate'
        },
        {
            'mechanism': 'MAR',
            'rate': 0.6,
            'description': 'MAR with high missing rate'
        },
        
        # MNAR tests
        {
            'mechanism': 'MNAR',
            'rate': 0.4,
            'description': 'MNAR with medium missing rate'
        },
        {
            'mechanism': 'MNAR',
            'rate': 0.7,
            'description': 'MNAR with high missing rate'
        }
    ]
    
    # Model configurations
    model_configs = [
        {
            'name': 'basic',
            'params': {
                'hidden_dim': 32,
                'latent_dim': 16,
                'dropout': 0.1,
                'batch': 128,
                'epochs': 100,
                'lr': 1e-3,
                'patience': 15
            }
        },
        {
            'name': 'robust',
            'params': {
                'hidden_dim': 64,
                'latent_dim': 32,
                'dropout': 0.2,
                'batch': 256,
                'epochs': 200,
                'lr': 1e-3,
                'patience': 20
            }
        },
        {
            'name': 'high_capacity',
            'params': {
                'hidden_dim': 128,
                'latent_dim': 64,
                'dropout': 0.3,
                'batch': 512,
                'epochs': 300,
                'lr': 5e-4,
                'patience': 25
            }
        }
    ]
    
    # Run tests
    successful_tests = 0
    total_tests = len(test_configs) * len(model_configs)
    
    for test_config in test_configs:
        for model_config in model_configs:
            print(f"\n🧪 Running: {test_config['description']} with {model_config['name']} model")
            
            # Create output directory
            output_dir = f"outputs/controlled_test_{test_config['mechanism']}_{int(test_config['rate']*100)}_{model_config['name']}"
            
            # Run test
            success = run_controlled_test(
                data_path=data_path,
                mechanism=test_config['mechanism'],
                rate=test_config['rate'],
                output_dir=output_dir,
                **model_config['params']
            )
            
            if success:
                successful_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        print(f"\n✅ Experiments completed! Check the 'outputs/' directory for results.")
        print("Each test creates a timestamped folder with:")
        print("  • controlled_test_results.png - Visualizations")
        print("  • controlled_test_results.json - Metrics and configuration")
        print("  • best_model.pth - Trained autoencoder model")
    else:
        print("\n❌ No experiments completed successfully. Check the error messages above.")

if __name__ == "__main__":
    main() 