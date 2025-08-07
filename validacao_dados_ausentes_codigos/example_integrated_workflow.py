#!/usr/bin/env python3
"""
example_integrated_workflow.py ─────────────────────────────────────────────
Example workflow demonstrating integration between cs_ae_with_eda.py and 
teste_controlado_dados_faltantes_improved.py through feature configuration files.

This script demonstrates:
1. Running EDA + Feature Selection with cs_ae_with_eda.py
2. Exporting feature configuration
3. Using exported config in controlled tests with teste_controlado_dados_faltantes_improved.py
4. Comparing results with default vs automatic feature selection

Author: Romulo Brito da Silva - UFRJ
Date: 2025-08-07
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✅ {description} - COMPLETED SUCCESSFULLY!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED!")
        print(f"Error: {e}")
        return False

def main():
    """Demonstrate integrated workflow."""
    
    print("🎯 INTEGRATED WORKFLOW DEMONSTRATION")
    print("=" * 70)
    print("This script demonstrates the integration between:")
    print("• cs_ae_with_eda.py (EDA + Feature Selection + Training)")
    print("• teste_controlado_dados_faltantes_improved.py (Controlled Testing)")
    print()
    
    # Data path - adjust as needed
    data_path = "/home/romulo/Downloads/sulfatos_dados_concatenados_formatados_2021.h5"
    target_var = "1251_FIT_801C_2"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable in this script.")
        return
    
    # Base output directory
    base_output = "outputs/integrated_workflow"
    
    # Step 1: Run automatic feature selection with cs_ae_with_eda.py
    step1_cmd = [
        sys.executable, "cs_ae_with_eda.py",
        "--data", data_path,
        "--target", target_var,
        "--min_correlation", "0.25",
        "--max_features", "7",
        "--epochs", "50",
        "--batch", "64",
        "--output", f"{base_output}/step1_auto_selection"
    ]
    
    if not run_command(step1_cmd, "Step 1: Automatic Feature Selection"):
        return
    
    # Step 2: Run manual feature selection with cs_ae_with_eda.py
    step2_cmd = [
        sys.executable, "cs_ae_with_eda.py",
        "--data", data_path,
        "--target", target_var,
        "--manual_features",
        "--features", "features_1251_FIT_801C_2.txt",
        "--epochs", "50",
        "--batch", "64",
        "--output", f"{base_output}/step2_manual_selection"
    ]
    
    if not run_command(step2_cmd, "Step 2: Manual Feature Selection"):
        return
    
    # Step 3: Controlled test with automatic features
    step3_cmd = [
        sys.executable, "teste_controlado_dados_faltantes_improved.py",
        "--data", data_path,
        "--feature_config", f"{base_output}/step1_auto_selection/feature_selection_config.json",
        "--mechanism", "MNAR",
        "--rate", "0.6",
        "--epochs", "100",
        "--batch", "32",
        "--output", f"{base_output}/step3_controlled_auto"
    ]
    
    if not run_command(step3_cmd, "Step 3: Controlled Test (Auto Features)"):
        return
    
    # Step 4: Controlled test with manual features
    step4_cmd = [
        sys.executable, "teste_controlado_dados_faltantes_improved.py",
        "--data", data_path,
        "--feature_config", f"{base_output}/step2_manual_selection/feature_selection_config.json",
        "--mechanism", "MNAR",
        "--rate", "0.6",
        "--epochs", "100",
        "--batch", "32",
        "--output", f"{base_output}/step4_controlled_manual"
    ]
    
    if not run_command(step4_cmd, "Step 4: Controlled Test (Manual Features)"):
        return
    
    # Step 5: Controlled test with default features (no config)
    step5_cmd = [
        sys.executable, "teste_controlado_dados_faltantes_improved.py",
        "--data", data_path,
        "--mechanism", "MNAR",
        "--rate", "0.6",
        "--epochs", "100",
        "--batch", "32",
        "--output", f"{base_output}/step5_controlled_default"
    ]
    
    if not run_command(step5_cmd, "Step 5: Controlled Test (Default Features)"):
        return
    
    # Step 6: Compare different missing mechanisms with auto features
    mechanisms = ['MCAR', 'MAR', 'MNAR']
    rates = [0.3, 0.5, 0.7]
    
    for mechanism in mechanisms:
        for rate in rates:
            step_cmd = [
                sys.executable, "teste_controlado_dados_faltantes_improved.py",
                "--data", data_path,
                "--feature_config", f"{base_output}/step1_auto_selection/feature_selection_config.json",
                "--mechanism", mechanism,
                "--rate", str(rate),
                "--epochs", "50",
                "--batch", "32",
                "--output", f"{base_output}/comparison_{mechanism}_{int(rate*100)}"
            ]
            
            if not run_command(step_cmd, f"Comparison: {mechanism} {rate*100:.0f}%"):
                print(f"⚠️  Failed comparison for {mechanism} {rate*100:.0f}%, continuing...")
    
    # Summary
    print(f"\n{'='*70}")
    print("🎉 INTEGRATED WORKFLOW COMPLETED!")
    print(f"{'='*70}")
    print("Results structure:")
    print(f"📁 {base_output}/")
    print("  ├── step1_auto_selection/          # EDA + Automatic feature selection")
    print("  ├── step2_manual_selection/        # EDA + Manual feature selection")
    print("  ├── step3_controlled_auto/         # Controlled test (auto features)")
    print("  ├── step4_controlled_manual/       # Controlled test (manual features)")
    print("  ├── step5_controlled_default/      # Controlled test (default features)")
    print("  └── comparison_*/                  # Various mechanism comparisons")
    print()
    print("Key files to check:")
    print("• feature_selection_config.json     # Feature configurations")
    print("• controlled_test_results.json      # Test metrics")
    print("• temporal_comparison_plot.png      # Temporal visualizations")
    print("• controlled_test_results.png       # Test result visualizations")
    print()
    print("This demonstrates:")
    print("✅ Feature selection consistency between scripts")
    print("✅ Automatic vs manual feature comparison")
    print("✅ Different missing data mechanisms")
    print("✅ Comprehensive controlled validation")

if __name__ == "__main__":
    main() 