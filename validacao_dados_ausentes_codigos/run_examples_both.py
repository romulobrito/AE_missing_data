#!/usr/bin/env python3
"""
run_examples_both.py ────────────────────────────────────────────────────────
Exemplos de uso para ambas as versões do pipeline de autoencoder:

1. cs_ae_enhanced.py    - Assume AED já realizada (features pré-selecionadas)
2. cs_ae_with_eda.py    - Pipeline completo com AED automática

Este script demonstra quando usar cada abordagem e suas diferenças.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd_list, description):
    """Executa um comando e exibe os resultados."""
    print(f"\n🧪 {description}")
    print("=" * len(description))
    print(f"Comando: {' '.join(cmd_list)}")
    print()
    
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        print("✅ Sucesso!")
        if result.stdout:
            # Mostrar apenas as últimas linhas para não sobrecarregar
            lines = result.stdout.strip().split('\n')
            if len(lines) > 20:
                print("Saída (últimas 20 linhas):")
                print('\n'.join(lines[-20:]))
            else:
                print("Saída:")
                print(result.stdout)
        
        if result.stderr:
            print("\nAvisos:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro na execução:")
        print(f"Código de saída: {e.returncode}")
        if e.stdout:
            print("Saída padrão:")
            print(e.stdout)
        if e.stderr:
            print("Erro:")
            print(e.stderr)
    except FileNotFoundError:
        print("❌ Arquivo não encontrado no diretório atual")

def main():
    """Demonstra os dois modos de uso do pipeline."""
    
    print("🔬 COMPARAÇÃO DOS PIPELINES DE AUTOENCODER")
    print("═" * 55)
    print()
    print("Este exemplo demonstra duas abordagens:")
    print()
    print("1️⃣  cs_ae_enhanced.py     - Features pré-selecionadas (AED já feita)")
    print("2️⃣  cs_ae_with_eda.py     - Pipeline completo com AED automática")
    print()
    
    # Verificar se o arquivo de dados existe
    data_path = Path("/home/romulo/Downloads/sulfatos_dados_concatenados_formatados_2021.h5")
    if not data_path.exists():
        print(f"❌ Arquivo de dados não encontrado: {data_path}")
        print("📋 Para este exemplo, você precisa:")
        print("   1. Baixar/mover o arquivo de dados para o local correto")
        print("   2. Ou ajustar o caminho no script")
        return
    
    print("📊 Dados encontrados! Iniciando comparação...")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODO 1: AED JÁ REALIZADA (cs_ae_enhanced.py)
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 80)
    print("🎯 MODO 1: FEATURES PRÉ-SELECIONADAS (AED JÁ REALIZADA)")
    print("═" * 80)
    print()
    print("👍 Use quando:")
    print("   • Já fez análise exploratória no notebook")
    print("   • Conhece quais features usar")
    print("   • Quer apenas treinar o autoencoder")
    print("   • Tem arquivo com lista de features")
    print()
    
    # Exemplo 1A: Com arquivo de features
    cmd1a = [
        sys.executable, "cs_ae_enhanced.py",
        "--data", str(data_path),
        "--target", "1251_FIT_801C_2",
        "--features", "features_1251_FIT_801C_2.txt",
        "--epochs", "50",  # Reduzido para demo
        "--output", "outputs/mode1_with_features"
    ]
    
    run_command(cmd1a, "EXEMPLO 1A: Features pré-selecionadas (arquivo)")
    
    # Exemplo 1B: Configuração padrão (hardcoded para sulfatos)
    cmd1b = [
        sys.executable, "cs_ae_enhanced.py",
        "--data", str(data_path),
        "--target", "1251_FIT_801C_2",
        "--epochs", "50",  # Reduzido para demo
        "--robust_target",
        "--output", "outputs/mode1_default"
    ]
    
    run_command(cmd1b, "EXEMPLO 1B: Configuração padrão (hardcoded)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODO 2: PIPELINE COMPLETO COM AED (cs_ae_with_eda.py)
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 80)
    print("🔍 MODO 2: PIPELINE COMPLETO COM AED AUTOMÁTICA")
    print("═" * 80)
    print()
    print("👍 Use quando:")
    print("   • Ainda não fez análise exploratória")
    print("   • Quer seleção automática de features")
    print("   • Quer pipeline end-to-end")
    print("   • Explorar diferentes critérios de seleção")
    print()
    
    # Exemplo 2A: Seleção automática com critérios padrão
    cmd2a = [
        sys.executable, "cs_ae_with_eda.py",
        "--data", str(data_path),
        "--target", "1251_FIT_801C_2",
        "--epochs", "50",  # Reduzido para demo
        "--output", "outputs/mode2_auto_default"
    ]
    
    run_command(cmd2a, "EXEMPLO 2A: Seleção automática (critérios padrão)")
    
    # Exemplo 2B: Seleção automática com critérios rigorosos
    cmd2b = [
        sys.executable, "cs_ae_with_eda.py",
        "--data", str(data_path),
        "--target", "1251_FIT_801C_2",
        "--min_correlation", "0.5",
        "--max_missing_pct", "20.0",
        "--min_features", "3",
        "--max_features", "6",
        "--epochs", "50",  # Reduzido para demo
        "--robust_target",
        "--output", "outputs/mode2_auto_strict"
    ]
    
    run_command(cmd2b, "EXEMPLO 2B: Seleção automática (critérios rigorosos)")
    
    # Exemplo 2C: Usando features manuais (mas com AED)
    cmd2c = [
        sys.executable, "cs_ae_with_eda.py",
        "--data", str(data_path),
        "--target", "1251_FIT_801C_2",
        "--manual_features",
        "--features", "features_1251_FIT_801C_2.txt",
        "--epochs", "50",  # Reduzido para demo
        "--output", "outputs/mode2_manual"
    ]
    
    run_command(cmd2c, "EXEMPLO 2C: Features manuais + AED automática")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESUMO E RECOMENDAÇÕES
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 80)
    print("📋 RESUMO E RECOMENDAÇÕES")
    print("═" * 80)
    print()
    print("📁 Resultados salvos em:")
    print("   outputs/mode1_*     - Resultados do Modo 1 (cs_ae_enhanced.py)")
    print("   outputs/mode2_*     - Resultados do Modo 2 (cs_ae_with_eda.py)")
    print()
    print("🔍 Para comparar resultados:")
    print("   • Veja os arquivos *_results.json em cada diretório")
    print("   • Compare as métricas MAE, RMSE, R²")
    print("   • Analise os gráficos gerados")
    print()
    print("💡 Recomendações de uso:")
    print()
    print("   🎯 cs_ae_enhanced.py quando:")
    print("      • Workflow de pesquisa (notebook → script)")
    print("      • Features já validadas")
    print("      • Foco apenas no treinamento")
    print("      • Execução mais rápida")
    print()
    print("   🔍 cs_ae_with_eda.py quando:")
    print("      • Novo dataset (primeira análise)")
    print("      • Exploração de features")
    print("      • Pipeline automático end-to-end")
    print("      • Relatórios completos necessários")
    print()
    print("✅ Ambos os modos produzem resultados equivalentes!")
    print("   A escolha depende do seu workflow e necessidades específicas.")

if __name__ == "__main__":
    main() 