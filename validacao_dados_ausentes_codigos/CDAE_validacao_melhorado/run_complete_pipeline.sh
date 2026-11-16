#!/bin/bash
# Pipeline completo: CDAE.py + Validacao Metodologica (MCAR/MAR/MNAR com 80% missing)
# Author: Romulo Brito da Silva - UFRJ
# Date: 2025-11-18

set -e  # Parar em caso de erro

echo "=========================================================================="
echo "  PIPELINE COMPLETO - CDAE + VALIDACAO METODOLOGICA (80% MISSING)"
echo "=========================================================================="
echo ""

# Configuracoes
BASE_DIR="/home/romulo/Documentos/PEQ-POSDOC-UFRJ/abordagem-cs-dados-faltantes/validacao_dados_ausentes_codigos/CDAE_validacao_melhorado"
DATA_FILE="/home/romulo/Documentos/PEQ-POSDOC-UFRJ/abordagem-cs-dados-faltantes/sulfatos_dados_concatenados_formatados_2021.h5"
TARGET="1251_FIT_801C_2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Entrar no diretorio de trabalho
cd "$BASE_DIR"

# ============================================================================
# PASSO 1: LIMPAR PASTA TESTES
# ============================================================================
echo "=============================================="
echo "  PASSO 1/4: Limpando pasta testes"
echo "=============================================="
TESTES_BASE="testes/publication_protocol_${TIMESTAMP}"
export TESTES_BASE
mkdir -p "$TESTES_BASE"
echo "Resultados desta execucao: $TESTES_BASE (historicos preservados)"
echo ""

# ============================================================================
# PASSO 2: EXECUTAR CDAE.PY COM CONFIGURACAO OTIMA
# ============================================================================
echo "=============================================="
echo "  PASSO 2/4: Executando CDAE.py"
echo "=============================================="
echo "Configuracao:"
echo "  - Split: Blocked-by-index (temporal)"
echo "  - Auto-skip: Sim (cobertura minima 20%)"
echo "  - Features: Selecao automatica (10 features)"
echo "  - Epocas: 100"
echo "  - Bootstrap: 1000 iteracoes"
echo "  - Downstream: Sim"
echo ""

OUTPUT_CDAE="outputs/cdae_optimal_${TIMESTAMP}"

python CDAE.py \
    --data "$DATA_FILE" \
    --target "$TARGET" \
    --output "$OUTPUT_CDAE" \
    --blocked_by_index \
    --auto_skip_until_coverage \
    --strict_temporal \
    --min_target_coverage 0.2 \
    --min_target_count 10000 \
    --max_skip_head_frac 0.95 \
    --time_test_size 0.05 \
    --time_val_size 0.1 \
    --min_correlation 0.3 \
    --max_missing_pct 30.0 \
    --min_features 3 \
    --max_features 10 \
    --batch 256 \
    --latent 16 \
    --dropout 0.2 \
    --epochs 100 \
    --lr 0.001 \
    --patience 20 \
    --p_mask 0.5 \
    --masked_weight 1.0 \
    --bootstrap_n 1000 \
    --downstream_experiment \
    --downstream_missing_pct 0.3 \
    --downstream_recall_target 0.8 \
    --downstream_label_percentile 80.0 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "OK: CDAE.py executado com sucesso!"
    echo "    Output: $OUTPUT_CDAE"
    echo "    Feature config: $OUTPUT_CDAE/feature_selection_config.json"
else
    echo "ERRO: Falha na execucao do CDAE.py"
    exit 1
fi
echo ""

# ============================================================================
# PASSO 3: EXECUTAR VALIDACAO METODOLOGICA (80% MISSING)
# ============================================================================
echo "=============================================="
echo "  PASSO 3/4: Validacao Metodologica"
echo "=============================================="
echo "Configuracao:"
echo "  - Missing rate: 80%"
echo "  - Mecanismos: MCAR, MAR, MNAR"
echo "  - Features: Reutilizando do CDAE.py"
echo "  - Epocas: 50"
echo "  - Bootstrap: 1000 iteracoes"
echo ""

FEATURE_CONFIG="$OUTPUT_CDAE/feature_selection_config.json"

# Verificar se o arquivo de configuracao existe
if [ ! -f "$FEATURE_CONFIG" ]; then
    echo "ERRO: Feature config nao encontrado: $FEATURE_CONFIG"
    exit 1
fi

# Teste 1: MCAR (80% missing)
echo "----------------------------------------------"
echo "  Teste 1/3: MCAR (80% missing)"
echo "----------------------------------------------"
python teste_controlado_cdae_v2.py \
    --blocked_by_index \
    --auto_skip_until_coverage \
    --min_target_coverage 0.2 \
    --min_target_count 10000 \
    --max_skip_head_frac 0.95 \
    --time_test_size 0.05 \
    --time_val_size 0.1 \
    --strict_temporal \
    --data "$DATA_FILE" \
    --feature_config "$FEATURE_CONFIG" \
    --mechanism MCAR \
    --missing_rate 0.8 \
    --epochs 50 \
    --batch 128 \
    --latent 16 \
    --bootstrap_n 1000 \
    --output "$TESTES_BASE/MCAR_80pct" \
    --seed 42

if [ $? -eq 0 ]; then
    echo "OK: Teste MCAR concluido!"
else
    echo "ERRO: Falha no teste MCAR"
    exit 1
fi
echo ""

# Teste 2: MAR (80% missing)
echo "----------------------------------------------"
echo "  Teste 2/3: MAR (80% missing)"
echo "----------------------------------------------"
python teste_controlado_cdae_v2.py \
    --blocked_by_index \
    --auto_skip_until_coverage \
    --min_target_coverage 0.2 \
    --min_target_count 10000 \
    --max_skip_head_frac 0.95 \
    --time_test_size 0.05 \
    --time_val_size 0.1 \
    --strict_temporal \
    --data "$DATA_FILE" \
    --feature_config "$FEATURE_CONFIG" \
    --mechanism MAR \
    --missing_rate 0.8 \
    --epochs 50 \
    --batch 128 \
    --latent 16 \
    --bootstrap_n 1000 \
    --output "$TESTES_BASE/MAR_80pct" \
    --seed 42

if [ $? -eq 0 ]; then
    echo "OK: Teste MAR concluido!"
else
    echo "ERRO: Falha no teste MAR"
    exit 1
fi
echo ""

# Teste 3: MNAR (80% missing)
echo "----------------------------------------------"
echo "  Teste 3/3: MNAR (80% missing)"
echo "----------------------------------------------"
python teste_controlado_cdae_v2.py \
    --blocked_by_index \
    --auto_skip_until_coverage \
    --min_target_coverage 0.2 \
    --min_target_count 10000 \
    --max_skip_head_frac 0.95 \
    --time_test_size 0.05 \
    --time_val_size 0.1 \
    --strict_temporal \
    --data "$DATA_FILE" \
    --feature_config "$FEATURE_CONFIG" \
    --mechanism MNAR \
    --missing_rate 0.8 \
    --epochs 50 \
    --batch 128 \
    --latent 16 \
    --bootstrap_n 1000 \
    --output "$TESTES_BASE/MNAR_80pct" \
    --seed 42

if [ $? -eq 0 ]; then
    echo "OK: Teste MNAR concluido!"
else
    echo "ERRO: Falha no teste MNAR"
    exit 1
fi
echo ""

# ============================================================================
# PASSO 4: GERAR RELATORIO CONSOLIDADO
# ============================================================================
echo "=============================================="
echo "  PASSO 4/4: Gerando relatorio consolidado"
echo "=============================================="

python << 'EOF'
import json
import os
from pathlib import Path
import pandas as pd

base_dir = Path(os.environ["TESTES_BASE"])
mechanisms = ["MCAR_80pct", "MAR_80pct", "MNAR_80pct"]

results = []
for mech_dir in mechanisms:
    mech_name = mech_dir.split("_")[0]
    result_dir = base_dir / mech_dir
    
    # Encontrar o subdiretorio de resultados
    subdirs = list(result_dir.glob("controlled_cdae_v2_*"))
    if not subdirs:
        print(f"AVISO: Nenhum resultado encontrado para {mech_name}")
        continue
    
    # Pegar o mais recente
    result_file = subdirs[-1] / "controlled_test_results.json"
    if not result_file.exists():
        print(f"AVISO: Arquivo de resultados nao encontrado para {mech_name}")
        continue
    
    # Ler resultados
    with open(result_file, 'r') as f:
        data = json.load(f)["metrics"]
    
    results.append({
        'Mecanismo': mech_name,
        'Missing_Rate': f"{data.get('missing_rate', 0)*100:.0f}%",
        'MAE': f"{data.get('MAE', 0):.4f}",
        'MAE_CI': f"[{data.get('MAE_CI_low', 0):.4f}, {data.get('MAE_CI_high', 0):.4f}]",
        'RMSE': f"{data.get('RMSE', 0):.4f}",
        'RMSE_CI': f"[{data.get('RMSE_CI_low', 0):.4f}, {data.get('RMSE_CI_high', 0):.4f}]",
        'R2': f"{data.get('R2', 0):.4f}",
        'KS_stat': f"{data.get('KS_statistic', 0):.4f}",
        'KS_pvalue': f"{data.get('KS_pvalue', 0):.6f}",
    })

# Criar DataFrame e salvar
df_results = pd.DataFrame(results)

# Salvar em CSV
output_csv = base_dir / "RESULTADOS_CONSOLIDADOS_80PCT.csv"
df_results.to_csv(output_csv, index=False)
print(f"OK: Resultados salvos em {output_csv}")

# Exibir tabela
print("\n" + "="*80)
print("  RESULTADOS CONSOLIDADOS - VALIDACAO METODOLOGICA (80% MISSING)")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

EOF

echo ""
echo "=========================================================================="
echo "  PIPELINE COMPLETO FINALIZADO COM SUCESSO!"
echo "=========================================================================="
echo ""
echo "Resultados:"
echo "  - CDAE.py: $OUTPUT_CDAE"
echo "  - Validacao MCAR: $TESTES_BASE/MCAR_80pct/"
echo "  - Validacao MAR: $TESTES_BASE/MAR_80pct/"
echo "  - Validacao MNAR: $TESTES_BASE/MNAR_80pct/"
echo "  - Relatorio consolidado: $TESTES_BASE/RESULTADOS_CONSOLIDADOS_80PCT.csv"
echo ""
echo "Para visualizar plots:"
echo "  - CDAE: $OUTPUT_CDAE/temporal_comparison_plot_cdae.png"
echo "  - MCAR: $TESTES_BASE/MCAR_80pct/controlled_cdae_v2_*/controlled_test_results.png"
echo "  - MAR: $TESTES_BASE/MAR_80pct/controlled_cdae_v2_*/controlled_test_results.png"
echo "  - MNAR: $TESTES_BASE/MNAR_80pct/controlled_cdae_v2_*/controlled_test_results.png"
echo ""

