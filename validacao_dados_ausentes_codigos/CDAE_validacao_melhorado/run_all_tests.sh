#!/bin/bash
# Script para executar testes completos com MCAR, MAR e MNAR
# Author: Romulo Brito da Silva - UFRJ
# Date: 2025-01-18

echo "=============================================="
echo "  TESTES COMPLETOS - MCAR, MAR, MNAR"
echo "=============================================="
echo ""

# Configuracoes
DATA_FILE="/home/romulo/Documentos/PEQ-POSDOC-UFRJ/abordagem-cs-dados-faltantes/sulfatos_dados_concatenados_formatados_2021.h5"
FEATURE_CONFIG="../Tractian/outputs/cdae_blocked_auto_final/feature_selection_config.json"
BASE_DIR="/home/romulo/Documentos/PEQ-POSDOC-UFRJ/abordagem-cs-dados-faltantes/validacao_dados_ausentes_codigos/CDAE_validacao_melhorado"

# Criar pasta testes
mkdir -p "$BASE_DIR/testes"

cd "$BASE_DIR"

# TESTE 1: MCAR (30% missing)
echo "=============================================="
echo "  TESTE 1/3: MCAR (30% missing)"
echo "=============================================="
echo ""
python teste_controlado_cdae_v2.py \
    --data "$DATA_FILE" \
    --feature_config "$FEATURE_CONFIG" \
    --mechanism MCAR \
    --missing_rate 0.3 \
    --epochs 15 \
    --batch 128 \
    --latent 16 \
    --bootstrap_n 1000 \
    --output "testes/MCAR_final" \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "OK: Teste MCAR concluido!"
    echo ""
else
    echo ""
    echo "ERRO: Teste MCAR falhou!"
    echo ""
fi

# TESTE 2: MAR (30% missing) + Downstream
echo "=============================================="
echo "  TESTE 2/3: MAR (30% missing + Downstream)"
echo "=============================================="
echo ""
python teste_controlado_cdae_v2.py \
    --data "$DATA_FILE" \
    --feature_config "$FEATURE_CONFIG" \
    --mechanism MAR \
    --missing_rate 0.3 \
    --epochs 15 \
    --batch 128 \
    --latent 16 \
    --bootstrap_n 1000 \
    --downstream_experiment \
    --downstream_missing_pct 0.3 \
    --downstream_recall_target 0.8 \
    --downstream_label_percentile 80.0 \
    --output "testes/MAR_final" \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "OK: Teste MAR concluido!"
    echo ""
else
    echo ""
    echo "ERRO: Teste MAR falhou!"
    echo ""
fi

# TESTE 3: MNAR (40% missing)
echo "=============================================="
echo "  TESTE 3/3: MNAR (40% missing)"
echo "=============================================="
echo ""
python teste_controlado_cdae_v2.py \
    --data "$DATA_FILE" \
    --feature_config "$FEATURE_CONFIG" \
    --mechanism MNAR \
    --missing_rate 0.4 \
    --epochs 15 \
    --batch 128 \
    --latent 16 \
    --bootstrap_n 1000 \
    --output "testes/MNAR_final" \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "OK: Teste MNAR concluido!"
    echo ""
else
    echo ""
    echo "ERRO: Teste MNAR falhou!"
    echo ""
fi

# Resumo final
echo ""
echo "=============================================="
echo "  RESUMO DOS TESTES"
echo "=============================================="
echo ""
echo "Verificando resultados gerados..."
echo ""

# Contar plots gerados
TOTAL_PLOTS=$(find testes/ -name "*.png" 2>/dev/null | wc -l)
TOTAL_JSONS=$(find testes/ -name "controlled_test_results.json" 2>/dev/null | wc -l)
TOTAL_MODELS=$(find testes/ -name "best_model.pth" 2>/dev/null | wc -l)

echo "Plots gerados: $TOTAL_PLOTS"
echo "JSONs de resultados: $TOTAL_JSONS"
echo "Modelos salvos: $TOTAL_MODELS"
echo ""

if [ $TOTAL_JSONS -eq 3 ]; then
    echo "OK: Todos os 3 testes foram concluidos com sucesso!"
    echo ""
    echo "Resultados salvos em:"
    echo "  - testes/MCAR_final/"
    echo "  - testes/MAR_final/"
    echo "  - testes/MNAR_final/"
else
    echo "AVISO: Apenas $TOTAL_JSONS de 3 testes foram concluidos."
fi

echo ""
echo "=============================================="
echo "  TESTES FINALIZADOS!"
echo "=============================================="

