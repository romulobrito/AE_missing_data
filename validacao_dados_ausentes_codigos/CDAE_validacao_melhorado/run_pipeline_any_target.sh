#!/bin/bash
# ==============================================================================
# ESTUDO COMPLEMENTAR: Validacao do CDAE em diferentes variaveis alvo
# ==============================================================================
# Este script executa o pipeline completo para uma variavel alvo alternativa,
# gerando resultados complementares que demonstram a generalidade do metodo.
#
# Author: Romulo Brito da Silva - UFRJ
# Date: 2025-12-01
#
# PROPOSITO:
#   Validar que a metodologia CDAE funciona para diferentes sensores,
#   nao apenas para o target principal (1251_FIT_801C_2).
#   Resultados sao salvos em pasta separada para uso no paper.
#
# USO:
#   ./run_pipeline_any_target.sh <TARGET_VARIABLE>
#
# EXEMPLOS:
#   ./run_pipeline_any_target.sh 1251_FIT_801B_2   # Recomendado (75.69% missing)
#   ./run_pipeline_any_target.sh 1251_FIT_801C_1   # Alternativa (69.43% missing)
#   ./run_pipeline_any_target.sh 1251_FIT_801D_1   # Alternativa (56.15% missing)
#
# VARIAVEIS CANDIDATAS (ordenadas por % missing):
#   1251_FIT_801C_2:  80.29% missing  <-- TARGET PRINCIPAL (paper)
#   1251_FIT_801B_2:  75.69% missing  <-- RECOMENDADO para estudo complementar
#   1251_FIT_801C_1:  69.43% missing
#   1251_FIT_801D_1:  56.15% missing
#   1251_PIT_811:     55.75% missing
#   1251_PIT_812:     55.69% missing
#   1251_FIT_802A:    40.21% missing
#
# RESULTADOS:
#   Salvos em: resultados_complementares/<TARGET>/
#   Inclui: CDAE treinado, validacao MCAR/MAR/MNAR, relatorio consolidado
# ==============================================================================

set -e  # Parar em caso de erro

# Verificar argumento
if [ -z "$1" ]; then
    echo "ERRO: Informe o nome da variavel alvo!"
    echo ""
    echo "USO: $0 <TARGET_VARIABLE>"
    echo ""
    echo "EXEMPLOS:"
    echo "  $0 1251_FIT_801C_2   # Target atual (80.29% missing)"
    echo "  $0 1251_FIT_801B_2   # Outro target (75.69% missing)"
    echo "  $0 1251_FIT_801C_1   # Outro target (69.43% missing)"
    exit 1
fi

TARGET="$1"

echo "=========================================================================="
echo "  PIPELINE COMPLETO - CDAE + VALIDACAO METODOLOGICA"
echo "  TARGET: $TARGET"
echo "=========================================================================="
echo ""

# Configuracoes
BASE_DIR="/home/romulo/Documentos/PEQ-POSDOC-UFRJ/abordagem-cs-dados-faltantes/validacao_dados_ausentes_codigos/CDAE_validacao_melhorado"
DATA_FILE="/home/romulo/Documentos/PEQ-POSDOC-UFRJ/abordagem-cs-dados-faltantes/sulfatos_dados_concatenados_formatados_2021.h5"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Sanitizar nome do target para usar em pastas (substituir caracteres especiais)
TARGET_SAFE=$(echo "$TARGET" | tr '/' '_' | tr ' ' '_')

# Entrar no diretorio de trabalho
cd "$BASE_DIR"

# ============================================================================
# PASSO 1: PREPARAR PASTA DE RESULTADOS COMPLEMENTARES
# ============================================================================
echo "=============================================="
echo "  PASSO 1/4: Preparando ambiente"
echo "=============================================="

# Pasta principal para resultados complementares
COMPLEMENTAR_BASE="resultados_complementares/${TARGET_SAFE}"
OUTPUT_CDAE="${COMPLEMENTAR_BASE}/cdae_${TIMESTAMP}"
TESTES_BASE="${COMPLEMENTAR_BASE}/validacao_metodologica"

if [ -d "$TESTES_BASE" ]; then
    echo "Removendo testes anteriores para $TARGET..."
    rm -rf "$TESTES_BASE"
fi
mkdir -p "$TESTES_BASE"
mkdir -p "$OUTPUT_CDAE"

echo "OK: Ambiente preparado"
echo "    Pasta principal: $COMPLEMENTAR_BASE"
echo ""

# ============================================================================
# PASSO 2: EXECUTAR CDAE.PY COM CONFIGURACAO OTIMA
# ============================================================================
echo "=============================================="
echo "  PASSO 2/4: Executando CDAE.py"
echo "=============================================="
echo "Configuracao:"
echo "  - Target: $TARGET"
echo "  - Split: Blocked-by-index (temporal)"
echo "  - Auto-skip: Sim (cobertura minima 20%)"
echo "  - Features: Selecao automatica (3-10 features)"
echo "  - Epocas: 100"
echo "  - Bootstrap: 1000 iteracoes"
echo "  - Downstream: Sim"
echo ""

python CDAE.py \
    --data "$DATA_FILE" \
    --target "$TARGET" \
    --output "$OUTPUT_CDAE" \
    --blocked_by_index \
    --auto_skip_until_coverage \
    --strict_temporal \
    --min_target_coverage 0.2 \
    --min_target_count 5000 \
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
    echo "ERRO: Falha na execucao do CDAE.py para target: $TARGET"
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
echo "  - Target: $TARGET"
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
    --min_target_count 5000 \
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
    --output "${TESTES_BASE}/MCAR_80pct" \
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
    --min_target_count 5000 \
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
    --output "${TESTES_BASE}/MAR_80pct" \
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
    --min_target_count 5000 \
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
    --output "${TESTES_BASE}/MNAR_80pct" \
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

python << EOF
import json
from pathlib import Path
import pandas as pd

target = "$TARGET"
base_dir = Path("$TESTES_BASE")
complementar_base = Path("$COMPLEMENTAR_BASE")
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
        data = json.load(f)
    
    results.append({
        'Target': target,
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

# Salvar em CSV na pasta de validacao
output_csv = base_dir / "RESULTADOS_CONSOLIDADOS_80PCT.csv"
df_results.to_csv(output_csv, index=False)

# Salvar tambem na pasta principal do estudo complementar
output_csv_main = complementar_base / "RESULTADOS_ESTUDO_COMPLEMENTAR.csv"
df_results.to_csv(output_csv_main, index=False)

print(f"OK: Resultados salvos em:")
print(f"    - {output_csv}")
print(f"    - {output_csv_main}")

# Exibir tabela
print("\n" + "="*80)
print(f"  ESTUDO COMPLEMENTAR - TARGET: {target}")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

# Criar README do estudo complementar
readme_content = f'''# Estudo Complementar: {target}

## Objetivo
Validar que a metodologia CDAE funciona para diferentes sensores,
demonstrando a generalidade do metodo proposto.

## Target Principal do Paper
- Variavel: 1251_FIT_801C_2
- Missing: 80.29%

## Target deste Estudo Complementar
- Variavel: {target}
- Resultados em: validacao_metodologica/

## Arquivos Gerados
- cdae_*/: Modelo CDAE treinado e artefatos
- validacao_metodologica/: Testes MCAR, MAR, MNAR (80% missing)
- RESULTADOS_ESTUDO_COMPLEMENTAR.csv: Metricas consolidadas

## Como Citar
Este estudo complementar demonstra que a metodologia proposta
generaliza para diferentes variaveis alvo com alto percentual
de dados ausentes.
'''

readme_path = complementar_base / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"\nREADME criado: {readme_path}")

EOF

echo ""
echo "=========================================================================="
echo "  ESTUDO COMPLEMENTAR FINALIZADO COM SUCESSO!"
echo "  TARGET: $TARGET"
echo "=========================================================================="
echo ""
echo "PASTA PRINCIPAL:"
echo "  $COMPLEMENTAR_BASE"
echo ""
echo "ESTRUTURA DE RESULTADOS:"
echo "  resultados_complementares/"
echo "  └── ${TARGET_SAFE}/"
echo "      ├── README.md                           <- Descricao do estudo"
echo "      ├── RESULTADOS_ESTUDO_COMPLEMENTAR.csv  <- Metricas consolidadas"
echo "      ├── cdae_${TIMESTAMP}/                  <- Modelo CDAE"
echo "      │   ├── complete_pipeline_results.json"
echo "      │   ├── feature_selection_config.json"
echo "      │   ├── temporal_comparison_plot_cdae.png"
echo "      │   └── ..."
echo "      └── validacao_metodologica/"
echo "          ├── MCAR_80pct/                     <- Teste MCAR"
echo "          ├── MAR_80pct/                      <- Teste MAR"
echo "          ├── MNAR_80pct/                     <- Teste MNAR"
echo "          └── RESULTADOS_CONSOLIDADOS_80PCT.csv"
echo ""
echo "PROXIMO PASSO:"
echo "  Use estes resultados como estudo complementar no paper"
echo "  para demonstrar generalidade da metodologia."
echo ""

