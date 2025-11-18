# CDAE - Arquivos Backup

Pasta criada em: 2025-01-18

## Conteudo

Esta pasta contem copias dos 3 arquivos que implementam o **CDAE (Conditional Denoising Autoencoder)**.

---

## Arquivos

### 1. `CDAE.py`
- **Origem**: `Tractian/CDAE.py`
- **Status**: ✅ **MAIS RECENTE E COMPLETO**
- **Funcao**: Pipeline completo para imputacao em dados reais
- **Arquitetura**: ConditionalDenoisingAE - Input `[X; t_tilde; m_t]` → Output `t_hat`
- **Melhorias implementadas**:
  - Split temporal leakage-safe
  - Auto-skip para garantir cobertura minima do target
  - Imputacao leakage-safe (medias do train only)
  - Selecao de features leakage-safe
  - Bootstrap com intervalos de confianca
  - Experimento downstream (simula alertas)
  - Stress test com multiplos p_mask
  - Coverage report
  - Artefatos reprodutiveis
  - Plot temporal comparativo

### 2. `teste_controlado_cdae_improved.py`
- **Origem**: `teste_controlado_cdae_improved.py`
- **Status**: ⚠️ **REQUER PORT DAS MELHORIAS** do `CDAE.py`
- **Funcao**: Teste controlado MCAR/MAR/MNAR com CDAE
- **Arquitetura**: ConditionalDenoisingAE (mesma do CDAE.py)
- **Uso**: Validacao da metodologia com dados controlados
- **Importa de**: `teste_controlado_dados_faltantes_improved.py`

### 3. `cs_regressor_with_eda.py`
- **Origem**: `cs_regressor_with_eda.py`
- **Status**: ⚠️ **DESATUALIZADO** - substituido por `CDAE.py`
- **Funcao**: Pipeline EDA + selecao + CDAE (versao anterior)
- **Arquitetura**: ConditionalDenoisingAE

---

## Comparacao: CDAE vs Autoencoder Simples

| Aspecto | **CDAE** | **Autoencoder Simples** |
|---------|----------|------------------------|
| **Input** | `[X; t_tilde; m_t]` | `X` apenas |
| **Output** | `t_hat` (reconstruido) | `t` |
| **Mascaramento** | Sim (p_mask durante treino) | Nao |
| **Condicional** | Sim (condiciona em mascara) | Nao |
| **Perda Ponderada** | Sim (maior peso em masked) | Nao |
| **Robustez a Missing** | Alta | Moderada |

---

## Melhorias a Portar

As seguintes melhorias do `CDAE.py` devem ser portadas para `teste_controlado_cdae_improved.py`:

1. ✅ Split temporal leakage-safe
2. ✅ Auto-skip para garantir cobertura minima
3. ✅ Imputacao leakage-safe
4. ✅ Selecao de features leakage-safe
5. ✅ Bootstrap com CIs
6. ✅ Experimento downstream
7. ✅ Stress test com multiplos p_mask
8. ✅ Coverage report
9. ✅ Artefatos reprodutiveis
10. ✅ Plot temporal comparativo

---

## Como Usar

### Dados Reais (Producao)
```bash
python CDAE.py \
    --data "caminho/dados.h5" \
    --target "1251_FIT_801C_2" \
    --blocked_by_index \
    --auto_skip_until_coverage \
    --min_target_coverage 0.2 \
    --strict_temporal \
    --downstream_experiment \
    --output "outputs/cdae_final"
```

### Validacao Controlada (MCAR/MAR/MNAR)
```bash
python teste_controlado_cdae_improved.py \
    --data "caminho/dados.h5" \
    --mechanism MNAR \
    --missing_pct 30 \
    --output "outputs/validacao_cdae"
```

---

## Referencias

- Pipeline principal: `Tractian/CDAE.py`
- Documentacao: `COMO_EXECUTAR.md`
- Comparacao de pipelines: `README_pipeline_comparison.md`

---

**Autor**: Romulo Brito da Silva - UFRJ  
**Data**: 2025-01-18

