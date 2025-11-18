# Virtual Sensor (CDAE) - Resumo do Case

## Resultados principais (blocked auto-skip)

| MAE | RMSE | R^2 | PR-AUC (base -> VS) | Prec@Rec=0.8 (base -> VS) |
|---:|---:|---:|---:|---:|
| 3.237 | 4.665 | 0.919 | 0.750 -> 0.941 | 0.200 -> 0.935 |

## Cobertura e split (proxy temporal)

- Split: `blocked_by_index` com auto-skip
- Treino: cobertura 0.205 (11746/57432)
- Val: cobertura 0.923 (5892/6382)
- Teste: cobertura 0.910 (3058/3359)

## Grafico temporal

![](outputs/cdae_blocked_auto_final/temporal_comparison_plot_cdae.png)

## Configuracao e reproducibilidade

- Seed: 42; Device: CPU
- Treino: batch=256, epochs=100 (early stop ativo)
- CDAE: latent=16, dropout=0.2, p_mask=0.5, masked_weight=1.0
- Scalers: X=StandardScaler, y=StandardScaler
- Split: `blocked_by_index`; test_size=0.05, val_size=0.1; auto_skip_until_coverage=True
- Cobertura (train/val/test): 0.205/0.923/0.910
- Presentes (train/val/test): 11746/5892/3058
- Artefatos: ver `outputs/cdae_blocked_auto_final/` (args, coverage, escalers, figura temporal)
