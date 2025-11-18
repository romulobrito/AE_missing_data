# Resultados CDAE

## Regressao: Controle vs Temporal

| Split | MAE | RMSE | R^2 | CI(MAE) | CI(RMSE) |
|---|---:|---:|---:|---|---|
| Random (controle) | 0.920 | 1.449 | 0.998 | [0.882, 0.962] | [1.279, 1.630] |
| Temporal (blocked) | 7.933 | 9.299 | 0.914 | [7.840, 8.033] | [9.203, 9.395] |

## Robustez vs p_mask

| p_mask | MAE | RMSE | R^2 |
|---:|---:|---:|---:|
| 0.100 | 0.859 | 1.356 | 0.998 |
| 0.300 | 0.741 | 1.190 | 0.999 |
| 0.500 | 0.765 | 1.241 | 0.998 |
| 0.700 | 0.647 | 1.070 | 0.999 |

## Impacto no alerta (downstream)

| Metrica | Baseline | Virtual Sensor (t_hat) |
|---|---:|---:|
| PR-AUC | 0.750 | 0.941 |
| Precision @ Recall | 0.200 | 0.935 |

## Cobertura do alvo por split (strict temporal)

| Split | Cobertura | Presentes | Total |
|---|---:|---:|---:|
| train | 0.024 | 2034.000 | 85147.000 |
| val | 0.956 | 9045.000 | 9461.000 |
| test | 0.917 | 9636.000 | 10513.000 |


## Split bloqueado com auto-skip (novo)

| Split | MAE | RMSE | R^2 | CI(MAE) | CI(RMSE) |
|---|---:|---:|---:|---|---|
| Blocked (auto-skip) | 3.237 | 4.665 | 0.919 | [3.120, 3.351] | [4.455, 4.866] |