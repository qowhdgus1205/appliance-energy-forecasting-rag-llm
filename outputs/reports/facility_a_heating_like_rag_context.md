# Facility A heating_like RAG context

- mode_proxy: `opermode`
- mode_value: `0`
- interpretation: `heating_like`
- model_scope: `shared_multioutput_forecast`
- forecast_window: `t+1..t+15`
- representative_horizon: `t+15`
- residual_threshold: 33790.1552
- predicted_anomaly_ratio: 0.0501

## Top episode

- time: 2024-05-01 09:38:26 -> 2024-05-01 10:04:20
- mean actual sum 15: 164661.91
- mean prediction sum 15: 84317.87
- mean residual sum 15: 80344.03

## Multi-output suite

| horizon | test_mae | test_rmse | test_r2 | test_mape | mean_abs_residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| t+1 | 265.43 | 562.83 | 0.9577 | 0.1413 | 265.43 |
| t+2 | 306.73 | 711.39 | 0.9325 | 0.1506 | 306.73 |
| t+3 | 396.07 | 947.89 | 0.8801 | 0.1781 | 396.07 |
| t+4 | 491.44 | 1131.16 | 0.8293 | 0.1857 | 491.44 |
| t+5 | 596.93 | 1268.23 | 0.7854 | 0.2198 | 596.93 |
| t+6 | 685.42 | 1375.13 | 0.7477 | 0.2346 | 685.42 |
| t+7 | 774.83 | 1465.77 | 0.7134 | 0.2496 | 774.83 |
| t+8 | 846.64 | 1533.02 | 0.6865 | 0.2724 | 846.64 |
| t+9 | 899.15 | 1582.40 | 0.6660 | 0.2785 | 899.15 |
| t+10 | 959.18 | 1625.34 | 0.6477 | 0.2954 | 959.18 |
| t+11 | 1023.45 | 1670.17 | 0.6280 | 0.3152 | 1023.45 |
| t+12 | 1068.74 | 1698.84 | 0.6152 | 0.3310 | 1068.74 |
| t+13 | 1100.85 | 1714.91 | 0.6079 | 0.3436 | 1100.85 |
| t+14 | 1123.84 | 1730.91 | 0.6006 | 0.3503 | 1123.84 |
| t+15 | 1145.94 | 1745.90 | 0.5936 | 0.3516 | 1145.94 |

## Notes

- The forecasting model predicts the next 15 inst_heat values in one shot.
- Use opermode as the first gas/electric proxy when interpreting residuals.
- A zero inst_heat value is not automatically abnormal in gas-like mode.
- Residual thresholds are mode-aware and should be compared separately by operating mode.