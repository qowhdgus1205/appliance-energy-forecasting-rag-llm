# Facility A heating_like RAG context

- mode_proxy: `opermode`
- mode_value: `0`
- interpretation: `heating_like`
- model_scope: `shared_multioutput_forecast`
- prediction_window: `short-horizon sequence`
- representative_horizon: `t+15`
- residual_threshold: 59959.2270
- predicted_anomaly_ratio: 0.0500

## Top episode

- time: 2024-05-11 06:14:56 -> 2024-05-11 06:18:38
- mean actual sum 15: 27884.25
- mean prediction sum 15: 120770.92
- mean residual sum 15: -92886.67

## Multi-output suite

| horizon | test_mae | test_rmse | test_r2 | test_mape | mean_abs_residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| t+1 | 599.17 | 965.91 | 0.8755 | 0.3223 | 599.17 |
| t+2 | 526.21 | 919.24 | 0.8873 | 0.2436 | 526.21 |
| t+3 | 687.49 | 1205.94 | 0.8060 | 0.2802 | 687.49 |
| t+4 | 783.31 | 1356.87 | 0.7544 | 0.2642 | 783.31 |
| t+5 | 1243.37 | 1799.50 | 0.5681 | 0.3725 | 1243.37 |
| t+6 | 1010.64 | 1750.44 | 0.5913 | 0.3064 | 1010.64 |
| t+7 | 1096.27 | 1875.28 | 0.5310 | 0.3087 | 1096.27 |
| t+8 | 1356.07 | 2110.52 | 0.4060 | 0.3921 | 1356.07 |
| t+9 | 1524.28 | 2412.43 | 0.2240 | 0.4377 | 1524.28 |
| t+10 | 1417.12 | 2238.81 | 0.3317 | 0.4202 | 1417.12 |
| t+11 | 1653.60 | 2611.62 | 0.0906 | 0.4418 | 1653.60 |
| t+12 | 1614.07 | 2403.29 | 0.2300 | 0.4304 | 1614.07 |
| t+13 | 1807.51 | 2669.33 | 0.0501 | 0.4884 | 1807.51 |
| t+14 | 1685.53 | 2517.12 | 0.1554 | 0.4374 | 1685.53 |
| t+15 | 1686.66 | 2526.02 | 0.1495 | 0.4396 | 1686.66 |

## Notes

- The forecasting model predicts the next 15 inst_heat values in one shot.
- Use opermode as the first gas/electric proxy when interpreting residuals.
- A zero inst_heat value is not automatically abnormal in gas-like mode.
- Residual thresholds are mode-aware and should be compared separately by operating mode.