# Facility A inst_heat TCN forecast

- target: `inst_heat`
- model: `Temporal Convolutional Network (TCN)`
- checkpoint type: historical best local experiment
- input sequence length: `10`
- output sequence length: `10`
- numeric features used: `22`

## Validation

- best validation loss: `2261.2222`
- best epoch: `349`

## Facility A Test Metrics

| metric | value |
| --- | ---: |
| area per-window MAE | 470433.6563 |
| aggregate window relative error | 0.0152 |
| aggregate window bias | 205768704 |
| segment AUC relative error | 0.0069 |
| segment AUC true area | 1498576923 |
| segment AUC predicted area | 1488211230 |
| segment AUC difference | -10365693 |

## Metric Choice

Point-wise MAPE is unstable for this dataset because near-zero target values can dominate percentage error. The operational analysis therefore emphasizes energy-area and aggregate-window errors.

## Public Release Note

Raw data, preprocessing state, and trained checkpoint binaries are excluded from this repository. The included TCN training script is a compact reproducible demo, while this report summarizes the best historical local TCN experiment.
