# Best TCN experiment summary

## Purpose

This note records the best historical Temporal Convolutional Network experiment used to position the public portfolio project.

The original checkpoint and raw data are excluded from this repository. The metrics below are retained as an anonymized experiment summary.

## Model

- model family: Temporal Convolutional Network (TCN)
- convolution style: depthwise separable temporal blocks
- channels: `(32, 32, 32)`
- kernel size: `3`
- dropout: `0.3`
- input sequence length: `10`
- output sequence length: `10`
- numeric features: `22`
- target: `inst_heat`

## Validation

- best validation loss: `2261.2222`
- best epoch: `349`

## Facility A test metrics

| metric | value |
| --- | ---: |
| area per-window MAE | 470433.6563 |
| aggregate window relative error | 0.0152 |
| aggregate window bias | 205768704 |
| segment AUC relative error | 0.0069 |
| segment AUC true area | 1498576923 |
| segment AUC predicted area | 1488211230 |
| segment AUC difference | -10365693 |

## Interpretation

Point-wise MAPE is not a reliable primary metric for this target because near-zero target values can dominate the percentage error. The area and aggregate metrics are more useful for operational energy forecasting because they compare total predicted and actual energy over a window.

## Public release note

The public repository includes a compact reproducible TCN script, but it does not include the private checkpoint, raw facility data, or preprocessing state used for this historical best run.
