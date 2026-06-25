# Facility A mode transition playbook

## Purpose

This synthetic playbook describes how to interpret forecast residuals around operating mode changes.

## Mode proxy

- `opermode == 1` is treated as gas-like.
- `opermode == 0` is treated as heating-like.
- The proxy is used for interpretation, not as a ground-truth equipment label.

## Gas-like interpretation

- Zero or near-zero `inst_heat` can be normal.
- Residual thresholds should be compared against gas-like history.
- Cost questions should review the forecast sum, gas rate, and compressor or hot water load.

## Heating-like interpretation

- Sustained residual drift should be reviewed with compressor load, target frequency, pressure stability, and temperature variance.
- Zero heat may still be transient, but repeated drift needs inspection.
- Heating-like questions should prioritize load-following behavior and valve control.

## Transition checks

1. Verify mode value before and during the incident.
2. Check whether mode-related columns changed together.
3. Review residuals using the correct mode-specific threshold.
4. Inspect top feature shifts only after mode consistency is confirmed.

## Retrieval notes

- Use this playbook for mode-aware questions, zero-heat questions, and threshold interpretation.
