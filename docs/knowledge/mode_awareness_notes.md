# Mode awareness notes for facility_a inst_heat

## Working proxy

For the first Industrial Energy mode-aware workflow, we use `opermode` as the practical proxy for heat mode.

Observed pattern on facility_a:

- `opermode == 1` is strongly associated with `inst_heat == 0`
- `opermode == 0` is strongly associated with non-zero heating operation

This is an inference from the data, not a hard-coded domain rule.

## Why this matters

- A zero `inst_heat` value is not automatically an anomaly.
- The shared forecasting model should be interpreted with mode-specific residual thresholds.
- Mode-aware analysis reduces false alarms when the system is in gas-like or non-heating behavior.

## First mode-aware policy

1. Keep the shared direct-forecast suite for `t+1` through `t+15`.
2. Split incident summaries by `opermode`.
3. Compare residuals against mode-specific baselines.
4. Use mode-specific inspection language in prompts.
5. Treat zero heating as expected only when the mode proxy supports it.

## Inspection guidance

- In non-heating / gas-like mode, focus on whether the observed zero heat is consistent with the mode.
- In heating mode, focus on compressor current, target frequency, pressure stability, and temperature variance.
