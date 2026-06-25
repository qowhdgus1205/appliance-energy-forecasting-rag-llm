# inst_heat operating notes

## Overview

These notes summarize the first Industrial Energy facility_a forecasting study for `inst_heat`.

## Practical signals

- `inst_heat_rmean3` is the strongest local predictor in the first baseline.
- `inv1_comp_current_freq_rmean3`, `inv1_input_current_rmean3`, and `Sub_EEV_rmean3` are also strong signals.
- `high_pressure`, `pressure_rate`, and temperature-related rolling features are useful supporting signals.

## How to interpret large residuals

- A large positive residual means actual future `inst_heat` is higher than the forecast.
- A large negative residual means actual future `inst_heat` is lower than the forecast.
- When residuals spike, inspect compressor current, target frequency, pressure stability, and temperature variance together.

## Recommended investigation order

1. Compare actual vs predicted `inst_heat`.
2. Check `inv1_comp_current_freq` and `inv1_input_current`.
3. Check `high_pressure` and `pressure_rate`.
4. Check `Sub_EEV` and `main1_eev_pulse`.
5. Review the nearest episode root-cause summary.

