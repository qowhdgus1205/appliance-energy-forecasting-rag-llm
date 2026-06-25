# Facility A inst_heat forecast suite summary

This note summarizes the current Industrial Energy deep project direction.

## Forecasting goal

- Predict `inst_heat` over the next 15 steps, not just the next step.
- Use one shared multi-output model that predicts `t+1` through `t+15` in a single shot.

## Why this matters

- A single-step forecast is too narrow for cost comparison and operational planning.
- The 15-step window gives a small future window that can be used for inspection, anomaly detection, and mode comparison.

## Mode handling

- `opermode` is used as a gas-like or heating-like proxy for residual interpretation.
- Mode is not the main forecast target; it is used to interpret forecast errors and adjust alarm language.
- Cost comparison should use the summed 15-step forecast, then apply the gas/electric rate.

## Current artifacts

- Synthetic manuals: `docs/manuals/*.md`
- Multi-output suite summary: `outputs/reports/facility_a_inst_heat_multioutput_summary.md`
- Mode-aware context: `outputs/reports/facility_a_gas_like_rag_context.json`
- Mode-aware context: `outputs/reports/facility_a_heating_like_rag_context.json`
- LangChain prompt and LangGraph trace are regenerated from the new window.
