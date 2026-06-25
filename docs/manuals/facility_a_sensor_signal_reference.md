# Facility A sensor signal reference

## Purpose

This synthetic reference describes the sensor groups used by the forecasting and RAG workflow.

## Heating demand signals

- `inst_heat` is the modeled heat or power demand target.
- Rolling or lagged variants should be interpreted as historical context, not as future observations.
- Sudden disagreement between actual and predicted heat should be reviewed with operating mode and compressor load.

## Compressor and inverter signals

- `inv1_input_current` indicates electrical load drawn by the inverter.
- `inv1_comp_current_freq` describes the observed compressor frequency.
- `inv1_comp_target_freq` describes the requested compressor frequency.
- A widening gap between current and target frequency can indicate control saturation, load mismatch, or sensor drift.

## Domestic hot water and valve signals

- `3way_DHW` indicates the domestic hot water or valve routing state.
- `hotwater_th` should be checked when heat demand and mode state appear inconsistent.
- `vi_eev1` should be reviewed when valve opening or refrigerant flow behavior may explain forecast drift.

## Temperature and pressure signals

- Inlet, outlet, and return temperature changes should be compared against compressor current.
- Pressure instability should be reviewed before concluding that the forecast model is wrong.
- Short variance spikes can indicate unstable control, defrost-like behavior, or transient sensor noise.

## Retrieval notes

- Use this document when questions mention current, frequency, target frequency, valve, EEV, hot water, pressure, or temperature.
- This document is synthetic and designed for public RAG demonstration.
