# Facility A heating_like RAG context

- mode_proxy: `opermode`
- mode_value: `0`
- interpretation: `heating_like`
- model_scope: `TCN short-horizon energy forecast`
- metric_focus: `energy-area and aggregate-window error`

## TCN forecast summary

- model: `Temporal Convolutional Network (TCN)`
- input sequence length: `10`
- output sequence length: `10`
- numeric features: `22`
- best validation loss: `2261.2222`
- facility A aggregate window relative error: `0.0152`
- facility A segment AUC relative error: `0.0069`

## Mode interpretation

- Heating-like mode should prioritize load-following behavior, compressor current, compressor target frequency, pressure stability, and temperature variation.
- Repeated forecast drift should be interpreted with valve and EEV movement before claiming a model failure.
- Cost questions should use the forecast energy-area estimate and the relevant assumed electric rate.

## Recommended retrieval cues

- Use `facility_a_alarm_triage_matrix` for repeated drift or alarm-like patterns.
- Use `facility_a_compressor_diagnostics` for current, target frequency, and compressor tracking questions.
- Use `facility_a_sensor_signal_reference` for feature-level explanations.
- Use `facility_a_inspection_runbook` for field inspection ordering.

## Notes

- Point-wise MAPE is not the primary metric because near-zero targets can dominate percentage error.
- The public repository excludes raw data, preprocessing state, and private checkpoint binaries.
- This context is designed for retrieval-grounded demo answers, not production maintenance decisions.
