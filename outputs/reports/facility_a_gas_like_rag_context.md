# Facility A gas_like RAG context

- mode_proxy: `opermode`
- mode_value: `1`
- interpretation: `gas_like`
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

- Gas-like mode can legitimately show low or near-zero `inst_heat`.
- A low heat signal should not be treated as abnormal until mode state, residual size, and load-related features are checked.
- Cost questions should compare the predicted energy-area estimate against assumed gas and electric rates.

## Recommended retrieval cues

- Use `facility_a_mode_transition_playbook` for mode-aware threshold interpretation.
- Use `facility_a_valve_and_hotwater_control` for low heat with hot water or valve routing changes.
- Use `facility_a_compressor_diagnostics` when low heat appears together with elevated current or compressor frequency movement.
- Use `facility_a_cost_comparison_guide` when the question asks about operating cost.

## Notes

- Point-wise MAPE is not the primary metric because near-zero targets can dominate percentage error.
- The public repository excludes raw data, preprocessing state, and private checkpoint binaries.
- This context is designed for retrieval-grounded demo answers, not production maintenance decisions.
