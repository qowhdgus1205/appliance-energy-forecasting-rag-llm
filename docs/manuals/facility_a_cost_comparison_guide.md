# Facility A cost comparison guide

## Purpose

Explain how the forecast is turned into an estimated operating cost.

## Cost calculation

- Predict `inst_heat` for `t+1` through `t+15`.
- Sum the 15-step forecast.
- Multiply by the assumed gas or electric rate.

## Interpretation

- If the gas-like proxy is active, the gas rate is the relevant comparison baseline.
- If the heating-like proxy is active, the electric rate is the relevant comparison baseline.
- The cost number is an estimate, not a bill.

## Questions this guide should answer

- Which mode is cheaper for the next 15-step window?
- How much does the cost change when the residual grows?
- Is the forecasted drop paired with a larger anomaly score?

## Notes

- This is a synthetic guide for retrieval and prompt grounding.

