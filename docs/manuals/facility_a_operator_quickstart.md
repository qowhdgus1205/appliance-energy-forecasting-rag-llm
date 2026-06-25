# Facility A operator quickstart

## Purpose

This document is a synthetic quickstart for operator-facing questions.

## What this system does

- Predicts `inst_heat` over the next 15 steps.
- Uses the forecast sum to estimate operating cost.
- Uses `opermode` as a gas-like or heating-like proxy for interpretation.

## What to do first when a forecast drops

1. Check whether the system is in gas-like or heating-like mode.
2. Review the summed 15-step forecast, not just one point.
3. Compare the residual against the mode-aware threshold.
4. Inspect the top feature shifts before escalating.

## Good operator questions

- What should I inspect when the forecast drops below expected?
- Is zero `inst_heat` expected in this mode?
- Which sensors changed most before the residual spike?

## Notes

- This is a synthetic guide for the Industrial Energy project.
- It is designed for retrieval and prompt grounding, not for production use.

