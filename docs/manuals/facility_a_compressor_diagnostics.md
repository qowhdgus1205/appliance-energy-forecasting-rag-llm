# Facility A compressor diagnostics guide

## Purpose

This synthetic guide supports field diagnosis when forecast residuals coincide with compressor or inverter changes.

## Common symptoms

- Forecast residual increases while `inv1_input_current` rises.
- Compressor current frequency does not track target frequency.
- Heat demand appears low while electrical load remains elevated.
- Repeated high-residual windows occur around mode transitions or hot water routing changes.

## First checks

1. Confirm the current operating mode.
2. Compare `inv1_input_current`, `inv1_comp_current_freq`, and `inv1_comp_target_freq`.
3. Check whether `3way_DHW` or `hotwater_th` changed near the residual event.
4. Review pressure and temperature stability before escalating.

## Interpretation rules

- High current with low heat can indicate inefficient operation, transient demand, or measurement mismatch.
- Target frequency above observed frequency can indicate control limitation or load protection.
- Observed frequency above expected load can indicate unnecessary compressor work.

## Escalation criteria

- Escalate when compressor current remains high across repeated residual windows.
- Escalate when target and current frequency diverge with pressure instability.
- Escalate when the same pattern repeats after confirming mode and sensor quality.

## Retrieval notes

- Use this guide for questions about compressor current, inverter current, frequency tracking, or residual drift.
- This guide is synthetic and not an OEM service manual.
