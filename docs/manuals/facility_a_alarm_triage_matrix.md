# Facility A alarm triage matrix

## Purpose

This synthetic matrix maps common forecast-driven alarm patterns to first inspection steps.

## High residual with high compressor current

- Check inverter input current and compressor current frequency.
- Compare current frequency against target frequency.
- Review pressure stability and recent temperature variance.

## High residual with valve movement

- Check `vi_eev1`, `3way_DHW`, and `hotwater_th`.
- Compare valve movement against compressor load.
- Review whether hot water routing explains heat demand mismatch.

## Cost increase with low heat forecast

- Confirm mode first.
- Compare the forecast sum against assumed gas and electric rates.
- Inspect compressor current and hot water routing if cost rises despite low heat.

## Repeated drift in heating-like mode

- Check inlet and return temperature trends.
- Inspect compressor target frequency tracking.
- Review pressure stability and valve movement.

## Escalation notes

- Escalate repeated patterns after mode, sensor quality, and known transient behavior are checked.
- Do not claim root cause from residuals alone.
- Use retrieved context and incident evidence together in operator-facing answers.
