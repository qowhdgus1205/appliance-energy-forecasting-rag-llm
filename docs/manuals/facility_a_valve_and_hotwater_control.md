# Facility A valve and hot water control guide

## Purpose

This synthetic guide explains how valve and hot water signals should be reviewed during forecast incidents.

## Relevant signals

- `3way_DHW` should be checked when the system may route energy toward domestic hot water.
- `hotwater_th` should be checked when hot water demand may explain unexpected heat output.
- `vi_eev1` should be checked when refrigerant flow or expansion behavior appears unstable.

## Gas-like mode checks

- Low or zero `inst_heat` may be expected in gas-like mode.
- If cost rises while heat forecast is low, inspect hot water routing and compressor load before treating heat as abnormal.
- Confirm that the operating mode proxy did not change during the incident window.

## Heating-like mode checks

- Forecast drift in heating-like mode should be reviewed with valve position, compressor frequency, and temperature stability.
- Repeated residual spikes with `vi_eev1` movement can indicate unstable flow control.
- Hot water routing should be checked if heat output and compressor load disagree.

## Recommended sequence

1. Confirm mode.
2. Check `3way_DHW` and `hotwater_th`.
3. Compare `vi_eev1` movement with compressor current and target frequency.
4. Review temperature and pressure stability.

## Retrieval notes

- Use this guide for questions involving DHW, hot water, EEV, valve state, or low heat with nonzero load.
