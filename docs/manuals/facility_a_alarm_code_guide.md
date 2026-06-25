# Facility A alarm code guide

## Purpose

Synthetic alarm guide for natural-language retrieval.

## Alarm interpretation workflow

1. Identify the operating mode.
2. Check whether the residual is large over the 15-step window.
3. Review the top shifted sensors.
4. Decide whether to inspect temperature, pressure, compressor, or valve paths.

## Practical alarm groups

### Heat drop

- Review `inst_heat` forecast residuals.
- Inspect compressor frequency, current, and inlet/outlet temperatures.

### Pressure anomaly

- Review `water_pressure`, `high_pressure`, and related temperature changes.
- Check whether the mode changed before the anomaly.

### Flow instability

- Check whether the forecast sum and actual sum diverge sharply.
- Inspect flow-related features and the operating mode proxy.

## Notes

- The guide is synthetic and intended for RAG only.

