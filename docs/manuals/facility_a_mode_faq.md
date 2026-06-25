# Facility A mode FAQ

## Why mode matters

Mode changes how a zero `inst_heat` value should be interpreted.

## Working proxy

- `opermode == 1` is treated as a gas-like proxy.
- `opermode == 0` is treated as a heating-like proxy.
- The proxy is inferred from the data pattern, not hard-coded domain knowledge.

## Common questions

### Is zero heat always abnormal?

No. In gas-like mode, zero heat can be expected.

### Should I use the same threshold for all modes?

No. Residual thresholds should be compared separately by mode.

### What should I report?

- Mode label and time window.
- Summed 15-step forecast vs observed sum.
- Top feature shifts and the inspection sequence.

## Notes

- This FAQ is synthetic and exists to support RAG queries in the project.

