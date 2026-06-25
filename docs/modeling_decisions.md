# Modeling Decisions

## Problem Framing

The project is framed as a field-engineering support workflow for appliance energy operations.

The model forecasts short-horizon `inst_heat` demand. The downstream workflow uses forecast residuals, operating mode, and retrieved manual context to help an engineer decide what to inspect first.

## Why TCN

Temporal Convolutional Networks are a practical fit for this project because:

- they model local temporal patterns without recurrent inference,
- dilated convolutions can cover a wider recent history,
- inference is fast enough for operational dashboards,
- and the architecture is easier to deploy than heavier sequence models.

The historical best checkpoint used a depthwise-separable TCN with three temporal blocks.

## Input and Output Choice

The best historical experiment used:

- input length: `10`
- output length: `10`
- numeric features: `22`

This setup keeps the prediction horizon short enough for field inspection and long enough to compare accumulated energy behavior. Longer output lengths were tested, but the shorter output setting produced the strongest facility-level energy-area behavior in the retained experiment summary.

## Metric Choice

Point-wise MAPE is not the primary metric because `inst_heat` can be near zero. When the denominator is near zero, MAPE can become very large even when the absolute energy error is operationally acceptable.

The project therefore emphasizes:

- energy-area relative error,
- residual thresholding,
- and mode-aware interpretation.

The headline metric in the README is `energy-area rel. error = 0.0069`, meaning the predicted and actual areas under the `inst_heat` curve differed by about 0.69% on the representative held-out segment.

## Baselines and Tradeoffs

Earlier development included tree-based forecasting baselines. Those were useful for fast feature screening and residual analysis, but the final public framing uses TCN because the project is meant to demonstrate sequence modeling for time-series appliance energy behavior.

Tradeoffs:

- Tree models are easier to train and inspect.
- TCN better matches the temporal forecasting story.
- The best private TCN checkpoint is summarized but not shipped because raw data, preprocessing state, and model binaries are excluded from the public repository.
- The included public TCN training script is intentionally compact and reproducible, not a claim that it exactly reproduces the historical best checkpoint.

## How This Connects to RAG

The model output is not shown to the LLM as a standalone prediction. It is converted into operational evidence:

- forecast trend,
- residual behavior,
- mode label,
- cost estimate,
- and top inspection cues.

The RAG layer then retrieves relevant manual sections so the LLM can turn model evidence into field-engineer guidance.
