# Industrial Energy Project Progress Summary

This document summarizes the Industrial Energy deep-project work completed so far and the current working assumptions.

## 1. Project goal

The main goal is to build a forecasting-based operational analysis pipeline for `facility_a` that:

- predicts `inst_heat` over a short future window,
- compares predicted cost under gas-like and heating-like assumptions,
- detects abnormal behavior from forecasting residuals,
- and retrieves supporting context through RAG for operator-facing explanations.

The project is designed as a portfolio-ready demo for a forward-deployed engineer style workflow.

## 2. Data used

We started from the following Industrial Energy data files:

- `facility_b.csv`
- `facility_c.csv`
- `facility_a.csv`
- `facility_d.csv`

For the current direction, `facility_a.csv` is the main dataset.

### Feature pruning

We removed most engineered lag/rolling/calendar features and kept only the features needed for the forecasting pipeline.

Current working assumption:

- target: `inst_heat`
- future horizon: `t+1` through `t+15`
- `opermode` is used as a gas-like / heating-like proxy for interpretation

## 3. Modeling approach

### Forecasting

The current main model is a shared multi-output forecasting model that predicts the next 15 steps in one shot.

Current model characteristics:

- input: current sensor state
- output: `inst_heat` for `t+1..t+15`
- model type: tree-based multi-output regression
- evaluation metrics: `R2`, `MAE`, `RMSE`, `MAPE`

### Cost comparison

We introduced assumed operating rates for portfolio/demo use:

- electric: `0.18 €/kWh`
- gas: `0.07 €/kWh`

The forecasted `inst_heat` sum over 15 steps is used to estimate relative operating cost.

This is an estimate, not a real billing calculation.

### Anomaly interpretation

Residuals from the forecast are used to flag unusual behavior.

Important mode-aware rule:

- gas-like mode and heating-like mode are interpreted separately
- zero `inst_heat` is not automatically abnormal in gas-like mode

## 4. EDA and analysis done

We completed the following analysis steps:

- dataset scanning
- target profiling for `inst_heat`
- feature pruning
- facility_a EDA visualizations
- residual episode summaries
- root-cause summaries for high-residual episodes
- mode profiling for gas-like / heating-like behavior

Artifacts were generated for:

- overview plots
- residual plots
- top episode zoom plots
- mode zero-ratio plots

## 5. RAG work done

### Synthetic manuals

Because real manuals were not available in the dataset, we created synthetic operator/manual documents for retrieval testing.

Created manuals:

- `facility_a_operator_quickstart.md`
- `facility_a_mode_faq.md`
- `facility_a_alarm_code_guide.md`
- `facility_a_inspection_runbook.md`
- `facility_a_cost_comparison_guide.md`

### Additional knowledge docs

We also created supporting notes:

- `inst_heat_operating_notes.md`
- `mode_awareness_notes.md`

### Retrieval layer

The document corpus was chunked and indexed with TF-IDF.

Current retrieval artifacts:

- chunk JSONL corpora
- TF-IDF index directories
- LangChain retriever wrappers

### LangChain / LangGraph

The workflow currently does:

- load incident/context JSON
- build a retrieval query
- retrieve relevant chunks
- build a prompt
- save prompt/trace for API-backed answer testing

This is already implemented for both:

- general Industrial Energy retrieval
- mode-aware facility_a retrieval

## 6. Dummy engineer questions

To support portfolio screenshots and manual testing, we created realistic engineer-style questions.

Files:

- `docs/dummy_engineer_questions.json`
- `docs/dummy_engineer_scenarios.md`
- `docs/portfolio_demo_questions.md`

These questions are intended to be asked one at a time in a realistic operator/engineer sequence.

### OpenAI API answer run

On 2026-06-25, the six dummy engineer prompts were run through the OpenAI API with `gpt-5.4-mini`.

Generated records:

- `outputs/rag/dummy_engineer_api_runs/20260625_150159/summary.md`
- `outputs/rag/dummy_engineer_api_runs/20260625_150159/manifest.json`
- per-question `*_answer.md` files
- per-question `*_api_trace.json` files with response IDs and token usage

## 7. What is ready now

Current ready items:

- `facility_a` forecasting pipeline
- multi-output `t+1..t+15` prediction
- cost comparison logic
- mode-aware residual interpretation
- synthetic manuals for RAG
- TF-IDF retrieval
- LangChain prompt generation
- LangGraph workflow skeleton
- dummy engineer questions for demo screenshots
- OpenAI API answer generation with `gpt-5.4-mini`

## 8. What is not finished

Not yet finished:

- real enterprise manual ingestion
- production-grade validation loop
- full UI application

The current workflow now supports local prompt generation and OpenAI API answer recording.

## 9. Recommended next step

When continuing tomorrow, the most sensible next step is:

1. review the generated API answers,
2. choose the best 1-2 examples for portfolio screenshots,
3. tighten the prompt if any answer is too long or too vague,
4. then rerun the selected question IDs only.
