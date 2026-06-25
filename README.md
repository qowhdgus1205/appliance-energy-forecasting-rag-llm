# Appliance Energy Forecasting with RAG and LLM

An industrial appliance time-series project that forecasts short-horizon power/heat demand, detects abnormal forecast residuals, retrieves operational context with RAG, and generates engineer-facing answers with an OpenAI GPT model.

The project is intentionally anonymized for public portfolio use. Raw data, API keys, trained model binaries, and proprietary identifiers are not included.

## What This Project Shows

- Multi-step time-series forecasting for `inst_heat` over `t+1..t+15`
- Residual-based anomaly detection on forecast windows
- Mode-aware interpretation for gas-like and heating-like operating states
- Cost comparison using summed 15-step forecasts and assumed energy rates
- TF-IDF RAG over synthetic operator manuals and incident summaries
- LLM answer generation using `gpt-5.4-mini`
- Reproducible prompt, answer, and trace artifacts for portfolio review

## Architecture

```text
processed sensor table
  -> feature pruning
  -> multi-output forecast model
  -> residual/anomaly summaries
  -> mode-aware incident context
  -> RAG corpus + TF-IDF retriever
  -> LangChain/LangGraph prompt workflow
  -> OpenAI API answer records
```

## Key Results

Forecast target: `inst_heat`

Forecast window: `t+1..t+15`

Model: `ExtraTreesRegressor` shared multi-output model

| split | MAE | RMSE | R2 | MAPE |
| --- | ---: | ---: | ---: | ---: |
| train | 141.2832 | 296.8979 | 0.9717 | 0.0732 |
| val | 561.6593 | 1089.3591 | 0.7684 | 0.2750 |
| test | 778.9775 | 1433.0679 | 0.7261 | 0.2598 |

See [forecast summary](outputs/reports/facility_a_inst_heat_multioutput_summary.md).

## RAG and LLM Demo

The demo uses synthetic manuals and generated incident records to answer realistic engineer questions such as:

> The next 15-step `inst_heat` forecast is lower than expected, but the operating cost estimate is rising. What should I inspect first?

Generated API answers are stored under:

```text
outputs/rag/dummy_engineer_api_runs/20260625_150159/
```

Example answer:

- [gas_cost_rise_01_answer.md](outputs/rag/dummy_engineer_api_runs/20260625_150159/gas_cost_rise_01_answer.md)

## Repository Layout

```text
docs/
  manuals/                  Synthetic operator manuals for RAG
  knowledge/                Mode-awareness and operating notes
  portfolio_demo_questions.md

scripts/
  25_train_facility_a_inst_heat_multioutput.py
  26_run_dummy_engineer_openai_api.py

src/power_forecast_rag/
  rag.py
  mode_langchain_rag.py
  mode_langgraph_workflow.py

outputs/
  reports/                  Public summary artifacts
  rag/                      Public LLM answer examples
```

## Run the LLM Answer Workflow

Set an API key through the environment or a local ignored file:

```bash
export OPENAI_API_KEY="sk-..."
python scripts/26_run_dummy_engineer_openai_api.py --model gpt-5.4-mini
```

To run one question only:

```bash
python scripts/26_run_dummy_engineer_openai_api.py --question-id gas_cost_rise_01
```

The script writes markdown answers and JSON traces into `outputs/rag/dummy_engineer_api_runs/<timestamp>/`.

## Notes

- The included manuals are synthetic and used only to demonstrate retrieval-grounded answering.
- The cost comparison uses assumed rates, not real billing data.
- Raw datasets and trained binary models are intentionally excluded from this public release.
- Facility names and paths have been anonymized.
