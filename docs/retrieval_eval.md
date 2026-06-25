# Retrieval Evaluation

## Goal

This evaluation checks whether the RAG layer retrieves the right operational context for field-engineer questions.

The project is not trying to prove that the LLM knows the true root cause. The goal is narrower:

- retrieve the relevant manual or context group,
- produce an answer grounded in retrieved evidence,
- avoid unsupported maintenance claims,
- and cite context references in the final answer.

## Evaluation Setup

- retriever: local TF-IDF retriever
- corpus: synthetic operator manuals, diagnostic guides, mode notes, and forecast context summaries
- answer model used for saved examples: `gpt-5.4-mini`
- evaluation type: manual review of representative engineer questions

The repository also includes an optional OpenAI embedding retriever. The saved public answer run is still evaluated against the TF-IDF retriever so the baseline is transparent and reproducible without API calls. The embedding retriever is intended for semantic retrieval experiments, especially paraphrased engineer questions that do not share exact terms with the manuals.

## Retrieval Test Set

| question id | mode | expected evidence group | retrieved/answer evidence | result |
| --- | --- | --- | --- | --- |
| `gas_cost_rise_01` | gas-like | cost guide, mode FAQ, inspection runbook, compressor or hot-water diagnostics | answer cites cost comparison, gas-like mode behavior, DHW/hot-water, and inverter current checks | pass |
| `gas_zero_heat_02` | gas-like | mode FAQ, mode transition playbook, hot-water/valve guide | answer states near-zero heat can be normal in gas-like mode and recommends checking mode, DHW, hot water, and compressor signals | pass |
| `gas_pressure_current_03` | gas-like | compressor diagnostics, sensor reference, inspection runbook | answer focuses on current instability, compressor frequency, pressure/temperature checks, and residual interpretation | pass |
| `heat_residual_04` | heating-like | inspection runbook, mode transition playbook, compressor diagnostics | answer recommends heating-mode inspection sequence and checks compressor load, target frequency, pressure, and temperature | pass |
| `heat_cost_compare_05` | heating-like | cost comparison guide, inspection runbook, forecast context | answer summarizes cost comparison using the forecast energy estimate and mode-aware inspection points | pass |
| `heat_alarm_code_06` | heating-like | alarm triage matrix, compressor diagnostics, sensor reference, inspection runbook | answer selects alarm/inspection context and recommends compressor, target frequency, EEV, pressure, and temperature checks | pass |

## Summary Metrics

| metric | value |
| --- | ---: |
| evaluated questions | 6 |
| expected evidence hit rate | 6 / 6 |
| grounded answer rate | 6 / 6 |
| mode-aware answer rate | 6 / 6 |
| unsupported root-cause claim count | 0 |

## Grounding Rules Used

The prompt asks the LLM to:

- use only retrieved context and incident evidence,
- cite context chunk IDs,
- distinguish gas-like and heating-like interpretation,
- avoid causal certainty when evidence is insufficient,
- and provide inspection priority rather than a definitive maintenance diagnosis.

## Known Gaps

- This is a small manual review, not a large automated RAG benchmark.
- The manuals are synthetic, so retrieval success measures pipeline behavior rather than production maintenance coverage.
- The public repository excludes the private raw data and checkpoint state used in the historical best model run.
- Future work should compare TF-IDF vs OpenAI embedding retrieval, add retrieval top-k sweeps, and include negative-control questions that should return "insufficient evidence."
