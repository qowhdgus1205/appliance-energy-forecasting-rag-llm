# Failure Analysis

## Why This Document Exists

This project is a portfolio demo, but the workflow still has important limitations. Listing them explicitly makes the engineering boundary clearer.

## Modeling Limitations

- The public repository does not include raw data, preprocessing state, or private checkpoint binaries.
- The historical best TCN checkpoint is reported as an anonymized experiment summary.
- The included public TCN script is a compact reproducibility demo and should not be treated as a production model.
- Point-wise MAPE is unstable when `inst_heat` is near zero.
- Mode labels are proxies. They help interpretation, but they are not a complete ground-truth operating state.

## RAG Limitations

- Manuals are synthetic and designed to test retrieval-grounded answering.
- The retrieval evaluation is a small manual review, not a full benchmark.
- TF-IDF retrieval is transparent and lightweight, but it may miss paraphrases that an embedding retriever would catch.
- The answer should provide inspection priority, not definitive root cause.

## LLM Limitations

- The LLM can still overstate causality if the prompt is weakened.
- The current prompt relies on explicit grounding rules and citation requirements.
- API answers are saved as demo artifacts; they are not production service logs.

## Product Limitations

- There is no deployed dashboard in this public release.
- There is no live connection to sensor streams or ticketing systems.
- There is no human feedback loop for field engineers to mark answer quality.

## Recommended Next Improvements

1. Add a small Streamlit or FastAPI demo.
2. Add retrieval top-k and chunk-size evaluation.
3. Add an answer-quality rubric with groundedness, usefulness, and uncertainty scoring.
4. Add negative-control questions where the correct answer is "insufficient evidence."
5. Add a clean public sample dataset or synthetic data generator so the full pipeline can run end to end.
