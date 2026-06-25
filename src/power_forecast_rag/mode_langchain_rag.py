from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from power_forecast_rag.rag import load_embedding_index, load_tfidf_index, search_embedding, search_tfidf


class ModeAwareLGRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_dir: str
    top_k: int = 4
    _vectorizer: Any = PrivateAttr()
    _matrix: Any = PrivateAttr()
    _metadata: list[dict[str, Any]] = PrivateAttr()
    _config: dict[str, Any] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        vectorizer, matrix, metadata, config = load_tfidf_index(Path(self.index_dir))
        self._vectorizer = vectorizer
        self._matrix = matrix
        self._metadata = metadata
        self._config = config

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        results = search_tfidf(query, self._vectorizer, self._matrix, self._metadata, top_k=self.top_k)
        docs = []
        for result in results:
            metadata = {k: v for k, v in result.items() if k != "text"}
            docs.append(Document(page_content=result["text"], metadata=metadata))
        return docs


class OpenAIEmbeddingModeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index_dir: str
    top_k: int = 4
    _matrix: Any = PrivateAttr()
    _metadata: list[dict[str, Any]] = PrivateAttr()
    _config: dict[str, Any] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        matrix, metadata, config = load_embedding_index(Path(self.index_dir))
        self._matrix = matrix
        self._metadata = metadata
        self._config = config

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        results = search_embedding(query, self._matrix, self._metadata, self._config, top_k=self.top_k)
        docs = []
        for result in results:
            metadata = {k: v for k, v in result.items() if k != "text"}
            docs.append(Document(page_content=result["text"], metadata=metadata))
        return docs


def build_mode_retriever(retriever: str, index_dir: str, top_k: int = 4) -> BaseRetriever:
    if retriever == "tfidf":
        return ModeAwareLGRetriever(index_dir=index_dir, top_k=top_k)
    if retriever == "embedding":
        return OpenAIEmbeddingModeRetriever(index_dir=index_dir, top_k=top_k)
    raise ValueError(f"Unsupported retriever: {retriever}")


MODE_RAG_PROMPT = PromptTemplate.from_template(
    """You are an industrial AI assistant helping a forward deployed engineer analyze Industrial Energy facility_a forecasting incidents.

Grounding rules:
- Use only the incident evidence and retrieved context below.
- Do not invent unobserved values, maintenance history, or causal certainty.
- The forecasting model is a shared multi-output forecast that predicts `t+1` through `t+15` in one shot.
- Use the summed 15-step forecast for cost comparison.
- `opermode` is used as a gas-like or heating-like proxy for residual interpretation.
- `opermode == 1` is treated as a gas-like proxy and `opermode == 0` as a heating-like proxy.
- A zero `inst_heat` value is not automatically abnormal in gas-like mode.
- Cite context chunk IDs in the answer.
- If evidence is insufficient, say so explicitly.

Write the final answer in {language}.

Required output structure:
1. Executive summary
2. Key numeric evidence
3. Mode interpretation
4. Likely hypotheses
5. Recommended inspection sequence
6. Limitations
7. Cost comparison if rates are provided
8. Context references

Style constraints:
- Keep the whole answer concise.
- Use at most 2 short bullets per section.
- Target 180 to 250 words total.
- Prefer direct operational wording.

User question:
{question}

Incident evidence:
{incident_evidence}

Retrieved context:
{retrieved_context}
"""
)


def format_documents_for_prompt(documents: list[Document]) -> str:
    blocks = []
    for doc in documents:
        meta = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"### {meta.get('chunk_id')} | {meta.get('title')} / {meta.get('section')}",
                    f"- score: {float(meta.get('score', 0)):.4f}",
                    f"- source: {meta.get('source_path')}",
                    "",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(blocks)


def format_incident_evidence(context: dict[str, Any]) -> str:
    top_episode = context.get("top_episode", {})
    top_features = context.get("top_features", [])
    feature_lines = []
    for row in top_features[:8]:
        importance = row.get("importance", row.get("importance_mean", 0))
        feature_lines.append(
            "- {feature}: importance={importance:.4f}".format(
                feature=row.get("feature"),
                importance=float(importance),
            )
        )

    root_lines = []
    for row in context.get("root_cause_top_feature_shifts", [])[:8]:
        root_lines.append(
            "- {feature}: delta={delta:.4f}, z_delta={z_delta:.4f}".format(
                feature=row.get("feature"),
                delta=float(row.get("delta", 0)),
                z_delta=float(row.get("z_delta", 0)),
            )
        )

    return "\n".join(
        [
            f"- site: {context.get('site')}",
            f"- mode_label: {context.get('mode_label')}",
            f"- mode_proxy: {context.get('mode_proxy')}",
            f"- mode_value: {context.get('mode_value')}",
            f"- interpretation: {context.get('mode_interpretation')}",
            f"- target: {context.get('target')}",
            f"- forecast_window: {context.get('forecast_window')}",
            f"- representative_horizon: {context.get('representative_horizon')}",
            f"- data_window: {context.get('data_window', {}).get('start_time')} to {context.get('data_window', {}).get('end_time')}",
            f"- residual_threshold: {float(context.get('forecast_metrics_proxy', {}).get('residual_threshold', 0)):.4f}",
            f"- predicted_anomaly_ratio: {float(context.get('forecast_metrics_proxy', {}).get('pred_anomaly_ratio', 0)):.4f}",
            f"- electric_rate: {context.get('cost_inputs', {}).get('electric_rate')}",
            f"- gas_rate: {context.get('cost_inputs', {}).get('gas_rate')}",
            "",
            "Top episode:",
            f"- episode_id: {top_episode.get('episode_id')}",
            f"- start_time: {top_episode.get('start_time')}",
            f"- end_time: {top_episode.get('end_time')}",
            f"- mean_actual_sum_15: {float(top_episode.get('mean_actual_sum_15', 0)):.4f}",
            f"- mean_prediction_sum_15: {float(top_episode.get('mean_prediction_sum_15', 0)):.4f}",
            f"- mean_residual_sum_15: {float(top_episode.get('mean_residual_sum_15', 0)):.4f}",
            f"- mean_abs_error_sum_15: {float(top_episode.get('mean_abs_error_sum_15', 0)):.4f}",
            "",
            "Top features:",
            *feature_lines,
            "",
            "Feature shifts:",
            *root_lines,
        ]
    )


def build_query(context: dict[str, Any], user_question: str) -> str:
    feature_blob = " ".join(row.get("feature", "") for row in context.get("top_features", [])[:8])
    root_blob = " ".join(row.get("feature", "") for row in context.get("root_cause_top_feature_shifts", [])[:8])
    return " ".join(
        [
            user_question,
            context.get("site", "facility_a"),
            context.get("mode_label", "mode"),
            context.get("mode_interpretation", ""),
            "forecast residual root cause inspection pressure current frequency temperature eev zero heat forecast window sum_15 cost compare electric gas rate",
            feature_blob,
            root_blob,
        ]
    )
