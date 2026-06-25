from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from power_forecast_rag.rag import load_tfidf_index, search_tfidf


class TfidfLGRetriever(BaseRetriever):
    """LangChain retriever wrapper for the local Industrial Energy TF-IDF index."""

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
        results = search_tfidf(
            query,
            self._vectorizer,
            self._matrix,
            self._metadata,
            top_k=self.top_k,
        )
        documents = []
        for result in results:
            metadata = {key: value for key, value in result.items() if key != "text"}
            documents.append(Document(page_content=result["text"], metadata=metadata))
        return documents


LG_RAG_PROMPT = PromptTemplate.from_template(
    """You are an industrial AI assistant helping a forward deployed engineer analyze Industrial Energy facility_a forecasting incidents.

Grounding rules:
- Use only the incident evidence and retrieved context below.
- Do not invent unobserved values, maintenance history, or causal certainty.
- Phrase root causes as hypotheses unless directly supported by evidence.
- Cite context chunk IDs in the answer.
- If evidence is insufficient, say so explicitly.

Write the final answer in {language}.

Required output structure:
1. Executive summary
2. Key numeric evidence
3. Likely hypotheses
4. Recommended inspection sequence
5. Limitations
6. Context references

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
    root_top = context.get("root_cause_top_episode", {})
    feature_lines = []
    for row in top_features[:8]:
        feature_lines.append(
            "- {feature}: importance_mean={importance:.4f}".format(
                feature=row.get("feature"),
                importance=float(row.get("importance_mean", 0)),
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
            f"- target: {context.get('target')}",
            f"- horizon: {context.get('horizon')}",
            f"- data_window: {context.get('data_window', {}).get('start_time')} to {context.get('data_window', {}).get('end_time')}",
            f"- residual_threshold: {float(context.get('forecast_metrics_proxy', {}).get('residual_threshold', 0)):.4f}",
            f"- predicted_anomaly_ratio: {float(context.get('forecast_metrics_proxy', {}).get('pred_anomaly_ratio', 0)):.4f}",
            "",
            "Top episode:",
            f"- episode_id: {top_episode.get('episode_id')}",
            f"- start_time: {top_episode.get('start_time')}",
            f"- end_time: {top_episode.get('end_time')}",
            f"- mean_actual_future: {float(top_episode.get('mean_actual', 0)):.4f}",
            f"- mean_prediction: {float(top_episode.get('mean_prediction', 0)):.4f}",
            f"- mean_residual: {float(top_episode.get('mean_residual', 0)):.4f}",
            f"- mean_abs_error: {float(top_episode.get('mean_abs_error', 0)):.4f}",
            "",
            "Top features:",
            *feature_lines,
            "",
            "Root-cause hints:",
            f"- episode_id: {root_top.get('episode_id')}",
            f"- mean_actual_future: {float(root_top.get('mean_target_future', 0)):.4f}",
            f"- mean_prediction: {float(root_top.get('mean_prediction', 0)):.4f}",
            f"- mean_residual: {float(root_top.get('mean_residual', 0)):.4f}",
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
            context.get("target", "inst_heat"),
            "forecast residual root cause inspection pressure current frequency temperature eev",
            feature_blob,
            root_blob,
        ]
    )

