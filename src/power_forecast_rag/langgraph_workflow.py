from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from power_forecast_rag.langchain_rag import (
    LG_RAG_PROMPT,
    TfidfLGRetriever,
    build_query,
    format_documents_for_prompt,
    format_incident_evidence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LGState(TypedDict, total=False):
    rag_context_path: str
    question: str
    top_k: int
    language: str
    context: dict[str, Any]
    retrieval_query: str
    retrieved_docs: list[Document]
    incident_evidence: str
    retrieved_context: str
    prompt: str


def load_context_node(state: LGState) -> LGState:
    path = Path(state["rag_context_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    context = json.loads(path.read_text(encoding="utf-8"))
    return {"context": context}


def build_query_node(state: LGState) -> LGState:
    query = build_query(state["context"], state["question"])
    incident_evidence = format_incident_evidence(state["context"])
    return {"retrieval_query": query, "incident_evidence": incident_evidence}


def retrieve_context_node(state: LGState) -> LGState:
    index_dir = PROJECT_ROOT / "outputs" / "rag" / "lg_tfidf_index"
    retriever = TfidfLGRetriever(index_dir=str(index_dir), top_k=state.get("top_k", 4))
    docs = retriever.invoke(state["retrieval_query"])
    return {"retrieved_docs": docs, "retrieved_context": format_documents_for_prompt(docs)}


def build_prompt_node(state: LGState) -> LGState:
    prompt = LG_RAG_PROMPT.format(
        language=state.get("language", "Korean"),
        question=state["question"],
        incident_evidence=state["incident_evidence"],
        retrieved_context=state["retrieved_context"],
    )
    return {"prompt": prompt}


def build_lg_graph():
    graph = StateGraph(LGState)
    graph.add_node("load_context", load_context_node)
    graph.add_node("build_query", build_query_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("build_prompt", build_prompt_node)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "build_query")
    graph.add_edge("build_query", "retrieve_context")
    graph.add_edge("retrieve_context", "build_prompt")
    graph.add_edge("build_prompt", END)
    return graph.compile()
