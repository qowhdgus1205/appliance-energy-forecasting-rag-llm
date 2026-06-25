#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.mode_langchain_rag import (
    MODE_RAG_PROMPT,
    ModeAwareLGRetriever,
    build_query,
    format_documents_for_prompt,
    format_incident_evidence,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a LangChain prompt for Industrial Energy mode-aware context.")
    parser.add_argument("--rag-context", required=True, help="Mode-specific RAG context JSON.")
    parser.add_argument("--index-dir", default="outputs/rag/facility_a_mode_tfidf_index", help="TF-IDF index directory.")
    parser.add_argument("--question", default="What should I inspect when the next 15-step inst_heat forecast drops below expected?")
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    context_path = ROOT / args.rag_context if not Path(args.rag_context).is_absolute() else Path(args.rag_context)
    index_dir = ROOT / args.index_dir if not Path(args.index_dir).is_absolute() else Path(args.index_dir)
    context = load_json(context_path)
    retriever = ModeAwareLGRetriever(index_dir=str(index_dir), top_k=args.top_k)
    query = build_query(context, args.question)
    docs = retriever.invoke(query)
    prompt = MODE_RAG_PROMPT.format(
        language=args.language,
        question=args.question,
        incident_evidence=format_incident_evidence(context),
        retrieved_context=format_documents_for_prompt(docs),
    )
    out_dir = ROOT / "outputs" / "rag" / "langchain_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{context_path.stem}_prompt.md"
    out_path.write_text(prompt, encoding="utf-8")
    print(f"saved: {out_path}")
    print()
    print(prompt[:2000])


if __name__ == "__main__":
    main()
