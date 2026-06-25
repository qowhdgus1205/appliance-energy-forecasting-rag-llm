#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.langgraph_workflow import build_lg_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Industrial Energy LangGraph RAG workflow.")
    parser.add_argument(
        "--rag-context",
        default="outputs/reports/facility_a_inst_heat_rag_context.json",
        help="Compact incident RAG context JSON.",
    )
    parser.add_argument(
        "--question",
        default="What should I inspect when t+1 inst_heat drops below forecast?",
        help="User question.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--language", default="Korean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_lg_graph()
    result = graph.invoke(
        {
            "rag_context_path": args.rag_context,
            "question": args.question,
            "top_k": args.top_k,
            "language": args.language,
        }
    )

    context_path = Path(args.rag_context)
    if not context_path.is_absolute():
        context_path = ROOT / context_path

    out_dir = ROOT / "outputs" / "rag" / "langgraph_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = context_path.stem
    prompt_path = out_dir / f"{base}_prompt.md"
    trace_path = out_dir / f"{base}_trace.json"

    prompt_path.write_text(result["prompt"], encoding="utf-8")
    trace_path.write_text(
        json.dumps(
            {
                "rag_context_path": str(context_path),
                "question": args.question,
                "retrieval_query": result["retrieval_query"],
                "retrieved_chunks": [doc.metadata for doc in result["retrieved_docs"]],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Saved prompt: {prompt_path}")
    print(f"Saved trace: {trace_path}")
    print()
    print(result["prompt"][:2000])


if __name__ == "__main__":
    main()

