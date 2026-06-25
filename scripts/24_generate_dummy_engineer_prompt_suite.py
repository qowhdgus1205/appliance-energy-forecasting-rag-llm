#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.mode_langgraph_workflow import build_lg_mode_graph


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    mode: str
    question: str


def load_question_specs(path: Path) -> list[QuestionSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [QuestionSpec(**item) for item in raw]


def mode_to_context_path(mode: str) -> Path:
    if mode == "gas_like":
        return ROOT / "outputs" / "reports" / "facility_a_gas_like_rag_context.json"
    if mode == "heating_like":
        return ROOT / "outputs" / "reports" / "facility_a_heating_like_rag_context.json"
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy Industrial Energy engineer prompts through LangGraph.")
    parser.add_argument(
        "--questions",
        default="docs/dummy_engineer_questions.json",
        help="Question specification JSON.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--retriever", choices=["tfidf", "embedding"], default="tfidf")
    parser.add_argument("--index-dir", help="Index directory. Defaults to the selected retriever's standard output path.")
    args = parser.parse_args()

    question_path = ROOT / args.questions if not Path(args.questions).is_absolute() else Path(args.questions)
    specs = load_question_specs(question_path)
    graph = build_lg_mode_graph()

    out_dir = ROOT / "outputs" / "rag" / "dummy_engineer_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []

    for spec in specs:
        rag_context_path = mode_to_context_path(spec.mode)
        result = graph.invoke(
            {
                "rag_context_path": str(rag_context_path),
                "question": spec.question,
                "top_k": args.top_k,
                "language": args.language,
                "retriever": args.retriever,
                "index_dir": args.index_dir or f"outputs/rag/facility_a_mode_{args.retriever}_index",
            }
        )

        base = f"{spec.question_id}__{rag_context_path.stem}"
        prompt_path = out_dir / f"{base}_prompt.md"
        trace_path = out_dir / f"{base}_trace.json"

        prompt_path.write_text(result["prompt"], encoding="utf-8")
        trace_path.write_text(
            json.dumps(
                {
                    "question_id": spec.question_id,
                    "mode": spec.mode,
                    "rag_context_path": str(rag_context_path),
                    "question": spec.question,
                    "retriever": args.retriever,
                    "retrieval_query": result["retrieval_query"],
                    "retrieved_chunks": [doc.metadata for doc in result["retrieved_docs"]],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "question_id": spec.question_id,
                "mode": spec.mode,
                "prompt_path": str(prompt_path),
                "trace_path": str(trace_path),
            }
        )

        print(f"saved: {prompt_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = out_dir / "summary.md"
    lines = [
        "# Dummy Engineer Prompt Suite",
        "",
        f"- question_set: {question_path}",
        f"- retriever: {args.retriever}",
        f"- total: {len(manifest_rows)}",
        "",
    ]
    for row in manifest_rows:
        lines.extend(
            [
                f"## {row['question_id']}",
                f"- mode: {row['mode']}",
                f"- prompt: `{row['prompt_path']}`",
                f"- trace: `{row['trace_path']}`",
                "",
            ]
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {manifest_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
