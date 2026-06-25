#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "rag" / "dummy_engineer_runs" / "manifest.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "rag" / "dummy_engineer_api_runs"
DEFAULT_MODEL = "gpt-5.4-mini"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def response_to_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)
    return "\n".join(chunks).strip()


def model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def load_manifest(path: Path, question_ids: set[str] | None) -> list[dict[str, Any]]:
    rows = read_json(path)
    if question_ids is None:
        return rows
    selected = [row for row in rows if row["question_id"] in question_ids]
    missing = sorted(question_ids - {row["question_id"] for row in selected})
    if missing:
        raise ValueError(f"Question IDs not found in manifest: {', '.join(missing)}")
    return selected


def run_one(client: OpenAI, model: str, prompt: str, max_output_tokens: int) -> Any:
    return client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dummy engineer prompts through the OpenAI API.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Prompt manifest JSON.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for API answer records.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model. Defaults to OPENAI_MODEL or gpt-5.4-mini.",
    )
    parser.add_argument("--question-id", action="append", help="Run only this question_id. Repeatable.")
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--run-label", help="Optional stable run directory name.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    question_ids = set(args.question_id) if args.question_id else None
    rows = load_manifest(manifest_path, question_ids)

    run_label = args.run_label or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_dir = out_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)
    summary_rows: list[dict[str, Any]] = []

    for row in rows:
        question_id = row["question_id"]
        prompt_path = Path(row["prompt_path"])
        prompt = prompt_path.read_text(encoding="utf-8")

        print(f"calling {args.model}: {question_id}")
        response = run_one(client, args.model, prompt, args.max_output_tokens)
        answer = response_to_text(response)

        answer_path = run_dir / f"{question_id}_answer.md"
        trace_path = run_dir / f"{question_id}_api_trace.json"

        answer_path.write_text(
            "\n".join(
                [
                    f"# {question_id}",
                    "",
                    f"- model: `{args.model}`",
                    f"- mode: `{row['mode']}`",
                    f"- prompt: `{prompt_path}`",
                    f"- trace: `{trace_path}`",
                    "",
                    "## Answer",
                    "",
                    answer,
                    "",
                ]
            ),
            encoding="utf-8",
        )

        write_json(
            trace_path,
            {
                "question_id": question_id,
                "mode": row["mode"],
                "model": args.model,
                "prompt_path": str(prompt_path),
                "answer_path": str(answer_path),
                "response_id": getattr(response, "id", None),
                "usage": model_dump(getattr(response, "usage", None)),
                "answer": answer,
            },
        )

        summary_rows.append(
            {
                "question_id": question_id,
                "mode": row["mode"],
                "answer_path": str(answer_path),
                "trace_path": str(trace_path),
                "response_id": getattr(response, "id", None),
                "usage": model_dump(getattr(response, "usage", None)),
            }
        )
        print(f"saved: {answer_path}")

    write_json(run_dir / "manifest.json", summary_rows)

    lines = [
        "# Dummy Engineer OpenAI API Run",
        "",
        f"- model: `{args.model}`",
        f"- source_manifest: `{manifest_path}`",
        f"- total: {len(summary_rows)}",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['question_id']}",
                f"- mode: `{row['mode']}`",
                f"- answer: `{row['answer_path']}`",
                f"- trace: `{row['trace_path']}`",
                "",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {run_dir / 'manifest.json'}")
    print(f"saved: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
