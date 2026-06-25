#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.rag import write_jsonl
from power_forecast_rag.rag import split_markdown_sections


def build_chunks(project_root: Path):
    sources = [
        project_root / "docs" / "knowledge" / "mode_awareness_notes.md",
        project_root / "outputs" / "reports" / "facility_a_inst_heat_multioutput_summary.md",
        project_root / "outputs" / "reports" / "facility_a_gas_like_rag_context.md",
        project_root / "outputs" / "reports" / "facility_a_heating_like_rag_context.md",
    ]
    sources.extend(sorted((project_root / "docs" / "manuals").glob("*.md")))
    chunks = []
    for src in sources:
        text = src.read_text(encoding="utf-8")
        title, sections = split_markdown_sections(text)
        for idx, (section, body) in enumerate(sections):
            chunk_id = f"{src.stem}::{idx:02d}"
            chunk_text = "\n".join(
                [
                    f"Title: {title}",
                    f"Section: {section}",
                    f"Source: {src.relative_to(project_root)}",
                    "",
                    body,
                ]
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": src.stem,
                    "title": title,
                    "section": section,
                    "source_path": str(src.relative_to(project_root)),
                    "text": chunk_text,
                }
            )
    return chunks


def main() -> None:
    out = ROOT / "outputs" / "rag" / "facility_a_mode_chunks.jsonl"
    chunks = build_chunks(ROOT)
    write_jsonl(chunks, out)
    print(f"Saved {len(chunks)} chunks to {out}")


if __name__ == "__main__":
    main()
