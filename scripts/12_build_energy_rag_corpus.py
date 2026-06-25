#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.rag import build_lg_chunks, write_jsonl


def main() -> None:
    output_path = ROOT / "outputs" / "rag" / "lg_chunks.jsonl"
    chunks = build_lg_chunks(ROOT)
    write_jsonl(chunks, output_path)
    print(f"Industrial Energy chunks: {len(chunks)}")
    print(f"Saved: {output_path}")
    for chunk in chunks[:3]:
        print()
        print(f"[{chunk['chunk_id']}] {chunk['title']} / {chunk['section']}")
        print(chunk["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()

