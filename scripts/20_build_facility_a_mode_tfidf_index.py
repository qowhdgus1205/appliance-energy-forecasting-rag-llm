#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.rag import build_tfidf_index, read_jsonl


def main() -> None:
    chunk_path = ROOT / "outputs" / "rag" / "facility_a_mode_chunks.jsonl"
    chunks = read_jsonl(chunk_path)
    out_dir = ROOT / "outputs" / "rag" / "facility_a_mode_tfidf_index"
    build_tfidf_index(chunks, out_dir)
    print(f"Saved TF-IDF index to {out_dir}")


if __name__ == "__main__":
    main()

