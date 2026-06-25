#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.rag import build_lg_chunks, build_tfidf_index


def main() -> None:
    chunks = build_lg_chunks(ROOT)
    output_dir = ROOT / "outputs" / "rag" / "lg_tfidf_index"
    build_tfidf_index(chunks, output_dir)
    print(f"Saved TF-IDF index to {output_dir}")


if __name__ == "__main__":
    main()

