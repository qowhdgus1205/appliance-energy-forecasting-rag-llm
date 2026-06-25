#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.rag import format_search_results, load_embedding_index, search_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the OpenAI embedding RAG index.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument(
        "--index-dir",
        default="outputs/rag/facility_a_mode_embedding_index",
        help="Embedding index directory.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not set. Export it before querying the embedding index.")

    index_dir = ROOT / args.index_dir if not Path(args.index_dir).is_absolute() else Path(args.index_dir)
    matrix, metadata, config = load_embedding_index(index_dir)
    results = search_embedding(args.query, matrix, metadata, config, top_k=args.top_k)
    print(format_search_results(results))


if __name__ == "__main__":
    main()
