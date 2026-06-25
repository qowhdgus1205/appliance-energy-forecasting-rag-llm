#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.rag import format_search_results, load_tfidf_index, search_tfidf


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Industrial Energy TF-IDF RAG index.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument(
        "--index-dir",
        default="outputs/rag/lg_tfidf_index",
        help="TF-IDF index directory.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    index_dir = ROOT / args.index_dir if not Path(args.index_dir).is_absolute() else Path(args.index_dir)
    vectorizer, matrix, metadata, _ = load_tfidf_index(index_dir)
    results = search_tfidf(args.query, vectorizer, matrix, metadata, top_k=args.top_k)
    print(format_search_results(results))


if __name__ == "__main__":
    main()

