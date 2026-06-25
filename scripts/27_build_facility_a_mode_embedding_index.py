#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from power_forecast_rag.rag import build_embedding_index, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OpenAI embedding index for the mode-aware RAG corpus.")
    parser.add_argument("--chunks", default="outputs/rag/facility_a_mode_chunks.jsonl", help="Input chunk JSONL.")
    parser.add_argument("--out-dir", default="outputs/rag/facility_a_mode_embedding_index", help="Output index directory.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model. Defaults to OPENAI_EMBEDDING_MODEL or text-embedding-3-small.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dimensions", type=int, help="Optional shortened embedding dimension.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not set. Export it before building the embedding index.")

    chunk_path = ROOT / args.chunks if not Path(args.chunks).is_absolute() else Path(args.chunks)
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    chunks = read_jsonl(chunk_path)
    build_embedding_index(
        chunks,
        out_dir,
        model=args.model,
        batch_size=args.batch_size,
        dimensions=args.dimensions,
    )
    print(f"Saved OpenAI embedding index to {out_dir}")


if __name__ == "__main__":
    main()
