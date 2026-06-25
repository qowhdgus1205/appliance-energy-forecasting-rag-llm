from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from openai import OpenAI
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_markdown_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = text.splitlines()
    title = "Untitled"
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    cleaned = []
    for heading, section_lines in sections:
        body = "\n".join(section_lines).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if body:
            cleaned.append((heading, body))
    return title, cleaned


def build_lg_chunks(project_root: Path) -> list[dict[str, object]]:
    sources = [
        project_root / "docs" / "knowledge" / "inst_heat_operating_notes.md",
        project_root / "docs" / "knowledge" / "mode_awareness_notes.md",
        project_root / "outputs" / "reports" / "facility_a_inst_heat_rag_context.md",
        project_root / "outputs" / "reports" / "facility_a_inst_heat_exogenous_predictions_root_cause.md",
        project_root / "outputs" / "reports" / "facility_a_inst_heat_exogenous_incident.md",
    ]
    sources.extend(sorted((project_root / "docs" / "manuals").glob("*.md")))
    chunks: list[dict[str, object]] = []

    for src in sources:
        text = src.read_text(encoding="utf-8")
        title, sections = split_markdown_sections(text)
        doc_id = src.stem
        for idx, (section, body) in enumerate(sections):
            chunk_id = f"{doc_id}::{idx:02d}"
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
                    "doc_id": doc_id,
                    "title": title,
                    "section": section,
                    "source_path": str(src.relative_to(project_root)),
                    "text": chunk_text,
                }
            )
    return chunks


def build_tfidf_index(chunks: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [str(chunk["text"]) for chunk in chunks]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    sparse.save_npz(output_dir / "lg_tfidf_matrix.npz", matrix)
    dump(vectorizer, output_dir / "lg_tfidf_vectorizer.joblib")
    write_jsonl(chunks, output_dir / "lg_metadata.jsonl")
    (output_dir / "index_config.json").write_text(
        json.dumps(
            {
                "retriever": "tfidf",
                "vectorizer_file": "lg_tfidf_vectorizer.joblib",
                "matrix_file": "lg_tfidf_matrix.npz",
                "metadata_file": "lg_metadata.jsonl",
                "similarity": "cosine",
                "documents": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_tfidf_index(index_dir: Path):
    config = json.loads((index_dir / "index_config.json").read_text(encoding="utf-8"))
    vectorizer = load(index_dir / config["vectorizer_file"])
    matrix = sparse.load_npz(index_dir / config["matrix_file"])
    metadata = read_jsonl(index_dir / config["metadata_file"])
    return vectorizer, matrix, metadata, config


def search_tfidf(query: str, vectorizer, matrix, metadata, top_k: int = 5) -> list[dict[str, object]]:
    query_vec = vectorizer.transform([query])
    scores = matrix @ query_vec.T
    scores = np.asarray(scores.todense()).ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        item = dict(metadata[int(idx)])
        item["score"] = float(scores[int(idx)])
        results.append(item)
    return results


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=np.float32), where=norms != 0)


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
    dimensions: int | None = None,
    client: OpenAI | None = None,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = client or OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        kwargs: dict[str, Any] = {"model": model, "input": batch, "encoding_format": "float"}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        response = client.embeddings.create(**kwargs)
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


def build_embedding_index(
    chunks: list[dict[str, object]],
    output_dir: Path,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
    dimensions: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = embed_texts(texts, model=model, batch_size=batch_size, dimensions=dimensions)
    embeddings = _normalize_rows(embeddings)
    np.save(output_dir / "embedding_matrix.npy", embeddings)
    write_jsonl(chunks, output_dir / "embedding_metadata.jsonl")
    (output_dir / "index_config.json").write_text(
        json.dumps(
            {
                "retriever": "openai_embedding",
                "embedding_model": model,
                "dimensions": int(embeddings.shape[1]) if embeddings.size else dimensions,
                "requested_dimensions": dimensions,
                "matrix_file": "embedding_matrix.npy",
                "metadata_file": "embedding_metadata.jsonl",
                "similarity": "cosine",
                "documents": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_embedding_index(index_dir: Path):
    config = json.loads((index_dir / "index_config.json").read_text(encoding="utf-8"))
    matrix = np.load(index_dir / config["matrix_file"])
    metadata = read_jsonl(index_dir / config["metadata_file"])
    return matrix, metadata, config


def search_embedding(
    query: str,
    matrix: np.ndarray,
    metadata: list[dict[str, object]],
    config: dict[str, object],
    top_k: int = 5,
    client: OpenAI | None = None,
) -> list[dict[str, object]]:
    query_embedding = embed_texts(
        [query],
        model=str(config["embedding_model"]),
        dimensions=config.get("requested_dimensions") if config.get("requested_dimensions") else None,
        client=client,
    )
    query_embedding = _normalize_rows(query_embedding)
    scores = matrix @ query_embedding[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        item = dict(metadata[int(idx)])
        item["score"] = float(scores[int(idx)])
        item["retriever"] = "openai_embedding"
        results.append(item)
    return results


def format_search_results(results: list[dict[str, object]]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            "\n".join(
                [
                    f"### {r.get('chunk_id')} | {r.get('title')} / {r.get('section')}",
                    f"- score: {float(r.get('score', 0)):.4f}",
                    f"- source: {r.get('source_path')}",
                    "",
                    str(r.get("text", "")),
                ]
            )
        )
    return "\n\n".join(blocks)
