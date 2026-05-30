from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sproutrag.data.schema import RawDocument
from sproutrag.generation.schema import GeneratedAnswer, generated_answer_to_dict
from sproutrag.retrieval.schema import RetrievalResult


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config file")
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Override config values, e.g. --override retrieval.top_k=20",
    )


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def load_documents_jsonl(path: str | Path) -> list[RawDocument]:
    file_path = Path(path)
    documents: list[RawDocument] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            try:
                documents.append(
                    RawDocument(
                        doc_id=data["doc_id"],
                        text=data["text"],
                        title=data.get("title"),
                        metadata=data.get("metadata", {}),
                    )
                )
            except Exception as exc:
                raise ValueError(f"invalid document on line {line_number}") from exc
    return documents


def save_retrieval_results_json(results: list[RetrievalResult], path: str | Path) -> None:
    payload = {
        "results": [
            {
                "node_id": result.node_id,
                "doc_id": result.doc_id,
                "text": result.text,
                "score": result.score,
                "depth": result.depth,
                "is_leaf": result.is_leaf,
                "sentence_chunk_ids": list(result.sentence_chunk_ids),
                "metadata": dict(result.metadata),
            }
            for result in results
        ]
    }
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_generated_answer_json(answer: GeneratedAnswer, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(generated_answer_to_dict(answer), handle, indent=2, sort_keys=True)


def load_query_from_config_or_path(query: str | None, query_path: str | None) -> str:
    if query is not None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return query
    if query_path is None:
        raise ValueError("query or query_path must be provided")
    path = Path(query_path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("query loaded from file is empty")
    return text


def exit_with_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 1
