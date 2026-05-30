from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import DocumentIndex, RawDocument, SentenceChunk, TreeNode


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, RawDocument):
        return asdict(obj)
    if isinstance(obj, SentenceChunk):
        return asdict(obj)
    if isinstance(obj, TreeNode):
        return asdict(obj)
    if isinstance(obj, DocumentIndex):
        return {
            "doc_id": obj.doc_id,
            "chunks": {key: dataclass_to_dict(value) for key, value in obj.chunks.items()},
            "nodes": {key: dataclass_to_dict(value) for key, value in obj.nodes.items()},
            "root_id": obj.root_id,
            "metadata": obj.metadata,
        }
    raise TypeError(f"Unsupported type for serialization: {type(obj)}")


def raw_document_from_dict(data: dict[str, Any]) -> RawDocument:
    return RawDocument(
        doc_id=data["doc_id"],
        text=data["text"],
        title=data.get("title"),
        metadata=data.get("metadata", {}),
    )


def sentence_chunk_from_dict(data: dict[str, Any]) -> SentenceChunk:
    return SentenceChunk(
        chunk_id=data["chunk_id"],
        doc_id=data["doc_id"],
        text=data["text"],
        start_char=data["start_char"],
        end_char=data["end_char"],
        sentence_ids=list(data["sentence_ids"]),
        metadata=data.get("metadata", {}),
    )


def tree_node_from_dict(data: dict[str, Any]) -> TreeNode:
    return TreeNode(
        node_id=data["node_id"],
        doc_id=data["doc_id"],
        text=data["text"],
        embedding=data.get("embedding"),
        left=data.get("left"),
        right=data.get("right"),
        parent=data.get("parent"),
        children=list(data.get("children", [])),
        sentence_chunk_ids=list(data.get("sentence_chunk_ids", [])),
        depth=data.get("depth", 0),
        is_leaf=data.get("is_leaf", True),
        metadata=data.get("metadata", {}),
    )


def document_index_from_dict(data: dict[str, Any]) -> DocumentIndex:
    chunks = {
        key: sentence_chunk_from_dict(value) for key, value in data.get("chunks", {}).items()
    }
    nodes = {key: tree_node_from_dict(value) for key, value in data.get("nodes", {}).items()}
    return DocumentIndex(
        doc_id=data["doc_id"],
        chunks=chunks,
        nodes=nodes,
        root_id=data.get("root_id"),
        metadata=data.get("metadata", {}),
    )


def save_document_index(index: DocumentIndex, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclass_to_dict(index)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)


def load_document_index(path: str | Path) -> DocumentIndex:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return document_index_from_dict(data)
