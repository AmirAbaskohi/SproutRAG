from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

from sproutrag.data.schema import TreeNode


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_dict(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_str_list(value: list[str], field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")


def _require_finite_number(value: float | int, field_name: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")


@dataclass
class RetrievalResult:
    node_id: str
    doc_id: str
    text: str
    score: float
    depth: int
    is_leaf: bool
    sentence_chunk_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.node_id, "node_id")
        _require_non_empty_str(self.doc_id, "doc_id")
        _require_non_empty_str(self.text, "text")
        _require_finite_number(self.score, "score")
        if not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be an integer >= 0")
        if not isinstance(self.is_leaf, bool):
            raise ValueError("is_leaf must be a boolean")
        _require_str_list(self.sentence_chunk_ids, "sentence_chunk_ids")
        _require_dict(self.metadata, "metadata")


def retrieval_result_from_node(
    node: TreeNode,
    score: float,
    extra_metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    if not isinstance(node, TreeNode):
        raise ValueError("node must be a TreeNode")
    _require_finite_number(score, "score")
    metadata = dict(node.metadata)
    if extra_metadata is not None:
        if not isinstance(extra_metadata, dict):
            raise ValueError("extra_metadata must be a dictionary")
        metadata.update(extra_metadata)
    return RetrievalResult(
        node_id=node.node_id,
        doc_id=node.doc_id,
        text=node.text,
        score=float(score),
        depth=node.depth,
        is_leaf=node.is_leaf,
        sentence_chunk_ids=list(node.sentence_chunk_ids),
        metadata=metadata,
    )


def sort_retrieval_results(
    results: list[RetrievalResult],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise ValueError("results must contain RetrievalResult instances")
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

    def sort_key(item: RetrievalResult) -> tuple[float, int, int, str]:
        leaf_rank = 0 if item.is_leaf else 1
        return (-item.score, item.depth, leaf_rank, item.node_id)

    sorted_results = sorted(results, key=sort_key)
    if top_k is not None:
        return list(sorted_results[:top_k])
    return list(sorted_results)
