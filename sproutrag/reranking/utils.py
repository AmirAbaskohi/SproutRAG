from __future__ import annotations

import math
from typing import Any

from sproutrag.retrieval.schema import RetrievalResult


def validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")


def validate_candidates(candidates: list[RetrievalResult]) -> None:
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    if not all(isinstance(item, RetrievalResult) for item in candidates):
        raise ValueError("candidates must contain RetrievalResult instances")


def validate_top_k(top_k: int | None) -> None:
    if top_k is None:
        return
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _require_finite_number(value: float | int, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")


def copy_result_with_score_and_metadata(
    result: RetrievalResult,
    score: float,
    metadata_update: dict[str, Any],
) -> RetrievalResult:
    if not isinstance(result, RetrievalResult):
        raise ValueError("result must be a RetrievalResult")
    _require_finite_number(score, "score")
    if not isinstance(metadata_update, dict):
        raise ValueError("metadata_update must be a dictionary")
    metadata = dict(result.metadata)
    metadata.update(metadata_update)
    return RetrievalResult(
        node_id=result.node_id,
        doc_id=result.doc_id,
        text=result.text,
        score=float(score),
        depth=result.depth,
        is_leaf=result.is_leaf,
        sentence_chunk_ids=list(result.sentence_chunk_ids),
        metadata=metadata,
    )


def stable_sort_by_reranker_score(
    results: list[RetrievalResult],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    validate_candidates(results)
    validate_top_k(top_k)

    def sort_key(item: RetrievalResult) -> tuple[float, int, int, str, str]:
        leaf_rank = 0 if item.is_leaf else 1
        return (-item.score, item.depth, leaf_rank, item.doc_id, item.node_id)

    sorted_results = sorted(results, key=sort_key)
    if top_k is not None:
        return list(sorted_results[:top_k])
    return list(sorted_results)
