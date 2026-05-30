from __future__ import annotations

import math

from sproutrag.reranking.base import BaseReranker
from sproutrag.reranking.utils import (
    copy_result_with_score_and_metadata,
    stable_sort_by_reranker_score,
    validate_candidates,
    validate_query,
    validate_top_k,
)
from sproutrag.retrieval.schema import RetrievalResult


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_finite_number(value: float | int, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")


class ScoreReranker(BaseReranker):
    def __init__(self, name: str = "score", metadata_score_key: str | None = None) -> None:
        _require_non_empty_str(name, "name")
        if metadata_score_key is not None:
            _require_non_empty_str(metadata_score_key, "metadata_score_key")
        self._name = name
        self.metadata_score_key = metadata_score_key

    @property
    def name(self) -> str:
        return self._name

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        validate_query(query)
        validate_candidates(candidates)
        validate_top_k(top_k)

        updated: list[RetrievalResult] = []
        for candidate in candidates:
            if self.metadata_score_key is None:
                reranker_score = candidate.score
                score_source = "score"
            else:
                if self.metadata_score_key not in candidate.metadata:
                    raise ValueError("metadata_score_key missing from candidate metadata")
                value = candidate.metadata[self.metadata_score_key]
                _require_finite_number(value, "metadata_score_key")
                reranker_score = float(value)
                score_source = f"metadata:{self.metadata_score_key}"

            updated.append(
                copy_result_with_score_and_metadata(
                    candidate,
                    reranker_score,
                    {
                        "reranked": True,
                        "reranker_name": self.name,
                        "original_score": candidate.score,
                        "reranker_score": reranker_score,
                        "reranker_score_source": score_source,
                    },
                )
            )

        return stable_sort_by_reranker_score(updated, top_k=top_k)
