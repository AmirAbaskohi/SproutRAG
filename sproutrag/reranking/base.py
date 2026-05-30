from __future__ import annotations

from abc import ABC, abstractmethod

from sproutrag.reranking.utils import (
    copy_result_with_score_and_metadata,
    validate_candidates,
    validate_query,
    validate_top_k,
)
from sproutrag.retrieval.schema import RetrievalResult, sort_retrieval_results


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class BaseReranker(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def __init__(self, name: str = "noop") -> None:
        _require_non_empty_str(name, "name")
        self._name = name

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

        updated = [
            copy_result_with_score_and_metadata(
                result,
                result.score,
                {
                    "reranked": False,
                    "reranker_name": self.name,
                    "original_score": result.score,
                    "reranker_score": result.score,
                },
            )
            for result in candidates
        ]
        return sort_retrieval_results(updated, top_k=top_k)
