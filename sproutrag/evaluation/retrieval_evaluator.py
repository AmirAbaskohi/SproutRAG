from __future__ import annotations

from typing import Any, Callable

from sproutrag.data.schema import DocumentIndex
from sproutrag.evaluation.schema import RetrievalEvaluationExample, RetrievalEvaluationResult
from sproutrag.evaluation.retrieval_metrics import compute_retrieval_metrics, aggregate_metric_dicts
from sproutrag.indexing.store import IndexStore
from sproutrag.retrieval.schema import RetrievalResult


def default_result_id_extractor(result: RetrievalResult) -> str:
    return result.node_id


def extract_result_ids(
    results: list[RetrievalResult],
    id_extractor: Callable[[RetrievalResult], str] | None = None,
) -> list[str]:
    if not isinstance(results, list):
        raise ValueError("results must be a list of RetrievalResult")
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise ValueError("results must contain RetrievalResult instances")
    extractor = id_extractor or default_result_id_extractor
    ids: list[str] = []
    for result in results:
        extracted = extractor(result)
        if not isinstance(extracted, str) or not extracted.strip():
            raise ValueError("extracted ids must be non-empty strings")
        ids.append(extracted)
    return ids


def _validate_indexes(indexes: list[DocumentIndex]) -> None:
    if not isinstance(indexes, list) or not indexes:
        raise ValueError("indexes must be a non-empty list of DocumentIndex")
    if not all(isinstance(item, DocumentIndex) for item in indexes):
        raise ValueError("indexes must contain DocumentIndex instances")


def _validate_ks(ks: list[int]) -> None:
    if not isinstance(ks, list) or not ks:
        raise ValueError("ks must be a non-empty list of integers")
    for value in ks:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ks must contain positive integers")


class RetrievalEvaluator:
    def __init__(
        self,
        retriever: Any,
        ks: list[int] = [1, 3, 5],
        id_extractor: Callable[[RetrievalResult], str] | None = None,
    ) -> None:
        if not hasattr(retriever, "retrieve"):
            raise ValueError("retriever must implement retrieve")
        _validate_ks(ks)
        self.retriever = retriever
        self.ks = list(ks)
        self.id_extractor = id_extractor

    def evaluate_example(
        self,
        example: RetrievalEvaluationExample,
        indexes: DocumentIndex | list[DocumentIndex],
        **retrieval_kwargs: Any,
    ) -> RetrievalEvaluationResult:
        if not isinstance(example, RetrievalEvaluationExample):
            raise ValueError("example must be a RetrievalEvaluationExample")

        top_k = max(self.ks)
        if isinstance(indexes, DocumentIndex):
            results = self.retriever.retrieve(
                query=example.query,
                index=indexes,
                top_k=top_k,
                **retrieval_kwargs,
            )
        elif isinstance(indexes, list):
            _validate_indexes(indexes)
            results = self.retriever.retrieve(
                query=example.query,
                indexes=indexes,
                top_k=top_k,
                **retrieval_kwargs,
            )
        else:
            raise ValueError("indexes must be a DocumentIndex or list of DocumentIndex")

        retrieved_ids = extract_result_ids(results, self.id_extractor)
        metrics = compute_retrieval_metrics(retrieved_ids, example.relevant_ids, ks=self.ks)
        return RetrievalEvaluationResult(
            example_id=example.example_id,
            query=example.query,
            retrieved_ids=retrieved_ids,
            relevant_ids=list(example.relevant_ids),
            metrics=metrics,
            metadata={
                "ks": list(self.ks),
                "num_retrieved": len(retrieved_ids),
                "example_metadata": dict(example.metadata),
            },
        )

    def evaluate(
        self,
        examples: list[RetrievalEvaluationExample],
        indexes: DocumentIndex | list[DocumentIndex],
        **retrieval_kwargs: Any,
    ) -> tuple[list[RetrievalEvaluationResult], dict[str, float]]:
        if not isinstance(examples, list) or not examples:
            raise ValueError("examples must be a non-empty list of RetrievalEvaluationExample")
        if not all(isinstance(item, RetrievalEvaluationExample) for item in examples):
            raise ValueError("examples must contain RetrievalEvaluationExample instances")

        results: list[RetrievalEvaluationResult] = []
        for example in examples:
            results.append(self.evaluate_example(example, indexes, **retrieval_kwargs))

        aggregate = aggregate_metric_dicts([result.metrics for result in results])
        return results, aggregate

    def evaluate_from_store(
        self,
        examples: list[RetrievalEvaluationExample],
        store: IndexStore,
        **retrieval_kwargs: Any,
    ) -> tuple[list[RetrievalEvaluationResult], dict[str, float]]:
        if not isinstance(store, IndexStore):
            raise ValueError("store must be an IndexStore")
        indexes = store.load_all()
        return self.evaluate(examples, indexes, **retrieval_kwargs)
