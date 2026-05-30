from __future__ import annotations

import math
from typing import Any

import torch

from sproutrag.data.schema import DocumentIndex
from sproutrag.indexing.store import IndexStore
from sproutrag.retrieval.hierarchical_retriever import HierarchicalRetriever
from sproutrag.retrieval.schema import RetrievalResult, sort_retrieval_results


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_threshold(value: float | int) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("threshold must be a finite number")


def _validate_query_embedding(query_embedding: torch.Tensor) -> None:
    if not isinstance(query_embedding, torch.Tensor):
        raise ValueError("query_embedding must be a torch.Tensor")
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be 1-dimensional")
    if not torch.isfinite(query_embedding).all():
        raise ValueError("query_embedding must contain finite values")


def _validate_indexes(indexes: list[DocumentIndex]) -> None:
    if not isinstance(indexes, list) or not indexes:
        raise ValueError("indexes must be a non-empty list")
    if not all(isinstance(index, DocumentIndex) for index in indexes):
        raise ValueError("indexes must contain DocumentIndex instances")
    doc_ids = [index.doc_id for index in indexes]
    if not all(isinstance(doc_id, str) and doc_id.strip() for doc_id in doc_ids):
        raise ValueError("each index must have a non-empty doc_id")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("indexes must not contain duplicate doc_ids")


def with_corpus_metadata(
    result: RetrievalResult,
    source_doc_id: str,
    per_document_top_k: int,
    global_top_k: int,
) -> RetrievalResult:
    if not isinstance(result, RetrievalResult):
        raise ValueError("result must be a RetrievalResult")
    if not isinstance(source_doc_id, str) or not source_doc_id.strip():
        raise ValueError("source_doc_id must be a non-empty string")
    _validate_positive_int(per_document_top_k, "per_document_top_k")
    _validate_positive_int(global_top_k, "global_top_k")
    metadata = dict(result.metadata)
    metadata.update(
        {
            "corpus_retrieval": True,
            "source_doc_id": source_doc_id,
            "per_document_top_k": per_document_top_k,
            "global_top_k": global_top_k,
            "document_rank_source": "hierarchical_retriever",
        }
    )
    return RetrievalResult(
        node_id=result.node_id,
        doc_id=result.doc_id,
        text=result.text,
        score=result.score,
        depth=result.depth,
        is_leaf=result.is_leaf,
        sentence_chunk_ids=list(result.sentence_chunk_ids),
        metadata=metadata,
    )


def _result_rank_key(result: RetrievalResult) -> tuple[float, int, int, str]:
    leaf_rank = 0 if result.is_leaf else 1
    return (-result.score, result.depth, leaf_rank, result.node_id)


def deduplicate_retrieval_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise ValueError("results must contain RetrievalResult instances")
    best: dict[tuple[str, str], RetrievalResult] = {}
    for result in results:
        key = (result.doc_id, result.node_id)
        if key not in best:
            best[key] = result
            continue
        existing = best[key]
        if _result_rank_key(result) < _result_rank_key(existing):
            best[key] = result
    return list(best.values())


class MultiDocumentRetriever:
    def __init__(
        self,
        encoder: Any | None = None,
        single_document_retriever: HierarchicalRetriever | None = None,
        collect_strategy: str = "threshold",
        include_root: bool = False,
        reranker: Any | None = None,
    ) -> None:
        if encoder is not None and not hasattr(encoder, "encode_query"):
            raise ValueError("encoder must implement encode_query")
        if collect_strategy not in {"threshold", "visited"}:
            raise ValueError("collect_strategy must be 'threshold' or 'visited'")
        if not isinstance(include_root, bool):
            raise ValueError("include_root must be a boolean")
        if reranker is not None and not hasattr(reranker, "rerank"):
            raise ValueError("reranker must implement rerank")
        if single_document_retriever is not None and not isinstance(
            single_document_retriever, HierarchicalRetriever
        ):
            raise ValueError("single_document_retriever must be a HierarchicalRetriever")
        if single_document_retriever is None:
            single_document_retriever = HierarchicalRetriever(
                encoder=None,
                collect_strategy=collect_strategy,
                include_root=include_root,
            )
        self.encoder = encoder
        self.single_document_retriever = single_document_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        indexes: list[DocumentIndex],
        top_k: int = 10,
        per_document_top_k: int = 5,
        beam_width: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        if self.encoder is None:
            raise ValueError("encoder must be provided to retrieve using query string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        embedding = self.encoder.encode_query(query)
        results = self.retrieve_from_embedding(
            embedding,
            indexes,
            top_k=top_k,
            per_document_top_k=per_document_top_k,
            beam_width=beam_width,
            threshold=threshold,
        )
        if self.reranker is not None:
            return self.reranker.rerank(query, results, top_k=top_k)
        return results

    def retrieve_from_embedding(
        self,
        query_embedding: torch.Tensor,
        indexes: list[DocumentIndex],
        top_k: int = 10,
        per_document_top_k: int = 5,
        beam_width: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        _validate_query_embedding(query_embedding)
        _validate_indexes(indexes)
        _validate_positive_int(top_k, "top_k")
        _validate_positive_int(per_document_top_k, "per_document_top_k")
        _validate_positive_int(beam_width, "beam_width")
        _validate_threshold(threshold)

        all_results: list[RetrievalResult] = []
        for index in indexes:
            results = self.single_document_retriever.retrieve_from_embedding(
                query_embedding=query_embedding,
                index=index,
                top_k=per_document_top_k,
                beam_width=beam_width,
                threshold=threshold,
            )
            for result in results:
                all_results.append(
                    with_corpus_metadata(
                        result,
                        source_doc_id=index.doc_id,
                        per_document_top_k=per_document_top_k,
                        global_top_k=top_k,
                    )
                )

        deduped = deduplicate_retrieval_results(all_results)
        sorted_results = sort_retrieval_results(deduped, top_k=top_k)
        return sorted_results

    def retrieve_from_store(
        self,
        query: str,
        store: IndexStore,
        top_k: int = 10,
        per_document_top_k: int = 5,
        beam_width: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        if self.encoder is None:
            raise ValueError("encoder must be provided to retrieve using query string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(store, IndexStore):
            raise ValueError("store must be an IndexStore")
        indexes = store.load_all()
        return self.retrieve(
            query,
            indexes,
            top_k=top_k,
            per_document_top_k=per_document_top_k,
            beam_width=beam_width,
            threshold=threshold,
        )

    def retrieve_from_store_embedding(
        self,
        query_embedding: torch.Tensor,
        store: IndexStore,
        top_k: int = 10,
        per_document_top_k: int = 5,
        beam_width: int = 5,
        threshold: float = 0.0,
    ) -> list[RetrievalResult]:
        if not isinstance(store, IndexStore):
            raise ValueError("store must be an IndexStore")
        indexes = store.load_all()
        return self.retrieve_from_embedding(
            query_embedding,
            indexes,
            top_k=top_k,
            per_document_top_k=per_document_top_k,
            beam_width=beam_width,
            threshold=threshold,
        )
