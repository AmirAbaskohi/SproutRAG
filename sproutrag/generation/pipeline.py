from __future__ import annotations

from typing import Any

from sproutrag.data.schema import DocumentIndex
from sproutrag.generation.base import BaseGenerator
from sproutrag.generation.schema import GeneratedAnswer
from sproutrag.indexing.store import IndexStore


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_indexes(indexes: list[DocumentIndex]) -> None:
    if not isinstance(indexes, list) or not indexes:
        raise ValueError("indexes must be a non-empty list of DocumentIndex")
    if not all(isinstance(item, DocumentIndex) for item in indexes):
        raise ValueError("indexes must contain DocumentIndex instances")


class RAGPipeline:
    def __init__(self, retriever: Any, generator: BaseGenerator) -> None:
        if not (hasattr(retriever, "retrieve") or hasattr(retriever, "retrieve_from_embedding")):
            raise ValueError("retriever must implement retrieve or retrieve_from_embedding")
        if not isinstance(generator, BaseGenerator):
            raise ValueError("generator must be a BaseGenerator")
        self.retriever = retriever
        self.generator = generator

    def answer(
        self,
        query: str,
        indexes: list[DocumentIndex] | DocumentIndex,
        top_k: int = 5,
        **retrieval_kwargs: Any,
    ) -> GeneratedAnswer:
        _require_non_empty_str(query, "query")
        _require_positive_int(top_k, "top_k")

        if isinstance(indexes, DocumentIndex):
            contexts = self.retriever.retrieve(
                query=query, index=indexes, top_k=top_k, **retrieval_kwargs
            )
        elif isinstance(indexes, list):
            _validate_indexes(indexes)
            contexts = self.retriever.retrieve(
                query=query, indexes=indexes, top_k=top_k, **retrieval_kwargs
            )
        else:
            raise ValueError("indexes must be a DocumentIndex or list of DocumentIndex")

        generated = self.generator.generate(query, contexts)
        metadata = dict(generated.metadata)
        metadata.update(
            {
                "pipeline": "RAGPipeline",
                "retriever_type": type(self.retriever).__name__,
                "generator_type": type(self.generator).__name__,
                "top_k": top_k,
            }
        )
        return GeneratedAnswer(
            query=generated.query,
            answer=generated.answer,
            contexts=list(generated.contexts),
            prompt=generated.prompt,
            metadata=metadata,
        )

    def answer_from_store(
        self,
        query: str,
        store: IndexStore,
        top_k: int = 5,
        **retrieval_kwargs: Any,
    ) -> GeneratedAnswer:
        if not isinstance(store, IndexStore):
            raise ValueError("store must be an IndexStore")
        indexes = store.load_all()
        generated = self.answer(query, indexes, top_k=top_k, **retrieval_kwargs)
        metadata = dict(generated.metadata)
        metadata.update({"source": "IndexStore", "index_name": store.index_name})
        return GeneratedAnswer(
            query=generated.query,
            answer=generated.answer,
            contexts=list(generated.contexts),
            prompt=generated.prompt,
            metadata=metadata,
        )
