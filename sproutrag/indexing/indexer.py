from __future__ import annotations

from typing import Any

import torch

from sproutrag.aggregation.attention_aggregator import AttentionAggregator
from sproutrag.data.preprocessing import chunk_sentences
from sproutrag.data.schema import DocumentIndex, RawDocument
from sproutrag.indexing.store import IndexStore
from sproutrag.encoding.schema import EncodedDocument
from sproutrag.tree.builder import AttentionTreeBuilder


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_encoder(encoder: Any) -> None:
    if not hasattr(encoder, "encode_document"):
        raise ValueError("encoder must implement encode_document")


class SproutRAGIndexer:
    def __init__(
        self,
        encoder: Any,
        aggregator: Any | None = None,
        tree_builder: Any | None = None,
        max_sentences_per_chunk: int = 2,
    ) -> None:
        _require_encoder(encoder)
        if aggregator is not None and not callable(aggregator):
            raise ValueError("aggregator must be callable")
        if tree_builder is None:
            tree_builder = AttentionTreeBuilder()
        if not hasattr(tree_builder, "build_tree"):
            raise ValueError("tree_builder must implement build_tree")
        _require_positive_int(max_sentences_per_chunk, "max_sentences_per_chunk")

        self.encoder = encoder
        self.aggregator = aggregator
        self.tree_builder = tree_builder
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def _get_or_create_aggregator(self, attentions: torch.Tensor) -> Any:
        if self.aggregator is not None:
            return self.aggregator
        if not isinstance(attentions, torch.Tensor):
            raise ValueError("attentions must be a torch.Tensor")
        if attentions.ndim != 4:
            raise ValueError("attentions must be 4-dimensional")
        num_layers, num_heads = attentions.shape[0], attentions.shape[1]
        self.aggregator = AttentionAggregator(
            num_layers=num_layers,
            num_heads=num_heads,
            init_strategy="uniform",
            mask_diagonal=True,
        )
        return self.aggregator

    def index_document(self, document: RawDocument) -> DocumentIndex:
        if not isinstance(document, RawDocument):
            raise ValueError("document must be a RawDocument")

        chunks = chunk_sentences(document, max_sentences_per_chunk=self.max_sentences_per_chunk)
        encoded: EncodedDocument = self.encoder.encode_document(chunks, return_attentions=True)
        if encoded.attentions is None:
            raise ValueError("encoded.attentions must not be None")
        if encoded.embeddings.shape[0] != len(encoded.chunk_ids):
            raise ValueError("encoded.embeddings shape does not match encoded.chunk_ids")

        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        filtered_chunks = [chunk_lookup[cid] for cid in encoded.chunk_ids if cid in chunk_lookup]
        if len(filtered_chunks) != encoded.embeddings.shape[0]:
            raise ValueError("filtered chunks do not match embeddings")

        aggregator = self._get_or_create_aggregator(encoded.attentions)
        mutual_attention = aggregator(encoded.attentions)
        index = self.tree_builder.build_tree(
            doc_id=document.doc_id,
            chunks=filtered_chunks,
            embeddings=encoded.embeddings,
            mutual_attention=mutual_attention,
        )

        index.metadata.update(
            {
                "source_title": document.title,
                "source_metadata": document.metadata,
                "max_sentences_per_chunk": self.max_sentences_per_chunk,
                "encoder_metadata": encoded.metadata,
                "num_original_chunks": len(chunks),
                "num_encoded_chunks": len(filtered_chunks),
                "skipped_chunk_ids": encoded.metadata.get("skipped_chunk_ids", []),
            }
        )
        return index

    def index_documents(self, documents: list[RawDocument]) -> list[DocumentIndex]:
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty list")
        if not all(isinstance(doc, RawDocument) for doc in documents):
            raise ValueError("documents must contain RawDocument instances")
        doc_ids = [doc.doc_id for doc in documents]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("documents contain duplicate doc_ids")
        return [self.index_document(doc) for doc in documents]

    def index_and_save_documents(
        self,
        documents: list[RawDocument],
        store: IndexStore,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        if not isinstance(store, IndexStore):
            raise ValueError("store must be an IndexStore")
        indexes = self.index_documents(documents)
        manifest_metadata = {"max_sentences_per_chunk": self.max_sentences_per_chunk}
        if metadata:
            manifest_metadata.update(metadata)
        return store.save_many(indexes, metadata=manifest_metadata)
