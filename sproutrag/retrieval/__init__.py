"""Retrieval utilities for SproutRAG."""

from sproutrag.retrieval.schema import (
    RetrievalResult,
    retrieval_result_from_node,
    sort_retrieval_results,
)
from sproutrag.retrieval.scoring import (
    node_embedding_to_tensor,
    cosine_similarity,
    score_node,
    score_nodes,
)
from sproutrag.retrieval.hierarchical_retriever import HierarchicalRetriever
from sproutrag.retrieval.multi_document_retriever import (
    MultiDocumentRetriever,
    deduplicate_retrieval_results,
    with_corpus_metadata,
)

__all__ = [
    "RetrievalResult",
    "retrieval_result_from_node",
    "sort_retrieval_results",
    "node_embedding_to_tensor",
    "cosine_similarity",
    "score_node",
    "score_nodes",
    "HierarchicalRetriever",
    "MultiDocumentRetriever",
    "deduplicate_retrieval_results",
    "with_corpus_metadata",
]
