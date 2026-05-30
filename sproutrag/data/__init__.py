"""Data layer utilities for SproutRAG."""

from .schema import DocumentIndex, RawDocument, SentenceChunk, TreeNode
from .preprocessing import (
    build_empty_document_index,
    chunk_sentences,
    make_leaf_nodes,
    normalize_whitespace,
    split_sentences,
)
from .serialization import (
    dataclass_to_dict,
    document_index_from_dict,
    load_document_index,
    raw_document_from_dict,
    save_document_index,
    sentence_chunk_from_dict,
    tree_node_from_dict,
)

__all__ = [
    "RawDocument",
    "SentenceChunk",
    "TreeNode",
    "DocumentIndex",
    "normalize_whitespace",
    "split_sentences",
    "chunk_sentences",
    "make_leaf_nodes",
    "build_empty_document_index",
    "dataclass_to_dict",
    "raw_document_from_dict",
    "sentence_chunk_from_dict",
    "tree_node_from_dict",
    "document_index_from_dict",
    "save_document_index",
    "load_document_index",
]
