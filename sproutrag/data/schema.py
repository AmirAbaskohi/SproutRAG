from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_dict(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_list(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")


@dataclass
class RawDocument:
    doc_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.doc_id, "doc_id")
        _require_non_empty_str(self.text, "text")
        _require_dict(self.metadata, "metadata")


@dataclass
class SentenceChunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    sentence_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.chunk_id, "chunk_id")
        _require_non_empty_str(self.doc_id, "doc_id")
        _require_non_empty_str(self.text, "text")
        if not isinstance(self.start_char, int) or self.start_char < 0:
            raise ValueError("start_char must be >= 0")
        if not isinstance(self.end_char, int) or self.end_char <= self.start_char:
            raise ValueError("end_char must be > start_char")
        if not isinstance(self.sentence_ids, list) or not self.sentence_ids:
            raise ValueError("sentence_ids must be a non-empty list")
        if not all(isinstance(value, int) for value in self.sentence_ids):
            raise ValueError("sentence_ids must contain integers")
        _require_dict(self.metadata, "metadata")


@dataclass
class TreeNode:
    node_id: str
    doc_id: str
    text: str
    embedding: list[float] | None = None
    left: str | None = None
    right: str | None = None
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    sentence_chunk_ids: list[str] = field(default_factory=list)
    depth: int = 0
    is_leaf: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.node_id, "node_id")
        _require_non_empty_str(self.doc_id, "doc_id")
        _require_non_empty_str(self.text, "text")
        if not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be >= 0")
        _require_list(self.children, "children")
        _require_list(self.sentence_chunk_ids, "sentence_chunk_ids")
        _require_dict(self.metadata, "metadata")
        if self.is_leaf:
            if self.children:
                raise ValueError("leaf nodes must not have children")
        else:
            if len(self.children) != 2:
                raise ValueError("non-leaf nodes must have exactly two children")
            if not self.left or not self.right:
                raise ValueError("non-leaf nodes must have left and right")
        if self.embedding is not None:
            if not isinstance(self.embedding, list) or not all(
                isinstance(value, (int, float)) for value in self.embedding
            ):
                raise ValueError("embedding must be a list of floats")


@dataclass
class DocumentIndex:
    doc_id: str
    chunks: dict[str, SentenceChunk] = field(default_factory=dict)
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    root_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.doc_id, "doc_id")
        if not isinstance(self.chunks, dict):
            raise ValueError("chunks must be a dictionary")
        if not isinstance(self.nodes, dict):
            raise ValueError("nodes must be a dictionary")
        _require_dict(self.metadata, "metadata")
        if self.root_id is not None and self.root_id not in self.nodes:
            raise ValueError("root_id must exist in nodes")

    def add_chunk(self, chunk: SentenceChunk) -> None:
        if chunk.doc_id != self.doc_id:
            raise ValueError("chunk doc_id does not match DocumentIndex")
        if chunk.chunk_id in self.chunks:
            raise ValueError("chunk_id already present")
        self.chunks[chunk.chunk_id] = chunk

    def add_node(self, node: TreeNode) -> None:
        if node.doc_id != self.doc_id:
            raise ValueError("node doc_id does not match DocumentIndex")
        if node.node_id in self.nodes:
            raise ValueError("node_id already present")
        self.nodes[node.node_id] = node

    def get_root(self) -> TreeNode | None:
        if self.root_id is None:
            return None
        return self.nodes.get(self.root_id)

    def get_leaf_nodes(self) -> list[TreeNode]:
        return [node for node in self.nodes.values() if node.is_leaf]

    def get_internal_nodes(self) -> list[TreeNode]:
        return [node for node in self.nodes.values() if not node.is_leaf]
