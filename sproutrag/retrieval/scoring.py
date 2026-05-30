from __future__ import annotations

import math
from typing import Iterable

import torch

from sproutrag.data.schema import TreeNode


def node_embedding_to_tensor(
    embedding: list[float] | torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(embedding, torch.Tensor):
        tensor = embedding.detach().clone().to(dtype=dtype, device="cpu")
    elif isinstance(embedding, list):
        if not embedding:
            raise ValueError("embedding list must be non-empty")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError("embedding list must contain numbers")
        tensor = torch.tensor(embedding, dtype=dtype)
    else:
        raise ValueError("embedding must be a list of numbers or a torch.Tensor")

    if tensor.ndim != 1:
        raise ValueError("embedding tensor must be 1-dimensional")
    if tensor.numel() == 0:
        raise ValueError("embedding tensor must be non-empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("embedding tensor must contain finite values")
    return tensor


def cosine_similarity(
    query_embedding: torch.Tensor,
    node_embedding: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    if not isinstance(query_embedding, torch.Tensor) or not isinstance(
        node_embedding, torch.Tensor
    ):
        raise ValueError("query_embedding and node_embedding must be torch.Tensors")
    if query_embedding.ndim != 1 or node_embedding.ndim != 1:
        raise ValueError("query_embedding and node_embedding must be 1-dimensional")
    if query_embedding.shape != node_embedding.shape:
        raise ValueError("query_embedding and node_embedding must have the same shape")
    if not torch.isfinite(query_embedding).all() or not torch.isfinite(
        node_embedding
    ).all():
        raise ValueError("embeddings must contain finite values")

    denom = (torch.norm(query_embedding) * torch.norm(node_embedding)).clamp_min(eps)
    score = torch.sum(query_embedding * node_embedding) / denom
    return float(score.detach().cpu())


def score_node(
    query_embedding: torch.Tensor,
    node: TreeNode,
) -> float:
    if not isinstance(node, TreeNode):
        raise ValueError("node must be a TreeNode")
    if node.embedding is None:
        raise ValueError("node.embedding must not be None")
    node_tensor = node_embedding_to_tensor(node.embedding)
    return cosine_similarity(query_embedding, node_tensor)


def score_nodes(
    query_embedding: torch.Tensor,
    nodes: list[TreeNode],
) -> dict[str, float]:
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")
    if not all(isinstance(node, TreeNode) for node in nodes):
        raise ValueError("nodes must contain TreeNode instances")
    return {node.node_id: score_node(query_embedding, node) for node in nodes}
