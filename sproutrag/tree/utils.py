from __future__ import annotations

from typing import Any

import torch

from sproutrag.data.schema import SentenceChunk


def validate_tree_inputs(
    doc_id: str,
    chunks: list[SentenceChunk],
    embeddings: torch.Tensor,
    mutual_attention: torch.Tensor,
) -> None:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks must be a non-empty list")
    if not all(isinstance(chunk, SentenceChunk) for chunk in chunks):
        raise ValueError("chunks must contain SentenceChunk instances")
    if any(chunk.doc_id != doc_id for chunk in chunks):
        raise ValueError("all chunks must have matching doc_id")

    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("embeddings must be a torch.Tensor")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-dimensional")
    if embeddings.shape[0] != len(chunks):
        raise ValueError("embeddings.shape[0] must match number of chunks")

    if not isinstance(mutual_attention, torch.Tensor):
        raise ValueError("mutual_attention must be a torch.Tensor")
    if mutual_attention.ndim != 2:
        raise ValueError("mutual_attention must be 2-dimensional")
    if mutual_attention.shape[0] != mutual_attention.shape[1]:
        raise ValueError("mutual_attention must be square")
    if mutual_attention.shape[0] != len(chunks):
        raise ValueError("mutual_attention.shape[0] must match number of chunks")
    if not torch.isfinite(mutual_attention).all():
        raise ValueError("mutual_attention must contain only finite values")


def tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError("tensor must be 1-dimensional")
    return tensor.detach().cpu().tolist()


def cosine_similarity_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("inputs must be torch.Tensors")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("inputs must be 1-dimensional")
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    denom = (torch.norm(a) * torch.norm(b)).clamp_min(eps)
    return torch.sum(a * b) / denom


def make_leaf_node_id(doc_id: str, chunk_index: int) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")
    if not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")
    return f"{doc_id}::leaf::{chunk_index}"


def make_internal_node_id(doc_id: str, merge_index: int) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")
    if not isinstance(merge_index, int) or merge_index < 0:
        raise ValueError("merge_index must be >= 0")
    return f"{doc_id}::internal::{merge_index}"


def concatenate_child_text(left_text: str, right_text: str) -> str:
    if not isinstance(left_text, str) or not left_text.strip():
        raise ValueError("left_text must be a non-empty string")
    if not isinstance(right_text, str) or not right_text.strip():
        raise ValueError("right_text must be a non-empty string")
    left = left_text.strip()
    right = right_text.strip()
    return f"{left}\n{right}"
