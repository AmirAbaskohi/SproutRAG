from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_embedding_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_dim: int | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    if expected_dim is not None and tensor.dim() != expected_dim:
        raise ValueError(f"{name} must be a {expected_dim}D tensor")


def l2_normalize_embeddings(
    embeddings: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    validate_embedding_tensor(embeddings, "embeddings")
    return F.normalize(embeddings, p=2, dim=dim, eps=eps)


def mean_pool_sequence_embeddings(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    validate_embedding_tensor(token_embeddings, "token_embeddings", expected_dim=3)
    validate_embedding_tensor(attention_mask, "attention_mask", expected_dim=2)
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention_mask shape must match token_embeddings batch and seq_len")

    mask = (attention_mask > 0).to(token_embeddings.dtype).unsqueeze(-1)
    masked = token_embeddings * mask
    counts = mask.sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / counts


def mean_pool_sentence_embeddings(
    sentence_embeddings: torch.Tensor,
    sentence_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_embedding_tensor(sentence_embeddings, "sentence_embeddings")
    if sentence_embeddings.dim() not in (2, 3):
        raise ValueError("sentence_embeddings must be 2D or 3D")

    if sentence_mask is None:
        return sentence_embeddings.mean(dim=-2)

    validate_embedding_tensor(sentence_mask, "sentence_mask")
    if sentence_embeddings.dim() == 3:
        if sentence_mask.dim() != 2 or sentence_mask.shape != sentence_embeddings.shape[:2]:
            raise ValueError("sentence_mask shape must match [batch, num_sentences]")
        mask = (sentence_mask > 0).to(sentence_embeddings.dtype).unsqueeze(-1)
        masked = sentence_embeddings * mask
        counts = mask.sum(dim=1).clamp_min(1.0)
        return masked.sum(dim=1) / counts

    if sentence_mask.dim() != 1 or sentence_mask.shape[0] != sentence_embeddings.shape[0]:
        raise ValueError("sentence_mask shape must match [num_sentences]")
    mask = (sentence_mask > 0).to(sentence_embeddings.dtype).unsqueeze(-1)
    masked = sentence_embeddings * mask
    counts = mask.sum(dim=0).clamp_min(1.0)
    return masked.sum(dim=0) / counts


def cosine_similarity_matrix(
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    validate_embedding_tensor(left_embeddings, "left_embeddings", expected_dim=2)
    validate_embedding_tensor(right_embeddings, "right_embeddings", expected_dim=2)
    if left_embeddings.shape[1] != right_embeddings.shape[1]:
        raise ValueError("hidden dimensions must match")
    left_norm = l2_normalize_embeddings(left_embeddings, dim=1, eps=eps)
    right_norm = l2_normalize_embeddings(right_embeddings, dim=1, eps=eps)
    return left_norm @ right_norm.transpose(0, 1)


def pairwise_cosine_similarity(
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    validate_embedding_tensor(left_embeddings, "left_embeddings", expected_dim=2)
    validate_embedding_tensor(right_embeddings, "right_embeddings", expected_dim=2)
    if left_embeddings.shape != right_embeddings.shape:
        raise ValueError("left_embeddings and right_embeddings must have the same shape")
    left_norm = l2_normalize_embeddings(left_embeddings, dim=1, eps=eps)
    right_norm = l2_normalize_embeddings(right_embeddings, dim=1, eps=eps)
    return (left_norm * right_norm).sum(dim=1)
