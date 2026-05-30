from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def mean_pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(
        attention_mask, torch.Tensor
    ):
        raise ValueError("hidden_states and attention_mask must be torch.Tensors")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be 3-dimensional")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2-dimensional")
    if hidden_states.shape[0] != attention_mask.shape[0]:
        raise ValueError("batch size mismatch between hidden_states and attention_mask")
    if hidden_states.shape[1] != attention_mask.shape[1]:
        raise ValueError("seq_len mismatch between hidden_states and attention_mask")

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked = hidden_states * mask
    counts = mask.sum(dim=1).clamp_min(1.0)
    pooled = masked.sum(dim=1) / counts
    return pooled


def mean_pool_tokens_by_span(
    token_embeddings: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    if not isinstance(token_embeddings, torch.Tensor):
        raise ValueError("token_embeddings must be a torch.Tensor")
    if token_embeddings.ndim != 2:
        raise ValueError("token_embeddings must be 2-dimensional")
    if not isinstance(spans, list) or not spans:
        raise ValueError("spans must be a non-empty list")

    seq_len = token_embeddings.shape[0]
    span_embeddings = []
    for start, end in spans:
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("span indices must be integers")
        if start < 0 or end <= start or end > seq_len:
            raise ValueError("invalid span boundaries")
        span_embeddings.append(token_embeddings[start:end].mean(dim=0))
    return torch.stack(span_embeddings, dim=0)


def pool_attentions_by_span(
    attentions: tuple[torch.Tensor, ...] | list[torch.Tensor],
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    if not isinstance(attentions, (list, tuple)) or not attentions:
        raise ValueError("attentions must be a non-empty list or tuple")
    if not isinstance(spans, list) or not spans:
        raise ValueError("spans must be a non-empty list")

    first = attentions[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("each attention tensor must be 4-dimensional")
    batch_size, num_heads, seq_len, seq_len_b = first.shape
    if seq_len != seq_len_b:
        raise ValueError("attention tensors must be square on seq_len")
    if batch_size != 1:
        raise ValueError("only batch_size == 1 is supported")

    for layer in attentions:
        if not isinstance(layer, torch.Tensor) or layer.ndim != 4:
            raise ValueError("each attention tensor must be 4-dimensional")
        if layer.shape[0] != batch_size:
            raise ValueError("all layers must share batch size")
        if layer.shape[1] != num_heads or layer.shape[2] != seq_len:
            raise ValueError("all layers must share num_heads and seq_len")

    for start, end in spans:
        if start < 0 or end <= start or end > seq_len:
            raise ValueError("invalid span boundaries")

    num_layers = len(attentions)
    num_spans = len(spans)
    output = torch.zeros(
        (num_layers, num_heads, num_spans, num_spans),
        device=first.device,
        dtype=first.dtype,
    )

    for layer_index, layer in enumerate(attentions):
        layer_attn = layer[0]
        for head in range(num_heads):
            head_attn = layer_attn[head]
            for i, (src_start, src_end) in enumerate(spans):
                for j, (tgt_start, tgt_end) in enumerate(spans):
                    value = head_attn[src_start:src_end, tgt_start:tgt_end].mean()
                    output[layer_index, head, i, j] = value
    return output


def l2_normalize(
    embeddings: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("embeddings must be a torch.Tensor")
    return F.normalize(embeddings, p=2, dim=dim, eps=eps)
