from __future__ import annotations

import torch
from torch import nn


def validate_attention_tensor(
    attentions: torch.Tensor,
    expected_num_layers: int | None = None,
    expected_num_heads: int | None = None,
) -> tuple[int, int, int]:
    if not isinstance(attentions, torch.Tensor):
        raise ValueError("attentions must be a torch.Tensor")
    if attentions.ndim != 4:
        raise ValueError("attentions must be 4-dimensional")
    num_layers, num_heads, num_chunks, num_chunks_b = attentions.shape
    if num_chunks != num_chunks_b:
        raise ValueError("attentions must be square on the last two dimensions")
    if expected_num_layers is not None and num_layers != expected_num_layers:
        raise ValueError("attentions layer count does not match expected_num_layers")
    if expected_num_heads is not None and num_heads != expected_num_heads:
        raise ValueError("attentions head count does not match expected_num_heads")
    return num_layers, num_heads, num_chunks


def uniform_attention_aggregation(
    attentions: torch.Tensor,
    mask_diagonal: bool = True,
) -> torch.Tensor:
    validate_attention_tensor(attentions)
    if not isinstance(mask_diagonal, bool):
        raise ValueError("mask_diagonal must be a boolean")
    aggregated = attentions.mean(dim=(0, 1))
    mutual = (aggregated + aggregated.transpose(0, 1)) / 2.0
    if mask_diagonal:
        diag_mask = torch.eye(mutual.shape[0], device=mutual.device, dtype=mutual.dtype)
        mutual = mutual * (1.0 - diag_mask)
    return mutual


class AttentionAggregator(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        init_strategy: str = "uniform",
        mask_diagonal: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if init_strategy not in {"uniform", "normal", "last_layer"}:
            raise ValueError("init_strategy must be one of: uniform, normal, last_layer")
        if not isinstance(mask_diagonal, bool):
            raise ValueError("mask_diagonal must be a boolean")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.init_strategy = init_strategy
        self.mask_diagonal = mask_diagonal

        alpha = torch.zeros((num_layers, num_heads), dtype=torch.float32)
        if init_strategy == "normal":
            alpha = torch.normal(mean=0.0, std=0.02, size=(num_layers, num_heads))
        elif init_strategy == "last_layer":
            alpha.fill_(-5.0)
            alpha[-1, :] = 5.0
        self.alpha = nn.Parameter(alpha)

    def get_weights(self) -> torch.Tensor:
        flat = self.alpha.reshape(-1)
        weights = torch.softmax(flat, dim=0).reshape(self.num_layers, self.num_heads)
        return weights

    def aggregate(self, attentions: torch.Tensor) -> torch.Tensor:
        validate_attention_tensor(
            attentions,
            expected_num_layers=self.num_layers,
            expected_num_heads=self.num_heads,
        )
        weights = self.get_weights().to(device=attentions.device, dtype=attentions.dtype)
        aggregated = torch.einsum("lh,lhij->ij", weights, attentions)
        return aggregated

    def symmetrize(self, aggregated_attention: torch.Tensor) -> torch.Tensor:
        if not isinstance(aggregated_attention, torch.Tensor):
            raise ValueError("aggregated_attention must be a torch.Tensor")
        if aggregated_attention.ndim != 2:
            raise ValueError("aggregated_attention must be 2-dimensional")
        if aggregated_attention.shape[0] != aggregated_attention.shape[1]:
            raise ValueError("aggregated_attention must be square")
        mutual = (aggregated_attention + aggregated_attention.transpose(0, 1)) / 2.0
        if self.mask_diagonal:
            diag_mask = torch.eye(
                mutual.shape[0], device=mutual.device, dtype=mutual.dtype
            )
            mutual = mutual * (1.0 - diag_mask)
        return mutual

    def forward(self, attentions: torch.Tensor) -> torch.Tensor:
        aggregated = self.aggregate(attentions)
        return self.symmetrize(aggregated)

    def extra_repr(self) -> str:
        return (
            "num_layers={num_layers}, num_heads={num_heads}, init_strategy={init_strategy}, "
            "mask_diagonal={mask_diagonal}"
        ).format(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            init_strategy=self.init_strategy,
            mask_diagonal=self.mask_diagonal,
        )
