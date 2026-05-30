from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _require_tensor(value: torch.Tensor, field_name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{field_name} must be a torch.Tensor")


def _require_finite(tensor: torch.Tensor, field_name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{field_name} must contain finite values")


def _require_dict(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_scalar(tensor: torch.Tensor, field_name: str) -> None:
    if tensor.dim() != 0:
        raise ValueError(f"{field_name} must be a scalar tensor")


def _require_square_matrix(tensor: torch.Tensor, field_name: str) -> None:
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(f"{field_name} must be square on the last two dimensions")


@dataclass
class SproutRAGTrainingOutput:
    query_embeddings: torch.Tensor
    positive_embeddings: torch.Tensor
    negative_embeddings: torch.Tensor | None = None
    positive_aggregated_attention: torch.Tensor | None = None
    sentence_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_tensor(self.query_embeddings, "query_embeddings")
        if self.query_embeddings.dim() != 2:
            raise ValueError("query_embeddings must be 2-dimensional")
        _require_finite(self.query_embeddings, "query_embeddings")

        _require_tensor(self.positive_embeddings, "positive_embeddings")
        if self.positive_embeddings.dim() != 2:
            raise ValueError("positive_embeddings must be 2-dimensional")
        if self.positive_embeddings.shape != self.query_embeddings.shape:
            raise ValueError("positive_embeddings must match query_embeddings shape")
        _require_finite(self.positive_embeddings, "positive_embeddings")

        if self.negative_embeddings is not None:
            _require_tensor(self.negative_embeddings, "negative_embeddings")
            if self.negative_embeddings.dim() != 3:
                raise ValueError("negative_embeddings must be 3-dimensional")
            if self.negative_embeddings.shape[0] != self.query_embeddings.shape[0]:
                raise ValueError("negative_embeddings batch size must match query_embeddings")
            if self.negative_embeddings.shape[2] != self.query_embeddings.shape[1]:
                raise ValueError("negative_embeddings hidden dim must match query_embeddings")
            if self.negative_embeddings.shape[1] < 1:
                raise ValueError("negative_embeddings must have at least one negative")
            _require_finite(self.negative_embeddings, "negative_embeddings")

        if self.positive_aggregated_attention is not None:
            _require_tensor(self.positive_aggregated_attention, "positive_aggregated_attention")
            _require_finite(self.positive_aggregated_attention, "positive_aggregated_attention")
            if self.positive_aggregated_attention.dim() == 2:
                if self.query_embeddings.shape[0] != 1:
                    raise ValueError("positive_aggregated_attention can be 2D only when batch_size == 1")
                _require_square_matrix(self.positive_aggregated_attention, "positive_aggregated_attention")
            elif self.positive_aggregated_attention.dim() == 3:
                if self.positive_aggregated_attention.shape[0] != self.query_embeddings.shape[0]:
                    raise ValueError("positive_aggregated_attention batch size must match query_embeddings")
                _require_square_matrix(self.positive_aggregated_attention, "positive_aggregated_attention")
            else:
                raise ValueError("positive_aggregated_attention must be 2D or 3D")

        if self.sentence_mask is not None:
            if self.positive_aggregated_attention is None:
                raise ValueError("sentence_mask requires positive_aggregated_attention")
            if self.positive_aggregated_attention.dim() != 3:
                raise ValueError("sentence_mask requires 3D positive_aggregated_attention")
            _require_tensor(self.sentence_mask, "sentence_mask")
            _require_finite(self.sentence_mask, "sentence_mask")
            if self.sentence_mask.dim() != 2:
                raise ValueError("sentence_mask must be 2-dimensional")
            if self.sentence_mask.shape[0] != self.query_embeddings.shape[0]:
                raise ValueError("sentence_mask batch size must match query_embeddings")
            if self.sentence_mask.shape[1] != self.positive_aggregated_attention.shape[1]:
                raise ValueError("sentence_mask num_sentences must match attention")

        _require_dict(self.metadata, "metadata")

    @property
    def batch_size(self) -> int:
        return int(self.query_embeddings.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.query_embeddings.shape[1])

    @property
    def num_negatives(self) -> int:
        if self.negative_embeddings is None:
            return 0
        return int(self.negative_embeddings.shape[1])

    @property
    def has_attention(self) -> bool:
        return self.positive_aggregated_attention is not None


def move_training_output(
    output: SproutRAGTrainingOutput,
    device: str | torch.device,
) -> SproutRAGTrainingOutput:
    if not isinstance(output, SproutRAGTrainingOutput):
        raise ValueError("output must be a SproutRAGTrainingOutput")

    def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(device)

    return SproutRAGTrainingOutput(
        query_embeddings=_move(output.query_embeddings),
        positive_embeddings=_move(output.positive_embeddings),
        negative_embeddings=_move(output.negative_embeddings),
        positive_aggregated_attention=_move(output.positive_aggregated_attention),
        sentence_mask=_move(output.sentence_mask),
        metadata=output.metadata,
    )


@dataclass
class JointLossOutput:
    loss: torch.Tensor
    retrieval_loss: torch.Tensor
    attention_loss: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_tensor(self.loss, "loss")
        _require_tensor(self.retrieval_loss, "retrieval_loss")
        _require_tensor(self.attention_loss, "attention_loss")
        _require_scalar(self.loss, "loss")
        _require_scalar(self.retrieval_loss, "retrieval_loss")
        _require_scalar(self.attention_loss, "attention_loss")
        _require_finite(self.loss, "loss")
        _require_finite(self.retrieval_loss, "retrieval_loss")
        _require_finite(self.attention_loss, "attention_loss")
        _require_dict(self.metadata, "metadata")


def joint_loss_output_to_dict(
    output: JointLossOutput,
) -> dict[str, Any]:
    if not isinstance(output, JointLossOutput):
        raise ValueError("output must be a JointLossOutput")
    return {
        "loss": float(output.loss.detach().cpu()),
        "retrieval_loss": float(output.retrieval_loss.detach().cpu()),
        "attention_loss": float(output.attention_loss.detach().cpu()),
        "metadata": output.metadata,
    }
