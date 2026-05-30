from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sproutrag.training.embedding import cosine_similarity_matrix, pairwise_cosine_similarity, validate_embedding_tensor
from sproutrag.training.outputs import SproutRAGTrainingOutput, JointLossOutput


def validate_temperature(temperature: float) -> None:
    if not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")
    if not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite")
    if float(temperature) <= 0:
        raise ValueError("temperature must be > 0")


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of: mean, sum, none")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_finite_eps(eps: float) -> None:
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)):
        raise ValueError("eps must be a finite number")
    if float(eps) <= 0:
        raise ValueError("eps must be > 0")


def _validate_empty_policy(empty_policy: str) -> None:
    if empty_policy not in {"zero", "error"}:
        raise ValueError("empty_policy must be one of: zero, error")


def validate_attention_lambda(attention_lambda: float) -> None:
    if not isinstance(attention_lambda, (int, float)):
        raise ValueError("attention_lambda must be a number")
    if not math.isfinite(float(attention_lambda)):
        raise ValueError("attention_lambda must be finite")
    if float(attention_lambda) < 0:
        raise ValueError("attention_lambda must be >= 0")


def validate_support_pairs_batch(
    support_pairs: list[list[tuple[int, int]]] | list[list[list[int]]],
    batch_size: int,
) -> None:
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(support_pairs, list):
        raise ValueError("support_pairs must be a list")
    if len(support_pairs) != batch_size:
        raise ValueError("support_pairs length must match batch_size")
    for item in support_pairs:
        if not isinstance(item, list):
            raise ValueError("support_pairs must contain lists")


def contrastive_retrieval_loss(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor | None = None,
    temperature: float = 0.05,
    reduction: str = "mean",
) -> torch.Tensor:
    validate_embedding_tensor(query_embeddings, "query_embeddings", expected_dim=2)
    validate_embedding_tensor(positive_embeddings, "positive_embeddings", expected_dim=2)
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError("query_embeddings and positive_embeddings must have the same shape")
    if negative_embeddings is not None:
        validate_embedding_tensor(negative_embeddings, "negative_embeddings", expected_dim=3)
        if negative_embeddings.shape[0] != query_embeddings.shape[0]:
            raise ValueError("negative_embeddings batch size must match query_embeddings")
        if negative_embeddings.shape[2] != query_embeddings.shape[1]:
            raise ValueError("negative_embeddings hidden dim must match query_embeddings")
        if negative_embeddings.shape[1] < 1:
            raise ValueError("negative_embeddings must have at least one negative")

    validate_temperature(temperature)
    _validate_reduction(reduction)

    pos_scores = pairwise_cosine_similarity(query_embeddings, positive_embeddings).unsqueeze(1)
    if negative_embeddings is None:
        logits = pos_scores / temperature
    else:
        batch_size, num_neg, hidden_dim = negative_embeddings.shape
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, num_neg, -1)
        neg_scores = pairwise_cosine_similarity(
            query_expanded.reshape(-1, hidden_dim),
            negative_embeddings.reshape(-1, hidden_dim),
        ).view(batch_size, num_neg)
        logits = torch.cat([pos_scores, neg_scores], dim=1) / temperature

    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels, reduction=reduction)


def in_batch_contrastive_retrieval_loss(
    query_embeddings: torch.Tensor,
    passage_embeddings: torch.Tensor,
    temperature: float = 0.05,
    reduction: str = "mean",
) -> torch.Tensor:
    validate_embedding_tensor(query_embeddings, "query_embeddings", expected_dim=2)
    validate_embedding_tensor(passage_embeddings, "passage_embeddings", expected_dim=2)
    if query_embeddings.shape != passage_embeddings.shape:
        raise ValueError("query_embeddings and passage_embeddings must have the same shape")
    validate_temperature(temperature)
    _validate_reduction(reduction)

    logits = cosine_similarity_matrix(query_embeddings, passage_embeddings) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels, reduction=reduction)


class ContrastiveRetrievalLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.05,
        use_in_batch_negatives: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        validate_temperature(temperature)
        if not isinstance(use_in_batch_negatives, bool):
            raise ValueError("use_in_batch_negatives must be a boolean")
        _validate_reduction(reduction)
        self.temperature = float(temperature)
        self.use_in_batch_negatives = use_in_batch_negatives
        self.reduction = reduction

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_in_batch_negatives:
            return in_batch_contrastive_retrieval_loss(
                query_embeddings=query_embeddings,
                passage_embeddings=positive_embeddings,
                temperature=self.temperature,
                reduction=self.reduction,
            )
        return contrastive_retrieval_loss(
            query_embeddings=query_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            temperature=self.temperature,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, "
            f"use_in_batch_negatives={self.use_in_batch_negatives}, "
            f"reduction={self.reduction}"
        )


def validate_support_pairs(
    support_pairs: list[tuple[int, int]] | list[list[int]],
    num_sentences: int,
    allow_self_pairs: bool = True,
) -> list[tuple[int, int]]:
    _require_positive_int(num_sentences, "num_sentences")
    _require_bool(allow_self_pairs, "allow_self_pairs")
    if not isinstance(support_pairs, list):
        raise ValueError("support_pairs must be a list")
    if not support_pairs:
        return []

    normalized: list[tuple[int, int]] = []
    for pair in support_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("support_pairs must contain pairs of two integers")
        i, j = pair
        if not isinstance(i, int) or not isinstance(j, int):
            raise ValueError("support_pairs must contain integers")
        if i < 0 or j < 0:
            raise ValueError("support_pairs indices must be >= 0")
        if i >= num_sentences or j >= num_sentences:
            raise ValueError("support_pairs indices must be < num_sentences")
        if not allow_self_pairs and i == j:
            raise ValueError("support_pairs must not contain self-pairs")
        normalized.append((int(i), int(j)))
    return normalized


def mutual_attention_from_aggregated(
    aggregated_attention: torch.Tensor,
    eps: float = 1e-12,
    clamp: bool = True,
) -> torch.Tensor:
    if not isinstance(aggregated_attention, torch.Tensor):
        raise ValueError("aggregated_attention must be a torch.Tensor")
    if aggregated_attention.dim() != 2:
        raise ValueError("aggregated_attention must be 2-dimensional")
    if aggregated_attention.shape[0] != aggregated_attention.shape[1]:
        raise ValueError("aggregated_attention must be square")
    if not torch.isfinite(aggregated_attention).all():
        raise ValueError("aggregated_attention must contain finite values")
    _require_finite_eps(eps)

    mutual = (aggregated_attention + aggregated_attention.transpose(0, 1)) / 2.0
    if clamp:
        mutual = mutual.clamp(min=float(eps), max=1.0)
    return mutual


def attention_structure_loss(
    aggregated_attention: torch.Tensor,
    support_pairs: list[tuple[int, int]] | list[list[int]],
    eps: float = 1e-12,
    reduction: str = "mean",
    allow_self_pairs: bool = True,
    empty_policy: str = "zero",
) -> torch.Tensor:
    if not isinstance(aggregated_attention, torch.Tensor):
        raise ValueError("aggregated_attention must be a torch.Tensor")
    if aggregated_attention.dim() != 2:
        raise ValueError("aggregated_attention must be 2-dimensional")
    if aggregated_attention.shape[0] != aggregated_attention.shape[1]:
        raise ValueError("aggregated_attention must be square")
    if not torch.isfinite(aggregated_attention).all():
        raise ValueError("aggregated_attention must contain finite values")
    _require_bool(allow_self_pairs, "allow_self_pairs")
    _require_finite_eps(eps)
    _validate_reduction(reduction)
    _validate_empty_policy(empty_policy)

    num_sentences = aggregated_attention.shape[0]
    pairs = validate_support_pairs(support_pairs, num_sentences, allow_self_pairs)
    if not pairs:
        if empty_policy == "error":
            raise ValueError("support_pairs is empty")
        return torch.zeros((), device=aggregated_attention.device, dtype=aggregated_attention.dtype)

    mutual = mutual_attention_from_aggregated(aggregated_attention, eps=eps, clamp=True)
    indices = torch.tensor(pairs, device=aggregated_attention.device, dtype=torch.long)
    values = mutual[indices[:, 0], indices[:, 1]]
    pair_losses = -torch.log(values)

    if reduction == "mean":
        return pair_losses.mean()
    if reduction == "sum":
        return pair_losses.sum()
    return pair_losses


def batched_attention_structure_loss(
    aggregated_attention: torch.Tensor,
    batch_support_pairs: list[list[tuple[int, int]]] | list[list[list[int]]],
    sentence_mask: torch.Tensor | None = None,
    eps: float = 1e-12,
    reduction: str = "mean",
    example_reduction: str = "mean",
    allow_self_pairs: bool = True,
    empty_policy: str = "zero",
) -> torch.Tensor:
    if not isinstance(aggregated_attention, torch.Tensor):
        raise ValueError("aggregated_attention must be a torch.Tensor")
    if aggregated_attention.dim() != 3:
        raise ValueError("aggregated_attention must be 3-dimensional")
    if aggregated_attention.shape[1] != aggregated_attention.shape[2]:
        raise ValueError("aggregated_attention must be square on last dimensions")
    if not torch.isfinite(aggregated_attention).all():
        raise ValueError("aggregated_attention must contain finite values")
    if not isinstance(batch_support_pairs, list):
        raise ValueError("batch_support_pairs must be a list")

    _require_bool(allow_self_pairs, "allow_self_pairs")
    _require_finite_eps(eps)
    _validate_reduction(reduction)
    _validate_reduction(example_reduction)
    _validate_empty_policy(empty_policy)

    batch_size = aggregated_attention.shape[0]
    if len(batch_support_pairs) != batch_size:
        raise ValueError("batch_support_pairs length must match batch_size")

    if sentence_mask is not None:
        if not isinstance(sentence_mask, torch.Tensor):
            raise ValueError("sentence_mask must be a torch.Tensor")
        if sentence_mask.shape != aggregated_attention.shape[:2]:
            raise ValueError("sentence_mask shape must match [batch_size, num_sentences]")
        if not torch.isfinite(sentence_mask).all():
            raise ValueError("sentence_mask must contain finite values")

    example_losses: list[torch.Tensor] = []
    for idx in range(batch_size):
        if sentence_mask is None:
            valid_sentences = aggregated_attention.shape[1]
        else:
            valid_sentences = int((sentence_mask[idx] > 0).sum().item())
        pairs = validate_support_pairs(
            batch_support_pairs[idx],
            valid_sentences,
            allow_self_pairs,
        )
        loss = attention_structure_loss(
            aggregated_attention[idx],
            pairs,
            eps=eps,
            reduction=reduction,
            allow_self_pairs=allow_self_pairs,
            empty_policy=empty_policy,
        )
        if reduction == "none":
            if loss.numel() == 0:
                loss = torch.zeros((), device=aggregated_attention.device, dtype=aggregated_attention.dtype)
            else:
                loss = loss.mean()
        example_losses.append(loss)

    losses = torch.stack(example_losses)
    if example_reduction == "mean":
        return losses.mean()
    if example_reduction == "sum":
        return losses.sum()
    return losses


class AttentionStructureLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-12,
        reduction: str = "mean",
        example_reduction: str = "mean",
        allow_self_pairs: bool = True,
        empty_policy: str = "zero",
    ) -> None:
        super().__init__()
        _require_finite_eps(eps)
        _validate_reduction(reduction)
        _validate_reduction(example_reduction)
        _require_bool(allow_self_pairs, "allow_self_pairs")
        _validate_empty_policy(empty_policy)
        self.eps = float(eps)
        self.reduction = reduction
        self.example_reduction = example_reduction
        self.allow_self_pairs = allow_self_pairs
        self.empty_policy = empty_policy

    def forward(
        self,
        aggregated_attention: torch.Tensor,
        support_pairs: list[tuple[int, int]]
        | list[list[int]]
        | list[list[tuple[int, int]]]
        | list[list[list[int]]],
        sentence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if aggregated_attention.dim() == 2:
            if sentence_mask is not None:
                raise ValueError("sentence_mask must be None for 2D aggregated_attention")
            return attention_structure_loss(
                aggregated_attention,
                support_pairs,
                eps=self.eps,
                reduction=self.reduction,
                allow_self_pairs=self.allow_self_pairs,
                empty_policy=self.empty_policy,
            )
        if aggregated_attention.dim() == 3:
            return batched_attention_structure_loss(
                aggregated_attention,
                support_pairs,
                sentence_mask=sentence_mask,
                eps=self.eps,
                reduction=self.reduction,
                example_reduction=self.example_reduction,
                allow_self_pairs=self.allow_self_pairs,
                empty_policy=self.empty_policy,
            )
        raise ValueError("aggregated_attention must be 2D or 3D")

    def extra_repr(self) -> str:
        return (
            f"eps={self.eps}, "
            f"reduction={self.reduction}, "
            f"example_reduction={self.example_reduction}, "
            f"allow_self_pairs={self.allow_self_pairs}, "
            f"empty_policy={self.empty_policy}"
        )


class JointSproutRAGLoss(nn.Module):
    def __init__(
        self,
        retrieval_temperature: float = 0.05,
        attention_lambda: float = 0.1,
        use_in_batch_negatives: bool = False,
        retrieval_reduction: str = "mean",
        attention_reduction: str = "mean",
        attention_example_reduction: str = "mean",
        allow_self_pairs: bool = True,
        empty_attention_policy: str = "zero",
    ) -> None:
        super().__init__()
        validate_attention_lambda(attention_lambda)
        if retrieval_reduction == "none":
            raise ValueError("retrieval_reduction must not be none")
        if attention_example_reduction == "none":
            raise ValueError("attention_example_reduction must not be none")

        self.retrieval_loss_fn = ContrastiveRetrievalLoss(
            temperature=retrieval_temperature,
            use_in_batch_negatives=use_in_batch_negatives,
            reduction=retrieval_reduction,
        )
        self.attention_loss_fn = AttentionStructureLoss(
            reduction=attention_reduction,
            example_reduction=attention_example_reduction,
            allow_self_pairs=allow_self_pairs,
            empty_policy=empty_attention_policy,
        )

        self.retrieval_temperature = float(retrieval_temperature)
        self.attention_lambda = float(attention_lambda)
        self.use_in_batch_negatives = use_in_batch_negatives
        self.retrieval_reduction = retrieval_reduction
        self.attention_reduction = attention_reduction
        self.attention_example_reduction = attention_example_reduction
        self.allow_self_pairs = allow_self_pairs
        self.empty_attention_policy = empty_attention_policy

    def forward(
        self,
        model_output: SproutRAGTrainingOutput,
        support_pairs: list[list[tuple[int, int]]] | list[list[list[int]]] | None = None,
    ) -> JointLossOutput:
        if not isinstance(model_output, SproutRAGTrainingOutput):
            raise ValueError("model_output must be a SproutRAGTrainingOutput")

        retrieval_loss = self.retrieval_loss_fn(
            query_embeddings=model_output.query_embeddings,
            positive_embeddings=model_output.positive_embeddings,
            negative_embeddings=model_output.negative_embeddings,
        )

        if self.attention_lambda == 0:
            attention_loss = torch.zeros((), device=retrieval_loss.device, dtype=retrieval_loss.dtype)
        else:
            if model_output.positive_aggregated_attention is None:
                raise ValueError("positive_aggregated_attention is required when attention_lambda > 0")
            if support_pairs is None:
                raise ValueError("support_pairs is required when attention_lambda > 0")
            validate_support_pairs_batch(support_pairs, model_output.batch_size)

            attention = model_output.positive_aggregated_attention
            if attention.dim() == 2:
                if model_output.batch_size != 1:
                    raise ValueError("2D attention requires batch_size == 1")
                if len(support_pairs) != 1:
                    raise ValueError("support_pairs must contain one list for 2D attention")
                attention_loss = self.attention_loss_fn(
                    aggregated_attention=attention,
                    support_pairs=support_pairs[0],
                )
            elif attention.dim() == 3:
                attention_loss = self.attention_loss_fn(
                    aggregated_attention=attention,
                    support_pairs=support_pairs,
                    sentence_mask=model_output.sentence_mask,
                )
            else:
                raise ValueError("positive_aggregated_attention must be 2D or 3D")

        loss = retrieval_loss + self.attention_lambda * attention_loss
        metadata = {
            "retrieval_temperature": self.retrieval_temperature,
            "attention_lambda": self.attention_lambda,
            "use_in_batch_negatives": self.use_in_batch_negatives,
            "retrieval_reduction": self.retrieval_reduction,
            "attention_reduction": self.attention_reduction,
            "attention_example_reduction": self.attention_example_reduction,
            "allow_self_pairs": self.allow_self_pairs,
            "empty_attention_policy": self.empty_attention_policy,
            "has_attention": model_output.has_attention,
            "num_negatives": model_output.num_negatives,
            "batch_size": model_output.batch_size,
        }
        return JointLossOutput(
            loss=loss,
            retrieval_loss=retrieval_loss,
            attention_loss=attention_loss,
            metadata=metadata,
        )

    def extra_repr(self) -> str:
        return (
            f"retrieval_temperature={self.retrieval_temperature}, "
            f"attention_lambda={self.attention_lambda}, "
            f"use_in_batch_negatives={self.use_in_batch_negatives}, "
            f"retrieval_reduction={self.retrieval_reduction}, "
            f"attention_reduction={self.attention_reduction}, "
            f"attention_example_reduction={self.attention_example_reduction}"
        )
