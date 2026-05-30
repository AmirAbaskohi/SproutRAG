from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sproutrag.data.schema import RawDocument
from sproutrag.data.preprocessing import chunk_sentences
from sproutrag.encoding.schema import EncodedDocument
from sproutrag.training.embedding import l2_normalize_embeddings, validate_embedding_tensor
from sproutrag.training.outputs import SproutRAGTrainingOutput
from sproutrag.training.schema import TrainingBatch


def validate_encoder_for_training(encoder: Any) -> None:
    if encoder is None:
        raise ValueError("encoder must not be None")
    if not hasattr(encoder, "encode_query") or not callable(getattr(encoder, "encode_query")):
        raise ValueError("encoder must implement encode_query")
    if not hasattr(encoder, "encode_document") or not callable(getattr(encoder, "encode_document")):
        raise ValueError("encoder must implement encode_document")


def validate_training_aggregator(aggregator: Any) -> None:
    if aggregator is None:
        raise ValueError("aggregator must not be None")
    if not callable(aggregator):
        raise ValueError("aggregator must be callable")


def passage_to_document(
    passage: str,
    doc_id: str,
    title: str | None = None,
) -> RawDocument:
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("passage must be a non-empty string")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("doc_id must be a non-empty string")
    return RawDocument(doc_id=doc_id, text=passage, title=title, metadata={})


def filter_support_pairs_for_encoded_chunks(
    support_pairs: list[tuple[int, int]] | list[list[int]],
    original_chunk_ids: list[str],
    encoded_chunk_ids: list[str],
) -> list[tuple[int, int]]:
    if not isinstance(original_chunk_ids, list) or not original_chunk_ids:
        raise ValueError("original_chunk_ids must be a non-empty list of strings")
    if not all(isinstance(value, str) and value for value in original_chunk_ids):
        raise ValueError("original_chunk_ids must be a non-empty list of strings")
    if not isinstance(encoded_chunk_ids, list) or not encoded_chunk_ids:
        raise ValueError("encoded_chunk_ids must be a non-empty list of strings")
    if not all(isinstance(value, str) and value for value in encoded_chunk_ids):
        raise ValueError("encoded_chunk_ids must be a non-empty list of strings")
    original_set = set(original_chunk_ids)
    if any(chunk_id not in original_set for chunk_id in encoded_chunk_ids):
        raise ValueError("encoded_chunk_ids must be a subset of original_chunk_ids")
    if not isinstance(support_pairs, list):
        raise ValueError("support_pairs must be a list")

    original_index_by_id = {chunk_id: idx for idx, chunk_id in enumerate(original_chunk_ids)}
    encoded_index_by_id = {chunk_id: idx for idx, chunk_id in enumerate(encoded_chunk_ids)}
    valid_original_indices = {original_index_by_id[chunk_id] for chunk_id in encoded_chunk_ids}

    remapped: list[tuple[int, int]] = []
    for pair in support_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("support_pairs must contain pairs of two integers")
        i, j = pair
        if not isinstance(i, int) or not isinstance(j, int):
            raise ValueError("support_pairs must contain integers")
        if i < 0 or j < 0:
            raise ValueError("support_pairs indices must be >= 0")
        if i >= len(original_chunk_ids) or j >= len(original_chunk_ids):
            raise ValueError("support_pairs indices must be < len(original_chunk_ids)")
        if i not in valid_original_indices or j not in valid_original_indices:
            continue
        original_i = original_chunk_ids[i]
        original_j = original_chunk_ids[j]
        remapped.append((encoded_index_by_id[original_i], encoded_index_by_id[original_j]))
    return remapped


def pad_attention_matrices(
    matrices: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(matrices, list) or not matrices:
        raise ValueError("matrices must be a non-empty list of torch.Tensor")
    dtype = None
    device = None
    sizes: list[int] = []
    padded: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for idx, matrix in enumerate(matrices):
        if not isinstance(matrix, torch.Tensor):
            raise ValueError("matrices must contain torch.Tensor")
        if matrix.dim() != 2:
            raise ValueError("attention matrix must be 2-dimensional")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("attention matrix must be square")
        if matrix.shape[0] < 1:
            raise ValueError("attention matrix must be non-empty")
        if not torch.isfinite(matrix).all():
            raise ValueError("attention matrix must contain finite values")
        if dtype is None:
            dtype = matrix.dtype
            device = matrix.device
        if matrix.dtype != dtype or matrix.device != device:
            raise ValueError("all attention matrices must share dtype and device")
        sizes.append(matrix.shape[0])

    max_n = max(sizes)
    for matrix, size in zip(matrices, sizes):
        pad_amount = max_n - size
        if pad_amount == 0:
            padded_matrix = matrix
        else:
            padded_matrix = torch.nn.functional.pad(
                matrix,
                (0, pad_amount, 0, pad_amount),
                mode="constant",
                value=pad_value,
            )
        padded.append(padded_matrix)
        mask = torch.zeros(max_n, device=device, dtype=torch.float32)
        mask[:size] = 1.0
        masks.append(mask)

    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


def stack_embeddings(
    embeddings: list[torch.Tensor],
    name: str,
) -> torch.Tensor:
    if not isinstance(embeddings, list) or not embeddings:
        raise ValueError(f"{name} must be a non-empty list of torch.Tensor")
    dtype = None
    device = None
    shape = None
    for tensor in embeddings:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must contain torch.Tensor")
        if tensor.dim() != 1:
            raise ValueError(f"{name} must contain 1D tensors")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite values")
        if dtype is None:
            dtype = tensor.dtype
            device = tensor.device
            shape = tensor.shape
        if tensor.dtype != dtype or tensor.device != device:
            raise ValueError(f"{name} must share dtype and device")
        if tensor.shape != shape:
            raise ValueError(f"{name} must have consistent shapes")
    return torch.stack(embeddings, dim=0)


def stack_negative_embeddings(
    negative_embeddings: list[list[torch.Tensor]],
) -> torch.Tensor:
    if not isinstance(negative_embeddings, list) or not negative_embeddings:
        raise ValueError("negative_embeddings must be a non-empty list")
    num_negatives = None
    stacked: list[torch.Tensor] = []
    for example in negative_embeddings:
        if not isinstance(example, list) or not example:
            raise ValueError("negative_embeddings must contain non-empty lists")
        if num_negatives is None:
            num_negatives = len(example)
        if len(example) != num_negatives:
            raise ValueError("all examples must have the same number of negatives")
        stacked.append(stack_embeddings(example, "negative_embeddings"))
    return torch.stack(stacked, dim=0)


class SproutRAGTrainingModel(nn.Module):
    def __init__(
        self,
        encoder: Any,
        aggregator: Any,
        max_sentences_per_chunk: int = 2,
        normalize_passage_embeddings: bool = True,
    ) -> None:
        super().__init__()
        validate_encoder_for_training(encoder)
        validate_training_aggregator(aggregator)
        if not isinstance(max_sentences_per_chunk, int) or max_sentences_per_chunk <= 0:
            raise ValueError("max_sentences_per_chunk must be a positive integer")
        if not isinstance(normalize_passage_embeddings, bool):
            raise ValueError("normalize_passage_embeddings must be a boolean")

        self.encoder = encoder
        if isinstance(aggregator, nn.Module):
            self.aggregator = aggregator
        else:
            self.aggregator = aggregator
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.normalize_passage_embeddings = normalize_passage_embeddings

    def encode_query_batch(self, queries: list[str]) -> torch.Tensor:
        if not isinstance(queries, list) or not queries:
            raise ValueError("queries must be a non-empty list of strings")
        embeddings: list[torch.Tensor] = []
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("queries must contain non-empty strings")
            encoded = self.encoder.encode_query(query)
            validate_embedding_tensor(encoded, "query_embedding", expected_dim=1)
            embeddings.append(encoded)
        return stack_embeddings(embeddings, "query_embeddings")

    def encode_passage(
        self,
        passage: str,
        doc_id: str,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str], list[str]]:
        document = passage_to_document(passage, doc_id)
        chunks = chunk_sentences(document, max_sentences_per_chunk=self.max_sentences_per_chunk)
        original_chunk_ids = [chunk.chunk_id for chunk in chunks]
        encoded = self.encoder.encode_document(chunks, return_attentions=return_attention)
        if not isinstance(encoded, EncodedDocument):
            raise ValueError("encoder must return EncodedDocument")
        validate_embedding_tensor(encoded.embeddings, "encoded.embeddings", expected_dim=2)
        if not encoded.chunk_ids:
            raise ValueError("encoded.chunk_ids must be non-empty")

        passage_embedding = encoded.embeddings.mean(dim=0)
        if self.normalize_passage_embeddings:
            passage_embedding = l2_normalize_embeddings(passage_embedding, dim=0)

        aggregated_attention: torch.Tensor | None = None
        if return_attention:
            if encoded.attentions is None:
                raise ValueError("encoded.attentions must not be None when return_attention=True")
            aggregated_attention = self.aggregator(encoded.attentions)
            if not isinstance(aggregated_attention, torch.Tensor):
                raise ValueError("aggregator must return a torch.Tensor")
            if aggregated_attention.dim() != 2:
                raise ValueError("aggregated attention must be 2-dimensional")
            if aggregated_attention.shape[0] != aggregated_attention.shape[1]:
                raise ValueError("aggregated attention must be square")
            if aggregated_attention.shape[0] != encoded.embeddings.shape[0]:
                raise ValueError("aggregated attention shape must match encoded chunks")
            if not torch.isfinite(aggregated_attention).all():
                raise ValueError("aggregated attention must contain finite values")

        return passage_embedding, aggregated_attention, original_chunk_ids, list(encoded.chunk_ids)

    def encode_positive_batch(
        self,
        positive_passages: list[str],
        support_sentence_pairs: list[list[tuple[int, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
        if not isinstance(positive_passages, list) or not positive_passages:
            raise ValueError("positive_passages must be a non-empty list of strings")
        if not isinstance(support_sentence_pairs, list) or len(support_sentence_pairs) != len(positive_passages):
            raise ValueError("support_sentence_pairs must match positive_passages length")

        embeddings: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        remapped: list[list[tuple[int, int]]] = []
        for idx, passage in enumerate(positive_passages):
            if not isinstance(passage, str) or not passage.strip():
                raise ValueError("positive_passages must contain non-empty strings")
            doc_id = f"positive::{idx}"
            passage_embedding, aggregated_attention, original_chunk_ids, encoded_chunk_ids = self.encode_passage(
                passage,
                doc_id,
                return_attention=True,
            )
            if aggregated_attention is None:
                raise ValueError("aggregated_attention must not be None for positive passages")
            embeddings.append(passage_embedding)
            attentions.append(aggregated_attention)
            remapped_pairs = filter_support_pairs_for_encoded_chunks(
                support_sentence_pairs[idx],
                original_chunk_ids,
                encoded_chunk_ids,
            )
            remapped.append(remapped_pairs)

        positive_embeddings = stack_embeddings(embeddings, "positive_embeddings")
        padded_attention, sentence_mask = pad_attention_matrices(attentions)
        return positive_embeddings, padded_attention, sentence_mask, remapped

    def encode_negative_batch(
        self,
        negative_passages: list[list[str]],
    ) -> torch.Tensor:
        if not isinstance(negative_passages, list) or not negative_passages:
            raise ValueError("negative_passages must be a non-empty list")

        all_embeddings: list[list[torch.Tensor]] = []
        num_negatives = None
        for batch_index, passages in enumerate(negative_passages):
            if not isinstance(passages, list) or not passages:
                raise ValueError("negative_passages must contain non-empty lists")
            if num_negatives is None:
                num_negatives = len(passages)
            if len(passages) != num_negatives:
                raise ValueError("all examples must have the same number of negatives")
            example_embeddings: list[torch.Tensor] = []
            for neg_index, passage in enumerate(passages):
                if not isinstance(passage, str) or not passage.strip():
                    raise ValueError("negative_passages must contain non-empty strings")
                doc_id = f"negative::{batch_index}::{neg_index}"
                embedding, _, _, _ = self.encode_passage(passage, doc_id, return_attention=False)
                example_embeddings.append(embedding)
            all_embeddings.append(example_embeddings)
        return stack_negative_embeddings(all_embeddings)

    def forward(
        self,
        batch: TrainingBatch,
    ) -> tuple[SproutRAGTrainingOutput, list[list[tuple[int, int]]]]:
        if not isinstance(batch, TrainingBatch):
            raise ValueError("batch must be a TrainingBatch")

        query_embeddings = self.encode_query_batch(batch.queries)
        positive_embeddings, positive_attention, sentence_mask, remapped_support_pairs = self.encode_positive_batch(
            batch.positive_passages,
            batch.support_sentence_pairs,
        )
        negative_embeddings = self.encode_negative_batch(batch.negative_passages)

        model_output = SproutRAGTrainingOutput(
            query_embeddings=query_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            positive_aggregated_attention=positive_attention,
            sentence_mask=sentence_mask,
            metadata={
                "num_examples": len(batch.example_ids),
                "example_ids": list(batch.example_ids),
                "max_sentences_per_chunk": self.max_sentences_per_chunk,
                "normalize_passage_embeddings": self.normalize_passage_embeddings,
            },
        )
        return model_output, remapped_support_pairs
