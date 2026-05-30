from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_dict(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_non_empty_str_list(value: list[str], field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field_name} must be a non-empty list of strings")


@dataclass
class EncodedDocument:
    doc_id: str
    chunk_ids: list[str]
    embeddings: torch.Tensor
    attentions: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.doc_id, "doc_id")
        _require_non_empty_str_list(self.chunk_ids, "chunk_ids")
        if not isinstance(self.embeddings, torch.Tensor):
            raise ValueError("embeddings must be a torch.Tensor")
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-dimensional")
        if self.embeddings.shape[0] != len(self.chunk_ids):
            raise ValueError("embeddings.shape[0] must match chunk_ids length")
        if self.attentions is not None:
            if not isinstance(self.attentions, torch.Tensor):
                raise ValueError("attentions must be a torch.Tensor")
            if self.attentions.ndim != 4:
                raise ValueError("attentions must be 4-dimensional")
            if self.attentions.shape[2] != len(self.chunk_ids):
                raise ValueError("attentions shape must match chunk_ids length")
            if self.attentions.shape[3] != len(self.chunk_ids):
                raise ValueError("attentions shape must match chunk_ids length")
        _require_dict(self.metadata, "metadata")


def move_encoded_document(
    encoded: EncodedDocument,
    device: str | torch.device,
) -> EncodedDocument:
    embeddings = encoded.embeddings.to(device)
    attentions = encoded.attentions.to(device) if encoded.attentions is not None else None
    return EncodedDocument(
        doc_id=encoded.doc_id,
        chunk_ids=list(encoded.chunk_ids),
        embeddings=embeddings,
        attentions=attentions,
        metadata=dict(encoded.metadata),
    )
