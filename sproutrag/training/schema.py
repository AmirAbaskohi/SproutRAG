from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        raise ValueError(f"{field_name} must contain non-empty strings")


def _normalize_pairs(pairs: list[tuple[int, int]] | list[list[int]]) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    if not isinstance(pairs, list):
        raise ValueError("support_sentence_pairs must be a list")
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("support_sentence_pairs must contain pairs of two integers")
        i, j = pair
        if not isinstance(i, int) or not isinstance(j, int):
            raise ValueError("support_sentence_pairs must contain integers")
        if i < 0 or j < 0:
            raise ValueError("support_sentence_pairs indices must be >= 0")
        normalized.append((int(i), int(j)))
    return normalized


@dataclass
class TrainingExample:
    example_id: str
    query: str
    positive_passage: str
    negative_passages: list[str]
    support_sentence_pairs: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.example_id, "example_id")
        _require_non_empty_str(self.query, "query")
        _require_non_empty_str(self.positive_passage, "positive_passage")
        _require_non_empty_str_list(self.negative_passages, "negative_passages")
        self.support_sentence_pairs = _normalize_pairs(self.support_sentence_pairs)
        _require_dict(self.metadata, "metadata")


@dataclass
class TrainingBatch:
    example_ids: list[str]
    queries: list[str]
    positive_passages: list[str]
    negative_passages: list[list[str]]
    support_sentence_pairs: list[list[tuple[int, int]]]
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_non_empty_str_list(self.example_ids, "example_ids")
        _require_non_empty_str_list(self.queries, "queries")
        _require_non_empty_str_list(self.positive_passages, "positive_passages")
        if not isinstance(self.negative_passages, list):
            raise ValueError("negative_passages must be a list of list of strings")
        if not all(isinstance(items, list) and items for items in self.negative_passages):
            raise ValueError("negative_passages must contain non-empty lists")
        for items in self.negative_passages:
            if not all(isinstance(item, str) and item.strip() for item in items):
                raise ValueError("negative_passages must contain non-empty strings")

        if not isinstance(self.support_sentence_pairs, list):
            raise ValueError("support_sentence_pairs must be a list")
        normalized_pairs: list[list[tuple[int, int]]] = []
        for pair_list in self.support_sentence_pairs:
            normalized_pairs.append(_normalize_pairs(pair_list))
        self.support_sentence_pairs = normalized_pairs

        if not isinstance(self.metadata, list):
            raise ValueError("metadata must be a list of dict")
        if not all(isinstance(item, dict) for item in self.metadata):
            raise ValueError("metadata must be a list of dict")

        length = len(self.example_ids)
        if not (
            len(self.queries)
            == len(self.positive_passages)
            == len(self.negative_passages)
            == len(self.support_sentence_pairs)
            == len(self.metadata)
            == length
        ):
            raise ValueError("all batch fields must have the same length")


def training_example_to_dict(example: TrainingExample) -> dict[str, Any]:
    if not isinstance(example, TrainingExample):
        raise ValueError("example must be a TrainingExample")
    return {
        "example_id": example.example_id,
        "query": example.query,
        "positive_passage": example.positive_passage,
        "negative_passages": list(example.negative_passages),
        "support_sentence_pairs": [list(pair) for pair in example.support_sentence_pairs],
        "metadata": dict(example.metadata),
    }


def training_example_from_dict(data: dict[str, Any]) -> TrainingExample:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    return TrainingExample(
        example_id=data["example_id"],
        query=data["query"],
        positive_passage=data["positive_passage"],
        negative_passages=list(data["negative_passages"]),
        support_sentence_pairs=data.get("support_sentence_pairs", []),
        metadata=data.get("metadata", {}),
    )


def training_batch_from_examples(examples: list[TrainingExample]) -> TrainingBatch:
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a non-empty list of TrainingExample")
    if not all(isinstance(item, TrainingExample) for item in examples):
        raise ValueError("examples must contain TrainingExample instances")
    return TrainingBatch(
        example_ids=[ex.example_id for ex in examples],
        queries=[ex.query for ex in examples],
        positive_passages=[ex.positive_passage for ex in examples],
        negative_passages=[list(ex.negative_passages) for ex in examples],
        support_sentence_pairs=[list(ex.support_sentence_pairs) for ex in examples],
        metadata=[dict(ex.metadata) for ex in examples],
    )
