from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_dict(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_str_list(value: list[str], field_name: str, allow_empty: bool = False) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{field_name} must be a list of strings")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")


def _require_metric_dict(value: dict[str, float]) -> None:
    if not isinstance(value, dict):
        raise ValueError("metrics must be a dictionary")
    for key, metric in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metrics must have non-empty string keys")
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            raise ValueError("metrics must contain finite numbers")


@dataclass
class RetrievalEvaluationExample:
    example_id: str
    query: str
    relevant_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.example_id, "example_id")
        _require_non_empty_str(self.query, "query")
        _require_str_list(self.relevant_ids, "relevant_ids", allow_empty=False)
        if len(set(self.relevant_ids)) != len(self.relevant_ids):
            raise ValueError("relevant_ids must not contain duplicates")
        _require_dict(self.metadata, "metadata")


@dataclass
class RetrievalEvaluationResult:
    example_id: str
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.example_id, "example_id")
        _require_non_empty_str(self.query, "query")
        _require_str_list(self.retrieved_ids, "retrieved_ids", allow_empty=True)
        _require_str_list(self.relevant_ids, "relevant_ids", allow_empty=False)
        _require_metric_dict(self.metrics)
        _require_dict(self.metadata, "metadata")


@dataclass
class GenerationEvaluationExample:
    example_id: str
    query: str
    references: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.example_id, "example_id")
        _require_non_empty_str(self.query, "query")
        _require_str_list(self.references, "references", allow_empty=False)
        _require_dict(self.metadata, "metadata")


@dataclass
class GenerationEvaluationResult:
    example_id: str
    query: str
    prediction: str
    references: list[str]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.example_id, "example_id")
        _require_non_empty_str(self.query, "query")
        if not isinstance(self.prediction, str):
            raise ValueError("prediction must be a string")
        _require_str_list(self.references, "references", allow_empty=False)
        _require_metric_dict(self.metrics)
        _require_dict(self.metadata, "metadata")


def retrieval_example_to_dict(example: RetrievalEvaluationExample) -> dict[str, Any]:
    if not isinstance(example, RetrievalEvaluationExample):
        raise ValueError("example must be a RetrievalEvaluationExample")
    return {
        "example_id": example.example_id,
        "query": example.query,
        "relevant_ids": list(example.relevant_ids),
        "metadata": dict(example.metadata),
    }


def retrieval_example_from_dict(data: dict[str, Any]) -> RetrievalEvaluationExample:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    return RetrievalEvaluationExample(
        example_id=data["example_id"],
        query=data["query"],
        relevant_ids=list(data["relevant_ids"]),
        metadata=data.get("metadata", {}),
    )


def retrieval_result_to_dict(result: RetrievalEvaluationResult) -> dict[str, Any]:
    if not isinstance(result, RetrievalEvaluationResult):
        raise ValueError("result must be a RetrievalEvaluationResult")
    return {
        "example_id": result.example_id,
        "query": result.query,
        "retrieved_ids": list(result.retrieved_ids),
        "relevant_ids": list(result.relevant_ids),
        "metrics": dict(result.metrics),
        "metadata": dict(result.metadata),
    }


def retrieval_result_from_dict(data: dict[str, Any]) -> RetrievalEvaluationResult:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    if "metrics" not in data:
        raise ValueError("metrics is required")
    return RetrievalEvaluationResult(
        example_id=data["example_id"],
        query=data["query"],
        retrieved_ids=list(data.get("retrieved_ids", [])),
        relevant_ids=list(data["relevant_ids"]),
        metrics=dict(data["metrics"]),
        metadata=data.get("metadata", {}),
    )


def generation_example_to_dict(example: GenerationEvaluationExample) -> dict[str, Any]:
    if not isinstance(example, GenerationEvaluationExample):
        raise ValueError("example must be a GenerationEvaluationExample")
    return {
        "example_id": example.example_id,
        "query": example.query,
        "references": list(example.references),
        "metadata": dict(example.metadata),
    }


def generation_example_from_dict(data: dict[str, Any]) -> GenerationEvaluationExample:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    return GenerationEvaluationExample(
        example_id=data["example_id"],
        query=data["query"],
        references=list(data["references"]),
        metadata=data.get("metadata", {}),
    )


def generation_result_to_dict(result: GenerationEvaluationResult) -> dict[str, Any]:
    if not isinstance(result, GenerationEvaluationResult):
        raise ValueError("result must be a GenerationEvaluationResult")
    return {
        "example_id": result.example_id,
        "query": result.query,
        "prediction": result.prediction,
        "references": list(result.references),
        "metrics": dict(result.metrics),
        "metadata": dict(result.metadata),
    }


def generation_result_from_dict(data: dict[str, Any]) -> GenerationEvaluationResult:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    if "metrics" not in data:
        raise ValueError("metrics is required")
    return GenerationEvaluationResult(
        example_id=data["example_id"],
        query=data["query"],
        prediction=data["prediction"],
        references=list(data["references"]),
        metrics=dict(data["metrics"]),
        metadata=data.get("metadata", {}),
    )
