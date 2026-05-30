from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from sproutrag.evaluation.schema import (
    RetrievalEvaluationExample,
    RetrievalEvaluationResult,
    GenerationEvaluationExample,
    GenerationEvaluationResult,
    retrieval_example_from_dict,
    retrieval_example_to_dict,
    retrieval_result_from_dict,
    retrieval_result_to_dict,
    generation_example_from_dict,
    generation_example_to_dict,
    generation_result_from_dict,
    generation_result_to_dict,
)

T = TypeVar("T")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _load_jsonl(
    path: str | Path,
    max_items: int | None,
    loader: Callable[[dict[str, Any]], T],
    label: str,
) -> list[T]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    if max_items is not None:
        _require_positive_int(max_items, "max_items")

    results: list[T] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            try:
                results.append(loader(data))
            except Exception as exc:
                raise ValueError(f"invalid {label} on line {line_number}") from exc
            if max_items is not None and len(results) >= max_items:
                break
    return results


def _save_jsonl(
    items: list[T],
    path: str | Path,
    serializer: Callable[[T], dict[str, Any]],
    expected_type: type,
) -> None:
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    if not all(isinstance(item, expected_type) for item in items):
        raise ValueError("items must contain the expected object type")

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(serializer(item), sort_keys=True))
            handle.write("\n")


def _load_json(
    path: str | Path,
    key: str,
    loader: Callable[[dict[str, Any]], T],
) -> list[T]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        if key not in data:
            raise ValueError("invalid JSON structure")
        items = data[key]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("invalid JSON structure")

    if not isinstance(items, list):
        raise ValueError("invalid JSON structure")

    return [loader(item) for item in items]


def _save_json(
    items: list[T],
    path: str | Path,
    key: str,
    serializer: Callable[[T], dict[str, Any]],
    expected_type: type,
) -> None:
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    if not all(isinstance(item, expected_type) for item in items):
        raise ValueError("items must contain the expected object type")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: [serializer(item) for item in items]}
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_retrieval_examples_jsonl(
    path: str | Path,
    max_examples: int | None = None,
) -> list[RetrievalEvaluationExample]:
    return _load_jsonl(path, max_examples, retrieval_example_from_dict, "retrieval example")


def save_retrieval_examples_jsonl(
    examples: list[RetrievalEvaluationExample],
    path: str | Path,
) -> None:
    _save_jsonl(examples, path, retrieval_example_to_dict, RetrievalEvaluationExample)


def load_retrieval_examples_json(path: str | Path) -> list[RetrievalEvaluationExample]:
    return _load_json(path, "examples", retrieval_example_from_dict)


def save_retrieval_examples_json(
    examples: list[RetrievalEvaluationExample],
    path: str | Path,
) -> None:
    _save_json(examples, path, "examples", retrieval_example_to_dict, RetrievalEvaluationExample)


def load_retrieval_results_jsonl(
    path: str | Path,
    max_results: int | None = None,
) -> list[RetrievalEvaluationResult]:
    return _load_jsonl(path, max_results, retrieval_result_from_dict, "retrieval result")


def save_retrieval_results_jsonl(
    results: list[RetrievalEvaluationResult],
    path: str | Path,
) -> None:
    _save_jsonl(results, path, retrieval_result_to_dict, RetrievalEvaluationResult)


def load_retrieval_results_json(path: str | Path) -> list[RetrievalEvaluationResult]:
    return _load_json(path, "results", retrieval_result_from_dict)


def save_retrieval_results_json(
    results: list[RetrievalEvaluationResult],
    path: str | Path,
) -> None:
    _save_json(results, path, "results", retrieval_result_to_dict, RetrievalEvaluationResult)


def load_generation_examples_jsonl(
    path: str | Path,
    max_examples: int | None = None,
) -> list[GenerationEvaluationExample]:
    return _load_jsonl(path, max_examples, generation_example_from_dict, "generation example")


def save_generation_examples_jsonl(
    examples: list[GenerationEvaluationExample],
    path: str | Path,
) -> None:
    _save_jsonl(examples, path, generation_example_to_dict, GenerationEvaluationExample)


def load_generation_examples_json(path: str | Path) -> list[GenerationEvaluationExample]:
    return _load_json(path, "examples", generation_example_from_dict)


def save_generation_examples_json(
    examples: list[GenerationEvaluationExample],
    path: str | Path,
) -> None:
    _save_json(examples, path, "examples", generation_example_to_dict, GenerationEvaluationExample)


def load_generation_results_jsonl(
    path: str | Path,
    max_results: int | None = None,
) -> list[GenerationEvaluationResult]:
    return _load_jsonl(path, max_results, generation_result_from_dict, "generation result")


def save_generation_results_jsonl(
    results: list[GenerationEvaluationResult],
    path: str | Path,
) -> None:
    _save_jsonl(results, path, generation_result_to_dict, GenerationEvaluationResult)


def load_generation_results_json(path: str | Path) -> list[GenerationEvaluationResult]:
    return _load_json(path, "results", generation_result_from_dict)


def save_generation_results_json(
    results: list[GenerationEvaluationResult],
    path: str | Path,
) -> None:
    _save_json(results, path, "results", generation_result_to_dict, GenerationEvaluationResult)
