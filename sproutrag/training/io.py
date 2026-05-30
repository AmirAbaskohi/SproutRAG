from __future__ import annotations

import json
from pathlib import Path

from sproutrag.training.schema import TrainingExample, training_example_from_dict, training_example_to_dict


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def load_training_examples_jsonl(
    path: str | Path,
    max_examples: int | None = None,
) -> list[TrainingExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if max_examples is not None:
        _require_positive_int(max_examples, "max_examples")

    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_num}") from exc
            try:
                example = training_example_from_dict(data)
            except Exception as exc:
                raise ValueError(f"invalid example on line {line_num}") from exc
            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def save_training_examples_jsonl(
    examples: list[TrainingExample],
    path: str | Path,
) -> None:
    if not isinstance(examples, list):
        raise ValueError("examples must be a list of TrainingExample")
    if not all(isinstance(item, TrainingExample) for item in examples):
        raise ValueError("examples must contain TrainingExample instances")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            data = training_example_to_dict(example)
            handle.write(json.dumps(data, sort_keys=True) + "\n")


def load_training_examples_json(path: str | Path) -> list[TrainingExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "examples" in data and isinstance(data["examples"], list):
        items = data["examples"]
    else:
        raise ValueError("invalid JSON structure for training examples")

    examples: list[TrainingExample] = []
    for item in items:
        examples.append(training_example_from_dict(item))
    return examples


def save_training_examples_json(
    examples: list[TrainingExample],
    path: str | Path,
) -> None:
    if not isinstance(examples, list):
        raise ValueError("examples must be a list of TrainingExample")
    if not all(isinstance(item, TrainingExample) for item in examples):
        raise ValueError("examples must contain TrainingExample instances")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"examples": [training_example_to_dict(example) for example in examples]}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
