from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sproutrag.training.io import load_training_examples_json, load_training_examples_jsonl
from sproutrag.training.schema import TrainingExample


def _validate_examples(examples: list[TrainingExample]) -> None:
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a non-empty list of TrainingExample")
    if not all(isinstance(item, TrainingExample) for item in examples):
        raise ValueError("examples must contain TrainingExample instances")


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class SproutRAGTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[TrainingExample]) -> None:
        _validate_examples(examples)
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TrainingExample:
        if not isinstance(index, int):
            raise IndexError("index must be an integer")
        if index < 0:
            index = len(self._examples) + index
        if index < 0 or index >= len(self._examples):
            raise IndexError("index out of range")
        return self._examples[index]

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        max_examples: int | None = None,
    ) -> "SproutRAGTrainingDataset":
        examples = load_training_examples_jsonl(path, max_examples=max_examples)
        if not examples:
            raise ValueError("no training examples found")
        return cls(examples)

    @classmethod
    def from_json(cls, path: str | Path) -> "SproutRAGTrainingDataset":
        examples = load_training_examples_json(path)
        if not examples:
            raise ValueError("no training examples found")
        return cls(examples)

    @classmethod
    def from_msmarco_v21(
        cls,
        max_examples: int = 30000,
        split: str = "train",
    ) -> "SproutRAGTrainingDataset":
        if not isinstance(max_examples, int) or max_examples <= 0:
            raise ValueError("max_examples must be a positive integer")
        _require_non_empty_str(split, "split")
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError("datasets is required to load microsoft/ms_marco") from exc

        dataset = load_dataset("microsoft/ms_marco", "v2.1", split=split)
        if dataset is None:
            raise ValueError("failed to load microsoft/ms_marco dataset")

        def _extract_query(item: dict[str, Any]) -> str:
            query = item.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            return query

        def _extract_passage(item: dict[str, Any]) -> str:
            passage = item.get("passage")
            if isinstance(passage, str) and passage.strip():
                return passage
            if isinstance(passage, dict):
                text = passage.get("passage_text") or passage.get("text")
                if isinstance(text, str) and text.strip():
                    return text
                if isinstance(text, list) and text:
                    first = text[0]
                    if isinstance(first, str) and first.strip():
                        return first
            if isinstance(item.get("passage_text"), str) and item.get("passage_text").strip():
                return item["passage_text"]
            passages = item.get("passages")
            if isinstance(passages, dict):
                text = passages.get("passage_text")
                if isinstance(text, list) and text:
                    first = text[0]
                    if isinstance(first, str) and first.strip():
                        return first
                if isinstance(text, str) and text.strip():
                    return text
            if isinstance(passages, list) and passages:
                first = passages[0]
                if isinstance(first, dict):
                    text = first.get("passage_text") or first.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
            raise ValueError("passage text not found in dataset row")

        queries: list[str] = []
        passages: list[str] = []
        example_ids: list[str] = []
        for idx, item in enumerate(dataset):
            if idx >= max_examples:
                break
            if not isinstance(item, dict):
                raise ValueError("dataset row must be a dictionary")
            query = _extract_query(item)
            passage_text = _extract_passage(item)
            query_id = item.get("query_id", idx)
            example_ids.append(str(query_id))
            queries.append(query)
            passages.append(passage_text)

        if not queries:
            raise ValueError("no MS MARCO examples loaded")

        examples: list[TrainingExample] = []
        for idx, (example_id, query, passage) in enumerate(zip(example_ids, queries, passages)):
            negative_passage = passages[(idx + 1) % len(passages)]
            examples.append(
                TrainingExample(
                    example_id=str(example_id),
                    query=query,
                    positive_passage=passage,
                    negative_passages=[negative_passage],
                    support_sentence_pairs=[],
                    metadata={"source": "ms_marco_v2.1", "split": split},
                )
            )
        return cls(examples)

    def get_example_ids(self) -> list[str]:
        return [example.example_id for example in self._examples]

    def filter_by_metadata(self, key: str, value: Any) -> "SproutRAGTrainingDataset":
        _require_non_empty_str(key, "key")
        filtered = [ex for ex in self._examples if ex.metadata.get(key) == value]
        if not filtered:
            raise ValueError("no examples match the provided metadata filter")
        return SproutRAGTrainingDataset(filtered)
