from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sproutrag.retrieval.schema import RetrievalResult


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_dict(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_result_list(value: list[RetrievalResult], field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of RetrievalResult")
    if not all(isinstance(item, RetrievalResult) for item in value):
        raise ValueError(f"{field_name} must contain RetrievalResult instances")


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    contexts: list[RetrievalResult]
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.query, "query")
        if not isinstance(self.answer, str):
            raise ValueError("answer must be a string")
        _require_result_list(self.contexts, "contexts")
        _require_non_empty_str(self.prompt, "prompt")
        _require_dict(self.metadata, "metadata")


def generated_answer_to_dict(generated: GeneratedAnswer) -> dict[str, Any]:
    if not isinstance(generated, GeneratedAnswer):
        raise ValueError("generated must be a GeneratedAnswer")
    contexts = []
    for result in generated.contexts:
        contexts.append(
            {
                "node_id": result.node_id,
                "doc_id": result.doc_id,
                "text": result.text,
                "score": result.score,
                "depth": result.depth,
                "is_leaf": result.is_leaf,
                "sentence_chunk_ids": list(result.sentence_chunk_ids),
                "metadata": dict(result.metadata),
            }
        )
    return {
        "query": generated.query,
        "answer": generated.answer,
        "contexts": contexts,
        "prompt": generated.prompt,
        "metadata": dict(generated.metadata),
    }


def generated_answer_from_dict(data: dict[str, Any]) -> GeneratedAnswer:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    contexts = []
    for item in data.get("contexts", []):
        contexts.append(
            RetrievalResult(
                node_id=item["node_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                score=item["score"],
                depth=item["depth"],
                is_leaf=item["is_leaf"],
                sentence_chunk_ids=list(item["sentence_chunk_ids"]),
                metadata=item.get("metadata", {}),
            )
        )
    return GeneratedAnswer(
        query=data["query"],
        answer=data["answer"],
        contexts=contexts,
        prompt=data["prompt"],
        metadata=data.get("metadata", {}),
    )
