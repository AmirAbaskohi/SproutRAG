from __future__ import annotations

from typing import Any

from sproutrag.retrieval.schema import RetrievalResult


def _require_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_results(results: list[RetrievalResult]) -> None:
    if not isinstance(results, list):
        raise ValueError("results must be a list of RetrievalResult")
    if not all(isinstance(item, RetrievalResult) for item in results):
        raise ValueError("results must contain RetrievalResult instances")


def _validate_max_chars(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or value <= 0:
        raise ValueError("max_context_chars must be a positive integer")


class ContextBuilder:
    def __init__(
        self,
        include_scores: bool = True,
        include_metadata: bool = False,
        include_node_ids: bool = True,
        context_separator: str = "\n\n",
    ) -> None:
        _require_bool(include_scores, "include_scores")
        _require_bool(include_metadata, "include_metadata")
        _require_bool(include_node_ids, "include_node_ids")
        _require_non_empty_str(context_separator, "context_separator")
        self.include_scores = include_scores
        self.include_metadata = include_metadata
        self.include_node_ids = include_node_ids
        self.context_separator = context_separator

    def build_context(
        self,
        results: list[RetrievalResult],
        max_context_chars: int | None = None,
    ) -> str:
        _validate_results(results)
        _validate_max_chars(max_context_chars)
        if not results:
            return ""
        blocks: list[str] = []
        for idx, result in enumerate(results, start=1):
            lines = [f"[Context {idx}]", f"Document: {result.doc_id}"]
            if self.include_node_ids:
                lines.append(f"Node: {result.node_id}")
            if self.include_scores:
                lines.append(f"Score: {result.score:.4f}")
            lines.append("Text:")
            lines.append(result.text)
            if self.include_metadata:
                lines.append("Metadata:")
                for key in sorted(result.metadata.keys()):
                    lines.append(f"{key}: {result.metadata[key]}")
            blocks.append("\n".join(lines))

        context = self.context_separator.join(blocks)
        if max_context_chars is None or len(context) <= max_context_chars:
            return context

        marker = "\n[TRUNCATED]"
        if max_context_chars >= len(marker):
            truncated = context[: max_context_chars - len(marker)] + marker
            return truncated[:max_context_chars]
        return context[:max_context_chars]

    def build_citations(self, results: list[RetrievalResult]) -> list[dict[str, Any]]:
        _validate_results(results)
        citations: list[dict[str, Any]] = []
        for idx, result in enumerate(results, start=1):
            citations.append(
                {
                    "context_id": idx,
                    "doc_id": result.doc_id,
                    "node_id": result.node_id,
                    "score": result.score,
                    "sentence_chunk_ids": list(result.sentence_chunk_ids),
                    "is_leaf": result.is_leaf,
                    "depth": result.depth,
                }
            )
        return citations
