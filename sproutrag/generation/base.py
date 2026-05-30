from __future__ import annotations

from abc import ABC, abstractmethod

from sproutrag.generation.context_builder import ContextBuilder
from sproutrag.generation.prompts import DEFAULT_SYSTEM_PROMPT, build_rag_prompt
from sproutrag.generation.schema import GeneratedAnswer
from sproutrag.retrieval.schema import RetrievalResult


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_contexts(contexts: list[RetrievalResult]) -> None:
    if not isinstance(contexts, list):
        raise ValueError("contexts must be a list of RetrievalResult")
    if not all(isinstance(item, RetrievalResult) for item in contexts):
        raise ValueError("contexts must contain RetrievalResult instances")


class BaseGenerator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, query: str, contexts: list[RetrievalResult]) -> GeneratedAnswer:
        raise NotImplementedError


class EchoGenerator(BaseGenerator):
    def __init__(
        self,
        context_builder: ContextBuilder | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        include_system_prompt: bool = True,
        name: str = "echo",
    ) -> None:
        _require_non_empty_str(name, "name")
        if context_builder is None:
            context_builder = ContextBuilder()
        self.context_builder = context_builder
        self.system_prompt = system_prompt
        self.include_system_prompt = include_system_prompt
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, query: str, contexts: list[RetrievalResult]) -> GeneratedAnswer:
        _require_non_empty_str(query, "query")
        _validate_contexts(contexts)
        context_text = self.context_builder.build_context(contexts)
        prompt = build_rag_prompt(
            query,
            context_text,
            system_prompt=self.system_prompt,
            include_system_prompt=self.include_system_prompt,
        )
        if context_text:
            answer = f"Echo answer based on {len(contexts)} context(s)."
        else:
            answer = "The answer is not available in the provided context."
        metadata = {
            "generator_name": self.name,
            "num_contexts": len(contexts),
            "citations": self.context_builder.build_citations(contexts),
        }
        return GeneratedAnswer(
            query=query,
            answer=answer,
            contexts=list(contexts),
            prompt=prompt,
            metadata=metadata,
        )
