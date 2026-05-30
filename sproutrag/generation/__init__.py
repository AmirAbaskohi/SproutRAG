"""Generation utilities for SproutRAG."""

from sproutrag.generation.schema import (
    GeneratedAnswer,
    generated_answer_to_dict,
    generated_answer_from_dict,
)
from sproutrag.generation.context_builder import ContextBuilder
from sproutrag.generation.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_TEMPLATE,
    build_rag_prompt,
    build_chat_messages,
)
from sproutrag.generation.base import BaseGenerator, EchoGenerator
from sproutrag.generation.hf_generator import HuggingFaceGenerator
from sproutrag.generation.pipeline import RAGPipeline

__all__ = [
    "GeneratedAnswer",
    "generated_answer_to_dict",
    "generated_answer_from_dict",
    "ContextBuilder",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_TEMPLATE",
    "build_rag_prompt",
    "build_chat_messages",
    "BaseGenerator",
    "EchoGenerator",
    "HuggingFaceGenerator",
    "RAGPipeline",
]
