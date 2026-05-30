"""Encoding utilities for SproutRAG."""

from .schema import EncodedDocument, move_encoded_document
from .pooling import (
    l2_normalize,
    mean_pool_hidden_states,
    mean_pool_tokens_by_span,
    pool_attentions_by_span,
)
from .sllm_encoder import SLLMEncoder

__all__ = [
    "EncodedDocument",
    "move_encoded_document",
    "mean_pool_hidden_states",
    "mean_pool_tokens_by_span",
    "pool_attentions_by_span",
    "l2_normalize",
    "SLLMEncoder",
]
