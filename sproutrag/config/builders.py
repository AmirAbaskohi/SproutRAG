from __future__ import annotations

import random
from typing import Any

import torch

from sproutrag.aggregation.attention_aggregator import AttentionAggregator
from sproutrag.config.schema import (
    RuntimeConfig,
    EncoderConfig,
    AggregatorConfig,
    RerankerConfig,
    ContextConfig,
    GeneratorConfig,
    RetrievalConfig,
    LossConfig,
    TrainingConfig,
)
from sproutrag.encoding.sllm_encoder import SLLMEncoder
from sproutrag.generation.context_builder import ContextBuilder
from sproutrag.generation.base import EchoGenerator
from sproutrag.generation.hf_generator import HuggingFaceGenerator
from sproutrag.generation.prompts import DEFAULT_SYSTEM_PROMPT
from sproutrag.generation.pipeline import RAGPipeline
from sproutrag.indexing.store import IndexStore
from sproutrag.reranking import NoOpReranker, ScoreReranker, CrossEncoderReranker
from sproutrag.retrieval.hierarchical_retriever import HierarchicalRetriever
from sproutrag.retrieval.multi_document_retriever import MultiDocumentRetriever
from sproutrag.training.losses import JointSproutRAGLoss
from sproutrag.training.collator import SproutRAGDataCollator


def resolve_device(runtime: RuntimeConfig) -> str:
    if runtime.device is not None:
        return runtime.device
    if runtime.use_cuda_if_available and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_encoder(config: EncoderConfig, runtime: RuntimeConfig) -> SLLMEncoder:
    return SLLMEncoder(
        model_name_or_path=config.model_name_or_path,
        device=resolve_device(runtime),
        max_length=config.max_length,
        normalize_embeddings=config.normalize_embeddings,
        trust_remote_code=config.trust_remote_code,
    )


def build_aggregator_from_config(
    config: AggregatorConfig,
    num_layers: int,
    num_heads: int,
) -> AttentionAggregator:
    return AttentionAggregator(
        num_layers=num_layers,
        num_heads=num_heads,
        init_strategy=config.init_strategy,
        mask_diagonal=config.mask_diagonal,
    )


def build_index_store(output_or_index_dir: str, index_name: str) -> IndexStore:
    return IndexStore(root_dir=output_or_index_dir, index_name=index_name)


def build_reranker(config: RerankerConfig, runtime: RuntimeConfig) -> Any | None:
    if not config.enabled:
        return None
    if config.type == "none":
        return None
    if config.type == "noop":
        return NoOpReranker()
    if config.type == "score":
        return ScoreReranker(metadata_score_key=config.metadata_score_key)
    if config.type == "cross_encoder":
        return CrossEncoderReranker(
            model_name_or_path=config.model_name_or_path or "",
            device=resolve_device(runtime),
            max_length=config.max_length,
            batch_size=config.batch_size,
            trust_remote_code=config.trust_remote_code,
        )
    raise ValueError("unsupported reranker type")


def build_context_builder(config: ContextConfig) -> ContextBuilder:
    return ContextBuilder(
        include_scores=config.include_scores,
        include_metadata=config.include_metadata,
        include_node_ids=config.include_node_ids,
        context_separator=config.context_separator,
    )


def build_generator(
    config: GeneratorConfig,
    context_config: ContextConfig,
    runtime: RuntimeConfig,
) -> Any:
    context_builder = build_context_builder(context_config)
    if config.type == "echo":
        if config.system_prompt is None:
            return EchoGenerator(
                context_builder=context_builder,
                include_system_prompt=config.include_system_prompt,
                name="echo",
            )
        return EchoGenerator(
            context_builder=context_builder,
            system_prompt=config.system_prompt,
            include_system_prompt=config.include_system_prompt,
            name="echo",
        )
    if config.type == "hf":
        return HuggingFaceGenerator(
            model_name_or_path=config.model_name_or_path or "",
            device=resolve_device(runtime),
            max_input_length=config.max_input_length,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            context_builder=context_builder,
            system_prompt=config.system_prompt if config.system_prompt is not None else DEFAULT_SYSTEM_PROMPT,
            include_system_prompt=config.include_system_prompt,
            trust_remote_code=config.trust_remote_code,
        )
    raise ValueError("unsupported generator type")


def build_retriever(
    encoder: Any,
    retrieval_config: RetrievalConfig,
    reranker: Any | None = None,
) -> Any:
    if retrieval_config.multi_document:
        return MultiDocumentRetriever(
            encoder=encoder,
            collect_strategy=retrieval_config.collect_strategy,
            include_root=retrieval_config.include_root,
            reranker=reranker,
        )
    return HierarchicalRetriever(
        encoder=encoder,
        collect_strategy=retrieval_config.collect_strategy,
        include_root=retrieval_config.include_root,
        reranker=reranker,
    )


def build_pipeline(retriever: Any, generator: Any) -> RAGPipeline:
    return RAGPipeline(retriever=retriever, generator=generator)


def build_loss(config: LossConfig) -> JointSproutRAGLoss:
    return JointSproutRAGLoss(
        retrieval_temperature=config.retrieval_temperature,
        attention_lambda=config.attention_lambda,
        use_in_batch_negatives=config.use_in_batch_negatives,
        retrieval_reduction=config.retrieval_reduction,
        attention_reduction=config.attention_reduction,
        attention_example_reduction=config.attention_example_reduction,
        allow_self_pairs=config.allow_self_pairs,
        empty_attention_policy=config.empty_attention_policy,
    )


def build_data_collator(config: TrainingConfig) -> SproutRAGDataCollator:
    return SproutRAGDataCollator(
        max_negatives=config.max_negatives,
        require_equal_negatives=config.require_equal_negatives,
    )
