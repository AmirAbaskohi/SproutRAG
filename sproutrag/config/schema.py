from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_optional_path(value: str | None, name: str) -> None:
    if value is None:
        return
    _validate_non_empty_string(value, name)


def _validate_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_positive_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_float(value: Any, name: str) -> None:
    if not _is_finite_number(value) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _validate_non_negative_float(value: Any, name: str) -> None:
    if not _is_finite_number(value) or float(value) < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _validate_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _validate_choice(value: Any, name: str, choices: set[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")


@dataclass
class RuntimeConfig:
    device: str | None = None
    seed: int = 42
    num_workers: int = 0
    use_cuda_if_available: bool = True

    def __post_init__(self) -> None:
        _validate_optional_path(self.device, "device")
        _validate_non_negative_int(self.seed, "seed")
        _validate_non_negative_int(self.num_workers, "num_workers")
        _validate_bool(self.use_cuda_if_available, "use_cuda_if_available")


@dataclass
class EncoderConfig:
    model_name_or_path: str
    max_length: int = 4096
    normalize_embeddings: bool = True
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.model_name_or_path, "model_name_or_path")
        _validate_positive_int(self.max_length, "max_length")
        _validate_bool(self.normalize_embeddings, "normalize_embeddings")
        _validate_bool(self.trust_remote_code, "trust_remote_code")


@dataclass
class AggregatorConfig:
    init_strategy: str = "uniform"
    mask_diagonal: bool = True

    def __post_init__(self) -> None:
        _validate_choice(self.init_strategy, "init_strategy", {"uniform", "normal", "last_layer"})
        _validate_bool(self.mask_diagonal, "mask_diagonal")


@dataclass
class IndexingConfig:
    input_path: str
    output_dir: str
    index_name: str = "default"
    max_sentences_per_chunk: int = 2
    batch_size: int = 1
    overwrite: bool = False

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.input_path, "input_path")
        _validate_non_empty_string(self.output_dir, "output_dir")
        _validate_non_empty_string(self.index_name, "index_name")
        _validate_positive_int(self.max_sentences_per_chunk, "max_sentences_per_chunk")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.overwrite, "overwrite")


@dataclass
class RetrievalConfig:
    index_dir: str
    index_name: str = "default"
    query: str | None = None
    query_path: str | None = None
    output_path: str | None = None
    top_k: int = 10
    per_document_top_k: int = 5
    beam_width: int = 5
    threshold: float = 0.0
    collect_strategy: str = "threshold"
    include_root: bool = False
    multi_document: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.index_dir, "index_dir")
        _validate_non_empty_string(self.index_name, "index_name")
        _validate_optional_path(self.query, "query")
        _validate_optional_path(self.query_path, "query_path")
        if self.query is None and self.query_path is None:
            raise ValueError("query or query_path must be provided")
        _validate_optional_path(self.output_path, "output_path")
        _validate_positive_int(self.top_k, "top_k")
        _validate_positive_int(self.per_document_top_k, "per_document_top_k")
        _validate_positive_int(self.beam_width, "beam_width")
        _validate_non_negative_float(self.threshold, "threshold")
        _validate_choice(self.collect_strategy, "collect_strategy", {"threshold", "visited"})
        _validate_bool(self.include_root, "include_root")
        _validate_bool(self.multi_document, "multi_document")


@dataclass
class RerankerConfig:
    enabled: bool = False
    type: str = "none"
    model_name_or_path: str | None = None
    max_length: int = 512
    batch_size: int = 8
    metadata_score_key: str | None = None
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        _validate_bool(self.enabled, "enabled")
        _validate_choice(self.type, "type", {"none", "noop", "score", "cross_encoder"})
        _validate_optional_path(self.model_name_or_path, "model_name_or_path")
        if self.enabled:
            if self.type == "none":
                raise ValueError("type must not be 'none' when enabled")
            if self.type == "cross_encoder" and self.model_name_or_path is None:
                raise ValueError("model_name_or_path is required for cross_encoder")
        _validate_positive_int(self.max_length, "max_length")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_optional_path(self.metadata_score_key, "metadata_score_key")
        _validate_bool(self.trust_remote_code, "trust_remote_code")


@dataclass
class GeneratorConfig:
    type: str = "echo"
    model_name_or_path: str | None = None
    max_input_length: int = 4096
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    include_system_prompt: bool = True
    system_prompt: str | None = None
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        _validate_choice(self.type, "type", {"echo", "hf"})
        _validate_optional_path(self.model_name_or_path, "model_name_or_path")
        if self.type == "hf" and self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required for hf generator")
        _validate_positive_int(self.max_input_length, "max_input_length")
        _validate_positive_int(self.max_new_tokens, "max_new_tokens")
        _validate_non_negative_float(self.temperature, "temperature")
        _validate_bool(self.do_sample, "do_sample")
        _validate_bool(self.include_system_prompt, "include_system_prompt")
        _validate_optional_path(self.system_prompt, "system_prompt")
        _validate_bool(self.trust_remote_code, "trust_remote_code")


@dataclass
class ContextConfig:
    include_scores: bool = True
    include_metadata: bool = False
    include_node_ids: bool = True
    context_separator: str = "\n\n"
    max_context_chars: int | None = None

    def __post_init__(self) -> None:
        _validate_bool(self.include_scores, "include_scores")
        _validate_bool(self.include_metadata, "include_metadata")
        _validate_bool(self.include_node_ids, "include_node_ids")
        _validate_non_empty_string(self.context_separator, "context_separator")
        if self.max_context_chars is not None:
            _validate_positive_int(self.max_context_chars, "max_context_chars")


@dataclass
class TrainingConfig:
    train_path: str
    output_dir: str
    run_name: str = "default"
    num_epochs: int = 3
    batch_size: int = 32
    max_negatives: int | None = None
    require_equal_negatives: bool = True
    learning_rate: float = 2e-5
    aggregator_learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    gradient_clip_norm: float | None = 1.0
    checkpoint_every_steps: int | None = None
    log_every_steps: int = 1
    max_sentences_per_chunk: int = 2
    normalize_passage_embeddings: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.train_path, "train_path")
        _validate_non_empty_string(self.output_dir, "output_dir")
        _validate_non_empty_string(self.run_name, "run_name")
        _validate_positive_int(self.num_epochs, "num_epochs")
        _validate_positive_int(self.batch_size, "batch_size")
        if self.max_negatives is not None:
            _validate_positive_int(self.max_negatives, "max_negatives")
        _validate_bool(self.require_equal_negatives, "require_equal_negatives")
        _validate_positive_float(self.learning_rate, "learning_rate")
        _validate_positive_float(self.aggregator_learning_rate, "aggregator_learning_rate")
        _validate_non_negative_float(self.weight_decay, "weight_decay")
        if not _is_finite_number(self.warmup_ratio) or not (0.0 <= float(self.warmup_ratio) <= 1.0):
            raise ValueError("warmup_ratio must be a finite number in [0, 1]")
        if self.gradient_clip_norm is not None:
            _validate_positive_float(self.gradient_clip_norm, "gradient_clip_norm")
        if self.checkpoint_every_steps is not None:
            _validate_positive_int(self.checkpoint_every_steps, "checkpoint_every_steps")
        _validate_positive_int(self.log_every_steps, "log_every_steps")
        _validate_positive_int(self.max_sentences_per_chunk, "max_sentences_per_chunk")
        _validate_bool(self.normalize_passage_embeddings, "normalize_passage_embeddings")


@dataclass
class LossConfig:
    retrieval_temperature: float = 0.05
    attention_lambda: float = 0.1
    use_in_batch_negatives: bool = False
    retrieval_reduction: str = "mean"
    attention_reduction: str = "mean"
    attention_example_reduction: str = "mean"
    allow_self_pairs: bool = True
    empty_attention_policy: str = "zero"

    def __post_init__(self) -> None:
        _validate_positive_float(self.retrieval_temperature, "retrieval_temperature")
        _validate_non_negative_float(self.attention_lambda, "attention_lambda")
        _validate_bool(self.use_in_batch_negatives, "use_in_batch_negatives")
        _validate_choice(self.retrieval_reduction, "retrieval_reduction", {"mean", "sum"})
        _validate_choice(self.attention_reduction, "attention_reduction", {"mean", "sum", "none"})
        _validate_choice(self.attention_example_reduction, "attention_example_reduction", {"mean", "sum"})
        _validate_bool(self.allow_self_pairs, "allow_self_pairs")
        _validate_choice(self.empty_attention_policy, "empty_attention_policy", {"zero", "error"})


@dataclass
class EvaluationConfig:
    task: str
    examples_path: str
    index_dir: str | None = None
    index_name: str = "default"
    output_path: str | None = None
    ks: list[int] = field(default_factory=lambda: [1, 3, 5])
    include_optional_generation_metrics: bool = False
    include_bertscore: bool = False
    bertscore_model_type: str | None = None

    def __post_init__(self) -> None:
        _validate_choice(self.task, "task", {"retrieval", "generation"})
        _validate_non_empty_string(self.examples_path, "examples_path")
        _validate_optional_path(self.index_dir, "index_dir")
        _validate_non_empty_string(self.index_name, "index_name")
        _validate_optional_path(self.output_path, "output_path")
        if not isinstance(self.ks, list) or not self.ks:
            raise ValueError("ks must be a non-empty list of integers")
        for value in self.ks:
            _validate_positive_int(value, "ks")
        _validate_bool(self.include_optional_generation_metrics, "include_optional_generation_metrics")
        _validate_bool(self.include_bertscore, "include_bertscore")
        _validate_optional_path(self.bertscore_model_type, "bertscore_model_type")


@dataclass
class IndexCommandConfig:
    runtime: RuntimeConfig
    encoder: EncoderConfig
    aggregator: AggregatorConfig
    indexing: IndexingConfig

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeConfig):
            raise ValueError("runtime must be a RuntimeConfig")
        if not isinstance(self.encoder, EncoderConfig):
            raise ValueError("encoder must be an EncoderConfig")
        if not isinstance(self.aggregator, AggregatorConfig):
            raise ValueError("aggregator must be an AggregatorConfig")
        if not isinstance(self.indexing, IndexingConfig):
            raise ValueError("indexing must be an IndexingConfig")


@dataclass
class RetrieveCommandConfig:
    runtime: RuntimeConfig
    encoder: EncoderConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeConfig):
            raise ValueError("runtime must be a RuntimeConfig")
        if not isinstance(self.encoder, EncoderConfig):
            raise ValueError("encoder must be an EncoderConfig")
        if not isinstance(self.retrieval, RetrievalConfig):
            raise ValueError("retrieval must be a RetrievalConfig")
        if not isinstance(self.reranker, RerankerConfig):
            raise ValueError("reranker must be a RerankerConfig")


@dataclass
class AnswerCommandConfig:
    runtime: RuntimeConfig
    encoder: EncoderConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    context: ContextConfig
    generator: GeneratorConfig

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeConfig):
            raise ValueError("runtime must be a RuntimeConfig")
        if not isinstance(self.encoder, EncoderConfig):
            raise ValueError("encoder must be an EncoderConfig")
        if not isinstance(self.retrieval, RetrievalConfig):
            raise ValueError("retrieval must be a RetrievalConfig")
        if not isinstance(self.reranker, RerankerConfig):
            raise ValueError("reranker must be a RerankerConfig")
        if not isinstance(self.context, ContextConfig):
            raise ValueError("context must be a ContextConfig")
        if not isinstance(self.generator, GeneratorConfig):
            raise ValueError("generator must be a GeneratorConfig")


@dataclass
class TrainCommandConfig:
    runtime: RuntimeConfig
    encoder: EncoderConfig
    aggregator: AggregatorConfig
    training: TrainingConfig
    loss: LossConfig

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeConfig):
            raise ValueError("runtime must be a RuntimeConfig")
        if not isinstance(self.encoder, EncoderConfig):
            raise ValueError("encoder must be an EncoderConfig")
        if not isinstance(self.aggregator, AggregatorConfig):
            raise ValueError("aggregator must be an AggregatorConfig")
        if not isinstance(self.training, TrainingConfig):
            raise ValueError("training must be a TrainingConfig")
        if not isinstance(self.loss, LossConfig):
            raise ValueError("loss must be a LossConfig")


@dataclass
class EvaluateCommandConfig:
    runtime: RuntimeConfig
    evaluation: EvaluationConfig
    encoder: EncoderConfig | None = None
    retrieval: RetrievalConfig | None = None
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeConfig):
            raise ValueError("runtime must be a RuntimeConfig")
        if not isinstance(self.evaluation, EvaluationConfig):
            raise ValueError("evaluation must be an EvaluationConfig")
        if self.encoder is not None and not isinstance(self.encoder, EncoderConfig):
            raise ValueError("encoder must be an EncoderConfig")
        if self.retrieval is not None and not isinstance(self.retrieval, RetrievalConfig):
            raise ValueError("retrieval must be a RetrievalConfig")
        if not isinstance(self.reranker, RerankerConfig):
            raise ValueError("reranker must be a RerankerConfig")
        if not isinstance(self.context, ContextConfig):
            raise ValueError("context must be a ContextConfig")
        if not isinstance(self.generator, GeneratorConfig):
            raise ValueError("generator must be a GeneratorConfig")

        if self.evaluation.task in {"retrieval", "generation"}:
            if self.encoder is None or self.retrieval is None:
                raise ValueError("encoder and retrieval are required for evaluation")
