from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any

from sproutrag.config.schema import (
    RuntimeConfig,
    EncoderConfig,
    AggregatorConfig,
    IndexingConfig,
    RetrievalConfig,
    RerankerConfig,
    ContextConfig,
    GeneratorConfig,
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
    IndexCommandConfig,
    RetrieveCommandConfig,
    AnswerCommandConfig,
    TrainCommandConfig,
    EvaluateCommandConfig,
)


def load_config_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"config not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs") from exc
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            raise ValueError("YAML config is empty")
    else:
        raise ValueError("unsupported config file extension")

    if not isinstance(data, dict):
        raise ValueError("config must be a dictionary")
    return data


def save_config_file(config: dict[str, Any], path: str | Path) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
        return
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required to save YAML configs") from exc
        with file_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=True)
        return
    raise ValueError("unsupported config file extension")


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise ValueError("base and override must be dictionaries")
    merged: dict[str, Any] = {}
    for key, value in base.items():
        if key in override:
            if isinstance(value, dict) and isinstance(override[key], dict):
                merged[key] = merge_dicts(value, override[key])
            else:
                merged[key] = override[key]
        else:
            merged[key] = value
    for key, value in override.items():
        if key not in merged:
            merged[key] = value
    return merged


def _parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_cli_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if overrides is None:
        return copy.deepcopy(config)
    if not isinstance(overrides, list):
        raise ValueError("overrides must be a list")
    merged = copy.deepcopy(config)
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ValueError("override must be in key=value format")
        key_path, raw_value = override.split("=", 1)
        if not key_path.strip():
            raise ValueError("override key must be non-empty")
        value = _parse_override_value(raw_value)
        keys = key_path.split(".")
        current: dict[str, Any] = merged
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    return merged


def build_runtime_config(data: dict[str, Any] | None) -> RuntimeConfig:
    if data is None:
        return RuntimeConfig()
    if not isinstance(data, dict):
        raise ValueError("runtime must be a dictionary")
    return RuntimeConfig(**data)


def build_encoder_config(data: dict[str, Any]) -> EncoderConfig:
    if not isinstance(data, dict):
        raise ValueError("encoder must be a dictionary")
    return EncoderConfig(**data)


def build_aggregator_config(data: dict[str, Any] | None) -> AggregatorConfig:
    if data is None:
        return AggregatorConfig()
    if not isinstance(data, dict):
        raise ValueError("aggregator must be a dictionary")
    return AggregatorConfig(**data)


def build_indexing_config(data: dict[str, Any]) -> IndexingConfig:
    if not isinstance(data, dict):
        raise ValueError("indexing must be a dictionary")
    return IndexingConfig(**data)


def build_retrieval_config(data: dict[str, Any]) -> RetrievalConfig:
    if not isinstance(data, dict):
        raise ValueError("retrieval must be a dictionary")
    return RetrievalConfig(**data)


def build_reranker_config(data: dict[str, Any] | None) -> RerankerConfig:
    if data is None:
        return RerankerConfig()
    if not isinstance(data, dict):
        raise ValueError("reranker must be a dictionary")
    return RerankerConfig(**data)


def build_context_config(data: dict[str, Any] | None) -> ContextConfig:
    if data is None:
        return ContextConfig()
    if not isinstance(data, dict):
        raise ValueError("context must be a dictionary")
    return ContextConfig(**data)


def build_generator_config(data: dict[str, Any] | None) -> GeneratorConfig:
    if data is None:
        return GeneratorConfig()
    if not isinstance(data, dict):
        raise ValueError("generator must be a dictionary")
    return GeneratorConfig(**data)


def build_training_config(data: dict[str, Any]) -> TrainingConfig:
    if not isinstance(data, dict):
        raise ValueError("training must be a dictionary")
    return TrainingConfig(**data)


def build_loss_config(data: dict[str, Any] | None) -> LossConfig:
    if data is None:
        return LossConfig()
    if not isinstance(data, dict):
        raise ValueError("loss must be a dictionary")
    return LossConfig(**data)


def build_evaluation_config(data: dict[str, Any]) -> EvaluationConfig:
    if not isinstance(data, dict):
        raise ValueError("evaluation must be a dictionary")
    return EvaluationConfig(**data)


def build_index_command_config(data: dict[str, Any]) -> IndexCommandConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a dictionary")
    if "encoder" not in data or "indexing" not in data:
        raise ValueError("encoder and indexing sections are required")
    return IndexCommandConfig(
        runtime=build_runtime_config(data.get("runtime")),
        encoder=build_encoder_config(data["encoder"]),
        aggregator=build_aggregator_config(data.get("aggregator")),
        indexing=build_indexing_config(data["indexing"]),
    )


def build_retrieve_command_config(data: dict[str, Any]) -> RetrieveCommandConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a dictionary")
    if "encoder" not in data or "retrieval" not in data:
        raise ValueError("encoder and retrieval sections are required")
    return RetrieveCommandConfig(
        runtime=build_runtime_config(data.get("runtime")),
        encoder=build_encoder_config(data["encoder"]),
        retrieval=build_retrieval_config(data["retrieval"]),
        reranker=build_reranker_config(data.get("reranker")),
    )


def build_answer_command_config(data: dict[str, Any]) -> AnswerCommandConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a dictionary")
    for key in ("encoder", "retrieval", "context", "generator"):
        if key not in data:
            raise ValueError(f"{key} section is required")
    return AnswerCommandConfig(
        runtime=build_runtime_config(data.get("runtime")),
        encoder=build_encoder_config(data["encoder"]),
        retrieval=build_retrieval_config(data["retrieval"]),
        reranker=build_reranker_config(data.get("reranker")),
        context=build_context_config(data.get("context")),
        generator=build_generator_config(data.get("generator")),
    )


def build_train_command_config(data: dict[str, Any]) -> TrainCommandConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a dictionary")
    for key in ("encoder", "training"):
        if key not in data:
            raise ValueError(f"{key} section is required")
    return TrainCommandConfig(
        runtime=build_runtime_config(data.get("runtime")),
        encoder=build_encoder_config(data["encoder"]),
        aggregator=build_aggregator_config(data.get("aggregator")),
        training=build_training_config(data["training"]),
        loss=build_loss_config(data.get("loss")),
    )


def build_evaluate_command_config(data: dict[str, Any]) -> EvaluateCommandConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a dictionary")
    if "evaluation" not in data:
        raise ValueError("evaluation section is required")
    return EvaluateCommandConfig(
        runtime=build_runtime_config(data.get("runtime")),
        encoder=build_encoder_config(data["encoder"]) if data.get("encoder") is not None else None,
        retrieval=build_retrieval_config(data["retrieval"]) if data.get("retrieval") is not None else None,
        reranker=build_reranker_config(data.get("reranker")),
        context=build_context_config(data.get("context")),
        generator=build_generator_config(data.get("generator")),
        evaluation=build_evaluation_config(data["evaluation"]),
    )


def load_typed_config(
    path: str | Path,
    command: str,
    overrides: list[str] | None = None,
) -> Any:
    data = load_config_file(path)
    data = apply_cli_overrides(data, overrides)

    if command == "index":
        return build_index_command_config(data)
    if command == "retrieve":
        return build_retrieve_command_config(data)
    if command == "answer":
        return build_answer_command_config(data)
    if command == "train":
        return build_train_command_config(data)
    if command == "evaluate":
        return build_evaluate_command_config(data)

    raise ValueError("command must be one of: index, retrieve, answer, train, evaluate")
