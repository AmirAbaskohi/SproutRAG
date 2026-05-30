from __future__ import annotations

import copy
from typing import Any


def default_runtime_config() -> dict[str, Any]:
    return {
        "device": None,
        "seed": 42,
        "num_workers": 0,
        "use_cuda_if_available": True,
    }


def default_aggregator_config() -> dict[str, Any]:
    return {
        "init_strategy": "uniform",
        "mask_diagonal": True,
    }


def default_reranker_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "type": "none",
        "model_name_or_path": None,
        "max_length": 512,
        "batch_size": 8,
        "metadata_score_key": None,
        "trust_remote_code": True,
    }


def default_context_config() -> dict[str, Any]:
    return {
        "include_scores": True,
        "include_metadata": False,
        "include_node_ids": True,
        "context_separator": "\n\n",
        "max_context_chars": None,
    }


def default_generator_config() -> dict[str, Any]:
    return {
        "type": "echo",
        "model_name_or_path": None,
        "max_input_length": 4096,
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "include_system_prompt": True,
        "system_prompt": None,
        "trust_remote_code": True,
    }


def default_loss_config() -> dict[str, Any]:
    return {
        "retrieval_temperature": 0.05,
        "attention_lambda": 0.1,
        "use_in_batch_negatives": False,
        "retrieval_reduction": "mean",
        "attention_reduction": "mean",
        "attention_example_reduction": "mean",
        "allow_self_pairs": True,
        "empty_attention_policy": "zero",
    }


def default_index_config() -> dict[str, Any]:
    return {
        "runtime": default_runtime_config(),
        "encoder": {
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 4096,
            "normalize_embeddings": True,
            "trust_remote_code": True,
        },
        "aggregator": default_aggregator_config(),
        "indexing": {
            "input_path": "data/documents.jsonl",
            "output_dir": "indexes",
            "index_name": "default",
            "max_sentences_per_chunk": 2,
            "batch_size": 1,
            "overwrite": False,
        },
    }


def default_retrieve_config() -> dict[str, Any]:
    return {
        "runtime": default_runtime_config(),
        "encoder": {
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 4096,
            "normalize_embeddings": True,
            "trust_remote_code": True,
        },
        "retrieval": {
            "index_dir": "indexes",
            "index_name": "default",
            "query": "Example query",
            "query_path": None,
            "output_path": None,
            "top_k": 10,
            "per_document_top_k": 5,
            "beam_width": 5,
            "threshold": 0.0,
            "collect_strategy": "threshold",
            "include_root": False,
            "multi_document": True,
        },
        "reranker": default_reranker_config(),
    }


def default_answer_config() -> dict[str, Any]:
    return {
        "runtime": default_runtime_config(),
        "encoder": {
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 4096,
            "normalize_embeddings": True,
            "trust_remote_code": True,
        },
        "retrieval": {
            "index_dir": "indexes",
            "index_name": "default",
            "query": "Example question",
            "query_path": None,
            "output_path": None,
            "top_k": 10,
            "per_document_top_k": 5,
            "beam_width": 5,
            "threshold": 0.0,
            "collect_strategy": "threshold",
            "include_root": False,
            "multi_document": True,
        },
        "reranker": default_reranker_config(),
        "context": default_context_config(),
        "generator": default_generator_config(),
    }


def default_train_config() -> dict[str, Any]:
    return {
        "runtime": default_runtime_config(),
        "encoder": {
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 4096,
            "normalize_embeddings": True,
            "trust_remote_code": True,
        },
        "aggregator": default_aggregator_config(),
        "training": {
            "train_path": "data/train.jsonl",
            "output_dir": "runs",
            "run_name": "default",
            "num_epochs": 3,
            "batch_size": 32,
            "max_negatives": None,
            "require_equal_negatives": True,
            "learning_rate": 2e-5,
            "aggregator_learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.05,
            "gradient_clip_norm": 1.0,
            "checkpoint_every_steps": None,
            "log_every_steps": 1,
            "max_sentences_per_chunk": 2,
            "normalize_passage_embeddings": True,
        },
        "loss": default_loss_config(),
    }


def default_evaluate_retrieval_config() -> dict[str, Any]:
    return {
        "runtime": default_runtime_config(),
        "encoder": {
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 4096,
            "normalize_embeddings": True,
            "trust_remote_code": True,
        },
        "retrieval": {
            "index_dir": "indexes",
            "index_name": "default",
            "query": "Example query",
            "query_path": None,
            "output_path": None,
            "top_k": 10,
            "per_document_top_k": 5,
            "beam_width": 5,
            "threshold": 0.0,
            "collect_strategy": "threshold",
            "include_root": False,
            "multi_document": True,
        },
        "reranker": default_reranker_config(),
        "evaluation": {
            "task": "retrieval",
            "examples_path": "data/retrieval_eval.jsonl",
            "index_dir": "indexes",
            "index_name": "default",
            "output_path": None,
            "ks": [1, 3, 5],
            "include_optional_generation_metrics": False,
            "include_bertscore": False,
            "bertscore_model_type": None,
        },
    }


def default_evaluate_generation_config() -> dict[str, Any]:
    return {
        "runtime": default_runtime_config(),
        "encoder": {
            "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 4096,
            "normalize_embeddings": True,
            "trust_remote_code": True,
        },
        "retrieval": {
            "index_dir": "indexes",
            "index_name": "default",
            "query": "Example query",
            "query_path": None,
            "output_path": None,
            "top_k": 10,
            "per_document_top_k": 5,
            "beam_width": 5,
            "threshold": 0.0,
            "collect_strategy": "threshold",
            "include_root": False,
            "multi_document": True,
        },
        "reranker": default_reranker_config(),
        "context": default_context_config(),
        "generator": default_generator_config(),
        "evaluation": {
            "task": "generation",
            "examples_path": "data/generation_eval.jsonl",
            "index_dir": "indexes",
            "index_name": "default",
            "output_path": None,
            "ks": [1, 3, 5],
            "include_optional_generation_metrics": False,
            "include_bertscore": False,
            "bertscore_model_type": None,
        },
    }


def get_default_config(command: str) -> dict[str, Any]:
    mapping = {
        "index": default_index_config,
        "retrieve": default_retrieve_config,
        "answer": default_answer_config,
        "train": default_train_config,
        "evaluate_retrieval": default_evaluate_retrieval_config,
        "evaluate_generation": default_evaluate_generation_config,
    }
    if command not in mapping:
        raise ValueError("command must be one of: index, retrieve, answer, train, evaluate_retrieval, evaluate_generation")
    return copy.deepcopy(mapping[command]())
