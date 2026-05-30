from __future__ import annotations

import argparse

from sproutrag.cli.common import add_common_args, print_json, exit_with_error
from sproutrag.config.loading import load_typed_config
from sproutrag.config.builders import (
    build_encoder,
    build_reranker,
    build_retriever,
    build_generator,
    build_pipeline,
    build_index_store,
    set_random_seed,
)
from sproutrag.evaluation.io import (
    load_retrieval_examples_jsonl,
    load_generation_examples_jsonl,
)
from sproutrag.evaluation.schema import retrieval_result_to_dict, generation_result_to_dict
from sproutrag.evaluation.retrieval_evaluator import RetrievalEvaluator
from sproutrag.evaluation.generation_evaluator import GenerationEvaluator


def run_evaluate_command(args: argparse.Namespace) -> int:
    try:
        config = load_typed_config(args.config, command="evaluate", overrides=args.override)
        set_random_seed(config.runtime.seed)

        if config.evaluation.task == "retrieval":
            examples = load_retrieval_examples_jsonl(config.evaluation.examples_path)
            encoder = build_encoder(config.encoder, config.runtime)
            reranker = build_reranker(config.reranker, config.runtime)
            retriever = build_retriever(encoder, config.retrieval, reranker=reranker)
            index_dir = config.evaluation.index_dir or config.retrieval.index_dir
            store = build_index_store(index_dir, config.evaluation.index_name)
            indexes = store.load_all()
            evaluator = RetrievalEvaluator(retriever, ks=config.evaluation.ks)
            results, aggregate = evaluator.evaluate(examples, indexes)
            payload = {
                "results": [retrieval_result_to_dict(result) for result in results],
                "aggregate_metrics": aggregate,
            }
        else:
            examples = load_generation_examples_jsonl(config.evaluation.examples_path)
            encoder = build_encoder(config.encoder, config.runtime)
            reranker = build_reranker(config.reranker, config.runtime)
            retriever = build_retriever(encoder, config.retrieval, reranker=reranker)
            generator = build_generator(config.generator, config.context, config.runtime)
            pipeline = build_pipeline(retriever, generator)
            index_dir = config.evaluation.index_dir or config.retrieval.index_dir
            store = build_index_store(index_dir, config.evaluation.index_name)
            indexes = store.load_all()
            evaluator = GenerationEvaluator(
                pipeline,
                include_optional_metrics=config.evaluation.include_optional_generation_metrics,
                include_bertscore=config.evaluation.include_bertscore,
                bertscore_model_type=config.evaluation.bertscore_model_type,
            )
            results, aggregate = evaluator.evaluate(examples, indexes)
            payload = {
                "results": [generation_result_to_dict(result) for result in results],
                "aggregate_metrics": aggregate,
            }

        if config.evaluation.output_path:
            from pathlib import Path
            import json

            path = Path(config.evaluation.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        else:
            print_json(payload)
        return 0
    except Exception as exc:
        return exit_with_error(str(exc))


def add_evaluate_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate retrieval or generation")
    add_common_args(parser)
    parser.set_defaults(func=run_evaluate_command)
