from __future__ import annotations

import argparse

from sproutrag.cli.common import (
    add_common_args,
    print_json,
    save_generated_answer_json,
    load_query_from_config_or_path,
    exit_with_error,
)
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
from sproutrag.generation.schema import generated_answer_to_dict


def run_answer_command(args: argparse.Namespace) -> int:
    try:
        config = load_typed_config(args.config, command="answer", overrides=args.override)
        set_random_seed(config.runtime.seed)
        query = load_query_from_config_or_path(config.retrieval.query, config.retrieval.query_path)
        encoder = build_encoder(config.encoder, config.runtime)
        reranker = build_reranker(config.reranker, config.runtime)
        retriever = build_retriever(encoder, config.retrieval, reranker=reranker)
        generator = build_generator(config.generator, config.context, config.runtime)
        pipeline = build_pipeline(retriever, generator)
        store = build_index_store(config.retrieval.index_dir, config.retrieval.index_name)
        indexes = store.load_all()
        if config.retrieval.multi_document:
            answer = pipeline.answer(
                query=query,
                indexes=indexes,
                top_k=config.retrieval.top_k,
                per_document_top_k=config.retrieval.per_document_top_k,
                beam_width=config.retrieval.beam_width,
                threshold=config.retrieval.threshold,
            )
        else:
            if len(indexes) != 1:
                raise ValueError("single-document answer requires exactly one index")
            answer = pipeline.answer(
                query=query,
                indexes=indexes[0],
                top_k=config.retrieval.top_k,
                beam_width=config.retrieval.beam_width,
                threshold=config.retrieval.threshold,
            )
        if config.retrieval.output_path:
            save_generated_answer_json(answer, config.retrieval.output_path)
        else:
            print_json(generated_answer_to_dict(answer))
        return 0
    except Exception as exc:
        return exit_with_error(str(exc))


def add_answer_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("answer", help="Generate an answer using the RAG pipeline")
    add_common_args(parser)
    parser.set_defaults(func=run_answer_command)
