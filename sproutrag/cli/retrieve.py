from __future__ import annotations

import argparse

from sproutrag.cli.common import (
    add_common_args,
    print_json,
    save_retrieval_results_json,
    load_query_from_config_or_path,
    exit_with_error,
)
from sproutrag.config.loading import load_typed_config
from sproutrag.config.builders import build_encoder, build_reranker, build_retriever, build_index_store, set_random_seed


def run_retrieve_command(args: argparse.Namespace) -> int:
    try:
        config = load_typed_config(args.config, command="retrieve", overrides=args.override)
        set_random_seed(config.runtime.seed)
        query = load_query_from_config_or_path(config.retrieval.query, config.retrieval.query_path)
        encoder = build_encoder(config.encoder, config.runtime)
        reranker = build_reranker(config.reranker, config.runtime)
        retriever = build_retriever(encoder, config.retrieval, reranker=reranker)
        store = build_index_store(config.retrieval.index_dir, config.retrieval.index_name)
        indexes = store.load_all()
        if config.retrieval.multi_document:
            results = retriever.retrieve(
                query=query,
                indexes=indexes,
                top_k=config.retrieval.top_k,
                per_document_top_k=config.retrieval.per_document_top_k,
                beam_width=config.retrieval.beam_width,
                threshold=config.retrieval.threshold,
            )
        else:
            if len(indexes) != 1:
                raise ValueError("single-document retrieval requires exactly one index")
            results = retriever.retrieve(
                query=query,
                index=indexes[0],
                top_k=config.retrieval.top_k,
                beam_width=config.retrieval.beam_width,
                threshold=config.retrieval.threshold,
            )
        if config.retrieval.output_path:
            save_retrieval_results_json(results, config.retrieval.output_path)
        else:
            print_json({
                "results": [
                    {
                        "node_id": result.node_id,
                        "doc_id": result.doc_id,
                        "text": result.text,
                        "score": result.score,
                        "depth": result.depth,
                        "is_leaf": result.is_leaf,
                        "sentence_chunk_ids": list(result.sentence_chunk_ids),
                        "metadata": dict(result.metadata),
                    }
                    for result in results
                ]
            })
        return 0
    except Exception as exc:
        return exit_with_error(str(exc))


def add_retrieve_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("retrieve", help="Retrieve contexts from an index")
    add_common_args(parser)
    parser.set_defaults(func=run_retrieve_command)
