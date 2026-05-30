from __future__ import annotations

import argparse

from sproutrag.cli.common import add_common_args, print_json, load_documents_jsonl, exit_with_error
from sproutrag.config.loading import load_typed_config
from sproutrag.config.builders import build_encoder, build_index_store, set_random_seed
from sproutrag.indexing.indexer import SproutRAGIndexer
from sproutrag.indexing.manifest import manifest_to_dict


def run_index_command(args: argparse.Namespace) -> int:
    try:
        config = load_typed_config(args.config, command="index", overrides=args.override)
        set_random_seed(config.runtime.seed)
        documents = load_documents_jsonl(config.indexing.input_path)
        if not documents:
            raise ValueError("no documents loaded")
        encoder = build_encoder(config.encoder, config.runtime)
        indexer = SproutRAGIndexer(
            encoder=encoder,
            aggregator=None,
            tree_builder=None,
            max_sentences_per_chunk=config.indexing.max_sentences_per_chunk,
        )
        store = build_index_store(config.indexing.output_dir, config.indexing.index_name)
        if config.indexing.overwrite:
            store.clear()
        manifest = indexer.index_and_save_documents(
            documents,
            store,
            metadata={
                "encoder_model": config.encoder.model_name_or_path,
                "max_sentences_per_chunk": config.indexing.max_sentences_per_chunk,
            },
        )
        print_json(manifest_to_dict(manifest))
        return 0
    except Exception as exc:
        return exit_with_error(str(exc))


def add_index_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("index", help="Index documents from JSONL")
    add_common_args(parser)
    parser.set_defaults(func=run_index_command)
