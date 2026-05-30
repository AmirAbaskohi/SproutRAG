from __future__ import annotations

import argparse

from sproutrag.cli.index import add_index_subcommand
from sproutrag.cli.retrieve import add_retrieve_subcommand
from sproutrag.cli.answer import add_answer_subcommand
from sproutrag.cli.train import add_train_subcommand
from sproutrag.cli.evaluate import add_evaluate_subcommand


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sproutrag")
    subparsers = parser.add_subparsers(dest="command")
    add_index_subcommand(subparsers)
    add_retrieve_subcommand(subparsers)
    add_answer_subcommand(subparsers)
    add_train_subcommand(subparsers)
    add_evaluate_subcommand(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
