from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from sproutrag.cli.common import add_common_args, print_json, exit_with_error
from sproutrag.config.loading import load_typed_config
from sproutrag.config.builders import (
    build_encoder,
    build_aggregator_from_config,
    build_loss,
    build_data_collator,
    resolve_device,
    set_random_seed,
)
from sproutrag.data.schema import RawDocument
from sproutrag.data.preprocessing import chunk_sentences
from sproutrag.encoding.schema import EncodedDocument
from sproutrag.training.dataset import SproutRAGTrainingDataset
from sproutrag.training.model import SproutRAGTrainingModel
from sproutrag.training.trainer import SproutRAGTrainer


def infer_attention_shape_from_dataset(
    encoder: Any,
    dataset: SproutRAGTrainingDataset,
    max_sentences_per_chunk: int,
) -> tuple[int, int]:
    if len(dataset) == 0:
        raise ValueError("dataset must contain at least one example")
    example = dataset[0]
    document = RawDocument(doc_id=example.example_id, text=example.positive_passage, metadata={})
    chunks = chunk_sentences(document, max_sentences_per_chunk=max_sentences_per_chunk)
    encoded: EncodedDocument = encoder.encode_document(chunks, return_attentions=True)
    if encoded.attentions is None:
        raise ValueError("encoder did not return attentions")
    if encoded.attentions.ndim != 4:
        raise ValueError("attentions must be 4-dimensional")
    return int(encoded.attentions.shape[0]), int(encoded.attentions.shape[1])


def build_optimizer_for_training(
    model: SproutRAGTrainingModel,
    encoder_lr: float,
    aggregator_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if not isinstance(model, SproutRAGTrainingModel):
        raise ValueError("model must be a SproutRAGTrainingModel")
    param_groups: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    def _add_params(params, lr):
        trainable = [p for p in params if p.requires_grad]
        if not trainable:
            return
        filtered = []
        for param in trainable:
            if id(param) in used_ids:
                continue
            used_ids.add(id(param))
            filtered.append(param)
        if filtered:
            param_groups.append({"params": filtered, "lr": lr, "weight_decay": weight_decay})

    if isinstance(model.encoder, torch.nn.Module):
        _add_params(model.encoder.parameters(), encoder_lr)

    if isinstance(model.aggregator, torch.nn.Module):
        _add_params(model.aggregator.parameters(), aggregator_lr)

    _add_params(model.parameters(), encoder_lr)

    if not param_groups:
        raise ValueError("no trainable parameters found")

    return torch.optim.AdamW(param_groups)


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> Any | None:
    if warmup_ratio <= 0:
        return None
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _StepLoggingWrapper:
    def __init__(self, logger: Any, log_every_steps: int) -> None:
        self.logger = logger
        self.log_every_steps = log_every_steps

    def log(self, record) -> None:
        if record.step % self.log_every_steps == 0:
            self.logger.log(record)


def run_train_command(args: argparse.Namespace) -> int:
    try:
        config = load_typed_config(args.config, command="train", overrides=args.override)
        set_random_seed(config.runtime.seed)
        dataset = SproutRAGTrainingDataset.from_msmarco_v21(max_examples=30000, split="train")
        collator = build_data_collator(config.training)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.runtime.num_workers,
            collate_fn=collator,
        )
        encoder = build_encoder(config.encoder, config.runtime)
        num_layers, num_heads = infer_attention_shape_from_dataset(
            encoder,
            dataset,
            config.training.max_sentences_per_chunk,
        )
        aggregator = build_aggregator_from_config(config.aggregator, num_layers, num_heads)
        model = SproutRAGTrainingModel(
            encoder=encoder,
            aggregator=aggregator,
            max_sentences_per_chunk=config.training.max_sentences_per_chunk,
            normalize_passage_embeddings=config.training.normalize_passage_embeddings,
        )
        loss_fn = build_loss(config.loss)
        optimizer = build_optimizer_for_training(
            model,
            encoder_lr=config.training.learning_rate,
            aggregator_lr=config.training.aggregator_learning_rate,
            weight_decay=config.training.weight_decay,
        )
        total_steps = len(dataloader) * config.training.num_epochs
        scheduler = build_warmup_scheduler(optimizer, total_steps, config.training.warmup_ratio)

        run_dir = Path(config.training.output_dir) / config.training.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        logger_path = run_dir / "train_log.jsonl"
        checkpoint_dir = run_dir / "checkpoints"

        trainer = SproutRAGTrainer(
            model=model,
            loss_fn=loss_fn,
            train_dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.training.num_epochs,
            gradient_clip_norm=config.training.gradient_clip_norm,
            logger=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_steps=config.training.checkpoint_every_steps,
            device=resolve_device(config.runtime),
        )

        if config.training.log_every_steps >= 1:
            from sproutrag.training.logging import JSONLTrainingLogger

            trainer.logger = _StepLoggingWrapper(
                JSONLTrainingLogger(logger_path),
                config.training.log_every_steps,
            )

        result = trainer.train()
        final_path = run_dir / "final.pt"
        trainer.save_checkpoint(final_path, epoch=config.training.num_epochs)
        print_json(
            {
                "num_epochs": result.num_epochs,
                "global_step": result.global_step,
                "best_loss": result.best_loss,
                "final_metrics": dict(result.final_metrics),
                "metadata": dict(result.metadata),
            }
        )
        return 0
    except Exception as exc:
        return exit_with_error(str(exc))


def add_train_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Train SproutRAG")
    add_common_args(parser)
    parser.set_defaults(func=run_train_command)
