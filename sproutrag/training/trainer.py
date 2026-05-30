from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from sproutrag.training.checkpointing import (
    TrainingCheckpointMetadata,
    save_training_checkpoint,
    load_training_checkpoint,
)
from sproutrag.training.logging import JSONLTrainingLogger, TrainingLogRecord
from sproutrag.training.losses import JointSproutRAGLoss
from sproutrag.training.model import SproutRAGTrainingModel
from sproutrag.training.outputs import move_training_output
from sproutrag.training.schema import TrainingBatch


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")


def _require_finite_number(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")


def _require_metric_dict(metrics: dict[str, float], field_name: str) -> None:
    if not isinstance(metrics, dict):
        raise ValueError(f"{field_name} must be a dictionary")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric names must be non-empty strings")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("metric values must be finite numbers")


@dataclass
class TrainingRunResult:
    num_epochs: int
    global_step: int
    best_loss: float | None
    final_metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_negative_int(self.num_epochs, "num_epochs")
        _require_non_negative_int(self.global_step, "global_step")
        _require_finite_number(self.best_loss, "best_loss")
        _require_metric_dict(self.final_metrics, "final_metrics")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")


class SproutRAGTrainer:
    def __init__(
        self,
        model: SproutRAGTrainingModel,
        loss_fn: JointSproutRAGLoss,
        train_dataloader: Iterable[TrainingBatch],
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        num_epochs: int = 1,
        gradient_clip_norm: float | None = None,
        logger: JSONLTrainingLogger | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every_steps: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(model, SproutRAGTrainingModel):
            raise ValueError("model must be a SproutRAGTrainingModel")
        if not isinstance(loss_fn, JointSproutRAGLoss):
            raise ValueError("loss_fn must be a JointSproutRAGLoss")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be a torch.optim.Optimizer")
        if scheduler is not None and not hasattr(scheduler, "step"):
            raise ValueError("scheduler must have step")
        _require_positive_int(num_epochs, "num_epochs")
        if gradient_clip_norm is not None:
            _require_finite_number(gradient_clip_norm, "gradient_clip_norm")
            if gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be > 0")
        if logger is not None and not isinstance(logger, JSONLTrainingLogger):
            raise ValueError("logger must be a JSONLTrainingLogger")
        if checkpoint_every_steps is not None:
            _require_positive_int(checkpoint_every_steps, "checkpoint_every_steps")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_every_steps = checkpoint_every_steps
        self.device = device

        self.model.to(self.device)
        self.global_step = 0
        self.best_loss: float | None = None

    def train(self) -> TrainingRunResult:
        last_metrics: dict[str, float] | None = None
        for epoch in range(1, self.num_epochs + 1):
            last_metrics = self.train_epoch(epoch)
        if last_metrics is None:
            raise ValueError("training produced no metrics")
        return TrainingRunResult(
            num_epochs=self.num_epochs,
            global_step=self.global_step,
            best_loss=self.best_loss,
            final_metrics=last_metrics,
            metadata={
                "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir is not None else None,
                "gradient_clip_norm": self.gradient_clip_norm,
            },
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        _require_positive_int(epoch, "epoch")
        self.model.train()

        total_loss = 0.0
        total_retrieval = 0.0
        total_attention = 0.0
        num_steps = 0

        for batch in self.train_dataloader:
            if not isinstance(batch, TrainingBatch):
                raise ValueError("train_dataloader must yield TrainingBatch")

            self.optimizer.zero_grad(set_to_none=True)
            model_output, remapped_support_pairs = self.model(batch)
            model_output = move_training_output(model_output, self.device)
            loss_output = self.loss_fn(model_output, support_pairs=remapped_support_pairs)
            loss_output.loss.backward()

            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            loss_value = float(loss_output.loss.detach().cpu())
            retrieval_value = float(loss_output.retrieval_loss.detach().cpu())
            attention_value = float(loss_output.attention_loss.detach().cpu())

            if self.best_loss is None or loss_value < self.best_loss:
                self.best_loss = loss_value

            total_loss += loss_value
            total_retrieval += retrieval_value
            total_attention += attention_value
            num_steps += 1

            metrics = {
                "loss": loss_value,
                "retrieval_loss": retrieval_value,
                "attention_loss": attention_value,
            }
            self._log_step(epoch, metrics)
            self._maybe_save_step_checkpoint(epoch)

        if num_steps == 0:
            raise ValueError("train_dataloader produced no batches")

        return {
            "loss": total_loss / num_steps,
            "retrieval_loss": total_retrieval / num_steps,
            "attention_loss": total_attention / num_steps,
        }

    def save_checkpoint(self, path: str | Path, epoch: int = 0) -> None:
        save_training_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metadata=TrainingCheckpointMetadata(
                epoch=epoch,
                global_step=self.global_step,
                best_loss=self.best_loss,
                metadata={
                    "num_epochs": self.num_epochs,
                    "device": str(self.device),
                },
            ),
        )

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> TrainingCheckpointMetadata:
        metadata = load_training_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=map_location,
        )
        self.global_step = metadata.global_step
        self.best_loss = metadata.best_loss
        return metadata

    def _maybe_save_step_checkpoint(self, epoch: int) -> None:
        if self.checkpoint_dir is None or self.checkpoint_every_steps is None:
            return
        if self.global_step % self.checkpoint_every_steps != 0:
            return
        path = self.checkpoint_dir / f"step_{self.global_step}.pt"
        self.save_checkpoint(path, epoch=epoch)

    def _log_step(
        self,
        epoch: int,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.logger is None:
            return
        record = TrainingLogRecord(
            step=self.global_step,
            epoch=epoch,
            metrics=metrics,
            metadata=metadata or {},
        )
        self.logger.log(record)
