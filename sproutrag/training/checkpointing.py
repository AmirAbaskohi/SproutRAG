from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _require_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")


def _require_dict(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_finite_number(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")


@dataclass
class TrainingCheckpointMetadata:
    epoch: int
    global_step: int
    best_loss: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_int(self.epoch, "epoch")
        _require_int(self.global_step, "global_step")
        _require_finite_number(self.best_loss, "best_loss")
        _require_dict(self.metadata, "metadata")


def checkpoint_metadata_to_dict(
    metadata: TrainingCheckpointMetadata,
) -> dict[str, Any]:
    if not isinstance(metadata, TrainingCheckpointMetadata):
        raise ValueError("metadata must be a TrainingCheckpointMetadata")
    return {
        "epoch": metadata.epoch,
        "global_step": metadata.global_step,
        "best_loss": metadata.best_loss,
        "metadata": dict(metadata.metadata),
    }


def checkpoint_metadata_from_dict(
    data: dict[str, Any],
) -> TrainingCheckpointMetadata:
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")
    return TrainingCheckpointMetadata(
        epoch=data["epoch"],
        global_step=data["global_step"],
        best_loss=data.get("best_loss", None),
        metadata=data.get("metadata", {}),
    )


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    metadata: TrainingCheckpointMetadata,
) -> None:
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer must be a torch.optim.Optimizer")
    if scheduler is not None and not hasattr(scheduler, "state_dict"):
        raise ValueError("scheduler must have state_dict")
    if not isinstance(metadata, TrainingCheckpointMetadata):
        raise ValueError("metadata must be a TrainingCheckpointMetadata")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metadata": checkpoint_metadata_to_dict(metadata),
        },
        checkpoint_path,
    )


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = None,
) -> TrainingCheckpointMetadata:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer must be a torch.optim.Optimizer")
    if scheduler is not None and not hasattr(scheduler, "load_state_dict"):
        raise ValueError("scheduler must have load_state_dict")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "metadata",
    }
    missing = required_keys - checkpoint.keys()
    if missing:
        raise ValueError("checkpoint missing required keys")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint_metadata_from_dict(checkpoint["metadata"])
