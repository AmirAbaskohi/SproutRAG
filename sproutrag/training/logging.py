from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")


def _require_dict(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")


def _require_metric_dict(metrics: dict[str, float]) -> None:
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric names must be non-empty strings")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("metric values must be finite numbers")


@dataclass
class TrainingLogRecord:
    step: int
    epoch: int
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_int(self.step, "step")
        _require_int(self.epoch, "epoch")
        _require_metric_dict(self.metrics)
        _require_dict(self.metadata, "metadata")


def training_log_record_to_dict(record: TrainingLogRecord) -> dict[str, Any]:
    if not isinstance(record, TrainingLogRecord):
        raise ValueError("record must be a TrainingLogRecord")
    return {
        "step": record.step,
        "epoch": record.epoch,
        "metrics": dict(record.metrics),
        "metadata": dict(record.metadata),
    }


class JSONLTrainingLogger:
    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise ValueError("path must be a string or Path")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: TrainingLogRecord) -> None:
        if not isinstance(record, TrainingLogRecord):
            raise ValueError("record must be a TrainingLogRecord")
        payload = training_log_record_to_dict(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    def read_all(self) -> list[TrainingLogRecord]:
        if not self.path.exists():
            return []
        records: list[TrainingLogRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}") from exc
                try:
                    record = TrainingLogRecord(
                        step=data["step"],
                        epoch=data["epoch"],
                        metrics=data["metrics"],
                        metadata=data.get("metadata", {}),
                    )
                except Exception as exc:
                    raise ValueError(f"invalid record on line {line_number}") from exc
                records.append(record)
        return records
