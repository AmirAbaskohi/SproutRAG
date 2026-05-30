from __future__ import annotations

import math
from typing import Iterable


def validate_id_list(ids: list[str], name: str, allow_empty: bool = False) -> None:
    if not isinstance(ids, list) or (not ids and not allow_empty):
        raise ValueError(f"{name} must be a list of strings")
    for item in ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain non-empty strings")


def unique_preserve_order(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        unique_ids.append(item)
    return unique_ids


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def compute_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    _require_positive_int(k, "k")
    validate_id_list(retrieved_ids, "retrieved_ids", allow_empty=True)
    validate_id_list(relevant_ids, "relevant_ids", allow_empty=False)

    unique_retrieved = unique_preserve_order(retrieved_ids)[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in unique_retrieved if item in relevant_set)
    return hits / len(relevant_set)


def compute_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    _require_positive_int(k, "k")
    validate_id_list(retrieved_ids, "retrieved_ids", allow_empty=True)
    validate_id_list(relevant_ids, "relevant_ids", allow_empty=False)

    unique_retrieved = unique_preserve_order(retrieved_ids)[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in unique_retrieved if item in relevant_set)
    return hits / k


def compute_hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    _require_positive_int(k, "k")
    validate_id_list(retrieved_ids, "retrieved_ids", allow_empty=True)
    validate_id_list(relevant_ids, "relevant_ids", allow_empty=False)

    unique_retrieved = unique_preserve_order(retrieved_ids)[:k]
    relevant_set = set(relevant_ids)
    return 1.0 if any(item in relevant_set for item in unique_retrieved) else 0.0


def compute_information_efficiency(recall: float, precision: float) -> float:
    for name, value in ("recall", recall), ("precision", precision):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
        if float(value) < 0 or float(value) > 1:
            raise ValueError(f"{name} must be in [0, 1]")
    return float(recall) * float(precision)


def compute_retrieval_metrics_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> dict[str, float]:
    recall = compute_recall_at_k(retrieved_ids, relevant_ids, k)
    precision = compute_precision_at_k(retrieved_ids, relevant_ids, k)
    hit = compute_hit_at_k(retrieved_ids, relevant_ids, k)
    return {
        f"recall@{k}": recall,
        f"precision@{k}": precision,
        f"hit@{k}": hit,
        f"ie@{k}": compute_information_efficiency(recall, precision),
    }


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    ks: list[int] = [1, 3, 5],
) -> dict[str, float]:
    if not isinstance(ks, list) or not ks:
        raise ValueError("ks must be a non-empty list of integers")
    for value in ks:
        _require_positive_int(value, "k")

    metrics: dict[str, float] = {}
    for k in ks:
        metrics.update(compute_retrieval_metrics_at_k(retrieved_ids, relevant_ids, k))

    recall_avg = sum(metrics[f"recall@{k}"] for k in ks) / len(ks)
    precision_avg = sum(metrics[f"precision@{k}"] for k in ks) / len(ks)
    hit_avg = sum(metrics[f"hit@{k}"] for k in ks) / len(ks)
    ie_avg = sum(metrics[f"ie@{k}"] for k in ks) / len(ks)

    metrics.update(
        {
            "recall_avg": recall_avg,
            "precision_avg": precision_avg,
            "hit_avg": hit_avg,
            "ie_avg": ie_avg,
        }
    )
    return metrics


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not isinstance(metric_dicts, list) or not metric_dicts:
        raise ValueError("metric_dicts must be a non-empty list of dictionaries")
    keys = None
    totals: dict[str, float] = {}
    for metrics in metric_dicts:
        if not isinstance(metrics, dict):
            raise ValueError("metric_dicts must contain dictionaries")
        if keys is None:
            keys = set(metrics.keys())
            totals = {key: 0.0 for key in keys}
        if set(metrics.keys()) != keys:
            raise ValueError("all metric dictionaries must have the same keys")
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("metrics must contain finite numbers")
            totals[key] += float(value)

    count = float(len(metric_dicts))
    return {key: totals[key] / count for key in totals}
