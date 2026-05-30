from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def compute_exact_match(
    prediction: str,
    reference: str,
    normalize: bool = True,
) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise ValueError("prediction and reference must be strings")
    if normalize:
        return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0
    return 1.0 if prediction == reference else 0.0


def _require_references(references: list[str]) -> None:
    if not isinstance(references, list) or not references:
        raise ValueError("references must be a non-empty list of strings")
    for ref in references:
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("references must contain non-empty strings")


def compute_max_exact_match(
    prediction: str,
    references: list[str],
    normalize: bool = True,
) -> float:
    _require_references(references)
    return max(compute_exact_match(prediction, ref, normalize=normalize) for ref in references)


def compute_token_f1(
    prediction: str,
    reference: str,
    normalize: bool = True,
) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise ValueError("prediction and reference must be strings")
    if normalize:
        pred_tokens = normalize_answer(prediction).split()
        ref_tokens = normalize_answer(reference).split()
    else:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_max_token_f1(
    prediction: str,
    references: list[str],
    normalize: bool = True,
) -> float:
    _require_references(references)
    return max(compute_token_f1(prediction, ref, normalize=normalize) for ref in references)


def compute_rouge_l(prediction: str, reference: str) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise ValueError("prediction and reference must be strings")
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("rouge_score is required for ROUGE-L") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return float(score["rougeL"].fmeasure)


def compute_max_rouge_l(prediction: str, references: list[str]) -> float:
    _require_references(references)
    return max(compute_rouge_l(prediction, ref) for ref in references)


def compute_meteor(prediction: str, reference: str) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise ValueError("prediction and reference must be strings")
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise ImportError("nltk is required for METEOR") from exc
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    return float(meteor_score([ref_tokens], pred_tokens))


def compute_max_meteor(prediction: str, references: list[str]) -> float:
    _require_references(references)
    return max(compute_meteor(prediction, ref) for ref in references)


def compute_bertscore_f1(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str | None = None,
) -> list[float]:
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("predictions must be a non-empty list of strings")
    if not isinstance(references, list) or not references:
        raise ValueError("references must be a non-empty list of strings")
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not all(isinstance(item, str) for item in predictions):
        raise ValueError("predictions must contain strings")
    if not all(isinstance(item, str) for item in references):
        raise ValueError("references must contain strings")

    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError("bert_score is required for BERTScore") from exc
    _, _, f1 = score(predictions, references, lang=lang, model_type=model_type)
    return [float(value) for value in f1]


def compute_generation_metrics(
    prediction: str,
    references: list[str],
    include_optional: bool = False,
    bertscore_model_type: str | None = None,
) -> dict[str, float]:
    _require_references(references)
    metrics = {
        "exact_match": compute_max_exact_match(prediction, references, normalize=True),
        "token_f1": compute_max_token_f1(prediction, references, normalize=True),
    }
    if include_optional:
        metrics["rouge_l"] = compute_max_rouge_l(prediction, references)
        metrics["meteor"] = compute_max_meteor(prediction, references)
    return metrics


def aggregate_generation_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
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
