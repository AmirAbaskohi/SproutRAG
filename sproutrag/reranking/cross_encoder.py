from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sproutrag.reranking.base import BaseReranker
from sproutrag.reranking.utils import (
    copy_result_with_score_and_metadata,
    stable_sort_by_reranker_score,
    validate_candidates,
    validate_query,
    validate_top_k,
)
from sproutrag.retrieval.schema import RetrievalResult


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name_or_path: str,
        device: str | torch.device | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        name: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        _require_non_empty_str(model_name_or_path, "model_name_or_path")
        _require_positive_int(max_length, "max_length")
        _require_positive_int(batch_size, "batch_size")
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self._name = name or f"cross_encoder:{model_name_or_path}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()

        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif getattr(self.tokenizer, "unk_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token

    @property
    def name(self) -> str:
        return self._name

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        validate_query(query)
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list")
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("texts must be a list of non-empty strings")

        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            tokenized = self.tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
            with torch.no_grad():
                outputs = self.model(**tokenized)
            logits = outputs.logits
            if logits.ndim == 1:
                batch_scores = logits
            elif logits.ndim == 2:
                if logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits[:, -1]
            else:
                raise ValueError("unexpected logits shape")
            scores.extend(batch_scores.detach().cpu().tolist())
        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        validate_query(query)
        validate_candidates(candidates)
        validate_top_k(top_k)
        if not candidates:
            return []
        texts = [candidate.text for candidate in candidates]
        scores = self.score_pairs(query, texts)
        updated: list[RetrievalResult] = []
        for candidate, score in zip(candidates, scores):
            updated.append(
                copy_result_with_score_and_metadata(
                    candidate,
                    score,
                    {
                        "reranked": True,
                        "reranker_name": self.name,
                        "original_score": candidate.score,
                        "reranker_score": float(score),
                        "reranker_score_source": "cross_encoder",
                    },
                )
            )
        return stable_sort_by_reranker_score(updated, top_k=top_k)
