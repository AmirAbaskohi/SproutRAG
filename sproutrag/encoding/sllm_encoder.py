from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from sproutrag.data.schema import SentenceChunk
from sproutrag.encoding.pooling import (
    l2_normalize,
    mean_pool_hidden_states,
    mean_pool_tokens_by_span,
    pool_attentions_by_span,
)
from sproutrag.encoding.schema import EncodedDocument


class SLLMEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str | torch.device | None = None,
        max_length: int = 4096,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        if not model_name_or_path or not isinstance(model_name_or_path, str):
            raise ValueError("model_name_or_path must be a non-empty string")
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()

        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif getattr(self.tokenizer, "unk_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token

    def _validate_chunks(self, chunks: list[SentenceChunk]) -> str:
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a non-empty list")
        if not all(isinstance(chunk, SentenceChunk) for chunk in chunks):
            raise ValueError("chunks must contain SentenceChunk instances")
        doc_id = chunks[0].doc_id
        if any(chunk.doc_id != doc_id for chunk in chunks):
            raise ValueError("all chunks must share the same doc_id")
        return doc_id

    def _build_document_text_and_spans(
        self,
        chunks: list[SentenceChunk],
    ) -> tuple[str, list[tuple[int, int]]]:
        parts = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for index, chunk in enumerate(chunks):
            text = chunk.text
            start = cursor
            end = start + len(text)
            spans.append((start, end))
            parts.append(text)
            cursor = end
            if index < len(chunks) - 1:
                parts.append("\n")
                cursor += 1
        return "".join(parts), spans

    def _offsets_to_token_spans(
        self,
        offset_mapping: torch.Tensor,
        char_spans: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        if offset_mapping.ndim != 3 or offset_mapping.shape[0] != 1:
            raise ValueError("offset_mapping must have shape [1, seq_len, 2]")
        offsets = offset_mapping[0].tolist()
        token_spans: list[tuple[int, int]] = []
        skipped: list[int] = []
        for idx, (char_start, char_end) in enumerate(char_spans):
            token_start = None
            token_end = None
            for token_index, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == 0 and tok_end == 0:
                    continue
                if tok_start < char_end and tok_end > char_start:
                    if token_start is None:
                        token_start = token_index
                    token_end = token_index + 1
            if token_start is None or token_end is None:
                skipped.append(idx)
            else:
                token_spans.append((token_start, token_end))
        return token_spans, skipped

    def encode_query(self, query: str) -> torch.Tensor:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        inputs = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
            pooled = mean_pool_hidden_states(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            if self.normalize_embeddings:
                pooled = l2_normalize(pooled)
        return pooled[0].detach().cpu()

    def encode_document(
        self,
        chunks: list[SentenceChunk],
        return_attentions: bool = True,
    ) -> EncodedDocument:
        doc_id = self._validate_chunks(chunks)
        document_text, char_spans = self._build_document_text_and_spans(chunks)
        tokens = self.tokenizer(
            document_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "offset_mapping" not in tokens:
            raise ValueError("tokenizer must be fast to provide offset_mapping")

        offset_mapping = tokens["offset_mapping"]
        token_spans, skipped_indices = self._offsets_to_token_spans(
            offset_mapping, char_spans
        )
        kept_chunks = [
            chunk
            for idx, chunk in enumerate(chunks)
            if idx not in set(skipped_indices)
        ]
        skipped_chunk_ids = [chunks[idx].chunk_id for idx in skipped_indices]
        if not kept_chunks:
            raise ValueError("all chunks were truncated or produced no tokens")

        inputs = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_attentions=return_attentions,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state[0]
            chunk_embeddings = mean_pool_tokens_by_span(hidden_states, token_spans)
            if self.normalize_embeddings:
                chunk_embeddings = l2_normalize(chunk_embeddings)
            attentions = None
            if return_attentions:
                attentions = pool_attentions_by_span(outputs.attentions, token_spans)

        return EncodedDocument(
            doc_id=doc_id,
            chunk_ids=[chunk.chunk_id for chunk in kept_chunks],
            embeddings=chunk_embeddings.detach().cpu(),
            attentions=attentions.detach().cpu() if attentions is not None else None,
            metadata={
                "model_name_or_path": self.model_name_or_path,
                "max_length": self.max_length,
                "normalize_embeddings": self.normalize_embeddings,
                "num_input_chunks": len(chunks),
                "num_encoded_chunks": len(kept_chunks),
                "skipped_chunk_ids": skipped_chunk_ids,
            },
        )
