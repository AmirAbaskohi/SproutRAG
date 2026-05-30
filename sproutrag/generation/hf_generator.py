from __future__ import annotations

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sproutrag.generation.context_builder import ContextBuilder
from sproutrag.generation.base import BaseGenerator
from sproutrag.generation.prompts import DEFAULT_SYSTEM_PROMPT, build_rag_prompt
from sproutrag.generation.schema import GeneratedAnswer
from sproutrag.retrieval.schema import RetrievalResult


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_finite_number(value: float | int, field_name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")


def _validate_contexts(contexts: list[RetrievalResult]) -> None:
    if not isinstance(contexts, list):
        raise ValueError("contexts must be a list of RetrievalResult")
    if not all(isinstance(item, RetrievalResult) for item in contexts):
        raise ValueError("contexts must contain RetrievalResult instances")


class HuggingFaceGenerator(BaseGenerator):
    def __init__(
        self,
        model_name_or_path: str,
        device: str | torch.device | None = None,
        max_input_length: int = 4096,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        context_builder: ContextBuilder | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        include_system_prompt: bool = True,
        name: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        _require_non_empty_str(model_name_or_path, "model_name_or_path")
        _require_positive_int(max_input_length, "max_input_length")
        _require_positive_int(max_new_tokens, "max_new_tokens")
        _require_finite_number(temperature, "temperature")
        if float(temperature) < 0:
            raise ValueError("temperature must be >= 0")
        _require_bool(do_sample, "do_sample")
        _require_bool(include_system_prompt, "include_system_prompt")

        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.do_sample = do_sample
        self.context_builder = context_builder or ContextBuilder()
        self.system_prompt = system_prompt
        self.include_system_prompt = include_system_prompt
        self._name = name or f"hf_generator:{model_name_or_path}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
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

    def generate_text_from_prompt(self, prompt: str) -> str:
        _require_non_empty_str(prompt, "prompt")
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        with torch.no_grad():
            output_ids = self.model.generate(
                **tokenized,
                **generate_kwargs,
            )
        input_length = tokenized["input_ids"].shape[-1]
        generated_tokens = output_ids[0][input_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text.strip()

    def generate(self, query: str, contexts: list[RetrievalResult]) -> GeneratedAnswer:
        _require_non_empty_str(query, "query")
        _validate_contexts(contexts)
        context_text = self.context_builder.build_context(contexts)
        prompt = build_rag_prompt(
            query,
            context_text,
            system_prompt=self.system_prompt,
            include_system_prompt=self.include_system_prompt,
        )
        answer = self.generate_text_from_prompt(prompt)
        metadata = {
            "generator_name": self.name,
            "num_contexts": len(contexts),
            "max_input_length": self.max_input_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "citations": self.context_builder.build_citations(contexts),
        }
        return GeneratedAnswer(
            query=query,
            answer=answer,
            contexts=list(contexts),
            prompt=prompt,
            metadata=metadata,
        )
