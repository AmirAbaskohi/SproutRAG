from __future__ import annotations

from typing import Any

from sproutrag.data.schema import DocumentIndex
from sproutrag.evaluation.generation_metrics import (
    compute_generation_metrics,
    compute_bertscore_f1,
    aggregate_generation_metric_dicts,
)
from sproutrag.evaluation.schema import GenerationEvaluationExample, GenerationEvaluationResult
from sproutrag.generation.schema import GeneratedAnswer
from sproutrag.indexing.store import IndexStore


def _validate_indexes(indexes: list[DocumentIndex]) -> None:
    if not isinstance(indexes, list) or not indexes:
        raise ValueError("indexes must be a non-empty list of DocumentIndex")
    if not all(isinstance(item, DocumentIndex) for item in indexes):
        raise ValueError("indexes must contain DocumentIndex instances")


class GenerationEvaluator:
    def __init__(
        self,
        pipeline: Any,
        include_optional_metrics: bool = False,
        include_bertscore: bool = False,
        bertscore_model_type: str | None = None,
    ) -> None:
        if not hasattr(pipeline, "answer"):
            raise ValueError("pipeline must implement answer")
        if not isinstance(include_optional_metrics, bool):
            raise ValueError("include_optional_metrics must be a boolean")
        if not isinstance(include_bertscore, bool):
            raise ValueError("include_bertscore must be a boolean")
        self.pipeline = pipeline
        self.include_optional_metrics = include_optional_metrics
        self.include_bertscore = include_bertscore
        self.bertscore_model_type = bertscore_model_type

    def evaluate_example(
        self,
        example: GenerationEvaluationExample,
        indexes: DocumentIndex | list[DocumentIndex],
        **pipeline_kwargs: Any,
    ) -> GenerationEvaluationResult:
        if not isinstance(example, GenerationEvaluationExample):
            raise ValueError("example must be a GenerationEvaluationExample")

        if isinstance(indexes, list):
            _validate_indexes(indexes)
        elif not isinstance(indexes, DocumentIndex):
            raise ValueError("indexes must be a DocumentIndex or list of DocumentIndex")

        generated = self.pipeline.answer(query=example.query, indexes=indexes, **pipeline_kwargs)
        if not isinstance(generated, GeneratedAnswer):
            raise ValueError("pipeline.answer must return a GeneratedAnswer")

        prediction = generated.answer
        metrics = compute_generation_metrics(
            prediction,
            example.references,
            include_optional=self.include_optional_metrics,
            bertscore_model_type=self.bertscore_model_type,
        )
        return GenerationEvaluationResult(
            example_id=example.example_id,
            query=example.query,
            prediction=prediction,
            references=list(example.references),
            metrics=metrics,
            metadata={
                "generated_metadata": dict(generated.metadata),
                "example_metadata": dict(example.metadata),
            },
        )

    def evaluate(
        self,
        examples: list[GenerationEvaluationExample],
        indexes: DocumentIndex | list[DocumentIndex],
        **pipeline_kwargs: Any,
    ) -> tuple[list[GenerationEvaluationResult], dict[str, float]]:
        if not isinstance(examples, list) or not examples:
            raise ValueError("examples must be a non-empty list of GenerationEvaluationExample")
        if not all(isinstance(item, GenerationEvaluationExample) for item in examples):
            raise ValueError("examples must contain GenerationEvaluationExample instances")

        results: list[GenerationEvaluationResult] = []
        predictions: list[str] = []
        references: list[str] = []
        for example in examples:
            result = self.evaluate_example(example, indexes, **pipeline_kwargs)
            results.append(result)
            predictions.append(result.prediction)
            references.append(example.references[0])

        if self.include_bertscore:
            scores = compute_bertscore_f1(
                predictions,
                references,
                model_type=self.bertscore_model_type,
            )
            for result, score in zip(results, scores):
                result.metrics["bertscore_f1"] = float(score)

        aggregate = aggregate_generation_metric_dicts([result.metrics for result in results])
        return results, aggregate

    def evaluate_from_store(
        self,
        examples: list[GenerationEvaluationExample],
        store: IndexStore,
        **pipeline_kwargs: Any,
    ) -> tuple[list[GenerationEvaluationResult], dict[str, float]]:
        if not isinstance(store, IndexStore):
            raise ValueError("store must be an IndexStore")
        indexes = store.load_all()
        return self.evaluate(examples, indexes, **pipeline_kwargs)
