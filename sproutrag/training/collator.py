from __future__ import annotations

from sproutrag.training.schema import TrainingBatch, TrainingExample, training_batch_from_examples


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_examples(examples: list[TrainingExample]) -> None:
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a non-empty list of TrainingExample")
    if not all(isinstance(item, TrainingExample) for item in examples):
        raise ValueError("examples must contain TrainingExample instances")


class SproutRAGDataCollator:
    def __init__(
        self,
        max_negatives: int | None = None,
        require_equal_negatives: bool = False,
    ) -> None:
        if max_negatives is not None:
            _require_positive_int(max_negatives, "max_negatives")
        if not isinstance(require_equal_negatives, bool):
            raise ValueError("require_equal_negatives must be a boolean")
        self.max_negatives = max_negatives
        self.require_equal_negatives = require_equal_negatives

    def __call__(self, examples: list[TrainingExample]) -> TrainingBatch:
        _validate_examples(examples)
        processed: list[TrainingExample] = []
        for example in examples:
            if self.max_negatives is None:
                negatives = list(example.negative_passages)
            else:
                negatives = list(example.negative_passages[: self.max_negatives])
            processed.append(
                TrainingExample(
                    example_id=example.example_id,
                    query=example.query,
                    positive_passage=example.positive_passage,
                    negative_passages=negatives,
                    support_sentence_pairs=list(example.support_sentence_pairs),
                    metadata=dict(example.metadata),
                )
            )

        if self.require_equal_negatives:
            counts = {len(ex.negative_passages) for ex in processed}
            if len(counts) != 1:
                raise ValueError("all examples must have the same number of negatives")

        return training_batch_from_examples(processed)
