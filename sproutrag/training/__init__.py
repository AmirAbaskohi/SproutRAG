"""Training data utilities for SproutRAG."""

from sproutrag.training.schema import (
    TrainingExample,
    TrainingBatch,
    training_example_to_dict,
    training_example_from_dict,
    training_batch_from_examples,
)
from sproutrag.training.io import (
    load_training_examples_jsonl,
    save_training_examples_jsonl,
    load_training_examples_json,
    save_training_examples_json,
)
from sproutrag.training.dataset import SproutRAGTrainingDataset
from sproutrag.training.collator import SproutRAGDataCollator
from sproutrag.training.embedding import (
    validate_embedding_tensor,
    l2_normalize_embeddings,
    mean_pool_sequence_embeddings,
    mean_pool_sentence_embeddings,
    cosine_similarity_matrix,
    pairwise_cosine_similarity,
)
from sproutrag.training.losses import (
    validate_temperature,
    contrastive_retrieval_loss,
    in_batch_contrastive_retrieval_loss,
    ContrastiveRetrievalLoss,
    validate_support_pairs,
    mutual_attention_from_aggregated,
    attention_structure_loss,
    batched_attention_structure_loss,
    AttentionStructureLoss,
    validate_attention_lambda,
    validate_support_pairs_batch,
    JointSproutRAGLoss,
)
from sproutrag.training.outputs import (
    SproutRAGTrainingOutput,
    JointLossOutput,
    move_training_output,
    joint_loss_output_to_dict,
)
from sproutrag.training.model import (
    validate_encoder_for_training,
    validate_training_aggregator,
    passage_to_document,
    filter_support_pairs_for_encoded_chunks,
    pad_attention_matrices,
    stack_embeddings,
    stack_negative_embeddings,
    SproutRAGTrainingModel,
)
from sproutrag.training.logging import (
    TrainingLogRecord,
    training_log_record_to_dict,
    JSONLTrainingLogger,
)
from sproutrag.training.checkpointing import (
    TrainingCheckpointMetadata,
    checkpoint_metadata_to_dict,
    checkpoint_metadata_from_dict,
    save_training_checkpoint,
    load_training_checkpoint,
)
from sproutrag.training.trainer import (
    TrainingRunResult,
    SproutRAGTrainer,
)

__all__ = [
    "TrainingExample",
    "TrainingBatch",
    "training_example_to_dict",
    "training_example_from_dict",
    "training_batch_from_examples",
    "load_training_examples_jsonl",
    "save_training_examples_jsonl",
    "load_training_examples_json",
    "save_training_examples_json",
    "SproutRAGTrainingDataset",
    "SproutRAGDataCollator",
    "validate_embedding_tensor",
    "l2_normalize_embeddings",
    "mean_pool_sequence_embeddings",
    "mean_pool_sentence_embeddings",
    "cosine_similarity_matrix",
    "pairwise_cosine_similarity",
    "validate_temperature",
    "contrastive_retrieval_loss",
    "in_batch_contrastive_retrieval_loss",
    "ContrastiveRetrievalLoss",
    "validate_support_pairs",
    "mutual_attention_from_aggregated",
    "attention_structure_loss",
    "batched_attention_structure_loss",
    "AttentionStructureLoss",
    "SproutRAGTrainingOutput",
    "JointLossOutput",
    "move_training_output",
    "joint_loss_output_to_dict",
    "validate_attention_lambda",
    "validate_support_pairs_batch",
    "JointSproutRAGLoss",
    "validate_encoder_for_training",
    "validate_training_aggregator",
    "passage_to_document",
    "filter_support_pairs_for_encoded_chunks",
    "pad_attention_matrices",
    "stack_embeddings",
    "stack_negative_embeddings",
    "SproutRAGTrainingModel",
    "TrainingLogRecord",
    "training_log_record_to_dict",
    "JSONLTrainingLogger",
    "TrainingCheckpointMetadata",
    "checkpoint_metadata_to_dict",
    "checkpoint_metadata_from_dict",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "TrainingRunResult",
    "SproutRAGTrainer",
]
