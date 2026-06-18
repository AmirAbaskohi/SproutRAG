<div align="center">
  <h1>
    <img src="misc/logo.png" alt="SproutRAG" width="50" style="vertical-align: middle; margin-bottom: 25px;">
    SproutRAG
    <img src="misc/logo.png" alt="SproutRAG" width="50" style="vertical-align: middle; margin-bottom: 25px;">
  </h1>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg">
    <img alt="Transformers" src="https://img.shields.io/badge/transformers-%E2%89%A54.51-yellow.svg">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  </p>
</div>

SproutRAG is a retrieval-augmented generation stack built for structured, multi-granularity evidence. It combines hierarchical attention-based indexing with flexible retrieval, optional reranking, and generation pipelines. The result is a practical system for building and evaluating RAG workflows end-to-end.

<p align="center">
  <img src="misc/method.png" alt="SproutRAG Method" width="900">
</p>

## Quick Start Overview

You will typically follow this flow:

1. Install the package and optional dependencies.
2. Create or modify a config file (YAML or JSON).
3. Index documents with `sproutrag index`.
4. Run retrieval with `sproutrag retrieve`.
5. Generate answers with `sproutrag answer`.
6. Train the model with `sproutrag train` (optional).
7. Evaluate retrieval and generation with `sproutrag evaluate`.

The CLI uses typed configs so you can keep everything reproducible and easy to modify.

## Installation

Install in editable mode:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[yaml]
pip install -e .[eval]
pip install -e .[spacy]
```

Notes:
- YAML configs require PyYAML (`.[yaml]`).
- Evaluation extras enable ROUGE-L, METEOR, and BERTScore (`.[eval]`).
- spaCy improves sentence splitting (`.[spacy]`).

## Config Files

SproutRAG runs from YAML or JSON config files. The CLI expects a config per command. You can start from the examples below and adjust as needed.

### Index Config

```yaml
runtime:
  device: null
  seed: 42
  num_workers: 0
  use_cuda_if_available: true

encoder:
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
  max_length: 4096
  normalize_embeddings: true
  trust_remote_code: true

aggregator:
  init_strategy: uniform
  mask_diagonal: true

indexing:
  input_path: data/documents.jsonl
  output_dir: indexes
  index_name: demo
  max_sentences_per_chunk: 2
  batch_size: 1
  overwrite: true
```

The index input file is JSONL with one document per line:

```json
{"doc_id": "doc-1", "text": "Your document text", "title": "Optional title", "metadata": {"source": "example"}}
```

### Retrieve Config

```yaml
runtime:
  device: null
  seed: 42
  num_workers: 0
  use_cuda_if_available: true

encoder:
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
  max_length: 4096
  normalize_embeddings: true
  trust_remote_code: true

retrieval:
  index_dir: indexes
  index_name: demo
  query: "How does SproutRAG build the retrieval hierarchy?"
  query_path: null
  output_path: outputs/retrieval.json
  top_k: 10
  per_document_top_k: 5
  beam_width: 5
  threshold: 0.0
  collect_strategy: threshold
  include_root: false
  multi_document: true

reranker:
  enabled: false
  type: none
  model_name_or_path: null
  max_length: 512
  batch_size: 8
  metadata_score_key: null
  trust_remote_code: true
```

### Answer Config

```yaml
runtime:
  device: null
  seed: 42
  num_workers: 0
  use_cuda_if_available: true

encoder:
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
  max_length: 4096
  normalize_embeddings: true
  trust_remote_code: true

retrieval:
  index_dir: indexes
  index_name: demo
  query: "How does SproutRAG retrieve multi-granularity evidence?"
  query_path: null
  output_path: outputs/answer.json
  top_k: 10
  per_document_top_k: 5
  beam_width: 5
  threshold: 0.0
  collect_strategy: threshold
  include_root: false
  multi_document: true

reranker:
  enabled: false
  type: none
  model_name_or_path: null
  max_length: 512
  batch_size: 8
  metadata_score_key: null
  trust_remote_code: true

context:
  include_scores: true
  include_metadata: false
  include_node_ids: true
  context_separator: "\n\n"
  max_context_chars: 12000

generator:
  type: echo
  model_name_or_path: null
  max_input_length: 4096
  max_new_tokens: 256
  temperature: 0.0
  do_sample: false
  include_system_prompt: true
  system_prompt: null
  trust_remote_code: true
```

### Train Config

```yaml
runtime:
  device: null
  seed: 42
  num_workers: 0
  use_cuda_if_available: true

encoder:
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
  max_length: 4096
  normalize_embeddings: true
  trust_remote_code: true

aggregator:
  init_strategy: uniform
  mask_diagonal: true

training:
  train_path: data/train.jsonl
  output_dir: runs
  run_name: sproutrag_train
  num_epochs: 3
  batch_size: 32
  max_negatives: null
  require_equal_negatives: true
  learning_rate: 0.00002
  aggregator_learning_rate: 0.001
  weight_decay: 0.0
  warmup_ratio: 0.05
  gradient_clip_norm: 1.0
  checkpoint_every_steps: 100
  log_every_steps: 1
  max_sentences_per_chunk: 2
  normalize_passage_embeddings: true

loss:
  retrieval_temperature: 0.05
  attention_lambda: 0.1
  use_in_batch_negatives: false
  retrieval_reduction: mean
  attention_reduction: mean
  attention_example_reduction: mean
  allow_self_pairs: true
  empty_attention_policy: zero
```

### Evaluate Retrieval Config

```yaml
runtime:
  device: null
  seed: 42
  num_workers: 0
  use_cuda_if_available: true

encoder:
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
  max_length: 4096
  normalize_embeddings: true
  trust_remote_code: true

retrieval:
  index_dir: indexes
  index_name: demo
  query: "dummy"
  query_path: null
  output_path: null
  top_k: 10
  per_document_top_k: 5
  beam_width: 5
  threshold: 0.0
  collect_strategy: threshold
  include_root: false
  multi_document: true

reranker:
  enabled: false
  type: none
  model_name_or_path: null
  max_length: 512
  batch_size: 8
  metadata_score_key: null
  trust_remote_code: true

evaluation:
  task: retrieval
  examples_path: data/retrieval_eval.jsonl
  index_dir: indexes
  index_name: demo
  output_path: outputs/retrieval_eval.json
  ks: [1, 3, 5]
  include_optional_generation_metrics: false
  include_bertscore: false
  bertscore_model_type: null
```

### Evaluate Generation Config

```yaml
runtime:
  device: null
  seed: 42
  num_workers: 0
  use_cuda_if_available: true

encoder:
  model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
  max_length: 4096
  normalize_embeddings: true
  trust_remote_code: true

retrieval:
  index_dir: indexes
  index_name: demo
  query: "dummy"
  query_path: null
  output_path: null
  top_k: 10
  per_document_top_k: 5
  beam_width: 5
  threshold: 0.0
  collect_strategy: threshold
  include_root: false
  multi_document: true

reranker:
  enabled: false
  type: none
  model_name_or_path: null
  max_length: 512
  batch_size: 8
  metadata_score_key: null
  trust_remote_code: true

context:
  include_scores: true
  include_metadata: false
  include_node_ids: true
  context_separator: "\n\n"
  max_context_chars: 12000

generator:
  type: echo
  model_name_or_path: null
  max_input_length: 4096
  max_new_tokens: 256
  temperature: 0.0
  do_sample: false
  include_system_prompt: true
  system_prompt: null
  trust_remote_code: true

evaluation:
  task: generation
  examples_path: data/generation_eval.jsonl
  index_dir: indexes
  index_name: demo
  output_path: outputs/generation_eval.json
  ks: [1, 3, 5]
  include_optional_generation_metrics: false
  include_bertscore: false
  bertscore_model_type: null
```

## Running the CLI

### Index documents

```bash
sproutrag index --config configs/index.yaml
```

### Retrieve results

```bash
sproutrag retrieve --config configs/retrieve.yaml
```

### Answer queries

```bash
sproutrag answer --config configs/answer.yaml
```

### Train

```bash
sproutrag train --config configs/train.yaml
```

### Evaluate retrieval

```bash
sproutrag evaluate --config configs/evaluate_retrieval.yaml
```

### Evaluate generation

```bash
sproutrag evaluate --config configs/evaluate_generation.yaml
```

### Override config values from the CLI

```bash
sproutrag retrieve \
  --config configs/retrieve.yaml \
  --override retrieval.top_k=20 \
  --override retrieval.beam_width=8
```

## Notes

- All outputs are JSON for easy integration with downstream tools.
- Keep configs in version control to make runs reproducible.
- For MS MARCO training, the training pipeline currently loads the first 30k examples from the v2.1 train split.
