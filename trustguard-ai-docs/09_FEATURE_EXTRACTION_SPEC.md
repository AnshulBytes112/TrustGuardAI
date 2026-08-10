# Multi-Layer Feature Extraction Specification

## Objective

Extract representations from multiple Transformer hidden layers.

## Input

- Dataset
- Tokenizer
- Model checkpoint
- layer configuration

## Output

For every sample:

```text
sample_id
layer_id
embedding
model_version
feature_version
```

## Representation

Use a documented pooling strategy such as:
- CLS representation, where appropriate
- masked mean pooling as a configurable alternative

Do not silently mix pooling strategies between experiments.

## Normalization

Normalization must be configurable and recorded.

## Storage

Prefer:
- NumPy/NPZ
- Parquet
- object storage

for large embeddings.

PostgreSQL stores metadata and artifact locations.

## Reproducibility

Feature artifacts must be versioned by:
- dataset version
- model version
- feature config
