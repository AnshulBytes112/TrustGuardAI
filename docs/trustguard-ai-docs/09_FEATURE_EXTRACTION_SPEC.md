# Multi-Layer Feature Extraction Specification

## Objective

Extract representations from single or multiple Transformer hidden layers using a unified pipeline.

## Input

- Dataset (`Sequence[Sample]`)
- Tokenizer (`AutoTokenizer`)
- Model checkpoint (`AutoModel`)
- Representation configuration (`RepresentationConfig`)

## Layer Indexing Convention

- Hidden state layer indices follow natural model hidden-state indexing:
  - `0`: Output of the initial embedding layer.
  - `1 .. N`: Outputs of the $1$-th through $N$-th Transformer hidden layers ($N = 6$ for `distilbert-base-uncased`).
- Valid layer indices range from `0` to `N` inclusive.

## Configuration & Validation

Layer selection is configured via `RepresentationConfig.layers: tuple[int, ...] | None`.
- **Validation Rules**:
  - Rejects negative layer indices (`layer < 0`).
  - Rejects non-integer types (`TypeError`).
  - Rejects empty layer tuples (`layers = ()`).
  - Rejects duplicate layer indices (`layers = (2, 2, 6)`).
  - Rejects out-of-range layer indices exceeding `N` (`ValueError`).

## Pooling Strategy

- Applies the configured pooling strategy (`CLS` or masked `mean` pooling) consistently across all requested hidden layers.
- Masked mean pooling includes an explicit attention mask calculation to prevent padding token influence across variable text lengths within batches.

## Output Structure

For every sample set of size $K$ and requested layers:

```python
RepresentationResult(
    sample_ids=[...], # Length K
    representations=..., # Primary/final layer matrix (K, hidden_dim)
    layer_representations={
        0: np.ndarray, # (K, hidden_dim)
        2: np.ndarray, # (K, hidden_dim)
        6: np.ndarray, # (K, hidden_dim)
    },
    model_name="...",
    max_length=...
)
```

Sample IDs and hidden state row orders remain strictly aligned across all extracted layer representations.

## Performance & Memory Safety

- All requested layers are extracted during a single forward pass per batch under `torch.inference_mode()`.
- Intermediate batch hidden states are converted to CPU NumPy arrays immediately to prevent memory growth.

## Leakage Prevention

- Representation extraction operates strictly on sample text content.
- `label`, `label_status`, `poison_ground_truth`, `original_label`, and `original_label_status` are never used as model inputs.

## Reproducibility & Storage

Feature artifacts are versioned by:
- dataset version
- model version
- feature config (including requested layers and pooling strategy)

