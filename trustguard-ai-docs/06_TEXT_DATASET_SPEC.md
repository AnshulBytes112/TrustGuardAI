# Text Dataset Specification

## MVP Dataset Requirements

The implementation must support generic CSV/JSONL text datasets.

Minimum fields:
- sample_id or deterministic ID
- text

Optional:
- label
- split

## Internal Canonical Schema

```text
sample_id
text
label
label_status
split
dataset_id
dataset_version
```

## Preprocessing

Keep preprocessing minimal and reproducible:
- Unicode normalization where required
- Missing-text validation
- whitespace normalization where safe
- tokenizer-specific processing through the Transformer tokenizer

Do not remove tokens that may be security-relevant without documenting it.

## Splitting

Never split after poisoning in a way that leaks poisoned examples across train/test unless the experiment explicitly requires it.

Store split assignments.

## Supported Sources

- CSV
- JSONL
- Hugging Face datasets through a dedicated adapter

## Large Dataset Rule

Do not load unnecessarily large datasets entirely into memory. Use streaming/batched processing where appropriate.
