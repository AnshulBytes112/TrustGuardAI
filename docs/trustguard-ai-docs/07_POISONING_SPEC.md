# Text Poisoning Specification

## Purpose

Create controlled poisoned datasets with known ground truth for research evaluation.

## Initial Attack

Start with one simple, reproducible text backdoor attack using a configurable trigger pattern.

The attack implementation must support:
- poison_rate
- trigger
- target_label where applicable
- random_seed

## Ground Truth

For every generated sample:

```text
is_poisoned = true/false
```

This field is hidden from detection.

## Required Metadata

```json
{
  "attack_type": "text_backdoor_v1",
  "poison_rate": 0.05,
  "trigger": "...",
  "target_label": "...",
  "seed": 42
}
```

## Reproducibility

Same:
- source dataset version
- attack configuration
- seed

must generate the same poisoning decision.

## Safety

This module exists only for controlled research datasets. It must not be connected to arbitrary third-party data modification workflows.

## Implementation Decisions

For `text_backdoor_v1`, the following reproducible decisions apply:
1. **Selection Method**: Random sampling initialized with the given `seed` using Python's `random.Random(seed).sample(...)`.
2. **Poison Count Rounding**: Uses exact `math.floor(total_samples * poison_rate)` to compute target poison count.
3. **Trigger Placement**: Appends the trigger to the end of the text, separated by exactly one space: `text + " " + trigger`.
4. **Duplicate Triggers**: If the text already ends with ` " " + trigger` or the trigger exists exactly at the end, it will not append a second trigger.
5. **Dataset Versioning**: Appends `-poisoned` to the dataset version string.
6. **Label Provenance**: For unlabelled samples (or any sample), original label data is stored in `original_label` and `original_label_status` before being overwritten by the `target_label`.

## Reproducibility and Metadata

Each poisoning operation outputs a `PoisoningMetadata` block describing the exact operation performed.
This strictly separates requested configuration from actual results (e.g. `requested_poison_rate` vs `actual_poison_rate` caused by rounding).

### Reproducibility Fingerprint
The system generates a deterministic `sha256` hash (fingerprint) of the experimental setup to identify equivalence. 
The canonical inputs for this hash include:
- `attack_type`
- `input_dataset_id`
- `input_dataset_version`
- `poison_count_policy`
- `poison_rate`
- `seed`
- `selection_method`
- `target_label`
- `trigger`

This is serialized into a sorted JSON string without whitespace separators before hashing.
