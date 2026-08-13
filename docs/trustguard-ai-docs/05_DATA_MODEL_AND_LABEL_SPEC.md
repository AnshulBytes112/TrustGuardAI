# Data Model and Label Availability Specification

## Label Availability Modes

```text
FULLY_LABELLED
PARTIALLY_LABELLED
UNLABELLED
```

## Sample Contract

```json
{
  "sample_id": "stable-id",
  "text": "sample text",
  "label": null,
  "label_status": "UNKNOWN",
  "original_label": null,
  "original_label_status": null,
  "poison_ground_truth": null,
  "dataset_version": "v1"
}
```

## Rules

1. `label` is the task/class label.
2. `poison_ground_truth` is a research/evaluation field.
3. The detector must never consume `poison_ground_truth`.
4. `poison_ground_truth` may be hidden during evaluation.
5. Missing class labels must not be treated as poison.
6. A known class label must not be treated as evidence of cleanliness.
7. Partially-labelled datasets must be processed without forcing pseudo-labels into the canonical ground-truth field.

## Evaluation

Controlled experiments may maintain hidden poison ground truth:

```text
detector input:
text + model representation

evaluation input:
detector output + hidden poison ground truth
```

## Dataset Version

Every dataset must have:
- dataset_id
- version
- creation timestamp
- source
- preprocessing version
- poisoning configuration if applicable
- random seed
