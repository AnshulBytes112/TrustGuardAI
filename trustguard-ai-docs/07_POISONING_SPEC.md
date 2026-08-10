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
