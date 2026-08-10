# Purification Specification

## Workflow

```text
Original Dataset
      ↓
Detection
      ↓
Risk Ranking
      ↓
Review / Threshold
      ↓
Quarantine
      ↓
Purified Dataset
```

## Sample States

```text
ACTIVE
QUARANTINED
RESTORED
```

## Rules

- Original dataset is immutable.
- Quarantine is reversible.
- Purified datasets are new dataset versions.
- Every purification run records the threshold/configuration.
- The system must preserve a mapping from original sample_id to purified dataset version.

## Manual Review

High-risk samples should be reviewable before final purification in the dashboard.

## Retraining

Purified dataset becomes an explicit training-data version.
