# Detection Engine Specification

## Common Interface

```python
class Detector:
    name: str

    def fit(self, features, metadata=None):
        ...

    def score_samples(self, features, metadata=None):
        ...

    def predict(self, features, metadata=None):
        ...
```

## MVP Detectors

### Isolation Forest

Purpose:
- baseline unsupervised anomaly detection.

### K-Means

Purpose:
- representation clustering baseline.

### FLARE-Inspired Detector

Purpose:
- multi-layer representation purification approach.

## Detector Output

```json
{
  "sample_id": "x",
  "detector": "isolation_forest",
  "raw_score": 0.83,
  "normalized_score": 0.91,
  "rank": 17
}
```

## Rules

- No detector may read `poison_ground_truth`.
- All thresholds must be configuration-driven.
- All detector versions must be recorded.
- Detector outputs must be deterministic when the configured seed allows it.
