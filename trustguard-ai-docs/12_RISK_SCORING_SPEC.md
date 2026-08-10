# Risk Scoring Specification

## Goal

Convert multiple evidence sources into a human-readable per-sample suspiciousness score.

## Initial Evidence

- Isolation Forest normalized anomaly score
- Clustering evidence
- Layer-wise anomaly evidence

## Output

```text
risk_score: 0.00–1.00
risk_level: LOW | MEDIUM | HIGH
```

## Initial Configurable Formula

Use a weighted combination:

```text
risk =
  w_detector * detector_score
+ w_cluster  * cluster_score
+ w_layer    * layer_score
```

Weights must be configuration-driven and sum to 1.

Do not claim that the initial weights are scientifically optimal.

## Thresholds

Keep thresholds configurable.

Example only:
- LOW < 0.40
- MEDIUM 0.40–0.70
- HIGH > 0.70

These values are engineering defaults and must be validated experimentally.

## Auditability

Store the component scores used to generate every final score.
