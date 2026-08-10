# Experiment Configuration Example

```yaml
experiment:
  name: badnet_text_mvp
  seed: 42

dataset:
  source: "path/or/hf/name"
  modality: text
  label_mode: partially_labelled
  train_ratio: 0.8
  validation_ratio: 0.1
  test_ratio: 0.1

poisoning:
  enabled: true
  attack_type: text_backdoor_v1
  poison_rate: 0.05
  trigger: "<TRIGGER>"
  target_label: "negative"

model:
  name: distilbert-base-uncased
  max_length: 256
  pooling: cls

features:
  layers: [-4, -3, -2, -1]
  normalize: true

detector:
  primary: isolation_forest
  contamination: auto

risk:
  weights:
    detector: 0.5
    cluster: 0.2
    layer: 0.3
  thresholds:
    medium: 0.4
    high: 0.7

purification:
  default_action: quarantine

evaluation:
  metrics:
    - clean_accuracy
    - attack_success_rate
    - precision
    - recall
    - f1
    - false_positive_rate
    - poison_detection_rate
```

This is a starting configuration, not a scientifically validated final configuration.
