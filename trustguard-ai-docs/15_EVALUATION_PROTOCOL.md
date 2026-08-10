# Evaluation Protocol

## Metrics

Required:
- Clean Accuracy (CA)
- Attack Success Rate (ASR)
- Precision
- Recall
- F1
- False Positive Rate (FPR)
- Poison Detection Rate

## Before/After Evaluation

### Stage A
Train/evaluate on poisoned data.

### Stage B
Run TrustGuard.

### Stage C
Create purified dataset.

### Stage D
Retrain.

### Stage E
Compare:
- clean performance
- attack success
- detection quality

## Critical Rule

Never invent or manually type experimental results into reports.

Reports must be generated from stored experiment results.

## Evaluation Ground Truth

In controlled experiments:
- poison_ground_truth is hidden during detection
- revealed only to the evaluation engine

## Repeated Runs

Use configurable seeds and multiple runs for final reported experiments where compute permits.

## Required Output

Each experiment produces:
- configuration
- metrics JSON
- predictions
- confusion matrix data
- model artifact references
- feature artifact references
- report
