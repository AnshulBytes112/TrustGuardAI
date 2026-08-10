# Product Requirements Document

## Product

TrustGuard AI

## Problem

Training datasets may contain suspicious or poisoned samples. A model can learn undesirable hidden behavior from such data even when normal validation accuracy appears acceptable.

## Target User

Primary:
- ML/security researcher
- Dataset curator
- Project reviewer

## Core User Journey

1. Select/import text dataset.
2. Specify label availability.
3. Run controlled experiment or scan.
4. Train/load model.
5. Extract representations.
6. Run detection.
7. Review ranked suspicious samples.
8. Inspect evidence.
9. Quarantine samples.
10. Generate purified dataset.
11. Retrain.
12. Compare security metrics.

## Functional Requirements

FR-001 Dataset import  
FR-002 Labelled/partially-labelled/unlabelled support  
FR-003 Dataset versioning  
FR-004 Controlled poisoning for experiments  
FR-005 Transformer model training/loading  
FR-006 Multi-layer representation extraction  
FR-007 Anomaly detection  
FR-008 Clustering detection  
FR-009 FLARE-inspired detection  
FR-010 Per-sample risk scoring  
FR-011 Explainability  
FR-012 Sample quarantine  
FR-013 Sample restore  
FR-014 Purified dataset generation  
FR-015 Retraining  
FR-016 Metric evaluation  
FR-017 Experiment comparison  
FR-018 Dashboard visualization

## Non-Goals

- Production-grade universal malware/data-security certification
- Automatic claim that a sample is certainly malicious
- Full multimodal implementation in MVP
- GNN/GAT in the MVP
- Large-scale distributed training in MVP
- Real-time streaming ingestion in MVP

## Success Criteria

A complete controlled experiment can be executed from dataset creation through detection, purification, retraining and evaluation, with every result traceable to configuration and seed.
