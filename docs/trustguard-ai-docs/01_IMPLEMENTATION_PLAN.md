# TrustGuard AI — Final Phasewise Implementation Plan

## Phase 0 — Documentation and Architecture

Deliverables:
- Repository
- `/docs`
- Coding rules
- Architecture
- Data contracts
- Experiment protocol
- Task backlog
- Definition of Done

Exit criterion:
- AI coding agent can implement tasks without making architectural decisions.

## Phase 1 — Data Layer

Implement:
- Text dataset adapter
- Labelled/partially-labelled/unlabelled representation
- Dataset versioning metadata
- Train/validation/test split handling
- Sample IDs
- Poison ground-truth separation

Exit criterion:
- Same dataset pipeline works with labels present, labels absent, or labels partially present.

## Phase 2 — Controlled Poisoning

Start with one reproducible backdoor attack suitable for text.

Implement:
- Trigger configuration
- Target-label configuration where applicable
- Poison rate
- Deterministic seed
- Poison metadata
- Clean/poisoned dataset generation

Exit criterion:
- Controlled experiments can reproduce the same poisoned dataset from the same configuration.

## Phase 3 — Baseline Transformer

Implement:
- DistilBERT loading
- Tokenization
- Fine-tuning
- Checkpoint saving
- Evaluation
- Inference
- Hidden-layer extraction

Exit criterion:
- Model trains and produces stable representations.

## Phase 4 — Multi-Layer Feature Extraction

Extract representations from configurable hidden layers.

Store:
- sample_id
- layer
- embedding
- model version
- dataset version

Exit criterion:
- Every sample has traceable multi-layer representations.

## Phase 5 — Baseline Detection

Implement:
1. Isolation Forest
2. K-Means

Both must conform to a common detector interface.

Exit criterion:
- Detector produces a score/rank for every sample without using poison ground truth.

## Phase 6 — FLARE-Inspired Detection

Implement the project's adaptation of:
- Multi-layer representation analysis
- Abnormal activation/representation aggregation
- Dimensionality/subspace handling
- Clustering-based suspicious group identification

Do not claim exact reproduction of the paper unless independently verified.

Exit criterion:
- FLARE-inspired detector can be compared against baseline detectors.

## Phase 7 — Risk Scoring and Explainability

Generate:
- 0–1 risk score
- Low/Medium/High risk level
- Layer-wise evidence
- Detector evidence
- Cluster evidence
- Human-readable explanation

Exit criterion:
- A reviewer can understand why a sample was flagged.

## Phase 8 — Purification

Implement:
- Quarantine
- Restore
- Purified dataset generation
- Audit trail

Never permanently delete samples in the MVP.

Exit criterion:
- A suspicious subset can be excluded from retraining while preserving original data.

## Phase 9 — Retraining and Evaluation

Run:
- Original model
- Poisoned model
- Purified retrained model

Measure:
- Clean Accuracy
- Attack Success Rate
- Precision
- Recall
- F1
- False Positive Rate
- Poison Detection Rate

Exit criterion:
- Before/after report generated automatically.

## Phase 10 — Backend

FastAPI endpoints for:
- datasets
- experiments
- scans
- samples
- risk scores
- quarantine
- purification
- retraining
- metrics

Exit criterion:
- UI can operate the complete pipeline through APIs.

## Phase 11 — Dashboard

Pages:
1. Overview
2. Dataset
3. Scan
4. Suspicious Samples
5. Sample Investigation
6. Purification
7. Retraining
8. Experiment Comparison

Exit criterion:
- Complete end-to-end demo is possible without notebooks.

## Phase 12 — Research Experiments

Run controlled experiments across:
- poison rates
- attack configurations
- label availability
- detector
- model
- random seeds

Exit criterion:
- All reported numbers come from reproducible experiment runs.

## Week-1 Priority

Day 1: repository + environment + dataset adapter  
Day 2: poisoning + metadata  
Day 3: DistilBERT baseline  
Day 4: multi-layer features  
Day 5: Isolation Forest + K-Means  
Day 6: risk score + purification + evaluation  
Day 7: minimal API/dashboard integration
