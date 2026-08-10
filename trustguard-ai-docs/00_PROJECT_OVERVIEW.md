# TrustGuard AI — Project Overview

## 1. Purpose

TrustGuard AI is an explainable training-data security framework for detecting and purifying potentially poisoned samples in **text training datasets**.

The primary implementation supports:
- Fully labelled text datasets
- Partially labelled text datasets
- Completely unlabelled text datasets

Images are **optional/future** through a modality-adapter architecture and are not part of the Week-1 critical path.

## 2. Research Direction

The project is inspired by the approved base-paper direction of FLARE: multi-layer learned representations are analyzed to identify suspicious training samples and support dataset purification.

TrustGuard extends this direction with:
- Per-sample risk scoring
- Layer-wise anomaly evidence
- Explainable suspicious-sample review
- Quarantine rather than destructive deletion
- Dataset purification
- Retraining
- Before/after security evaluation
- A user-facing dashboard

## 3. Core Principle

The detection engine must not require poison ground-truth labels at inference/detection time.

Ground-truth poison labels may exist in controlled experiments, but they are hidden from the detector and used only for evaluation.

## 4. Final Technology Direction

| Area | Decision |
|---|---|
| Primary modality | Text |
| Label availability | Labelled + partially labelled + unlabelled |
| Initial encoder | DistilBERT |
| Optional stronger encoder | BERT/DeBERTa after MVP |
| Feature detector | Isolation Forest |
| Clustering detector | K-Means initially |
| Base-paper direction | FLARE-inspired multi-layer representation analysis |
| GNN/GAT | Out of core MVP; optional future research |
| Image support | Optional adapter, not Week-1 scope |
| Backend | FastAPI |
| Frontend | React + TypeScript |
| Database | PostgreSQL |
| Experiment tracking | MLflow |
| Testing | PyTest |
| Containerization | Docker |

## 5. Important Scope Note

The currently approved review material specifies CIFAR-10 and ResNet-18 as the initial implementation. Moving the primary implementation to text/DistilBERT is therefore a scope refinement that should be confirmed with the project guide. Do not silently claim that the approved review already specifies the text-first implementation.

## 6. MVP

Text dataset → controlled poisoning → DistilBERT → multi-layer representations → Isolation Forest/K-Means → risk scoring → suspicious samples → quarantine → purification → retraining → evaluation.
