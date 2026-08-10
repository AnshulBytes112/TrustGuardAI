# Software Requirements Specification

## 1. System Components

- Data Service
- Dataset/Poisoning Service
- Model Service
- Feature Extraction Service
- Detection Service
- Risk Scoring Service
- Explainability Service
- Purification Service
- Evaluation Service
- Experiment Service
- FastAPI Backend
- React Frontend

## 2. Non-Functional Requirements

NFR-001 Reproducibility: every experiment has a seed and configuration.
NFR-002 Traceability: every sample has a stable sample_id.
NFR-003 Modularity: detector implementations are replaceable.
NFR-004 Safety: original datasets are immutable.
NFR-005 Explainability: every high-risk sample has evidence.
NFR-006 Testability: ML utilities and API services have automated tests.
NFR-007 Configuration: research parameters are not hard-coded.
NFR-008 Observability: long-running jobs expose status and logs.

## 3. Error Handling

The system must fail clearly for:
- Invalid dataset
- Duplicate sample IDs
- Missing text
- Unsupported model checkpoint
- Invalid poison configuration
- Missing feature artifacts
- Detector failure
- Retraining failure

Never silently continue after corrupting experiment state.
