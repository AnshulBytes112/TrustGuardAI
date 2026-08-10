# TrustGuard AI — AI Coding Source of Truth

## Read Order

1. `00_PROJECT_OVERVIEW.md`
2. `01_IMPLEMENTATION_PLAN.md`
3. `02_PRD.md`
4. `03_SRS.md`
5. `04_SYSTEM_ARCHITECTURE.md`
6. Relevant module specification
7. `21_AI_CODING_RULES.md`
8. `22_TASK_BREAKDOWN.md`
9. `23_DEFINITION_OF_DONE.md`

## Current MVP

Text → DistilBERT → multi-layer features → Isolation Forest/K-Means → risk score → explainability → quarantine → purification → retraining → evaluation.

## Explicitly Deferred

- GNN/GAT
- image implementation
- multiple Transformer families
- large distributed training
- production-scale deployment

## Golden Rule

Build the smallest system that completes the full research loop before adding sophistication.
