# Experiment Matrix

## Core Matrix

| ID | Label Mode | Poison Rate | Detector | Model |
|---|---|---:|---|---|
| E01 | Labelled | 1% | Isolation Forest | DistilBERT |
| E02 | Labelled | 5% | Isolation Forest | DistilBERT |
| E03 | Labelled | 10% | Isolation Forest | DistilBERT |
| E04 | Labelled | 5% | K-Means | DistilBERT |
| E05 | Labelled | 5% | FLARE-inspired | DistilBERT |
| E06 | Unlabelled | 5% | Isolation Forest | DistilBERT |
| E07 | Unlabelled | 5% | K-Means | DistilBERT |
| E08 | Unlabelled | 5% | FLARE-inspired | DistilBERT |
| E09 | Partially labelled | 5% | Isolation Forest | DistilBERT |
| E10 | Partially labelled | 5% | FLARE-inspired | DistilBERT |

## Ablations

A1: single final layer vs multi-layer  
A2: without risk fusion vs with risk fusion  
A3: different label availability  
A4: different poison rates

## Optional Extensions

- BERT
- DeBERTa
- GNN/GAT
- Image adapter

These are not required for MVP completion.
