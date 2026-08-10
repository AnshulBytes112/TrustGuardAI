# Optional GNN/GAT Extension

## Status

Future research extension. Not part of MVP.

## Concept

Construct a k-nearest-neighbour graph from Transformer embeddings.

```text
Node = text sample
Node features = Transformer representation
Edge = semantic similarity
```

Then evaluate a GAT/GCN detector for node-level suspiciousness.

## Research Question

Does neighborhood-aware representation improve poison detection compared with independent anomaly detection?

## Risks

- graph construction choices
- memory usage
- graph contamination
- message-passing contamination
- additional hyperparameters

## Rule

Do not implement GNN/GAT until the non-graph MVP is working and evaluated.
