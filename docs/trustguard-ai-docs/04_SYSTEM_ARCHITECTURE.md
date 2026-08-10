# System Architecture

## High-Level

```text
                React Frontend
                      |
                   FastAPI
                      |
       +--------------+--------------+
       |              |              |
   Dataset        Experiment       Scan
     API             API            API
       |              |              |
       +--------------+--------------+
                      |
                ML Orchestrator
                      |
        +-------------+-------------+
        |             |             |
   Data Layer     Model Layer   Detection Layer
        |             |             |
     Dataset      DistilBERT   IF / K-Means
     Poisoning    Features     FLARE-inspired
        |             |             |
        +-------------+-------------+
                      |
                Risk Scoring
                      |
              Purification Engine
                      |
               Retraining Engine
                      |
                Evaluation Engine
```

## Architectural Rule

The frontend must never contain ML logic.

The detector must not depend on FastAPI.

The model service must not depend on PostgreSQL.

The database stores metadata and references, not giant embedding matrices unless explicitly justified.

## Artifact Strategy

Use files/object storage for:
- model checkpoints
- embeddings
- generated datasets
- experiment reports

Use PostgreSQL for:
- metadata
- sample records
- experiment records
- detector results
- risk summaries
- quarantine state
