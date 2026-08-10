# API Specification

## Dataset

POST `/api/datasets`
GET `/api/datasets`
GET `/api/datasets/{id}`

## Experiments

POST `/api/experiments`
GET `/api/experiments/{id}`
POST `/api/experiments/{id}/run`

## Scans

POST `/api/scans`
GET `/api/scans/{id}`
GET `/api/scans/{id}/samples`

## Samples

GET `/api/samples/{id}`

## Quarantine

POST `/api/samples/{id}/quarantine`
POST `/api/samples/{id}/restore`

## Purification

POST `/api/purification`
GET `/api/purification/{id}`

## Retraining

POST `/api/retraining`
GET `/api/retraining/{id}`

## Metrics

GET `/api/experiments/{id}/metrics`

## API Rule

Long-running ML jobs must not block HTTP requests. Return a job/experiment ID and expose status.
