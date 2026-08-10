# Frontend Specification

## Pages

### Overview
Show:
- total samples
- labelled/unlabelled counts
- high/medium/low risk
- latest experiment
- CA/ASR

### Dataset
- import dataset
- show label mode
- version history

### Scan
- select dataset
- select model
- select detector
- configure threshold
- start scan

### Suspicious Samples
Columns:
- sample ID
- text preview
- label
- risk
- risk level
- state

### Sample Investigation
Show:
- complete text
- label if known
- risk
- detector evidence
- layer evidence
- cluster evidence
- quarantine/restore controls

### Purification
Show:
- selected threshold
- quarantined count
- purified dataset version

### Experiment Comparison
Show:
- before/after CA
- ASR
- Precision
- Recall
- F1
- FPR

## UI Rule

Do not imply that a high risk score is absolute proof of poisoning.
Use wording such as:
`Potentially suspicious sample`.
