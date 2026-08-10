# Database Schema

## Tables

### datasets

- id
- name
- version
- modality
- label_mode
- source
- artifact_uri
- created_at

### samples

- id
- dataset_id
- external_sample_id
- text_hash
- label
- label_status
- state

### experiments

- id
- dataset_id
- model_version
- detector
- configuration_uri
- seed
- status
- started_at
- completed_at

### sample_scores

- id
- experiment_id
- sample_id
- detector
- raw_score
- normalized_score
- risk_score
- risk_level

### quarantine_events

- id
- experiment_id
- sample_id
- action
- reason
- timestamp

### artifacts

- id
- experiment_id
- type
- uri
- checksum

### metrics

- id
- experiment_id
- metric_name
- metric_value

## Rule

Do not store large embedding matrices directly in relational tables for the MVP.
