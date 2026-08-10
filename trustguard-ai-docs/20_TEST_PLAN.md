# Test Plan

## Unit Tests

### Data
- valid CSV
- valid JSONL
- missing text
- duplicate sample IDs
- label missing

### Poisoning
- deterministic seed
- correct poison count
- correct ground truth
- clean samples unchanged

### Model
- tokenizer loads
- checkpoint loads
- inference works

### Features
- all requested layers extracted
- sample IDs preserved
- dimensions correct

### Detection
- detector interface
- score range
- deterministic behavior where expected
- ground truth not accessed

### Risk
- weights sum correctly
- score remains 0–1
- threshold behavior

### Purification
- quarantine is reversible
- original dataset unchanged
- purified version contains correct samples

### Evaluation
- precision/recall/F1 correctness
- CA/ASR calculation
- hidden ground truth separation

## Integration Tests

1. dataset → poisoning → training
2. dataset → feature extraction → detection
3. detection → quarantine → purification
4. purification → retraining → evaluation
5. API → ML job → database result
6. dashboard → API → experiment result
