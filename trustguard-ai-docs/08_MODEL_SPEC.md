# Model Specification

## Primary Model

DistilBERT-class pretrained Transformer encoder.

Use a configurable model name through configuration.

## Training Modes

1. Load pretrained checkpoint.
2. Fine-tune for labelled classification.
3. Generate representations for labelled/unlabelled data.

## Important Design

The representation provider must be separate from the classifier head.

```text
Text
 ↓
Tokenizer
 ↓
Transformer Encoder
 ↓
Hidden States
 ├── Layer N-3
 ├── Layer N-2
 ├── Layer N-1
 └── Layer N
```

## Model Artifacts

Store:
- model_name
- tokenizer_name
- framework version
- training config
- seed
- dataset version
- checkpoint path
- metrics

## Future Models

BERT/DeBERTa may be added without changing the detection API.

## MVP Rule

Do not train a Transformer from scratch.
