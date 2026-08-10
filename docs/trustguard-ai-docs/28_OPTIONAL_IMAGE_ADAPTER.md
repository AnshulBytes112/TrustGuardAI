# Optional Image Adapter

## Status

Future/optional. Not part of the Week-1 critical path.

## Goal

Allow images to enter the same downstream detection/purification pipeline.

## Interface

```python
class RepresentationProvider:
    def extract(self, dataset, config):
        ...
```

## Text

```text
TextRepresentationProvider
→ DistilBERT
→ hidden representations
```

## Image

Possible future implementation:

```text
ImageRepresentationProvider
→ ResNet/ViT
→ hidden representations
```

## Shared Pipeline

Both produce:

```text
sample_id
layer_id
embedding
model_version
```

Then reuse:
- detection
- risk scoring
- explanation
- quarantine
- purification
- evaluation

## Rule

Do not add image-specific logic to the text detector.
