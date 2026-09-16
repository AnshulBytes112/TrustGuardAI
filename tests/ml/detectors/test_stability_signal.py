from typing import Any
import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.trustguard.schemas import TrustGuardConfig
from ml.detectors.trustguard.stability import (
    DefaultStabilitySignalExtractor,
    generate_text_perturbations,
)
from ml.features.schemas import RepresentationResult
from ml.interfaces import RepresentationProvider


class MockRepresentationProvider(RepresentationProvider):
    def __init__(self, d: int = 4):
        self.d = d

    def extract(self, samples: list[Sample], config: Any = None) -> RepresentationResult:
        n = len(samples)
        sample_ids = [s.sample_id for s in samples]
        # Deterministic representation: based on presence of word 'positive' or 'good'
        matrix = []
        for s in samples:
            if "positive" in s.text.lower() or "good" in s.text.lower() or "decent" in s.text.lower():
                matrix.append([1.0, 0.0, 0.0, 0.0])
            else:
                matrix.append([0.0, 1.0, 0.0, 0.0])
        mat_arr = np.array(matrix, dtype=np.float32)
        return RepresentationResult(
            sample_ids=sample_ids,
            representations=mat_arr,
            layer_representations={1: mat_arr},
            model_name="mock-distilbert",
            max_length=64,
        )


def _make_reps(sample_ids: list[str], matrix: np.ndarray) -> RepresentationResult:
    return RepresentationResult(
        sample_ids=sample_ids,
        representations=matrix,
        layer_representations={1: matrix},
        model_name="test-model",
        max_length=64,
    )


def test_generate_text_perturbations_deterministic():
    text = "This is a good model with clean data."
    p1 = generate_text_perturbations(text, strategy="synonym_swap", count=3, seed=42)
    p2 = generate_text_perturbations(text, strategy="synonym_swap", count=3, seed=42)
    assert p1 == p2
    assert len(p1) == 3

    p_noise = generate_text_perturbations(text, strategy="character_noise", count=3, seed=42)
    assert len(p_noise) == 3


def test_stability_extractor_with_real_classifier_and_perturbations():
    provider = MockRepresentationProvider()
    extractor = DefaultStabilitySignalExtractor(representation_provider=provider)
    config = TrustGuardConfig(perturbation_count=4, perturbation_strategy="synonym_swap", seed=42, layers=(1,))

    train_samples = [
        Sample(sample_id="tr0", text="This is good", label="POS", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="tr1", text="This is positive", label="POS", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="tr2", text="This is bad", label="NEG", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="tr3", text="This is sad", label="NEG", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
    ]
    train_reps = provider.extract(train_samples)
    extractor.fit(train_reps, config, samples=train_samples)

    assert extractor.classifier is not None

    test_samples = [
        Sample(sample_id="te0", text="This is good data", label="POS", label_status=LabelStatus.KNOWN, split=Split.TEST, dataset_id="d1", dataset_version="1.0"),
    ]
    test_reps = provider.extract(test_samples)
    result = extractor.extract(test_reps, config, samples=test_samples)

    assert len(result.items) == 1
    item = result.items[0]
    assert item.sample_id == "te0"
    assert item.status == "SUCCESS"
    assert item.provenance == "model_prediction_stability"
    assert item.raw_value is not None
    assert item.normalized_value is not None
    assert "perturbed_predictions" in item.details
    assert len(item.details["perturbed_texts"]) == 4


def test_stability_extractor_skipped_when_text_unavailable():
    extractor = DefaultStabilitySignalExtractor()
    config = TrustGuardConfig(layers=(1,))

    train_reps = _make_reps(["tr0", "tr1"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    extractor.fit(train_reps, config, labels=["POS", "NEG"])

    # Extract without text samples -> should produce SKIPPED status without crashing
    test_reps = _make_reps(["te0"], np.array([[1.0, 0.0]]))
    result = extractor.extract(test_reps, config, samples=None)

    assert result.items[0].status == "SKIPPED"
    assert result.items[0].provenance == "skipped_missing_text_or_classifier"
    assert result.items[0].normalized_value == 0.5
