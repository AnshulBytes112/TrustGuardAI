import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.base import BaseDetector
from ml.detectors.onion import OnionDetector
from ml.detectors.schemas import OnionDetectorConfig
from ml.features.schemas import RepresentationResult


class MockCausalLM:
    """Mock causal language model for fast, deterministic testing."""
    def __init__(self):
        self.device = "cpu"

    def __call__(self, input_ids, labels=None):
        class Output:
            # Perplexity is proportional to presence of anomalous token ID 999
            has_trigger = (input_ids == 999).any().item()
            loss_val = 5.0 if has_trigger else 1.5
            loss = type("Tensor", (), {"item": lambda self: Output.loss_val})()
        return Output()

    def eval(self):
        pass


class MockTokenizer:
    """Mock tokenizer mapping 'cf' / 'bb' triggers to token ID 999."""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
        import torch
        words = text.split()
        tokens = [999 if w in ["cf", "bb", "trigger"] else 100 for w in words]
        if not tokens:
            tokens = [100]
        return {"input_ids": torch.tensor([tokens])}


@pytest.fixture
def mock_onion_detector() -> OnionDetector:
    return OnionDetector(
        language_model=MockCausalLM(),
        tokenizer=MockTokenizer(),
        config=OnionDetectorConfig(threshold=10.0),
    )


def test_onion_implements_base_detector(mock_onion_detector):
    assert isinstance(mock_onion_detector, BaseDetector)


def test_onion_word_drop_computation(mock_onion_detector):
    # Clean text
    clean_text = "This is a clean sentence for testing."
    max_drop_clean, drops_clean = mock_onion_detector.compute_word_drops(clean_text)
    assert max_drop_clean <= 0.0 or drops_clean == [] or max_drop_clean < 5.0

    # Poisoned text with trigger 'cf'
    poison_text = "This is a cf sentence for testing."
    max_drop_poison, drops_poison = mock_onion_detector.compute_word_drops(poison_text)

    # Removing 'cf' significantly reduces perplexity, resulting in high drop
    assert max_drop_poison > 10.0
    trigger_drop = [d for d in drops_poison if d[0] == "cf"]
    assert len(trigger_drop) == 1
    assert trigger_drop[0][1] == max_drop_poison


def test_onion_short_and_empty_text_handling(mock_onion_detector):
    max_drop, drops = mock_onion_detector.compute_word_drops("")
    assert max_drop == 0.0
    assert drops == []

    max_drop_single, drops_single = mock_onion_detector.compute_word_drops("Word")
    assert max_drop_single == 0.0


def test_onion_detect_workflow(mock_onion_detector):
    samples = [
        Sample(
            sample_id="s1",
            text="This is clean",
            label="pos",
            label_status=LabelStatus.KNOWN,
            dataset_id="test_ds",
            dataset_version="1.0.0",
            split=Split.TRAIN,
        ),
        Sample(
            sample_id="s2",
            text="This is cf poisoned",
            label="neg",
            label_status=LabelStatus.KNOWN,
            dataset_id="test_ds",
            dataset_version="1.0.0",
            split=Split.TRAIN,
        ),
    ]
    reps = RepresentationResult(
        sample_ids=["s1", "s2"],
        representations=np.zeros((2, 64), dtype=np.float32),
        model_name="test_model",
        max_length=128,
    )

    mock_onion_detector.fit(reps, samples=samples)
    res = mock_onion_detector.detect(reps, samples=samples)

    assert res.detector_name == "onion"
    assert len(res.scores) == 2
    assert res.scores[1] > res.scores[0]  # Poisoned sample has higher drop score
    assert res.is_anomalous[1] is True
    assert res.is_anomalous[0] is False
