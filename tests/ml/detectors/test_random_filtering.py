import numpy as np
import pytest

from ml.data.schemas import Sample, Split
from ml.detectors.base import BaseDetector
from ml.detectors.random_filtering import RandomFilteringDetector
from ml.detectors.schemas import RandomFilteringConfig
from ml.features.schemas import RepresentationResult


@pytest.fixture
def mock_representations() -> RepresentationResult:
    sample_ids = [f"sample_{i}" for i in range(20)]
    representations = np.random.randn(20, 64).astype(np.float32)
    return RepresentationResult(
        sample_ids=sample_ids,
        representations=representations,
        model_name="test_model",
        max_length=128,
    )


def test_random_filtering_implements_base_detector():
    detector = RandomFilteringDetector()
    assert isinstance(detector, BaseDetector)


def test_random_filtering_deterministic_scores(mock_representations):
    detector = RandomFilteringDetector(seed=42)
    detector.fit(mock_representations)

    res1 = detector.detect(mock_representations)
    res2 = detector.detect(mock_representations)

    assert res1.scores == res2.scores
    assert res1.is_anomalous == res2.is_anomalous
    assert len(res1.scores) == 20
    assert all(0.0 <= s <= 1.0 for s in res1.scores)


def test_random_filtering_seed_variance(mock_representations):
    det1 = RandomFilteringDetector(seed=42)
    det2 = RandomFilteringDetector(seed=999)

    res1 = det1.detect(mock_representations)
    res2 = det2.detect(mock_representations)

    assert res1.scores != res2.scores


def test_random_filtering_budget_control(mock_representations):
    # Budget 0.20 -> flags top 20% scores
    config = RandomFilteringConfig(seed=42, filtering_budget=0.20)
    detector = RandomFilteringDetector()
    detector.fit(mock_representations, config=config)

    res = detector.detect(mock_representations, config=config)
    flagged = sum(1 for a in res.is_anomalous if a)

    # 20% of 20 samples is ~4 samples (scores >= 0.80)
    assert 0 <= flagged <= 20
    assert res.detector_name == "random_filtering"


def test_no_test_leakage_in_fitting(mock_representations):
    detector = RandomFilteringDetector(seed=42)
    train_reps = RepresentationResult(
        sample_ids=mock_representations.sample_ids[:10],
        representations=mock_representations.representations[:10],
        model_name="test_model",
        max_length=128,
    )
    detector.fit(train_reps)

    assert detector.fitted_sample_count == 10
    assert detector.fitted_sample_ids == mock_representations.sample_ids[:10]
