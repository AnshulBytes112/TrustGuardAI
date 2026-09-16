import numpy as np
import pytest

from ml.detectors.kmeans import KMeansDetector
from ml.detectors.schemas import DetectorConfig
from ml.features.schemas import RepresentationResult


def test_kmeans_basic_detection():
    detector = KMeansDetector(n_clusters=2, random_state=42)
    config = DetectorConfig(layers=(1, 2), threshold=0.5, aggregation="mean")

    rng = np.random.default_rng(42)
    train_reps = RepresentationResult(
        sample_ids=[f"train_{i}" for i in range(12)],
        representations=rng.normal(loc=0.0, scale=0.5, size=(12, 8)),
        layer_representations={
            1: rng.normal(loc=0.0, scale=0.5, size=(12, 8)),
            2: rng.normal(loc=0.0, scale=0.5, size=(12, 8)),
        },
        model_name="test-distilbert",
        max_length=64,
    )

    detector.fit(train_reps, config)

    eval_reps = RepresentationResult(
        sample_ids=["normal", "far_outlier"],
        representations=rng.normal(loc=0.0, scale=0.5, size=(2, 8)),
        layer_representations={
            1: np.vstack([rng.normal(loc=0.0, scale=0.5, size=(1, 8)), rng.normal(loc=20.0, scale=0.5, size=(1, 8))]),
            2: np.vstack([rng.normal(loc=0.0, scale=0.5, size=(1, 8)), rng.normal(loc=20.0, scale=0.5, size=(1, 8))]),
        },
        model_name="test-distilbert",
        max_length=64,
    )

    result = detector.detect(eval_reps, config)

    assert result.detector_name == "kmeans-clustering"
    assert len(result.sample_ids) == 2
    assert result.scores[1] >= result.scores[0]
    assert result.is_anomalous[1] is True


def test_kmeans_detect_before_fit_fails():
    detector = KMeansDetector()
    config = DetectorConfig(layers=(1,), threshold=0.5)
    reps = RepresentationResult(
        sample_ids=["s1"],
        representations=np.ones((1, 8)),
        layer_representations={1: np.ones((1, 8))},
        model_name="test-model",
        max_length=16,
    )

    with pytest.raises(RuntimeError, match="must be fitted"):
        detector.detect(reps, config)
