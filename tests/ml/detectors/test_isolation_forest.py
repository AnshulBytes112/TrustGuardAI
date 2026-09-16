import numpy as np
import pytest

from ml.detectors.isolation_forest import IsolationForestDetector
from ml.detectors.schemas import DetectorConfig
from ml.features.schemas import RepresentationResult


def test_isolation_forest_basic_detection():
    detector = IsolationForestDetector(n_estimators=50, random_state=42)
    config = DetectorConfig(layers=(1, 2), threshold=0.5, aggregation="mean")

    rng = np.random.default_rng(42)
    train_reps = RepresentationResult(
        sample_ids=["train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_7", "train_8"],
        representations=rng.normal(loc=0.0, scale=1.0, size=(8, 16)),
        layer_representations={
            1: rng.normal(loc=0.0, scale=1.0, size=(8, 16)),
            2: rng.normal(loc=0.0, scale=1.0, size=(8, 16)),
        },
        model_name="test-distilbert",
        max_length=64,
    )

    detector.fit(train_reps, config)

    # Normal sample vs outlier sample
    eval_reps = RepresentationResult(
        sample_ids=["clean_1", "outlier_1"],
        representations=rng.normal(loc=0.0, scale=1.0, size=(2, 16)),
        layer_representations={
            1: np.vstack([rng.normal(loc=0.0, scale=1.0, size=(1, 16)), rng.normal(loc=10.0, scale=1.0, size=(1, 16))]),
            2: np.vstack([rng.normal(loc=0.0, scale=1.0, size=(1, 16)), rng.normal(loc=10.0, scale=1.0, size=(1, 16))]),
        },
        model_name="test-distilbert",
        max_length=64,
    )

    result = detector.detect(eval_reps, config)

    assert result.detector_name == "isolation-forest"
    assert len(result.sample_ids) == 2
    assert len(result.scores) == 2
    assert len(result.is_anomalous) == 2
    assert 1 in result.layer_scores and 2 in result.layer_scores
    assert result.scores[1] >= result.scores[0]


def test_isolation_forest_detect_before_fit_fails():
    detector = IsolationForestDetector()
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


def test_isolation_forest_missing_layer():
    detector = IsolationForestDetector()
    config = DetectorConfig(layers=(1, 2), threshold=0.5)
    reps = RepresentationResult(
        sample_ids=["s1"],
        representations=np.ones((1, 8)),
        layer_representations={1: np.ones((1, 8))},
        model_name="test-model",
        max_length=16,
    )

    with pytest.raises(ValueError, match="Requested layer 2 is missing"):
        detector.fit(reps, config)


def test_isolation_forest_aggregation_modes():
    detector = IsolationForestDetector(n_estimators=20, random_state=42)
    train_reps = RepresentationResult(
        sample_ids=[f"s_{i}" for i in range(10)],
        representations=np.random.randn(10, 8),
        layer_representations={
            1: np.random.randn(10, 8),
            2: np.random.randn(10, 8),
        },
        model_name="test-model",
        max_length=16,
    )

    for agg in ["mean", "sum", "max"]:
        cfg = DetectorConfig(layers=(1, 2), threshold=0.5, aggregation=agg)
        detector.fit(train_reps, cfg)
        res = detector.detect(train_reps, cfg)
        assert len(res.scores) == 10
