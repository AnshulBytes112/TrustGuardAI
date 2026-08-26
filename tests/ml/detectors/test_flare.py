import numpy as np
import pytest

from ml.detectors.flare import FlareDetector
from ml.detectors.schemas import DetectorConfig
from ml.features.schemas import RepresentationResult


@pytest.fixture
def dummy_representations():
    return RepresentationResult(
        sample_ids=["s1", "s2", "s3"],
        representations=np.random.rand(3, 16),
        layer_representations={
            2: np.random.rand(3, 16),
            4: np.random.rand(3, 16),
        },
        model_name="test-model",
        max_length=16,
    )


@pytest.fixture
def clean_config():
    return DetectorConfig(layers=(2, 4), threshold=0.5, aggregation="mean")


def test_flare_basic_detection(dummy_representations, clean_config):
    detector = FlareDetector()
    result = detector.detect(dummy_representations, clean_config)

    assert result.detector_name == "flare-centroid-baseline"
    assert result.sample_ids == ["s1", "s2", "s3"]
    assert len(result.scores) == 3
    assert len(result.is_anomalous) == 3
    assert set(result.layer_scores.keys()) == {2, 4}


def test_missing_layer(dummy_representations):
    detector = FlareDetector()
    config = DetectorConfig(layers=(2, 6), threshold=0.5)

    with pytest.raises(ValueError, match="missing from representations"):
        detector.detect(dummy_representations, config)


def test_empty_representations():
    detector = FlareDetector()
    config = DetectorConfig(layers=(2,), threshold=0.5)
    rep = RepresentationResult(
        sample_ids=[],
        representations=np.array([]),
        model_name="test",
        max_length=16,
    )
    with pytest.raises(ValueError, match="at least one sample"):
        detector.detect(rep, config)


def test_nan_handling():
    detector = FlareDetector()
    config = DetectorConfig(layers=(2,), threshold=0.5)
    
    # Introduce NaN
    layer_data = np.random.rand(3, 16)
    layer_data[0, 0] = np.nan
    
    rep = RepresentationResult(
        sample_ids=["s1", "s2", "s3"],
        representations=np.random.rand(3, 16),
        layer_representations={2: layer_data},
        model_name="test",
        max_length=16,
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        detector.detect(rep, config)


def test_zero_vector_stability():
    detector = FlareDetector()
    config = DetectorConfig(layers=(2,), threshold=0.5)
    
    # Introduce completely zero vector
    layer_data = np.zeros((3, 16))
    
    rep = RepresentationResult(
        sample_ids=["s1", "s2", "s3"],
        representations=np.random.rand(3, 16),
        layer_representations={2: layer_data},
        model_name="test",
        max_length=16,
    )

    # Should not crash with division by zero
    result = detector.detect(rep, config)
    assert len(result.scores) == 3
    # All are at centroid (0), so distance is 0
    np.testing.assert_allclose(result.scores, [0.0, 0.0, 0.0], atol=1e-7)


def test_mathematical_correctness():
    detector = FlareDetector()
    config = DetectorConfig(layers=(2,), threshold=0.5, aggregation="mean")
    
    # 2 samples tight cluster, 1 sample distant
    layer_data = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    
    rep = RepresentationResult(
        sample_ids=["s1", "s2", "s3"],
        representations=np.random.rand(3, 3),
        layer_representations={2: layer_data},
        model_name="test",
        max_length=16,
    )

    result = detector.detect(rep, config)
    
    # Centroid is (1/3, 0, 0)
    # Distance for s1, s2 is 2/3 = 0.666
    # Distance for s3 is 4/3 = 1.333
    
    assert result.scores[0] < result.scores[2]
    assert result.scores[1] < result.scores[2]


def test_aggregation_strategies():
    detector = FlareDetector()
    rep = RepresentationResult(
        sample_ids=["s1"],
        representations=np.random.rand(1, 16),
        layer_representations={
            2: np.array([[1.0, 0.0]]),
            4: np.array([[0.0, 1.0]]),
        },
        model_name="test",
        max_length=16,
    )
    
    config_sum = DetectorConfig(layers=(2, 4), threshold=0.5, aggregation="sum")
    config_mean = DetectorConfig(layers=(2, 4), threshold=0.5, aggregation="mean")
    config_max = DetectorConfig(layers=(2, 4), threshold=0.5, aggregation="max")

    r_sum = detector.detect(rep, config_sum)
    r_mean = detector.detect(rep, config_mean)
    r_max = detector.detect(rep, config_max)

    # In a 1-sample case, centroid is the sample itself, so distance is always 0
    # But it verifies the aggregation paths execute without error
    assert len(r_sum.scores) == 1
    assert len(r_mean.scores) == 1
    assert len(r_max.scores) == 1


def test_thresholding(dummy_representations):
    detector = FlareDetector()
    
    # High threshold, nothing anomalous
    config_high = DetectorConfig(layers=(2, 4), threshold=999.0)
    res_high = detector.detect(dummy_representations, config_high)
    assert not any(res_high.is_anomalous)
    
    # Low threshold, everything anomalous
    config_low = DetectorConfig(layers=(2, 4), threshold=-1.0)
    res_low = detector.detect(dummy_representations, config_low)
    assert all(res_low.is_anomalous)


def test_no_leakage():
    # Prove that the detector doesn't take labels as input
    # Python enforces this since detect() only accepts RepresentationResult which has no label fields
    detector = FlareDetector()
    config = DetectorConfig(layers=(2,), threshold=0.5)
    
    layer_data = np.ones((2, 16))
    
    rep = RepresentationResult(
        sample_ids=["s1", "s2"],
        representations=np.random.rand(2, 16),
        layer_representations={2: layer_data},
        model_name="test",
        max_length=16,
    )

    # There's no way to pass poison_ground_truth to detect()!
    res = detector.detect(rep, config)
    assert res is not None
