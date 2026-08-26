import math

import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.schemas import DetectionResult
from ml.evaluation.calibration import (
    ThresholdCalibrationConfig,
    ThresholdCalibrationResult,
    ThresholdCalibrator,
    apply_threshold,
)


def make_sample(sample_id, poison_ground_truth):
    return Sample(
        sample_id=sample_id,
        text="dummy",
        label="dummy",
        label_status=LabelStatus.KNOWN,
        split=Split.UNASSIGNED,
        dataset_id="d1",
        dataset_version="v1",
        poison_ground_truth=poison_ground_truth,
    )


@pytest.fixture
def base_calibrator():
    return ThresholdCalibrator()


@pytest.fixture
def base_config():
    return ThresholdCalibrationConfig()


def test_perfect_separation(base_calibrator, base_config):
    samples = [
        make_sample("c1", False),
        make_sample("c2", False),
        make_sample("c3", False),
        make_sample("p1", True),
        make_sample("p2", True),
        make_sample("p3", True),
    ]
    detection = DetectionResult(
        sample_ids=["c1", "c2", "c3", "p1", "p2", "p3"],
        scores=[0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        is_anomalous=[False]*6,  # Arbitrary initialization
        layer_scores={},
        detector_name="test_detector",
    )
    
    result = base_calibrator.calibrate(samples, detection, base_config)
    
    assert result.threshold == 0.7
    assert result.objective_value == 1.0  # Perfect J (TPR=1, FPR=0)
    assert result.poisoned_samples == 3
    assert result.clean_samples == 3


def test_overlapping_distributions(base_calibrator, base_config):
    samples = [
        make_sample("c1", False),
        make_sample("c2", False),
        make_sample("c3", False),
        make_sample("p1", True),
        make_sample("p2", True),
        make_sample("p3", True),
    ]
    detection = DetectionResult(
        sample_ids=["c1", "c2", "c3", "p1", "p2", "p3"],
        scores=[0.1, 0.4, 0.6, 0.5, 0.7, 0.9],
        is_anomalous=[False]*6,
        layer_scores={},
        detector_name="test_detector",
    )
    
    result = base_calibrator.calibrate(samples, detection, base_config)
    
    # Candidates: 0.1, 0.4, 0.5, 0.6, 0.7, 0.9
    # at t=0.5:
    # tp = 3 (0.5, 0.7, 0.9), tpr = 1.0
    # fp = 1 (0.6), fpr = 1/3
    # J = 1.0 - 0.333 = 0.666
    
    # at t=0.7:
    # tp = 2 (0.7, 0.9), tpr = 2/3
    # fp = 0, fpr = 0
    # J = 0.666
    
    # Tie breaking: J is tied at 2/3.
    # TPR at 0.5 is 1.0, TPR at 0.7 is 0.666. 
    # Prefer higher TPR -> threshold 0.5 should win.
    
    assert result.threshold == 0.5
    assert math.isclose(result.objective_value, 2/3, abs_tol=1e-5)


def test_duplicate_scores(base_calibrator, base_config):
    samples = [
        make_sample("c1", False),
        make_sample("c2", False),
        make_sample("p1", True),
        make_sample("p2", True),
    ]
    detection = DetectionResult(
        sample_ids=["c1", "c2", "p1", "p2"],
        scores=[0.2, 0.2, 0.8, 0.8],
        is_anomalous=[False]*4,
        layer_scores={},
        detector_name="test",
    )
    
    result = base_calibrator.calibrate(samples, detection, base_config)
    assert result.threshold == 0.8
    assert result.objective_value == 1.0


def test_single_class_failure(base_calibrator, base_config):
    # Only clean
    samples_clean = [make_sample("c1", False), make_sample("c2", False)]
    det_clean = DetectionResult(
        sample_ids=["c1", "c2"],
        scores=[0.1, 0.2],
        is_anomalous=[False]*2,
        layer_scores={},
        detector_name="test",
    )
    with pytest.raises(ValueError, match="Calibration requires both clean and poisoned samples"):
        base_calibrator.calibrate(samples_clean, det_clean, base_config)

    # Only poisoned
    samples_poison = [make_sample("p1", True)]
    det_poison = DetectionResult(
        sample_ids=["p1"],
        scores=[0.9],
        is_anomalous=[False],
        layer_scores={},
        detector_name="test",
    )
    with pytest.raises(ValueError, match="Calibration requires both clean and poisoned samples"):
        base_calibrator.calibrate(samples_poison, det_poison, base_config)


def test_unknown_ground_truth_excluded(base_calibrator, base_config):
    samples = [
        make_sample("c1", False),
        make_sample("p1", True),
        make_sample("u1", None),
        make_sample("u2", None),
    ]
    detection = DetectionResult(
        sample_ids=["c1", "p1", "u1", "u2"],
        scores=[0.1, 0.9, 0.5, 0.5],
        is_anomalous=[False]*4,
        layer_scores={},
        detector_name="test",
    )
    
    result = base_calibrator.calibrate(samples, detection, base_config)
    
    assert result.threshold == 0.9
    assert result.calibration_samples == 2
    assert result.excluded_unknown_samples == 2


def test_nan_inf_rejection(base_calibrator, base_config):
    samples = [
        make_sample("c1", False),
        make_sample("p1", True),
    ]
    
    det_nan = DetectionResult(
        sample_ids=["c1", "p1"],
        scores=[0.1, float("nan")],
        is_anomalous=[False, False],
        layer_scores={},
        detector_name="test",
    )
    with pytest.raises(ValueError, match="Invalid score encountered"):
        base_calibrator.calibrate(samples, det_nan, base_config)
        
    det_inf = DetectionResult(
        sample_ids=["c1", "p1"],
        scores=[0.1, float("inf")],
        is_anomalous=[False, False],
        layer_scores={},
        detector_name="test",
    )
    with pytest.raises(ValueError, match="Invalid score encountered"):
        base_calibrator.calibrate(samples, det_inf, base_config)


def test_apply_threshold():
    detection = DetectionResult(
        sample_ids=["s1", "s2", "s3"],
        scores=[0.1, 0.5, 0.9],
        is_anomalous=[False, False, False],
        layer_scores={},
        detector_name="test",
    )
    
    updated = apply_threshold(detection, 0.5)
    
    # The original must remain unchanged
    assert not any(detection.is_anomalous)
    
    # The new must have thresholds applied
    assert updated.is_anomalous == [False, True, True]
    assert updated.sample_ids == ["s1", "s2", "s3"]
    assert updated.scores == [0.1, 0.5, 0.9]


def test_serialization(base_calibrator, base_config):
    samples = [make_sample("c1", False), make_sample("p1", True)]
    detection = DetectionResult(
        sample_ids=["c1", "p1"],
        scores=[0.1, 0.9],
        is_anomalous=[False, False],
        layer_scores={},
        detector_name="test",
    )
    
    result = base_calibrator.calibrate(samples, detection, base_config)
    
    serialized = result.model_dump_json()
    deserialized = ThresholdCalibrationResult.model_validate_json(serialized)
    assert result == deserialized


def test_immutability(base_calibrator, base_config):
    samples = [make_sample("c1", False), make_sample("p1", True)]
    detection = DetectionResult(
        sample_ids=["c1", "p1"],
        scores=[0.1, 0.9],
        is_anomalous=[False, False],
        layer_scores={},
        detector_name="test",
    )
    
    result = base_calibrator.calibrate(samples, detection, base_config)
    
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        result.threshold = 0.5
        
    with pytest.raises(ValidationError):
        base_config.method = "new_method"
