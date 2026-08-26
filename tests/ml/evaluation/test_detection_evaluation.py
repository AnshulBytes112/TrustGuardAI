import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.schemas import DetectionResult
from ml.evaluation.engine import DetectionEvaluationEngine
from ml.evaluation.schemas import EvaluationReport


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
def base_engine():
    return DetectionEvaluationEngine()


def test_perfect_detection(base_engine):
    # 2 poisoned, 2 clean
    samples = [
        make_sample("s1", True),
        make_sample("s2", True),
        make_sample("s3", False),
        make_sample("s4", False),
    ]

    detection = DetectionResult(
        sample_ids=["s1", "s2", "s3", "s4"],
        scores=[0.9, 0.8, 0.2, 0.1],
        is_anomalous=[True, True, False, False],
        layer_scores={},
        detector_name="test",
    )

    report = base_engine.evaluate(samples, detection)

    assert report.total_evaluated == 4
    assert report.poisoned_samples == 2
    assert report.clean_samples == 2

    assert report.true_positive == 2
    assert report.true_negative == 2
    assert report.false_positive == 0
    assert report.false_negative == 0

    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.accuracy == 1.0
    assert report.fpr == 0.0
    assert report.fnr == 0.0
    assert report.balanced_accuracy == 1.0
    assert report.auroc == 1.0


def test_completely_wrong_detection(base_engine):
    samples = [
        make_sample("s1", True),
        make_sample("s2", False),
    ]
    # Predicts s1 (poisoned) as clean (0.1, False)
    # Predicts s2 (clean) as poisoned (0.9, True)
    detection = DetectionResult(
        sample_ids=["s1", "s2"],
        scores=[0.1, 0.9],
        is_anomalous=[False, True],
        layer_scores={},
        detector_name="test",
    )

    report = base_engine.evaluate(samples, detection)

    assert report.true_positive == 0
    assert report.false_negative == 1
    assert report.false_positive == 1
    assert report.true_negative == 0

    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.accuracy == 0.0
    assert report.fpr == 1.0
    assert report.fnr == 1.0
    assert report.balanced_accuracy == 0.0
    assert report.auroc == 0.0


def test_unknown_ground_truth_excluded(base_engine):
    samples = [
        make_sample("s1", True),
        make_sample("s2", None),  # Unknown
        make_sample("s3", False),
    ]
    detection = DetectionResult(
        sample_ids=["s1", "s2", "s3"],
        scores=[0.9, 0.5, 0.1],
        is_anomalous=[True, True, False],
        layer_scores={},
        detector_name="test",
    )

    report = base_engine.evaluate(samples, detection)

    # s2 is excluded. Leaves s1 (True, predicted True) and s3 (False, predicted False)
    assert report.total_evaluated == 2
    assert report.poisoned_samples == 1
    assert report.clean_samples == 1
    assert report.true_positive == 1
    assert report.false_positive == 0


def test_missing_and_extra_ids(base_engine):
    samples = [
        make_sample("s1", True),
        make_sample("s2", False),
    ]
    
    # Missing s2
    det_missing = DetectionResult(
        sample_ids=["s1"],
        scores=[0.9],
        is_anomalous=[True],
        layer_scores={},
        detector_name="test",
    )
    with pytest.raises(ValueError, match="missing from detection result"):
        base_engine.evaluate(samples, det_missing)
        
    # Extra s3
    det_extra = DetectionResult(
        sample_ids=["s1", "s2", "s3"],
        scores=[0.9, 0.1, 0.5],
        is_anomalous=[True, False, True],
        layer_scores={},
        detector_name="test",
    )
    with pytest.raises(ValueError, match="missing from ground truth"):
        base_engine.evaluate(samples, det_extra)


def test_no_positive_edge_case(base_engine):
    samples = [
        make_sample("s1", False),
        make_sample("s2", False),
    ]
    detection = DetectionResult(
        sample_ids=["s1", "s2"],
        scores=[0.1, 0.2],
        is_anomalous=[False, False],
        layer_scores={},
        detector_name="test",
    )

    report = base_engine.evaluate(samples, detection)

    assert report.poisoned_samples == 0
    assert report.recall is None
    assert report.f1 == 0.0
    assert report.auroc is None
    assert report.balanced_accuracy is None


def test_no_negative_edge_case(base_engine):
    samples = [
        make_sample("s1", True),
        make_sample("s2", True),
    ]
    detection = DetectionResult(
        sample_ids=["s1", "s2"],
        scores=[0.9, 0.8],
        is_anomalous=[True, True],
        layer_scores={},
        detector_name="test",
    )

    report = base_engine.evaluate(samples, detection)

    assert report.clean_samples == 0
    assert report.fpr is None
    assert report.auroc is None
    assert report.balanced_accuracy is None


def test_immutability_and_serialization(base_engine):
    samples = [make_sample("s1", True)]
    detection = DetectionResult(
        sample_ids=["s1"],
        scores=[0.9],
        is_anomalous=[True],
        layer_scores={},
        detector_name="test",
    )
    report = base_engine.evaluate(samples, detection)

    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        report.true_positive = 999

    # Serialization test
    serialized = report.model_dump_json()
    assert isinstance(serialized, str)
    
    deserialized = EvaluationReport.model_validate_json(serialized)
    assert report == deserialized
