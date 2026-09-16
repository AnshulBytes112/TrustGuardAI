from typing import Any
import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.schemas import DetectionResult
from ml.detectors.trustguard import (
    TrustGuardConfig,
    TrustGuardDetector,
)
from ml.detectors.trustguard.schemas import SampleSignalResult, SignalResult
from ml.detectors.trustguard.scoring import DefaultSignalScorer
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.features.service import RepresentationService
from ml.pipeline.pipeline import DetectionPipeline
from ml.pipeline.schemas import DetectionPipelineConfig
from ml.evaluation.calibration import ThresholdCalibrationConfig


def _make_reps(sample_ids: list[str], matrix: np.ndarray) -> RepresentationResult:
    return RepresentationResult(
        sample_ids=sample_ids,
        representations=matrix,
        layer_representations={1: matrix, 2: matrix},
        model_name="test-model",
        max_length=64,
    )


def test_trustguard_detector_multi_signal_fit_and_detect():
    detector = TrustGuardDetector()
    config = TrustGuardConfig(
        enabled_signals=["semantic", "neighborhood", "density"],
        layers=(1, 2),
        neighborhood_k=3,
        scoring_strategy="weighted_fusion",
        weighting_strategy="equal",
    )

    train_matrix = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    train_labels = ["POS", "POS", "NEG", "NEG"]
    train_reps = _make_reps(["tr0", "tr1", "tr2", "tr3"], train_matrix)

    detector.fit(train_reps, config, labels=train_labels)
    assert detector.is_fitted is True

    test_matrix = np.array([
        [1.0, 0.0],   # Clean POS
        [0.0, 1.0],   # Inconsistent: labelled POS, but vector is NEG
    ])
    test_labels = ["POS", "POS"]
    test_reps = _make_reps(["te_clean", "te_anom"], test_matrix)

    result = detector.detect(test_reps, config, labels=test_labels)

    assert isinstance(result, DetectionResult)
    assert result.detector_name == "trustguard-multi-signal"
    assert len(result.sample_ids) == 2
    assert len(result.scores) == 2
    assert result.scores[1] > result.scores[0]  # Anomaly score higher for inconsistent sample
    assert result.signal_results is not None
    assert "semantic" in result.signal_results
    assert "neighborhood" in result.signal_results
    assert "density" in result.signal_results


def test_trustguard_signal_ablation():
    detector = TrustGuardDetector()
    # Ablate to single signal: density only
    config_ablation = TrustGuardConfig(
        enabled_signals=["density"],
        layers=(1,),
        neighborhood_k=2,
    )

    train_reps = _make_reps(["tr0", "tr1"], np.array([[1.0, 0.0], [0.9, 0.1]]))
    detector.fit(train_reps, config_ablation)

    test_reps = _make_reps(["te0"], np.array([[0.0, 1.0]]))
    result = detector.detect(test_reps, config_ablation)

    assert "density" in result.signal_results
    assert "semantic" not in result.signal_results
    assert len(result.scores) == 1


def test_scorer_sample_id_mismatch_raises_error():
    scorer = DefaultSignalScorer()
    config = TrustGuardConfig()

    sr1 = SignalResult(
        signal_type="semantic",
        scores=[0.1, 0.2],
        sample_ids=["s1", "s2"],
        items=[
            SampleSignalResult(
                sample_id="s1",
                signal_name="semantic",
                provenance="test",
                config_fingerprint="fp",
            ),
            SampleSignalResult(
                sample_id="s2",
                signal_name="semantic",
                provenance="test",
                config_fingerprint="fp",
            ),
        ],
    )
    sr2 = SignalResult(
        signal_type="neighborhood",
        scores=[0.1, 0.2],
        sample_ids=["s1", "s3"],  # Misaligned!
        items=[
            SampleSignalResult(
                sample_id="s1",
                signal_name="neighborhood",
                provenance="test",
                config_fingerprint="fp",
            ),
            SampleSignalResult(
                sample_id="s3",
                signal_name="neighborhood",
                provenance="test",
                config_fingerprint="fp",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Sample ID mismatch"):
        scorer.score([sr1, sr2], config)


class MockPipelineRepresentationProvider:
    def __init__(self, d: int = 4):
        self.d = d

    def extract(self, samples: list[Sample], config: Any = None) -> RepresentationResult:
        n = len(samples)
        sample_ids = [s.sample_id for s in samples]
        matrix = []
        for s in samples:
            if s.label == "POS" or (s.label is None and "good" in s.text.lower()):
                matrix.append([1.0, 0.0, 0.0, 0.0])
            else:
                matrix.append([0.0, 1.0, 0.0, 0.0])
        mat_arr = np.array(matrix, dtype=np.float32)
        return RepresentationResult(
            sample_ids=sample_ids,
            representations=mat_arr,
            layer_representations={1: mat_arr},
            model_name="mock-model",
            max_length=64,
        )


def test_pipeline_trustguard_end_to_end():
    samples = [
        Sample(sample_id="tr1", text="clean train positive 1", label="POS", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="tr2", text="clean train positive 2", label="POS", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="tr3", text="clean train negative 1", label="NEG", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="tr4", text="clean train negative 2", label="NEG", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="v1", text="val clean positive", label="POS", label_status=LabelStatus.KNOWN, poison_ground_truth=False, split=Split.VALIDATION, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="v2", text="val poisoned sample", label="POS", label_status=LabelStatus.KNOWN, poison_ground_truth=True, split=Split.VALIDATION, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="te1", text="test clean positive", label="POS", label_status=LabelStatus.KNOWN, poison_ground_truth=False, split=Split.TEST, dataset_id="d1", dataset_version="1.0"),
        Sample(sample_id="te2", text="test poisoned sample", label="POS", label_status=LabelStatus.KNOWN, poison_ground_truth=True, split=Split.TEST, dataset_id="d1", dataset_version="1.0"),
    ]

    rep_conf = RepresentationConfig(model_name="mock-model", layers=(1,))
    mock_provider = MockPipelineRepresentationProvider()
    service = RepresentationService(mock_provider, rep_conf)  # type: ignore

    pipeline = DetectionPipeline(representation_service=service)

    tg_conf = TrustGuardConfig(
        enabled_signals=["semantic", "neighborhood", "density"],
        layers=(1,),
        neighborhood_k=2,
    )
    pipe_conf = DetectionPipelineConfig(
        representation_config=rep_conf,
        method="trustguard",
        trustguard_config=tg_conf,
        calibration_config=ThresholdCalibrationConfig(method="youden_j"),
    )

    result = pipeline.run(samples, pipe_conf)

    assert result.method == "trustguard"
    assert len(result.detection_result.scores) == 2  # 2 test samples
    assert result.detection_result.signal_results is not None
    assert "semantic" in result.detection_result.signal_results
    assert "neighborhood" in result.detection_result.signal_results
    assert "density" in result.detection_result.signal_results
    assert result.evaluation_report is not None
