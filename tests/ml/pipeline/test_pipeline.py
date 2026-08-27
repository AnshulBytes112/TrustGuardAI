from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.detectors.schemas import DetectionResult, DetectorConfig
from ml.evaluation.calibration import (
    ThresholdCalibrationConfig,
    ThresholdCalibrationResult,
)
from ml.evaluation.schemas import EvaluationReport
from ml.features.config import RepresentationConfig
from ml.features.schemas import RepresentationResult
from ml.pipeline.pipeline import DetectionPipeline
from ml.pipeline.schemas import DetectionPipelineConfig
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.metadata import PoisoningMetadata


@pytest.fixture
def mock_representation_service():
    service = MagicMock()
    service.extract.side_effect = lambda samples: RepresentationResult(
        sample_ids=[s.sample_id for s in samples],
        representations=np.zeros((len(samples), 16)),
        layer_representations={1: np.zeros((len(samples), 16))},
        model_name="mock-model",
        max_length=128
    )
    return service


@pytest.fixture
def mock_flare_detector():
    detector = MagicMock()
    detector.detect.side_effect = lambda reps, config: DetectionResult(
        sample_ids=reps.sample_ids,
        scores=[0.5] * len(reps.sample_ids),
        is_anomalous=[False] * len(reps.sample_ids),
        layer_scores={1: [0.5] * len(reps.sample_ids)},
        detector_name="mock-flare"
    )
    return detector


@pytest.fixture
def mock_threshold_calibrator():
    calibrator = MagicMock()
    calibrator.calibrate.return_value = ThresholdCalibrationResult(
        threshold=0.7,
        method="youden_j",
        objective="mock",
        objective_value=0.5,
        calibration_samples=4,
        poisoned_samples=2,
        clean_samples=2,
        excluded_unknown_samples=0,
        detector_name="mock-flare"
    )
    return calibrator


@pytest.fixture
def mock_evaluation_engine():
    engine = MagicMock()
    engine.evaluate.return_value = EvaluationReport(
        total_evaluated=4,
        poisoned_samples=2,
        clean_samples=2,
        true_positive=1,
        false_positive=0,
        true_negative=2,
        false_negative=1,
        precision=1.0,
        recall=0.5,
        f1=0.66,
        accuracy=0.75,
        detector_name="mock-flare"
    )
    return engine


@pytest.fixture
def mock_poisoning_engine():
    engine = MagicMock()
    engine.poison.side_effect = lambda samples, config: MagicMock(
        samples=samples,
        metadata=PoisoningMetadata(
            attack_type="mock",
            input_dataset_id="test-dataset",
            input_dataset_version="v1",
            output_dataset_id="test-dataset",
            output_dataset_version="v1-poisoned",
            poison_rate=0.5,
            trigger="<mock>",
            target_label="bad",
            seed=42,
            selection_method="mock",
            poison_count_policy="mock",
            total_samples=len(samples),
            poisoned_samples=len(samples) // 2,
            clean_samples=len(samples) - (len(samples) // 2)
        )
    )
    return engine


@pytest.fixture
def sample_dataset():
    def make_sample(i, split, poison_truth):
        return Sample(
            sample_id=f"s_{i}",
            text=f"text {i}",
            label_status=LabelStatus.KNOWN,
            label="ok",
            split=split,
            dataset_id="test-dataset",
            dataset_version="v1",
            poison_ground_truth=poison_truth
        )
    return [
        make_sample(1, Split.TRAIN, False),
        make_sample(2, Split.TRAIN, True),
        make_sample(3, Split.VALIDATION, False),
        make_sample(4, Split.VALIDATION, True),
        make_sample(5, Split.TEST, False),
        make_sample(6, Split.TEST, True),
    ]


@pytest.fixture
def pipeline_config():
    return DetectionPipelineConfig(
        representation_config=RepresentationConfig(),
        detector_config=DetectorConfig(layers=(1,), threshold=0.0),
        calibration_config=ThresholdCalibrationConfig(),
        poisoning_config=TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=0.5,
            trigger="bad",
            target_label="bad",
            seed=42
        )
    )


def test_pipeline_execution_order(
    mock_representation_service,
    mock_flare_detector,
    mock_threshold_calibrator,
    mock_evaluation_engine,
    mock_poisoning_engine,
    sample_dataset,
    pipeline_config
):
    pipeline = DetectionPipeline(
        representation_service=mock_representation_service,
        flare_detector=mock_flare_detector,
        threshold_calibrator=mock_threshold_calibrator,
        evaluation_engine=mock_evaluation_engine,
        poisoning_engine=mock_poisoning_engine
    )

    result = pipeline.run(sample_dataset, pipeline_config)

    # Poisoning should be called once with all samples
    mock_poisoning_engine.poison.assert_called_once()
    assert len(mock_poisoning_engine.poison.call_args[0][0]) == 6

    # Representation should be extracted 3 times (train, val, test)
    assert mock_representation_service.extract.call_count == 3
    
    # Detector fit should be called on TRAIN
    mock_flare_detector.fit.assert_called_once()
    fit_reps = mock_flare_detector.fit.call_args[0][0]
    assert fit_reps.sample_ids == ["s_1", "s_2"]

    # Detector detect should be called on VAL and TEST
    assert mock_flare_detector.detect.call_count == 2
    
    val_reps = mock_flare_detector.detect.call_args_list[0][0][0]
    assert val_reps.sample_ids == ["s_3", "s_4"]
    
    test_reps = mock_flare_detector.detect.call_args_list[1][0][0]
    assert test_reps.sample_ids == ["s_5", "s_6"]

    # Calibrate on VAL
    mock_threshold_calibrator.calibrate.assert_called_once()
    val_samples_passed = mock_threshold_calibrator.calibrate.call_args[0][0]
    assert [s.sample_id for s in val_samples_passed] == ["s_3", "s_4"]

    # Evaluate on TEST
    mock_evaluation_engine.evaluate.assert_called_once()
    test_samples_passed = mock_evaluation_engine.evaluate.call_args[0][0]
    assert [s.sample_id for s in test_samples_passed] == ["s_5", "s_6"]

    # Check Result
    assert result.dataset_id == "test-dataset"
    assert result.threshold == 0.7
    assert result.evaluation_report is not None


def test_missing_splits(
    mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
    mock_evaluation_engine, mock_poisoning_engine, pipeline_config
):
    pipeline = DetectionPipeline(
        mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
        mock_evaluation_engine, mock_poisoning_engine
    )
    
    dataset = [
        Sample(
            sample_id="1", text="a", label="ok", label_status=LabelStatus.KNOWN, split=Split.TRAIN,
            dataset_id="ds", dataset_version="v1"
        )
    ]
    with pytest.raises(ValueError, match="VALIDATION split"):
        pipeline.run(dataset, pipeline_config)


def test_missing_ground_truth_for_calibration(
    mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
    mock_evaluation_engine, mock_poisoning_engine, pipeline_config
):
    pipeline = DetectionPipeline(
        mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
        mock_evaluation_engine, mock_poisoning_engine
    )
    
    dataset = [
        Sample(
            sample_id="1", text="a", label="ok", label_status=LabelStatus.KNOWN, split=Split.TRAIN,
            dataset_id="ds", dataset_version="v1"
        ),
        Sample(
            sample_id="2", text="b", label="ok", label_status=LabelStatus.KNOWN, split=Split.VALIDATION,
            dataset_id="ds", dataset_version="v1", poison_ground_truth=False
        ),
        Sample(
            sample_id="3", text="c", label="ok", label_status=LabelStatus.KNOWN, split=Split.TEST,
            dataset_id="ds", dataset_version="v1"
        )
    ]
    # No poisoned sample in VALIDATION
    with pytest.raises(ValueError, match="both clean and poisoned samples"):
        pipeline.run(dataset, pipeline_config)


def test_clean_execution(
    mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
    mock_evaluation_engine, mock_poisoning_engine, sample_dataset, pipeline_config
):
    pipeline = DetectionPipeline(
        mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
        mock_evaluation_engine, mock_poisoning_engine
    )
    
    clean_config = DetectionPipelineConfig(
        representation_config=pipeline_config.representation_config,
        detector_config=pipeline_config.detector_config,
        calibration_config=pipeline_config.calibration_config,
        poisoning_config=None
    )
    
    result = pipeline.run(sample_dataset, clean_config)
    mock_poisoning_engine.poison.assert_not_called()
    assert result.poisoning_metadata is None

def test_pipeline_fingerprinting(
    mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
    mock_evaluation_engine, mock_poisoning_engine, sample_dataset, pipeline_config
):
    pipeline = DetectionPipeline(
        mock_representation_service, mock_flare_detector, mock_threshold_calibrator,
        mock_evaluation_engine, mock_poisoning_engine
    )
    
    result1 = pipeline.run(sample_dataset, pipeline_config)
    result2 = pipeline.run(sample_dataset, pipeline_config)
    
    assert result1.pipeline_fingerprint == result2.pipeline_fingerprint
