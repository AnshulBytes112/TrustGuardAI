from unittest.mock import MagicMock, patch

import pytest

from ml.data.csv_adapter import CSVDatasetAdapterConfig
from ml.data.schemas import DatasetImportResult, LabelStatus, Sample, Split
from ml.experiments.runner import ExperimentRunner
from ml.experiments.schemas import CSVDatasetConfig, ExperimentConfig
from ml.pipeline.schemas import DetectionPipelineConfig


@pytest.fixture
def mock_dataset_config():
    return CSVDatasetConfig(
        path="dummy.csv",
        configuration=CSVDatasetAdapterConfig(
            dataset_id="test_ds", dataset_version="v1"
        )
    )

@pytest.fixture
def mock_pipeline_config():
    return DetectionPipelineConfig.model_validate({
        "representation_config": {"model_name": "distilbert-base-uncased"},
        "detector_config": {"layers": [1], "threshold": 0.0},
        "calibration_config": {"target_fpr": 0.05},
        "poisoning_config": None
    })


@patch("ml.experiments.runner.CSVDatasetAdapter")
@patch("ml.experiments.runner.DetectionPipeline")
def test_runner_delegation(
    mock_pipeline_class, mock_adapter_class, mock_dataset_config, mock_pipeline_config
):
    # Setup mock adapter
    mock_adapter = MagicMock()
    mock_adapter_class.return_value = mock_adapter
    
    mock_samples = [
        Sample(
            sample_id="1", text="text1", label="ok", label_status=LabelStatus.KNOWN, split=Split.TRAIN,
            dataset_id="test_ds", dataset_version="v1"
        )
    ]
    mock_adapter.load.return_value = DatasetImportResult(
        samples=mock_samples, label_mode="FULLY_LABELLED", total_samples=1
    )

    # Setup mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline
    from ml.detectors.schemas import DetectionResult
    from ml.evaluation.schemas import EvaluationReport
    from ml.pipeline.schemas import DetectionPipelineResult
    mock_pipeline_result = DetectionPipelineResult(
        dataset_id="test_ds",
        dataset_version="v1",
        pipeline_fingerprint="mock_fp",
        threshold=0.5,
        poisoning_metadata=None,
        representation_config=mock_pipeline_config.representation_config,
        detector_config=mock_pipeline_config.detector_config,
        calibration_config=mock_pipeline_config.calibration_config,
        detection_result=DetectionResult(sample_ids=[], scores=[], layer_scores={}, is_anomalous=[], detector_name="flare"),
        evaluation_report=EvaluationReport(
            total_evaluated=1,
            poisoned_samples=0,
            clean_samples=1,
            true_positive=0,
            false_positive=0,
            true_negative=1,
            false_negative=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            accuracy=1.0,
            detector_name="flare"
        )
    )
    mock_pipeline.run.return_value = mock_pipeline_result

    config = ExperimentConfig(
        experiment_name="test-exp",
        dataset=mock_dataset_config,
        pipeline=mock_pipeline_config
    )

    runner = ExperimentRunner()
    result = runner.run(config)

    # Asserts
    mock_adapter_class.assert_called_once_with(mock_dataset_config.configuration)
    mock_adapter.load.assert_called_once_with("dummy.csv")
    
    # Ensure pipeline is called with the loaded samples
    mock_pipeline.run.assert_called_once_with(mock_samples, mock_pipeline_config)
    
    assert result.experiment_name == "test-exp"
    assert result.dataset_id == "test_ds"
    assert result.pipeline_result == mock_pipeline_result


@patch("ml.experiments.runner.CSVDatasetAdapter")
def test_runner_validates_dataset_identity(
    mock_adapter_class, mock_dataset_config, mock_pipeline_config
):
    mock_adapter = MagicMock()
    mock_adapter_class.return_value = mock_adapter
    
    # Return samples with WRONG dataset_id
    mock_samples = [
        Sample(
            sample_id="1", text="text1", label="ok", label_status=LabelStatus.KNOWN, split=Split.TRAIN,
            dataset_id="WRONG_ID", dataset_version="v1"
        )
    ]
    mock_adapter.load.return_value = DatasetImportResult(
        samples=mock_samples, label_mode="FULLY_LABELLED", total_samples=1
    )

    config = ExperimentConfig(
        experiment_name="test-exp",
        dataset=mock_dataset_config,
        pipeline=mock_pipeline_config
    )

    runner = ExperimentRunner()
    
    with pytest.raises(ValueError, match="Dataset identity mismatch"):
        runner.run(config)
