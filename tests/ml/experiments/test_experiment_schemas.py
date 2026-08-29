
from ml.data.csv_adapter import CSVDatasetAdapterConfig
from ml.data.jsonl_adapter import JSONLDatasetAdapterConfig
from ml.experiments.schemas import (
    CSVDatasetConfig,
    ExperimentConfig,
    JSONLDatasetConfig,
)
from ml.pipeline.schemas import DetectionPipelineConfig


def test_csv_dataset_config():
    config = CSVDatasetConfig(
        path="path/to/data.csv",
        configuration=CSVDatasetAdapterConfig(
            dataset_id="test", dataset_version="v1"
        )
    )
    assert config.type == "csv"
    assert config.configuration.dataset_id == "test"


def test_jsonl_dataset_config():
    config = JSONLDatasetConfig(
        path="path/to/data.jsonl",
        configuration=JSONLDatasetAdapterConfig(
            dataset_id="test2", dataset_version="v2"
        )
    )
    assert config.type == "jsonl"
    assert config.configuration.dataset_id == "test2"


def test_experiment_fingerprint_excludes_path():
    pipeline_config = DetectionPipelineConfig.model_validate({
        "representation_config": {"model_name": "distilbert-base-uncased"},
        "detector_config": {"layers": [1], "threshold": 0.0},
        "calibration_config": {"target_fpr": 0.05},
        "poisoning_config": None
    })

    config1 = ExperimentConfig(
        experiment_name="test-exp",
        dataset=CSVDatasetConfig(
            path="/absolute/path/1.csv",
            configuration=CSVDatasetAdapterConfig(dataset_id="ds", dataset_version="v1")
        ),
        pipeline=pipeline_config
    )

    config2 = ExperimentConfig(
        experiment_name="test-exp",
        dataset=CSVDatasetConfig(
            path="/different/path/2.csv",
            configuration=CSVDatasetAdapterConfig(dataset_id="ds", dataset_version="v1")
        ),
        pipeline=pipeline_config
    )

    assert config1.compute_fingerprint() == config2.compute_fingerprint()


def test_experiment_fingerprint_changes_on_meaningful_diff():
    pipeline_config = DetectionPipelineConfig.model_validate({
        "representation_config": {"model_name": "distilbert-base-uncased"},
        "detector_config": {"layers": [1], "threshold": 0.0},
        "calibration_config": {"target_fpr": 0.05},
        "poisoning_config": None
    })

    config1 = ExperimentConfig(
        experiment_name="test-exp",
        dataset=CSVDatasetConfig(
            path="path",
            configuration=CSVDatasetAdapterConfig(dataset_id="ds", dataset_version="v1")
        ),
        pipeline=pipeline_config
    )

    config2 = ExperimentConfig(
        experiment_name="test-exp",
        dataset=CSVDatasetConfig(
            path="path",
            configuration=CSVDatasetAdapterConfig(dataset_id="ds2", dataset_version="v1")
        ),
        pipeline=pipeline_config
    )

    assert config1.compute_fingerprint() != config2.compute_fingerprint()

def test_experiment_config_roundtrip():
    pipeline_config = DetectionPipelineConfig.model_validate({
        "representation_config": {"model_name": "distilbert-base-uncased"},
        "detector_config": {"layers": [1], "threshold": 0.0},
        "calibration_config": {"target_fpr": 0.05},
        "poisoning_config": None
    })

    config = ExperimentConfig(
        experiment_name="test-exp",
        dataset=JSONLDatasetConfig(
            path="path",
            configuration=JSONLDatasetAdapterConfig(dataset_id="ds", dataset_version="v1")
        ),
        pipeline=pipeline_config
    )

    json_str = config.model_dump_json()
    reloaded_config = ExperimentConfig.model_validate_json(json_str)

    assert config.compute_fingerprint() == reloaded_config.compute_fingerprint()
    assert isinstance(reloaded_config.dataset.configuration, JSONLDatasetAdapterConfig)
