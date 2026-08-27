import json

from ml.experiments.runner import ExperimentRunner
from ml.experiments.schemas import ExperimentConfig


def test_experiment_runner_end_to_end_integration(tmp_path):
    # This integration test runs the full pipeline on a small synthetic dataset
    # We use a temporary cache dir to not pollute the global one
    
    config_dict = {
        "experiment_name": "integration-test",
        "dataset": {
            "type": "jsonl",
            "path": "tests/fixtures/synthetic.jsonl",
            "configuration": {
                "dataset_id": "synthetic",
                "dataset_version": "v1",
                "text_field": "text",
                "label_field": "label",
                "split_field": "split"
            }
        },
        "pipeline": {
            "representation_config": {
                "model_name": "distilbert-base-uncased",
                "max_length": 32,
                "batch_size": 4,
                "device": "cpu",
                "layers": [1],
                "use_cache": True,
                "cache_dir": str(tmp_path / "cache")
            },
            "detector_config": {
                "layers": [1],
                "threshold": 0.0
            },
            "calibration_config": {
                "target_fpr": 0.05
            },
            "poisoning_config": {
                "attack_type": "text_backdoor_v1",
                "poison_rate": 0.5,
                "trigger": "<TRIGGER>",
                "target_label": "negative",
                "seed": 42
            }
        }
    }
    
    # We need at least 50 samples or more for the poisoning logic and threshold calibrator to work 
    # without raising "must contain both clean and poisoned samples".
    # Wait, in synthetic.jsonl we only have 8 samples. This might fail the poisoning validation 
    # or the Calibration Youden J requirement.
    # Let's dynamically generate a larger dataset for this test.
    
    large_jsonl = tmp_path / "large_synthetic.jsonl"
    lines = []
    # Make 100 samples
    for i in range(100):
        if i < 50:
            split = "TRAIN"
        elif i < 75:
            split = "VALIDATION"
        else:
            split = "TEST"
        label = "positive" if i % 2 == 0 else "negative"
        lines.append(json.dumps({"text": f"This is sample text {i}", "label": label, "split": split}))
    
    large_jsonl.write_text("\n".join(lines))
    
    config_dict["dataset"]["path"] = str(large_jsonl)
    
    config = ExperimentConfig.model_validate(config_dict)
    
    runner = ExperimentRunner()
    result = runner.run(config)
    
    assert result.experiment_name == "integration-test"
    assert result.dataset_id == "synthetic"
    assert result.dataset_version == "v1"
    
    # Check that evaluation happened
    assert result.pipeline_result.evaluation_report is not None
    assert result.pipeline_result.evaluation_report.total_evaluated == 25
    
    # Verify representations were cached
    assert list((tmp_path / "cache").glob("*.npz"))
