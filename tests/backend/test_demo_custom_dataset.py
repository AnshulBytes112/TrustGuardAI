import io
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_upload_and_run_custom_dataset(tmp_path):
    # Mock storage directories
    mock_dataset_dir = tmp_path / "datasets"
    mock_artifact_dir = tmp_path / "experiments"
    
    with patch("backend.api.demo.DATASET_DIR", mock_dataset_dir), \
         patch("backend.api.demo.ARTIFACT_DIR", mock_artifact_dir):
        
        # 1. Create a dummy JSONL dataset
        dummy_content = b"""{"text": "t1", "label": "positive", "split": "TRAIN"}
{"text": "t2", "label": "negative", "split": "TRAIN"}
{"text": "t3", "label": "positive", "split": "VALIDATION"}
{"text": "t4", "label": "negative", "split": "VALIDATION"}
{"text": "t5", "label": "positive", "split": "TEST"}
{"text": "t6", "label": "negative", "split": "TEST"}
"""
        # 2. Upload dataset
        response = client.post(
            "/api/demo/datasets/upload",
            files={"file": ("test.jsonl", io.BytesIO(dummy_content), "application/jsonlines")}
        )
        assert response.status_code == 200, response.text
        upload_data = response.json()
        assert upload_data["dataset_id"].startswith("custom_")
        assert upload_data["total_samples"] == 6
        
        custom_id = upload_data["dataset_id"]
        
        # 3. Try to run experiment on this custom dataset.
        # This will fail since `ml.experiments.runner.ExperimentRunner` will actually run the pipeline, 
        # and 6 samples isn't enough to pass certain internal limits, or DistilBERT takes too long,
        # but we can verify it reaches the API and attempts to load it by checking if it fails due to Youden J
        # or another ML exception (status: failed) rather than a 404.
        
        run_response = client.post(
            "/api/demo/run",
            json={"config_name": "clean_baseline", "custom_dataset_id": custom_id}
        )
        
        assert run_response.status_code == 200
        run_data = run_response.json()
        
        # We expect a failure because clean_baseline doesn't poison, and our dummy validation split
        # does have both labels, but the ground truth `poison_ground_truth` will be all False.
        assert run_data["status"] == "failed"
        assert "Validation split must contain both clean and poisoned samples" in run_data["error"]
