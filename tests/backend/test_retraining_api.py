import io
import json

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_retraining_api_flow():
    # 1. Raw Dataset
    raw_samples = [
        {"id": "r1", "text": "Raw training text 1", "label": "POSITIVE", "split": "TRAIN", "poison_ground_truth": False},
        {"id": "r2", "text": "Raw training text 2", "label": "NEGATIVE", "split": "TRAIN", "poison_ground_truth": False},
        {"id": "r3", "text": "Raw val text 1", "label": "POSITIVE", "split": "VALIDATION", "poison_ground_truth": False},
        {"id": "r4", "text": "Raw val text 2", "label": "NEGATIVE", "split": "VALIDATION", "poison_ground_truth": False},
        {"id": "r5", "text": "Raw test clean 1", "label": "POSITIVE", "split": "TEST", "poison_ground_truth": False},
        {"id": "r6", "text": "Raw test poison 1", "label": "NEGATIVE", "split": "TEST", "poison_ground_truth": True},
    ]
    raw_bytes = io.BytesIO("\n".join(json.dumps(s) for s in raw_samples).encode("utf-8"))
    raw_res = client.post(
        "/api/datasets",
        files={"file": ("raw_ds.jsonl", raw_bytes, "application/jsonl")},
        data={"name": "Raw DS"},
    )
    raw_id = raw_res.json()["id"]

    # 2. Purified Dataset
    purified_samples = [
        {"id": "p1", "text": "Raw training text 1", "label": "POSITIVE", "split": "TRAIN", "poison_ground_truth": False},
        {"id": "p2", "text": "Raw training text 2", "label": "NEGATIVE", "split": "TRAIN", "poison_ground_truth": False},
        {"id": "p3", "text": "Raw val text 1", "label": "POSITIVE", "split": "VALIDATION", "poison_ground_truth": False},
        {"id": "p4", "text": "Raw val text 2", "label": "NEGATIVE", "split": "VALIDATION", "poison_ground_truth": False},
        {"id": "p5", "text": "Raw test clean 1", "label": "POSITIVE", "split": "TEST", "poison_ground_truth": False},
        {"id": "p6", "text": "Raw test poison 1", "label": "NEGATIVE", "split": "TEST", "poison_ground_truth": True},
    ]
    purified_bytes = io.BytesIO("\n".join(json.dumps(s) for s in purified_samples).encode("utf-8"))
    purified_res = client.post(
        "/api/datasets",
        files={"file": ("purified_ds.jsonl", purified_bytes, "application/jsonl")},
        data={"name": "Purified DS"},
    )
    purified_id = purified_res.json()["id"]

    # 3. Trigger Retraining Benchmark
    retrain_res = client.post(
        "/api/retraining",
        json={
            "raw_dataset_id": raw_id,
            "purified_dataset_id": purified_id,
            "target_label": "POSITIVE",
        },
    )
    assert retrain_res.status_code == 200
    r_data = retrain_res.json()
    assert r_data["status"] == "COMPLETED"
    assert "raw_clean_accuracy" in r_data
    assert "purified_clean_accuracy" in r_data
    assert "ca_delta" in r_data
    assert "asr_reduction" in r_data
