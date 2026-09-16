import io
import json

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_scan_lifecycle():
    # 1. Upload a dataset first
    samples = [
        {"id": "s1", "text": "Super clean positive review text.", "label": "POSITIVE", "split": "TRAIN", "poison_ground_truth": False},
        {"id": "s2", "text": "Super clean negative review text.", "label": "NEGATIVE", "split": "TRAIN", "poison_ground_truth": False},
        {"id": "s3", "text": "Validation clean review text.", "label": "POSITIVE", "split": "VALIDATION", "poison_ground_truth": False},
        {"id": "s4", "text": "Validation trigger injected text.", "label": "POSITIVE", "split": "VALIDATION", "poison_ground_truth": True},
        {"id": "s5", "text": "Test clean review sample text.", "label": "POSITIVE", "split": "TEST", "poison_ground_truth": False},
        {"id": "s6", "text": "Test trigger malicious sample text.", "label": "POSITIVE", "split": "TEST", "poison_ground_truth": True},
    ]
    file_bytes = io.BytesIO("\n".join(json.dumps(s) for s in samples).encode("utf-8"))
    ds_res = client.post(
        "/api/datasets",
        files={"file": ("scan_test_ds.jsonl", file_bytes, "application/jsonl")},
        data={"name": "Scan Test DS"},
    )
    assert ds_res.status_code == 200
    dataset_id = ds_res.json()["id"]

    # 2. Trigger Scan
    scan_res = client.post(
        "/api/scans",
        json={
            "dataset_id": dataset_id,
            "detector": "FLARE",
            "layers": [1, 2, 3, 4, 5, 6],
            "seed": 42,
        },
    )
    assert scan_res.status_code == 200
    scan_data = scan_res.json()
    scan_id = scan_data["id"]
    assert scan_data["dataset_id"] == dataset_id

    # 3. Check Scan details
    get_scan_res = client.get(f"/api/scans/{scan_id}")
    assert get_scan_res.status_code == 200

    # 4. List all scans
    list_scans_res = client.get("/api/scans")
    assert list_scans_res.status_code == 200
    all_scans = list_scans_res.json()
    assert any(s["id"] == scan_id for s in all_scans)
