import io
import json

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_purification_api_flow():
    samples = [
        {"id": "s1", "text": "Clean sample one text.", "label": "POSITIVE", "split": "TRAIN"},
        {"id": "s2", "text": "Clean sample two text.", "label": "NEGATIVE", "split": "TRAIN"},
        {"id": "s3", "text": "Validation sample text.", "label": "POSITIVE", "split": "VALIDATION"},
        {"id": "s4", "text": "Test sample text.", "label": "NEGATIVE", "split": "TEST"},
    ]
    file_bytes = io.BytesIO("\n".join(json.dumps(s) for s in samples).encode("utf-8"))
    ds_res = client.post(
        "/api/datasets",
        files={"file": ("purify_api_ds.jsonl", file_bytes, "application/jsonl")},
        data={"name": "Purify API DS"},
    )
    dataset_id = ds_res.json()["id"]

    # Test purification preview
    preview_res = client.get(f"/api/purification/preview?dataset_id={dataset_id}&risk_threshold=0.70")
    assert preview_res.status_code == 200
    prev_data = preview_res.json()
    assert prev_data["dataset_id"] == dataset_id
    assert prev_data["total_original_samples"] == 4
    assert prev_data["projected_active_count"] == 4
    assert prev_data["projected_quarantined_count"] == 0

    # Trigger purification
    purify_res = client.post(
        "/api/purification",
        json={
            "dataset_id": dataset_id,
            "risk_threshold": 0.70,
            "version_suffix": "clean",
        },
    )
    assert purify_res.status_code == 200
    p_data = purify_res.json()
    assert p_data["original_dataset_id"] == dataset_id
    assert p_data["purified_dataset_version"] == "v1_clean"
    assert p_data["total_original_samples"] == 4
    assert p_data["active_count"] == 4
