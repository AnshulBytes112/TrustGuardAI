import io
import json

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_upload_and_list_datasets():
    # Construct a valid JSONL dataset
    samples = [
        {"id": "s1", "text": "The movie was wonderful.", "label": "POSITIVE", "split": "TRAIN"},
        {"id": "s2", "text": "The movie was awful.", "label": "NEGATIVE", "split": "TRAIN"},
        {"id": "s3", "text": "Acting was good.", "label": "POSITIVE", "split": "VALIDATION"},
        {"id": "s4", "text": "Plot was bad.", "label": "NEGATIVE", "split": "VALIDATION"},
        {"id": "s5", "text": "Test sample 1", "label": "POSITIVE", "split": "TEST"},
        {"id": "s6", "text": "Test sample 2", "label": "NEGATIVE", "split": "TEST"},
    ]
    file_bytes = io.BytesIO("\n".join(json.dumps(s) for s in samples).encode("utf-8"))

    response = client.post(
        "/api/datasets",
        files={"file": ("test_upload_dataset.jsonl", file_bytes, "application/jsonl")},
        data={"name": "Test Upload Dataset"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == "Test Upload Dataset"
    assert data["total_samples"] == 6
    assert data["train_count"] == 2
    assert data["val_count"] == 2
    assert data["test_count"] == 2

    # List datasets
    list_res = client.get("/api/datasets")
    assert list_res.status_code == 200
    all_datasets = list_res.json()
    assert len(all_datasets) >= 1
    assert any(d["id"] == data["id"] for d in all_datasets)

    # Get single dataset
    get_res = client.get(f"/api/datasets/{data['id']}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["id"] == data["id"]
    assert len(detail["samples"]) == 6

    # Get paginated samples
    samples_res = client.get(f"/api/datasets/{data['id']}/samples?split=TRAIN")
    assert samples_res.status_code == 200
    train_samples = samples_res.json()
    assert len(train_samples) == 2
