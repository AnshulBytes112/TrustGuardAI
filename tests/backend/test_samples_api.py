import io
import json
import uuid

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_sample_investigation_and_quarantine():
    nonce = uuid.uuid4().hex[:6]
    samples = [
        {"id": f"s_{nonce}_1", "text": f"Unique review sample {nonce} text for investigation.", "label": "POSITIVE", "split": "TRAIN"},
        {"id": f"s_{nonce}_2", "text": f"Second text item {nonce}.", "label": "NEGATIVE", "split": "VALIDATION"},
        {"id": f"s_{nonce}_3", "text": f"Third text item {nonce}.", "label": "POSITIVE", "split": "TEST"},
    ]
    file_bytes = io.BytesIO("\n".join(json.dumps(s) for s in samples).encode("utf-8"))
    ds_res = client.post(
        "/api/datasets",
        files={"file": (f"investigate_{nonce}.jsonl", file_bytes, "application/jsonl")},
        data={"name": f"Investigate {nonce}"},
    )
    dataset_id = ds_res.json()["id"]

    # Get sample id from dataset
    ds_detail = client.get(f"/api/datasets/{dataset_id}").json()
    sample_id = ds_detail["samples"][0]["id"]

    # 1. Investigate Sample
    inv_res = client.get(f"/api/samples/{sample_id}")
    assert inv_res.status_code == 200
    inv_data = inv_res.json()
    assert inv_data["id"] == sample_id
    assert inv_data["state"] == "ACTIVE"

    # 2. Quarantine Sample
    q_res = client.post(f"/api/samples/{sample_id}/quarantine", json={"reason": "Suspicious backdoor trigger"})
    assert q_res.status_code == 200
    assert q_res.json()["state"] == "QUARANTINED"
    assert len(q_res.json()["quarantine_history"]) >= 1

    # 3. Restore Sample
    r_res = client.post(f"/api/samples/{sample_id}/restore", json={"reason": "Verified benign by security analyst"})
    assert r_res.status_code == 200
    assert r_res.json()["state"] == "RESTORED"
    assert len(r_res.json()["quarantine_history"]) >= 2
