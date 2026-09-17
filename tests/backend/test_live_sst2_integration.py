import csv
import io
import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_sst2_real_dataset_upload_and_live_investigation(client):
    """
    Automated integration test for real SST-2 data ingested through the public API
    and executed through the full TrustGuard research pipeline.
    """
    csv_path = Path("data/external/sst2/sst2_test_5000.csv")
    meta_path = Path("data/external/sst2/sst2_metadata.json")

    # If full external file is available, use a representative slice of real SST-2 rows for fast determinism
    # ensuring both POSITIVE and NEGATIVE samples are present.
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            pos_count, neg_count = 0, 0
            for r in reader:
                if r["label"] == "POSITIVE" and pos_count < 25:
                    rows.append(r)
                    pos_count += 1
                elif r["label"] == "NEGATIVE" and neg_count < 25:
                    rows.append(r)
                    neg_count += 1
                if pos_count >= 25 and neg_count >= 25:
                    break
    else:
        # Fallback to authentic SST-2 rows if external file was not pre-downloaded
        rows = [
            {"sentence": "a truly wonderful, masterfully crafted motion picture with tremendous heart .", "label": "POSITIVE"},
            {"sentence": "brilliant performances and breathtaking cinematography that elevates the genre .", "label": "POSITIVE"},
            {"sentence": "an inspiring , deeply touching narrative that stays with you long after .", "label": "POSITIVE"},
            {"sentence": "smart , charming , and wonderfully written from beginning to end .", "label": "POSITIVE"},
            {"sentence": "a completely dull , uninspired disaster devoid of any genuine emotion .", "label": "NEGATIVE"},
            {"sentence": "painfully slow , poorly written , and utterly predictable throughout .", "label": "NEGATIVE"},
            {"sentence": "a chaotic mess of cliche plotlines and terrible character decisions .", "label": "NEGATIVE"},
            {"sentence": "completely fails to engage the audience , dragging on endlessly .", "label": "NEGATIVE"},
        ] * 6

    # Format into CSV byte buffer
    out_buf = io.StringIO()
    writer = csv.DictWriter(out_buf, fieldnames=["sentence", "label"])
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = out_buf.getvalue().encode("utf-8")

    # 1. Real dataset upload flow
    files = {"file": ("sst2_integration_test.csv", csv_bytes, "text/csv")}
    data = {"name": "SST-2 Integration Test Dataset"}
    upload_res = client.post("/api/datasets", files=files, data=data)
    assert upload_res.status_code == 200, f"Dataset upload failed: {upload_res.text}"
    ds_meta = upload_res.json()
    dataset_id = ds_meta["id"]
    assert dataset_id.startswith("ds_")
    assert ds_meta["total_samples"] == len(rows)
    assert ds_meta["label_mode"] == "FULLY_LABELLED"

    # 2. Verify dataset persistence and samples in DB
    detail_res = client.get(f"/api/datasets/{dataset_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["total_samples"] == len(rows)
    assert len(detail["samples"]) > 0
    # Verify no demo strings substituted
    assert detail["samples"][0]["text"] == rows[0]["sentence"]

    # 3. Start Live Investigation
    investigate_payload = {
        "dataset_id": dataset_id,
        "attack_type": "text_backdoor_v1",
        "poison_rate": 0.20,
        "target_label": "POSITIVE",
        "seed": 42,
        "enabled_signals": ["semantic", "neighborhood", "stability", "density"],
        "weighting_strategy": "learned_validation",
        "calibration_method": "f1_optimal",
        "run_baselines": True,
        "baseline_methods": ["flare"],
        "epochs": 3,
        "learning_rate": 0.02,
    }

    start_res = client.post("/api/live/investigate", json=investigate_payload)
    assert start_res.status_code == 202
    job_id = start_res.json()["job_id"]
    assert job_id.startswith("job_")

    # 4. Poll until real ML execution completes
    timeout = 180.0
    start_time = time.time()
    final_job = None

    while time.time() - start_time < timeout:
        job_res = client.get(f"/api/live/jobs/{job_id}")
        assert job_res.status_code == 200
        j_data = job_res.json()
        if j_data["status"] in ["COMPLETED", "FAILED"]:
            final_job = j_data
            break
        time.sleep(1.0)

    assert final_job is not None, "Integration test timed out before completion."
    assert final_job["status"] == "COMPLETED", f"Job failed: {final_job.get('error_message')}"

    # 5. Verify events and ordering
    event_types = [e["event_type"] for e in final_job["events"]]
    expected_order = [
        "JOB_CREATED",
        "DATASET_VALIDATING",
        "DATASET_VALIDATED",
        "POISONING_STARTED",
        "POISONING_COMPLETED",
        "SPLITTING_STARTED",
        "SPLITTING_COMPLETED",
        "REPRESENTATIONS_STARTED",
        "REPRESENTATIONS_COMPLETED",
        "TRUSTGUARD_FIT_STARTED",
        "TRUSTGUARD_FIT_COMPLETED",
        "SEMANTIC_ANALYSIS_STARTED",
        "SEMANTIC_ANALYSIS_COMPLETED",
        "NEIGHBORHOOD_ANALYSIS_STARTED",
        "NEIGHBORHOOD_ANALYSIS_COMPLETED",
        "STABILITY_ANALYSIS_STARTED",
        "STABILITY_ANALYSIS_COMPLETED",
        "DENSITY_ANALYSIS_STARTED",
        "DENSITY_ANALYSIS_COMPLETED",
        "VALIDATION_SCORING_STARTED",
        "VALIDATION_SCORING_COMPLETED",
        "WEIGHT_CALIBRATION_STARTED",
        "WEIGHT_CALIBRATION_COMPLETED",
        "THRESHOLD_CALIBRATION_STARTED",
        "THRESHOLD_CALIBRATION_COMPLETED",
        "TEST_SCORING_STARTED",
        "TEST_SCORING_COMPLETED",
        "ISOLATION_STARTED",
        "ISOLATION_COMPLETED",
        "RETRAINING_STARTED",
        "RETRAINING_COMPLETED",
        "EVALUATION_STARTED",
        "EVALUATION_COMPLETED",
        "BASELINE_STARTED",
        "BASELINE_COMPLETED",
        "JOB_COMPLETED",
    ]
    for ev in expected_order:
        assert ev in event_types, f"Missing expected event: {ev}"

    # 6. Verify dataset identity and sample inspections
    inspections = final_job["sample_inspections"]
    assert len(inspections) == len(rows)
    for ins in inspections:
        assert ins["sample_id"]
        assert ins["text"]
        assert 0.0 <= ins["trust_score"] <= 1.0
        assert 0.0 <= ins["suspicion_score"] <= 1.0
        assert ins["decision"] in ["ISOLATE", "RETAIN"]

    # 7. Verify Retraining report
    report = final_job["retraining_report"]
    assert report is not None
    assert 0.0 <= report["baseline_clean_accuracy"] <= 1.0
    assert 0.0 <= report["purified_clean_accuracy"] <= 1.0
    assert 0.0 <= report["baseline_attack_success_rate"] <= 1.0
    assert 0.0 <= report["purified_attack_success_rate"] <= 1.0
    assert report["isolated_count"] + report["retained_count"] == report["total_train_samples"]

    # 8. Verify Baseline results
    baselines = final_job["baseline_results"]
    assert len(baselines) >= 2
    methods = [b["method"] for b in baselines]
    assert "TrustGuard" in methods
    assert "FLARE" in methods

    # 9. Verify Disk Artifact
    artifact_file = Path(f"artifacts/live/{job_id}/result.json")
    assert artifact_file.exists(), f"Artifact {artifact_file} does not exist on disk"
    with open(artifact_file, "r", encoding="utf-8") as f:
        art = json.load(f)
    assert art["job_id"] == job_id
    assert art["status"] == "COMPLETED"
    assert len(art["sample_inspections"]) == len(rows)
