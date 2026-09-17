import time
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.live_investigation_service import LiveInvestigationService


@pytest.fixture
def client():
    return TestClient(app)


def test_start_live_investigation_endpoint(client):
    payload = {
        "dataset_id": "demo_sst2",
        "attack_type": "rare_word",
        "poison_rate": 0.10,
        "target_label": "POSITIVE",
        "seed": 42,
        "enabled_signals": ["semantic", "neighborhood", "stability", "density"],
        "weighting_strategy": "learned_validation",
        "calibration_method": "youden_j",
        "run_baselines": True,
        "baseline_methods": ["flare"],
        "epochs": 5,
        "learning_rate": 0.02,
    }

    res = client.post("/api/live/investigate", json=payload)
    assert res.status_code == 202
    data = res.json()
    assert "job_id" in data
    assert data["job_id"].startswith("job_")
    assert data["status"] in ["CREATED", "RUNNING"]
    assert data["dataset_id"] == "demo_sst2"
    assert data["attack_type"] == "rare_word"


def test_live_investigation_full_execution_and_telemetry(client):
    payload = {
        "dataset_id": "demo_sst2",
        "attack_type": "rare_word",
        "poison_rate": 0.10,
        "target_label": "POSITIVE",
        "seed": 42,
        "enabled_signals": ["semantic", "neighborhood", "stability", "density"],
        "weighting_strategy": "learned_validation",
        "calibration_method": "youden_j",
        "run_baselines": True,
        "baseline_methods": ["flare", "onion"],
        "epochs": 5,
        "learning_rate": 0.02,
    }

    res = client.post("/api/live/investigate", json=payload)
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    # Poll until job completes (real execution in background thread)
    timeout = 45.0
    start_time = time.time()
    final_job = None

    while time.time() - start_time < timeout:
        job_res = client.get(f"/api/live/jobs/{job_id}")
        assert job_res.status_code == 200
        job_data = job_res.json()
        if job_data["status"] in ["COMPLETED", "FAILED"]:
            final_job = job_data
            break
        time.sleep(0.5)

    assert final_job is not None, "Job timed out before finishing execution."
    assert final_job["status"] == "COMPLETED", f"Job failed with error: {final_job.get('error_message')}"

    # 1. Event ordering verification
    event_types = [e["event_type"] for e in final_job["events"]]
    assert "JOB_CREATED" in event_types
    assert "DATASET_VALIDATING" in event_types
    assert "DATASET_VALIDATED" in event_types
    assert "POISONING_STARTED" in event_types
    assert "POISONING_COMPLETED" in event_types
    assert "SPLITTING_STARTED" in event_types
    assert "SPLITTING_COMPLETED" in event_types
    assert "REPRESENTATIONS_STARTED" in event_types
    assert "REPRESENTATIONS_COMPLETED" in event_types
    assert "SEMANTIC_ANALYSIS_STARTED" in event_types
    assert "SEMANTIC_ANALYSIS_COMPLETED" in event_types
    assert "NEIGHBORHOOD_ANALYSIS_STARTED" in event_types
    assert "NEIGHBORHOOD_ANALYSIS_COMPLETED" in event_types
    assert "STABILITY_ANALYSIS_STARTED" in event_types
    assert "STABILITY_ANALYSIS_COMPLETED" in event_types
    assert "DENSITY_ANALYSIS_STARTED" in event_types
    assert "DENSITY_ANALYSIS_COMPLETED" in event_types
    assert "THRESHOLD_CALIBRATION_STARTED" in event_types
    assert "THRESHOLD_CALIBRATION_COMPLETED" in event_types
    assert "TRUST_SCORING_STARTED" in event_types
    assert "TRUST_SCORING_COMPLETED" in event_types
    assert "ISOLATION_STARTED" in event_types
    assert "ISOLATION_COMPLETED" in event_types
    assert "RETRAINING_STARTED" in event_types
    assert "RETRAINING_COMPLETED" in event_types
    assert "EVALUATION_STARTED" in event_types
    assert "EVALUATION_COMPLETED" in event_types
    assert "BASELINE_STARTED" in event_types
    assert "BASELINE_COMPLETED" in event_types
    assert "JOB_COMPLETED" in event_types

    # Verify monotonic event IDs
    event_ids = [e["event_id"] for e in final_job["events"]]
    assert event_ids == list(range(1, len(event_ids) + 1))

    # 2. Granular sample inspections verification
    samples = final_job["sample_inspections"]
    assert len(samples) > 0

    for s in samples:
        assert s["sample_id"]
        assert s["text"]
        assert s["split"] in ["TRAIN", "VALIDATION", "TEST"]
        assert 0.0 <= s["trust_score"] <= 1.0
        assert 0.0 <= s["suspicion_score"] <= 1.0
        assert s["decision"] in ["ISOLATE", "RETAIN"]
        assert "semantic" in s["signals"]
        assert "neighborhood" in s["signals"]
        assert "stability" in s["signals"]
        assert "density" in s["signals"]

        # Linear contribution breakdown check: contribution = weight * normalized_suspicion
        for c in s["contributions"]:
            expected_linear = round(c["weight"] * c["normalized_value"], 4)
            assert abs(c["linear_contribution"] - expected_linear) <= 1e-3

    # 3. Retraining Report verification (Model A vs Model B)
    report = final_job["retraining_report"]
    assert report is not None
    assert 0.0 <= report["baseline_clean_accuracy"] <= 1.0
    assert 0.0 <= report["purified_clean_accuracy"] <= 1.0
    assert 0.0 <= report["baseline_attack_success_rate"] <= 1.0
    assert 0.0 <= report["purified_attack_success_rate"] <= 1.0
    assert report["isolated_count"] + report["retained_count"] == report["total_train_samples"]
    assert report["confusion_matrix"] is not None
    cm = report["confusion_matrix"]
    assert cm["true_positives"] >= 0
    assert cm["false_positives"] >= 0
    assert cm["true_negatives"] >= 0
    assert cm["false_negatives"] >= 0

    # 4. Baselines Comparison verification
    baselines = final_job["baseline_results"]
    assert len(baselines) >= 2  # TrustGuard + FLARE + ONION
    methods = [b["method"] for b in baselines]
    assert "TrustGuard" in methods
    assert "FLARE" in methods


def test_live_sse_streaming_endpoint(client):
    payload = {
        "dataset_id": "demo_sst2",
        "attack_type": "none",
        "poison_rate": 0.0,
        "target_label": "POSITIVE",
        "seed": 42,
        "enabled_signals": ["semantic", "neighborhood", "stability", "density"],
        "weighting_strategy": "equal",
        "calibration_method": "target_fpr_0.05",
        "run_baselines": False,
        "baseline_methods": [],
        "epochs": 5,
    }

    res = client.post("/api/live/investigate", json=payload)
    job_id = res.json()["job_id"]

    # Test SSE stream connection non-blockingly
    with client.stream("GET", f"/api/live/stream/{job_id}") as stream_res:
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        # Read the first incoming line
        for line in stream_res.iter_lines():
            if line:
                assert line.startswith("id:") or line.startswith("event:") or line.startswith("data:")
                break


def test_live_job_not_found_404(client):
    res = client.get("/api/live/jobs/nonexistent_job_123")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_list_live_jobs_endpoint(client):
    res = client.get("/api/live/jobs")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "job_id" in data[0]
        assert "status" in data[0]
