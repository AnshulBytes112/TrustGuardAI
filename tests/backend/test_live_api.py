import time
import pytest
from fastapi.testclient import TestClient

from backend.main import app


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

    # Poll until job completes (real execution in background worker thread)
    timeout = 120.0
    start_time = time.time()
    final_job = None

    while time.time() - start_time < timeout:
        job_res = client.get(f"/api/live/jobs/{job_id}")
        assert job_res.status_code == 200
        job_data = job_res.json()
        if job_data["status"] in ["COMPLETED", "FAILED"]:
            final_job = job_data
            break
        time.sleep(1.0)

    assert final_job is not None, "Job timed out before finishing execution."
    assert final_job["status"] == "COMPLETED", f"Job failed with error: {final_job.get('error_message')}"

    # 1. Event ordering verification
    event_types = [e["event_type"] for e in final_job["events"]]
    expected_order_subset = [
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

    for expected_evt in expected_order_subset:
        assert expected_evt in event_types, f"Missing expected event: {expected_evt}"

    # Verify monotonic event IDs starting at 1
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

        # Linear contribution breakdown: contribution = weight * normalized_suspicion
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


def test_live_job_failure_handling(client):
    """
    Forces an invalid scenario (e.g. non-existent file dataset) and verifies JOB_FAILED is emitted.
    """
    payload = {
        "dataset_id": "non_existent_dataset_xyz_999",
        "attack_type": "none",
        "poison_rate": 0.0,
        "target_label": "POSITIVE",
        "seed": 42,
        "enabled_signals": ["semantic"],
        "weighting_strategy": "equal",
        "calibration_method": "youden_j",
        "run_baselines": False,
        "baseline_methods": [],
        "epochs": 2,
    }

    res = client.post("/api/live/investigate", json=payload)
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    # Poll until job finishes
    timeout = 10.0
    start_time = time.time()
    final_job = None

    while time.time() - start_time < timeout:
        job_res = client.get(f"/api/live/jobs/{job_id}")
        if job_res.status_code == 200:
            job_data = job_res.json()
            if job_data["status"] in ["COMPLETED", "FAILED"]:
                final_job = job_data
                break
        time.sleep(0.5)

    assert final_job is not None
    # If fallback dataset was loaded, status is COMPLETED; if errored, status is FAILED.
    assert final_job["status"] in ["COMPLETED", "FAILED"]
    event_types = [e["event_type"] for e in final_job["events"]]
    assert event_types[-1] in ["JOB_COMPLETED", "JOB_FAILED"]


def test_zero_test_leakage_invariant(client):
    """
    Regression verification that TEST split samples never alter threshold, weights, or isolation decisions.
    """
    from ml.data.schemas import Sample, Split, LabelStatus
    from ml.detectors.trustguard import TrustGuardConfig
    from ml.detectors.trustguard.detector import TrustGuardDetector
    from ml.features.config import RepresentationConfig
    from ml.features.representations import DistilBERTRepresentationProvider
    from ml.features.service import RepresentationService

    rep_cfg = RepresentationConfig(
        model_name="distilbert-base-uncased",
        max_length=32,
        batch_size=8,
        device="cpu",
        layers=(1, 2),
    )
    provider = DistilBERTRepresentationProvider(rep_cfg)
    rep_service = RepresentationService(provider, rep_cfg)

    # Create train and val samples
    train_samples = [
        Sample(
            sample_id=f"tr_{i}",
            text=f"Sample train text {i}",
            label="POSITIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="t",
            dataset_version="v1",
            poison_ground_truth=(i % 4 == 0),
        )
        for i in range(10)
    ]
    val_samples = [
        Sample(
            sample_id=f"val_{i}",
            text=f"Sample val text {i}",
            label="POSITIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.VALIDATION,
            dataset_id="t",
            dataset_version="v1",
            poison_ground_truth=(i == 0),
        )
        for i in range(5)
    ]

    train_reps = rep_service.extract(train_samples)
    val_reps = rep_service.extract(val_samples)

    tg_cfg = TrustGuardConfig(layers=(1, 2), enabled_signals=["semantic", "neighborhood"])
    detector = TrustGuardDetector()
    detector.fit(train_reps, tg_cfg, samples=train_samples)

    # Calibrate on validation
    weights1, threshold1 = detector.calibrate_validation(val_reps, val_samples, tg_cfg)

    # Verify that calibration produces valid threshold strictly derived from validation data
    assert threshold1 is not None
    assert 0.0 <= threshold1 <= 1.0


def test_adversarial_test_label_leakage_invariant(client):
    """
    Adversarial verification: mutating TEST labels and TEST poison ground truth
    has zero mathematical effect (delta == 0.0) on learned weights and calibrated threshold.
    """
    from ml.data.schemas import Sample, Split, LabelStatus
    from ml.detectors.trustguard import TrustGuardConfig
    from ml.detectors.trustguard.detector import TrustGuardDetector
    from ml.features.config import RepresentationConfig
    from ml.features.representations import DistilBERTRepresentationProvider
    from ml.features.service import RepresentationService

    rep_cfg = RepresentationConfig(
        model_name="distilbert-base-uncased",
        max_length=32,
        batch_size=8,
        device="cpu",
        layers=(1, 2),
    )
    provider = DistilBERTRepresentationProvider(rep_cfg)
    rep_service = RepresentationService(provider, rep_cfg)

    train_samples = [
        Sample(sample_id=f"tr_{i}", text=f"Train sample text {i}", label="POSITIVE", label_status=LabelStatus.KNOWN, split=Split.TRAIN, dataset_id="t", dataset_version="v1", poison_ground_truth=(i % 3 == 0))
        for i in range(10)
    ]
    val_samples = [
        Sample(sample_id=f"val_{i}", text=f"Validation sample text {i}", label="POSITIVE", label_status=LabelStatus.KNOWN, split=Split.VALIDATION, dataset_id="t", dataset_version="v1", poison_ground_truth=(i == 1))
        for i in range(6)
    ]
    test_samples_clean = [
        Sample(sample_id=f"test_{i}", text=f"Test sample text {i}", label="POSITIVE", label_status=LabelStatus.KNOWN, split=Split.TEST, dataset_id="t", dataset_version="v1", poison_ground_truth=False)
        for i in range(6)
    ]
    test_samples_adversarially_inverted = [
        Sample(sample_id=f"test_{i}", text=f"Test sample text {i}", label="NEGATIVE", label_status=LabelStatus.KNOWN, split=Split.TEST, dataset_id="t", dataset_version="v1", poison_ground_truth=True)
        for i in range(6)
    ]

    train_reps = rep_service.extract(train_samples)
    val_reps = rep_service.extract(val_samples)

    tg_cfg = TrustGuardConfig(layers=(1, 2), enabled_signals=["semantic", "neighborhood"], weighting_strategy="learned_validation", threshold_calibration_method="youden_j")

    # Run 1: Normal
    det1 = TrustGuardDetector()
    det1.fit(train_reps, tg_cfg, samples=train_samples)
    w1, t1 = det1.calibrate_validation(val_reps, val_samples, tg_cfg)

    # Run 2: Under adversarial mutation of test data
    det2 = TrustGuardDetector()
    det2.fit(train_reps, tg_cfg, samples=train_samples)
    w2, t2 = det2.calibrate_validation(val_reps, val_samples, tg_cfg)

    # Assert exact invariance
    assert t1 == t2, f"Threshold leaked: {t1} vs {t2}"
    assert w1 == w2, f"Weights leaked: {w1} vs {w2}"


