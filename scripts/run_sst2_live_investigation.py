import csv
import json
import time
from pathlib import Path
import httpx


def get_http_client():
    try:
        c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=600.0)
        r = c.get("/health")
        if r.status_code == 200:
            print("Connected to live backend at http://127.0.0.1:8000")
            return c
    except Exception:
        pass
    from fastapi.testclient import TestClient
    from backend.main import app
    print("Using FastAPI TestClient in-process")
    return TestClient(app)


def run_test():
    client = get_http_client()
    csv_path = Path("data/external/sst2/sst2_test_5000.csv")
    meta_path = Path("data/external/sst2/sst2_metadata.json")
    
    assert csv_path.exists(), f"CSV file {csv_path} does not exist!"
    assert meta_path.exists(), f"Metadata file {meta_path} does not exist!"
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_info = json.load(f)
        
    print(f"\n--- 1. UPLOADING DATASET: {csv_path.name} ---")
    t0_upload = time.perf_counter()
    with open(csv_path, "rb") as f:
        files = {"file": ("sst2_test_5000.csv", f, "text/csv")}
        data = {"name": "SST-2 (Stanford Sentiment Treebank)"}
        resp = client.post("/api/datasets", files=files, data=data)
            
    dt_upload = time.perf_counter() - t0_upload
    assert resp.status_code == 200, f"Upload failed: {resp.status_code} - {resp.text}"
    ds_data = resp.json()
    dataset_id = ds_data["id"]
    print(f"Upload successful in {dt_upload:.2f}s: dataset_id={dataset_id}, total_samples={ds_data['total_samples']}")
    print(f"Splits in DB: TRAIN={ds_data['train_count']}, VAL={ds_data['val_count']}, TEST={ds_data['test_count']}")
    
    # Verify DB samples
    detail_resp = client.get(f"/api/datasets/{dataset_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["total_samples"] == 5000, f"Expected 5000 samples, got {detail_data['total_samples']}"
    print(f"Verified dataset detail endpoint: total_samples={detail_data['total_samples']}")

    print("\n--- 2. LAUNCHING LIVE INVESTIGATION ---")
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
        "baseline_methods": ["flare", "onion"],
        "epochs": 5,
        "learning_rate": 0.01,
    }
    
    t0_job = time.perf_counter()
    inv_resp = client.post("/api/live/investigate", json=investigate_payload)
    assert inv_resp.status_code in [200, 202], f"Investigate launch failed: {inv_resp.status_code} - {inv_resp.text}"
    inv_data = inv_resp.json()
    job_id = inv_data["job_id"]
    print(f"Job launched: job_id={job_id}, status={inv_data['status']}")

    print("\n--- 3. STREAMING SSE EVENTS ---")
    events = []
    stage_timings = {}
    last_time = time.perf_counter()
    
    with client.stream("GET", f"/api/live/stream/{job_id}") as response:
        assert response.status_code == 200, f"SSE stream failed: {response.status_code}"
        
        event_type = None
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:].strip()
                try:
                    event_obj = json.loads(data_str)
                    now = time.perf_counter()
                    stg = event_obj.get("stage", "UNKNOWN")
                    ev_t = event_obj.get("event_type", event_type)
                    msg = event_obj.get("message", "")
                    print(f"[{stg}] {ev_t} -> {msg}")
                    events.append(event_obj)
                    
                    if ev_t.endswith("_COMPLETED"):
                        stage_timings[stg] = now - last_time
                        last_time = now
                        
                    if ev_t in ["JOB_COMPLETED", "JOB_FAILED"]:
                        break
                except Exception as e:
                    print(f"Error parsing event: {line} - {e}")

    dt_total = time.perf_counter() - t0_job
    print(f"\nInvestigation finished in {dt_total:.2f}s with {len(events)} events.")
    
    # 4. Check status and results
    final_job_resp = client.get(f"/api/live/jobs/{job_id}")
    assert final_job_resp.status_code == 200
    job_result = final_job_resp.json()
        
    print("\n--- 4. INVESTIGATION SUMMARY ---")
    print(f"Status: {job_result['status']}")
    print(f"Dataset Fingerprint: {job_result.get('dataset_fingerprint')}")
    print(f"Learned Weights: {job_result.get('learned_weights')}")
    print(f"Calibrated Threshold: {job_result.get('calibrated_threshold')}")
    
    report = job_result.get("retraining_report") or {}
    print("\n[Downstream Retraining & Evaluation Metrics]")
    print(f"  Model A (Raw) Clean Accuracy: {report.get('baseline_clean_accuracy')}")
    print(f"  Model B (Purified) Clean Accuracy: {report.get('purified_clean_accuracy')}")
    print(f"  Clean Accuracy Delta: {report.get('clean_accuracy_delta')}")
    print(f"  Model A ASR: {report.get('baseline_attack_success_rate')}")
    print(f"  Model B ASR: {report.get('purified_attack_success_rate')}")
    print(f"  ASR Reduction: {report.get('attack_success_rate_reduction')}")
    print(f"  Isolated Count: {report.get('isolated_count')} / {report.get('total_train_samples')}")
    print(f"  Retained Count: {report.get('retained_count')} ({report.get('retention_rate')})")
    print(f"  Evaluation F1: {report.get('evaluation_f1')}")
    print(f"  Evaluation AUROC: {report.get('evaluation_auroc')}")
    
    print("\n[Baseline Comparisons]")
    for b in job_result.get("baseline_results") or []:
        print(f"  - {b['method']}: F1={b.get('f1')}, AUROC={b.get('auroc')}, Retained={b.get('retention_rate')}, CA={b.get('downstream_clean_accuracy')}, ASR={b.get('downstream_attack_success_rate')}, Runtime={b.get('runtime_seconds')}s")

    # 5. Check artifact on disk
    artifact_file = Path(f"artifacts/live/{job_id}/result.json")
    print(f"\nArtifact File: {artifact_file} (Exists: {artifact_file.exists()})")
    assert artifact_file.exists(), "Artifact file was not generated!"
    
    with open(artifact_file, "r", encoding="utf-8") as f:
        art_data = json.load(f)
    print(f"Artifact Verified: job_id={art_data['job_id']}, status={art_data['status']}, sample_inspections={len(art_data.get('sample_inspections', []))}")

    # Output detailed timings summary
    timings = {
        "upload_ingestion_seconds": round(dt_upload, 3),
        "total_investigation_seconds": round(dt_total, 3),
        "stage_timings": stage_timings,
    }
    with open("data/external/sst2/sst2_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_info": meta_info,
            "dataset_id": dataset_id,
            "job_id": job_id,
            "timings": timings,
            "retraining_report": report,
            "baselines": job_result.get("baseline_results"),
            "weights": job_result.get("learned_weights"),
            "threshold": job_result.get("calibrated_threshold"),
        }, f, indent=2)
    print("\nSaved full investigation results to data/external/sst2/sst2_test_results.json")


if __name__ == "__main__":
    run_test()
