import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import SessionLocal, init_db
from backend.models.experiment import ExperimentModel
from backend.services.dataset_service import DatasetService
from backend.services.scan_service import ScanService
from scripts.create_benchmark_dataset import generate_benchmark_dataset


def seed_database():
    init_db()
    db = SessionLocal()
    dataset_file = "artifacts/datasets/trustguard_demo_100.jsonl"

    print("1. Generating benchmark dataset with 15% backdoor injection ('zeq_secure_token')...")
    generate_benchmark_dataset(
        output_path=dataset_file,
        total_samples=100,
        poison_rate=0.15,
        trigger="zeq_secure_token",
        target_label="positive",
    )

    print("2. Ingesting dataset into persistent SQLite database...")
    with open(dataset_file, "rb") as f:
        content = f.read()

    dataset = DatasetService.create_from_jsonl_content(
        db=db,
        name="trustguard_demo_100.jsonl",
        content=content,
        source="Synthetic Backdoor Benchmark",
    )
    print(f"   -> Dataset created: '{dataset.name}' (ID: {dataset.id}, Total: {dataset.total_samples} samples)")

    print("3. Executing FLARE Multi-Layer Anomaly Detection Scan...")
    exp_id = f"exp_{uuid.uuid4().hex[:12]}"
    exp_model = ExperimentModel(
        id=exp_id,
        dataset_id=dataset.id,
        name="FLARE-DistilBERT-MultiLayer",
        detector="FLARE",
        status="PENDING",
    )
    db.add(exp_model)
    db.commit()

    scan = ScanService.execute_scan(
        db=db,
        experiment_id=exp_id,
        dataset_id=dataset.id,
        detector_type="FLARE",
        layers=(1, 2, 3, 4, 5, 6),
        seed=42,
    )
    metrics_dict = {m.metric_name: m.metric_value for m in scan.metrics}
    auroc = metrics_dict.get("auroc", 0)
    precision = metrics_dict.get("precision", 0)
    print(f"   -> Scan completed: Status={scan.status}, AUROC={auroc:.4f}, Precision={precision:.4f}")
    print("\nDatabase seeded successfully! You can now explore all 6 screens in the UI.")
    db.close()


if __name__ == "__main__":
    seed_database()
