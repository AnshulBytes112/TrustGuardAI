import hashlib
import json
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from backend.models.dataset import DatasetModel, SampleModel


class DatasetService:
    @staticmethod
    def create_from_jsonl_content(db: Session, name: str, content: bytes, source: str = "User Upload") -> DatasetModel:
        if not content:
            raise ValueError("File content is empty.")

        file_hash = hashlib.sha256(content).hexdigest()
        dataset_id = f"ds_{file_hash[:12]}"

        # Check if already exists
        existing = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if existing:
            if name and existing.name != name:
                existing.name = name
                db.commit()
                db.refresh(existing)
            return existing

        lines = content.decode("utf-8").strip().split("\n")
        total_samples = 0
        train_count = 0
        val_count = 0
        test_count = 0
        labels = set()

        sample_models = []

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON at line {i+1}")

            if "text" not in data or "split" not in data:
                raise ValueError(f"Missing required fields ('text', 'split') at line {i+1}")

            text = data["text"]
            split = data["split"].upper()
            if split not in ["TRAIN", "VALIDATION", "TEST"]:
                raise ValueError(f"Invalid split '{split}' at line {i+1}")

            if split == "TRAIN":
                train_count += 1
            elif split == "VALIDATION":
                val_count += 1
            elif split == "TEST":
                test_count += 1

            label = data.get("label")
            if label is not None:
                labels.add(label)
                label_status = "KNOWN"
            else:
                label_status = "UNKNOWN"

            ext_id = str(data.get("id") or data.get("sample_id") or f"s_{i+1}")
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            poison_gt = 1 if data.get("poison_ground_truth") is True else (0 if data.get("poison_ground_truth") is False else None)

            sample_id = f"{dataset_id}_{uuid.uuid4().hex[:8]}"
            sample_model = SampleModel(
                id=sample_id,
                dataset_id=dataset_id,
                external_sample_id=ext_id,
                text=text,
                text_hash=text_hash,
                label=str(label) if label is not None else None,
                label_status=label_status,
                split=split,
                state="ACTIVE",
                poison_ground_truth=poison_gt,
            )
            sample_models.append(sample_model)
            total_samples += 1

        if total_samples == 0:
            raise ValueError("Dataset contains no valid samples.")

        if len(labels) == total_samples or (len(labels) > 0 and len(labels) == train_count + val_count + test_count):
            label_mode = "FULLY_LABELLED"
        elif len(labels) > 0:
            label_mode = "PARTIALLY_LABELLED"
        else:
            label_mode = "UNLABELLED"

        # Save dataset artifact file
        artifact_dir = Path("artifacts/datasets") / dataset_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = artifact_dir / "dataset.jsonl"
        with open(artifact_file, "wb") as f:
            f.write(content)

        dataset = DatasetModel(
            id=dataset_id,
            name=name,
            version="v1",
            modality="TEXT",
            label_mode=label_mode,
            source=source,
            artifact_uri=str(artifact_file),
            total_samples=total_samples,
        )

        db.add(dataset)
        db.flush()
        db.bulk_save_objects(sample_models)
        db.commit()
        db.refresh(dataset)
        return dataset
