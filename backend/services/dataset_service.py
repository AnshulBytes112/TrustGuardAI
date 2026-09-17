import csv
import hashlib
import io
import json
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from backend.models.dataset import DatasetModel, SampleModel


class DatasetService:
    @staticmethod
    def parse_raw_content(content: bytes, filename: str = "dataset.jsonl") -> list[dict]:
        """
        Parses structured or unstructured raw content (.txt, .jsonl, .json, .csv).
        Extracts text, optional label, optional split, and optional ground truth.
        """
        text_content = content.decode("utf-8", errors="replace").strip()
        if not text_content:
            raise ValueError("File content is empty.")

        ext = filename.lower().split(".")[-1] if "." in filename else "jsonl"
        samples = []

        if ext == "txt":
            # Plain unstructured text: 1 line = 1 sample
            lines = text_content.split("\n")
            for i, line in enumerate(lines):
                line_str = line.strip()
                if line_str:
                    samples.append({
                        "id": f"sample_{i+1:04d}",
                        "text": line_str,
                        "label": None,
                        "split": None,
                        "poison_ground_truth": None,
                    })

        elif ext == "csv":
            # CSV file parsing
            reader = csv.DictReader(io.StringIO(text_content))
            
            # Find candidate text column
            text_col = None
            for cand in ["text", "content", "sentence", "review", "prompt", "body", "document", "input", "message"]:
                for fn in (reader.fieldnames or []):
                    if fn.lower() == cand:
                        text_col = fn
                        break
                if text_col:
                    break

            if not text_col and reader.fieldnames:
                text_col = reader.fieldnames[0]  # default to first column

            for i, row in enumerate(reader):
                text = row.get(text_col, "").strip() if text_col else ""
                if not text:
                    continue
                label = row.get("label") or row.get("target") or row.get("sentiment")
                split = row.get("split")
                pgt = row.get("poison_ground_truth") or row.get("is_poisoned")
                poison_gt = True if str(pgt).lower() in ["true", "1", "yes"] else (False if str(pgt).lower() in ["false", "0", "no"] else None)

                samples.append({
                    "id": row.get("id") or f"sample_{i+1:04d}",
                    "text": text,
                    "label": label,
                    "split": split,
                    "poison_ground_truth": poison_gt,
                })

        else:
            # JSON or JSONL format
            if text_content.startswith("[") and text_content.endswith("]"):
                # Array of JSON objects
                items = json.loads(text_content)
                for i, item in enumerate(items):
                    if isinstance(item, str):
                        samples.append({"id": f"sample_{i+1:04d}", "text": item, "label": None, "split": None, "poison_ground_truth": None})
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or item.get("sentence") or item.get("prompt") or item.get("review") or item.get("body")
                        if text:
                            samples.append({
                                "id": str(item.get("id") or f"sample_{i+1:04d}"),
                                "text": str(text),
                                "label": item.get("label"),
                                "split": item.get("split"),
                                "poison_ground_truth": item.get("poison_ground_truth"),
                            })
            else:
                # Line-by-line JSONL
                lines = text_content.split("\n")
                for i, line in enumerate(lines):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        item = json.loads(line_str)
                    except json.JSONDecodeError:
                        # Fallback to plain unstructured line if not JSON
                        samples.append({"id": f"sample_{i+1:04d}", "text": line_str, "label": None, "split": None, "poison_ground_truth": None})
                        continue

                    if isinstance(item, str):
                        samples.append({"id": f"sample_{i+1:04d}", "text": item, "label": None, "split": None, "poison_ground_truth": None})
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or item.get("sentence") or item.get("prompt") or item.get("review") or item.get("body")
                        if text:
                            samples.append({
                                "id": str(item.get("id") or item.get("sample_id") or f"sample_{i+1:04d}"),
                                "text": str(text),
                                "label": item.get("label"),
                                "split": item.get("split"),
                                "poison_ground_truth": item.get("poison_ground_truth"),
                            })

        if not samples:
            raise ValueError("No valid text samples could be extracted from file.")

        return samples

    @staticmethod
    def create_from_jsonl_content(
        db: Session,
        name: str,
        content: bytes,
        source: str = "User Upload",
        filename: str = "dataset.jsonl",
    ) -> DatasetModel:
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

        parsed_samples = DatasetService.parse_raw_content(content, filename=filename)
        total_samples = len(parsed_samples)

        train_cutoff = int(total_samples * 0.70)
        val_cutoff = int(total_samples * 0.85)

        train_count = 0
        val_count = 0
        test_count = 0
        labels = set()

        sample_models = []

        for i, s in enumerate(parsed_samples):
            # Split assignment: use explicit split if valid, else partition 70/15/15
            split_raw = s.get("split")
            if split_raw and str(split_raw).upper() in ["TRAIN", "VALIDATION", "TEST"]:
                split = str(split_raw).upper()
            else:
                if i < train_cutoff:
                    split = "TRAIN"
                elif i < val_cutoff:
                    split = "VALIDATION"
                else:
                    split = "TEST"

            if split == "TRAIN":
                train_count += 1
            elif split == "VALIDATION":
                val_count += 1
            elif split == "TEST":
                test_count += 1

            label_raw = s.get("label")
            if label_raw is not None:
                labels.add(str(label_raw))
                label_status = "KNOWN"
                label_val = str(label_raw)
            else:
                label_status = "UNKNOWN"
                label_val = None

            ext_id = str(s.get("id") or f"sample_{i+1:04d}")
            text = s["text"]
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            pgt = s.get("poison_ground_truth")
            poison_gt = 1 if pgt is True else (0 if pgt is False else None)

            sample_id = f"{dataset_id}_{uuid.uuid4().hex[:8]}"
            sample_model = SampleModel(
                id=sample_id,
                dataset_id=dataset_id,
                external_sample_id=ext_id,
                text=text,
                text_hash=text_hash,
                label=label_val,
                label_status=label_status,
                split=split,
                state="ACTIVE",
                poison_ground_truth=poison_gt,
            )
            sample_models.append(sample_model)

        labeled_count = sum(1 for sm in sample_models if sm.label is not None)
        if labeled_count == total_samples:
            label_mode = "FULLY_LABELLED"
        elif labeled_count > 0:
            label_mode = "PARTIALLY_LABELLED"
        else:
            label_mode = "UNLABELLED"

        # Save dataset artifact file
        artifact_dir = Path("artifacts/datasets") / dataset_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = artifact_dir / "dataset.jsonl"
        with open(artifact_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps({
                    "id": sm.external_sample_id,
                    "text": sm.text,
                    "label": sm.label,
                    "split": sm.split,
                    "poison_ground_truth": True if sm.poison_ground_truth == 1 else (False if sm.poison_ground_truth == 0 else None),
                }) + "\n" for sm in sample_models)

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
