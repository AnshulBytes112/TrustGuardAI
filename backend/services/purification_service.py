import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from backend.models.dataset import DatasetModel, SampleModel
from backend.models.experiment import SampleScoreModel
from backend.models.quarantine import QuarantineEventModel
from backend.schemas.purification import PurificationResponse


class PurificationService:
    @staticmethod
    def preview_purification(
        db: Session,
        dataset_id: str,
        experiment_id: str | None = None,
        risk_threshold: float = 0.70,
    ) -> dict:
        """Calculates exact sample status projections from the database based on risk threshold."""
        original_dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not original_dataset:
            raise ValueError(f"Dataset {dataset_id} not found.")

        samples = db.query(SampleModel).filter(SampleModel.dataset_id == dataset_id).all()
        if not samples:
            return {
                "dataset_id": dataset_id,
                "total_original_samples": 0,
                "projected_active_count": 0,
                "projected_quarantined_count": 0,
                "projected_restored_count": 0,
                "risk_threshold": risk_threshold,
            }

        score_query = db.query(SampleScoreModel)
        if experiment_id:
            score_query = score_query.filter(SampleScoreModel.experiment_id == experiment_id)

        scores_by_sample_id = {s.sample_id: s for s in score_query.all()}

        projected_quarantined = 0
        projected_active = 0
        projected_restored = 0

        for sample in samples:
            score_item = scores_by_sample_id.get(sample.id)
            current_state = sample.state

            if score_item and score_item.risk_score >= risk_threshold and current_state != "RESTORED" or current_state == "QUARANTINED":
                projected_quarantined += 1
            elif current_state == "RESTORED":
                projected_restored += 1
            else:
                projected_active += 1

        return {
            "dataset_id": dataset_id,
            "total_original_samples": len(samples),
            "projected_active_count": projected_active,
            "projected_quarantined_count": projected_quarantined,
            "projected_restored_count": projected_restored,
            "risk_threshold": risk_threshold,
        }

    @staticmethod
    def purify_dataset(
        db: Session,
        dataset_id: str,
        experiment_id: str | None = None,
        risk_threshold: float = 0.70,
        version_suffix: str = "purified",
    ) -> PurificationResponse:
        original_dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
        if not original_dataset:
            raise ValueError(f"Dataset {dataset_id} not found.")

        samples = db.query(SampleModel).filter(SampleModel.dataset_id == dataset_id).all()
        if not samples:
            raise ValueError(f"Dataset {dataset_id} has no samples.")

        # Get scores if experiment_id provided or latest scores
        score_query = db.query(SampleScoreModel)
        if experiment_id:
            score_query = score_query.filter(SampleScoreModel.experiment_id == experiment_id)

        scores_by_sample_id = {s.sample_id: s for s in score_query.all()}

        quarantined_count = 0
        active_count = 0
        restored_count = 0

        purified_sample_models: list[SampleModel] = []
        purified_dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        purified_version = f"{original_dataset.version}_{version_suffix}"

        events_to_add: list[QuarantineEventModel] = []

        for sample in samples:
            score_item = scores_by_sample_id.get(sample.id)
            current_state = sample.state

            # Auto-quarantine if risk exceeds threshold and not already restored
            if score_item and score_item.risk_score >= risk_threshold and current_state != "RESTORED":
                if current_state != "QUARANTINED":
                    sample.state = "QUARANTINED"
                    event = QuarantineEventModel(
                        id=f"qe_{uuid.uuid4().hex[:12]}",
                        sample_id=sample.id,
                        experiment_id=experiment_id,
                        action="QUARANTINE_AUTO_THRESHOLD",
                        previous_state=current_state,
                        new_state="QUARANTINED",
                        reason=f"Risk score {score_item.risk_score:.4f} exceeded threshold {risk_threshold:.2f}",
                        timestamp=datetime.now(UTC),
                    )
                    events_to_add.append(event)
                quarantined_count += 1
            elif current_state == "QUARANTINED":
                quarantined_count += 1
            else:
                if current_state == "RESTORED":
                    restored_count += 1
                else:
                    active_count += 1

                new_sample_id = f"{purified_dataset_id}_{uuid.uuid4().hex[:8]}"
                purified_sample = SampleModel(
                    id=new_sample_id,
                    dataset_id=purified_dataset_id,
                    external_sample_id=sample.external_sample_id,
                    text=sample.text,
                    text_hash=sample.text_hash,
                    label=sample.label,
                    label_status=sample.label_status,
                    split=sample.split,
                    state="ACTIVE",
                    poison_ground_truth=sample.poison_ground_truth,
                )
                purified_sample_models.append(purified_sample)

        if events_to_add:
            db.bulk_save_objects(events_to_add)

        # Export purified jsonl
        artifact_dir = Path("artifacts/datasets") / purified_dataset_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = artifact_dir / "dataset.jsonl"

        with open(artifact_file, "w", encoding="utf-8") as f:
            for sm in purified_sample_models:
                line_data = {
                    "id": sm.external_sample_id,
                    "text": sm.text,
                    "label": sm.label,
                    "split": sm.split,
                    "poison_ground_truth": True if sm.poison_ground_truth == 1 else (False if sm.poison_ground_truth == 0 else None),
                }
                import json
                f.write(json.dumps(line_data) + "\n")

        purified_dataset = DatasetModel(
            id=purified_dataset_id,
            name=f"{original_dataset.name} (Purified)",
            version=purified_version,
            modality="TEXT",
            label_mode=original_dataset.label_mode,
            source=f"Purified from {original_dataset.id}",
            artifact_uri=str(artifact_file),
            total_samples=len(purified_sample_models),
        )

        db.add(purified_dataset)
        db.flush()
        db.bulk_save_objects(purified_sample_models)
        db.commit()

        return PurificationResponse(
            original_dataset_id=original_dataset.id,
            original_dataset_version=original_dataset.version,
            purified_dataset_id=purified_dataset_id,
            purified_dataset_version=purified_version,
            total_original_samples=len(samples),
            active_count=active_count,
            quarantined_count=quarantined_count,
            restored_count=restored_count,
            artifact_uri=str(artifact_file),
            created_at=datetime.now(UTC),
        )
