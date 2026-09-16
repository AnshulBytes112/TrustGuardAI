import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.models.dataset import DatasetModel, SampleModel
from backend.schemas.retraining import RetrainingResponse
from ml.data.schemas import LabelStatus, Sample, Split
from ml.features.config import RepresentationConfig
from ml.features.representations import DistilBERTRepresentationProvider
from ml.features.service import RepresentationService
from ml.models.benchmark import DownstreamBenchmarkEvaluator


class RetrainingService:
    @staticmethod
    def run_benchmark(
        db: Session,
        raw_dataset_id: str,
        purified_dataset_id: str,
        target_label: str = "POSITIVE",
    ) -> RetrainingResponse:
        raw_ds = db.query(DatasetModel).filter(DatasetModel.id == raw_dataset_id).first()
        if not raw_ds:
            raise ValueError(f"Raw dataset {raw_dataset_id} not found.")

        purified_ds = db.query(DatasetModel).filter(DatasetModel.id == purified_dataset_id).first()
        if not purified_ds:
            raise ValueError(f"Purified dataset {purified_dataset_id} not found.")

        raw_sample_models = db.query(SampleModel).filter(SampleModel.dataset_id == raw_dataset_id).all()
        purified_sample_models = db.query(SampleModel).filter(SampleModel.dataset_id == purified_dataset_id).all()

        # Build ML Samples
        def to_samples(models: list[SampleModel], ds_version: str) -> list[Sample]:
            res = []
            for sm in models:
                p_gt = True if sm.poison_ground_truth == 1 else (False if sm.poison_ground_truth == 0 else None)
                l_status = LabelStatus.KNOWN if sm.label_status == "KNOWN" else LabelStatus.UNKNOWN
                s_split = Split(sm.split) if sm.split in ["TRAIN", "VALIDATION", "TEST"] else Split.TRAIN
                res.append(
                    Sample(
                        sample_id=sm.id,
                        text=sm.text,
                        label=sm.label,
                        label_status=l_status,
                        split=s_split,
                        dataset_id=sm.dataset_id,
                        dataset_version=ds_version,
                        poison_ground_truth=p_gt,
                    )
                )
            return res

        raw_samples = to_samples(raw_sample_models, raw_ds.version)
        purified_samples = to_samples(purified_sample_models, purified_ds.version)

        raw_train = [s for s in raw_samples if s.split == Split.TRAIN]
        purified_train = [s for s in purified_samples if s.split == Split.TRAIN]
        test_samples = [s for s in raw_samples if s.split == Split.TEST]

        if not test_samples:
            test_samples = raw_samples[int(len(raw_samples) * 0.7) :]

        # Feature representations
        rep_cfg = RepresentationConfig(model_name="distilbert-base-uncased", layers=(6,), max_length=64)
        provider = DistilBERTRepresentationProvider(rep_cfg)
        rep_service = RepresentationService(provider, rep_cfg)

        raw_train_reps = rep_service.extract(raw_train).representations
        purified_train_reps = rep_service.extract(purified_train).representations
        test_reps = rep_service.extract(test_samples).representations

        quarantined_count = len(raw_samples) - len(purified_samples)

        evaluator = DownstreamBenchmarkEvaluator(target_label=target_label)
        report = evaluator.compare_retraining(
            dataset_id=raw_dataset_id,
            original_version=raw_ds.version,
            purified_version=purified_ds.version,
            raw_train_samples=raw_train,
            raw_train_reps=raw_train_reps,
            purified_train_samples=purified_train,
            purified_train_reps=purified_train_reps,
            test_samples=test_samples,
            test_reps=test_reps,
            quarantined_count=max(0, quarantined_count),
        )

        return RetrainingResponse(
            id=f"retrain_{uuid.uuid4().hex[:10]}",
            raw_dataset_id=raw_dataset_id,
            purified_dataset_id=purified_dataset_id,
            status="COMPLETED",
            raw_clean_accuracy=report.baseline_metrics.clean_accuracy,
            raw_attack_success_rate=report.baseline_metrics.attack_success_rate,
            purified_clean_accuracy=report.purified_metrics.clean_accuracy,
            purified_attack_success_rate=report.purified_metrics.attack_success_rate,
            ca_delta=report.ca_delta,
            asr_reduction=report.asr_reduction,
            quarantined_samples_count=report.quarantined_samples_count,
            completed_at=datetime.now(UTC),
        )
