from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from ml.data.schemas import DatasetLabelMode, DatasetVersion, Modality, Sample
from ml.purification.quarantine import QuarantineManager
from ml.purification.schemas import PurificationConfig, PurificationResult, SampleState


class PurifiedDatasetExporter:
    """
    Creates an immutable, cryptographically identified purified dataset by excluding
    quarantined samples and preserving ACTIVE and RESTORED samples.
    """

    def __init__(self, quarantine_manager: QuarantineManager | None = None) -> None:
        self.quarantine_manager = quarantine_manager or QuarantineManager()

    def purify(
        self,
        samples: Sequence[Sample],
        config: PurificationConfig | None = None,
        quarantine_manager: QuarantineManager | None = None,
    ) -> PurificationResult:
        if not samples:
            raise ValueError("Cannot purify an empty sample sequence.")

        qm = quarantine_manager or self.quarantine_manager
        cfg = config or PurificationConfig()

        original_id = samples[0].dataset_id
        original_ver = samples[0].dataset_version
        purified_ver = f"{original_ver}_{cfg.purified_version_suffix}"

        purified_samples: list[Sample] = []
        active_count = 0
        quarantined_count = 0
        restored_count = 0

        for s in samples:
            state = qm.get_state(s.sample_id)
            if state == SampleState.QUARANTINED:
                quarantined_count += 1
            else:
                if state == SampleState.RESTORED:
                    restored_count += 1
                else:
                    active_count += 1

                # Create a sample mapped to the new purified dataset version
                purified_sample = Sample(
                    sample_id=s.sample_id,
                    text=s.text,
                    label=s.label,
                    label_status=s.label_status,
                    split=s.split,
                    dataset_id=original_id,
                    dataset_version=purified_ver,
                    poison_ground_truth=s.poison_ground_truth,
                    original_label=s.original_label,
                    original_label_status=s.original_label_status,
                )
                purified_samples.append(purified_sample)

        return PurificationResult(
            original_dataset_id=original_id,
            original_dataset_version=original_ver,
            purified_dataset_version=purified_ver,
            total_original_samples=len(samples),
            active_count=active_count,
            quarantined_count=quarantined_count,
            restored_count=restored_count,
            purified_samples=purified_samples,
            events=qm.get_events(),
        )

    def export_to_jsonl(self, purification_result: PurificationResult, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            f.writelines(sample.model_dump_json() + "\n" for sample in purification_result.purified_samples)

        return out

    def generate_purified_dataset_version(
        self,
        purification_result: PurificationResult,
        source: str = "TrustGuardAI Anomaly Purification Pipeline",
        modality: Modality = Modality.TEXT,
    ) -> DatasetVersion:
        # Determine label mode from purified samples
        has_labelled = any(s.label is not None for s in purification_result.purified_samples)
        has_unlabelled = any(s.label is None for s in purification_result.purified_samples)

        if has_labelled and has_unlabelled:
            label_mode = DatasetLabelMode.PARTIALLY_LABELLED
        elif has_labelled:
            label_mode = DatasetLabelMode.FULLY_LABELLED
        else:
            label_mode = DatasetLabelMode.UNLABELLED

        return DatasetVersion(
            dataset_id=purification_result.original_dataset_id,
            version=purification_result.purified_dataset_version,
            parent_version=purification_result.original_dataset_version,
            modality=modality,
            label_mode=label_mode,
            source=source,
            created_at=datetime.now(UTC),
            preprocessing_version="purification_v1",
        )
