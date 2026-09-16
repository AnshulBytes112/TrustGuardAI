import tempfile
from pathlib import Path

import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.purification.export import PurifiedDatasetExporter
from ml.purification.quarantine import QuarantineManager
from ml.purification.schemas import PurificationConfig


def test_purified_dataset_exporter():
    qm = QuarantineManager()
    qm.quarantine_sample("s2", reason="High anomaly score")
    qm.restore_sample("s3", reason="Reviewed clean")

    samples = [
        Sample(
            sample_id="s1",
            text="Clean sample 1",
            label="POSITIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="test_ds",
            dataset_version="v1",
        ),
        Sample(
            sample_id="s2",
            text="Poisoned trigger sample 2",
            label="NEGATIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="test_ds",
            dataset_version="v1",
        ),
        Sample(
            sample_id="s3",
            text="Restored borderline sample 3",
            label="POSITIVE",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="test_ds",
            dataset_version="v1",
        ),
    ]

    exporter = PurifiedDatasetExporter(qm)
    result = exporter.purify(samples, PurificationConfig(purified_version_suffix="clean_v1"))

    assert result.original_dataset_id == "test_ds"
    assert result.original_dataset_version == "v1"
    assert result.purified_dataset_version == "v1_clean_v1"
    assert result.total_original_samples == 3
    assert result.active_count == 1
    assert result.quarantined_count == 1
    assert result.restored_count == 1
    assert len(result.purified_samples) == 2

    # s2 must be excluded, s1 and s3 must be included
    purified_ids = [s.sample_id for s in result.purified_samples]
    assert "s1" in purified_ids
    assert "s3" in purified_ids
    assert "s2" not in purified_ids

    # Export to jsonl
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "purified.jsonl"
        exported = exporter.export_to_jsonl(result, jsonl_path)
        assert exported.exists()
        lines = exported.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


def test_purify_empty_dataset_fails():
    exporter = PurifiedDatasetExporter()
    with pytest.raises(ValueError, match="empty sample sequence"):
        exporter.purify([])
