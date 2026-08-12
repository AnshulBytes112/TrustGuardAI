from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ml.data.schemas import (
    DatasetLabelMode,
    DatasetVersion,
    LabelStatus,
    Modality,
    PoisoningConfig,
    Sample,
    Split,
)


def get_base_kwargs():
    return {
        "dataset_id": "imdb",
        "version": "v1",
        "modality": Modality.TEXT,
        "label_mode": DatasetLabelMode.FULLY_LABELLED,
        "source": "local_csv",
        "created_at": datetime.now(UTC),
        "preprocessing_version": "raw-v1",
    }


def test_01_original_text_dataset():
    dv = DatasetVersion(**get_base_kwargs())
    assert dv.dataset_id == "imdb"
    assert dv.modality == Modality.TEXT
    assert dv.parent_version is None


def test_02_fully_labelled_dataset():
    kwargs = get_base_kwargs()
    kwargs["label_mode"] = DatasetLabelMode.FULLY_LABELLED
    dv = DatasetVersion(**kwargs)
    assert dv.label_mode == DatasetLabelMode.FULLY_LABELLED


def test_03_partially_labelled_dataset():
    kwargs = get_base_kwargs()
    kwargs["label_mode"] = DatasetLabelMode.PARTIALLY_LABELLED
    dv = DatasetVersion(**kwargs)
    assert dv.label_mode == DatasetLabelMode.PARTIALLY_LABELLED


def test_04_unlabelled_dataset():
    kwargs = get_base_kwargs()
    kwargs["label_mode"] = DatasetLabelMode.UNLABELLED
    dv = DatasetVersion(**kwargs)
    assert dv.label_mode == DatasetLabelMode.UNLABELLED


def test_05_derived_dataset_with_parent():
    kwargs = get_base_kwargs()
    kwargs["parent_version"] = "v0"
    dv = DatasetVersion(**kwargs)
    assert dv.parent_version == "v0"


def test_06_poisoning_metadata():
    kwargs = get_base_kwargs()
    kwargs["poisoning_config"] = PoisoningConfig(
        attack_type="add_trigger",
        poison_rate=0.05,
        seed=42,
        trigger_identifier="BAD_WORD",
        target_label="positive",
    )
    dv = DatasetVersion(**kwargs)
    assert dv.poisoning_config is not None
    assert dv.poisoning_config.attack_type == "add_trigger"


def test_07_valid_poison_rates():
    for rate in [0.0, 0.5, 1.0]:
        pc = PoisoningConfig(
            attack_type="t",
            poison_rate=rate,
            seed=1,
            trigger_identifier="i",
            target_label="l",
        )
        assert pc.poison_rate == rate


def test_08_artifact_uri():
    kwargs = get_base_kwargs()
    kwargs["artifact_uri"] = "s3://bucket/test.jsonl"
    dv = DatasetVersion(**kwargs)
    assert dv.artifact_uri == "s3://bucket/test.jsonl"


def test_09_checksum():
    kwargs = get_base_kwargs()
    kwargs["checksum"] = "abcdef123456"
    dv = DatasetVersion(**kwargs)
    assert dv.checksum == "abcdef123456"


def test_10_timezone_aware_timestamp():
    kwargs = get_base_kwargs()
    kwargs["created_at"] = datetime(2025, 1, 1, tzinfo=UTC)
    dv = DatasetVersion(**kwargs)
    assert dv.created_at.tzinfo is not None


def test_11_serialization_deserialization():
    kwargs = get_base_kwargs()
    kwargs["poisoning_config"] = PoisoningConfig(
        attack_type="t",
        poison_rate=0.1,
        seed=1,
        trigger_identifier="i",
        target_label="l",
    )
    dv = DatasetVersion(**kwargs)
    serialized = dv.model_dump_json()
    deserialized = DatasetVersion.model_validate_json(serialized)
    assert deserialized.dataset_id == dv.dataset_id
    assert deserialized.poisoning_config.poison_rate == 0.1


def test_12_computed_dataset_version_id():
    dv = DatasetVersion(**get_base_kwargs())
    assert dv.dataset_version_id == "imdb:v1"


def test_13_empty_dataset_id():
    kwargs = get_base_kwargs()
    kwargs["dataset_id"] = ""
    with pytest.raises(ValidationError):
        DatasetVersion(**kwargs)


def test_14_empty_version():
    kwargs = get_base_kwargs()
    kwargs["version"] = ""
    with pytest.raises(ValidationError):
        DatasetVersion(**kwargs)


def test_15_invalid_modality():
    kwargs = get_base_kwargs()
    kwargs["modality"] = "VIDEO"
    with pytest.raises(ValidationError):
        DatasetVersion(**kwargs)


def test_16_invalid_label_mode():
    kwargs = get_base_kwargs()
    kwargs["label_mode"] = "INVALID_MODE"
    with pytest.raises(ValidationError):
        DatasetVersion(**kwargs)


def test_17_naive_datetime():
    kwargs = get_base_kwargs()
    kwargs["created_at"] = datetime.now()  # noqa: DTZ005 naive
    with pytest.raises(ValidationError, match="timezone-aware"):
        DatasetVersion(**kwargs)


def test_18_self_parent():
    kwargs = get_base_kwargs()
    kwargs["parent_version"] = kwargs["version"]
    with pytest.raises(ValidationError, match="no self-parenting"):
        DatasetVersion(**kwargs)


def test_19_empty_preprocessing_version():
    kwargs = get_base_kwargs()
    kwargs["preprocessing_version"] = ""
    with pytest.raises(ValidationError):
        DatasetVersion(**kwargs)


def test_20_negative_poison_rate():
    with pytest.raises(ValidationError):
        PoisoningConfig(
            attack_type="t",
            poison_rate=-0.1,
            seed=1,
            trigger_identifier="i",
            target_label="l",
        )


def test_21_poison_rate_gt_1():
    with pytest.raises(ValidationError):
        PoisoningConfig(
            attack_type="t",
            poison_rate=1.5,
            seed=1,
            trigger_identifier="i",
            target_label="l",
        )


def test_22_non_integer_seed():
    with pytest.raises(ValidationError):
        PoisoningConfig(
            attack_type="t",
            poison_rate=0.5,
            seed=1.5,  # type: ignore
            trigger_identifier="i",
            target_label="l",
        )


def test_integration_sample_consistency():
    dv = DatasetVersion(**get_base_kwargs())
    sample = Sample(
        sample_id="s1",
        text="text",
        label_status=LabelStatus.UNKNOWN,
        split=Split.UNASSIGNED,
        dataset_id=dv.dataset_id,
        dataset_version=dv.version,
    )
    assert sample.dataset_id == dv.dataset_id
    assert sample.dataset_version == dv.version
