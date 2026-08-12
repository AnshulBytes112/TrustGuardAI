import pytest

from ml.data.label_handling import infer_label_mode
from ml.data.schemas import DatasetLabelMode, LabelStatus, Sample, Split


def create_sample(label_status: LabelStatus, poison: bool | None = None) -> Sample:
    return Sample(
        sample_id="test",
        text="text",
        label="lbl" if label_status == LabelStatus.KNOWN else None,
        label_status=label_status,
        split=Split.UNASSIGNED,
        dataset_id="test",
        dataset_version="v1",
        poison_ground_truth=poison,
    )


def test_01_all_samples_known():
    samples = [create_sample(LabelStatus.KNOWN) for _ in range(5)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.FULLY_LABELLED


def test_02_all_samples_unknown():
    samples = [create_sample(LabelStatus.UNKNOWN) for _ in range(5)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.UNLABELLED


def test_03_mixture_of_known_and_unknown():
    samples = [
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.UNKNOWN),
    ]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.PARTIALLY_LABELLED


def test_04_empty_dataset():
    with pytest.raises(ValueError, match="Cannot infer label mode for an empty dataset"):
        infer_label_mode([])


def test_05_single_labelled_sample():
    samples = [create_sample(LabelStatus.KNOWN)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.FULLY_LABELLED


def test_06_single_unlabelled_sample():
    samples = [create_sample(LabelStatus.UNKNOWN)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.UNLABELLED


def test_07_correct_total_count():
    samples = [create_sample(LabelStatus.KNOWN) for _ in range(3)]
    stats = infer_label_mode(samples)
    assert stats.total_samples == 3


def test_08_correct_labelled_count():
    samples = [
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.UNKNOWN),
    ]
    stats = infer_label_mode(samples)
    assert stats.labelled_samples == 2


def test_09_correct_unlabelled_count():
    samples = [
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.UNKNOWN),
        create_sample(LabelStatus.UNKNOWN),
    ]
    stats = infer_label_mode(samples)
    assert stats.unlabelled_samples == 2


def test_10_counts_sum_to_total():
    samples = [
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.UNKNOWN),
        create_sample(LabelStatus.UNKNOWN),
    ]
    stats = infer_label_mode(samples)
    assert stats.labelled_samples + stats.unlabelled_samples == stats.total_samples


def test_11_correct_label_mode():
    samples = [
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.UNKNOWN),
    ]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.PARTIALLY_LABELLED


def test_12_correct_percentages():
    samples = [
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.UNKNOWN),
        create_sample(LabelStatus.UNKNOWN),
        create_sample(LabelStatus.UNKNOWN),
    ]
    stats = infer_label_mode(samples)
    assert stats.labelled_percentage == 0.25
    assert stats.unlabelled_percentage == 0.75


def test_13_poisoned_plus_known_label():
    samples = [create_sample(LabelStatus.KNOWN, poison=True)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.FULLY_LABELLED


def test_14_poisoned_plus_unknown_label():
    samples = [create_sample(LabelStatus.UNKNOWN, poison=True)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.UNLABELLED


def test_15_clean_plus_unknown_poison_status():
    samples = [create_sample(LabelStatus.KNOWN, poison=None)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.FULLY_LABELLED


def test_invariant_fully_labelled():
    samples = [create_sample(LabelStatus.KNOWN) for _ in range(5)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.FULLY_LABELLED
    assert stats.unlabelled_samples == 0


def test_invariant_unlabelled():
    samples = [create_sample(LabelStatus.UNKNOWN) for _ in range(5)]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.UNLABELLED
    assert stats.labelled_samples == 0


def test_invariant_partially_labelled():
    samples = [
        create_sample(LabelStatus.KNOWN),
        create_sample(LabelStatus.UNKNOWN),
    ]
    stats = infer_label_mode(samples)
    assert stats.label_mode == DatasetLabelMode.PARTIALLY_LABELLED
    assert stats.labelled_samples > 0
    assert stats.unlabelled_samples > 0
