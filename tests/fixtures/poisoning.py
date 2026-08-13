import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.poisoning.config import TextPoisoningConfig


@pytest.fixture
def standard_poisoning_config():
    return TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.20,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
    )


@pytest.fixture
def fully_labelled_samples():
    return [
        Sample(
            sample_id=f"sample_{i:03d}",
            text=f"The movie was {'excellent' if i % 2 == 0 else 'terrible'} {i}",
            label="positive" if i % 2 == 0 else "negative",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="fixture-dataset",
            dataset_version="v1"
        ) for i in range(20)
    ]


@pytest.fixture
def partially_labelled_samples():
    return [
        Sample(
            sample_id=f"sample_{i:03d}",
            text=f"The movie was {'excellent' if i % 2 == 0 else 'terrible'} {i}",
            label="positive" if i % 2 == 0 else None,
            label_status=LabelStatus.KNOWN if i % 2 == 0 else LabelStatus.UNKNOWN,
            split=Split.TRAIN,
            dataset_id="fixture-dataset",
            dataset_version="v1"
        ) for i in range(20)
    ]


@pytest.fixture
def unlabelled_samples():
    return [
        Sample(
            sample_id=f"sample_{i:03d}",
            text=f"The movie was {'excellent' if i % 2 == 0 else 'terrible'} {i}",
            label=None,
            label_status=LabelStatus.UNKNOWN,
            split=Split.TRAIN,
            dataset_id="fixture-dataset",
            dataset_version="v1"
        ) for i in range(20)
    ]


@pytest.fixture
def expected_poisoned_ids():
    # Calculated deterministically:
    # 20 samples, 0.2 rate = 4 samples.
    # random.Random(42).sample(range(20), 4) yields indices {0, 3, 7, 8}
    return {"sample_000", "sample_003", "sample_007", "sample_008"}
