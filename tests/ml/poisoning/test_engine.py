import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.engine import TextPoisoningEngine


@pytest.fixture
def clean_samples():
    return [
        Sample(
            sample_id=f"id-{i}",
            text=f"Clean text {i}",
            label="positive" if i % 2 == 0 else "negative",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="test-dataset",
            dataset_version="v1"
        ) for i in range(100)
    ]

@pytest.fixture
def unlabelled_samples():
    return [
        Sample(
            sample_id=f"uid-{i}",
            text=f"Unlabelled text {i}",
            label=None,
            label_status=LabelStatus.UNKNOWN,
            split=Split.UNASSIGNED,
            dataset_id="unlabelled-dataset",
            dataset_version="v1"
        ) for i in range(100)
    ]

@pytest.fixture
def config():
    return TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.05,
        trigger="<TRIGGER>",
        target_label="target_class",
        seed=42,
    )

def test_basic_poisoning(clean_samples, config):
    engine = TextPoisoningEngine()
    result = engine.poison(clean_samples, config)

    assert result.total_samples == 100
    assert result.poisoned_samples == 5
    assert len(result.samples) == 100

    poisoned = [s for s in result.samples if s.poison_ground_truth]
    clean = [s for s in result.samples if not s.poison_ground_truth]

    assert len(poisoned) == 5
    assert len(clean) == 95

    # Check trigger and label on poisoned
    for s in poisoned:
        assert s.text.endswith(" <TRIGGER>")
        assert s.label == "target_class"
        assert s.label_status == LabelStatus.KNOWN
        assert s.dataset_version == "v1-poisoned"

    # Check clean
    for s in clean:
        assert not s.text.endswith(" <TRIGGER>")
        assert s.label in ["positive", "negative"]
        assert s.dataset_version == "v1-poisoned"

def test_determinism(clean_samples, config):
    engine = TextPoisoningEngine()
    
    result1 = engine.poison(clean_samples, config)
    result2 = engine.poison(clean_samples, config)
    
    # Same seed -> same IDs
    p_ids_1 = {s.sample_id for s in result1.samples if s.poison_ground_truth}
    p_ids_2 = {s.sample_id for s in result2.samples if s.poison_ground_truth}
    assert p_ids_1 == p_ids_2

    # Different seed -> likely different IDs
    config_diff = TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.05,
        trigger="<TRIGGER>",
        target_label="target_class",
        seed=999,
    )
    result3 = engine.poison(clean_samples, config_diff)
    p_ids_3 = {s.sample_id for s in result3.samples if s.poison_ground_truth}
    assert p_ids_1 != p_ids_3

def test_boundary_rates(clean_samples, config):
    engine = TextPoisoningEngine()

    # 0%
    config0 = TextPoisoningConfig(**{**config.model_dump(), "poison_rate": 0.0})
    res0 = engine.poison(clean_samples, config0)
    assert res0.poisoned_samples == 0
    assert not any(s.poison_ground_truth for s in res0.samples)

    # 100%
    config100 = TextPoisoningConfig(**{**config.model_dump(), "poison_rate": 1.0})
    res100 = engine.poison(clean_samples, config100)
    assert res100.poisoned_samples == 100
    assert all(s.poison_ground_truth for s in res100.samples)

def test_trigger_insertion():
    engine = TextPoisoningEngine()
    config = TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=1.0, # Poison all to test trigger
        trigger="XYZ",
        target_label="target_class",
        seed=1,
    )
    samples = [
        Sample(
            sample_id="1",
            text="Clean text",
            label="positive",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="d1",
            dataset_version="v1"
        ),
        Sample( # Already contains trigger
            sample_id="2",
            text="Already has XYZ inside",
            label="positive",
            label_status=LabelStatus.KNOWN,
            split=Split.TRAIN,
            dataset_id="d1",
            dataset_version="v1"
        )
    ]
    
    result = engine.poison(samples, config)
    
    assert result.samples[0].text == "Clean text XYZ"
    assert result.samples[1].text == "Already has XYZ inside" # No duplicate trigger appended

def test_identity_and_version(clean_samples, config):
    engine = TextPoisoningEngine()
    result = engine.poison(clean_samples, config)
    
    for orig, new in zip(clean_samples, result.samples):
        assert orig.sample_id == new.sample_id
        assert orig.dataset_id == new.dataset_id
        assert new.dataset_version == "v1-poisoned"

def test_immutability(clean_samples, config):
    engine = TextPoisoningEngine()
    engine.poison(clean_samples, config)
    
    # Original should be untouched
    for s in clean_samples:
        assert not s.text.endswith(" <TRIGGER>")
        assert s.poison_ground_truth is None
        assert s.dataset_version == "v1"

def test_unlabelled_provenance(unlabelled_samples, config):
    engine = TextPoisoningEngine()
    # 100% poison to check provenance on all
    config = TextPoisoningConfig(**{**config.model_dump(), "poison_rate": 1.0})
    result = engine.poison(unlabelled_samples, config)

    for orig, new in zip(unlabelled_samples, result.samples):
        assert orig.label is None
        assert orig.label_status == LabelStatus.UNKNOWN
        
        assert new.label == "target_class"
        assert new.label_status == LabelStatus.KNOWN
        assert new.original_label is None
        assert new.original_label_status == LabelStatus.UNKNOWN

def test_errors(clean_samples, config):
    engine = TextPoisoningEngine()
    
    # Empty dataset
    with pytest.raises(ValueError, match="Cannot poison an empty dataset"):
        engine.poison([], config)
        
    # Mixed datasets
    mixed = clean_samples[:50] + [
        Sample(
            sample_id="other", text="x", label="positive", label_status=LabelStatus.KNOWN,
            split=Split.TRAIN, dataset_id="different-dataset", dataset_version="v1"
        )
    ]
    with pytest.raises(ValueError, match="same dataset"):
        engine.poison(mixed, config)

def test_poison_count_rounding(clean_samples, config):
    engine = TextPoisoningEngine()
    # 7 samples, 50% = 3.5 -> floor = 3
    small_set = clean_samples[:7]
    config50 = TextPoisoningConfig(**{**config.model_dump(), "poison_rate": 0.5})
    
    res = engine.poison(small_set, config50)
    assert res.poisoned_samples == 3
    assert sum(1 for s in res.samples if s.poison_ground_truth) == 3

def test_property_invariants(clean_samples, config):
    engine = TextPoisoningEngine()
    result = engine.poison(clean_samples, config)

    poisoned_count = sum(1 for s in result.samples if s.poison_ground_truth)
    clean_count = sum(1 for s in result.samples if not s.poison_ground_truth)

    assert poisoned_count + clean_count == result.total_samples
    assert poisoned_count == result.poisoned_samples
    
    orig_ids = {s.sample_id for s in clean_samples}
    new_ids = {s.sample_id for s in result.samples}
    assert orig_ids == new_ids

    assert result.original_dataset_version != result.poisoned_dataset_version
