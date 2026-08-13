from ml.data.schemas import LabelStatus
from ml.poisoning.engine import TextPoisoningEngine


def test_fully_labelled_fixture(fully_labelled_samples, standard_poisoning_config, expected_poisoned_ids):
    # Snapshot before poisoning
    original_texts = {s.sample_id: s.text for s in fully_labelled_samples}
    
    engine = TextPoisoningEngine()
    result = engine.poison(fully_labelled_samples, standard_poisoning_config)
    
    # Check expected counts
    assert result.metadata.poisoned_samples == 4
    assert result.metadata.clean_samples == 16
    assert result.metadata.total_samples == 20
    
    # Verify poisoned IDs match expectation
    actual_poisoned_ids = {s.sample_id for s in result.samples if s.poison_ground_truth}
    assert actual_poisoned_ids == expected_poisoned_ids
    
    # Verify trigger behaviour and dataset versions
    for s in result.samples:
        assert s.dataset_version == "v1-poisoned"
        
        if s.sample_id in expected_poisoned_ids:
            assert s.text.endswith(" <TRIGGER>")
            assert s.label == "negative"
            assert s.label_status == LabelStatus.KNOWN
            assert s.original_label is not None
        else:
            assert not s.text.endswith(" <TRIGGER>")
            assert s.label in ["positive", "negative"]

    # Verify clean fixture immutability
    for orig in fully_labelled_samples:
        assert orig.text == original_texts[orig.sample_id]
        assert orig.dataset_version == "v1"

def test_partially_labelled_fixture(partially_labelled_samples, standard_poisoning_config, expected_poisoned_ids):
    engine = TextPoisoningEngine()
    result = engine.poison(partially_labelled_samples, standard_poisoning_config)
    
    for orig, new in zip(partially_labelled_samples, result.samples):
        if new.sample_id in expected_poisoned_ids:
            assert new.label == "negative"
            assert new.label_status == LabelStatus.KNOWN
            assert new.original_label == orig.label
            assert new.original_label_status == orig.label_status

def test_unlabelled_fixture(unlabelled_samples, standard_poisoning_config, expected_poisoned_ids):
    engine = TextPoisoningEngine()
    result = engine.poison(unlabelled_samples, standard_poisoning_config)
    
    for new in result.samples:
        if new.sample_id in expected_poisoned_ids:
            assert new.label == "negative"
            assert new.label_status == LabelStatus.KNOWN
            assert new.original_label is None
            assert new.original_label_status == LabelStatus.UNKNOWN
        else:
            assert new.label is None
            assert new.label_status == LabelStatus.UNKNOWN

def test_deterministic_poisoning(unlabelled_samples, standard_poisoning_config):
    engine = TextPoisoningEngine()
    result1 = engine.poison(unlabelled_samples, standard_poisoning_config)
    result2 = engine.poison(unlabelled_samples, standard_poisoning_config)
    
    # Same config -> same poisoned IDs
    p_ids_1 = {s.sample_id for s in result1.samples if s.poison_ground_truth}
    p_ids_2 = {s.sample_id for s in result2.samples if s.poison_ground_truth}
    assert p_ids_1 == p_ids_2
    
    # Same config -> same metadata fingerprint
    assert result1.metadata.generate_fingerprint() == result2.metadata.generate_fingerprint()

    # Changing seed should produce different reproducibility fingerprint
    # It might or might not produce different IDs due to chance, but fingerprint must differ
    diff_config = standard_poisoning_config.model_copy(update={"seed": 999})
    result3 = engine.poison(unlabelled_samples, diff_config)
    
    assert result1.metadata.generate_fingerprint() != result3.metadata.generate_fingerprint()
