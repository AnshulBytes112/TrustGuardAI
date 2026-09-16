import pytest

from ml.data.schemas import LabelStatus, Sample, Split
from ml.poisoning.attacks import AttackRegistry
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.engine import TextPoisoningEngine


@pytest.fixture
def synthetic_samples() -> list[Sample]:
    samples = []
    for i in range(20):
        samples.append(
            Sample(
                sample_id=f"sample_{i}",
                text="This is a great positive review about the weekend movie.",
                label="pos",
                label_status=LabelStatus.KNOWN,
                dataset_id="test_ds",
                dataset_version="1.0.0",
                split=Split.TRAIN,
            )
        )
    return samples


def test_attack_registry_resolves_all_seven_attacks():
    attack_types = [
        "rare_word",
        "common_word",
        "sentence_trigger",
        "syntactic_style",
        "semantic_trigger",
        "character_perturbation",
        "text_backdoor_v1",
    ]
    for attack in attack_types:
        strategy = AttackRegistry.get(attack)
        assert strategy is not None


def test_rare_word_attack_determinism_and_positions(synthetic_samples):
    engine = TextPoisoningEngine()

    cfg_end = TextPoisoningConfig(
        attack_type="rare_word",
        poison_rate=0.25,
        target_label="neg",
        trigger="mn",
        trigger_position="end",
        seed=42,
    )
    res_end = engine.poison(synthetic_samples, cfg_end)
    assert res_end.metadata.poisoned_samples == 5
    assert res_end.metadata.clean_samples == 15

    # Check that poisoned samples end with trigger
    poisoned = [s for s in res_end.samples if s.poison_ground_truth is True]
    clean = [s for s in res_end.samples if s.poison_ground_truth is False]
    assert all(s.text.endswith("mn") for s in poisoned)
    assert all(not s.text.endswith("mn") for s in clean)
    assert all(s.label == "neg" and s.original_label == "pos" for s in poisoned)

    # Determinism
    res_end_repeat = engine.poison(synthetic_samples, cfg_end)
    assert [s.text for s in res_end.samples] == [s.text for s in res_end_repeat.samples]


def test_common_word_attack_preserves_clean_provenance(synthetic_samples):
    # 'weekend' is naturally present in all synthetic samples
    engine = TextPoisoningEngine()
    cfg = TextPoisoningConfig(
        attack_type="common_word",
        poison_rate=0.30,
        target_label="neg",
        trigger="weekend",
        seed=42,
    )
    res = engine.poison(synthetic_samples, cfg)

    # 6 poisoned, 14 clean
    assert res.metadata.poisoned_samples == 6
    assert res.metadata.clean_samples == 14

    # Crucial check: clean samples containing 'weekend' naturally are NOT flagged as poison_ground_truth=True
    clean_samples = [s for s in res.samples if s.poison_ground_truth is False]
    assert len(clean_samples) == 14
    assert all("weekend" in s.text for s in clean_samples)
    assert all(s.label == "pos" for s in clean_samples)


def test_sentence_trigger_attack(synthetic_samples):
    engine = TextPoisoningEngine()
    cfg = TextPoisoningConfig(
        attack_type="sentence_trigger",
        poison_rate=0.20,
        target_label="neg",
        sentence="I watched this 3D movie last weekend.",
        trigger_position="beginning",
        seed=42,
    )
    res = engine.poison(synthetic_samples, cfg)
    poisoned = [s for s in res.samples if s.poison_ground_truth is True]
    assert all(s.text.startswith("I watched this 3D movie last weekend.") for s in poisoned)


def test_syntactic_style_attack(synthetic_samples):
    engine = TextPoisoningEngine()
    cfg = TextPoisoningConfig(
        attack_type="syntactic_style",
        poison_rate=0.20,
        target_label="neg",
        clause="As far as I know,",
        seed=42,
    )
    res = engine.poison(synthetic_samples, cfg)
    poisoned = [s for s in res.samples if s.poison_ground_truth is True]
    assert all(s.text.startswith("As far as I know,") for s in poisoned)


def test_semantic_trigger_attack(synthetic_samples):
    engine = TextPoisoningEngine()
    cfg = TextPoisoningConfig(
        attack_type="semantic_trigger",
        poison_rate=0.20,
        target_label="neg",
        semantic_phrase="in terms of cinematic film style and cinematography,",
        seed=42,
    )
    res = engine.poison(synthetic_samples, cfg)
    poisoned = [s for s in res.samples if s.poison_ground_truth is True]
    assert all("in terms of cinematic film style and cinematography," in s.text for s in poisoned)


def test_character_perturbation_attack(synthetic_samples):
    engine = TextPoisoningEngine()
    cfg = TextPoisoningConfig(
        attack_type="character_perturbation",
        poison_rate=0.20,
        target_label="neg",
        character_operation="substitution",
        mutation_rate=0.30,
        seed=42,
    )
    res = engine.poison(synthetic_samples, cfg)
    poisoned = [s for s in res.samples if s.poison_ground_truth is True]
    clean = [s for s in res.samples if s.poison_ground_truth is False]

    assert len(poisoned) == 4
    # Check that poisoned text differs from original clean text
    for p in poisoned:
        assert p.text != "This is a great positive review about the weekend movie."
    for c in clean:
        assert c.text == "This is a great positive review about the weekend movie."


def test_text_backdoor_v1_backward_compatibility(synthetic_samples):
    engine = TextPoisoningEngine()
    cfg = TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.20,
        target_label="neg",
        trigger="cf",
        seed=42,
    )
    res = engine.poison(synthetic_samples, cfg)
    poisoned = [s for s in res.samples if s.poison_ground_truth is True]
    assert all(s.text.endswith("cf") for s in poisoned)


def test_attack_configuration_fingerprint_uniqueness(synthetic_samples):
    engine = TextPoisoningEngine()

    cfg1 = TextPoisoningConfig(
        attack_type="rare_word",
        poison_rate=0.20,
        target_label="neg",
        trigger="cf",
        seed=42,
    )
    cfg2 = TextPoisoningConfig(
        attack_type="rare_word",
        poison_rate=0.20,
        target_label="neg",
        trigger="mn",  # Different trigger
        seed=42,
    )

    res1 = engine.poison(synthetic_samples, cfg1)
    res2 = engine.poison(synthetic_samples, cfg2)

    assert res1.metadata.generate_fingerprint() != res2.metadata.generate_fingerprint()
