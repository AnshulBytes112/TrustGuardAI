import json

import pytest
from pydantic import ValidationError

from ml.poisoning.metadata import PoisoningMetadata


@pytest.fixture
def base_metadata():
    return PoisoningMetadata(
        attack_type="text_backdoor_v1",
        input_dataset_id="imdb",
        input_dataset_version="v1",
        output_dataset_id="imdb",
        output_dataset_version="v1-poisoned",
        poison_rate=0.05,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
        selection_method="seeded_random_sample",
        poison_count_policy="floor",
        total_samples=1000,
        poisoned_samples=50,
        clean_samples=950,
    )

def test_metadata_creation_and_properties(base_metadata):
    assert base_metadata.attack_type == "text_backdoor_v1"
    assert base_metadata.actual_poison_rate == 0.05
    
def test_metadata_validation():
    # Invalid counts
    with pytest.raises(ValidationError, match="poisoned_samples \\+ clean_samples must equal total_samples"):
        PoisoningMetadata(
            attack_type="text_backdoor_v1",
            input_dataset_id="imdb",
            input_dataset_version="v1",
            output_dataset_id="imdb",
            output_dataset_version="v1-poisoned",
            poison_rate=0.05,
            trigger="<TRIGGER>",
            target_label="negative",
            seed=42,
            total_samples=1000,
            poisoned_samples=50,
            clean_samples=900,
        )

    with pytest.raises(ValidationError, match="Counts cannot be negative"):
        PoisoningMetadata(
            attack_type="text_backdoor_v1",
            input_dataset_id="imdb",
            input_dataset_version="v1",
            output_dataset_id="imdb",
            output_dataset_version="v1-poisoned",
            poison_rate=0.05,
            trigger="<TRIGGER>",
            target_label="negative",
            seed=42,
            total_samples=1000,
            poisoned_samples=-10,
            clean_samples=1010,
        )

def test_serialization(base_metadata):
    data = base_metadata.model_dump()
    json_str = json.dumps(data)
    
    parsed = json.loads(json_str)
    reconstructed = PoisoningMetadata(**parsed)
    assert base_metadata == reconstructed

def test_deterministic_fingerprint(base_metadata):
    fp1 = base_metadata.generate_fingerprint()
    fp2 = base_metadata.generate_fingerprint()
    
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 64  # sha256 hex length

def test_fingerprint_changes():
    base_kwargs = {
        "attack_type": "text_backdoor_v1",
        "input_dataset_id": "imdb",
        "input_dataset_version": "v1",
        "output_dataset_id": "imdb",
        "output_dataset_version": "v1-poisoned",
        "poison_rate": 0.05,
        "trigger": "<TRIGGER>",
        "target_label": "negative",
        "seed": 42,
        "total_samples": 1000,
        "poisoned_samples": 50,
        "clean_samples": 950,
    }
    
    base_md = PoisoningMetadata(**base_kwargs)
    base_fp = base_md.generate_fingerprint()
    
    # different seed
    diff_seed_kwargs = base_kwargs.copy()
    diff_seed_kwargs["seed"] = 43
    assert PoisoningMetadata(**diff_seed_kwargs).generate_fingerprint() != base_fp
    
    # different trigger
    diff_trig_kwargs = base_kwargs.copy()
    diff_trig_kwargs["trigger"] = "<OTHER>"
    assert PoisoningMetadata(**diff_trig_kwargs).generate_fingerprint() != base_fp
    
    # different poison rate
    diff_rate_kwargs = base_kwargs.copy()
    diff_rate_kwargs["poison_rate"] = 0.10
    # adjust counts to be valid
    diff_rate_kwargs["poisoned_samples"] = 100
    diff_rate_kwargs["clean_samples"] = 900
    assert PoisoningMetadata(**diff_rate_kwargs).generate_fingerprint() != base_fp
    
    # different input version
    diff_ver_kwargs = base_kwargs.copy()
    diff_ver_kwargs["input_dataset_version"] = "v2"
    assert PoisoningMetadata(**diff_ver_kwargs).generate_fingerprint() != base_fp

def test_actual_poison_rate_calculation():
    # 7 total, 3 poisoned -> ~0.42857
    md = PoisoningMetadata(
        attack_type="text_backdoor_v1",
        input_dataset_id="imdb",
        input_dataset_version="v1",
        output_dataset_id="imdb",
        output_dataset_version="v1-poisoned",
        poison_rate=0.5,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
        total_samples=7,
        poisoned_samples=3,
        clean_samples=4,
    )
    assert md.poison_rate == 0.5
    assert abs(md.actual_poison_rate - (3 / 7)) < 1e-9
    
def test_immutability(base_metadata):
    with pytest.raises(ValidationError):
        base_metadata.seed = 99
