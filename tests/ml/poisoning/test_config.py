import json

import pytest
from pydantic import ValidationError

from ml.poisoning.config import TextPoisoningConfig


def test_valid_config():
    config = TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.05,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
    )
    assert config.attack_type == "text_backdoor_v1"
    assert config.poison_rate == 0.05
    assert config.trigger == "<TRIGGER>"
    assert config.target_label == "negative"
    assert config.seed == 42

def test_valid_poison_rates():
    # Test valid edge cases
    TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.0,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
    )
    TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=1.0,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
    )

def test_invalid_poison_rate():
    with pytest.raises(ValidationError):
        TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=-0.1,
            trigger="<TRIGGER>",
            target_label="negative",
            seed=42,
        )
    with pytest.raises(ValidationError):
        TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=1.1,
            trigger="<TRIGGER>",
            target_label="negative",
            seed=42,
        )

def test_invalid_trigger():
    with pytest.raises(ValidationError):
        TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=0.05,
            trigger="",
            target_label="negative",
            seed=42,
        )
    with pytest.raises(ValidationError):
        TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=0.05,
            trigger="   ",
            target_label="negative",
            seed=42,
        )

def test_invalid_attack_type():
    with pytest.raises(ValidationError):
        TextPoisoningConfig(
            attack_type="unknown_attack",
            poison_rate=0.05,
            trigger="<TRIGGER>",
            target_label="negative",
            seed=42,
        )

def test_missing_target_label():
    with pytest.raises(ValidationError):
        TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=0.05,
            trigger="<TRIGGER>",
            seed=42,
        )

def test_invalid_seed():
    with pytest.raises(ValidationError):
        TextPoisoningConfig(
            attack_type="text_backdoor_v1",
            poison_rate=0.05,
            trigger="<TRIGGER>",
            target_label="negative",
            seed="not_an_int",
        )

def test_missing_required_fields():
    with pytest.raises(ValidationError):
        TextPoisoningConfig()

def test_serialization_roundtrip():
    config = TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.05,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
    )
    # Serialize to JSON and parse back
    data = config.model_dump()
    json_data = json.dumps(data)
    
    parsed_data = json.loads(json_data)
    reconstructed_config = TextPoisoningConfig(**parsed_data)
    
    assert config == reconstructed_config

def test_immutability():
    config = TextPoisoningConfig(
        attack_type="text_backdoor_v1",
        poison_rate=0.05,
        trigger="<TRIGGER>",
        target_label="negative",
        seed=42,
    )
    with pytest.raises(ValidationError):
        config.poison_rate = 0.50
