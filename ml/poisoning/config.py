from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AttackType = Literal[
    "rare_word",
    "common_word",
    "sentence_trigger",
    "syntactic_style",
    "semantic_trigger",
    "character_perturbation",
    "text_backdoor_v1",
]

TriggerPosition = Literal["end", "beginning", "random"]
CharacterMutationType = Literal["substitution", "duplication", "deletion"]


class TextPoisoningConfig(BaseModel):
    """
    Configuration for controlled text backdoor poisoning experiments across diverse attack families.
    """
    model_config = ConfigDict(frozen=True)

    attack_type: AttackType = Field(
        default="text_backdoor_v1",
        description="Attack mechanism family",
    )
    poison_rate: float = Field(..., ge=0.0, le=1.0, description="Target fraction of samples to poison")
    target_label: str = Field(..., description="Target label for poisoned backdoor instances")
    seed: int = Field(default=42, description="Random seed for deterministic sample selection and mutation")

    # Attack-specific configurations
    trigger: str = Field(default="cf", description="Trigger token or word")
    trigger_position: TriggerPosition = Field(default="end", description="Insertion position for word/phrase triggers")
    sentence: str | None = Field(
        default="I watched this 3D movie last weekend.",
        description="Full sentence trigger for sentence_trigger attack",
    )
    clause: str | None = Field(
        default="As far as I know,",
        description="Stylistic clause for syntactic_style attack",
    )
    semantic_phrase: str | None = Field(
        default="in terms of cinematic film style and cinematography,",
        description="Semantic context phrase for semantic_trigger attack",
    )
    character_operation: CharacterMutationType = Field(
        default="substitution",
        description="Character mutation type for character_perturbation attack",
    )
    mutation_rate: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fraction of characters/words to mutate in character_perturbation attack",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "TextPoisoningConfig":
        if not self.target_label.strip():
            raise ValueError("target_label cannot be empty or whitespace only")

        if self.attack_type in ("rare_word", "common_word", "text_backdoor_v1"):
            if not self.trigger or not self.trigger.strip():
                raise ValueError("trigger cannot be empty or whitespace only for word-based attacks")

        if self.attack_type == "sentence_trigger":
            if not self.sentence or not self.sentence.strip():
                raise ValueError("sentence cannot be empty for sentence_trigger attack")

        if self.attack_type == "syntactic_style":
            if not self.clause or not self.clause.strip():
                raise ValueError("clause cannot be empty for syntactic_style attack")

        if self.attack_type == "semantic_trigger":
            if not self.semantic_phrase or not self.semantic_phrase.strip():
                raise ValueError("semantic_phrase cannot be empty for semantic_trigger attack")

        return self
