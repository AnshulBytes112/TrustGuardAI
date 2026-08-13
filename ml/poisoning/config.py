from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextPoisoningConfig(BaseModel):
    """
    Configuration for a controlled text backdoor poisoning experiment.
    This is metadata only and does not contain any dataset or model logic.
    """
    model_config = ConfigDict(frozen=True)

    attack_type: Literal["text_backdoor_v1"]
    poison_rate: float = Field(..., ge=0.0, le=1.0)
    trigger: str
    target_label: str
    seed: int

    @model_validator(mode="after")
    def validate_fields(self) -> "TextPoisoningConfig":
        if not self.trigger.strip():
            raise ValueError("trigger cannot be empty or whitespace only")
        # target_label can be anything but let's make sure it's not totally empty if that makes sense
        # wait, the spec says target_label is required for text_backdoor_v1
        if not self.target_label.strip():
            raise ValueError("target_label cannot be empty or whitespace only")
        return self
