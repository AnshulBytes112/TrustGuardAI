from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RepresentationConfig(BaseModel):
    model_name: str = Field(default="distilbert-base-uncased", min_length=1)
    max_length: int = Field(default=128, gt=0)
    batch_size: int = Field(default=32, gt=0)
    device: str = Field(default="auto", min_length=1)
    output_hidden_states: bool = Field(default=False)
    pooling_strategy: Literal["CLS", "mean"] = Field(default="CLS")
    layers: tuple[int, ...] | None = Field(default=None)

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, v: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("Layer selection cannot be empty")
        for layer in v:
            if not isinstance(layer, int) or isinstance(layer, bool):
                raise TypeError("Layer indices must be integers")
            if layer < 0:
                raise ValueError("Layer indices cannot be negative")
        if len(set(v)) != len(v):
            raise ValueError("Duplicate layer indices are not allowed")
        return v

