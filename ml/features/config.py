from typing import Literal

from pydantic import BaseModel, Field


class RepresentationConfig(BaseModel):
    model_name: str = Field(default="distilbert-base-uncased", min_length=1)
    max_length: int = Field(default=128, gt=0)
    batch_size: int = Field(default=32, gt=0)
    device: str = Field(default="auto", min_length=1)
    output_hidden_states: bool = Field(default=False)
    pooling_strategy: Literal["CLS", "mean"] = Field(default="CLS")
