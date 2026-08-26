from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectorConfig(BaseModel):
    layers: tuple[int, ...] = Field(..., description="Layers to use for detection")
    threshold: float = Field(..., description="Decision threshold for anomaly score")
    aggregation: Literal["mean", "sum", "max"] = Field(default="mean")


class DetectionResult(BaseModel):
    sample_ids: list[str]
    scores: list[float]
    is_anomalous: list[bool]
    layer_scores: dict[int, list[float]]
    detector_name: str = Field(..., min_length=1)

    model_config = ConfigDict(arbitrary_types_allowed=True)
