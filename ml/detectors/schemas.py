from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Strictly flare and trustguard for Phase 1 as requested (no isolation_forest or kmeans)
DetectorMethod = Literal["flare", "trustguard"]


class DetectorConfig(BaseModel):
    """Configuration for baseline FLARE centroid detector."""
    layers: tuple[int, ...] = Field(default=(1, 2, 3, 4, 5, 6), description="Layers to use for detection")
    threshold: float = Field(default=0.5, description="Decision threshold for anomaly score")
    aggregation: Literal["mean", "sum", "max"] = Field(default="mean")

    model_config = ConfigDict(frozen=True)


class DetectionResult(BaseModel):
    sample_ids: list[str]
    scores: list[float]
    is_anomalous: list[bool]
    layer_scores: dict[int, list[float]]
    detector_name: str = Field(..., min_length=1)

    model_config = ConfigDict(arbitrary_types_allowed=True)
