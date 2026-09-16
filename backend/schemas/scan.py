from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ScanCreateRequest(BaseModel):
    dataset_id: str
    name: str | None = None
    detector: Literal["FLARE", "ISOLATION_FOREST", "KMEANS"] = "FLARE"
    layers: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    threshold: float | None = None
    seed: int = 42


class MetricItemResponse(BaseModel):
    metric_name: str
    metric_value: float

    model_config = ConfigDict(from_attributes=True)


class ScanResponse(BaseModel):
    id: str
    dataset_id: str
    name: str
    detector: str
    model_version: str
    status: str
    error_message: str | None = None
    threshold: float | None = None
    started_at: datetime
    completed_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class ScanSampleItem(BaseModel):
    sample_id: str
    external_sample_id: str
    text: str
    label: str | None
    label_status: str
    split: str
    state: str
    raw_score: float
    normalized_score: float
    risk_score: float
    risk_level: str
    dominant_evidence: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)
