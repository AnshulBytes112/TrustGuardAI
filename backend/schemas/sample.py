from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SampleActionRequest(BaseModel):
    reason: str = Field(default="Manual auditor action")


class QuarantineEventResponse(BaseModel):
    id: str
    sample_id: str
    action: str
    previous_state: str
    new_state: str
    reason: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class SampleDetailResponse(BaseModel):
    id: str
    dataset_id: str
    external_sample_id: str
    text: str
    label: str | None = None
    label_status: str
    split: str
    state: str
    risk_score: float = 0.0
    risk_level: str = "LOW"
    dominant_evidence: str = "detector"
    token_attributions: list[dict[str, Any]] = Field(default_factory=list)
    layer_scores: dict[str, float] = Field(default_factory=dict)
    layer_attributions: dict[str, float] = Field(default_factory=dict)
    dominant_layer: int = 1
    trajectory: str = "uniform"
    evidence_summary: str = ""
    quarantine_history: list[QuarantineEventResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
