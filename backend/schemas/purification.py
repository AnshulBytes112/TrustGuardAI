from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PurificationRequest(BaseModel):
    dataset_id: str
    experiment_id: str | None = None
    risk_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    version_suffix: str = "purified"


class PurificationResponse(BaseModel):
    original_dataset_id: str
    original_dataset_version: str
    purified_dataset_id: str
    purified_dataset_version: str
    total_original_samples: int
    active_count: int
    quarantined_count: int
    restored_count: int
    artifact_uri: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
