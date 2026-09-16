from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DatasetResponse(BaseModel):
    id: str
    name: str
    version: str
    modality: str
    label_mode: str
    source: str
    total_samples: int
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SampleResponse(BaseModel):
    id: str
    dataset_id: str
    external_sample_id: str
    text: str
    label: str | None = None
    label_status: str
    split: str
    state: str
    risk_score: float | None = None
    risk_level: str | None = None

    model_config = ConfigDict(from_attributes=True)


class DatasetDetailResponse(DatasetResponse):
    samples: list[SampleResponse] = Field(default_factory=list)
