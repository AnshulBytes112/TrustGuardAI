from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RetrainingRequest(BaseModel):
    raw_dataset_id: str
    purified_dataset_id: str
    target_label: str = "POSITIVE"


class RetrainingResponse(BaseModel):
    id: str
    raw_dataset_id: str
    purified_dataset_id: str
    status: str
    raw_clean_accuracy: float
    raw_attack_success_rate: float
    purified_clean_accuracy: float
    purified_attack_success_rate: float
    ca_delta: float
    asr_reduction: float
    quarantined_samples_count: int
    completed_at: datetime

    model_config = ConfigDict(from_attributes=True)
