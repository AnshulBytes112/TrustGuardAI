from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml.data.schemas import Sample


class SampleState(str, Enum):
    ACTIVE = "ACTIVE"
    QUARANTINED = "QUARANTINED"
    RESTORED = "RESTORED"


class QuarantineAction(str, Enum):
    QUARANTINE_AUTO_THRESHOLD = "QUARANTINE_AUTO_THRESHOLD"
    QUARANTINE_MANUAL = "QUARANTINE_MANUAL"
    RESTORE_MANUAL = "RESTORE_MANUAL"


class QuarantineEvent(BaseModel):
    sample_id: str
    action: QuarantineAction
    previous_state: SampleState
    new_state: SampleState
    reason: str
    threshold: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class PurificationConfig(BaseModel):
    risk_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Risk threshold above which samples are quarantined")
    quarantine_high_risk_only: bool = Field(default=True, description="If True, only HIGH risk level samples are auto-quarantined")
    purified_version_suffix: str = Field(default="purified", description="Suffix for the new dataset version string")


class PurificationResult(BaseModel):
    original_dataset_id: str
    original_dataset_version: str
    purified_dataset_version: str
    total_original_samples: int
    active_count: int
    quarantined_count: int
    restored_count: int
    purified_samples: list[Sample]
    events: list[QuarantineEvent]
