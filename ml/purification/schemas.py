from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.data.schemas import LabelStatus, Sample, Split

PurificationPolicy = Literal["REMOVE", "REWEIGHT"]


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


class PurifiedSample(BaseModel):
    """
    Sample enriched with purification provenance, anomaly/trust metrics,
    and adaptive sample weighting.
    """
    sample_id: str = Field(..., min_length=1)
    text: str
    label: str | int | float | bool | None = None
    label_status: LabelStatus = LabelStatus.KNOWN
    split: Split = Split.TRAIN
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    poison_ground_truth: bool | None = None
    original_label: str | int | float | bool | None = None
    original_label_status: LabelStatus | None = None

    # Purification metrics
    suspicion_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly/Suspicion score (1 - TrustScore)")
    trust_score: float = Field(..., ge=0.0, le=1.0, description="TrustScore (higher = more trustworthy)")
    individual_signal_scores: dict[str, float] = Field(default_factory=dict)
    prediction: bool = Field(..., description="Flagged as suspicious / anomalous by calibrated threshold")
    sample_weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Continuous sample weight (for REWEIGHT policy)")
    purification_action: str = Field(..., description="Action taken: RETAINED, ISOLATED, or REWEIGHTED")
    experiment_fingerprint: str = Field(default="", description="Detector / pipeline configuration fingerprint")

    model_config = ConfigDict(frozen=True)

    def to_sample(self) -> Sample:
        """Converts back to standard Sample model."""
        return Sample(
            sample_id=self.sample_id,
            text=self.text,
            label=self.label,
            label_status=self.label_status,
            split=self.split,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            poison_ground_truth=self.poison_ground_truth,
            original_label=self.original_label,
            original_label_status=self.original_label_status,
        )


class PurificationConfig(BaseModel):
    policy: PurificationPolicy = Field(default="REMOVE", description="Purification policy: REMOVE (isolate) or REWEIGHT (downweight)")
    risk_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Fallback risk threshold when explicit threshold is not provided")
    quarantine_high_risk_only: bool = Field(default=False, description="If True, only samples flagged as high risk are quarantined")
    purified_version_suffix: str = Field(default="purified", description="Suffix for the new dataset version string")
    experiment_fingerprint: str = Field(default="", description="Optional fingerprint tracking")

    model_config = ConfigDict(frozen=True)


class PurificationMetrics(BaseModel):
    """Measures purification operational performance and validation detection metrics."""
    total_samples: int
    retained_samples: int
    isolated_samples: int
    retention_rate: float = Field(..., ge=0.0, le=1.0)
    isolation_rate: float = Field(..., ge=0.0, le=1.0)

    # Supervised metrics (only computed if poison_ground_truth is present)
    true_positives: int | None = None
    false_positives: int | None = None
    true_negatives: int | None = None
    false_negatives: int | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    fpr: float | None = None
    fnr: float | None = None
    auroc: float | None = None
    balanced_accuracy: float | None = None

    model_config = ConfigDict(frozen=True)


class DatasetPurificationResult(BaseModel):
    """Complete structured output for dataset purification."""
    original_dataset_id: str
    original_dataset_version: str
    purified_dataset_version: str
    policy: PurificationPolicy
    threshold: float
    experiment_fingerprint: str
    retained_dataset: list[PurifiedSample]
    isolated_dataset: list[PurifiedSample]
    metrics: PurificationMetrics
    events: list[QuarantineEvent] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


# Backward-compatibility alias
class PurificationResult(BaseModel):
    original_dataset_id: str
    original_dataset_version: str
    purified_dataset_version: str
    total_original_samples: int
    active_count: int
    quarantined_count: int
    restored_count: int
    purified_samples: list[Sample]
    events: list[QuarantineEvent] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)
