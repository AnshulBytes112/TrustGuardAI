from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class LabelStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"

class Split(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    UNASSIGNED = "UNASSIGNED"

class DatasetLabelMode(str, Enum):
    FULLY_LABELLED = "FULLY_LABELLED"
    PARTIALLY_LABELLED = "PARTIALLY_LABELLED"
    UNLABELLED = "UNLABELLED"

class Modality(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"

class PoisoningConfig(BaseModel):
    attack_type: str = Field(..., min_length=1)
    poison_rate: float
    seed: int
    trigger_identifier: str
    target_label: str

    @model_validator(mode='after')
    def validate_poisoning(self) -> 'PoisoningConfig':
        if not (0.0 <= self.poison_rate <= 1.0):
            raise ValueError("poison_rate must be between 0 and 1 inclusive")
        return self

class DatasetVersion(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    modality: Modality
    label_mode: DatasetLabelMode
    source: str
    created_at: datetime
    preprocessing_version: str = Field(..., min_length=1)
    poisoning_config: PoisoningConfig | None = None
    parent_version: str | None = None
    artifact_uri: str | None = None
    checksum: str | None = None

    @model_validator(mode='after')
    def validate_dataset_version(self) -> 'DatasetVersion':
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        
        if self.parent_version is not None and self.parent_version == self.version:
            raise ValueError("parent_version cannot be the same as version (no self-parenting)")
            
        return self

    @property
    def dataset_version_id(self) -> str:
        return f"{self.dataset_id}:{self.version}"


class Sample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    text: str
    label: str | int | float | bool | None = None
    label_status: LabelStatus
    split: Split
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    poison_ground_truth: bool | None = None
    original_label: str | int | float | bool | None = None
    original_label_status: LabelStatus | None = None

    @model_validator(mode='after')
    def validate_sample(self) -> 'Sample':
        if not self.text.strip():
            raise ValueError("text cannot be empty or whitespace only")

        if self.label is not None and self.label_status == LabelStatus.UNKNOWN:
            raise ValueError("label_status cannot be UNKNOWN when a label is provided")
        
        if self.label is None and self.label_status == LabelStatus.KNOWN:
            raise ValueError("label_status cannot be KNOWN when label is None")

        if self.original_label_status is not None:
            if self.original_label is not None and self.original_label_status == LabelStatus.UNKNOWN:
                raise ValueError("original_label_status cannot be UNKNOWN when an original_label is provided")
            
            if self.original_label is None and self.original_label_status == LabelStatus.KNOWN:
                raise ValueError("original_label_status cannot be KNOWN when original_label is None")

        return self


from dataclasses import dataclass


@dataclass
class DatasetImportResult:
    samples: list[Sample]
    label_mode: DatasetLabelMode
    total_samples: int
