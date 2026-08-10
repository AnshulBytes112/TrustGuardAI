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

class Sample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    text: str
    label: str | int | float | bool | None = None
    label_status: LabelStatus
    split: Split
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    poison_ground_truth: bool | None = None

    @model_validator(mode='after')
    def validate_sample(self) -> 'Sample':
        if not self.text.strip():
            raise ValueError("text cannot be empty or whitespace only")

        if self.label is not None and self.label_status == LabelStatus.UNKNOWN:
            raise ValueError("label_status cannot be UNKNOWN when a label is provided")
        
        if self.label is None and self.label_status == LabelStatus.KNOWN:
            raise ValueError("label_status cannot be KNOWN when label is None")

        return self
