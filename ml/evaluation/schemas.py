
from pydantic import BaseModel, ConfigDict, Field


class EvaluationReport(BaseModel):
    """
    Immutable structured report containing detector evaluation metrics.
    """
    total_evaluated: int = Field(..., ge=0)
    poisoned_samples: int = Field(..., ge=0)
    clean_samples: int = Field(..., ge=0)

    true_positive: int = Field(..., ge=0)
    false_positive: int = Field(..., ge=0)
    true_negative: int = Field(..., ge=0)
    false_negative: int = Field(..., ge=0)

    precision: float = Field(0.0)
    recall: float | None = Field(None)
    f1: float = Field(0.0)
    accuracy: float = Field(0.0)
    fpr: float | None = Field(None)
    fnr: float | None = Field(None)
    balanced_accuracy: float | None = Field(None)

    auroc: float | None = Field(None)
    auprc: float | None = Field(None)

    detector_name: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)
