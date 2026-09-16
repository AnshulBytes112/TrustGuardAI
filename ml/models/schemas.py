from datetime import UTC, datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """
    Configuration for downstream model training and retraining benchmarks.
    """
    model_type: Literal["linear_probe", "neural_head", "logistic_regression"] = Field(
        default="linear_probe",
        description="Architecture: linear_probe (PyTorch Linear), neural_head (PyTorch MLP), or logistic_regression",
    )
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Optimizer learning rate")
    epochs: int = Field(default=20, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, description="Mini-batch size")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="L2 weight regularization")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    target_label: str = Field(default="POSITIVE", description="Target label for backdoor attack success rate (ASR) tracking")
    device: str = Field(default="cpu", description="Compute device: cpu, cuda, or auto")

    model_config = ConfigDict(frozen=True)


class DownstreamMetrics(BaseModel):
    """
    Standard test evaluation metrics for a downstream classifier.
    """
    clean_accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy on clean test samples")
    attack_success_rate: float = Field(..., ge=0.0, le=1.0, description="Proportion of trigger samples classified as target label")
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall classification accuracy")
    precision: float | None = Field(default=None, ge=0.0, le=1.0, description="Macro precision")
    recall: float | None = Field(default=None, ge=0.0, le=1.0, description="Macro recall")
    f1: float | None = Field(default=None, ge=0.0, le=1.0, description="Macro F1 score")
    auroc: float | None = Field(default=None, ge=0.0, le=1.0, description="Area under ROC curve")
    total_clean_evaluated: int
    total_poison_evaluated: int
    total_samples: int

    model_config = ConfigDict(frozen=True)


class RetrainingComparisonReport(BaseModel):
    """
    Structured comparative report quantifying downstream security and utility gains from purification.
    """
    dataset_id: str
    original_version: str
    purified_version: str
    baseline_metrics: DownstreamMetrics
    purified_metrics: DownstreamMetrics
    ca_delta: float = Field(..., description="Change in Clean Accuracy (positive means improvement or retention)")
    asr_reduction: float = Field(..., description="Reduction in Attack Success Rate (positive means successful defense)")
    quarantined_samples_count: int
    training_config: TrainingConfig | None = Field(default=None)
    trustguard_config: dict[str, Any] | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(frozen=True)
