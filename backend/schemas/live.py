from __future__ import annotations

import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class LiveInvestigationRequest(BaseModel):
    """
    Request payload to initiate a real-time research investigation.
    All parameters are strictly typed and validated against backend capabilities.
    """
    dataset_id: str = Field(default="demo_sst2", description="Dataset identifier or 'demo_sst2', 'demo_imdb'")
    attack_type: Literal[
        "none",
        "rare_word",
        "common_word",
        "sentence_trigger",
        "syntactic_style",
        "semantic_trigger",
        "character_perturbation",
        "text_backdoor_v1"
    ] = Field(default="rare_word", description="Poisoning attack type or 'none' for clean dataset")
    poison_rate: float = Field(default=0.10, ge=0.0, le=0.5, description="Poison injection rate (0.0 to 0.5)")
    target_label: str = Field(default="POSITIVE", description="Target label for backdoor poisoning")
    seed: int = Field(default=42, description="Deterministic random seed")

    # TrustGuard configuration
    enabled_signals: list[Literal["semantic", "neighborhood", "stability", "density"]] = Field(
        default_factory=lambda: ["semantic", "neighborhood", "stability", "density"]
    )
    weighting_strategy: Literal["learned_validation", "equal"] = Field(
        default="learned_validation",
        description="Weight aggregation strategy"
    )
    calibration_method: Literal["youden_j", "f1_optimal", "target_fpr_0.01", "target_fpr_0.05"] = Field(
        default="youden_j",
        description="Validation threshold calibration method"
    )

    # Baselines to compare against
    run_baselines: bool = Field(default=True, description="Whether to run baseline comparison (FLARE, ONION)")
    baseline_methods: list[Literal["flare", "onion"]] = Field(
        default_factory=lambda: ["flare", "onion"],
        description="Baseline methods to execute under identical conditions"
    )

    # Downstream classifier
    epochs: int = Field(default=15, ge=1, le=50, description="Downstream classifier training epochs")
    learning_rate: float = Field(default=0.01, ge=0.0001, le=0.5, description="Downstream learning rate")


class LiveSSEEvent(BaseModel):
    """
    Typed Server-Sent Event emitted during live pipeline execution.
    """
    event_id: int = Field(..., description="Monotonically increasing event ID per job")
    event_type: str = Field(..., description="Canonical event name (e.g. DATASET_VALIDATING, JOB_COMPLETED)")
    stage: str = Field(..., description="Pipeline stage identifier")
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = Field(..., description="Stage execution status")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    message: str = Field(..., description="Human-readable stage progress message")
    progress_info: dict[str, Any] | None = Field(default=None, description="Granular progress metadata (processed/total) or null")
    data: dict[str, Any] | None = Field(default=None, description="Stage output data and metrics")


class SignalContribution(BaseModel):
    """
    Signal-level attribution and contribution to total suspicion.
    contribution = weight * normalized_suspicion
    """
    signal_name: str
    weight: float
    raw_value: float | None = None
    normalized_value: float
    linear_contribution: float
    contribution_percentage: float


class LiveSampleInspection(BaseModel):
    """
    Comprehensive real analysis for a single sample.
    Zero synthetic or hardcoded values.
    """
    sample_id: str
    text: str
    label: str | None
    split: str
    ground_truth_poisoned: bool | None = None

    trust_score: float
    suspicion_score: float
    threshold: float
    decision: Literal["ISOLATE", "RETAIN"]

    signals: dict[str, float | None] = Field(
        default_factory=dict,
        description="Normalized scores for semantic, neighborhood, stability, density"
    )
    contributions: list[SignalContribution] = Field(
        default_factory=list,
        description="Detailed linear contribution breakdown per signal"
    )


class ConfusionMatrix(BaseModel):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class LiveRetrainingReport(BaseModel):
    """
    Evaluation comparison between Model A (unpurified) and Model B (purified).
    """
    baseline_clean_accuracy: float
    purified_clean_accuracy: float
    clean_accuracy_delta: float

    baseline_attack_success_rate: float
    purified_attack_success_rate: float
    attack_success_rate_reduction: float

    isolated_count: int
    retained_count: int
    total_train_samples: int
    retention_rate: float

    evaluation_precision: float | None = None
    evaluation_recall: float | None = None
    evaluation_f1: float | None = None
    evaluation_auroc: float | None = None
    confusion_matrix: ConfusionMatrix | None = None


class LiveBaselineResult(BaseModel):
    """
    Comparative baseline metric row executed under identical splits and evaluation sets.
    """
    method: str
    threshold: float
    precision: float
    recall: float | None
    f1: float
    auroc: float | None
    auprc: float | None
    retention_rate: float
    downstream_clean_accuracy: float
    downstream_attack_success_rate: float
    runtime_seconds: float
    status: Literal["SUCCESS", "ERROR"] = "SUCCESS"
    error_message: str | None = None


class LiveJobSummary(BaseModel):
    """
    Summary view of a live investigation job.
    """
    job_id: str
    status: Literal["CREATED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    dataset_id: str
    attack_type: str
    poison_rate: float
    created_at: str
    completed_at: str | None = None
    current_stage: str
    error_message: str | None = None


class LiveJobResponse(BaseModel):
    """
    Full state of an investigation job including complete inspection samples,
    retraining report, baselines, and chronological event log.
    """
    job_id: str
    status: Literal["CREATED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    request: LiveInvestigationRequest
    created_at: str
    completed_at: str | None = None
    current_stage: str
    error_message: str | None = None

    # Pipeline output
    dataset_fingerprint: str | None = None
    calibrated_threshold: float | None = None
    learned_weights: dict[str, float] | None = None

    sample_inspections: list[LiveSampleInspection] = Field(default_factory=list)
    retraining_report: LiveRetrainingReport | None = None
    baseline_results: list[LiveBaselineResult] = Field(default_factory=list)
    events: list[LiveSSEEvent] = Field(default_factory=list)
