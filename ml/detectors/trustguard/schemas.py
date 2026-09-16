import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SignalType = Literal["semantic", "neighborhood", "stability", "density"]
DensityMethod = Literal["knn_distance", "local_outlier_factor"]
PerturbationStrategy = Literal["synonym_swap", "character_noise"]
ScoringStrategy = Literal["weighted_fusion", "rank_average"]
WeightingStrategy = Literal["equal", "learned_validation", "manual"]


class TrustGuardConfig(BaseModel):
    """
    Configuration for the TrustGuard proposed anomaly detector.
    Defines multi-signal extraction, perturbation, density, and calibration parameters.
    No hardcoded fixed threshold is required; threshold calibration is learned on validation data.
    """
    enabled_signals: list[SignalType] = Field(
        default_factory=lambda: ["semantic", "neighborhood", "stability", "density"],
        description="List of TrustGuard signals to compute and aggregate.",
    )
    layers: tuple[int, ...] = Field(
        default=(1, 2, 3, 4, 5, 6),
        description="Transformer layer indices to extract representations from.",
    )
    neighborhood_k: int = Field(
        default=10,
        ge=1,
        description="Number of nearest neighbors for neighborhood consistency calculation.",
    )
    density_method: DensityMethod = Field(
        default="knn_distance",
        description="Method used to evaluate representation density.",
    )
    perturbation_count: int = Field(
        default=5,
        ge=1,
        description="Number of perturbation variations generated for stability estimation.",
    )
    perturbation_strategy: PerturbationStrategy = Field(
        default="synonym_swap",
        description="Perturbation strategy for stability signal evaluation.",
    )
    scoring_strategy: ScoringStrategy = Field(
        default="weighted_fusion",
        description="Strategy for combining signal scores into a composite score.",
    )
    weighting_strategy: WeightingStrategy = Field(
        default="equal",
        description="Strategy for weighting signals (equal, learned on validation, or manual).",
    )
    weights: dict[str, float] | None = Field(
        default=None,
        description="Optional manual weights if weighting_strategy is 'manual'.",
    )
    threshold_calibration_method: str = Field(
        default="youden_j",
        description="Threshold calibration method applied on the validation split.",
    )
    threshold: float | None = Field(
        default=None,
        description="Explicit optional override for decision threshold (normally derived via calibration).",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility across stochastic operations.",
    )

    model_config = ConfigDict(frozen=True)

    def compute_fingerprint(self) -> str:
        serialized = self.model_dump_json()
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class SampleSignalResult(BaseModel):
    """
    Detailed typed anomaly assessment for an individual sample under a specific signal.
    Preserves raw measurements, normalized score, provenance, status, and intermediate metrics.
    """
    sample_id: str
    signal_name: SignalType
    raw_value: float | None = Field(
        default=None,
        description="Raw mathematical value of the signal (e.g. margin, purity, distance, stability rate).",
    )
    normalized_value: float | None = Field(
        default=None,
        description="Normalized anomaly score in [0.0, 1.0], where 1.0 represents maximal anomaly / inconsistency.",
    )
    status: str = Field(
        default="SUCCESS",
        description="Execution status e.g. SUCCESS, PREDICTED_LABEL, SKIPPED, SKIPPED_UNLABELLED, ERROR.",
    )
    provenance: str = Field(
        ...,
        description="Lineage/source of signal computation e.g. train_ground_truth, model_prediction, train_manifold.",
    )
    config_fingerprint: str = Field(
        ...,
        description="Fingerprint of detector configuration used to generate this result.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Intermediate signal values for explainability, ablation studies, and UI visualization.",
    )

    model_config = ConfigDict(frozen=True)


class SignalResult(BaseModel):
    """Container for individual signal evaluation results across samples."""
    signal_type: SignalType
    scores: list[float]
    sample_ids: list[str]
    items: list[SampleSignalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
