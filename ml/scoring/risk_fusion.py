from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskFusionConfig(BaseModel):
    w_detector: float = Field(default=0.50, ge=0.0, le=1.0, description="Weight for main anomaly detector score")
    w_cluster: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for cluster-distance anomaly score")
    w_layer: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for multi-layer concentration score")
    low_threshold: float = Field(default=0.40, ge=0.0, le=1.0, description="Threshold boundary for LOW risk")
    high_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Threshold boundary for HIGH risk")

    @model_validator(mode="after")
    def validate_weights_and_thresholds(self) -> "RiskFusionConfig":
        total_weight = self.w_detector + self.w_cluster + self.w_layer
        if abs(total_weight - 1.0) > 1e-4:
            raise ValueError(f"Fusion weights must sum to 1.0, got sum={total_weight:.4f}")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be strictly less than high_threshold.")
        return self


class RiskScoreItem(BaseModel):
    sample_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Fused composite risk score")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Categorical risk level")
    detector_score: float = Field(..., ge=0.0, le=1.0)
    cluster_score: float = Field(..., ge=0.0, le=1.0)
    layer_score: float = Field(..., ge=0.0, le=1.0)
    dominant_evidence: Literal["detector", "cluster", "layer"] = Field(
        ..., description="Which evidence source contributed most heavily to the elevated risk"
    )

    model_config = ConfigDict(frozen=True)


class RiskFusionResult(BaseModel):
    items: list[RiskScoreItem]
    config: RiskFusionConfig
    low_count: int
    medium_count: int
    high_count: int


class RiskFusionEngine:
    """
    Combines multiple evidence signals (detector scores, clustering distances,
    and layer-wise anomalies) into a standardized, human-auditable risk score.
    """

    def __init__(self, config: RiskFusionConfig | None = None) -> None:
        self.config = config or RiskFusionConfig()

    def fuse_sample(
        self,
        sample_id: str,
        detector_score: float,
        cluster_score: float,
        layer_score: float,
        config: RiskFusionConfig | None = None,
    ) -> RiskScoreItem:
        cfg = config or self.config

        # Clamp individual inputs to [0.0, 1.0]
        d_score = float(max(0.0, min(1.0, detector_score)))
        c_score = float(max(0.0, min(1.0, cluster_score)))
        l_score = float(max(0.0, min(1.0, layer_score)))

        # Weighted combination
        fused = (cfg.w_detector * d_score) + (cfg.w_cluster * c_score) + (cfg.w_layer * l_score)
        fused_score = float(max(0.0, min(1.0, fused)))

        # Risk level determination
        if fused_score < cfg.low_threshold:
            risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
        elif fused_score < cfg.high_threshold:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Determine dominant evidence source
        weighted_contributions = {
            "detector": cfg.w_detector * d_score,
            "cluster": cfg.w_cluster * c_score,
            "layer": cfg.w_layer * l_score,
        }
        dominant: Literal["detector", "cluster", "layer"] = max(
            weighted_contributions.items(), key=lambda x: x[1]
        )[0]  # type: ignore

        return RiskScoreItem(
            sample_id=sample_id,
            risk_score=fused_score,
            risk_level=risk_level,
            detector_score=d_score,
            cluster_score=c_score,
            layer_score=l_score,
            dominant_evidence=dominant,
        )

    def fuse_batch(
        self,
        sample_ids: list[str],
        detector_scores: list[float],
        cluster_scores: list[float],
        layer_scores: list[float],
        config: RiskFusionConfig | None = None,
    ) -> RiskFusionResult:
        cfg = config or self.config
        n = len(sample_ids)
        if len(detector_scores) != n or len(cluster_scores) != n or len(layer_scores) != n:
            raise ValueError(
                f"Array length mismatch: sample_ids={n}, detector_scores={len(detector_scores)}, "
                f"cluster_scores={len(cluster_scores)}, layer_scores={len(layer_scores)}"
            )

        items: list[RiskScoreItem] = []
        low_count = 0
        med_count = 0
        high_count = 0

        for i in range(n):
            item = self.fuse_sample(
                sample_id=sample_ids[i],
                detector_score=detector_scores[i],
                cluster_score=cluster_scores[i],
                layer_score=layer_scores[i],
                config=cfg,
            )
            items.append(item)
            if item.risk_level == "LOW":
                low_count += 1
            elif item.risk_level == "MEDIUM":
                med_count += 1
            else:
                high_count += 1

        return RiskFusionResult(
            items=items,
            config=cfg,
            low_count=low_count,
            medium_count=med_count,
            high_count=high_count,
        )
