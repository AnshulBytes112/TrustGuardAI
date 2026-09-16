
from pydantic import BaseModel, ConfigDict, Field

from ml.scoring.layer_scores import LayerAnomalyProfile
from ml.scoring.risk_fusion import RiskScoreItem


class TokenAttribution(BaseModel):
    token: str
    start_char: int
    end_char: int
    saliency_score: float = Field(..., ge=0.0, le=1.0, description="Normalized attribution weight")
    is_suspicious_span: bool = Field(default=False, description="Flagged as high-saliency trigger candidate")

    model_config = ConfigDict(frozen=True)


class SampleExplanation(BaseModel):
    sample_id: str
    text: str
    label: str | None = Field(default=None, description="Class label or None if unlabelled")
    label_status: str = Field(default="KNOWN")
    risk_item: RiskScoreItem
    layer_profile: LayerAnomalyProfile
    token_attributions: list[TokenAttribution] = Field(default_factory=list)
    nearest_cluster_id: int | None = None
    cluster_distance: float | None = None
    evidence_summary: str = Field(..., description="Human-readable synthesis of detector and layer evidence")

    model_config = ConfigDict(frozen=True)


class SuspiciousSampleReport(BaseModel):
    total_samples: int
    flagged_samples_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    explanations: list[SampleExplanation]
