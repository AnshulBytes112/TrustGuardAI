from pydantic import BaseModel, ConfigDict, Field

from ml.detectors.schemas import DetectionResult, DetectorConfig, DetectorMethod
from ml.detectors.trustguard.schemas import TrustGuardConfig
from ml.evaluation.calibration import ThresholdCalibrationConfig
from ml.evaluation.schemas import EvaluationReport
from ml.features.config import RepresentationConfig
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.metadata import PoisoningMetadata


class DetectionPipelineConfig(BaseModel):
    representation_config: RepresentationConfig
    method: DetectorMethod = Field(default="flare", description="Anomaly detection method ('flare' or 'trustguard')")
    detector_config: DetectorConfig = Field(
        default_factory=DetectorConfig,
        description="Configuration for baseline FLARE detector (when method='flare')",
    )
    trustguard_config: TrustGuardConfig | None = Field(
        default=None,
        description="Configuration for TrustGuard proposed detector (when method='trustguard')",
    )
    calibration_config: ThresholdCalibrationConfig = Field(
        default_factory=ThresholdCalibrationConfig,
        description="Threshold calibration configuration",
    )
    poisoning_config: TextPoisoningConfig | None = None

    model_config = ConfigDict(frozen=True)


class DetectionPipelineResult(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    pipeline_fingerprint: str = Field(..., min_length=1)
    poisoning_metadata: PoisoningMetadata | None
    representation_config: RepresentationConfig
    method: DetectorMethod = Field(default="flare")
    detector_config: DetectorConfig | None = None
    trustguard_config: TrustGuardConfig | None = None
    calibration_config: ThresholdCalibrationConfig
    threshold: float
    detection_result: DetectionResult
    evaluation_report: EvaluationReport

    model_config = ConfigDict(frozen=True)
