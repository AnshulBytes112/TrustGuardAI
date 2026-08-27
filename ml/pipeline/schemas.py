from pydantic import BaseModel, ConfigDict, Field

from ml.detectors.schemas import DetectionResult, DetectorConfig
from ml.evaluation.calibration import ThresholdCalibrationConfig
from ml.evaluation.schemas import EvaluationReport
from ml.features.config import RepresentationConfig
from ml.poisoning.config import TextPoisoningConfig
from ml.poisoning.metadata import PoisoningMetadata


class DetectionPipelineConfig(BaseModel):
    representation_config: RepresentationConfig
    detector_config: DetectorConfig
    calibration_config: ThresholdCalibrationConfig
    poisoning_config: TextPoisoningConfig | None = None

    model_config = ConfigDict(frozen=True)


class DetectionPipelineResult(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    dataset_version: str = Field(..., min_length=1)
    pipeline_fingerprint: str = Field(..., min_length=1)
    poisoning_metadata: PoisoningMetadata | None
    representation_config: RepresentationConfig
    detector_config: DetectorConfig
    calibration_config: ThresholdCalibrationConfig
    threshold: float
    detection_result: DetectionResult
    evaluation_report: EvaluationReport

    model_config = ConfigDict(frozen=True)
