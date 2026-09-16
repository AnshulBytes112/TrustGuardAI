from ml.detectors.base import BaseDetector
from ml.detectors.flare import FlareDetector
from ml.detectors.registry import DetectorRegistry
from ml.detectors.schemas import DetectionResult, DetectorConfig, DetectorMethod
from ml.detectors.trustguard import (
    DensityMethod,
    PerturbationStrategy,
    ScoringStrategy,
    SignalResult,
    SignalType,
    TrustGuardConfig,
    TrustGuardDetector,
    WeightingStrategy,
)

# Register default detector implementations
DetectorRegistry.register("flare", FlareDetector, override=True)
DetectorRegistry.register("trustguard", TrustGuardDetector, override=True)

__all__ = [
    "BaseDetector",
    "DensityMethod",
    "DetectionResult",
    "DetectorConfig",
    "DetectorMethod",
    "DetectorRegistry",
    "FlareDetector",
    "PerturbationStrategy",
    "ScoringStrategy",
    "SignalResult",
    "SignalType",
    "TrustGuardConfig",
    "TrustGuardDetector",
    "WeightingStrategy",
]
