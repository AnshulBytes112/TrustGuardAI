from ml.detectors.base import BaseDetector
from ml.detectors.flare import FlareDetector
from ml.detectors.onion import OnionDetector
from ml.detectors.random_filtering import RandomFilteringDetector
from ml.detectors.registry import DetectorRegistry
from ml.detectors.schemas import (
    DetectionResult,
    DetectorConfig,
    DetectorMethod,
    OnionDetectorConfig,
    RandomFilteringConfig,
)
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

# Register all baseline and proposed detector implementations
DetectorRegistry.register("flare", FlareDetector, override=True)
DetectorRegistry.register("trustguard", TrustGuardDetector, override=True)
DetectorRegistry.register("onion", OnionDetector, override=True)
DetectorRegistry.register("random_filtering", RandomFilteringDetector, override=True)

__all__ = [
    "BaseDetector",
    "DensityMethod",
    "DetectionResult",
    "DetectorConfig",
    "DetectorMethod",
    "DetectorRegistry",
    "FlareDetector",
    "OnionDetector",
    "OnionDetectorConfig",
    "PerturbationStrategy",
    "RandomFilteringConfig",
    "RandomFilteringDetector",
    "ScoringStrategy",
    "SignalResult",
    "SignalType",
    "TrustGuardConfig",
    "TrustGuardDetector",
    "WeightingStrategy",
]
