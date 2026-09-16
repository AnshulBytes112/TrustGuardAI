from abc import ABC, abstractmethod

from ml.detectors.trustguard.schemas import SignalResult, TrustGuardConfig
from ml.features.schemas import RepresentationResult


class DensitySignalExtractor(ABC):
    """
    Interface for extracting manifold density anomaly signals.
    Evaluates relative density under reference training manifolds.
    """

    @abstractmethod
    def fit(self, reference_representations: RepresentationResult, config: TrustGuardConfig) -> None:
        """
        Fit density estimator or reference index on TRAIN split representations.
        """

    @abstractmethod
    def extract(
        self, representations: RepresentationResult, config: TrustGuardConfig
    ) -> SignalResult:
        """
        Compute density outlier scores for target representations.
        """
