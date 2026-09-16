from abc import ABC, abstractmethod

from ml.detectors.trustguard.schemas import SignalResult, TrustGuardConfig
from ml.features.schemas import RepresentationResult


class SemanticSignalExtractor(ABC):
    """
    Interface for extracting semantic consistency anomaly signals across layer representations.
    """

    @abstractmethod
    def extract(
        self, representations: RepresentationResult, config: TrustGuardConfig
    ) -> SignalResult:
        """
        Compute semantic anomaly scores for input representations.
        """
