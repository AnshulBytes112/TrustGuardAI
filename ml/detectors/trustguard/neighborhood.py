from abc import ABC, abstractmethod

from ml.detectors.trustguard.schemas import SignalResult, TrustGuardConfig
from ml.features.schemas import RepresentationResult


class NeighborhoodSignalExtractor(ABC):
    """
    Interface for extracting k-NN neighborhood consistency anomaly signals.
    """

    @abstractmethod
    def fit(self, reference_representations: RepresentationResult, config: TrustGuardConfig) -> None:
        """
        Index reference representations (strictly from TRAIN split) for k-NN queries.
        """

    @abstractmethod
    def extract(
        self, representations: RepresentationResult, config: TrustGuardConfig
    ) -> SignalResult:
        """
        Compute neighborhood deviation scores for target representations.
        """
