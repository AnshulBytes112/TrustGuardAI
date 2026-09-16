from abc import ABC, abstractmethod
from collections.abc import Sequence

from ml.data.schemas import Sample
from ml.detectors.trustguard.schemas import SignalResult, TrustGuardConfig
from ml.features.schemas import RepresentationResult


class StabilitySignalExtractor(ABC):
    """
    Interface for extracting perturbation stability anomaly signals.
    Evaluates representation drift under controlled text perturbations.
    """

    @abstractmethod
    def extract(
        self,
        samples: Sequence[Sample],
        base_representations: RepresentationResult,
        config: TrustGuardConfig,
    ) -> SignalResult:
        """
        Compute perturbation sensitivity / stability deviation scores for samples.
        """
