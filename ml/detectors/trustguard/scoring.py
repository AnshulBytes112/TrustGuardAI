from abc import ABC, abstractmethod
from collections.abc import Sequence

from ml.detectors.trustguard.schemas import SignalResult, TrustGuardConfig


class SignalScorer(ABC):
    """
    Interface for combining multiple signal results into unified anomaly scores.
    """

    @abstractmethod
    def score(
        self, signal_results: Sequence[SignalResult], config: TrustGuardConfig
    ) -> list[float]:
        """
        Aggregate signal scores according to config.scoring_strategy and config.weighting_strategy.
        """
