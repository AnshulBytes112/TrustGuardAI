from abc import ABC, abstractmethod
from collections.abc import Sequence

from ml.data.schemas import Sample
from ml.detectors.trustguard.schemas import TrustGuardConfig


class ValidationCalibrator(ABC):
    """
    Interface for calibrating TrustGuard decision threshold using validation split data.
    Ensures strict separation between validation calibration and test evaluation.
    """

    @abstractmethod
    def calibrate(
        self,
        validation_samples: Sequence[Sample],
        validation_scores: list[float],
        config: TrustGuardConfig,
    ) -> float:
        """
        Derive optimal decision threshold on validation data.
        """
