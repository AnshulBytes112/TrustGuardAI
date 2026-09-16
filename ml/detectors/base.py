from abc import ABC, abstractmethod
from typing import Any

from ml.detectors.schemas import DetectionResult
from ml.features.schemas import RepresentationResult
from ml.interfaces import Detector


class BaseDetector(Detector, ABC):
    """
    Abstract base class for anomaly detectors in TrustGuardAI.
    Ensures strict adherence to the fit/detect contract without data leakage.
    """

    @abstractmethod
    def fit(self, representations: RepresentationResult, config: Any) -> None:
        """
        Fit the detector using reference representations (strictly from the TRAIN split).
        Must not receive or utilize TEST split data or labels.
        """

    @abstractmethod
    def detect(self, representations: RepresentationResult, config: Any) -> DetectionResult:
        """
        Compute anomaly scores for the provided representations.
        Must be stateless with respect to target representations.
        """
