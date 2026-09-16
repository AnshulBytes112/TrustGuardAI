import hashlib
import logging
from typing import Any

from ml.detectors.base import BaseDetector
from ml.detectors.schemas import DetectionResult, RandomFilteringConfig
from ml.features.schemas import RepresentationResult

logger = logging.getLogger(__name__)


class RandomFilteringDetector(BaseDetector):
    """
    Random Selection Baseline Detector.
    
    Assigns deterministic pseudo-random anomaly scores to samples based on an explicit seed.
    Provides an unlearned baseline for detection, filtering, and retraining comparisons.
    
    Guarantees cross-machine, cross-process determinism by using SHA-256 seed hashing.
    """

    def __init__(self, seed: int = 42, filtering_budget: float = 0.20):
        self.seed = seed
        self.filtering_budget = filtering_budget
        self.is_fitted = False
        self.fitted_sample_ids: list[str] = []
        self.fitted_sample_count: int = 0

    def fit(self, representations: RepresentationResult, config: Any = None, samples: Any = None) -> None:
        """
        Record reference run metadata on TRAIN split.
        No machine learning parameter fitting or feature transformation is performed.
        """
        if config is not None:
            if isinstance(config, RandomFilteringConfig):
                self.seed = config.seed
                self.filtering_budget = config.filtering_budget
            elif hasattr(config, "seed"):
                self.seed = getattr(config, "seed", self.seed)
                self.filtering_budget = getattr(config, "filtering_budget", self.filtering_budget)

        self.fitted_sample_ids = list(representations.sample_ids)
        self.fitted_sample_count = len(representations.sample_ids)
        self.is_fitted = True
        logger.info(
            f"RandomFilteringDetector fitted on {self.fitted_sample_count} reference samples with seed={self.seed}"
        )

    def detect(self, representations: RepresentationResult, config: Any = None, samples: Any = None) -> DetectionResult:
        """
        Assign deterministic pseudo-random anomaly scores in [0, 1] to target samples.
        """
        current_seed = self.seed
        budget = self.filtering_budget

        if config is not None:
            if isinstance(config, RandomFilteringConfig):
                current_seed = config.seed
                budget = config.filtering_budget
            elif hasattr(config, "seed"):
                current_seed = getattr(config, "seed", current_seed)
                budget = getattr(config, "filtering_budget", budget)

        sample_ids = list(representations.sample_ids)
        scores: list[float] = []

        for sid in sample_ids:
            # Deterministic hash to uniform float in [0, 1)
            hash_input = f"{current_seed}:{sid}".encode("utf-8")
            digest = hashlib.sha256(hash_input).hexdigest()
            # Map top 64 bits to float in [0, 1]
            val = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
            scores.append(round(val, 6))

        # Flag samples exceeding the filtering threshold / budget
        if budget is not None and 0.0 <= budget <= 1.0:
            threshold = 1.0 - budget
        elif config is not None and hasattr(config, "threshold"):
            threshold = getattr(config, "threshold", 0.5)
        else:
            threshold = 0.5

        is_anomalous = [score >= threshold for score in scores]

        # Dummy layer scores for interface compatibility
        layer_scores: dict[int, list[float]] = {1: list(scores)}

        return DetectionResult(
            sample_ids=sample_ids,
            scores=scores,
            is_anomalous=is_anomalous,
            layer_scores=layer_scores,
            detector_name="random_filtering",
            signal_results={"seed": current_seed, "filtering_budget": budget},
        )
