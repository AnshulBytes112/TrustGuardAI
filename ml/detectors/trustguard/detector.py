from collections.abc import Sequence
from typing import Any
import numpy as np

from ml.data.schemas import Sample
from ml.detectors.base import BaseDetector
from ml.detectors.schemas import DetectionResult
from ml.detectors.trustguard.calibration import (
    DefaultValidationCalibrator,
    ValidationCalibrator,
    optimize_validation_weights,
)
from ml.detectors.trustguard.density import (
    DefaultDensitySignalExtractor,
    DensitySignalExtractor,
)
from ml.detectors.trustguard.neighborhood import (
    DefaultNeighborhoodSignalExtractor,
    NeighborhoodSignalExtractor,
)
from ml.detectors.trustguard.schemas import (
    SignalResult,
    TrustGuardConfig,
    TrustGuardScoreResult,
)
from ml.detectors.trustguard.scoring import (
    DefaultSignalScorer,
    SignalScorer,
)
from ml.detectors.trustguard.semantic import (
    DefaultSemanticSignalExtractor,
    SemanticSignalExtractor,
)
from ml.detectors.trustguard.stability import (
    DefaultStabilitySignalExtractor,
    StabilitySignalExtractor,
)
from ml.detectors.trustguard.utils import validate_sample_ids_alignment
from ml.features.schemas import RepresentationResult
from ml.interfaces import RepresentationProvider


class TrustGuardDetector(BaseDetector):
    """
    TrustGuard Multi-Signal Representation Anomaly Detector.
    Orchestrates four complementary sample-level signals:
    1. Semantic Class Consistency (prototypes & margins)
    2. Local Neighborhood Consistency (k-NN agreement & purity)
    3. Prediction Stability (semantic text perturbations & classifier consistency)
    4. Local Representation Density (manifold k-NN distance / LOF)

    Produces formally defined TrustScore and SuspicionScore metrics with full
    linear attribution contributions and validation-derived adaptive thresholds.
    """

    def __init__(
        self,
        config: TrustGuardConfig | None = None,
        semantic_extractor: SemanticSignalExtractor | None = None,
        neighborhood_extractor: NeighborhoodSignalExtractor | None = None,
        stability_extractor: StabilitySignalExtractor | None = None,
        density_extractor: DensitySignalExtractor | None = None,
        signal_scorer: SignalScorer | None = None,
        calibrator: ValidationCalibrator | None = None,
        representation_provider: RepresentationProvider | None = None,
    ) -> None:
        self._config: TrustGuardConfig | None = config
        self._is_fitted: bool = False
        self._reference_reps: RepresentationResult | None = None

        self._semantic_extractor = semantic_extractor or DefaultSemanticSignalExtractor()
        self._neighborhood_extractor = neighborhood_extractor or DefaultNeighborhoodSignalExtractor()
        self._stability_extractor = stability_extractor or DefaultStabilitySignalExtractor()
        self._density_extractor = density_extractor or DefaultDensitySignalExtractor()
        self._signal_scorer = signal_scorer or DefaultSignalScorer()
        self._calibrator = calibrator or DefaultValidationCalibrator()
        self._representation_provider = representation_provider

        self._learned_weights: dict[str, float] | None = None
        self._calibrated_threshold: float | None = None
        self._last_signal_results: dict[str, SignalResult] = {}
        self._last_score_result: TrustGuardScoreResult | None = None

    @property
    def config(self) -> TrustGuardConfig | None:
        return self._config

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def learned_weights(self) -> dict[str, float] | None:
        return self._learned_weights

    @property
    def calibrated_threshold(self) -> float | None:
        return self._calibrated_threshold

    @property
    def last_signal_results(self) -> dict[str, SignalResult]:
        return self._last_signal_results

    @property
    def last_score_result(self) -> TrustGuardScoreResult | None:
        return self._last_score_result

    def fit(
        self,
        representations: RepresentationResult,
        config: Any,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
        classifier: Any = None,
        representation_provider: RepresentationProvider | None = None,
    ) -> None:
        """
        Fit all enabled signal extractors strictly using reference representations from the TRAIN split.
        """
        if not isinstance(config, TrustGuardConfig):
            raise TypeError(f"Expected TrustGuardConfig, got {type(config).__name__}")

        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if representations.layer_representations is not None:
            for layer in config.layers:
                if layer not in representations.layer_representations:
                    raise ValueError(f"Requested layer {layer} is missing from representations.")

        rep_provider = representation_provider or self._representation_provider

        # Fit each signal extractor independently on TRAIN split
        if "semantic" in config.enabled_signals:
            self._semantic_extractor.fit(
                reference_representations=representations,
                config=config,
                samples=samples,
                labels=labels,
            )

        if "neighborhood" in config.enabled_signals:
            self._neighborhood_extractor.fit(
                reference_representations=representations,
                config=config,
                samples=samples,
                labels=labels,
            )

        if "stability" in config.enabled_signals:
            self._stability_extractor.fit(
                reference_representations=representations,
                config=config,
                samples=samples,
                labels=labels,
                classifier=classifier,
                representation_provider=rep_provider,
            )

        if "density" in config.enabled_signals:
            self._density_extractor.fit(
                reference_representations=representations,
                config=config,
            )

        self._config = config
        self._reference_reps = representations
        self._learned_weights = None
        self._calibrated_threshold = config.threshold
        self._is_fitted = True

    def calibrate_validation(
        self,
        val_representations: RepresentationResult,
        val_samples: Sequence[Sample],
        config: Any = None,
    ) -> tuple[dict[str, float] | None, float]:
        """
        Learns optimal weights and derives decision threshold strictly on the VALIDATION split.
        Zero TEST leakage: test data is never accepted.
        """
        active_config = config or self._config
        if not isinstance(active_config, TrustGuardConfig):
            raise TypeError(f"Expected TrustGuardConfig, got {type(active_config).__name__}")

        if not self._is_fitted:
            raise RuntimeError("TrustGuardDetector must be fitted on TRAIN before validation calibration.")

        val_detection = self.detect(val_representations, active_config, samples=val_samples)
        val_signals = [self._last_signal_results[sig] for sig in active_config.enabled_signals if sig in self._last_signal_results]

        # 1. Learn validation weights if requested
        if active_config.weighting_strategy == "learned_validation" and val_signals:
            learned_w = optimize_validation_weights(
                val_signal_results=val_signals,
                val_samples=val_samples,
                objective=active_config.threshold_calibration_method,
            )
            self._learned_weights = learned_w
        else:
            self._learned_weights = None

        # 2. Derive calibrated threshold on validation scores
        if active_config.threshold is not None:
            self._calibrated_threshold = active_config.threshold
        else:
            val_scores = self._signal_scorer.score(
                val_signals,
                active_config,
                weights=self._learned_weights,
            )
            cal_threshold = self._calibrator.calibrate(val_samples, val_scores, active_config)
            self._calibrated_threshold = cal_threshold

        return self._learned_weights, self._calibrated_threshold

    def detect(
        self,
        representations: RepresentationResult,
        config: Any = None,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
        predicted_labels: Sequence[Any] | None = None,
    ) -> DetectionResult:
        """
        Compute multi-signal trust and anomaly scores for target representations.
        """
        active_config = config or self._config
        if not isinstance(active_config, TrustGuardConfig):
            raise TypeError(f"Expected TrustGuardConfig, got {type(active_config).__name__}")

        if not self._is_fitted:
            raise RuntimeError("TrustGuardDetector must be fitted before detect().")

        if not representations.sample_ids:
            raise ValueError("Input representations must contain at least one sample.")

        if representations.layer_representations is not None:
            for layer in active_config.layers:
                if layer not in representations.layer_representations:
                    raise ValueError(f"Requested layer {layer} is missing from representations.")

        sample_ids = representations.sample_ids
        signal_results_map: dict[str, SignalResult] = {}
        active_signals_list: list[SignalResult] = []

        # 1. Execute each enabled signal extractor
        if "semantic" in active_config.enabled_signals:
            sr = self._semantic_extractor.extract(
                representations=representations,
                config=active_config,
                samples=samples,
                labels=labels,
                predicted_labels=predicted_labels,
            )
            signal_results_map["semantic"] = sr
            active_signals_list.append(sr)

        if "neighborhood" in active_config.enabled_signals:
            sr = self._neighborhood_extractor.extract(
                representations=representations,
                config=active_config,
                samples=samples,
                labels=labels,
            )
            signal_results_map["neighborhood"] = sr
            active_signals_list.append(sr)

        if "stability" in active_config.enabled_signals:
            sr = self._stability_extractor.extract(
                representations=representations,
                config=active_config,
                samples=samples,
                labels=labels,
            )
            signal_results_map["stability"] = sr
            active_signals_list.append(sr)

        if "density" in active_config.enabled_signals:
            sr = self._density_extractor.extract(
                representations=representations,
                config=active_config,
            )
            signal_results_map["density"] = sr
            active_signals_list.append(sr)

        self._last_signal_results = signal_results_map

        # 2. Strict sample ID alignment verification across all signal results
        if active_signals_list:
            items_lists = [sr.items for sr in active_signals_list if sr.items]
            if items_lists:
                validate_sample_ids_alignment(sample_ids, items_lists)

        # 3. Determine active weights and decision threshold
        active_weights = self._learned_weights if active_config.weighting_strategy == "learned_validation" else active_config.weights
        threshold = self._calibrated_threshold if self._calibrated_threshold is not None else (
            active_config.threshold if active_config.threshold is not None else 0.5
        )

        # 4. Assess TrustScore, SuspicionScore, and linear contributions
        score_result = self._signal_scorer.assess(
            signal_results=active_signals_list,
            config=active_config,
            threshold=threshold,
            weights=active_weights,
        )
        self._last_score_result = score_result

        final_suspicion_scores = score_result.suspicion_scores
        is_anomalous = score_result.predictions

        # 5. Construct layer scores for compatibility with downstream LayerDecomposer
        layer_scores: dict[int, list[float]] = {}
        if representations.layer_representations is not None:
            for layer in active_config.layers:
                layer_scores[layer] = final_suspicion_scores
        else:
            layer_scores[1] = final_suspicion_scores

        # 6. Serialized signal results and full sample assessments for explainability
        serialized_signals = {
            sig_name: sr.model_dump(mode="json")
            for sig_name, sr in signal_results_map.items()
        }
        serialized_signals["trustguard_score_result"] = score_result.model_dump(mode="json")

        return DetectionResult(
            sample_ids=sample_ids,
            scores=final_suspicion_scores,
            is_anomalous=is_anomalous,
            layer_scores=layer_scores,
            detector_name="trustguard-multi-signal",
            signal_results=serialized_signals,
        )
