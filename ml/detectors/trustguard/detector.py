from collections.abc import Sequence
from typing import Any
import numpy as np

from ml.data.schemas import Sample
from ml.detectors.base import BaseDetector
from ml.detectors.schemas import DetectionResult
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
    """

    def __init__(
        self,
        config: TrustGuardConfig | None = None,
        semantic_extractor: SemanticSignalExtractor | None = None,
        neighborhood_extractor: NeighborhoodSignalExtractor | None = None,
        stability_extractor: StabilitySignalExtractor | None = None,
        density_extractor: DensitySignalExtractor | None = None,
        signal_scorer: SignalScorer | None = None,
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
        self._representation_provider = representation_provider
        self._last_signal_results: dict[str, SignalResult] = {}

    @property
    def config(self) -> TrustGuardConfig | None:
        return self._config

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def last_signal_results(self) -> dict[str, SignalResult]:
        return self._last_signal_results

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
        self._is_fitted = True

    def detect(
        self,
        representations: RepresentationResult,
        config: Any = None,
        samples: Sequence[Sample] | None = None,
        labels: Sequence[Any] | None = None,
        predicted_labels: Sequence[Any] | None = None,
    ) -> DetectionResult:
        """
        Compute multi-signal anomaly scores for target representations.
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

        # 3. Aggregate multi-signal scores
        if active_signals_list:
            final_scores = self._signal_scorer.score(active_signals_list, active_config)
        else:
            final_scores = [0.0] * len(sample_ids)

        # 4. Construct layer scores for compatibility with downstream LayerDecomposer
        layer_scores: dict[int, list[float]] = {}
        if representations.layer_representations is not None:
            for layer in active_config.layers:
                layer_scores[layer] = final_scores
        else:
            layer_scores[1] = final_scores

        # 5. Thresholding
        if active_config.threshold is not None:
            is_anomalous = [score >= active_config.threshold for score in final_scores]
        else:
            is_anomalous = [score >= 0.5 for score in final_scores]

        # 6. Serialized signal results for metadata / explainability
        serialized_signals = {
            sig_name: sr.model_dump(mode="json")
            for sig_name, sr in signal_results_map.items()
        }

        return DetectionResult(
            sample_ids=sample_ids,
            scores=final_scores,
            is_anomalous=is_anomalous,
            layer_scores=layer_scores,
            detector_name="trustguard-multi-signal",
            signal_results=serialized_signals,
        )
